"""Warp call detection and Torch/NumPy boundary propagation.

Tracing model
-------------
LEAPP does not decompose Warp kernels into Torch operations.  During
``annotate.warp_op(node_name)`` this module intercepts public ``warp.*`` calls,
propagates the active LEAPP trace through ``wp.array`` values, and records the
arrays read or written by the segment.  ``WarpOp.__exit__`` then saves the APIC
capture and inserts one ``leapp::warp_runner`` FX node whose ``getitem`` nodes
represent the segment outputs.

Torch-facing custom autograd wrappers are not inherently opaque to this
mechanism.  ``torch.autograd.Function.forward`` still executes eagerly with a
``TracedTensor``, so calls such as ``wp.from_torch``, ``wp.launch``, and
``wp.to_torch`` reach these wrappers.  They must still execute inside an
explicit Warp segment; patched calls outside ``annotate.warp_op`` run normally
but do not create an APIC-backed FX node.

Important lifecycle limitation: ``wp.to_torch`` inside an open segment
-----------------------------------------------------------------------
The boundary overload below correctly preserves the *current* proxy of its
source ``TracedWpArray``.  For an output array, however, the final Warp output
proxy does not exist until ``WarpOp.__exit__`` inserts the runner node.  This
creates the following sequence for a Torch-style Warp autograd function:

1. A traced Torch input is converted to Warp and gives the segment trace state.
2. A kernel writes a preallocated Warp output.  Until segment close, that
   output temporarily carries the input-derived proxy.
3. ``wp.to_torch(output_wp)`` runs inside ``forward`` and creates a
   ``TracedTensor`` alias using that temporary proxy.
4. Segment close rebinds the ``TracedWpArray`` to the new runner-output proxy,
   but currently does not rebind Torch aliases that were already returned.
5. ``annotate.output_tensors`` therefore sees the result as the original input.
   The runner has no live FX consumers and normal dead-code pruning removes it.

The eager numerical result is still correct, which makes this a particularly
dangerous silent tracing failure.  Conversion after the segment closes works:
the source Warp array has already been rebound, so the existing ``to_torch``
overload propagates the runner-output proxy.  A complete fix should register
Torch/NumPy aliases created from segment outputs while the segment is open and
rebind those aliases when ``_insert_warp_marker`` assigns output proxies.
Fail-closed validation should also reject a Warp-derived declared output that
still points at a pre-segment input proxy.

Shaped Warp dtype limitation
----------------------------
``_validate_boundary_dtype`` currently rejects vector and matrix dtypes such
as ``wp.vec3``.  Scalar-only examples do not need shaped-dtype support, and the
proxy lifecycle bug above reproduces with ``wp.float32`` alone.  Existing
fabrics-sim collision code does need ``wp.vec3`` because it reinterprets a
Torch ``(..., 3)`` tensor as a logical Warp vector array.  Kinematics also
returns ``wp.transform``/``wp.vec3`` arrays.  Supporting those boundaries
requires recording both the scalar Torch dtype and the expanded Torch storage
shape (for example, logical Warp shape ``(B, N)`` becomes Torch shape
``(B, N, 3)`` for ``wp.vec3``).  Alternatively, callers must change their
kernels to scalar arrays and construct vectors explicitly.

Standalone regression reproducer
---------------------------------
From ``leapp_repo`` run:

``.venv/bin/python reproduce_autograd_warp_segment.py all``

It demonstrates the stale-proxy pruning failure, the successful conversion
after segment close, and the current ``wp.vec3`` boundary rejection.  Keep
those three cases when changing boundary propagation or segment finalization.
"""

from typing import Any, Callable
from types import ModuleType
import contextlib
import functools
from dataclasses import dataclass
import inspect
import sys

import warp as wp
from warp._src.context import Function as WarpKernelLanguageFunction

from leapp.utils.caller_identity import caller_identity_has_same_anchor
from leapp.utils.logging import _get_logger

from .cupti_oracle import WarpCudaOracle
from .traced_wp_array import TracedWpArray
from .warp_segment import WarpSegment

_WRAPPER_MARKER = "__leapp_warp_detector_wrapper__"
_ALLOWED_DUNDER_METHODS = {"__init__"}
_MAX_PARAM_SCAN_DEPTH = 16
_MAX_CLASS_SCAN_DEPTH = 1

@dataclass
class _Patch:
    owner: Any
    attr_name: str
    original: Any
    wrapper: Any


@dataclass(frozen=True)
class _WarpTraceState:
    name: str
    context: Any
    proxy: Any



class WarpPatchBackend:
    """Warp monkeypatch backend and per-session trace routing.

    Patches loaded ``warp`` callables during :meth:`install` and records calls
    into the active :class:`~leapp.leapp_graph.datatypes.warp.warp_segment.WarpSegment`
    while a :class:`~leapp.leapp_graph.warp_op.WarpOp` block is open.
    """

    def __init__(self) -> None:
        self._patches: list[_Patch] = []
        self._wrappers_by_original_id: dict[int, Any] = {}
        self._recording_paused = False
        self._active_segment: Any | None = None
        self._installed = False
        self._boundary_function_ids: set[int] = set()
        self._boundary_array_init_id: int | None = None
        self._cuda_oracle = WarpCudaOracle()
    #########################################################
    # Properties
    #########################################################
    @property
    def patched_count(self) -> int:
        return len(self._patches)
    #########################################################
    # Public methods
    #########################################################
    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> "WarpPatchBackend":
        """Patch currently loaded Warp module functions and class methods."""

        if self._installed:
            return self

        self._register_boundary_functions()
        self._patch_warp_modules()

        self._patch_loaded_aliases()
        self._cuda_oracle.start()

        self._installed = True
        return self

    def begin_discovery_segment(
        self,
        *,
        node_name: str,
        call_stack: Any,
    ) -> WarpSegment:
        segment = WarpSegment(
            node_name=node_name,
            call_stack=call_stack,
        )
        self.activate_segment(segment)
        return segment

    def call_stack_matches_segment(
        self,
        segment: WarpSegment,
        call_stack: Any,
    ) -> bool:
        return caller_identity_has_same_anchor(segment.call_stack, call_stack)

    def end_discovery_segment(
        self,
        segment: WarpSegment,
        call_stack: Any,
    ) -> WarpSegment | None:
        if not self.call_stack_matches_segment(segment, call_stack):
            return None

        active = self.deactivate_segment(segment)
        if active is not segment:
            _get_logger().fatal(
                "Discovery Warp segment is not active.",
                error_type=RuntimeError,
            )

        warp_call_events = tuple(
            event for event in segment.events if event.get("kind") == "warp_call"
        )
        call_qualnames = tuple(str(event["qualname"]) for event in warp_call_events)
        segment.call_qualnames = call_qualnames
        segment.status = "closed"
        return segment

    def begin_capture_segment(
        self,
        *,
        segment: WarpSegment,
    ) -> WarpSegment:
        segment.status = "open"
        segment.events.clear()
        segment.input_refs.clear()
        segment.output_refs.clear()
        segment.marker_proxy = None
        segment.proxy_name = None
        segment.apic_graph = None
        self.activate_segment(segment)
        return segment

    def end_capture_segment(self, segment: WarpSegment, call_stack: Any) -> bool:
        if not self.call_stack_matches_segment(segment, call_stack):
            return False

        active = self.deactivate_segment(segment)
        if active is not segment:
            _get_logger().fatal(
                "Capture Warp segment is not active.",
                error_type=RuntimeError,
            )

        expected_qualnames = segment.call_qualnames
        actual_qualnames = tuple(
            str(event["qualname"])
            for event in segment.events
            if event.get("kind") == "warp_call"
        )
        if actual_qualnames != expected_qualnames:
            segment.invalidate()
            _get_logger().fatal(
                f"[{segment.node_name}] Warp segment diverged between "
                "discovery and capture. "
                f"Expected calls {expected_qualnames}, got {actual_qualnames}.",
                error_type=RuntimeError,
            )
        return True

    def activate_segment(self, segment: Any) -> None:
        """Make ``segment`` the active destination for detected Warp calls."""
        if self._active_segment is not None:
            _get_logger().fatal(
                "A Warp segment is already active; only one segment may be "
                "open globally at a time.",
                error_type=RuntimeError,
            )
        self._active_segment = segment
        self._cuda_oracle.set_segment(segment)

    def deactivate_segment(self, segment: Any | None = None) -> Any | None:
        """Deactivate and return the active Warp segment."""
        if self._active_segment is None:
            return None

        active = self._active_segment
        if segment is not None and active is not segment:
            _get_logger().fatal(
                "Warp segment is not the active segment.",
                error_type=ValueError,
            )
        self._active_segment = None
        self._cuda_oracle.set_segment(None)
        return active

    @property
    def active_segment(self) -> Any | None:
        return self._active_segment

    @contextlib.contextmanager
    def paused(self):
        """Suppress Warp call detection for the duration of the block.

        Temporarily routes patched Warp functions straight to their original
        implementations and allows soft-banned CUDA callbacks as Warp-owned.
        Previous state is restored so nested pauses compose correctly.
        """
        previous_recording_paused = self._recording_paused
        previous_cuda_allowed = self._cuda_oracle.set_warp_cuda_allowed(True)
        self._recording_paused = True
        try:
            yield
        finally:
            self._recording_paused = previous_recording_paused
            self._cuda_oracle.set_warp_cuda_allowed(previous_cuda_allowed)

    def uninstall(self) -> None:
        """Restore every attribute patched by this detector."""

        self._active_segment = None
        self._cuda_oracle.set_segment(None)
        self._cuda_oracle.stop()

        for patch in reversed(self._patches):
            try:
                current = inspect.getattr_static(patch.owner, patch.attr_name)
            except Exception:
                try:
                    current = getattr(patch.owner, patch.attr_name)
                except Exception:
                    continue
            if current is patch.wrapper:
                try:
                    setattr(patch.owner, patch.attr_name, patch.original)
                except Exception:
                    pass

        self._patches.clear()
        self._wrappers_by_original_id.clear()
        self._boundary_function_ids.clear()
        self._boundary_array_init_id = None
        self._installed = False

    #########################################################
    # Module patching
    #########################################################
    def _patch_loaded_aliases(self) -> None:
        """Patch already-imported aliases such as ``from warp import launch``.

        Only updates bindings outside public ``warp`` modules. Phase 1 already
        patched those namespaces; this pass catches imports that still reference
        the original callables by identity.
        """

        if not self._wrappers_by_original_id:
            return

        for module_name, module in list(sys.modules.items()):
            if not isinstance(module, ModuleType):
                continue
            if self._is_warp_module(module_name, module):
                continue

            try:
                attrs = list(vars(module).items())
            except Exception:
                continue

            for attr_name, value in attrs:
                if not self._is_warp_owned_callable(value):
                    continue
                wrapper = self._wrappers_by_original_id.get(id(value))
                if wrapper is None:
                    continue
                if getattr(value, _WRAPPER_MARKER, False):
                    continue
                self._patch_attr(
                    module,
                    attr_name,
                    value,
                    value,
                    f"{module.__name__}.{attr_name}",
                )

    def _patch_warp_modules(self) -> None:
        modules = [
            module
            for module_name, module in sys.modules.items()
            if self._is_warp_module(module_name, module)
        ]
        modules.sort(key=lambda module: (module.__name__.count("."), module.__name__))

        # recursive scan to patch modules
        for module in modules:
            self._patch_namespace(
                module,
                module.__name__,
                owner_module_name=module.__name__,
                class_depth=0,
            )

    def _patch_namespace(
        self,
        owner: Any,
        qualname: str,
        *,
        owner_module_name: str,
        class_depth: int,
    ) -> None:
        for attr_name, value in list(vars(owner).items()):
            attr_qualname = f"{qualname}.{attr_name}"
            if self._should_scan_class(value, owner_module_name, class_depth):
                self._patch_namespace(
                    value,
                    attr_qualname,
                    owner_module_name=owner_module_name,
                    class_depth=class_depth + 1,
                )
            else:
                # this is not a namespace; maybe it is a callable we should wrap, otherwise ignore it.
                self._patch_callable_attr(owner, attr_name, value, attr_qualname)

    def _should_scan_class(self, cls: Any, owner_module_name: str, class_depth: int) -> bool:
        # Class scan criteria:
        # - The depth is not too deep.
        # - Only scan actual class objects.
        # - Only scan classes defined by Warp (`warp` or `warp.*`).
        # - Only scan classes exposed through public Warp modules.
        # - Skip classes reached only through private modules (`warp._*`).
        # - Descend only one level to patch class methods.
        # - Avoid patching arbitrary user classes or private Warp internals.
        if class_depth >= _MAX_CLASS_SCAN_DEPTH:
            return False
        if not inspect.isclass(cls):
            return False
        if not isinstance(owner_module_name, str):
            return False

        class_module = self._definition_module_name(cls)
        if class_module != "warp" and not class_module.startswith("warp."):
            return False

        if owner_module_name.startswith("warp._"):
            return False

        return True

    def _patch_callable_attr(self, owner: Any, attr_name: str, raw_value: Any, qualname: str) -> None:
        # Patch only safe callable attributes:
        # - Preserve staticmethod/classmethod binding by unwrapping, wrapping, then
        #   restoring the original descriptor type.
        # - Skip unsupported dunders, existing wrappers, classes, properties, Warp kernel
        #   language functions, and non-callables.
        # - Patch normal callables directly.
        descriptor_type = None
        callable_original = raw_value
        if isinstance(raw_value, staticmethod):
            descriptor_type = staticmethod
            callable_original = raw_value.__func__
        elif isinstance(raw_value, classmethod):
            descriptor_type = classmethod
            callable_original = raw_value.__func__

        if attr_name.startswith("__") and attr_name not in _ALLOWED_DUNDER_METHODS:
            return
        if callable_original is None or getattr(callable_original, _WRAPPER_MARKER, False):
            return
        if inspect.isclass(callable_original) or isinstance(callable_original, property):
            return
        if isinstance(callable_original, WarpKernelLanguageFunction):
            """In Warp, symbols like these often look callable from Python:
                wp.dot
                wp.sin
                wp.frac
                etc.
                this function contains logic to filter them out.
            """
            return
        if not callable(callable_original):
            return
        if not self._is_warp_owned_callable(callable_original):
            return

        self._patch_attr(
            owner,
            attr_name,
            raw_value,
            callable_original,
            qualname,
            descriptor_type=descriptor_type,
        )

    def _patch_attr(
        self,
        owner: Any,
        attr_name: str,
        raw_original: Any,
        callable_original: Callable,
        qualname: str,
        *,
        descriptor_type: type[staticmethod] | type[classmethod] | None = None,
    ) -> None:
        if getattr(callable_original, _WRAPPER_MARKER, False):
            return

        for patch in self._patches:
            if patch.owner is owner and patch.attr_name == attr_name:
                return

        wrapper_func = self._get_or_make_wrapper(qualname, callable_original)
        wrapper = descriptor_type(wrapper_func) if descriptor_type is not None else wrapper_func

        try:
            setattr(owner, attr_name, wrapper)
        except Exception:
            return

        self._patches.append(_Patch(owner, attr_name, raw_original, wrapper))


    #########################################################
    # Wrapper creation and execution
    #########################################################


    def _get_or_make_wrapper(self, qualname: str, original: Callable) -> Callable:
        wrapper = self._wrappers_by_original_id.get(id(original))
        if wrapper is None:
            wrapper = self._make_wrapper(qualname, original)
            self._wrappers_by_original_id[id(original)] = wrapper
        return wrapper


    def _make_wrapper(self, qualname: str, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if self._recording_paused:
                return original(*args, **kwargs)

            if id(original) in self._boundary_function_ids:
                handled, result = self._handle_boundary_call(original, args, kwargs)
                if handled:
                    return result

            # Single pass: swap active traced arrays for raw ``.data`` views (so
            # Warp sees exact wp.array objects) and collect them so we can derive
            # the shared trace state. The original traced objects are left
            # untouched, so there is nothing to convert back afterward.
            traced_inputs, call_args, call_kwargs = self._normalize_and_collect(args, kwargs)
            trace_state = self._build_trace_state(qualname, traced_inputs)
            segment = self._resolve_segment(trace_state)

            if segment is not None:
                self._record_segment_inputs(segment, qualname, traced_inputs)

            with self.paused():
                result = original(*call_args, **call_kwargs)

            self._process_post_call_arrays(
                segment, args, kwargs, result, trace_state
            )

            return result

        setattr(wrapped, _WRAPPER_MARKER, True)
        return wrapped

    #########################################################
    # Boundary tracing (torch/numpy <-> warp)
    #########################################################

    def _register_boundary_functions(self) -> None:
        """Record original boundary callables before patching replaces them."""
        self._boundary_function_ids = set()
        self._boundary_array_init_id = None

        array_init = getattr(wp.array, "__init__", None)
        candidates = [
            array_init,
            getattr(wp, "from_torch", None),
            getattr(wp, "to_torch", None),
            getattr(wp, "from_numpy", None),
            getattr(wp.array, "numpy", None),
        ]
        for fn in candidates:
            if fn is None or not callable(fn):
                continue
            fn_id = id(fn)
            self._boundary_function_ids.add(fn_id)
            if fn is array_init:
                self._boundary_array_init_id = fn_id

    def _handle_boundary_call(
        self, original: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[bool, Any]:
        # lazy to avoid circular imports
        from leapp.leapp_graph.datatypes import as_traced, is_tracable_tensor_type

        is_init = id(original) == self._boundary_array_init_id
        if is_init and not args:
            return False, None

        search_args = args[1:] if is_init else args
        src = self._find_single_active_traced_data(search_args, kwargs)
        if src is None or not src.is_tracing:
            return False, None

        self._validate_boundary_dtype(kwargs)
        from leapp.leapp_graph.datatypes.traced_data import TracedData

        call_args = TracedData.unwrap_traced_data(args)
        call_kwargs = TracedData.unwrap_traced_data(kwargs)

        with self.paused():
            raw = original(*call_args, **call_kwargs)

        if is_init:
            TracedWpArray.make_traced_in_place(
                args[0], src.name, src.context_obj, src.proxy
            )
            return True, None
        if raw is src.data:
            return True, src
        if is_tracable_tensor_type(raw):
            traced_raw = as_traced(
                raw,
                src.name,
                src.context_obj,
                src.proxy,
                preserve_identity=isinstance(raw, wp.array),
            )
            return True, traced_raw
        return True, raw

    def _find_single_active_traced_data(
        self, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None
    ):
        # lazy to avoid circular imports
        from leapp.leapp_graph.datatypes.traced_data import TracedData

        contexts = TracedData.find_all_contexts([args, kwargs or {}])

        if not contexts:
            return None
        if len(contexts) > 1:
            _get_logger().fatal(
                "Warp boundary call received traced data from different LEAPP "
                "trace contexts. Mixing active contexts is not supported.",
                error_type=ValueError,
            )

        src = TracedData.find_traced_data([args, kwargs or {}])
        if src is None or not src.is_tracing:
            return None
        return src

    def _validate_boundary_dtype(self, kwargs: dict[str, Any]) -> None:
        dtype = kwargs.get("dtype")
        if dtype is not None and getattr(dtype, "_shape_", None):
            # TODO: location to look into dtypes
            raise NotImplementedError(
                "LEAPP warp boundary tracing does not yet support vector/matrix "
                f"warp dtypes (got {dtype}). Reshape in torch/numpy first or use a "
                "scalar warp dtype."
            )

    def _normalize_and_collect(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[list[TracedWpArray], tuple[Any, ...], dict[str, Any]]:
        """Single pass over ``args`` and ``kwargs``: substitute active traced
        arrays with raw ``.data`` views and collect them for trace-state
        validation.

        Returns ``(traced_inputs, call_args, call_kwargs)``. Only ``TracedWpArray``
        instances are substituted, so Warp receives exact ``wp.array`` objects
        (passing both the concrete ``pack_arg`` check and the generic
        ``infer_argument_types`` check). The view aliases the same memory
        (``copy=False``), so kernels compute on the real buffer and APIC capture
        records the correct allocation. Containers are rebuilt rather than
        mutated in place, so the caller's own args/lists are never touched and
        there is nothing to walk back afterward. Raw ``wp.array`` values and
        all other values pass through unchanged so the post-call traversal
        can still class-swap them later.
        """
        traced: list[TracedWpArray] = []
        call_args = self._normalize_node(args, traced, depth=0)
        call_kwargs = self._normalize_node(kwargs, traced, depth=0)
        return traced, call_args, call_kwargs

    def _normalize_node(
        self, obj: Any, traced: list[TracedWpArray], *, depth: int
    ) -> Any:
        if depth > _MAX_PARAM_SCAN_DEPTH:
            return obj

        if isinstance(obj, TracedWpArray):
            # Always hand Warp an exact ``wp.array`` view (so both concrete
            # and generic kernels accept it), but only record the array for
            # trace-state propagation when its owning context is actively
            # tracing. A traced array whose context has stopped tracing thus
            # flows through as plain data and never re-traces the outputs.
            if obj.is_tracing:
                traced.append(obj)
            return obj.data

        if isinstance(obj, dict):
            return {
                key: self._normalize_node(value, traced, depth=depth + 1)
                for key, value in obj.items()
            }
        if isinstance(obj, (list, tuple, set, frozenset)):
            return type(obj)(
                self._normalize_node(item, traced, depth=depth + 1) for item in obj
            )

        # Anything else (scalars like int/float/bool/str, raw wp.array, None,
        # Device, dtypes, ...) is not a traced array and is passed through as-is.
        return obj

    def _build_trace_state(
        self, qualname: str, traced_inputs: list[TracedWpArray]
    ) -> "_WarpTraceState | None":
        if not traced_inputs:
            return None

        source = traced_inputs[0]
        for candidate in traced_inputs[1:]:
            if candidate.context_obj is not source.context_obj:
                _get_logger().fatal(
                    f"{qualname} received traced Warp arrays from different LEAPP "
                    "trace contexts.",
                    error_type=ValueError,
                )

        return _WarpTraceState(source.name, source.context_obj, source.proxy)

    def _resolve_segment(self, trace_state: "_WarpTraceState | None") -> Any | None:
        if self.active_segment is not None:
            return self.active_segment
        return None

    def _record_segment_inputs(
        self,
        segment: Any,
        qualname: str,
        traced_inputs: list[TracedWpArray],
    ) -> None:
        segment.add_event({"kind": "warp_call", "qualname": qualname})

        for array in traced_inputs:
            if array.warp_segment is not segment:
                segment.add_input_ref(array)

    def _process_post_call_arrays(
        self,
        segment: Any | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        trace_state: "_WarpTraceState | None",
    ) -> None:
        seen: set[int] = set()
        for value in (args, kwargs, result):
            self._process_post_call_node(
                value,
                segment,
                trace_state,
                seen,
                depth=0,
            )

    def _process_post_call_node(
        self,
        obj: Any,
        segment: Any | None,
        trace_state: "_WarpTraceState | None",
        seen: set[int],
        *,
        depth: int,
    ) -> None:
        if depth > _MAX_PARAM_SCAN_DEPTH:
            return

        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        # needs to lazy import to avoid circular import
        from leapp.leapp_graph.datatypes import as_traced, is_tracable_tensor_type

        if is_tracable_tensor_type(obj):
            if trace_state is not None and isinstance(obj, wp.array):
                traced_array = as_traced(
                    obj,
                    trace_state.name,
                    trace_state.context,
                    trace_state.proxy,
                    preserve_identity=True,
                )
                if segment is not None:
                    segment.add_output_ref(traced_array)
                    traced_array.warp_segment = segment
            return

        if isinstance(obj, dict):
            for value in obj.values():
                self._process_post_call_node(
                    value,
                    segment,
                    trace_state,
                    seen,
                    depth=depth + 1,
                )
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                self._process_post_call_node(
                    item,
                    segment,
                    trace_state,
                    seen,
                    depth=depth + 1,
                )
        elif isinstance(obj, (set, frozenset)):
            for item in obj:
                self._process_post_call_node(
                    item,
                    segment,
                    trace_state,
                    seen,
                    depth=depth + 1,
                )



    #########################################################
    # static Helper functions
    #########################################################

    @staticmethod
    def _is_warp_module(module_name: str, module: Any) -> bool:
        if not isinstance(module, ModuleType):
            return False
        if not isinstance(module_name, str):
            return False
        if module_name == "warp":
            return True
        if not module_name.startswith("warp."):
            return False
        return not module_name.startswith("warp._")

    @staticmethod
    def _definition_module_name(obj: Any) -> str:
        """Return ``obj.__module__`` when it is a real string (not a descriptor)."""
        module = getattr(obj, "__module__", None)
        return module if isinstance(module, str) else ""

    @staticmethod
    def _is_warp_owned_callable(obj: Any) -> bool:
        """True when ``obj`` is a callable defined by warp-lang, not a re-export."""
        if not callable(obj):
            return False
        owner_module = WarpPatchBackend._definition_module_name(obj)
        return owner_module == "warp" or owner_module.startswith("warp.")