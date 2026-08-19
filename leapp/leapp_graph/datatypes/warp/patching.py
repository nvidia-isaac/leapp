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
``wp.to_torch`` reach these wrappers.  Patched calls can run inside an explicit
``annotate.warp_op`` context or trigger automatic ``WarpOp`` placement when
they first touch actively traced Warp data.

``wp.to_torch`` inside an open segment
---------------------------------------
A zero-copy conversion does not copy a proxy, it shares the source's
``ProxyView``, which is what lets a Torch alias of a Warp buffer stay correct
across a segment it was created inside:

1. A traced Torch input is converted to Warp and gives the segment trace state.
2. A kernel writes a preallocated Warp output.  Until segment close, that
   output temporarily carries the donor's proxy.
3. ``wp.to_torch(output_wp)`` runs inside ``forward``.  The layouts match, so
   the resulting ``TracedTensor`` adopts ``output_wp``'s view rather than
   snapshotting its temporary proxy.
4. Segment close replaces the proxy inside that one view, so the Warp array and
   the Torch alias both move to the runner output together.
5. ``annotate.output_tensors`` sees the runner output, which keeps it alive
   through pruning.

Before view sharing, step 3 copied the temporary proxy and step 4 rebound only
the Warp array, so the declared output still resolved to the pre-segment input
and the runner was pruned for having no consumers.  The eager numbers were
right either way, which is what made it a silent tracing failure.  Conversions
that genuinely copy -- a CUDA readback, a device move -- still get an
independent root, so the same hazard would return for them.

Standalone regression reproducer
---------------------------------
From ``leapp_repo`` run:

``.venv/bin/python reproduce_autograd_warp_segment.py all``

It covers conversion inside an open segment and after it closes. Keep those
cases when changing boundary propagation or segment finalization.
"""

from typing import Any, Callable
from types import ModuleType
import contextlib
import functools
from dataclasses import dataclass
import inspect
import sys

import torch
import warp as wp
from warp._src.context import Function as WarpKernelLanguageFunction

from leapp.utils.caller_identity import caller_identity_has_same_anchor, get_caller_stack_identity
from leapp.utils.logging import _get_logger

from ..proxy_view import ProxyView, may_adopt_view
from .cupti_oracle import WarpCudaOracle
from .session import WarpTraceSession
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
    # this is the simplification of the traced state provided by the proxies
    name: str
    context: Any
    view: ProxyView



class WarpPatchBackend:
    """Warp monkeypatch backend and per-session trace routing.

    Patches loaded ``warp`` callables during :meth:`install` and records calls
    into the active :class:`~leapp.leapp_graph.datatypes.warp.warp_segment.WarpSegment`
    while a :class:`~leapp.leapp_graph.warp_op.WarpOp` block is open.
    """

    def __init__(self) -> None:
        self._patches: list[_Patch] = []
        self._wrappers_by_original_id: dict[int, Any] = {}
        self._session: Any | None = None
        self._installed = False
        self._boundary_function_ids: set[int] = set()
        self._sync_boundary_function_ids: set[int] = set()
        self._readback_boundary_function_ids: set[int] = set()
        self._boundary_array_init_id: int | None = None
        self._full_copy_function_id: int | None = None
        self._cuda_oracle: WarpCudaOracle | None = None
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

        if torch.cuda.is_available():
            torch.cuda.init()
            torch.cuda.memory.caching_allocator_enable(False)

        self._cuda_oracle = WarpCudaOracle(self.close_warp_segment)
        self._session = WarpTraceSession()
        self._cuda_oracle.set_session(self._session)
        self._cuda_oracle.start()
        self._register_boundary_functions()
        self._patch_warp_modules()

        self._patch_loaded_aliases()

        self._installed = True
        return self

    def call_stack_matches_segment(
        self,
        segment: WarpSegment,
        call_stack: Any,
    ) -> bool:
        return caller_identity_has_same_anchor(segment.open_call_stack, call_stack)

    def create_warp_op(self, node_ref: Any):
        """Create a WarpOp owned by this patch backend."""
        if not self.installed or self._session is None:
            _get_logger().fatal(
                "LEAPP: the warp backend is not installed. "
                "Please call leapp.start(..., global_patching=True), and make sure warp-lang is installed.",
                error_type=ImportError,
            )
        from leapp.leapp_graph.warp_op import WarpOp

        return WarpOp(
            node_ref,
            session=self._session,
            capture=node_ref.is_warp_capture_active,
        )

    def close_warp_segment(self, *_args: Any, **_kwargs: Any) -> None:
        """Request closure of the active unowned Warp segment."""
        if self._session is None:
            return
        if self._session.paused:
            return
        closed = self._session.close_warp_segment(
            close_call_stack=get_caller_stack_identity(),
        )
        if not closed:
            _get_logger().warning(
                "Boundary requested closure of an explicit active WarpOp; "
                "leaving it open because it is protected by its owner token."
            )

    def _pause_context(self):
        if self._session is None:
            return contextlib.nullcontext()
        return self._session.pause()

    def uninstall(self) -> None:
        """Restore every attribute patched by this detector."""

        if self._cuda_oracle is not None:
            self._cuda_oracle.stop()
            self._cuda_oracle = None

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
        self._sync_boundary_function_ids.clear()
        self._readback_boundary_function_ids.clear()
        self._boundary_array_init_id = None
        if self._session is not None:
            self._session.reset()
        self._session = None
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
            if self._session is not None and self._session.paused:
                return original(*args, **kwargs)

            # Warp sync APIs are hard boundaries. Close before running them, and
            # do not pause: pausing would hide the CUDA sync from CUPTI.
            if id(original) in self._sync_boundary_function_ids:
                self.close_warp_segment()
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
            segment = self._resolve_or_begin_warp_segment(trace_state)

            if segment is not None:
                self._record_segment_inputs(segment, qualname, traced_inputs)

            with self._pause_context():
                result = original(*call_args, **call_kwargs)

            self._process_post_call_arrays(
                segment, args, kwargs, result, trace_state
            )
            self._carry_full_copy_port(original, args, kwargs)

            return result

        setattr(wrapped, _WRAPPER_MARKER, True)
        return wrapped

    #########################################################
    # Boundary tracing (torch/numpy <-> warp)
    #########################################################

    def _register_boundary_functions(self) -> None:
        """Record original boundary callables before patching replaces them."""
        self._boundary_function_ids = set()
        self._sync_boundary_function_ids = set()
        self._readback_boundary_function_ids = set()
        self._boundary_array_init_id = None

        full_copy = getattr(wp, "copy", None)
        self._full_copy_function_id = id(full_copy) if callable(full_copy) else None

        array_init = getattr(wp.array, "__init__", None)
        to_torch = getattr(wp, "to_torch", None)
        array_numpy = getattr(wp.array, "numpy", None)
        candidates = [
            array_init,
            getattr(wp, "from_torch", None),
            to_torch,
            getattr(wp, "from_numpy", None),
            array_numpy,
        ]
        for fn in candidates:
            if fn is None or not callable(fn):
                continue
            fn_id = id(fn)
            self._boundary_function_ids.add(fn_id)
            if fn is array_init:
                self._boundary_array_init_id = fn_id
            if fn is to_torch or fn is array_numpy:
                self._readback_boundary_function_ids.add(fn_id)

        for sync_name in (
            "synchronize",
            "synchronize_device",
            "synchronize_event",
            "synchronize_stream",
        ):
            sync_fn = getattr(wp, sync_name, None)
            if sync_fn is not None and callable(sync_fn):
                self._sync_boundary_function_ids.add(id(sync_fn))

    def _handle_boundary_call(
        self, original: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[bool, Any]:
        # lazy to avoid circular imports
        from leapp.leapp_graph.datatypes import as_traced, is_tracable_tensor_type
        from leapp.leapp_graph.datatypes.traced_data import TracedData

        is_init = id(original) == self._boundary_array_init_id
        # ``__init__`` with no args has no ``self`` to promote; fall through.
        if is_init and not args:
            return False, None

        search_args = args[1:] if is_init else args
        src = self._find_single_traced_data(search_args, kwargs)
        # earlly exit path if no traced data is involved.
        if src is None:
            if is_init:
                with self._pause_context():
                    original(*args, **kwargs)
                return True, None
            return False, None

        # Host conversion/readback ends the current Warp segment before the
        # CUDA copy/sync happens under pause (which CUPTI would otherwise miss).
        if id(original) in self._readback_boundary_function_ids:
            self.close_warp_segment()

        call_args = TracedData.unwrap_traced_data(args)
        call_kwargs = TracedData.unwrap_traced_data(kwargs)

        with self._pause_context():
            raw = original(*call_args, **call_kwargs)

        if is_init:
            if may_adopt_view(src, args[0]):
                view, proxy = src.proxy_view, None
            else:
                view, proxy = None, src.proxy
            TracedWpArray.make_traced_in_place(
                args[0], src.name, src.context_obj, proxy, view=view
            )
            return True, None
        if raw is src.data:
            return True, src
        if is_tracable_tensor_type(raw):
            if may_adopt_view(src, raw):
                view, proxy = src.proxy_view, None
            else:
                view, proxy = None, src.proxy
            traced_raw = as_traced(
                raw, src.name, src.context_obj, proxy, view=view
            )
            return True, traced_raw
        return True, raw

    def _find_single_traced_data(
        self, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None
    ):
        # lazy to avoid circular imports
        from leapp.leapp_graph.datatypes.traced_data import TracedData

        values = []

        def collect(obj):
            if isinstance(obj, TracedData):
                values.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    collect(value)
            elif isinstance(obj, (list, tuple, set, frozenset)):
                for value in obj:
                    collect(value)

        collect([args, kwargs or {}])
        if not values:
            return None

        context_ids = {id(value.context_obj) for value in values}
        if len(context_ids) > 1:
            _get_logger().fatal(
                "Warp boundary call received traced data from different LEAPP "
                "trace contexts. Mixing contexts is not supported.",
                error_type=ValueError,
            )
        return values[0]

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
            _get_logger().fatal(
                "When traversing a nested structure in a Warp function call, "
                f"exceeded LEAPP's max traversal depth ({_MAX_PARAM_SCAN_DEPTH}).",
                error_type=RuntimeError,
            )

        if isinstance(obj, TracedWpArray):
            # Always hand Warp an exact ``wp.array`` view. Active carriers
            # drive segment capture; inactive carriers only propagate tracing
            # state to array-valued results.
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

        return _WarpTraceState(source.name, source.context_obj, source.proxy_view)

    def _resolve_or_begin_warp_segment(
        self,
        trace_state: "_WarpTraceState | None",
    ) -> Any | None:
        active_segment = None if self._session is None else self._session.active_segment
        if active_segment is not None:
            if trace_state is None:
                return active_segment
            incoming_node_name = getattr(trace_state.context, "name", None)
            if incoming_node_name == active_segment.node_name:
                return active_segment # same node, no need to close
            else:
                # different node, close the active segment
                self.close_warp_segment()
                warp_op = self._begin_boundary_closeable_warp_op(trace_state)
                return None if warp_op is None else warp_op.segment
        if trace_state is not None:
            # no active segment, begin a new one
            warp_op = self._begin_boundary_closeable_warp_op(trace_state)
            return None if warp_op is None else warp_op.segment
        return None # no active segment, no need to close

    def _begin_boundary_closeable_warp_op(
        self,
        trace_state: "_WarpTraceState",
    ) -> Any | None:
        node_ref = trace_state.context
        if node_ref is None or not getattr(node_ref, "is_tracing", False):
            return None
        warp_op = self.create_warp_op(node_ref)
        return warp_op.begin(
            call_stack=get_caller_stack_identity(),
        )

    def _record_segment_inputs(
        self,
        segment: Any,
        qualname: str,
        traced_inputs: list[TracedWpArray],
    ) -> None:
        segment.add_event({"kind": "warp_call", "qualname": qualname})

        for array in traced_inputs:
            # An array the segment already stands for is not a new input, whether
            # an earlier launch produced it or this one passed it twice.
            if not segment.knows_array(array):
                segment.add_input_ref(array)

    def _carry_full_copy_port(
        self, original: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        """Give a full ``wp.copy`` destination the boundary identity of its source.

        The post-call traversal has already upgraded a raw destination into a
        traced array bound to the source's node. A copy covering the whole array
        leaves that destination holding the published data, so it also takes the
        port that connects it onward. Offsets or a partial count make it a
        different value, which keeps the default portless carrier.
        """
        if (
            id(original) != self._full_copy_function_id
            or len(args) < 2
            or any(args[2:])
            or kwargs.get("dest_offset")
            or kwargs.get("src_offset")
            or kwargs.get("count")
        ):
            return

        dest, src = args[0], args[1]
        if (
            getattr(src, "output_port", None) is None
            or not isinstance(dest, TracedWpArray)
            or dest.shape != src.shape
            or dest.dtype != src.dtype
        ):
            return
        src.preserve_port(dest)

    def _process_post_call_arrays(
        self,
        segment: Any | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        trace_state: "_WarpTraceState | None",
    ) -> None:
        seen: set[int] = set()
        # Warp kernels and runtime helpers can mutate arrays through arguments
        # without declaring intent, so conservatively inspect all call inputs and
        # return values for arrays that may need to be tracked as segment outputs.
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
        # needs to lazy import to avoid circular import
        from leapp.leapp_graph.datatypes import (
            as_traced,
            is_tracable_tensor_type,
            promote_in_place,
        )
        if depth > _MAX_PARAM_SCAN_DEPTH:
            _get_logger().fatal(
                "When traversing a nested structure in a Warp function call, "
                f"exceeded LEAPP's max traversal depth ({_MAX_PARAM_SCAN_DEPTH}).",
                error_type=RuntimeError,
            )

        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if is_tracable_tensor_type(obj):
            if trace_state is not None and isinstance(obj, wp.array):
                owner = getattr(obj, "context_obj", None)
                published = getattr(obj, "output_port", None) is not None
                if published or (owner is not None and owner is not trace_state.context):
                    # A value another node already published stays untouched, so
                    # it can still fan out; this call only gets an alias of it.
                    traced_array = as_traced(
                        obj,
                        trace_state.name,
                        trace_state.context,
                        trace_state.view.proxy,
                    )
                elif owner is not None:
                    # Already carries provenance for this node, so leave both its
                    # view and its proxy alone. Writing a neighbouring argument's
                    # proxy over it would discard whatever produced this buffer,
                    # and the close assigns this argument's own runner output
                    # regardless.
                    traced_array = obj
                else:
                    # The caller keeps using this exact array after the call, so
                    # its tracing state has to live on the object itself for the
                    # segment close to rebind it to the segment output proxy. It
                    # gets its own view and only borrows the donor's proxy as
                    # placeholder provenance until then: a kernel writing two
                    # arrays must leave them on two roots to receive two outputs.
                    traced_array = promote_in_place(
                        obj,
                        trace_state.name,
                        trace_state.context,
                        trace_state.view.proxy,
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
