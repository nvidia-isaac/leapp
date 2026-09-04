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
import inspect
import sys

import torch
import warp as wp
from warp._src.context import Function as WarpKernelLanguageFunction

from leapp.utils.caller_identity import caller_identity_has_same_anchor, get_caller_stack_identity
from leapp.utils.logging import _get_logger

from .._attribute_patching import AttributePatchRegistry
from ..proxy_view import may_adopt_view
from .cupti_oracle import WarpCudaOracle
from .session import WarpTraceSession
from .traced_wp_array import TracedWpArray
from .warp_segment import WarpSegment

_WRAPPER_MARKER = "__leapp_warp_detector_wrapper__"
_ALLOWED_DUNDER_METHODS = {"__init__"}
_MAX_CLASS_SCAN_DEPTH = 1


class WarpPatchBackend:
    """Warp monkeypatch backend and per-session trace routing.

    Patches loaded ``warp`` callables during :meth:`install` and records calls
    into the active :class:`~leapp.leapp_graph.datatypes.warp.warp_segment.WarpSegment`
    while a :class:`~leapp.leapp_graph.warp_op.WarpOp` block is open.
    """

    def __init__(self) -> None:
        self._patches = AttributePatchRegistry()
        self._wrappers_by_original_id: dict[int, Any] = {}
        self._qualnames_by_original_id: dict[int, str] = {}
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

        try:
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
        except Exception:
            self.uninstall()
            raise

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

    def pause_context(self):
        if self._session is None:
            return contextlib.nullcontext()
        return self._session.pause()

    def is_boundary_function(self, func: Callable) -> bool:
        return id(func) in self._boundary_function_ids

    def is_readback_boundary(self, func: Callable) -> bool:
        return id(func) in self._readback_boundary_function_ids

    def is_full_copy_function(self, func: Callable) -> bool:
        return id(func) == self._full_copy_function_id

    def function_qualname(self, func: Callable) -> str:
        return self._qualnames_by_original_id.get(
            id(func), f"warp.{func.__qualname__}"
        )

    def uninstall(self) -> None:
        """Restore every attribute patched by this detector."""

        if self._cuda_oracle is not None:
            self._cuda_oracle.stop()
            self._cuda_oracle = None

        self._patches.restore(suppress_errors=True)
        self._wrappers_by_original_id.clear()
        self._qualnames_by_original_id.clear()
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

        if self._patches.contains(owner, attr_name):
            return

        wrapper_func = self._get_or_make_wrapper(qualname, callable_original)
        wrapper = descriptor_type(wrapper_func) if descriptor_type is not None else wrapper_func

        try:
            self._patches.install(
                owner,
                attr_name,
                raw_original,
                wrapper,
            )
        except Exception:
            return

    #########################################################
    # Wrapper creation and execution
    #########################################################

    def _get_or_make_wrapper(self, qualname: str, original: Callable) -> Callable:
        wrapper = self._wrappers_by_original_id.get(id(original))
        if wrapper is None:
            wrapper = self._make_wrapper(qualname, original)
            self._wrappers_by_original_id[id(original)] = wrapper
            self._qualnames_by_original_id[id(original)] = qualname
        return wrapper

    def _make_wrapper(self, qualname: str, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            if self._session is not None and self._session.paused:
                return original(*args, **kwargs)

            if id(original) in self._sync_boundary_function_ids:
                self.close_warp_segment()
                return original(*args, **kwargs)

            if id(original) == self._boundary_array_init_id:
                handled, result = self._handle_array_init(original, args, kwargs)
                if handled:
                    return result

            return TracedWpArray.__warp_function__(
                original,
                (TracedWpArray,),
                args,
                kwargs,
            )

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

    def _handle_array_init(
        self, original: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[bool, Any]:
        from leapp.leapp_graph.datatypes.traced_data import TracedData

        if not args:
            return False, None

        src = TracedWpArray._find_single_traced_data(args[1:], kwargs)
        if src is None:
            with self.pause_context():
                original(*args, **kwargs)
            return True, None

        call_args = TracedData.unwrap_traced_data(args)
        call_kwargs = TracedData.unwrap_traced_data(kwargs)
        with self.pause_context():
            original(*call_args, **call_kwargs)

        if may_adopt_view(src, args[0]):
            view, proxy = src.proxy_view, None
        else:
            view, proxy = None, src.proxy
        TracedWpArray.make_traced_in_place(
            args[0], src.name, src.context_obj, proxy, view=view
        )
        return True, None

    def resolve_or_begin_warp_segment(
        self,
        trace_source: Any | None,
    ) -> Any | None:
        active_segment = None if self._session is None else self._session.active_segment
        if active_segment is not None:
            if trace_source is None:
                return active_segment
            incoming_node_name = getattr(trace_source.context_obj, "name", None)
            if incoming_node_name == active_segment.node_name:
                return active_segment
            self.close_warp_segment()
            warp_op = self._begin_boundary_closeable_warp_op(trace_source)
            return None if warp_op is None else warp_op.segment
        if trace_source is None:
            return None
        warp_op = self._begin_boundary_closeable_warp_op(trace_source)
        return None if warp_op is None else warp_op.segment

    def _begin_boundary_closeable_warp_op(
        self,
        trace_source: Any,
    ) -> Any | None:
        node_ref = trace_source.context_obj
        if node_ref is None or not getattr(node_ref, "is_tracing", False):
            return None
        warp_op = self.create_warp_op(node_ref)
        return warp_op.begin(
            call_stack=get_caller_stack_identity(),
        )

    def record_segment_inputs(
        self,
        segment: Any,
        qualname: str,
        traced_inputs: list[TracedWpArray],
    ) -> None:
        segment.add_event({"kind": "warp_call", "qualname": qualname})

        for array in traced_inputs:
            if not segment.knows_array(array):
                segment.add_input_ref(array)

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
        """Return ``obj.__module__`` when it is a real string (not a descriptor).

        The alias scan reads this from arbitrary third-party objects, and a lazy
        import placeholder standing in for a missing optional dependency answers
        every attribute with the ImportError it represents. ``getattr``'s default
        only absorbs ``AttributeError``, so that would escape and take
        ``leapp.start`` with it. An object unwilling to say where it came from is
        not a Warp callable, which makes the refusal a miss.
        """
        try:
            module = getattr(obj, "__module__", None)
        except Exception:
            return ""
        return module if isinstance(module, str) else ""

    @staticmethod
    def _is_warp_owned_callable(obj: Any) -> bool:
        """True when ``obj`` is a callable defined by warp-lang, not a re-export."""
        if not callable(obj):
            return False
        owner_module = WarpPatchBackend._definition_module_name(obj)
        return owner_module == "warp" or owner_module.startswith("warp.")
