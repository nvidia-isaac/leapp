from typing import Any, Callable
from types import ModuleType
import functools
from dataclasses import dataclass
import inspect
import sys

try:
    import warp as wp
    from warp._src.context import Function as WarpKernelLanguageFunction
    from leapp.leapp_graph.datatypes import is_tracable_tensor_type
    from .traced_wp_array import TracedWpArray
except ImportError:
    wp = None
    WarpKernelLanguageFunction = None
    TracedWpArray = None
else:
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



    class WarpLeappCallDetector:
        """Process-wide singleton: ``WarpLeappCallDetector()`` always returns the
        same instance so there is only ever one set of Warp patches in flight."""

        _instance: "WarpLeappCallDetector | None" = None

        def __new__(cls) -> "WarpLeappCallDetector":
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        @classmethod
        def instance(cls) -> "WarpLeappCallDetector":
            """Return the shared detector instance (constructing it if needed)."""
            return cls()

        def __init__(self) -> None:
            # ``__new__`` returns the shared instance on every call; guard so state
            # is initialized once and repeated ``WarpLeappCallDetector()`` calls do
            # not reset an active (installed) detector.
            if getattr(self, "_initialized", False):
                return
            self._patches: list[_Patch] = []
            self._wrappers_by_original_id: dict[int, Any] = {}
            self._recording_depth = 0
            self._segment_stack: list[Any] = []
            self._installed = False
            self._initialized = True
        #########################################################
        # Properties
        #########################################################
        @property
        def patched_count(self) -> int:
            return len(self._patches)
        #########################################################
        # Public methods
        #########################################################
        def install(self) -> "WarpLeappCallDetector":
            """Patch currently loaded Warp module functions and class methods."""

            if self._installed:
                return self

            self._patch_warp_modules()

            self._patch_loaded_aliases()

            self._installed = True
            return self

        def push_segment(self, segment: Any) -> None:
            """Make ``segment`` the active destination for detected Warp calls."""
            self._segment_stack.append(segment)

        def pop_segment(self, segment: Any | None = None) -> Any | None:
            """Remove and return the active Warp segment."""
            if not self._segment_stack:
                return None

            active = self._segment_stack[-1]
            if segment is not None and active is not segment:
                raise ValueError("Warp segment stack is not balanced.")
            return self._segment_stack.pop()

        @property
        def active_segment(self) -> Any | None:
            if not self._segment_stack:
                return None
            return self._segment_stack[-1]

        def uninstall(self) -> None:
            """Restore every attribute patched by this detector."""

            self._segment_stack.clear()

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
            self._installed = False

        #########################################################
        # Module patching
        #########################################################
        def _patch_loaded_aliases(self) -> None:
            """Patch already-imported aliases such as ``from warp import launch``."""

            if not self._wrappers_by_original_id:
                return

            for module in list(sys.modules.values()):
                if not isinstance(module, ModuleType):
                    continue

                try:
                    attrs = list(vars(module).items())
                except Exception:
                    continue

                for attr_name, value in attrs:
                    wrapper = self._wrappers_by_original_id.get(id(value))
                    if wrapper is None:
                        continue
                    if getattr(value, _WRAPPER_MARKER, False):
                        continue
                    self._patch_attr(module, attr_name, value, value, f"{module.__name__}.{attr_name}")

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

            class_module = getattr(cls, "__module__", "")
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
                if self._recording_depth:
                    return original(*args, **kwargs)

                # Single pass: swap active traced arrays for raw ``.data`` views (so
                # Warp sees exact wp.array objects) and collect them so we can derive
                # the shared trace state. The original traced objects are left
                # untouched, so there is nothing to convert back afterward.
                traced_inputs, call_args, call_kwargs = self._normalize_and_collect(args, kwargs)
                trace_state = self._build_trace_state(qualname, traced_inputs)
                segment = self._resolve_segment(trace_state)

                if segment is not None:
                    self._record_segment_inputs(segment, qualname, traced_inputs)

                self._recording_depth += 1
                try:
                    result = original(*call_args, **call_kwargs)
                finally:
                    self._recording_depth -= 1

                self._process_post_call_arrays(
                    segment, qualname, args, kwargs, result, trace_state
                )

                return result

            setattr(wrapped, _WRAPPER_MARKER, True)
            return wrapped

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
                    raise ValueError(
                        f"{qualname} received traced Warp arrays from different LEAPP "
                        "trace contexts. Propagating mixed Warp trace contexts is "
                        "not supported yet."
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
            qualname: str,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            result: Any,
            trace_state: "_WarpTraceState | None",
        ) -> None:
            seen: set[int] = set()
            for val, name in [(args, "args"), (kwargs, "kwargs"), (result, "return")]:
                self._process_post_call_node(
                    val,
                    name,
                    segment,
                    trace_state,
                    seen,
                    depth=0,
                )

        def _process_post_call_node(
            self,
            obj: Any,
            path: str,
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

            if is_tracable_tensor_type(obj):
                if trace_state is not None and isinstance(obj, wp.array):
                    traced_array = TracedWpArray.make_traced_in_place(
                        obj,
                        trace_state.name,
                        trace_state.context,
                        trace_state.proxy,
                    )
                    if segment is not None:
                        segment.add_output_ref(traced_array, path=path)
                        traced_array.warp_segment = segment
                return

            if isinstance(obj, dict):
                for key, value in obj.items():
                    self._process_post_call_node(
                        value,
                        f"{path}[{key!r}]",
                        segment,
                        trace_state,
                        seen,
                        depth=depth + 1,
                    )
            elif isinstance(obj, (list, tuple)):
                for index, item in enumerate(obj):
                    self._process_post_call_node(
                        item,
                        f"{path}[{index}]",
                        segment,
                        trace_state,
                        seen,
                        depth=depth + 1,
                    )
            elif isinstance(obj, (set, frozenset)):
                for index, item in enumerate(obj):
                    self._process_post_call_node(
                        item,
                        f"{path}[{index}]",
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
            if module_name == "warp":
                return True
            if not module_name.startswith("warp."):
                return False
            return not module_name.startswith("warp._")




    ## REMOVE THIS BEFORE MERGING
    if __name__ == "__main__":
        import warp as wp
        import numpy as np
        @wp.kernel
        def add_one_kernel(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float32)):
            i = wp.tid()
            dst[i] = src[i] + 1.0

        src = wp.array(np.ones(10), dtype=wp.float32)
        dst = wp.array(np.zeros(10), dtype=wp.float32)
        wp.launch(add_one_kernel, dim=src.size, inputs=[src], outputs=[dst])
        print(dst)
        detector = WarpLeappCallDetector()
        detector.install()
        print("launch after instasll")
        wp.launch(add_one_kernel, dim=src.size, inputs=[src], outputs=[dst])
        tmp = wp.zeros(src.size, dtype=wp.float32)
        ones = wp.ones(src.size, dtype=wp.float32)
        wp.copy(tmp, ones)
        print(detector.patched_count)
        detector.uninstall()
        print(detector.patched_count)