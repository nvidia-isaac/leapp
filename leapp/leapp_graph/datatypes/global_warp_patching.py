from typing import Any, Callable
from types import ModuleType
from warp._src.context import Function as WarpKernelLanguageFunction
import functools
from dataclasses import dataclass
import inspect
import sys


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



class WarpLeappCallDetector:
    def __init__(self) -> None:
        self._patches: list[_Patch] = []
        self._wrappers_by_original_id: dict[int, Any] = {}
        self._recording_depth = 0
        self._installed = False
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

    def uninstall(self) -> None:
        """Restore every attribute patched by this detector."""

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

            print(f"Calling {qualname}")

            self._recording_depth += 1
            try:
                result = original(*args, **kwargs)
            finally:
                self._recording_depth -= 1

            return result

        setattr(wrapped, _WRAPPER_MARKER, True)
        return wrapped




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