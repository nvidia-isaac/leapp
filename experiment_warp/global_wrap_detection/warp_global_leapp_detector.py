#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Prototype global Warp call profiler for LEAPP-traced Warp arrays.

This detector intentionally ignores ``__array_interface__`` and
``__cuda_array_interface__``. Instead, it patches Python-visible Warp functions
and a small set of important Warp class methods. A call is recorded as a valid
candidate when it:

1. receives at least one LEAPP-traced or detector-tracked Warp array, and
2. returns a Warp array, mutates a Warp array argument/receiver, or has explicit
   output Warp arrays in the function signature.

The detector also propagates tracked status to returned/output/mutated arrays so
raw intermediates can seed later launch detection.
"""

from __future__ import annotations

import functools
import inspect
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Iterable


_WRAPPER_MARKER = "__leapp_warp_detector_wrapper__"
_LEAPP_MARKER_ATTRS = (
    "_leapp_warp_array",
    "_leapp_trace_id",
    "_leapp_tensor_name",
)
_OUTPUT_KEYWORDS = {"dest", "dst", "out", "output", "outputs", "result", "results"}
_ARRAY_MUTATOR_METHODS = {"zero_", "fill_", "assign"}
_ARRAY_RETURNING_METHODS = {"flatten", "reshape", "view"}


@dataclass
class _Patch:
    owner: Any
    attr_name: str
    original: Any
    wrapper: Any


@dataclass(frozen=True)
class WarpArrayRef:
    path: str
    value: Any
    traced: bool
    tracked: bool

    @property
    def active(self) -> bool:
        return self.traced or self.tracked


@dataclass
class WarpCallEvent:
    index: int
    qualname: str
    valid: bool
    reasons: list[str]
    tracked_inputs: list[WarpArrayRef]
    output_arrays: list[WarpArrayRef] = field(default_factory=list)
    mutated_arrays: list[WarpArrayRef] = field(default_factory=list)
    returned_arrays: list[WarpArrayRef] = field(default_factory=list)
    newly_tracked_arrays: list[WarpArrayRef] = field(default_factory=list)
    ignored_reason: str | None = None
    inner_event_count: int = 0


class WarpLeappCallDetector:
    """Install/uninstall wrappers around Python-visible Warp functions."""

    def __init__(
        self,
        *,
        print_fn: Callable[[str], None] = print,
        patch_aliases: bool = True,
        include_private_warp_modules: bool = False,
        verbose_ignored: bool = False,
    ) -> None:
        self.print_fn = print_fn
        self.patch_aliases = patch_aliases
        self.include_private_warp_modules = include_private_warp_modules
        self.verbose_ignored = verbose_ignored
        self.events: list[WarpCallEvent] = []
        self.ignored_events: list[WarpCallEvent] = []
        self._patches: list[_Patch] = []
        self._wrappers_by_original_id: dict[int, Any] = {}
        self._tracked_array_ids: set[int] = set()
        self._tracked_array_ptrs: set[int] = set()
        self._installed = False
        self._wp = None

    @property
    def patched_count(self) -> int:
        return len(self._patches)

    def install(self) -> "WarpLeappCallDetector":
        """Patch currently loaded Warp functions and common array mutators."""

        if self._installed:
            return self

        import warp as wp

        self._wp = wp
        self._patch_warp_modules()
        self._patch_array_methods(wp)
        self._patch_extra_class_methods(wp)

        if self.patch_aliases:
            self._patch_loaded_aliases()

        self._installed = True
        return self

    def rescan(self) -> None:
        """Patch Warp modules or aliases imported after :meth:`install`."""

        if self._wp is None:
            raise RuntimeError("Detector must be installed before rescan().")

        self._patch_warp_modules()
        self._patch_array_methods(self._wp)
        self._patch_extra_class_methods(self._wp)

        if self.patch_aliases:
            self._patch_loaded_aliases()

    def uninstall(self) -> None:
        """Restore every attribute patched by this detector."""

        for patch in reversed(self._patches):
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

    def track_array(self, array: Any) -> None:
        """Seed or propagate detector-tracked status for a Warp array."""

        if self._wp is None:
            import warp as wp

            self._wp = wp

        if not _is_warp_array(array, self._wp):
            return

        self._tracked_array_ids.add(id(array))
        ptr = getattr(array, "ptr", 0)
        if ptr:
            self._tracked_array_ptrs.add(int(ptr))

    def is_tracked_array(self, array: Any) -> bool:
        """Return whether ``array`` is traced or tracked by propagation."""

        if self._wp is None or not _is_warp_array(array, self._wp):
            return False

        if _is_leapp_warp_array(array, self._wp):
            return True
        if id(array) in self._tracked_array_ids:
            return True

        ptr = getattr(array, "ptr", 0)
        return bool(ptr and int(ptr) in self._tracked_array_ptrs)

    def _patch_warp_modules(self) -> None:
        modules = [
            module
            for module_name, module in sys.modules.items()
            if self._is_warp_module(module_name, module)
        ]
        modules.sort(key=lambda module: (module.__name__.count("."), module.__name__))

        for module in modules:
            for attr_name, value in list(vars(module).items()):
                if self._should_wrap_callable(attr_name, value):
                    self._patch_attr(module, attr_name, value, f"{module.__name__}.{attr_name}")

    def _patch_array_methods(self, wp: ModuleType) -> None:
        for class_name in ("array", "indexedarray", "fabricarray", "indexedfabricarray"):
            cls = getattr(wp, class_name, None)
            if cls is None:
                continue

            for method_name in sorted(_ARRAY_MUTATOR_METHODS | _ARRAY_RETURNING_METHODS | {"numpy"}):
                value = getattr(cls, method_name, None)
                if self._should_wrap_callable(method_name, value):
                    self._patch_attr(cls, method_name, value, f"warp.{class_name}.{method_name}")

    def _patch_extra_class_methods(self, wp: ModuleType) -> None:
        method_map = {
            "Tape": ("backward", "zero", "reset"),
            "Launch": ("launch", "set_param_at_index", "set_param_by_name", "set_params"),
        }

        for class_name, method_names in method_map.items():
            cls = getattr(wp, class_name, None)
            if cls is None:
                continue
            for method_name in method_names:
                value = getattr(cls, method_name, None)
                if self._should_wrap_callable(method_name, value):
                    self._patch_attr(cls, method_name, value, f"warp.{class_name}.{method_name}")

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
                self._patch_attr(module, attr_name, value, f"{module.__name__}.{attr_name}", wrapper=wrapper)

    def _patch_attr(
        self,
        owner: Any,
        attr_name: str,
        original: Any,
        qualname: str,
        *,
        wrapper: Any | None = None,
    ) -> None:
        if getattr(original, _WRAPPER_MARKER, False):
            return

        for patch in self._patches:
            if patch.owner is owner and patch.attr_name == attr_name:
                return

        if wrapper is None:
            wrapper = self._wrappers_by_original_id.get(id(original))
        if wrapper is None:
            wrapper = self._make_wrapper(qualname, original)
            self._wrappers_by_original_id[id(original)] = wrapper

        try:
            setattr(owner, attr_name, wrapper)
        except Exception:
            return

        self._patches.append(_Patch(owner, attr_name, original, wrapper))

    def _make_wrapper(self, qualname: str, original: Callable) -> Callable:
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            refs_before = self._find_call_array_refs(args, kwargs)
            active_inputs = [ref for ref in refs_before if ref.active]

            if not active_inputs:
                return original(*args, **kwargs)

            before_event_count = len(self.events)
            result = original(*args, **kwargs)
            inner_event_count = len(self.events) - before_event_count

            event = self._classify_call(
                qualname,
                original,
                args,
                kwargs,
                result,
                refs_before,
                active_inputs,
                inner_event_count,
            )

            if event.valid:
                self.events.append(event)
                for ref in event.newly_tracked_arrays:
                    self.track_array(ref.value)
                self._print_event(event)
            else:
                self.ignored_events.append(event)
                if self.verbose_ignored:
                    self._print_event(event)

            return result

        setattr(wrapped, _WRAPPER_MARKER, True)
        return wrapped

    def _classify_call(
        self,
        qualname: str,
        original: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        refs_before: list[WarpArrayRef],
        active_inputs: list[WarpArrayRef],
        inner_event_count: int,
    ) -> WarpCallEvent:
        reasons: list[str] = []
        output_refs = self._output_refs_for_call(qualname, original, args, kwargs)
        mutated_refs = self._mutated_refs_for_call(qualname, args)
        returned_refs = self._find_array_refs(result, "return")

        record_cmd = bool(kwargs.get("record_cmd", False))
        if qualname.endswith(".launch") and record_cmd:
            return WarpCallEvent(
                index=len(self.events) + len(self.ignored_events),
                qualname=qualname,
                valid=False,
                reasons=[],
                tracked_inputs=active_inputs,
                output_arrays=output_refs,
                returned_arrays=returned_refs,
                ignored_reason="record_cmd=True creates a deferred Launch object; output arrays are not modified yet.",
                inner_event_count=inner_event_count,
            )

        if output_refs:
            reasons.append("has_output_warp_array_args")
        if mutated_refs:
            reasons.append("mutates_warp_array_args")
        if returned_refs:
            reasons.append("returns_warp_array")
        if qualname.endswith("Tape.backward") and inner_event_count > 0:
            reasons.append("emits_adjoint_launches")

        valid = bool(reasons)
        newly_tracked = _dedupe_refs([*output_refs, *mutated_refs, *returned_refs])
        ignored_reason = None if valid else "tracked/traced Warp arrays were present, but no valid output/mutation/return effect was detected."

        return WarpCallEvent(
            index=len(self.events) + len(self.ignored_events),
            qualname=qualname,
            valid=valid,
            reasons=reasons,
            tracked_inputs=active_inputs,
            output_arrays=output_refs,
            mutated_arrays=mutated_refs,
            returned_arrays=returned_refs,
            newly_tracked_arrays=newly_tracked,
            ignored_reason=ignored_reason,
            inner_event_count=inner_event_count,
        )

    def _output_refs_for_call(
        self,
        qualname: str,
        original: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> list[WarpArrayRef]:
        refs: list[WarpArrayRef] = []

        if qualname.endswith(".launch") or qualname.endswith(".launch_tiled"):
            adjoint = bool(kwargs.get("adjoint", False))
            if adjoint:
                refs.extend(self._refs_from_pos_or_kw(args, kwargs, 4, "adj_inputs"))
            else:
                refs.extend(self._refs_from_pos_or_kw(args, kwargs, 3, "outputs"))
            return _dedupe_refs(refs)

        if qualname.endswith(".copy"):
            refs.extend(self._refs_from_pos_or_kw(args, kwargs, 0, "dest"))
            return _dedupe_refs(refs)

        for key in sorted(_OUTPUT_KEYWORDS):
            if key in kwargs:
                refs.extend(self._find_array_refs(kwargs[key], f"kwargs[{key!r}]"))

        try:
            signature = inspect.signature(original)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            parameters = list(signature.parameters)
            for index, name in enumerate(parameters):
                if name in _OUTPUT_KEYWORDS and index < len(args):
                    refs.extend(self._find_array_refs(args[index], f"args[{index}]"))

        return _dedupe_refs(refs)

    def _mutated_refs_for_call(self, qualname: str, args: tuple[Any, ...]) -> list[WarpArrayRef]:
        method_name = qualname.rsplit(".", 1)[-1]
        if method_name in _ARRAY_MUTATOR_METHODS and args:
            return self._find_array_refs(args[0], "args[0]")
        return []

    def _refs_from_pos_or_kw(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        index: int,
        key: str,
    ) -> list[WarpArrayRef]:
        if key in kwargs:
            return self._find_array_refs(kwargs[key], f"kwargs[{key!r}]")
        if index < len(args):
            return self._find_array_refs(args[index], f"args[{index}]")
        return []

    def _find_call_array_refs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[WarpArrayRef]:
        refs: list[WarpArrayRef] = []
        for index, arg in enumerate(args):
            refs.extend(self._find_array_refs(arg, f"args[{index}]"))
        for key, value in kwargs.items():
            refs.extend(self._find_array_refs(value, f"kwargs[{key!r}]"))
        return _dedupe_refs(refs)

    def _find_array_refs(self, value: Any, path: str) -> list[WarpArrayRef]:
        refs: list[WarpArrayRef] = []
        _find_warp_array_refs(value, path, refs, set(), self._wp, self)
        return refs

    def _print_event(self, event: WarpCallEvent) -> None:
        state = "VALID" if event.valid else "ignored"
        pieces = [f"[leapp-warp-detector] {state} {event.qualname}"]
        if event.reasons:
            pieces.append(f"reasons={','.join(event.reasons)}")
        if event.ignored_reason:
            pieces.append(f"reason={event.ignored_reason}")
        if event.inner_event_count:
            pieces.append(f"inner_events={event.inner_event_count}")
        self.print_fn(" ".join(pieces))
        self.print_fn(f"  tracked inputs: {', '.join(_format_refs(event.tracked_inputs))}")
        if event.output_arrays:
            self.print_fn(f"  output arrays: {', '.join(_format_refs(event.output_arrays))}")
        if event.mutated_arrays:
            self.print_fn(f"  mutated arrays: {', '.join(_format_refs(event.mutated_arrays))}")
        if event.returned_arrays:
            self.print_fn(f"  returned arrays: {', '.join(_format_refs(event.returned_arrays))}")
        if event.newly_tracked_arrays:
            self.print_fn(f"  track after call: {', '.join(_format_refs(event.newly_tracked_arrays))}")

    def _is_warp_module(self, module_name: str, module: Any) -> bool:
        if not isinstance(module, ModuleType):
            return False
        if module_name == "warp":
            return True
        if not module_name.startswith("warp."):
            return False
        if self.include_private_warp_modules:
            return True
        return not module_name.startswith("warp._")

    def _should_wrap_callable(self, attr_name: str, value: Any) -> bool:
        if attr_name.startswith("__"):
            return False
        if value is None or getattr(value, _WRAPPER_MARKER, False):
            return False
        if inspect.isclass(value):
            return False
        return inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value)


def install_global_detector(**kwargs) -> WarpLeappCallDetector:
    """Convenience installer used by the POC script."""

    return WarpLeappCallDetector(**kwargs).install()


def find_leapp_warp_arrays(value: Any, *, wp=None) -> list[str]:
    """Return stable argument paths for LEAPP-traced Warp arrays in ``value``."""

    if wp is None:
        import warp as wp

    matches: list[str] = []
    _find_leapp_warp_arrays(value, "call", matches, set(), wp)
    return matches


def known_compute_like_surface() -> dict[str, Iterable[str]]:
    """Document the Warp operations this POC smoke-tests or flags as relevant."""

    return {
        "module_functions": (
            "warp.launch",
            "warp.launch_tiled",
            "warp.copy",
            "warp.clone",
            "warp.empty_like",
            "warp.zeros_like",
            "warp.full_like",
            "warp.to_torch",
        ),
        "array_methods": (
            "warp.array.zero_",
            "warp.array.fill_",
            "warp.array.assign",
            "warp.array.flatten",
            "warp.array.numpy",
        ),
        "class_methods": (
            "warp.Tape.backward",
            "warp.Launch.launch",
            "warp.Launch.set_param_at_index",
            "warp.Launch.set_param_by_name",
            "warp.Launch.set_params",
        ),
    }


def _find_warp_array_refs(
    value: Any,
    path: str,
    refs: list[WarpArrayRef],
    seen: set[int],
    wp: ModuleType,
    detector: WarpLeappCallDetector,
) -> None:
    value_id = id(value)
    if value_id in seen:
        return

    if _is_warp_array(value, wp):
        traced = _is_leapp_warp_array(value, wp)
        refs.append(
            WarpArrayRef(
                path=path,
                value=value,
                traced=traced,
                tracked=traced or detector.is_tracked_array(value),
            )
        )
        return

    if _is_container(value):
        seen.add(value_id)

    if isinstance(value, dict):
        for key, child in value.items():
            _find_warp_array_refs(child, f"{path}[{key!r}]", refs, seen, wp, detector)
        return

    if isinstance(value, tuple):
        labels = getattr(value, "_fields", None)
        for index, child in enumerate(value):
            field = labels[index] if labels else index
            _find_warp_array_refs(child, f"{path}[{field!r}]", refs, seen, wp, detector)
        return

    if isinstance(value, (list, set, frozenset)):
        for index, child in enumerate(value):
            _find_warp_array_refs(child, f"{path}[{index}]", refs, seen, wp, detector)
        return


def _find_leapp_warp_arrays(
    value: Any,
    path: str,
    matches: list[str],
    seen: set[int],
    wp: ModuleType,
) -> None:
    value_id = id(value)
    if value_id in seen:
        return

    if _is_container(value):
        seen.add(value_id)

    if _is_leapp_warp_array(value, wp):
        matches.append(f"{path}={_describe_array(value)}")
        return

    if isinstance(value, dict):
        for key, child in value.items():
            _find_leapp_warp_arrays(child, f"{path}[{key!r}]", matches, seen, wp)
        return

    if isinstance(value, tuple):
        labels = getattr(value, "_fields", None)
        for index, child in enumerate(value):
            field = labels[index] if labels else index
            _find_leapp_warp_arrays(child, f"{path}[{field!r}]", matches, seen, wp)
        return

    if isinstance(value, (list, set, frozenset)):
        for index, child in enumerate(value):
            _find_leapp_warp_arrays(child, f"{path}[{index}]", matches, seen, wp)
        return


def _is_container(value: Any) -> bool:
    return isinstance(value, (dict, list, tuple, set, frozenset))


def _is_warp_array(value: Any, wp: ModuleType) -> bool:
    try:
        return isinstance(value, wp.array)
    except Exception:
        return False


def _is_leapp_warp_array(value: Any, wp: ModuleType) -> bool:
    try:
        from leapp.leapp_graph.datatypes import TracedData
        from leapp.leapp_graph.datatypes.traced_warp_array import TracedWarpArray
    except Exception:
        TracedData = None
        TracedWarpArray = None

    if TracedWarpArray is not None and isinstance(value, TracedWarpArray):
        return True

    if TracedData is not None and isinstance(value, TracedData):
        try:
            unwrapped = TracedData.unwrap_traced_data(value)
        except Exception:
            unwrapped = getattr(value, "data", None)
        return isinstance(unwrapped, wp.array)

    if isinstance(value, wp.array):
        if all(hasattr(value, attr) for attr in ("_context", "_proxy", "_name")):
            return True
        return any(bool(getattr(value, attr, False)) for attr in _LEAPP_MARKER_ATTRS)

    return any(bool(getattr(value, attr, False)) for attr in _LEAPP_MARKER_ATTRS)


def _describe_array(value: Any) -> str:
    pieces = [type(value).__name__]
    name = getattr(value, "name", None) or getattr(value, "_name", None)
    if name:
        pieces.append(f"name={name}")

    for attr in ("shape", "dtype", "device"):
        attr_value = getattr(value, attr, None)
        if attr_value is not None:
            pieces.append(f"{attr}={attr_value}")

    ptr = getattr(value, "ptr", 0)
    if ptr:
        pieces.append(f"ptr=0x{int(ptr):x}")

    return "(" + ", ".join(pieces) + ")"


def _format_refs(refs: list[WarpArrayRef]) -> list[str]:
    return [
        f"{ref.path}:{'traced' if ref.traced else 'tracked'}{_describe_array(ref.value)}"
        for ref in refs
    ]


def _dedupe_refs(refs: list[WarpArrayRef]) -> list[WarpArrayRef]:
    deduped: list[WarpArrayRef] = []
    seen: set[tuple[int, int]] = set()
    for ref in refs:
        key = (id(ref.value), int(getattr(ref.value, "ptr", 0) or 0))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped
