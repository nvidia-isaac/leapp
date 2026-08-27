#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Coordinate tracing-session patches across torch, numpy, and warp backends."""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch

from leapp.utils.logging import _get_logger

from ._attribute_patching import AttributePatchRegistry
from .numpy.patching import NumpyPatchBackend
from .torch.patching import TorchPatchBackend
from .traced_data import TracedData

if TYPE_CHECKING:
    from .warp.patching import WarpPatchBackend


@dataclass(frozen=True)
class FunctionPatch:
    """Define a traceable replacement for one module-level function."""

    module: ModuleType
    function_name: str
    replacement: Callable[..., Any]


def _create_user_patch(original, replacement):
    @functools.wraps(original)
    def patched(*args, **kwargs):
        active = False

        def find_active(item):
            nonlocal active
            if isinstance(item, TracedData) and item.is_tracing:
                active = True
            return item

        TracedData._map_structure((args, kwargs), find_active)
        if active:
            return replacement(*args, **kwargs)
        return original(*args, **kwargs)

    return patched


def _validate_user_patches(
    patching: Sequence[FunctionPatch] | None,
) -> tuple[FunctionPatch, ...]:
    definitions = () if patching is None else tuple(patching)
    targets: set[tuple[int, str]] = set()

    for definition in definitions:
        if not isinstance(definition, FunctionPatch):
            raise TypeError("patching entries must be FunctionPatch instances")
        if not isinstance(definition.module, ModuleType):
            raise TypeError("FunctionPatch.module must be a Python module")
        if not isinstance(definition.function_name, str) or not definition.function_name:
            raise ValueError(
                "FunctionPatch.function_name must be a non-empty string"
            )
        try:
            target = getattr(definition.module, definition.function_name)
        except AttributeError:
            raise ValueError(
                f"module {definition.module.__name__!r} has no attribute "
                f"{definition.function_name!r}"
            ) from None
        if not callable(target):
            raise TypeError(
                f"{definition.module.__name__}."
                f"{definition.function_name} is not callable"
            )
        if not callable(definition.replacement):
            raise TypeError("FunctionPatch.replacement must be callable")

        key = (id(definition.module), definition.function_name)
        if key in targets:
            raise ValueError(
                f"duplicate patch target "
                f"{definition.module.__name__}.{definition.function_name}"
            )
        targets.add(key)

    return definitions


def _try_create_warp_backend() -> WarpPatchBackend | None:
    try:
        from .warp.patching import WarpPatchBackend

        return WarpPatchBackend()
    except ImportError:
        return None


class TracingPatcher:
    """Install and restore all tracing-session monkeypatches."""

    def __init__(self) -> None:
        self.torch = TorchPatchBackend()
        self.numpy = NumpyPatchBackend()

        self.warp = _try_create_warp_backend()
        self._user_patches = AttributePatchRegistry()
        self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    def install(
        self,
        *,
        patching: Sequence[FunctionPatch] | None = None,
    ) -> None:
        """Apply all available backends."""
        if self._installed:
            return

        definitions = _validate_user_patches(patching)
        try:
            if self.torch is not None:
                self.torch.install()
            if self.numpy is not None:
                self.numpy.install()
            if self.warp is not None:
                self.warp.install()
            for definition in definitions:
                original = getattr(
                    definition.module,
                    definition.function_name,
                )
                patched = _create_user_patch(
                    original,
                    definition.replacement,
                )
                self._user_patches.install(
                    definition.module,
                    definition.function_name,
                    original,
                    patched,
                )
            self._installed = True
        except Exception:
            self._user_patches.restore()
            self._uninstall_defaults()
            raise

    def uninstall(self) -> None:
        """Restore every backend that is still installed."""
        if not self._installed:
            return

        self._user_patches.restore()
        self._uninstall_defaults()
        self._installed = False

    def _uninstall_defaults(self) -> None:
        if self.warp is not None and self.warp.installed:
            self.warp.uninstall()
        if self.numpy.installed:
            self.numpy.uninstall()
        if self.torch.installed:
            self.torch.uninstall()


def get_warp_backend() -> WarpPatchBackend | None:
    """Return the active session's warp backend, if warp-lang is available."""
    from leapp.export_manager import ExportManager

    return ExportManager().patcher.warp


def warn_if_script_functions_in_scope():
    """Warn when TorchScript ScriptFunctions in scope may break tracing."""
    frame = sys._getframe(2)
    suspects = []

    for name, val in frame.f_locals.items():
        if isinstance(val, torch._C.ScriptFunction):
            suspects.append(f"{name} (local)")

    local_names = set(frame.f_locals.keys())
    for name, val in frame.f_globals.items():
        if name not in local_names and isinstance(val, torch._C.ScriptFunction):
            suspects.append(f"{name} (global)")

    if suspects:
        _get_logger().warning(
            f"Detected pre-compiled TorchScript ScriptFunction(s) in scope: {suspects}\n"
            "ScriptFunction.__call__ bypasses TracedTensor tracing — outputs will be "
            "plain torch.Tensors, breaking the trace chain.\n"
            "To fix this, either:\n"
            "  1. Call torch.jit._state.disable() before the @torch.jit.script "
            "decorators run (e.g. before importing the module that defines them)\n"
            "  2. Run the scripting code between leapp.start() and leapp.stop() signals"
            "(leapp disables JIT during tracing automatically)"
        )
