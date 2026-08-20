#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Coordinate tracing-session patches across torch, numpy, and warp backends."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Protocol

import torch

from leapp.utils.logging import _get_logger

from .numpy.patching import NumpyPatchBackend
from .torch.patching import TorchPatchBackend

if TYPE_CHECKING:
    from .warp.patching import WarpPatchBackend


class _PatchBackend(Protocol):
    @property
    def installed(self) -> bool: ...

    def install(self) -> None: ...

    def uninstall(self) -> None: ...


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
        self._installed = False

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        """Apply all available backends."""
        if self._installed:
            return

        installed: list[_PatchBackend] = []
        try:
            if self.torch is not None: # install torch patches
                self.torch.install()
                installed.append(self.torch)
            if self.numpy is not None: # install numpy patches
                self.numpy.install()
                installed.append(self.numpy)
            if self.warp is not None: # install warp patches
                self.warp.install()
                installed.append(self.warp)
            self._installed = True
        except Exception:
            for backend in reversed(installed):
                backend.uninstall()
            raise

    def uninstall(self) -> None:
        """Restore every backend that is still installed."""
        if not self._installed:
            return

        if self.warp is not None and self.warp.installed:
            self.warp.uninstall()
        if self.numpy.installed:
            self.numpy.uninstall()
        if self.torch.installed:
            self.torch.uninstall()
        self._installed = False


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
