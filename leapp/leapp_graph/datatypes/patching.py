#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Orchestrate deferred patching for torch and numpy during LEAPP tracing."""

import sys

import torch

from leapp.utils.logging import _get_logger

from .numpy.patching import build_numpy_patches
from .torch.patching import build_torch_patches

_torch_originals, _torch_patches = build_torch_patches()
_numpy_originals, _numpy_patches = build_numpy_patches()
_originals = {**_torch_originals, **_numpy_originals}
_patches = {**_torch_patches, **_numpy_patches}

_patches_applied = False


def apply_traced_data_patches():
    """Apply patches for torch and numpy functions when tracing starts."""
    global _patches_applied
    if not _patches_applied:
        for (module, name), patched in _patches.items():
            setattr(module, name, patched)
        torch.jit._state.disable()
        _patches_applied = True


def remove_traced_data_patches():
    """Restore original torch and numpy functions when tracing stops."""
    global _patches_applied
    if _patches_applied:
        for (module, name), original in _originals.items():
            setattr(module, name, original)
        torch.jit._state.enable()
        _patches_applied = False


def is_patching_enabled():
    return _patches_applied


is_numpy_patching_enabled = is_patching_enabled


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
