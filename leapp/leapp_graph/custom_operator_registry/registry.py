#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Generic export hooks for LEAPP custom operators.

Each custom op package (e.g. ``warp_operator``) registers callbacks here at import
time. ``leapp_node.compile_model`` calls :func:`prepare_and_validate` before any
export backend compiles, so backends stay unaware of individual ops.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# (module, backend) -> None
PreCompileHook = Callable[["torch.nn.Module", str], None]
# module -> bool
DetectInModuleHook = Callable[["torch.nn.Module"], bool]
# backend -> error message
UnsupportedMessageHook = Callable[[str], str]


@dataclass(frozen=True)
class CustomOpExportHooks:
    """Export constraints and optional graph prep for one custom op."""

    op_name: str
    detect_in_module: DetectInModuleHook
    supported_backends: frozenset[str]
    unsupported_message: UnsupportedMessageHook
    pre_compile: PreCompileHook | None = None


_HOOKS: list[CustomOpExportHooks] = []


def register_export_hooks(
    *,
    op_name: str,
    detect_in_module: DetectInModuleHook,
    supported_backends: frozenset[str],
    unsupported_message: UnsupportedMessageHook,
    pre_compile: PreCompileHook | None = None,
) -> None:
    """Register export hooks for a custom op (typically at import time)."""
    _HOOKS.append(
        CustomOpExportHooks(
            op_name=op_name,
            detect_in_module=detect_in_module,
            supported_backends=supported_backends,
            unsupported_message=unsupported_message,
            pre_compile=pre_compile,
        )
    )


def prepare_and_validate(module: "torch.nn.Module | None", backend: str) -> None:
    """Validate backend support, then run any registered ``pre_compile`` hooks.

    Called from ``leapp_node.compile_model`` before ``export_backend.compile``.
    Raises ``NotImplementedError`` when the module contains a registered custom op
    that does not support the chosen ``export_with`` backend.
    """
    if module is None:
        return
    active = [hooks for hooks in _HOOKS if hooks.detect_in_module(module)]
    for hooks in active:
        if backend not in hooks.supported_backends:
            raise NotImplementedError(hooks.unsupported_message(backend))
    for hooks in active:
        if hooks.pre_compile is not None:
            hooks.pre_compile(module, backend)
