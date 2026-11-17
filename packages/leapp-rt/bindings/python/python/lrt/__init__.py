# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronous LEAPP execution with retained NumPy and DLPack outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from ._native import (
    Bundle,
    BundleError,
    DLPackOutput,
    MetadataMapping,
    PortRef,
    PortSpec,
    RunResult,
    RuntimeError,
    Session,
    _DEFAULT_ENGINE,
)


class ErrorCode(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    NOT_FOUND = 2
    UNSUPPORTED = 3
    IO = 4
    INTERNAL = 5


@dataclass(frozen=True)
class SessionOptions:
    """Construction-time engine and device placement.

    See documentation for onnx_tensorrt options.
    """

    device: str = "cpu"
    output_device: str = "cpu"
    output_overrides: dict[str, str] = field(default_factory=dict)
    engine: str = _DEFAULT_ENGINE
    onnx_tensorrt: dict[str, dict[str, str]] = field(default_factory=dict)


__all__ = [
    "Bundle",
    "BundleError",
    "DLPackOutput",
    "ErrorCode",
    "MetadataMapping",
    "PortRef",
    "PortSpec",
    "RunResult",
    "RuntimeError",
    "SessionOptions",
    "Session",
]
