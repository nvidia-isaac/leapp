# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synchronous LEAPP execution with retained NumPy and DLPack outputs."""

from dataclasses import dataclass
from enum import IntEnum
from os import PathLike

from ._native import (
    Bundle,
    BundleError,
    DLPackView,
    MetadataMapping,
    PortRef,
    PortSpec,
    RunResult,
    Runtime,
    RuntimeError,
    Session,
    Tensor,
)


class ErrorCode(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    NOT_FOUND = 2
    UNSUPPORTED = 3
    IO = 4
    INTERNAL = 5


@dataclass(frozen=True)
class RuntimeOptions:
    """Model placement, independent of input/output storage."""

    device: str = "cpu"
    backend_directory: str | PathLike = "/opt/tritonserver/backends"


__all__ = [
    "Bundle",
    "BundleError",
    "DLPackView",
    "ErrorCode",
    "MetadataMapping",
    "PortRef",
    "PortSpec",
    "RunResult",
    "Runtime",
    "RuntimeError",
    "RuntimeOptions",
    "Session",
    "Tensor",
]
