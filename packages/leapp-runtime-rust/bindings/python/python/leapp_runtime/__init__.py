# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-independent LEAPP bundle loading and inspection."""

from enum import IntEnum

from ._native import (
    Bundle,
    BundleError,
    MetadataMapping,
    PortRef,
    PortSpec,
)


class ErrorCode(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    NOT_FOUND = 2
    UNSUPPORTED = 3
    IO = 4
    INTERNAL = 5


__all__ = [
    "Bundle",
    "BundleError",
    "ErrorCode",
    "MetadataMapping",
    "PortRef",
    "PortSpec",
]
