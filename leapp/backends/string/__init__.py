# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental operation capture for strings, containers and native resources."""

# Version of the metadata embedded in exported string models.
FORMAT = "leapp-utf8-fx-4"


class StringCaptureError(RuntimeError):
    """An operation cannot be captured without changing its behavior."""
