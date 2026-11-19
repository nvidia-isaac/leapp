#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Locate the checked-in TensorRT catalogs.

The catalogs are kept at ``packages/isaac_deploy_trt/tensorrt_catalogs``.
Wheel builds copy that directory into the installed package, so an installed
package finds it beside this module. A source checkout or editable install
falls back to the catalogs directory next to ``pyproject.toml``.
"""

from __future__ import annotations

from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = Path(__file__).resolve().parents[2]


def catalog_root() -> Path:
    installed = _PACKAGE_DIR / "tensorrt_catalogs"
    if (installed / "index.json").is_file():
        return installed
    return _PROJECT_DIR / "tensorrt_catalogs"
