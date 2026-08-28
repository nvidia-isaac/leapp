#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Locate LEAPP's native Warp runtime adapters."""

from __future__ import annotations

import os
import sys
from pathlib import Path


WARP_RUNTIME_ENVIRONMENTS = {
    "onnx": "LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY",
    "pt2": "LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY",
}


def _shared_library_name(target: str) -> str:
    if sys.platform == "win32":
        return f"{target}.dll"
    if sys.platform == "darwin":
        return f"lib{target}.dylib"
    return f"lib{target}.so"


def warp_runtime_build_dir() -> Path:
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        cache_root = (
            Path(local_app_data).expanduser()
            if local_app_data
            else Path.home() / "AppData" / "Local"
        )
        return cache_root / "leapp" / "warp-runtime" / "build"

    cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = (
        Path(cache_home).expanduser()
        if cache_home
        else Path.home() / ".cache"
    )
    return cache_root / "leapp" / "warp-runtime" / "build"


def warp_runtime_artifact_paths(
    build_dir: str | Path | None = None,
) -> dict[str, Path]:
    output_dir = (
        Path(build_dir).expanduser().resolve()
        if build_dir is not None
        else warp_runtime_build_dir()
    )
    return {
        "onnx": output_dir
        / _shared_library_name("leapp_wrp_onnx_custom_op"),
        "pt2": output_dir
        / _shared_library_name("leapp_wrp_torch_custom_op"),
    }


def resolve_warp_runtime_library(backend: str) -> Path:
    """Resolve one adapter, preferring its explicit environment override."""
    if backend not in WARP_RUNTIME_ENVIRONMENTS:
        raise ValueError(f"Unknown Warp runtime backend: {backend}")

    env_name = WARP_RUNTIME_ENVIRONMENTS[backend]
    configured_path = os.environ.get(env_name)
    if configured_path:
        path = Path(configured_path).expanduser()
        if path.is_file():
            return path
        raise FileNotFoundError(f"{env_name} does not exist: {path}")

    path = warp_runtime_artifact_paths()[backend]
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"LEAPP Warp {backend} runtime was not found at {path}. "
        f"Run leapp-build-warp-runtime or set {env_name}."
    )
