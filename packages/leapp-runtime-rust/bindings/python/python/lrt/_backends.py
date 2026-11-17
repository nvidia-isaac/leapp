# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover optional pip SDK wheels when a session first opens."""

from functools import cache
from importlib.metadata import version
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import re


@cache
def configure():
    """Validate one installed SDK build and preserve explicit/Pixi overrides."""
    if "LEAPP_NATIVE_BACKENDS" in os.environ or "ORT_DYLIB_PATH" in os.environ:
        return
    spec = find_spec("_leapp_rt_backends")
    if spec is None:
        return
    roots = list(spec.submodule_search_locations or [])
    if len(roots) != 1:
        raise RuntimeError(
            "Optional LEAPP backend wheels must share one installation prefix"
        )
    root = Path(roots[0])
    manifests = [
        json.loads(path.read_text())
        for path in sorted((root / "metadata").glob("*.json"))
    ]
    if not manifests:
        raise RuntimeError("Optional LEAPP backend wheels have no SDK metadata")
    if len({item["build"] for item in manifests}) != 1:
        raise RuntimeError(
            "Optional LEAPP backend wheels contain incompatible SDK builds; reinstall a matching set"
        )
    core = version("leapp-rt")
    release = Path("/etc/nv_tegra_release")
    l4t = (
        re.search(r"R(\d+)\s+\(release\),\s*REVISION:\s*([\d.]+)", release.read_text())
        if release.exists()
        else None
    )
    actual_l4t = f"{l4t[1]}.{l4t[2]}" if l4t else None
    target = {"x86_64": "linux-64", "aarch64": "linux-aarch64"}.get(platform.machine())
    for item in manifests:
        if item["platform"] != target:
            raise RuntimeError(
                f"Backend {item['package']} targets {item['platform']}, not {platform.machine()}"
            )
        if item["version"] != core:
            raise RuntimeError(
                f"Backend {item['package']} requires leapp-rt {item['version']}; installed {core}"
            )
        if item["l4t"] and item["l4t"] != actual_l4t:
            raise RuntimeError(
                f"Backend {item['package']} requires Jetson Linux {item['l4t']}"
            )
        if tuple(map(int, platform.libc_ver()[1].split("."))) < tuple(
            map(int, item["glibc"].split("."))
        ):
            raise RuntimeError(
                f"Backend {item['package']} requires glibc {item['glibc']}"
            )
    libraries = root / "lib"
    onnx = sorted(libraries.glob("libonnxruntime.so*"))
    defaults = {}
    if onnx:
        defaults["ORT_DYLIB_PATH"] = str(onnx[0])
    if (root / "backends").is_dir():
        defaults["LEAPP_NATIVE_BACKENDS"] = str(root / "backends")
    if list(libraries.glob("libcudart.so*")):
        defaults["CUDA_ROOT"] = str(root)
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
