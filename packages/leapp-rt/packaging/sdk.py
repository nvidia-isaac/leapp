# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inspect the supplied SDKs and native target without machine-specific profiles."""

import ctypes
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def native_platform():
    targets = {"x86_64": "linux-64", "aarch64": "linux-aarch64"}
    if sys.platform != "linux" or platform.machine() not in targets:
        raise RuntimeError("Build natively on Linux x86-64 or aarch64")
    return targets[platform.machine()]


def jetson_release(path=Path("/etc/nv_tegra_release")):
    if not path.exists():
        return None
    match = re.search(r"R(\d+)\s+\(release\),\s*REVISION:\s*([\d.]+)", path.read_text())
    if not match:
        raise RuntimeError(f"Cannot read the Jetson Linux release from {path}")
    return f"{match[1]}.{match[2]}"


def header_version(header, *macros):
    fields = subprocess.check_output(
        ["c++", "-E", "-P", "-x", "c++", "-include", str(header), "-"],
        input=" ".join(macros) + "\n",
        text=True,
    ).split()
    if len(fields) != len(macros) or not all(value.isdecimal() for value in fields):
        raise RuntimeError(f"Cannot read version macros from {header}: {fields}")
    return fields


def inspect_sdk(sdk, backends):
    """Read versions before compilation; leave adapter checks in their build files."""
    versions = {}
    if sdk.get("triton"):
        triton = Path(sdk["triton"])
        version = (triton / "TRITON_VERSION").read_text().strip()
        supported = re.search(
            r'const TRITON_VERSION: &str = "([^"]+)";',
            (ROOT / "src/engines/triton/native.rs").read_text(),
        )[1]
        if version != supported:
            raise RuntimeError(
                f"This runtime requires Triton {supported}, found {version}"
            )
        versions["triton"] = version
    if sdk.get("cuda"):
        cuda = cuda_library(sdk)
        runtime_version = ctypes.c_int()
        if (
            ctypes.CDLL(str(cuda)).cudaRuntimeGetVersion(ctypes.byref(runtime_version))
            != 0
        ):
            raise RuntimeError("Cannot read CUDA runtime version")
        value = runtime_version.value
        versions["cuda"] = f"{value // 1000}.{value % 1000 // 10}"
    if "onnx" in backends:
        ort = ctypes.CDLL(str(onnx_library(sdk)))

        class ApiBase(ctypes.Structure):
            _fields_ = [
                ("get_api", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)),
                ("version", ctypes.CFUNCTYPE(ctypes.c_char_p)),
            ]

        ort.OrtGetApiBase.restype = ctypes.POINTER(ApiBase)
        api = ort.OrtGetApiBase().contents
        features = tomllib.loads((ROOT / "Cargo.toml").read_text())["dependencies"][
            "ort"
        ]["features"]
        api_version = max(
            int(f.removeprefix("api-")) for f in features if f.startswith("api-")
        )
        if not api.get_api(api_version):
            raise RuntimeError(f"ONNX Runtime must support C API {api_version}")
        versions["onnxruntime"] = api.version().decode()
    if sdk["tensorrt"]:
        headers = sorted(
            (Path(sdk["tensorrt"]) / "include").glob("**/NvInferVersion.h")
        )
        if not headers:
            raise RuntimeError("Missing TensorRT version header")
        versions["tensorrt"] = ".".join(
            header_version(
                headers[0],
                "NV_TENSORRT_MAJOR",
                "NV_TENSORRT_MINOR",
                "NV_TENSORRT_PATCH",
                "NV_TENSORRT_BUILD",
            )
        )
    if sdk["torch"]:
        torch = Path(sdk["torch"])
        header = torch / "include/torch/headeronly/version.h"
        if not header.exists():
            header = torch / "include/torch/csrc/api/include/torch/version.h"
        versions["torch"] = ".".join(
            header_version(
                header,
                "TORCH_VERSION_MAJOR",
                "TORCH_VERSION_MINOR",
                "TORCH_VERSION_PATCH",
            )
        )
        # Match the adapters' ABI using a string-returning LibTorch symbol.
        # Recent Torch SDKs no longer publish this flag in TorchConfig.cmake.
        with tempfile.TemporaryDirectory(prefix="leapp-torch-abi-") as directory:
            result = subprocess.run(
                [
                    "c++",
                    "-std=c++20",
                    "-D_GLIBCXX_USE_CXX11_ABI=1",
                    f"-I{torch / 'include'}",
                    "-x",
                    "c++",
                    "-",
                    "-x",
                    "none",
                    str(torch / "lib/libc10.so"),
                    "-o",
                    str(Path(directory) / "check"),
                ],
                input="#include <c10/core/Device.h>\nint main() { return c10::Device(c10::DeviceType::CPU).str().empty(); }\n",
                text=True,
                capture_output=True,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(
                f"LibTorch failed the adapters' C++11 ABI (1) link check:\n{result.stderr}"
            )
        versions["cxx11_abi"] = 1
    return versions


def cuda_library(sdk):
    for directory in ("lib64", "lib"):
        library = Path(sdk["cuda"]) / directory / "libcudart.so"
        if library.is_file():
            return library
    raise RuntimeError("CUDA_ROOT must contain lib64/libcudart.so or lib/libcudart.so")


def onnx_library(sdk):
    if sdk.get("onnx"):
        return Path(sdk["onnx"])
    if sdk.get("triton"):
        return Path(sdk["triton"]) / "backends/onnxruntime/libonnxruntime.so"
    raise RuntimeError("Set ORT_DYLIB_PATH to the ONNX Runtime shared library")


def verify_binary(path, target, glibc):
    with path.open("rb") as stream:
        header = stream.read(20)
    machine = {"linux-64": 62, "linux-aarch64": 183}[target]
    if (
        header[:6] != b"\x7fELF\x02\x01"
        or int.from_bytes(header[18:20], "little") != machine
    ):
        raise RuntimeError(f"{path} is not a 64-bit {target} ELF binary")
    text = subprocess.check_output(["readelf", "--version-info", str(path)], text=True)
    required = [
        tuple(map(int, value.split(".")))
        for value in re.findall(r"GLIBC_([0-9.]+)", text)
    ]
    if required and max(required) > tuple(map(int, glibc.split("."))):
        version = ".".join(map(str, max(required)))
        raise RuntimeError(
            f"{path} requires glibc {version}, above declared baseline {glibc}"
        )
