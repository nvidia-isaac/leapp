# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build once on a native Linux SDK host; stage independently installable outputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tomllib

from sdk import ROOT, inspect_sdk, jetson_release, native_platform

BACKENDS = ("onnx", "torchscript", "nativert", "tensorrt")


def run(*args, **kwargs):
    print("+", *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def required(variable):
    value = os.environ.get(variable)
    if not value or not Path(value).exists():
        raise SystemExit(f"Set {variable} to an existing SDK path")
    return Path(value).resolve()


def cmake(source, build, prefix, *options):
    run(
        "cmake",
        "-S",
        ROOT / source,
        "-B",
        build,
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_TESTING=OFF",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        f"-DCMAKE_INSTALL_PREFIX={prefix}",
        *options,
    )
    run(
        "cmake",
        "--build",
        build,
        "--parallel",
        os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "2"),
    )
    run("cmake", "--install", build)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glibc", required=True, help="Minimum deployment glibc version, e.g. 2.39"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "target/packaging")
    parser.add_argument(
        "--backends", nargs="*", choices=BACKENDS, default=list(BACKENDS)
    )
    parser.add_argument(
        "--warp", type=Path, help="Directory containing built Warp ONNX/PT2 adapters"
    )
    parser.add_argument(
        "--sdk-origin",
        required=True,
        help="Immutable SDK image digest or artifact URL and SHA256",
    )
    args = parser.parse_args()
    target = native_platform()
    if not re.fullmatch(r"[0-9]+\.[0-9]+", args.glibc):
        parser.error("--glibc must be a major.minor version, e.g. 2.39")
    if platform.libc_ver()[0] != "glibc":
        parser.error("The native Linux build requires glibc")
    if os.environ.get("LEAPP_ANALYSIS_ONLY"):
        parser.error("LEAPP_ANALYSIS_ONLY cannot produce deployment binaries")
    triton, cuda = required("TRITON_ROOT"), required("CUDA_ROOT")
    torch = (
        required("TORCH_ROOT")
        if set(args.backends) & {"torchscript", "nativert"}
        else None
    )
    pytorch = required("PYTORCH_SOURCE") if "nativert" in args.backends else None
    sdk = {
        "triton": str(triton),
        "cuda": str(cuda),
        "torch": str(torch) if torch else None,
        "tensorrt": str(required("TENSORRT_ROOT"))
        if "tensorrt" in args.backends
        else None,
    }
    versions = inspect_sdk(sdk, args.backends)
    versions["python"] = f"{sys.version_info.major}.{sys.version_info.minor}"
    out = args.output.resolve()
    stage = out / "stage"
    if stage.exists():
        parser.error(
            f"{stage} already exists; select a new output directory for a clean build"
        )
    stage.mkdir(parents=True)
    version = tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"]
    source_hash = hashlib.sha256()
    # Include uncommitted source. Ignore caches and build products, not source edits.
    for path in sorted(ROOT.rglob("*")):
        if any(
            part in {"target", ".pixi", ".venv", "__pycache__", ".pytest_cache"}
            for part in path.relative_to(ROOT).parts
        ):
            continue
        if path.is_file():
            source_hash.update(str(path.relative_to(ROOT)).encode())
            source_hash.update(bytes.fromhex(digest(path)))
    provenance = {
        "platform": target,
        "version": version,
        "source_sha256": source_hash.hexdigest(),
        "sdk_origin": args.sdk_origin,
        "versions": versions,
        "glibc": args.glibc,
        "l4t": jetson_release(),
        "backends": args.backends,
        "host": platform.platform(),
        "tools": {},
    }
    for tool, flag in [
        ("rustc", "--version"),
        ("cmake", "--version"),
        ("c++", "--version"),
    ]:
        provenance["tools"][tool] = subprocess.check_output(
            [tool, flag], text=True
        ).splitlines()[0]
    if pytorch:
        source_files = []
        for directory in [
            "torch/nativert",
            "torch/csrc/inductor/aoti_torch",
            "third_party/nlohmann",
            "third_party/concurrentqueue",
        ]:
            source_files.extend(
                p for p in (pytorch / directory).rglob("*") if p.is_file()
            )
        source_digest = hashlib.sha256()
        for path in sorted(source_files):
            source_digest.update(str(path.relative_to(pytorch)).encode())
            source_digest.update(bytes.fromhex(digest(path)))
        provenance["nativert_source_sha256"] = source_digest.hexdigest()
    (out / "build.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "sdk.json").write_text(json.dumps(sdk, indent=2) + "\n")
    os.environ["CARGO_TARGET_DIR"] = str(out / "cargo")
    os.environ["PYO3_PYTHON"] = sys.executable
    wheel_dir = out / "wheels"
    run(
        "maturin",
        "build",
        "--locked",
        "--release",
        "--manifest-path",
        ROOT / "bindings/python/Cargo.toml",
        "--auditwheel",
        "skip",
        "--out",
        wheel_dir,
        "--interpreter",
        sys.executable,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one wheel, found {wheels}")
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-compile",
        "--no-index",
        "--prefix",
        stage / "leapp-runtime-python",
        wheels[0],
    )
    # pip's direct_url records the producer path, which is not part of the install.
    for path in (stage / "leapp-runtime-python").rglob("direct_url.json"):
        path.unlink()
    os.environ.pop("CARGO_TARGET_DIR")
    cmake(
        "bindings/cxx",
        out / "build-cxx",
        stage / "leapp-runtime-cxx",
        f"-DTRITON_ROOT={triton}",
        f"-DCUDA_ROOT={cuda}",
    )
    if pytorch:
        nativert = stage / "leapp-libs-nativert"
        cmake(
            "src/engines/triton/backends/torch/nativert",
            out / "build-nativert-library",
            nativert,
            f"-DPYTORCH_SOURCE={pytorch}",
            f"-DTORCH_ROOT={torch}",
        )
        os.environ["NATIVERT_LIBRARY"] = str(nativert / "lib/libleapp_nativert.so")
    for backend in args.backends:
        prefix = stage / f"leapp-backend-{backend}"
        native = prefix / "lib/leapp/native-backends"
        if backend == "onnx":
            destination = prefix / "lib/leapp/triton-backends/onnxruntime"
            destination.mkdir(parents=True)
            shutil.copy2(
                triton / "backends/onnxruntime/libtriton_onnxruntime.so", destination
            )
            continue
        options = [f"-DCUDA_ROOT={cuda}", f"-DCUDAToolkit_ROOT={cuda}"]
        if torch and backend != "tensorrt":
            options += [f"-DTORCH_ROOT={torch}"]
        if backend == "nativert":
            options += [
                f"-DPYTORCH_SOURCE={pytorch}",
                f"-DNATIVERT_LIBRARY={os.environ['NATIVERT_LIBRARY']}",
            ]
        if backend == "tensorrt":
            options += [f"-DTENSORRT_ROOT={required('TENSORRT_ROOT')}"]
        cmake(
            f"src/engines/native/backends/{backend}",
            out / f"build-{backend}",
            native,
            *options,
        )
        if backend == "tensorrt":
            destination = prefix / "lib/leapp/triton-backends/tensorrt"
            destination.mkdir(parents=True)
            shutil.copy2(
                triton / "backends/tensorrt/libtriton_tensorrt.so", destination
            )
        else:
            name = "jit" if backend == "torchscript" else "pt2"
            temporary = out / f"install-triton-{name}"
            cmake(
                f"src/engines/triton/backends/torch/{name}",
                out / f"build-triton-{name}",
                temporary,
                f"-DTRITON_ROOT={triton}",
                *options,
            )
            shutil.copytree(
                temporary / "backends",
                prefix / "lib/leapp/triton-backends",
                dirs_exist_ok=True,
            )
    if args.warp:
        for backend, library, native_name, triton_name in [
            ("onnx", "libleapp_wrp_onnx_custom_op.so", "onnxruntime", "onnxruntime"),
            ("nativert", "libleapp_wrp_torch_custom_op.so", "nativert", "leapp_pt2"),
        ]:
            if backend not in args.backends:
                continue
            source = args.warp.resolve() / library
            if not source.is_file():
                raise RuntimeError(f"Missing Warp adapter: {source}")
            prefix = stage / f"leapp-warp-{backend}"
            for engine, name in [("native", native_name), ("triton", triton_name)]:
                dest = prefix / f"lib/leapp/{engine}-backends/{name}"
                dest.mkdir(parents=True)
                shutil.copy2(source, dest)
    print(f"Built outputs in {stage}; next run package-runtime --input {out}")


if __name__ == "__main__":
    main()
