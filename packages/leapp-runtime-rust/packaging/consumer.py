# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lock a public Python core or source-built core to selected backend packages."""

import argparse
import json
from pathlib import Path
import tomllib

from sdk import ROOT, jetson_release, native_platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--channel",
        type=Path,
        help="Local backend channel; optional for core-only installs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binding", choices=["python", "cxx", "both", "none"], default="python"
    )
    parser.add_argument("--core-source", choices=["pypi", "channel"], default="pypi")
    parser.add_argument(
        "--runtime-version", help="Core version for an environment without a channel"
    )
    parser.add_argument(
        "--engines", nargs="+", choices=["native", "triton"], default=["native"]
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        choices=["onnx", "torchscript", "nativert", "tensorrt"],
        default=[],
    )
    parser.add_argument("--warp", nargs="*", choices=["onnx", "nativert"], default=[])
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--onnx-tensorrt", action="store_true")
    args = parser.parse_args()
    if "tensorrt" in args.backends and "triton" not in args.engines:
        parser.error("precompiled TensorRT engines require --engines triton")
    channel = args.channel.resolve() if args.channel else None
    report = (
        json.loads((channel / "packages.json").read_text())
        if channel
        else {
            "version": args.runtime_version
            or tomllib.loads((ROOT / "Cargo.toml").read_text())["package"]["version"],
            "platform": native_platform(),
            "glibc": "2.28",
            "l4t": None,
            "engines": ["native"],
            "versions": {},
            "packages": {},
        }
    )
    if args.runtime_version and args.runtime_version != report["version"]:
        parser.error(
            "--runtime-version must match the backend channel's runtime version"
        )
    if args.core_source == "pypi" and args.binding in {"cxx", "both"}:
        parser.error(
            "C++ consumers use --binding none and build with CMake; local binaries require --core-source channel"
        )
    if args.core_source == "channel" and not channel:
        parser.error("--core-source channel requires --channel")
    if not set(args.engines) <= set(report["engines"]):
        parser.error("Selected engines have not been built in this channel")
    if report["l4t"] and jetson_release() != report["l4t"]:
        parser.error(
            f"This channel requires Jetson Linux {report['l4t']}; generate the environment on that target"
        )
    if not set(args.warp) <= set(args.backends):
        parser.error("Select each Warp adapter's backend as well")
    if ("tensorrt" in args.backends or args.warp) and not args.cuda:
        parser.error("TensorRT and Warp require --cuda")
    if args.onnx_tensorrt and ("onnx" not in args.backends or not args.cuda):
        parser.error("--onnx-tensorrt requires --backends onnx and --cuda")
    if args.cuda and "cuda" not in report["versions"]:
        parser.error("This channel has no CUDA runtime")
    if args.cuda and "onnx" in args.backends and not report.get("onnx_cuda"):
        parser.error("Rebuild the ONNX backend channel with --cuda")
    if (
        channel
        and args.core_source == "pypi"
        and args.binding == "python"
        and not report.get("runtime_release")
    ):
        parser.error(
            "PyPI cores require plugins built from the clean matching runtime-vVERSION tag; use --core-source channel for development builds"
        )
    bindings = (
        ["python", "cxx"]
        if args.binding == "both"
        else ([] if args.binding == "none" else [args.binding])
    )
    selected = ["leapp-runtime-environment"] if channel else []
    if args.core_source == "channel":
        selected += [f"leapp-runtime-{binding}" for binding in bindings]
    for engine in sorted(set(args.engines)):
        kind = "backend" if engine == "native" else "triton"
        selected += [
            f"leapp-{kind}-{name}"
            for name in args.backends
            if engine != "native" or name != "tensorrt"
        ]
        selected += [f"leapp-warp-{engine}-{name}" for name in args.warp]
    if "triton" in args.engines:
        selected.append("leapp-libs-triton")
    if args.cuda:
        selected.append("leapp-libs-cudart")
    if args.onnx_tensorrt:
        selected.append("leapp-onnx-tensorrt")
    missing = set(selected) - report["packages"].keys()
    if missing:
        parser.error(f"Packages have not been built: {sorted(missing)}")
    platform = f'{{ platform = "{report["platform"]}", glibc = "{report["glibc"]}"'
    if args.cuda:
        platform += f', cuda = "{report["versions"]["cuda"]}"'
    platform += " }"
    text = (
        f'[workspace]\nname = "leapp-consumer"\nrequires-pixi = ">=0.84.0"\n'
        f'channels = [{json.dumps(channel.as_uri()) + ", " if channel else ""}"conda-forge"]\n'
        f"platforms = [{platform}]\n\n[dependencies]\n"
    )
    for name in selected:
        text += f'{name} = {{ version = "=={report["version"]}", build = "{report["build"]}" }}\n'
    if args.core_source == "pypi" and args.binding == "python":
        text += 'python = "3.12.*"\n\n[pypi-dependencies]\n'
        text += f'leapp-runtime = "=={report["version"]}"\n'
    # Explicit overrides are preserved. Pixi expands these at activation, not
    # when writing the manifest, so another install prefix works unchanged.
    text += (
        "\n[activation.env]\n"
        'LEAPP_NATIVE_BACKENDS = "${LEAPP_NATIVE_BACKENDS:-$CONDA_PREFIX/lib/leapp/native-backends}"\n'
        'LEAPP_TRITON_BACKENDS = "${LEAPP_TRITON_BACKENDS:-$CONDA_PREFIX/lib/leapp/triton-backends}"\n'
        'TRITON_ROOT = "${TRITON_ROOT:-$CONDA_PREFIX/lib/leapp/triton}"\n'
        'CUDA_ROOT = "${CUDA_ROOT:-$CONDA_PREFIX/lib/leapp/cuda}"\n'
    )
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "pixi.toml"
    if manifest.exists():
        parser.error(
            f"Refusing to overwrite {manifest}; edit its dependencies with pixi add/remove"
        )
    manifest.write_text(text)
    print(f"Created {manifest}. Run: pixi install --manifest-path {manifest}")


if __name__ == "__main__":
    main()
