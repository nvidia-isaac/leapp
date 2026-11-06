# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a small consumer workspace from a built local channel."""

import argparse
import json
from pathlib import Path

from sdk import jetson_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--binding", choices=["python", "cxx", "both"], default="python"
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        choices=["onnx", "torchscript", "nativert", "tensorrt"],
        default=[],
    )
    parser.add_argument("--warp", nargs="*", choices=["onnx", "nativert"], default=[])
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    channel = args.channel.resolve()
    report = json.loads((channel / "packages.json").read_text())
    if report["l4t"] and jetson_release() != report["l4t"]:
        parser.error(
            f"This channel requires Jetson Linux {report['l4t']}; generate the environment on that target"
        )
    if not set(args.warp) <= set(args.backends):
        parser.error("Select each Warp adapter's backend as well")
    if ("tensorrt" in args.backends or args.warp) and not args.cuda:
        parser.error("TensorRT and Warp require --cuda")
    bindings = ["python", "cxx"] if args.binding == "both" else [args.binding]
    selected = [f"leapp-runtime-{binding}" for binding in bindings]
    selected += [f"leapp-backend-{name}" for name in args.backends]
    selected += [f"leapp-warp-{name}" for name in args.warp]
    missing = set(selected) - report["packages"].keys()
    if missing:
        parser.error(f"Packages have not been built: {sorted(missing)}")
    platform = f'{{ platform = "{report["platform"]}", glibc = "{report["glibc"]}"'
    if args.cuda:
        platform += f', cuda = "{report["versions"]["cuda"]}"'
    platform += " }"
    text = (
        f'[workspace]\nname = "leapp-consumer"\nrequires-pixi = ">=0.84.0"\n'
        f'channels = [{json.dumps(channel.as_uri())}, "conda-forge"]\n'
        f"platforms = [{platform}]\n\n[dependencies]\n"
    )
    for name in selected:
        text += f'{name} = {{ version = "=={report["version"]}", build = "{report["build"]}" }}\n'
    # Explicit overrides are preserved. Pixi expands these at activation, not
    # when writing the manifest, so another install prefix works unchanged.
    text += (
        "\n[activation.env]\n"
        'LEAPP_NATIVE_BACKENDS = "${LEAPP_NATIVE_BACKENDS:-$CONDA_PREFIX/lib/leapp/native-backends}"\n'
        'LEAPP_TRITON_BACKENDS = "${LEAPP_TRITON_BACKENDS:-$CONDA_PREFIX/lib/leapp/triton-backends}"\n'
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
