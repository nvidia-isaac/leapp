# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and locate native ONNX and Torch tokenizer adapters sharing one engine."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


RUNTIME_ENVIRONMENTS = {
    "onnx": "LEAPP_TEXT_RUNTIME_LIBRARY",
    "pt2": "LEAPP_TEXT_PT2_RUNTIME_LIBRARY",
}


def library_name(backend="onnx"):
    if backend not in RUNTIME_ENVIRONMENTS:
        raise ValueError(f"Unknown text runtime backend: {backend}")
    stem = "leapp_text_runtime" if backend == "onnx" else "leapp_text_torch"
    if sys.platform == "win32":
        return stem + ".dll"
    return "lib" + stem + (".dylib" if sys.platform == "darwin" else ".so")


def target_directory():
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "leapp/text-runtime"
    )


def resolve_library(backend="onnx"):
    filename = library_name(backend)
    environment = RUNTIME_ENVIRONMENTS[backend]
    override = os.environ.get(environment)
    path = (
        Path(override).expanduser()
        if override
        else target_directory() / "release" / filename
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Native tokenizer {backend} runtime was not found at {path}. Build it "
            f"with 'python -m leapp.text_runtime --build', or set {environment}."
        )
    return path.resolve()


def build_torch_library(directory):
    """Link the CPU Torch dispatcher to the same engine used by ONNX Runtime."""
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths

    if sys.platform == "win32":
        raise RuntimeError("The native Torch tokenizer build supports Linux/macOS")
    directory = Path(directory).resolve()
    source = (
        Path(__file__).parent
        / "leapp_graph/custom_operator_registry/tokenizer_operator/torch_tokenizer.cc"
    )
    subprocess.run(
        [
            *shlex.split(os.environ.get("CXX", "c++")),
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *[f"-I{path}" for path in include_paths()],
            *[f"-L{path}" for path in library_paths()],
            f"-L{directory}",
            str(source),
            "-lleapp_text_runtime",
            "-ltorch_cpu",
            "-lc10",
            "-Wl,-rpath,$ORIGIN"
            if sys.platform != "darwin"
            else "-Wl,-rpath,@loader_path",
            *[f"-Wl,-rpath,{path}" for path in library_paths()],
            "-o",
            str(directory / library_name("pt2")),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--target-dir", type=Path, default=target_directory())
    arguments = parser.parse_args()
    if arguments.build:
        manifest = (
            Path(__file__).parent
            / "leapp_graph/custom_operator_registry/tokenizer_operator/Cargo.toml"
        )
        subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                str(manifest),
                "--target-dir",
                str(arguments.target_dir),
            ],
            check=True,
        )
        directory = arguments.target_dir / "release"
        build_torch_library(directory)
        for backend in RUNTIME_ENVIRONMENTS:
            print((directory / library_name(backend)).resolve())
    else:
        for backend in RUNTIME_ENVIRONMENTS:
            print(resolve_library(backend))


if __name__ == "__main__":
    main()
