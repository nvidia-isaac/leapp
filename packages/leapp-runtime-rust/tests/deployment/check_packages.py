# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit an installed consumer prefix, then execute the selected backends.

Run in a clean container without SDK mounts or compiler toolchains. Reports
package sizes, binary hashes and the libraries actually mapped by each engine.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(prefix, backends, report, onnx_tensorrt=False, engines=("native",)):
    installed = [
        json.loads(path.read_text()) for path in (prefix / "conda-meta").glob("*.json")
    ]
    names = {record["name"] for record in installed}
    assert ("leapp-onnx-tensorrt" in names) == onnx_tensorrt
    for engine, kind in (("native", "backend"), ("triton", "triton")):
        marker = f"leapp-{kind}-"
        expected = set(backends) if engine in engines else set()
        if engine == "native":
            expected.discard("tensorrt")
        assert {
            name.removeprefix(marker) for name in names if name.startswith(marker)
        } == expected, names
    forbidden = {
        "rust",
        "rust-std",
        "maturin",
        "cmake",
        "ninja",
        "rattler-build",
        "py-lief",
        "liblief",
        "patchelf",
        "pytorch",
        "torch",
        "onnx",
        "cuda-nvcc",
    }
    assert not names & forbidden, names & forbidden
    for tool in ["cargo", "rustc", "nvcc", "cmake", "c++", "g++"]:
        assert shutil.which(tool) is None, (
            f"Run the deployment audit without build tools: {tool}"
        )
    for name, patterns in {
        "onnx": ("libonnxruntime*",),
        "torchscript": ("libtorch*", "libarm_compute*", "libnvpl*"),
        "nativert": ("libleapp_nativert*",),
        "tensorrt": ("libnvinfer*",),
    }.items():
        needed = (
            name in backends
            or (name == "tensorrt" and onnx_tensorrt)
            or (name == "torchscript" and "nativert" in backends)
        )
        if not needed:
            for pattern in patterns:
                assert not list(prefix.rglob(pattern)), (
                    f"Unselected runtime found: {name}"
                )
    metadata = [
        json.loads(path.read_text())
        for path in (prefix / "share/leapp/packages").glob("*.json")
    ]
    builds = {item["build"] for item in metadata}
    assert len(builds) == 1, f"Mixed incompatible LEAPP builds: {builds}"
    for release in {item["l4t"] for item in metadata} - {None}:
        path = Path("/etc/nv_tegra_release")
        actual = re.search(
            r"R(\d+)\s+\(release\),\s*REVISION:\s*([\d.]+)",
            path.read_text() if path.exists() else "",
        )
        assert actual and f"{actual[1]}.{actual[2]}" == release, (
            f"These packages require Jetson Linux {release}"
        )
    for item in metadata:
        for relative, expected in item["files"].items():
            path = prefix / relative
            # Conda normalizes these wheel bookkeeping files during packaging.
            if ".dist-info/" in relative and path.name in {"INSTALLER", "RECORD"}:
                continue
            assert path.is_file() and sha256(path) == expected, (
                f"Changed or missing package file: {path}"
            )
    core = {}
    for pattern in [
        "lib/python*/site-packages/lrt/_native*.so",
        "lib/libleapp_runtime.so.*",
    ]:
        for path in prefix.glob(pattern):
            if not path.is_symlink():
                path = path.resolve()
                core[str(path.relative_to(prefix))] = sha256(path)
    report.update(
        build=builds.pop(),
        core=core,
        packages=[
            {
                "name": record["name"],
                "archive_bytes": record.get("size"),
                "installed_bytes": sum(
                    (prefix / path).stat().st_size
                    for path in record["files"]
                    if (prefix / path).is_file() and not (prefix / path).is_symlink()
                ),
            }
            for record in installed
        ],
    )


def loaded_libraries(prefix):
    # Standard platform libraries and the NVIDIA driver may come from the host.
    system = re.compile(
        r"^(ld-linux.*|lib(c|m|dl|pthread|rt|util|resolv|anl|gcc_s|stdc\+\+)\.so\..*|libcuda\.so.*|libnvidia-.*|"
        r"libnv(rm_(gpu|sync|mem|host1x|chip)|cu(compat|extend)|os|socsys|tegrahv|sciipc)\.so)$"
    )
    paths = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "/" in line and ".so" in line
        }
    )
    for text in paths:
        path = Path(text)
        assert path.is_relative_to(prefix) or system.match(path.name), (
            f"Library leaked from host: {path}"
        )
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends",
        nargs="*",
        choices=["onnx", "torchscript", "nativert", "tensorrt"],
        default=[],
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--onnx-tensorrt", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--engines", nargs="+", choices=["native", "triton"], default=["native"]
    )
    parser.add_argument(
        "--engine", choices=["native", "triton"], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    report = {"prefix": str(prefix), "backends": args.backends, "cuda": args.cuda}
    if args.engine:
        # Import only after process startup; each engine gets a fresh loader.
        import check_installation

        arguments = [
            "--backend",
            *(args.backends or ["none"]),
            "--engine",
            args.engine,
            "--installed-only",
        ]
        if args.cuda:
            arguments.append("--cuda")
        mapped = set()
        inspect = lambda: mapped.update(loaded_libraries(prefix))
        check_installation.main(inspect, argv=arguments)
        if "onnx" in args.backends and args.cuda and not args.onnx_tensorrt:
            import lrt

            try:
                with lrt.Session.open(
                    check_installation.fixture("onnx"),
                    lrt.SessionOptions(
                        engine=args.engine,
                        device="cuda:0",
                        onnx_tensorrt={"accumulator": {}},
                    ),
                ):
                    raise AssertionError(
                        "TensorRT acceleration succeeded without its provider package"
                    )
            except lrt.RuntimeError as error:
                assert "tensorrt" in str(error).lower(), str(error)
        if args.onnx_tensorrt:
            import check_onnx_tensorrt

            with tempfile.TemporaryDirectory() as cache:
                check_onnx_tensorrt.check(
                    Path(
                        os.environ.get(
                            "LEAPP_TRITON_FIXTURES", "target/triton-fixtures"
                        )
                    ),
                    Path(cache),
                    args.engine,
                )
        for backend in ["onnx", "nativert"]:
            if list(
                (prefix / "conda-meta").glob(
                    f"leapp-warp-{args.engine}-{backend}-*.json"
                )
            ):
                check_installation.execute_warp(backend, args.engine, inspect)
        inspect()
        report["libraries"] = sorted(mapped)
    else:
        audit(prefix, args.backends, report, args.onnx_tensorrt, args.engines)
        report["engines"] = {}
        for engine in sorted(set(args.engines)):
            child_report = args.report.with_suffix(f".{engine}.json")
            command = [
                sys.executable,
                __file__,
                "--backends",
                *(b for b in args.backends if engine != "native" or b != "tensorrt"),
                "--report",
                str(child_report),
                "--engine",
                engine,
            ]
            if args.cuda:
                command.append("--cuda")
            if args.onnx_tensorrt:
                command.append("--onnx-tensorrt")
            subprocess.run(command, check=True)
            report["engines"][engine] = json.loads(child_report.read_text())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Package audit passed: {args.report}")


if __name__ == "__main__":
    main()
