# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit an installed consumer prefix, then execute the selected backends.

Run in a clean container without SDK mounts or compiler toolchains. Reports
package sizes, binary hashes and the libraries actually mapped by each engine.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(prefix, backends, report):
    installed = [
        json.loads(path.read_text()) for path in (prefix / "conda-meta").glob("*.json")
    ]
    names = {record["name"] for record in installed}
    assert {
        name.removeprefix("leapp-backend-")
        for name in names
        if name.startswith("leapp-backend-")
    } == set(backends), names
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
    for name, pattern in {
        "onnx": "libonnxruntime*",
        "torchscript": "libtorch*",
        "nativert": "libleapp_nativert*",
        "tensorrt": "libnvinfer*",
    }.items():
        needed = name in backends or (name == "torchscript" and "nativert" in backends)
        if not needed:
            assert not list(prefix.rglob(pattern)), f"Unselected runtime found: {name}"
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
    if not {"torchscript", "nativert"} & set(backends):
        for pattern in ("libarm_compute*", "libnvpl*"):
            assert not list(prefix.rglob(pattern)), (
                f"Unselected Torch dependency: {pattern}"
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
    # Only libc and the host's NVIDIA driver may resolve outside the environment.
    system = re.compile(
        r"^(ld-linux.*|lib(c|m|dl|pthread|rt|util|resolv|anl)\.so\..*|libcuda\.so.*|libnvidia-.*|"
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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--engine", choices=["native", "triton"], help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    prefix = Path(sys.prefix).resolve()
    report = {"prefix": str(prefix), "backends": args.backends, "cuda": args.cuda}
    if args.engine:
        # Import only after process startup; each engine gets a fresh loader.
        import check_installation

        sys.argv = [
            str(Path(check_installation.__file__)),
            "--backend",
            *(args.backends or ["none"]),
            "--engine",
            args.engine,
        ]
        if args.cuda:
            sys.argv.append("--cuda")
        mapped = set()
        inspect = lambda: mapped.update(loaded_libraries(prefix))
        check_installation.main(inspect)
        for backend in ["onnx", "nativert"]:
            if list((prefix / "conda-meta").glob(f"leapp-warp-{backend}-*.json")):
                check_installation.execute_warp(backend, args.engine, inspect)
        inspect()
        report["libraries"] = sorted(mapped)
    else:
        audit(prefix, args.backends, report)
        report["engines"] = {}
        for engine in ["native", "triton"]:
            child_report = args.report.with_suffix(f".{engine}.json")
            command = [
                sys.executable,
                __file__,
                "--backends",
                *args.backends,
                "--report",
                str(child_report),
                "--engine",
                engine,
            ]
            if args.cuda:
                command.append("--cuda")
            subprocess.run(command, check=True)
            report["engines"][engine] = json.loads(child_report.read_text())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Package audit passed: {args.report}")


if __name__ == "__main__":
    main()
