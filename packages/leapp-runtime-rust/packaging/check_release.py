# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check runtime release versions and the core crate's distributable contents."""

import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
import tomllib
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {
    "Cargo.toml": "leapp-rt",
    "bindings/common/Cargo.toml": "leapp-rt-bindings",
    "bindings/python/Cargo.toml": "leapp-rt-python",
    "bindings/cxx/Cargo.toml": "leapp-rt-cxx",
}
MAX_CRATE_BYTES = 10_000_000
FIXTURES = {
    f"tests/fixtures/{name}.yaml" for name in ("feedback", "pipeline", "policy")
}
SOURCE_SUFFIXES = {
    ".rs",
    ".h",
    ".cc",
    ".txt",
    ".md",
    ".py",
    ".cmake",
    ".cfg",
    ".proto",
    ".in",
}
PACKAGE_FILES = {
    "Cargo.toml",
    "Cargo.toml.orig",
    "Cargo.lock",
    ".cargo_vcs_info.json",
    "README.md",
    "LICENSE",
    "build.rs",
}


def check_versions(root=ROOT, tag=None):
    """Require one version across the runtime manifests, lockfile, and release tag."""
    manifests = {path: tomllib.loads((root / path).read_text()) for path in MANIFESTS}
    version = manifests["Cargo.toml"]["package"]["version"]
    for path, manifest in manifests.items():
        package = manifest["package"]
        if (package["name"], package["version"]) != (MANIFESTS[path], version):
            raise RuntimeError(f"{path}: expected {MANIFESTS[path]} {version}")
        for section in ("dependencies", "build-dependencies", "dev-dependencies"):
            for name, dependency in manifest.get(section, {}).items():
                if (
                    name in MANIFESTS.values()
                    and dependency.get("version") != f"={version}"
                ):
                    raise RuntimeError(f"{path}: {name} must pin ={version}")
    project = tomllib.loads((root / "bindings/python/pyproject.toml").read_text())[
        "project"
    ]
    if (project["name"], project["version"]) != ("leapp-rt", version):
        raise RuntimeError(f"Python project must be leapp-rt {version}")
    cmake = (root / "bindings/cxx/CMakeLists.txt").read_text()
    match = re.search(r"project\(leapp-rt\s+VERSION\s+([^\s)]+)", cmake)
    if not match or match[1] != version:
        raise RuntimeError(f"CMake project must be leapp-rt {version}")
    locked = tomllib.loads((root / "Cargo.lock").read_text())["package"]
    for name in MANIFESTS.values():
        versions = [
            p["version"] for p in locked if p["name"] == name and "source" not in p
        ]
        if versions != [version]:
            raise RuntimeError(f"Cargo.lock: {name} must be {version}")
    if tag is not None and tag != f"runtime-v{version}":
        raise RuntimeError(f"Release tag must be runtime-v{version}, found {tag}")
    return version


def check_crate(path, version):
    """Reject oversized archives, exported models, and unexpected fixture payloads."""
    size = path.stat().st_size
    if size > MAX_CRATE_BYTES:
        raise RuntimeError(f"Crate is {size:,} bytes; limit is {MAX_CRATE_BYTES:,}")
    prefix = f"leapp-rt-{version}/"
    count = 0
    with tarfile.open(path) as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or not member.name.startswith(prefix):
                raise RuntimeError(f"Unexpected crate member: {member.name}")
            relative = PurePosixPath(member.name.removeprefix(prefix))
            parts = relative.parts
            if not parts:
                raise RuntimeError(f"Unexpected crate member: {member.name}")
            source = parts[0] == "src" and (
                relative.suffix in SOURCE_SUFFIXES
                or relative.name in {".clang-format", ".clang-tidy"}
            )
            test = (
                parts[:2] in (("tests", "unit_tests"), ("tests", "functional_tests"))
                and relative.suffix == ".rs"
            )
            example = (
                len(parts) == 2 and parts[0] == "examples" and relative.suffix == ".rs"
            )
            fixture = str(relative) in FIXTURES
            if ".." in parts or not (
                source or test or example or fixture or str(relative) in PACKAGE_FILES
            ):
                raise RuntimeError(f"Unexpected crate payload: {relative}")
            if fixture and member.size > 64 * 1024:
                raise RuntimeError(f"Fixture exceeds 64 KiB: {relative}")
            count += 1
    if not count:
        raise RuntimeError("Crate contains no files")
    print(
        f"Core crate: {size:,} compressed bytes, {count} files (limit {MAX_CRATE_BYTES:,} bytes)"
    )


def check_wheel(path, report_dir, root=ROOT):
    """Require dependency notices and one extension with only system ELF imports."""
    with ZipFile(path) as archive:
        libraries = []
        for member in archive.infolist():
            if member.is_dir():
                continue
            with archive.open(member) as source:
                magic = source.read(4)
            if magic == b"\x7fELF" or re.search(
                r"\.(so(?:\..*)?|a|dll|pyd)$", member.filename
            ):
                libraries.append(member.filename)
        if len(libraries) != 1 or not re.fullmatch(
            r"lrt/_native\.[^/]+\.so", libraries[0]
        ):
            raise RuntimeError(f"Unexpected wheel binaries: {libraries}")
        for name in ("LICENSE-LEAPP", "THIRD-PARTY-NOTICES.txt"):
            members = [
                p
                for p in archive.namelist()
                if p.endswith(f".dist-info/licenses/{name}")
            ]
            expected = (root / "bindings/python" / name).read_bytes()
            if (
                len(members) != 1
                or not expected
                or archive.read(members[0]) != expected
            ):
                raise RuntimeError(f"Missing or stale wheel license: {name}")
        with tempfile.TemporaryDirectory(prefix="leapp-wheel-") as directory:
            library = Path(directory) / "native.so"
            library.write_bytes(archive.read(libraries[0]))
            dynamic = subprocess.check_output(
                ["readelf", "-d", str(library)], text=True
            )
    needed = re.findall(r"\(NEEDED\).*?\[([^]]+)\]", dynamic)
    system = re.compile(
        r"(?:lib(?:c|m|dl|pthread|rt|util|gcc_s|stdc\+\+)\.so\.[0-9]+"
        r"|ld-linux-(?:x86-64|aarch64)\.so\.[0-9]+)"
    )
    unexpected = [name for name in needed if not system.fullmatch(name)]
    if unexpected:
        raise RuntimeError(
            f"Unexpected wheel shared-library dependencies: {unexpected}"
        )
    audit = subprocess.run(
        ["auditwheel", "show", str(path)], text=True, capture_output=True
    )
    report = (
        f"Wheel: {path.name}\nBinaries: {libraries}\nELF imports: {needed}\n"
        "LEAPP license and dependency notices match the source files.\n\n"
        + audit.stdout
        + audit.stderr
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{path.stem}.txt").write_text(report)
    print(report)
    audit.check_returncode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Expected runtime-vVERSION release tag")
    parser.add_argument("--crate", type=Path, help="Core .crate archive to inspect")
    parser.add_argument("--wheel", type=Path, nargs="+", help="Core wheels to audit")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "target/wheel-audits")
    args = parser.parse_args()
    version = check_versions(tag=args.tag)
    print(f"Runtime versions aligned: {version}")
    if args.crate:
        check_crate(args.crate, version)
    for wheel in args.wheel or []:
        check_wheel(wheel, args.report_dir)


if __name__ == "__main__":
    main()
