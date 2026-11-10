# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create a Python sdist with a lockfile matching its reduced Cargo workspace."""

import argparse
from pathlib import Path
import subprocess
import tarfile
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "target/distributions")
    args = parser.parse_args()
    version = tomllib.loads((ROOT / "bindings/python/pyproject.toml").read_text())[
        "project"
    ]["version"]
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="leapp-sdist-") as directory:
        work = Path(directory)
        subprocess.run(
            [
                "maturin",
                "sdist",
                "--manifest-path",
                str(ROOT / "bindings/python/Cargo.toml"),
                "--out",
                str(work),
            ],
            check=True,
        )
        source = work / f"leapp_runtime-{version}.tar.gz"
        with tarfile.open(source) as archive:
            archive.extractall(work / "source", filter="data")
        root = work / "source" / f"leapp_runtime-{version}"
        lock = root / "Cargo.lock"
        before = tomllib.loads(lock.read_text())["package"]
        # Maturin omits the unrelated C++ workspace member. Cargo must prune its
        # unused lock entries before a consumer can build this archive --locked.
        subprocess.run(
            [
                "cargo",
                "metadata",
                "--offline",
                "--format-version",
                "1",
                "--manifest-path",
                str(root / "bindings/python/Cargo.toml"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        after = tomllib.loads(lock.read_text())["package"]
        if any(package not in before for package in after):
            raise RuntimeError("Source packaging changed a locked dependency")
        destination = args.out / source.name
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(root, arcname=root.name)
    print(destination)


if __name__ == "__main__":
    main()
