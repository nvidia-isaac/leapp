# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refresh locked Rust dependency inventory and distributable license notices."""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tomllib
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
NOTICE_NAMES = ("license", "licence", "copying", "notice", "copyright", "authors")


def crate_notices(package):
    """Read notices from the exact published archive verified against Cargo.lock."""
    name, version = package["name"], package["version"]
    if package["source"] != "registry+https://github.com/rust-lang/crates.io-index":
        raise RuntimeError(f"Add license collection for {package['source']}")
    filename = f"{name}-{version}.crate"
    cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
    cached = sorted((cargo_home / "registry/cache").glob(f"*/{filename}"))
    if cached:
        data = cached[0].read_bytes()
    else:
        with urlopen(
            f"https://static.crates.io/crates/{name}/{filename}", timeout=60
        ) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != package["checksum"]:
        raise RuntimeError(f"Cargo.lock checksum mismatch: {filename}")
    prefix = f"{name}-{version}/"
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        manifest = archive.extractfile(prefix + "Cargo.toml")
        metadata = tomllib.loads(manifest.read().decode())["package"]
        notices = {}
        for member in archive:
            path = PurePosixPath(member.name.removeprefix(prefix))
            if member.isfile() and (
                path.name.lower().startswith(NOTICE_NAMES)
                or str(path) == metadata.get("license-file")
            ):
                notices[str(path)] = (
                    archive.extractfile(member)
                    .read()
                    .decode("utf-8")
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                )
    if not metadata.get("license") or not notices:
        raise RuntimeError(f"Missing license expression or notices: {filename}")
    return metadata["license"], dict(sorted(notices.items()))


def generate(root=ROOT):
    """Preserve manually inventoried SDKs; regenerate Cargo rows and notice text."""
    inventory = root / "dependency-licenses.csv"
    with inventory.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        manual = [row for row in reader if row["ecosystem"] != "Rust/Cargo"]
    locked = tomllib.loads((root / "Cargo.lock").read_text())["package"]
    by_name = defaultdict(list)
    for package in locked:
        by_name[package["name"]].append(package["version"])
    parents = defaultdict(list)
    for package in locked:
        for dependency in package.get("dependencies", []):
            name, *rest = dependency.split()
            version = rest[0] if rest else by_name[name][0]
            parents[name, version].append(f"{package['name']}@{package['version']}")
    metadata = json.loads(
        subprocess.check_output(
            [
                "cargo",
                "metadata",
                "--locked",
                "--format-version",
                "1",
                "--filter-platform",
                "x86_64-unknown-linux-gnu",
            ],
            cwd=root,
            text=True,
        )
    )
    active = {(p["name"], p["version"]) for p in metadata["packages"]}
    direct = defaultdict(list)
    for member in (".", "bindings/common", "bindings/python", "bindings/cxx"):
        manifest = tomllib.loads((root / member / "Cargo.toml").read_text())
        for section in ("dependencies", "build-dependencies", "dev-dependencies"):
            for name, dependency in manifest.get(section, {}).items():
                if isinstance(dependency, dict):
                    name = dependency.get("package", name)
                direct[name].append(f"{manifest['package']['name']} ({section})")
    rows, texts, index = [], defaultdict(list), []
    for package in sorted(locked, key=lambda p: (p["name"], p["version"])):
        if "source" not in package:
            continue
        name, version = package["name"], package["version"]
        label = f"{name} {version}"
        license_expression, notices = crate_notices(package)
        evidence = f"https://docs.rs/crate/{name}/{version}/source/Cargo.toml"
        rows.append(
            {
                "ecosystem": "Rust/Cargo",
                "dependency": name,
                "version_or_requirement": version,
                "license": license_expression,
                "relationship": "direct" if name in direct else "transitive",
                "used_for": "; ".join(direct[name])
                if name in direct
                else "Cargo workspace dependency",
                "linux_x86_64_cargo_resolution": "yes"
                if (name, version) in active
                else "no",
                "new_vs_origin_main": "not evaluated",
                "required_by": "; ".join(sorted(parents[name, version])),
                "license_evidence": evidence,
                "license_files": "; ".join(notices),
                "notes": "Generated from Cargo.lock and checksum-verified published crate; "
                "includes build-only, inactive-feature, and other-target dependencies.",
            }
        )
        index.append(f"{label}: {license_expression}\n  {evidence}")
        for path, text in notices.items():
            texts[text].append(f"{label}: {path}")
    # Preserve the BSD notice in the vendored schema used by generated Rust code.
    schema = root / "src/engines/triton/convert/model_config.proto"
    header = schema.read_text().split('syntax = "proto3";', 1)[0]
    texts[header].append("Vendored Triton model configuration schema")
    dlpack = root / "src/dlpack/include/dlpack/dlpack.h"
    header = dlpack.read_text().split("*/", 1)[0] + "*/\n"
    texts[header + (root / "LICENSE").read_text()].append(
        "Vendored DLPack header (Apache-2.0)"
    )
    notice = (
        "THIRD-PARTY LICENSES AND NOTICES\n\n"
        "Generated by packaging/licenses.py from Cargo.lock and published crates.\n"
        "Scope: all locked workspace dependencies, including build-only, inactive-\n"
        "feature and other-target crates, plus vendored schema/header notices.\n"
        "Listing a dependency does not imply it is linked into every wheel.\n"
        "Model runtime SDKs are installed separately and are not bundled here.\n"
        "Identical notice texts are reproduced once with all source files listed.\n\n"
        + "\n".join(index)
        + "\n\n"
    )
    for text, sources in texts.items():
        notice += "=" * 78 + "\n" + "\n".join(sources) + "\n\n" + text + "\n\n"
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows + manual)
    return {
        inventory: output.getvalue(),
        root / "bindings/python/THIRD-PARTY-NOTICES.txt": notice.rstrip() + "\n",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Fail on stale generated files"
    )
    args = parser.parse_args()
    for path, text in generate().items():
        if args.check:
            if not path.exists() or path.read_text() != text:
                raise RuntimeError(
                    f"Stale {path.name}; run python packaging/licenses.py"
                )
        else:
            path.write_text(text)
        print(path)


if __name__ == "__main__":
    main()
