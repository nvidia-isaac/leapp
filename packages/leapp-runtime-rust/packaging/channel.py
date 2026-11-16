# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble a local channel from staged builds and their inspected SDK libraries.

Run only against trusted SDK binaries: ldd uses the host's ELF loader. The source
libraries, checksums, dependencies and relocated file hashes are kept in the
channel report and in each package. Nothing is uploaded.
"""

import argparse
import json
from pathlib import Path
import tarfile

import yaml

from build import ROOT, digest, run
from payloads import prepare_payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "target/packaging")
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="Inspect and relocate without invoking rattler-build",
    )
    args = parser.parse_args()
    work = args.input.resolve()
    info, stage, dependencies, records = prepare_payloads(work)
    channel = work / "channel"
    channel.mkdir(exist_ok=True)
    components = {p.name for p in stage.iterdir()}
    build = info["build"]
    specs = {name: f"{name} =={info['version']} {build}" for name in components}
    recipes = work / "recipes"
    recipes.mkdir(exist_ok=True)
    template = yaml.safe_load((ROOT / "packaging/recipes/recipe.yaml").read_text())
    report = {**info, "build": build, "packages": {}}
    built = set()
    remaining = set(components)
    while remaining:
        ready = sorted(name for name in remaining if dependencies[name] <= built)
        if not ready:
            raise RuntimeError(f"Cyclic package dependencies: {dict(dependencies)}")
        for name in ready:
            prefix = stage / name
            payload = recipes / f"{name}.tar"
            with tarfile.open(payload, "w") as archive:
                for path in sorted(prefix.iterdir()):
                    archive.add(path, arcname=path.name)
            runtime = sorted(specs[dep] for dep in dependencies[name])
            runtime += [f"__glibc >={info['glibc']}"]
            if name == "leapp-rt-python":
                major, minor = map(int, info["versions"]["python"].split("."))
                runtime += [
                    f"python >={major}.{minor},<{major}.{minor + 1}.0a0",
                    "numpy >=1.26",
                ]
            recipe = {
                **template,
                "context": {
                    "name": name,
                    "version": info["version"],
                    "payload_url": payload.as_uri(),
                    "payload_sha256": digest(payload),
                    "build_string": build,
                    "summary": f"LEAPP {name.removeprefix('leapp-')} ({info['platform']})",
                    "license": "Apache-2.0 AND BSD-3-Clause"
                    if name == "leapp-libs-nativert"
                    else "LicenseRef-Vendor-Notices"
                    if records[name]
                    else "Apache-2.0",
                },
            }
            recipe["requirements"] = {"run": runtime}
            recipe_file = recipes / f"{name}.yaml"
            recipe_file.write_text(yaml.safe_dump(recipe, sort_keys=False))
            if not args.stage_only:
                run(
                    "rattler-build",
                    "build",
                    "--recipe",
                    recipe_file,
                    "--target-platform",
                    info["platform"],
                    "--output-dir",
                    channel,
                    "--channel",
                    channel.as_uri(),
                    "--channel",
                    "conda-forge",
                    "--no-config",
                    "--no-include-recipe",
                    "--skip-existing",
                    "local",
                    "--test",
                    "skip",
                    "--package-format",
                    "conda",
                    "--compression-threads",
                    "2",
                )
            report["packages"][name] = {
                "spec": specs[name],
                "dependencies": runtime,
                "installed_bytes": sum(
                    p.stat().st_size
                    for p in prefix.rglob("*")
                    if p.is_file() and not p.is_symlink()
                ),
            }
            if not args.stage_only:
                payload.unlink()
            built.add(name)
            remaining.remove(name)
    for package in channel.glob(
        f"{info['platform']}/*-{info['version']}-{build}.conda"
    ):
        name = package.name.rsplit("-", 2)[0]
        report["packages"][name].update(
            archive=package.name,
            sha256=digest(package),
            download_bytes=package.stat().st_size,
        )
    (channel / "packages.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Local channel: {channel}")


if __name__ == "__main__":
    main()
