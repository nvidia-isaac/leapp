# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build optional backend wheels from the same inspected payloads as Pixi."""

import argparse
import base64
import csv
from email.message import EmailMessage
import io
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

from build import ROOT, digest
from payloads import elf, prepare_payloads, relocate

NAMESPACE = "_leapp_rt_backends"


def distribution(component, info):
    """Keep CUDA families distinct in pip's dependency graph."""
    cuda = info["versions"].get("cuda", "").split(".")[0]
    if component.startswith("leapp-backend-"):
        backend = component.removeprefix("leapp-backend-")
        return f"leapp-rt-{backend}" + ("" if backend == "onnx" else f"-cuda{cuda}")
    if component == "leapp-onnx-cuda":
        return f"leapp-rt-onnx-cuda{cuda}"
    if component == "leapp-onnx-tensorrt":
        return f"leapp-rt-tensorrt-cuda{cuda}"
    family = component.removeprefix("leapp-libs-")
    suffix = "" if family in {"onnx", "system"} else f"-cuda{cuda}"
    return f"leapp-rt-libs-{family}{suffix}"


def build_wheels(work, output, allow_unreleased=False):
    """Write independently installable SDK components with exact dependencies."""
    info = json.loads((work / "build.json").read_text())
    if (
        not info["backends"]
        or not set(info["backends"]) <= {"onnx", "torchscript", "nativert"}
        or set(info["engines"]) != {"native"}
    ):
        raise RuntimeError(
            "Backend wheels support native ONNX, TorchScript, and NativeRT; use ONNX TensorRT acceleration for TensorRT"
        )
    if not info["runtime_release"] and not allow_unreleased:
        raise RuntimeError(
            "Build backend wheels from the clean matching runtime-vVERSION tag; "
            "--allow-unreleased permits local development wheels only"
        )
    info, stage, dependencies, _ = prepare_payloads(work)
    components = {
        p.name
        for p in stage.iterdir()
        if p.name.startswith(("leapp-libs-", "leapp-backend-"))
        or p.name in {"leapp-backend-onnx", "leapp-onnx-cuda", "leapp-onnx-tensorrt"}
    }
    architecture = {"linux-64": "x86_64", "linux-aarch64": "aarch64"}[info["platform"]]
    platform = (
        f"linux_{architecture}"
        if info["l4t"]
        else f"manylinux_{info['glibc'].replace('.', '_')}_{architecture}"
    )
    tag = f"py3-none-{platform}"
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "backend-wheels.json"
    if (
        report_path.exists()
        and json.loads(report_path.read_text())["build"] != info["build"]
    ):
        raise RuntimeError(
            "Output contains another SDK build; select a clean wheel directory"
        )
    report = {**info, "packages": {}}
    for component in sorted(components):
        name = distribution(component, info)
        normalized = name.replace("-", "_")
        dist = f"{normalized}-{info['version']}.dist-info"
        wheel = output / f"{normalized}-{info['version']}-{tag}.whl"
        required = dependencies[component] - {"leapp-rt-environment"}
        if required - components:
            raise RuntimeError(
                f"Missing wheel dependencies for {component}: {required - components}"
            )
        requirements = [f"leapp-rt=={info['version']}"] + [
            f"{distribution(dep, info)}=={info['version']}" for dep in sorted(required)
        ]
        metadata = EmailMessage()
        for key, value in [
            ("Metadata-Version", "2.4"),
            ("Name", name),
            ("Version", info["version"]),
            ("Summary", f"Optional LEAPP runtime libraries: {component}"),
            ("Description-Content-Type", "text/plain"),
            ("Requires-Python", ">=3.10"),
            ("License-Expression", "LicenseRef-Vendor-Notices"),
        ]:
            metadata[key] = value
        for requirement in requirements:
            metadata["Requires-Dist"] = requirement
        metadata.set_payload(
            f"Native SDK component {component} for LEAPP runtime {info['version']}.\n"
            "Install through the matching leapp-rt extra. Contains the selected "
            "native libraries and their license notices; no SDK is downloaded at runtime.\n"
        )
        rows = []
        installed = {}
        with tempfile.TemporaryDirectory(prefix="leapp-backend-wheel-") as temporary:
            with ZipFile(
                wheel, "w", ZIP_DEFLATED, compresslevel=6, allowZip64=True
            ) as archive:

                def add_bytes(path, data):
                    archive.writestr(path, data)
                    checksum = (
                        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                        .decode()
                        .rstrip("=")
                    )
                    rows.append((path, "sha256=" + checksum, len(data)))

                prefix = stage / component
                for source in sorted(prefix.rglob("*")):
                    if not source.is_file():
                        continue
                    relative = source.relative_to(prefix)
                    parts = relative.parts
                    if parts[:3] == ("lib", "leapp", "vendor"):
                        target = PurePosixPath("lib", *parts[3:])
                    elif parts[:3] == ("lib", "leapp", "native-backends"):
                        # ORT's Conda alias crosses package boundaries. Discovery
                        # selects the real library instead; wheels need no symlinks.
                        if source.is_symlink():
                            continue
                        target = PurePosixPath("backends", *parts[3:])
                    elif parts[:2] == ("share", "licenses"):
                        license_name = "/".join(parts[3:])
                        path = f"{dist}/licenses/{license_name}"
                        metadata["License-File"] = f"licenses/{license_name}"
                        add_bytes(path, source.read_bytes())
                        continue
                    else:
                        continue
                    # Resolve SONAME aliases into regular wheel members. Directory
                    # aliases and activation scripts belong only to Conda packages.
                    original = source.resolve(strict=True)
                    staged = Path(temporary) / str(len(rows))
                    shutil.copy2(original, staged)
                    if elf(staged):
                        relative_lib = os.path.relpath("lib", str(target.parent))
                        relocate(
                            staged,
                            "$ORIGIN"
                            + (
                                f":$ORIGIN/{relative_lib}"
                                if relative_lib != "."
                                else ""
                            ),
                        )
                    path = f"{NAMESPACE}/{target}"
                    archive.write(staged, path)
                    checksum = digest(staged)
                    installed[str(target)] = checksum
                    encoded = (
                        base64.urlsafe_b64encode(bytes.fromhex(checksum))
                        .decode()
                        .rstrip("=")
                    )
                    rows.append((path, "sha256=" + encoded, staged.stat().st_size))
                    staged.unlink()
                add_bytes(
                    f"{NAMESPACE}/metadata/{name}.json",
                    json.dumps(
                        {**info, "package": name, "files": installed}, indent=2
                    ).encode(),
                )
                add_bytes(f"{dist}/METADATA", metadata.as_bytes())
                add_bytes(
                    f"{dist}/WHEEL",
                    f"Wheel-Version: 1.0\nGenerator: leapp-rt\nRoot-Is-Purelib: false\nTag: {tag}\n".encode(),
                )
                record = io.StringIO(newline="")
                csv.writer(record, lineterminator="\n").writerows(
                    [*rows, (f"{dist}/RECORD", "", "")]
                )
                archive.writestr(f"{dist}/RECORD", record.getvalue())
        report["packages"][name] = {
            "archive": wheel.name,
            "sha256": digest(wheel),
            "dependencies": requirements,
            "download_bytes": wheel.stat().st_size,
        }
        print(wheel, flush=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "target/packaging")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-unreleased",
        action="store_true",
        help="Permit local development artifacts",
    )
    args = parser.parse_args()
    build_wheels(
        args.input.resolve(),
        (args.output or args.input / "backend-wheels").resolve(),
        args.allow_unreleased,
    )


if __name__ == "__main__":
    main()
