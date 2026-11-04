# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assemble a local channel from staged builds and their inspected SDK libraries.

Run only against trusted SDK binaries: ldd uses the host's ELF loader. The source
libraries, checksums, dependencies and relocated file hashes are kept in the
channel report and in each package. Nothing is uploaded.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

import lief
import yaml

from build import ROOT, digest, run
from sdk import cuda_library, inspect_sdk, onnx_library, verify_binary

SYSTEM = re.compile(r"^(ld-linux.*|lib(c|m|dl|pthread|rt|util|resolv|anl)\.so\..*)$")
DRIVER = re.compile(
    r"^(libcuda\.so.*|libnvidia-.*|"
    r"libnv(rm_(gpu|sync|mem|host1x|chip)|cu(compat|extend)|os|socsys|tegrahv|sciipc)\.so)$"
)
GROUPS = {
    "libtritonserver": "libs-triton",
    "libonnxruntime": "libs-onnx",
    "libopenvino": "libs-onnx",
    "libtbb": "libs-onnx",
    "libtorch": "libs-torch",
    "libc10": "libs-torch",
    "libarm_compute": "libs-torch",
    "libnvpl": "libs-torch",
    "libnvinfer_builder_resource": "libs-tensorrt-builder",
    "libnvonnxparser": "libs-tensorrt-parser",
    "libleapp_nativert.": "libs-nativert",
    "libnvinfer": "libs-tensorrt",
    "libcudart": "libs-cudart",
    "libcublas": "libs-cublas",
    "libcudnn": "libs-cudnn",
    "libcufft": "libs-cufft",
    "libcurand": "libs-curand",
    "libcusolver": "libs-cusolver",
    "libcusparseLt": "libs-cusparselt",
    "libcusparse": "libs-cusparse",
    "libnvJitLink": "libs-nvjitlink",
    "libnvrtc": "libs-nvrtc",
    "libnccl": "libs-nccl",
    "warp.so": "libs-warp",
    "libnvshmem": "libs-nvshmem",
    "libcupti": "libs-cupti",
    "libcufile": "libs-cufile",
}


def elf(path):
    if path.is_symlink() or not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def libraries(path, environment):
    output = subprocess.check_output(["ldd", str(path)], env=environment, text=True)
    if "not found" in output:
        raise RuntimeError(f"Unresolved SDK dependency for {path}:\n{output}")
    return [Path(match) for match in re.findall(r"(?:=>\s+)?(/[^\s]+)\s+\(", output)]


def relocate(path, runpath):
    binary = lief.ELF.parse(path)
    if binary is None:
        raise RuntimeError(f"Cannot read ELF binary: {path}")
    if any(name.startswith("/") for name in binary.libraries):
        raise RuntimeError(f"Absolute SDK library dependency in {path}")
    binary.remove(lief.ELF.DynamicEntry.TAG.RPATH)
    binary.remove(lief.ELF.DynamicEntry.TAG.RUNPATH)
    binary.add(lief.ELF.DynamicEntryRunPath(runpath))
    temporary = path.with_name(path.name + ".relocated")
    try:
        binary.write(temporary)
        if not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError(f"Cannot write relocated ELF binary: {path}")
        shutil.copymode(path, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def owner(path):
    return "leapp-" + next(
        (group for prefix, group in GROUPS.items() if path.name.startswith(prefix)),
        "libs-system",
    )


def licenses(path, sdk):
    candidates = []
    for installation in ("triton", "torch", "tensorrt", "cuda"):
        if sdk.get(installation):
            root = Path(sdk[installation])
            if path.is_relative_to(root):
                candidates += list(root.glob("LICENSE*"))
                candidates += list(path.parent.glob("LICENSE*"))
    if sdk.get("onnx") and path.name.startswith("libonnxruntime"):
        parent = Path(sdk["onnx"]).parent
        candidates += list(parent.glob("LICENSE*"))
        candidates += list(parent.parent.glob("LICENSE*"))
    # Wheel metadata holds licenses for PyTorch, Warp and NVIDIA CUDA libraries.
    for parent in path.parents:
        if parent.name in {"site-packages", "dist-packages"}:
            family = (
                "torch"
                if "torch" in path.parts
                else "warp"
                if "warp" in path.parts
                else path.parent.parent.name.replace("cu13", "cuda")
            )
            distributions = set(parent.glob(f"{family}*.dist-info"))
            if "nvidia" in path.parts:
                distributions.update(parent.glob("nvidia_*.dist-info"))
            for dist in distributions:
                candidates += [
                    p
                    for p in dist.rglob("*")
                    if p.is_file()
                    and ("license" in str(p).lower() or p.name == "NOTICE")
                ]
            break
    if shutil.which("dpkg-query"):
        result = subprocess.run(
            ["dpkg-query", "-S", str(path)], text=True, capture_output=True
        )
        for line in result.stdout.splitlines():
            package = line.split(": ", 1)[0].split(":", 1)[0]
            candidates.append(Path("/usr/share/doc") / package / "copyright")
    # CUDA /usr/local/cuda is commonly a symlink; dpkg records the real path.
    if not candidates and path != path.resolve():
        return licenses(path.resolve(), sdk)
    return sorted(set(p for p in candidates if p.is_file()))


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
    info = json.loads((work / "build.json").read_text())
    sdk = json.loads((work / "sdk.json").read_text())
    versions = {
        key: value for key, value in info["versions"].items() if key != "python"
    }
    if inspect_sdk(sdk, info["backends"]) != versions:
        raise RuntimeError(
            "The SDK versions changed since compilation; rebuild before packaging"
        )
    stage = work / "payloads"
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(work / "stage", stage, symlinks=True)
    channel = work / "channel"
    channel.mkdir(exist_ok=True)
    dependencies = defaultdict(set)
    records = defaultdict(list)
    roots = [
        (component.name, path)
        for component in stage.iterdir()
        for path in component.rglob("*")
        if elf(path)
    ]
    known = {path.name: (package, path) for package, path in roots}
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = (
        ":".join(str(p / "lib") for p in stage.iterdir())
        + ":"
        + environment.get("LD_LIBRARY_PATH", "")
    )
    vendor_roots = []
    if sdk.get("triton"):
        vendor_roots.append(Path(sdk["triton"]) / "lib/libtritonserver.so")
    if sdk.get("cuda"):
        vendor_roots.append(cuda_library(sdk))
    if "onnx" in info["backends"]:
        ort = onnx_library(sdk)
        vendor_roots.append(ort)
        if info["onnx_cuda"] or info.get("onnx_tensorrt"):
            vendor_roots.append(ort.parent / "libonnxruntime_providers_shared.so")
        if info["onnx_cuda"]:
            vendor_roots.append(ort.parent / "libonnxruntime_providers_cuda.so")
    pending = roots + [(owner(path), path) for path in vendor_roots]
    if info.get("onnx_tensorrt"):
        pending.append(
            (
                "leapp-onnx-tensorrt",
                onnx_library(sdk).parent / "libonnxruntime_providers_tensorrt.so",
            )
        )
    visited = set()
    copied = {}
    license_sources = defaultdict(set)
    while pending:
        package, source = pending.pop()
        key = (package, source)
        if key in visited:
            continue
        visited.add(key)
        if not source.is_relative_to(stage):
            dest = stage / package / "lib/leapp/vendor" / source.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and digest(dest) != digest(source):
                raise RuntimeError(f"Conflicting SDK libraries named {source.name}")
            original = (package, source.resolve())
            if not dest.exists():
                if original in copied:
                    dest.symlink_to(copied[original].name)
                else:
                    shutil.copy2(source, dest)
            copied.setdefault(original, dest)
            records[package].append(
                {
                    "source": str(source),
                    "sha256": digest(source),
                    "file": str(dest.relative_to(stage / package)),
                }
            )
            license_sources[package].update(licenses(source, sdk))
        known[source.name] = (package, source)
        for library in libraries(source, environment):
            if SYSTEM.match(library.name) or DRIVER.match(library.name):
                continue
            dependency, location = known.get(library.name, (owner(library), library))
            if dependency != package:
                dependencies[package].add(dependency)
            pending.append((dependency, location))
        # These runtimes load sibling kernels dynamically, outside DT_NEEDED.
        for prefix, pattern in (
            ("libcudnn.so", "libcudnn*.so*"),
            ("libnvrtc.so", "libnvrtc-builtins.so*"),
        ):
            if source.name.startswith(prefix):
                pending += [
                    (package, path)
                    for path in source.parent.glob(pattern)
                    if not path.is_symlink()
                ]
        if info.get("onnx_tensorrt") and source.name.startswith("libnvinfer.so"):
            resources = [
                p
                for p in source.parent.glob("libnvinfer_builder_resource*.so*")
                if not p.is_symlink() and "_win_" not in p.name
            ]
            if not resources:
                raise RuntimeError(
                    f"Missing TensorRT builder resources beside {source}"
                )
            pending += [(owner(path), path) for path in resources]
    # Vendor dependencies live privately so they cannot overwrite Conda's
    # Python/NumPy libraries. NativeRT is built locally but shares that layout.
    nativert = stage / "leapp-libs-nativert/lib/libleapp_nativert.so"
    if nativert.exists():
        destination = nativert.parent / "leapp/vendor" / nativert.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        nativert.rename(destination)
    if (stage / "leapp-onnx-tensorrt").exists():
        dependencies["leapp-onnx-tensorrt"].update(
            {"leapp-libs-onnx", "leapp-libs-tensorrt-builder"}
        )
    # Plugin lookup remains configurable through the existing environment API.
    core = stage / "leapp-runtime-environment"
    activate = core / "etc/conda/activate.d/leapp.sh"
    activate.parent.mkdir(parents=True, exist_ok=True)
    activate.write_text(
        'export LEAPP_NATIVE_BACKENDS="${LEAPP_NATIVE_BACKENDS:-$CONDA_PREFIX/lib/leapp/native-backends}"\n'
        'export LEAPP_TRITON_BACKENDS="${LEAPP_TRITON_BACKENDS:-$CONDA_PREFIX/lib/leapp/triton-backends}"\n'
        'export TRITON_ROOT="${TRITON_ROOT:-$CONDA_PREFIX/lib/leapp/triton}"\n'
        'export CUDA_ROOT="${CUDA_ROOT:-$CONDA_PREFIX/lib/leapp/cuda}"\n'
    )
    for package, directory in (
        ("leapp-libs-triton", "triton"),
        ("leapp-libs-cudart", "cuda"),
    ):
        if (stage / package).exists():
            alias = stage / package / "lib/leapp" / directory / "lib"
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.symlink_to("../vendor", target_is_directory=True)
    if (stage / "leapp-backend-onnx").exists():
        # Native canonicalizes its library path before loading ORT. Triton finds
        # the SONAME through its RPATH directly in vendor/, keeping provider
        # discovery and CUDA dependency lookup in that same directory.
        directory = stage / "leapp-backend-onnx/lib/leapp/native-backends/onnxruntime"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "libonnxruntime.so").symlink_to(
            "../../vendor/" + onnx_library(sdk).name
        )
        dependencies["leapp-backend-onnx"].add("leapp-libs-onnx")
    if (stage / "leapp-triton-onnx").exists():
        dependencies["leapp-triton-onnx"].add("leapp-libs-onnx")
    components = {p.name for p in stage.iterdir()}
    # Use one compatibility build across the binding and plugin packages. A
    # source or SDK change cannot silently mix a new plugin with an old core.
    identity = {
        "target": {key: info[key] for key in ("platform", "glibc", "l4t", "versions")},
        "tools": info["tools"],
        "source": info["source_sha256"],
        "nativert_source": info.get("nativert_source_sha256"),
        "packaging": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (ROOT / "packaging").rglob("*")
            if path.is_file() and path.suffix in {".py", ".yaml", ".toml"}
        },
        "sdk": sorted(
            (Path(r["source"]).name, r["sha256"])
            for rows in records.values()
            for r in rows
        ),
    }
    build = f"h{hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]}_0"
    specs = {name: f"{name} =={info['version']} {build}" for name in components}
    for name in components:
        if (
            name.startswith(
                ("leapp-runtime-", "leapp-backend-", "leapp-triton-", "leapp-warp-")
            )
            and name != "leapp-runtime-environment"
        ):
            dependencies[name].add("leapp-runtime-environment")
        if name.startswith("leapp-triton-"):
            dependencies[name].add("leapp-libs-triton")
        if name.startswith("leapp-warp-"):
            engine, backend = name.removeprefix("leapp-warp-").split("-", 1)
            dependencies[name].add(
                f"leapp-{'backend' if engine == 'native' else 'triton'}-{backend}"
            )
    if "leapp-libs-nativert" in components:
        license_sources["leapp-libs-nativert"].update(
            licenses(Path(sdk["torch"]) / "lib/libtorch.so", sdk)
        )
    for name in components:
        prefix = stage / name
        license_dir = prefix / "share/licenses" / name
        license_dir.mkdir(parents=True, exist_ok=True)
        if (
            name.startswith(
                ("leapp-runtime-", "leapp-backend-", "leapp-triton-", "leapp-warp-")
            )
            or name == "leapp-libs-nativert"
        ):
            shutil.copy2(ROOT.parents[1] / "LICENSE", license_dir / "LEAPP-LICENSE")
        if not license_sources[name] and records[name]:
            raise RuntimeError(
                f"Missing license files for {name}: {[r['source'] for r in records[name]]}"
            )
        for source in license_sources[name]:
            shutil.copy2(
                source, license_dir / (digest(source)[:12] + "-" + source.name)
            )
        for path in prefix.rglob("*"):
            if elf(path):
                verify_binary(path, info["platform"], info["glibc"])
                relative = os.path.relpath(prefix / "lib/leapp/vendor", path.parent)
                relocate(
                    path,
                    "$ORIGIN"
                    + (":" + "$ORIGIN/" + relative if relative != "." else ""),
                )
        metadata = prefix / "share/leapp/packages" / f"{name}.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(
            json.dumps(
                {
                    **info,
                    "package": name,
                    "build": build,
                    "libraries": records[name],
                    "files": {
                        str(p.relative_to(prefix)): digest(p)
                        for p in prefix.rglob("*")
                        if p.is_file() and not p.is_symlink()
                    },
                },
                indent=2,
            )
            + "\n"
        )
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
            if name == "leapp-runtime-python":
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
