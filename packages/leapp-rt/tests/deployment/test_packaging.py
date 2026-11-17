# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging checks without model SDKs or a GPU; ELF tests need Linux and cc."""

import contextlib
import io
import importlib.util
import os
from types import SimpleNamespace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tarfile
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import check_release
import consumer
import sdk


class PackagingTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cc"), "Linux cc")
    def test_backend_wheels_keep_cuda_optional_and_resolve_sonames(self):
        from wheels import build_wheels

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            sdk_root = work / "sdk"
            (sdk_root / "lib").mkdir(parents=True)
            (sdk_root / "LICENSE").write_text("SDK test license")
            (work / "stage/leapp-backend-onnx").mkdir(parents=True)

            def library(name, code, *link):
                subprocess.run(
                    [
                        "cc",
                        "-shared",
                        "-fPIC",
                        "-x",
                        "c",
                        "-",
                        "-o",
                        str(sdk_root / "lib" / name),
                        f"-Wl,-soname,{name}",
                        *link,
                    ],
                    input=code,
                    text=True,
                    check=True,
                )

            library("libcudart.so.13", "int cuda_value(void) { return 42; }")
            (sdk_root / "lib/libcudart.so").symlink_to("libcudart.so.13")
            library("libonnxruntime.so", "int ort_value(void) { return 1; }")
            library(
                "libonnxruntime_providers_shared.so",
                "int shared_value(void) { return 2; }",
            )
            library(
                "libonnxruntime_providers_cuda.so",
                "int cuda_value(void); int provider(void) { return cuda_value(); }",
                f"-L{sdk_root / 'lib'}",
                "-l:libcudart.so.13",
                f"-Wl,-rpath,{sdk_root / 'lib'}",
            )
            target = sdk.native_platform()
            info = {
                "version": "0.1.0",
                "runtime_release": True,
                "platform": target,
                "glibc": "2.28",
                "l4t": None,
                "engines": ["native"],
                "backends": ["onnx"],
                "onnx_cuda": True,
                "onnx_tensorrt": False,
                "versions": {"cuda": "13.0", "onnxruntime": "1.24.0"},
                "tools": {},
                "source_sha256": "test",
            }
            (work / "build.json").write_text(json.dumps(info))
            (work / "sdk.json").write_text(
                json.dumps(
                    {
                        "onnx": str(sdk_root / "lib/libonnxruntime.so"),
                        "cuda": str(sdk_root),
                        "torch": None,
                        "triton": None,
                        "tensorrt": None,
                    }
                )
            )
            with patch("payloads.inspect_sdk", return_value=info["versions"]):
                report = build_wheels(work, work / "wheels")
            packages = report["packages"]
            self.assertEqual(
                packages["leapp-rt-onnx"]["dependencies"],
                ["leapp-rt==0.1.0", "leapp-rt-libs-onnx==0.1.0"],
            )
            self.assertIn(
                "leapp-rt-libs-cudart-cuda13==0.1.0",
                packages["leapp-rt-onnx-cuda13"]["dependencies"],
            )
            for package in packages.values():
                with ZipFile(work / "wheels" / package["archive"]) as archive:
                    names = archive.namelist()
                    self.assertFalse(any(name.endswith(".sh") for name in names))
                    self.assertTrue(
                        any(".dist-info/licenses/" in name for name in names)
                    )
                    if "libs_cudart" in package["archive"]:
                        self.assertIn("_leapp_rt_backends/lib/libcudart.so.13", names)
                        self.assertEqual(
                            archive.read("_leapp_rt_backends/lib/libcudart.so"),
                            archive.read("_leapp_rt_backends/lib/libcudart.so.13"),
                        )
                    for name in names:
                        if name.endswith(".so.13") or name.endswith(".so"):
                            elf_path = work / "check.so"
                            elf_path.write_bytes(archive.read(name))
                            dynamic = subprocess.check_output(
                                ["readelf", "-d", elf_path], text=True
                            )
                            self.assertNotIn(str(sdk_root), dynamic)

    def test_pip_discovery_preserves_overrides_and_rejects_mixed_builds(self):
        path = check_release.ROOT / "bindings/python/python/lrt/_backends.py"
        spec = importlib.util.spec_from_file_location("pip_backends", path)
        backends = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backends)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lib").mkdir()
            (root / "metadata").mkdir()
            (root / "lib/libonnxruntime.so.1").touch()
            item = {
                "package": "leapp-rt-libs-onnx",
                "version": "0.1.0",
                "build": "one",
                "glibc": "2.28",
                "l4t": None,
                "platform": sdk.native_platform(),
            }
            (root / "metadata/onnx.json").write_text(json.dumps(item))
            with (
                patch.object(
                    backends,
                    "find_spec",
                    return_value=SimpleNamespace(
                        submodule_search_locations=[str(root)]
                    ),
                ),
                patch.object(backends, "version", return_value="0.1.0"),
                patch.object(
                    backends.platform, "libc_ver", return_value=("glibc", "2.39")
                ),
                patch.dict(os.environ, {}, clear=True),
            ):
                backends.configure()
                self.assertEqual(
                    os.environ["ORT_DYLIB_PATH"], str(root / "lib/libonnxruntime.so.1")
                )
                backends.configure.cache_clear()
                os.environ["LEAPP_NATIVE_BACKENDS"] = "/explicit"
                os.environ.pop("ORT_DYLIB_PATH")
                backends.configure()
                self.assertNotIn("ORT_DYLIB_PATH", os.environ)
                os.environ.pop("LEAPP_NATIVE_BACKENDS")
                backends.configure.cache_clear()
                (root / "metadata/cuda.json").write_text(
                    json.dumps({**item, "build": "two"})
                )
                with self.assertRaisesRegex(RuntimeError, "incompatible SDK builds"):
                    backends.configure()
                self.assertNotIn("ORT_DYLIB_PATH", os.environ)

    def test_release_version_drift(self):
        version = check_release.check_versions()
        with self.assertRaisesRegex(RuntimeError, "Release tag"):
            check_release.check_versions(tag="runtime-v9.9.9")
        files = [
            *check_release.MANIFESTS,
            "bindings/python/pyproject.toml",
            "bindings/cxx/CMakeLists.txt",
            "Cargo.lock",
        ]
        for changed in files:
            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                for name in files:
                    destination = root / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(check_release.ROOT / name, destination)
                manifest = root / changed
                manifest.write_text(manifest.read_text().replace(version, "9.9.9"))
                with self.assertRaises(RuntimeError):
                    check_release.check_versions(root)

    def test_crate_payload_limits(self):
        for name, size, valid in (
            ("src/lib.rs", 1, True),
            ("tests/fixtures/pipeline.yaml", 1, True),
            ("src/model.onnx", 1, False),
            ("tests/fixtures/export.yaml", 1, False),
            ("tests/fixtures/pipeline.yaml", 65537, False),
        ):
            with (
                self.subTest(name=name, size=size),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "core.crate"
                with tarfile.open(path, "w:gz") as archive:
                    member = tarfile.TarInfo(f"leapp-rt-0.1.0/{name}")
                    member.size = size
                    archive.addfile(member, io.BytesIO(b"x" * size))
                if valid:
                    with contextlib.redirect_stdout(io.StringIO()):
                        check_release.check_crate(path, "0.1.0")
                    with patch.object(check_release, "MAX_CRATE_BYTES", 1):
                        with self.assertRaisesRegex(RuntimeError, "limit"):
                            check_release.check_crate(path, "0.1.0")
                else:
                    with self.assertRaises(RuntimeError):
                        check_release.check_crate(path, "0.1.0")

    def test_wheel_rejects_missing_notices_and_sdk_binaries(self):
        for problem, message in (
            ("notices", "Missing or stale wheel license"),
            ("bundled", "Unexpected wheel binaries"),
            ("linked", "Unexpected wheel shared-library dependencies"),
        ):
            with (
                self.subTest(problem=problem),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                wheel = root / "core.whl"
                with ZipFile(wheel, "w") as archive:
                    archive.writestr(
                        "lrt/_native.cpython-312-x86_64-linux-gnu.so", b"\x7fELF"
                    )
                    for name in ("LICENSE-LEAPP", "THIRD-PARTY-NOTICES.txt"):
                        if problem == "notices" and name == "THIRD-PARTY-NOTICES.txt":
                            continue
                        archive.writestr(
                            f"leapp_rt-0.1.0.dist-info/licenses/{name}",
                            (
                                check_release.ROOT / "bindings/python" / name
                            ).read_bytes(),
                        )
                    if problem == "bundled":
                        archive.writestr("leapp_rt.libs/backend.bin", b"\x7fELF")
                with (
                    patch.object(
                        check_release.subprocess,
                        "check_output",
                        return_value="(NEEDED) Shared library: [libonnxruntime.so.1]",
                    ),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    check_release.check_wheel(wheel, root / "reports")

    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cc"), "Linux cc")
    def test_relocation(self):
        from payloads import relocate

        for tag in ("rpath", "runpath"):
            with self.subTest(tag=tag), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                library = root / "libanswer.so"
                subprocess.run(
                    [
                        "cc",
                        "-x",
                        "c",
                        "-shared",
                        "-fPIC",
                        "-Wl,-soname,libanswer.so",
                        "-o",
                        str(library),
                        "-",
                    ],
                    input="int answer(void) { return 42; }",
                    text=True,
                    check=True,
                )
                program = root / "program"
                subprocess.run(
                    [
                        "cc",
                        "-x",
                        "c",
                        "-",
                        "-L",
                        str(root),
                        "-lanswer",
                        f"-Wl,-rpath,{root}",
                        "-Wl,--disable-new-dtags"
                        if tag == "rpath"
                        else "-Wl,--enable-new-dtags",
                        "-o",
                        str(program),
                    ],
                    input="int answer(void); int main(void) { return answer() != 42; }",
                    text=True,
                    check=True,
                )
                mode = program.stat().st_mode
                relocate(program, "$ORIGIN/vendor")
                dynamic = subprocess.check_output(["readelf", "-d", program], text=True)
                self.assertIn("(RUNPATH)", dynamic)
                self.assertNotIn("(RPATH)", dynamic)
                self.assertNotIn(str(root), dynamic)
                moved = root / "moved"
                (moved / "vendor").mkdir(parents=True)
                library.rename(moved / "vendor" / library.name)
                program = program.rename(moved / program.name)
                self.assertEqual(program.stat().st_mode, mode)
                subprocess.run([program], check=True)

    def test_native_target(self):
        with patch.object(sys, "platform", "linux"):
            for machine, target in (
                ("x86_64", "linux-64"),
                ("aarch64", "linux-aarch64"),
            ):
                with patch.object(sdk.platform, "machine", return_value=machine):
                    self.assertEqual(sdk.native_platform(), target)
        with patch.object(sys, "platform", "darwin"):
            with self.assertRaisesRegex(RuntimeError, "Build natively"):
                sdk.native_platform()

    def test_binary_architecture_and_glibc_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "test.so"
            header = bytearray(20)
            header[:6] = b"\x7fELF\x02\x01"
            header[18:20] = (183).to_bytes(2, "little")
            library.write_bytes(header)
            with patch.object(
                sdk.subprocess, "check_output", return_value="Name: GLIBC_2.38"
            ):
                sdk.verify_binary(library, "linux-aarch64", "2.39")
                with self.assertRaisesRegex(RuntimeError, "above declared baseline"):
                    sdk.verify_binary(library, "linux-aarch64", "2.37")
                with self.assertRaisesRegex(RuntimeError, "not a 64-bit linux-64"):
                    sdk.verify_binary(library, "linux-64", "2.39")

    def test_sdk_reads_versions_without_unselected_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lib").mkdir()
            (root / "lib/libcudart.so").touch()
            (root / "TRITON_VERSION").write_text("2.69.0\n")
            paths = {
                "cuda": str(root),
                "triton": str(root),
                "torch": None,
                "tensorrt": None,
            }
            cuda = Mock()

            def version(result):
                result._obj.value = 13020
                return 0

            cuda.cudaRuntimeGetVersion.side_effect = version
            with patch.object(sdk.ctypes, "CDLL", return_value=cuda) as load:
                versions = sdk.inspect_sdk(paths, [])
                self.assertEqual(versions, {"triton": "2.69.0", "cuda": "13.2"})
                load.assert_called_once_with(str(root / "lib/libcudart.so"))
            (root / "TRITON_VERSION").write_text("0.0.0\n")
            with self.assertRaisesRegex(RuntimeError, "requires Triton"):
                sdk.inspect_sdk(paths, [])

    def test_jetson_release_and_consumer_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "nv_tegra_release"
            self.assertIsNone(sdk.jetson_release(release))
            release.write_text("# R39 (release), REVISION: 2.1, GCID: 123\n")
            self.assertEqual(sdk.jetson_release(release), "39.2.1")
            report = {
                "l4t": "39.2.1",
                "platform": "linux-aarch64",
                "glibc": "2.39",
                "version": "0.1.0",
                "build": "habc_0",
                "versions": {"cuda": "13.2"},
                "engines": ["native"],
                "packages": {"leapp-rt-cxx": {}, "leapp-rt-environment": {}},
            }
            (root / "packages.json").write_text(json.dumps(report))
            output = root / "consumer"
            args = [
                "consumer.py",
                "--channel",
                str(root),
                "--output",
                str(output),
                "--binding",
                "cxx",
                "--core-source",
                "channel",
            ]
            with patch.object(sys, "argv", args):
                for actual in (None, "39.1.0"):
                    with patch.object(consumer, "jetson_release", return_value=actual):
                        with (
                            contextlib.redirect_stderr(io.StringIO()),
                            self.assertRaises(SystemExit),
                        ):
                            consumer.main()
                        self.assertFalse(output.exists())
                with patch.object(consumer, "jetson_release", return_value="39.2.1"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        consumer.main()
                manifest = (output / "pixi.toml").read_text()
                self.assertIn('platform = "linux-aarch64", glibc = "2.39"', manifest)
                self.assertIn('build = "habc_0"', manifest)


if __name__ == "__main__":
    unittest.main()
