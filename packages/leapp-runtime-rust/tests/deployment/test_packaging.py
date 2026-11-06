# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging checks without model SDKs or a GPU; ELF tests need Linux and cc."""

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging"))
import consumer
import sdk


class PackagingTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux" and shutil.which("cc"), "Linux cc")
    def test_relocation(self):
        from channel import relocate

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
                "packages": {"leapp-runtime-cxx": {}},
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
