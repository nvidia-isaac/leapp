# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone build CLI checks; fixtures compile without model SDKs or a GPU."""

import contextlib
import ctypes
import importlib.machinery
import importlib.util
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[4] / "scripts/leapp-build-runtime-backends"
loader = importlib.machinery.SourceFileLoader("build_runtime_backends", str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
builder = importlib.util.module_from_spec(spec)
loader.exec_module(builder)


class BackendBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="leapp-backend-build-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "runtime source"
        self.torch = self.root / "torch sdk"
        self.pytorch = self.root / "pytorch source"
        self.torch.mkdir()
        self.pytorch.mkdir()
        for backend, library in builder.LIBRARIES.items():
            directory = self.source / "src/engines/native/backends" / backend
            directory.mkdir(parents=True)
            name = Path(library).stem.removeprefix("lib")
            (directory / "plugin.cc").write_text(
                'extern "C" int leapp_build_test() { return 42; }\n'
            )
            (directory / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.24)\n"
                f"project({name} LANGUAGES CXX)\n"
                f"add_library({name} SHARED plugin.cc)\n"
                f"install(TARGETS {name} LIBRARY DESTINATION {backend})\n"
            )
        dependency = self.source / "src/engines/triton/backends/torch/nativert"
        dependency.mkdir(parents=True)
        (dependency / "library.cc").write_text(
            "int nativert_build_test() { return 1; }\n"
        )
        (dependency / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.24)\n"
            "project(leapp_nativert LANGUAGES CXX)\n"
            "add_library(leapp_nativert SHARED library.cc)\n"
            "install(TARGETS leapp_nativert LIBRARY DESTINATION lib)\n"
        )
        environment = patch.dict(os.environ, {"TORCH_ROOT": str(self.torch)})
        environment.start()
        self.addCleanup(environment.stop)
        for variable in (
            "PYTORCH_SOURCE",
            "NATIVERT_LIBRARY",
            "CUDAToolkit_ROOT",
            "CUDA_ROOT",
        ):
            os.environ.pop(variable, None)
        self.prefix = self.root / "installed backends"

    @unittest.skipUnless(
        sys.platform == "linux" and shutil.which("cmake") and shutil.which("c++"),
        "requires Linux, CMake 3.24+, and a C++ compiler",
    )
    def test_cli_builds_and_installs_both_backends_with_spaces_in_paths(self):
        environment = {**os.environ, "PYTORCH_SOURCE": str(self.pytorch)}
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--source-dir",
                str(self.source),
                "--build-dir",
                str(self.root / "build files"),
                "--install-prefix",
                str(self.prefix),
                "--backends",
                "torchscript",
                "nativert",
                "--jobs",
                "1",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for library in builder.LIBRARIES.values():
            loaded = ctypes.CDLL(str(self.prefix / library))
            self.assertEqual(loaded.leapp_build_test(), 42)
        self.assertTrue((self.prefix / "nativert/lib/libleapp_nativert.so").is_file())
        self.assertIn(f"Set LEAPP_NATIVE_BACKENDS to {self.prefix}", result.stdout)

    @unittest.skipUnless(
        sys.platform == "linux" and shutil.which("cmake") and shutil.which("c++"),
        "requires Linux, CMake 3.24+, and a C++ compiler",
    )
    def test_reused_build_selects_libraries_from_changed_torch_and_cuda_sdks(self):
        source = (
            SCRIPT.parent.parent
            / "packages/leapp-rt/src/engines/native/backends/torchscript"
        )
        libraries = ("torch", "torch_cpu", "torch_cuda", "c10", "c10_cuda")
        torch_roots = [self.root / f"torch {name}" for name in ("A", "B")]
        cuda_roots = [self.root / f"cuda {name}" for name in ("A", "B")]
        for torch in torch_roots:
            (torch / "include/torch/csrc/api/include").mkdir(parents=True)
            (torch / "include/torch/script.h").touch()
            (torch / "lib").mkdir()
            for name in libraries:
                (torch / f"lib/lib{name}.so").touch()
        for cuda in cuda_roots:
            (cuda / "include").mkdir(parents=True)
            (cuda / "include/cuda_runtime.h").touch()
            (cuda / "version.txt").write_text("CUDA Version 12.0.0\n")
            (cuda / "bin").mkdir()
            (cuda / "lib64").mkdir()
            (cuda / "lib64/libcudart.so").touch()

        run = subprocess.run

        def configure_only(command, **kwargs):
            # Configure the real TorchScript project and FindCUDAToolkit module.
            # The stub SDKs support discovery, not compilation or installation.
            if "--build" in command or "--install" in command:
                return subprocess.CompletedProcess(command, 0)
            return run(command, capture_output=True, text=True, **kwargs)

        build = self.root / "reused build"
        for torch, cuda in (
            (torch_roots[0], cuda_roots[0]),
            (torch_roots[1], cuda_roots[0]),
            (torch_roots[1], cuda_roots[1]),
            (torch_roots[0], cuda_roots[1]),
        ):
            with self.subTest(torch=torch.name, cuda=cuda.name):
                with patch.object(
                    builder.subprocess, "run", side_effect=configure_only
                ):
                    try:
                        builder.cmake_build(
                            source,
                            build,
                            self.prefix,
                            [f"-DTORCH_ROOT={torch}", f"-DCUDAToolkit_ROOT={cuda}"],
                            1,
                        )
                    except subprocess.CalledProcessError as error:
                        self.fail(error.stdout + error.stderr)
                cache = {}
                for line in (build / "CMakeCache.txt").read_text().splitlines():
                    if line and not line.startswith(("#", "//")):
                        key, value = line.split("=", 1)
                        cache[key.split(":", 1)[0]] = value
                self.assertEqual(cache["TORCH_ROOT"], str(torch))
                self.assertEqual(cache["CUDAToolkit_ROOT"], str(cuda))
                for name in libraries:
                    self.assertEqual(
                        cache[f"LEAPP_{name}_LIBRARY"], str(torch / f"lib/lib{name}.so")
                    )
                for name in ("CUDA_CUDART", "CUDA_cudart_LIBRARY"):
                    self.assertEqual(cache[name], str(cuda / "lib64/libcudart.so"))

    def test_missing_sdk_fails_before_invoking_cmake(self):
        os.environ.pop("TORCH_ROOT")
        with patch.object(builder.shutil, "which", return_value="cmake"):
            with patch.object(builder.subprocess, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "TORCH_ROOT"):
                    builder.build_backends(["torchscript"], self.source)
                run.assert_not_called()

    def test_missing_source_explains_the_matching_checkout_requirement(self):
        with self.assertRaisesRegex(RuntimeError, "--source-dir.*matching"):
            builder.runtime_source(self.root / "missing")

    def test_missing_cmake_reports_the_required_tool(self):
        with patch.object(builder.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "CMake 3.24"):
                builder.build_backends(["torchscript"], self.source)

    def test_torchscript_does_not_require_pytorch_sources(self):
        def install(source, build, prefix, options, jobs):
            library = prefix / builder.LIBRARIES["torchscript"]
            library.parent.mkdir(parents=True)
            library.touch()

        with patch.object(builder.shutil, "which", return_value="cmake"):
            with patch.object(builder, "cmake_build", side_effect=install) as build:
                builder.build_backends(
                    ["torchscript"], self.source, install_prefix=self.prefix
                )
        self.assertEqual(build.call_count, 1)

    def test_explicit_invalid_nativert_library_is_not_silently_rebuilt(self):
        os.environ["PYTORCH_SOURCE"] = str(self.pytorch)
        os.environ["NATIVERT_LIBRARY"] = str(self.root / "missing.so")
        with patch.object(builder.shutil, "which", return_value="cmake"):
            with patch.object(builder, "cmake_build") as build:
                with self.assertRaisesRegex(RuntimeError, "NATIVERT_LIBRARY"):
                    builder.build_backends(["nativert"], self.source)
                build.assert_not_called()

    def test_existing_nativert_library_is_reused_and_cuda_override_is_forwarded(self):
        library = self.root / "existing NativeRT.so"
        library.touch()
        os.environ.update(
            PYTORCH_SOURCE=str(self.pytorch),
            NATIVERT_LIBRARY=str(library),
            CUDA_ROOT=str(self.root / "default cuda"),
            CUDAToolkit_ROOT=str(self.root / "selected cuda"),
        )

        def install(source, build, prefix, options, jobs):
            output = prefix / builder.LIBRARIES["nativert"]
            output.parent.mkdir(parents=True)
            output.touch()

        with patch.object(builder.shutil, "which", return_value="cmake"):
            with patch.object(builder, "cmake_build", side_effect=install) as build:
                builder.build_backends(
                    ["nativert"], self.source, install_prefix=self.prefix
                )
        self.assertEqual(build.call_count, 1)
        options = build.call_args.args[3]
        self.assertIn(f"-DNATIVERT_LIBRARY={library}", options)
        self.assertIn(f"-DCUDAToolkit_ROOT={self.root / 'selected cuda'}", options)

    def test_build_errors_propagate_and_cli_returns_failure(self):
        with patch.object(sys, "argv", [str(SCRIPT), "--backends", "torchscript"]):
            with patch.object(
                builder,
                "build_backends",
                side_effect=subprocess.CalledProcessError(7, "cmake"),
            ):
                with contextlib.redirect_stderr(io.StringIO()) as error:
                    self.assertEqual(builder.main(), 1)
        self.assertIn("Failed to build", error.getvalue())

    def test_no_output_library_is_reported_as_failure(self):
        with patch.object(builder.shutil, "which", return_value="cmake"):
            with patch.object(builder, "cmake_build"):
                with self.assertRaisesRegex(RuntimeError, "did not install"):
                    builder.build_backends(["torchscript"], self.source)

    def test_nonpositive_parallelism_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            builder.build_backends(["torchscript"], self.source, jobs=0)


if __name__ == "__main__":
    unittest.main()
