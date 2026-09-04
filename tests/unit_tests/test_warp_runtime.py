#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
from types import SimpleNamespace

import pytest

from leapp import warp_runtime

_BUILD_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "leapp-build-warp-runtime"
)
_BUILD_SCRIPT_LOADER = SourceFileLoader(
    "leapp_build_warp_runtime", str(_BUILD_SCRIPT_PATH)
)
_BUILD_SCRIPT_SPEC = spec_from_loader(
    _BUILD_SCRIPT_LOADER.name, _BUILD_SCRIPT_LOADER
)
assert _BUILD_SCRIPT_SPEC is not None
_BUILD_SCRIPT = module_from_spec(_BUILD_SCRIPT_SPEC)
_BUILD_SCRIPT_LOADER.exec_module(_BUILD_SCRIPT)


def _create_build_resources(tmp_path, windows=False):
    warp_dir = tmp_path / "warp"
    (warp_dir / "bin").mkdir(parents=True)
    (warp_dir / "native").mkdir()
    (warp_dir / "bin" / ("warp.dll" if windows else "warp.so")).touch()
    (warp_dir / "native" / "apic.h").touch()

    ort_dir = tmp_path / "onnxruntime"
    (ort_dir / "capi").mkdir(parents=True)
    ort_library = (
        "onnxruntime.dll" if windows else "libonnxruntime.so.1.2.3"
    )
    (ort_dir / "capi" / ort_library).touch()

    torch_dir = tmp_path / "torch"
    (torch_dir / "share" / "cmake" / "Torch").mkdir(parents=True)
    (torch_dir / "share" / "cmake" / "Torch" / "TorchConfig.cmake").touch()

    modules = {
        "warp": SimpleNamespace(__file__=str(warp_dir / "__init__.py")),
        "onnxruntime": SimpleNamespace(
            __file__=str(ort_dir / "__init__.py"),
            __version__="1.2.3",
        ),
        "torch": SimpleNamespace(
            utils=SimpleNamespace(
                cmake_prefix_path=str(torch_dir / "share" / "cmake")
            )
        ),
    }
    return modules


def test_cache_directory_respects_xdg_cache_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert warp_runtime.warp_runtime_build_dir() == (
        tmp_path / "leapp" / "warp-runtime" / "build"
    )


def test_windows_cache_and_artifact_names(tmp_path, monkeypatch):
    monkeypatch.setattr(warp_runtime.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    assert warp_runtime.warp_runtime_build_dir() == (
        tmp_path / "leapp" / "warp-runtime" / "build"
    )
    assert {
        backend: path.name
        for backend, path in warp_runtime.warp_runtime_artifact_paths().items()
    } == {
        "onnx": "leapp_wrp_onnx_custom_op.dll",
        "pt2": "leapp_wrp_torch_custom_op.dll",
    }


def test_environment_override_wins_over_cached_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    cache_library = (
        warp_runtime.warp_runtime_build_dir()
        / "libleapp_wrp_onnx_custom_op.so"
    )
    cache_library.parent.mkdir(parents=True)
    cache_library.touch()
    override = tmp_path / "override.so"
    override.touch()
    monkeypatch.setenv("LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY", str(override))

    assert warp_runtime.resolve_warp_runtime_library("onnx") == override


def test_missing_environment_override_does_not_fall_back(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    cached_library = (
        warp_runtime.warp_runtime_build_dir()
        / "libleapp_wrp_torch_custom_op.so"
    )
    cached_library.parent.mkdir(parents=True)
    cached_library.touch()
    missing_override = tmp_path / "missing.so"
    monkeypatch.setenv(
        "LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY", str(missing_override)
    )

    with pytest.raises(
        FileNotFoundError, match="LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY"
    ):
        warp_runtime.resolve_warp_runtime_library("pt2")


@pytest.mark.parametrize(
    ("missing_resource", "message"),
    [
        ("warp/bin/warp.so", "native library was not found"),
        (
            "onnxruntime/capi/libonnxruntime.so.1.2.3",
            "shared library was not found",
        ),
        (
            "torch/share/cmake/Torch/TorchConfig.cmake",
            "CMake resources were not found",
        ),
    ],
)
def test_build_preflight_reports_missing_native_resources(
    tmp_path, monkeypatch, missing_resource, message
):
    modules = _create_build_resources(tmp_path)
    (tmp_path / missing_resource).unlink()
    monkeypatch.setattr(
        _BUILD_SCRIPT.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        _BUILD_SCRIPT.shutil,
        "which",
        lambda name: f"/usr/bin/{name}",
    )
    monkeypatch.setattr(
        _BUILD_SCRIPT,
        "_find_nvcc",
        lambda: Path("/usr/local/cuda/bin/nvcc"),
    )

    with pytest.raises(RuntimeError, match=message):
        _BUILD_SCRIPT._validate_build_resources()


def test_missing_python_dependency_is_not_installed(monkeypatch):
    def missing_module(name):
        raise ImportError(name)

    monkeypatch.setattr(
        _BUILD_SCRIPT.importlib, "import_module", missing_module
    )

    with pytest.raises(RuntimeError, match="never installs dependencies"):
        _BUILD_SCRIPT._require_module("warp", 'pip install "leapp[warp]"')


def test_windows_preflight_reports_missing_warp_import_library(
    tmp_path, monkeypatch
):
    modules = _create_build_resources(tmp_path, windows=True)
    monkeypatch.setattr(_BUILD_SCRIPT.sys, "platform", "win32")
    monkeypatch.delenv("LEAPP_WARP_IMPORT_LIBRARY", raising=False)
    monkeypatch.setattr(
        _BUILD_SCRIPT.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        _BUILD_SCRIPT.shutil,
        "which",
        lambda name: f"C:/{name}.exe",
    )

    with pytest.raises(RuntimeError, match="LEAPP_WARP_IMPORT_LIBRARY"):
        _BUILD_SCRIPT._validate_build_resources()


def test_build_configures_both_adapters(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "CMakeLists.txt").touch()
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    stale_file = build_dir / "stale"
    stale_file.touch()
    commands = []

    def run(command, check):
        commands.append(command)
        if "--build" in command:
            for path in warp_runtime.warp_runtime_artifact_paths(
                build_dir
            ).values():
                path.touch()

    monkeypatch.setattr(
        _BUILD_SCRIPT, "_runtime_source_dir", lambda: source_dir
    )
    monkeypatch.setattr(
        _BUILD_SCRIPT,
        "_validate_build_resources",
        lambda: (Path("/usr/local/cuda/bin/nvcc"), None),
    )
    monkeypatch.setattr(_BUILD_SCRIPT.subprocess, "run", run)

    artifacts = _BUILD_SCRIPT.build_warp_runtime(build_dir)

    assert len(commands) == 2
    assert "-DLEAPP_WARP_BUILD_ONNX=ON" in commands[0]
    assert "-DLEAPP_WARP_BUILD_TORCH=ON" in commands[0]
    assert set(artifacts) == {"onnx", "pt2"}
    assert not stale_file.exists()


def test_discovery_reports_cache_and_missing_artifacts(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY", raising=False)
    monkeypatch.delenv("LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY", raising=False)
    onnx_library = (
        warp_runtime.warp_runtime_build_dir()
        / "libleapp_wrp_onnx_custom_op.so"
    )
    onnx_library.parent.mkdir(parents=True)
    onnx_library.touch()

    assert _BUILD_SCRIPT.show_status() == 1
    output = capsys.readouterr().out
    assert f"onnx: {onnx_library} (cache)" in output
    assert "pt2:" in output
    assert "(missing)" in output
