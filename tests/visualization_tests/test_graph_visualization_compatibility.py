#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path

import pytest
import torch

import leapp
from leapp import annotate
import leapp.leapp_graph.leapp_graph as leapp_graph_module


ROOT = Path(__file__).resolve().parents[2]


def test_visualization_dependency_is_conditional_on_python_311():
    root_pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    visualization_pyproject = (
        ROOT / "packages" / "leapp-visualization" / "pyproject.toml"
    ).read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in root_pyproject
    assert (
        '"leapp-visualization==0.5.2; python_version >= \'3.11\'"'
        in root_pyproject
    )
    assert 'requires-python = ">=3.11"' in visualization_pyproject
    assert '"fast-sugiyama>=0.5.3"' in visualization_pyproject


def test_compile_graph_warns_and_skips_unsupported_visualization(tmp_path, monkeypatch):
    def fail_render(*args, **kwargs):
        raise AssertionError("renderer must not be loaded below Python 3.11")

    monkeypatch.setattr(
        leapp_graph_module,
        "_visualization_supported",
        lambda: False,
        raising=False,
    )
    monkeypatch.setattr(
        leapp_graph_module,
        "_render_visual_graph",
        fail_render,
        raising=False,
    )

    leapp.start(name="demo", save_path=str(tmp_path), dry_run=True)
    traced_obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 2)})
    annotate.output_tensors("policy", {"action": traced_obs}, export_with="jit")
    leapp.stop()

    with pytest.warns(RuntimeWarning, match="requires Python 3.11 or later"):
        leapp.compile_graph(visualize=True, validate=False)

    assert (tmp_path / "demo" / "demo.yaml").exists()
    assert not (tmp_path / "demo" / "demo.png").exists()
