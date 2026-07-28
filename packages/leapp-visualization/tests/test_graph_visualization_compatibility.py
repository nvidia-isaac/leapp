#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pathlib import Path

import pytest
import torch

import leapp
from leapp import annotate
import leapp.leapp_graph.leapp_graph as leapp_graph_module


ROOT = Path(__file__).resolve().parents[3]


def test_visualization_dependency_is_conditional_on_python_311():
    root_pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in root_pyproject
    assert (
        '"fast-sugiyama>=0.5.3; python_version >= \'3.11\'"' in root_pyproject
    )
    assert '"Pillow>=10.0.0; python_version >= \'3.11\'"' in root_pyproject


def test_visualization_modules_are_shipped_with_leapp():
    root_pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    viz_src = ROOT / "packages" / "leapp-visualization" / "src" / "leapp_visualization"

    assert 'where = [".", "packages/leapp-visualization/src"]' in root_pyproject
    assert '"leapp_visualization*"' in root_pyproject
    assert "leapp-visualization==" not in root_pyproject
    assert "[tool.uv.sources]" not in root_pyproject
    assert viz_src.is_dir()
    assert (viz_src / "__init__.py").is_file()


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
