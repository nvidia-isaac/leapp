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

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from PIL import Image

import leapp
from leapp import annotate
import leapp.leapp_graph.leapp_graph as leapp_graph_module
from leapp.leapp_graph.leapp_graph import LeappGraph


@dataclass
class FakeTensorDescription:
    name_str: str
    dtype: str
    shape: tuple[int, ...]
    tag: str | None = None
    semantics: dict | None = None

    def get_semantics(self):
        return self.semantics or {}


@dataclass
class FakeNode:
    name: str
    inputs: list[FakeTensorDescription]
    outputs: list[FakeTensorDescription]
    backend: str | None = "jit-script"
    node_index: int = 0


def test_leapp_graph_visualize_writes_png(tmp_path: Path):
    node = FakeNode(
        name="policy",
        inputs=[FakeTensorDescription("obs", "float32", (1, 12))],
        outputs=[FakeTensorDescription("action", "float32", (1, 4), semantics={"kind": "command"})],
        backend="jit-script",
    )

    graph = LeappGraph(nodes={"policy": node})
    graph.visualize(save_path=str(tmp_path), graph_name="demo")

    png_path = tmp_path / "demo.png"
    assert png_path.exists()
    assert not (tmp_path / "demo.svg").exists()
    assert Image.open(png_path).size[0] > 0


def test_compile_graph_raises_when_visualization_fails(tmp_path: Path, monkeypatch):
    def fail_render(*args, **kwargs):
        raise RuntimeError("visualization exploded")

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

    with pytest.raises(RuntimeError, match="visualization exploded"):
        leapp.compile_graph(visualize=True, validate=False)
