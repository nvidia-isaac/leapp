#!/usr/bin/env python3

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

import shutil
import unittest

import torch

import leapp
from leapp import annotate
from leapp.export_manager import ExportManager
from leapp import InferenceManager
from leapp.leapp import _MANAGER as _annotate_manager

from .base_example_test import BaseExampleTest


class TestStatefulGRUExport(BaseExampleTest):
    """Tests for examples/stateful_gru_export.py — stateful GRU buffer tracking."""

    def setUp(self):
        super().setUp()
        self._output_dir = self.repo_root / "stateful_gru_test_output"
        if self._output_dir.exists():
            shutil.rmtree(self._output_dir)

    def tearDown(self):
        super().tearDown()
        # Reset singleton state left by inline export
        if ExportManager.is_interpret_graph_enabled():
            ExportManager.set_interpret_graph(False)
        _annotate_manager.reset_nodes()
        if self._output_dir.exists():
            shutil.rmtree(self._output_dir)

    def _export_gru(self):
        """Run GRU export inline and return the InferenceManager yaml path."""
        from examples.stateful_gru_export import GRUPolicy

        model = GRUPolicy()
        model.eval()
        obs = torch.randn(1, 16)

        leapp.start("stateful_gru", save_path=str(self._output_dir))
        obs_traced = annotate.input_tensors("policy", {"obs": obs})
        annotate.module("policy", model)
        action = model(obs_traced)
        annotate.output_tensors("policy", {"action": action}, export_with="onnx-torchscript")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        return self._output_dir / "stateful_gru" / "stateful_gru.yaml"

    def test_stateful_gru_export_files(self):
        """Export produces the expected ONNX model and YAML files."""
        yaml_path = self._export_gru()
        graph_dir = self._output_dir / "stateful_gru"

        self.assertTrue(yaml_path.exists(), f"YAML not found: {yaml_path}")
        self.assertTrue((graph_dir / "policy.onnx").exists(), "policy.onnx not found")

    def test_stateful_gru_inference_manager_loads(self):
        """InferenceManager loads the exported graph without errors."""
        yaml_path = self._export_gru()
        manager = InferenceManager(str(yaml_path))
        self.assertIsNotNone(manager)
        self.assertIn("policy/obs", manager.inputs)
        self.assertIn("policy/action", manager.outputs)

    def test_stateful_gru_h_state_updates_between_calls(self):
        """Hidden state (h_state) is updated between consecutive run_policy calls.

        This is TG5: verifies that feedback connections carry the GRU hidden
        state from one step to the next, so the model is truly stateful.
        """
        yaml_path = self._export_gru()
        manager = InferenceManager(str(yaml_path))

        obs = torch.randn(1, 16)
        inputs = {"policy/obs": obs}

        # Initial hidden state should be all zeros
        h_initial = manager.value_dict["policy"]["h_state_in"].clone()
        self.assertTrue(
            torch.allclose(h_initial, torch.zeros_like(h_initial)),
            "Expected h_state to start as zeros")

        # First call — h_state should change from zeros
        manager.run_policy(inputs)
        h_after_first = manager.value_dict["policy"]["h_state_in"].clone()
        self.assertFalse(
            torch.allclose(h_after_first, h_initial),
            "h_state was not updated after the first call")

        # Second call — h_state should change again (different from after first call)
        manager.run_policy(inputs)
        h_after_second = manager.value_dict["policy"]["h_state_in"].clone()
        self.assertFalse(
            torch.allclose(h_after_second, h_after_first),
            "h_state was not updated between the first and second calls")


if __name__ == "__main__":
    unittest.main(verbosity=2)
