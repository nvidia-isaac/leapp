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

import unittest

from tests.functional_tests.base import LEAPPFunctionalTestBase
from leapp import InferenceManager
from leapp import annotate
import torch
import leapp
import os
import yaml
import shutil
import hashlib
class TestInferenceManagerRobustness(LEAPPFunctionalTestBase):
    """Tests for InferenceManager resilience to YAML variations."""

    def _export_simple_model(self):
        """Export a minimal jit model and return the yaml path."""
        @annotate.method(export_with="jit")
        def simple_model(x: torch.Tensor):
            return x * 2.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        simple_model(torch.tensor([1.0, 2.0, 3.0]))
        leapp.stop()
        leapp.compile_graph(visualize=False)
        return os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")

    def test_sha256_mismatch_raises(self):
        """Corrupting the model file after export raises ValueError on load."""

        @annotate.method(export_with="jit")
        def simple_model(x: torch.Tensor):
            return x * 2.0

        input_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)
        simple_model(input_tensor)
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Corrupt the model file by appending garbage bytes
        model_path = os.path.join(self.TEST_GRAPH_NAME, "simple_model.pt")
        with open(model_path, "ab") as f:
            f.write(b"\x00\xFF\x00\xFF")

        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        with self.assertRaises(ValueError) as ctx:
            InferenceManager(yaml_path)

        self.assertIn("SHA256 checksum mismatch", str(ctx.exception))

    def test_inference_manager_bad_path_raises(self):
        """Loading a missing YAML path should raise FileNotFoundError."""
        missing_path = os.path.join(self.TEST_GRAPH_NAME, "does_not_exist.yaml")
        with self.assertRaises(FileNotFoundError) as ctx:
            InferenceManager(missing_path)
        self.assertIn("Leapp description file not found", str(ctx.exception))

    def test_inference_manager_non_yaml_extension_raises(self):
        """Loading a non-.yaml file should raise ValueError."""
        yaml_path = self._export_simple_model()
        txt_path = os.path.join(self.TEST_GRAPH_NAME, "graph.txt")
        shutil.copy(yaml_path, txt_path)

        with self.assertRaises(ValueError) as ctx:
            InferenceManager(txt_path)
        self.assertIn("must end with .yaml", str(ctx.exception))

    def test_inference_manager_missing_top_level_keys_raises(self):
        """Missing models/pipeline/system information should fail at load time."""
        yaml_path = self._export_simple_model()

        for missing_key in ["models", "pipeline", "system information"]:
            with self.subTest(missing_key=missing_key):
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)
                data.pop(missing_key, None)
                with open(yaml_path, "w") as f:
                    yaml.dump(data, f)

                with self.assertRaises(ValueError) as ctx:
                    InferenceManager(yaml_path)
                self.assertIn(
                    "Leapp description file must contain models, pipeline, and system_info",
                    str(ctx.exception),
                )

                yaml_path = self._export_simple_model()

    def test_inference_manager_missing_model_parameter_keys_raises(self):
        """Missing required parameter keys should fail before node creation."""
        yaml_path = self._export_simple_model()

        for missing_key in ["model_path", "md5sum", "sha256sum", "backend"]:
            with self.subTest(missing_key=missing_key):
                with open(yaml_path) as f:
                    data = yaml.safe_load(f)

                first_model = next(iter(data["models"].values()))
                first_model["parameters"].pop(missing_key, None)

                with open(yaml_path, "w") as f:
                    yaml.dump(data, f)

                with self.assertRaises(ValueError) as ctx:
                    InferenceManager(yaml_path)
                self.assertIn(
                    "Model description must contain model_path, md5sum, sha256sum, and backend",
                    str(ctx.exception),
                )

                yaml_path = self._export_simple_model()

    def test_inference_manager_missing_feedback_flow(self):
        """InferenceManager loads successfully when feedback_flow is absent from YAML."""

        yaml_path = self._export_simple_model()
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        data['pipeline'].pop('feedback_flow', None)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        manager = InferenceManager(yaml_path)
        self.assertIsNotNone(manager)
        self.assertEqual(manager.feedback_inputs, [])

    def test_inference_manager_missing_data_flow(self):
        """InferenceManager loads successfully when data_flow is absent from YAML."""
        yaml_path = self._export_simple_model()
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        data['pipeline'].pop('data_flow', None)
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        manager = InferenceManager(yaml_path)
        self.assertIsNotNone(manager)


    # -----------------------------------------------------------------
    # TG3: Feedback state loading
    # -----------------------------------------------------------------

    def test_feedback_initial_values_loaded_into_value_dict(self):
        """Initial state values from safetensors are loaded into value_dict, not left as zeros."""
        initial_counter = torch.tensor([42.0, 43.0, 44.0])
        obs = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"observation": obs})
        counter = annotate.state_tensors("policy", {"counter": initial_counter})
        new_counter = counter + obs_traced
        annotate.update_state("policy", {"counter": new_counter})
        annotate.output_tensors("policy", {"action": obs_traced * 2.0}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        manager = InferenceManager(yaml_path)

        loaded = manager.value_dict["policy"]["counter"]
        self.assertTrue(
            torch.allclose(loaded.cpu(), initial_counter),
            f"Expected initial_counter {initial_counter}, got {loaded}")

    def test_reset_zeros_runtime_buffers_then_repopulates_feedback(self):
        """reset() zeroes runtime caches and then restores feedback initial values."""
        initial_counter = torch.tensor([42.0, 43.0, 44.0])
        obs = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"observation": obs})
        counter = annotate.state_tensors("policy", {"counter": initial_counter})
        new_counter = counter + obs_traced
        annotate.update_state("policy", {"counter": new_counter})
        annotate.output_tensors("policy", {"action": obs_traced * 2.0}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        manager = InferenceManager(yaml_path)

        manager.run_policy({"policy/observation": obs.to(manager.nodes["policy"].device)})
        manager.reset()

        self.assertTrue(torch.allclose(
            manager.value_dict["policy"]["observation"].cpu(),
            torch.zeros_like(obs),
        ))
        self.assertTrue(torch.allclose(
            manager.value_dict["policy"]["counter"].cpu(),
            initial_counter,
        ))
        self.assertTrue(torch.allclose(
            manager.value_dict["==out=="]["policy/action"].cpu(),
            torch.zeros_like(obs),
        ))

    def test_feedback_malformed_safetensors_key_raises(self):
        """A safetensors key with more than one slash raises an error on load."""
        from leapp import InferenceManager
        from safetensors.torch import save_file

        initial_counter = torch.tensor([1.0, 2.0, 3.0])
        obs = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"observation": obs})
        counter = annotate.state_tensors("policy", {"counter": initial_counter})
        annotate.update_state("policy", {"counter": counter + obs_traced})
        annotate.output_tensors("policy", {"action": obs_traced * 2.0}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Overwrite the safetensors file with a malformed key (extra slash)
        safetensors_path = os.path.join(
            self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}_initial_values.safetensors")
        save_file({"policy/counter/extra": initial_counter}, safetensors_path)

        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        with self.assertRaises(ValueError) as ctx:
            InferenceManager(yaml_path)
        self.assertIn("too many values to unpack", str(ctx.exception))

    # -----------------------------------------------------------------
    # TG4: Fan-out clone independence
    # -----------------------------------------------------------------

    def test_fanout_clone_independence(self):
        """When one output feeds multiple targets, second+ consumers get independent clones."""
        # Export two independent single-input/output jit models
        trace_input = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)
        x_a = annotate.input_tensors("node_a", {"x": trace_input})
        annotate.output_tensors("node_a", {"out": x_a * 2.0}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Copy node_a's model as node_b (same architecture, different name in YAML)
        shutil.copy(
            os.path.join(self.TEST_GRAPH_NAME, "node_a.pt"),
            os.path.join(self.TEST_GRAPH_NAME, "node_b.pt"))

        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Add node_b as a second model (same shape as node_a)
        model_bytes = open(os.path.join(self.TEST_GRAPH_NAME, "node_b.pt"), "rb").read()
        sha256 = hashlib.sha256(model_bytes).hexdigest()
        md5 = hashlib.md5(model_bytes).hexdigest()
        data['models']['node_b'] = {
            'inputs': [{'name': 'x', 'dtype': 'float32', 'shape': [3], 'type': 'tensor'}],
            'outputs': [{'name': 'out', 'dtype': 'float32', 'shape': [3], 'type': 'tensor'}],
            'parameters': {'model_path': 'node_b.pt', 'backend': 'jit',
                           'md5sum': md5, 'sha256sum': sha256},
        }
        # node_a/out fans out: to node_b/x (data_flow) AND stays as pipeline output
        data['pipeline']['data_flow'] = {'node_a/out': ['node_b/x']}
        data['pipeline']['outputs'] = {'node_a': ['out'], 'node_b': ['out']}
        data['pipeline']['inputs'] = {'node_a': ['x']}
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)

        manager = InferenceManager(yaml_path)
        result = manager.run_policy({"node_a/x": trace_input})

        # Both consumers should have the same value
        node_a_out = result["node_a/out"]
        node_b_out = result["node_b/out"]
        self.assertTrue(torch.allclose(node_a_out, trace_input * 2.0))
        self.assertTrue(torch.allclose(node_b_out, trace_input * 2.0 * 2.0))

        # The value routed to node_b's input must be a distinct tensor from
        # the cached pipeline output for node_a/out.
        node_b_input = manager.value_dict["node_b"]["x"]
        node_a_output_cache = manager.value_dict["==out=="]["node_a/out"]
        self.assertFalse(
            node_b_input.data_ptr() == node_a_output_cache.data_ptr(),
            "Second consumer shares storage with first consumer",
        )

    def test_split_output_node_runs_without_keyerror(self):
        """A node can have one routed output and a different pipeline-only output."""
        from leapp import InferenceManager

        trace_input = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("node_a", {"x": trace_input})
        to_b, _ = annotate.output_tensors(
            "node_a",
            {"to_b": x * 2.0, "final_a": x + 10.0},
            export_with="jit",
        )
        y = annotate.input_tensors("node_b", {"x": to_b})
        annotate.output_tensors("node_b", {"final_b": y - 1.0}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        manager = InferenceManager(yaml_path)

        self.assertIn("x", manager.organized_pipeline_connections["node_a"])
        self.assertIn("final_a", manager.organized_pipeline_connections["node_a"])

        result = manager.run_policy({"node_a/x": trace_input})

        self.assertTrue(torch.allclose(result["node_a/final_a"], trace_input + 10.0))
        self.assertTrue(torch.allclose(result["node_b/final_b"], trace_input * 2.0 - 1.0))

    def test_unroutable_output_raises_on_load(self):
        """Malformed YAML with an output removed from routing should fail at load time."""
        from leapp import InferenceManager

        trace_input = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("node_a", {"x": trace_input})
        to_b, _ = annotate.output_tensors(
            "node_a",
            {"to_b": x * 2.0, "final_a": x + 10.0},
            export_with="jit",
        )
        y = annotate.input_tensors("node_b", {"x": to_b})
        annotate.output_tensors("node_b", {"final_b": y - 1.0}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        yaml_path = os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        data["pipeline"]["outputs"]["node_a"] = []

        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        with self.assertRaises(ValueError) as ctx:
            InferenceManager(yaml_path)

        self.assertIn("unroutable outputs", str(ctx.exception))
        self.assertIn("node_a", str(ctx.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)