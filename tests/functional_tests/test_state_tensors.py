#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for state tensor API (state_tensors and update_state)."""

import unittest
import torch
from leapp import annotate
from .base import LEAPPFunctionalTestBase


class TestStateTensors(LEAPPFunctionalTestBase):
    """Tests for the state_tensors API which handles tensors that are both inputs and outputs."""

    def test_state_tensor_basic(self):
        """Test basic state tensor: input -> computation -> updated state output."""
        annotate.start(name=self.TEST_GRAPH_NAME)

        # Create regular input
        obs = annotate.input_tensors(
            "policy", {"observation": torch.tensor([1.0, 2.0, 3.0])}
        )

        # Create state tensor (both input and output) — non-trivial initial value
        initial_counter = torch.tensor([5.0, -3.0, 0.5])
        counter = annotate.state_tensors(
            "policy", {"counter": initial_counter}
        )

        # Use state in computation
        new_counter = counter + obs

        # Update state with new value
        annotate.update_state("policy", {"counter": new_counter})

        # Regular output
        action = obs * 2.0
        annotate.output_tensors("policy", {"action": action}, export_with="jit")

        annotate.stop()
        annotate.compile_graph(visualize=False)

        # Verify structure: 1 dangling input (observation), 1 dangling output (action),
        # 1 feedback connection (counter -> counter_out)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0,
            feedback_connections=1
        )
        self.verify_all_models_exist("policy")

        # Verify model execution
        model_info = self.inspect_torchscript_model("policy")
        # Should have 2 inputs and 2 outputs
        self.assertEqual(len(model_info["inputs"]), 3)  # self + 2 tensors
        self.assertEqual(len(model_info["outputs"]), 2)

        # Verify feedback initial values safetensors file
        self.verify_feedback_initial_values({
            "policy/counter": initial_counter,
        })

    def test_state_tensor_history_buffer(self):
        """Test state tensor for observation history buffer (shift and append pattern)."""
        history_length = 3
        obs_dim = 4

        annotate.start(name=self.TEST_GRAPH_NAME)

        # Current observation
        current_obs = annotate.input_tensors(
            "obs_processor", {"current_obs": torch.tensor([1.0, 2.0, 3.0, 4.0])}
        )

        # History buffer as state [batch, history_len, obs_dim] — non-trivial initial value
        initial_history = torch.arange(history_length * obs_dim, dtype=torch.float32).reshape(1, history_length, obs_dim)
        history = annotate.state_tensors(
            "obs_processor", {"observation_history": initial_history}
        )

        # Shift history: drop oldest, append newest
        new_history = torch.cat(
            [history[:, 1:, :], current_obs.unsqueeze(0).unsqueeze(1)], dim=1
        )

        # Update state
        annotate.update_state("obs_processor", {"observation_history": new_history})

        # Output flattened history for policy
        flat_history = new_history.reshape(1, -1)
        annotate.output_tensors(
            "obs_processor", {"flat_history": flat_history}, export_with="jit"
        )

        annotate.stop()
        annotate.compile_graph(visualize=False)

        # Verify structure: 1 dangling input, 1 dangling output,
        # 1 feedback connection (observation_history)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0,
            feedback_connections=1
        )
        self.verify_all_models_exist("obs_processor")

        # Verify feedback initial values safetensors file
        self.verify_feedback_initial_values({
            "obs_processor/observation_history": initial_history,
        })

    def test_state_tensor_multiple_states(self):
        """Test multiple state tensors in a single node."""
        annotate.start(name=self.TEST_GRAPH_NAME)

        # Input
        obs = annotate.input_tensors(
            "policy", {"observation": torch.tensor([1.0, 2.0, 3.0])}
        )

        # Multiple state tensors - returns tuple — non-trivial initial values
        initial_mean = torch.tensor([0.1, -0.2, 0.3])
        initial_var = torch.tensor([1.5, 2.0, 0.8])
        initial_count = torch.tensor([42.0])
        running_mean, running_var, step_count = annotate.state_tensors(
            "policy",
            {
                "running_mean": initial_mean,
                "running_var": initial_var,
                "step_count": initial_count,
            },
        )

        # Update running statistics (simplified)
        new_mean = running_mean * 0.9 + obs * 0.1
        new_var = running_var * 0.9 + (obs - new_mean) ** 2 * 0.1
        new_count = step_count + 1.0

        # Update all states
        annotate.update_state(
            "policy",
            {
                "running_mean": new_mean,
                "running_var": new_var,
                "step_count": new_count,
            },
        )

        # Normalized output
        normalized = (obs - new_mean) / (new_var.sqrt() + 1e-8)
        annotate.output_tensors("policy", {"normalized": normalized}, export_with="jit")

        annotate.stop()
        annotate.compile_graph(visualize=False)

        # Verify structure: 1 dangling input (observation), 1 dangling output (normalized),
        # 3 feedback connections (running_mean, running_var, step_count)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0,
            feedback_connections=3
        )
        self.verify_all_models_exist("policy")

        # Verify feedback initial values safetensors file
        self.verify_feedback_initial_values({
            "policy/running_mean": initial_mean,
            "policy/running_var": initial_var,
            "policy/step_count": initial_count,
        })

    def test_state_tensor_passthrough(self):
        """Test that state passes through unchanged when update_state is not called."""
        annotate.start(name=self.TEST_GRAPH_NAME)

        # Input
        obs = annotate.input_tensors(
            "policy", {"observation": torch.tensor([1.0, 2.0, 3.0])}
        )

        # State tensor without calling update_state — non-trivial initial value
        initial_hidden = torch.tensor([-1.0, 0.5, 2.5])
        hidden = annotate.state_tensors("policy", {"hidden": initial_hidden})

        # Use state but don't update it (passthrough)
        action = obs + hidden

        # Output without updating state
        annotate.output_tensors("policy", {"action": action}, export_with="jit")

        annotate.stop()
        annotate.compile_graph(visualize=False)

        # State should still appear in outputs as passthrough
        # 1 dangling input (observation), 1 dangling output (action),
        # 1 feedback connection (hidden passthrough)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0,
            feedback_connections=1
        )
        self.verify_all_models_exist("policy")

        # Verify feedback initial values safetensors file
        self.verify_feedback_initial_values({
            "policy/hidden": initial_hidden,
        })

    def test_state_tensor_multi_step_simulation(self):
        """Test state tensors with multiple simulation steps (like real usage)."""
        annotate.start(name=self.TEST_GRAPH_NAME)

        # Simulate multiple steps — non-trivial initial history
        history = torch.linspace(-1.0, 1.0, 12).reshape(1, 3, 4)

        for step in range(5):
            # Current observation
            current_obs = annotate.input_tensors(
                "policy", {"current_obs": torch.randn(1, 4)}
            )

            # History as state
            history_state = annotate.state_tensors("policy", {"history": history})

            # Update history
            new_history = torch.cat(
                [history_state[:, 1:, :], current_obs.unsqueeze(1)], dim=1
            )
            annotate.update_state("policy", {"history": new_history})

            # Compute action from history
            flat = new_history.reshape(1, -1)
            action = flat.mean(dim=1, keepdim=True)
            annotate.output_tensors("policy", {"action": action}, export_with="jit")

            # Update for next step (only affects non-traced execution)
            history = new_history.tensor if hasattr(new_history, "tensor") else new_history

        annotate.stop()
        annotate.compile_graph(visualize=False)

        # Should have traced just one iteration
        # 1 dangling input (current_obs), 1 dangling output (action),
        # 1 feedback connection (history)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0,
            feedback_connections=1
        )
        self.verify_all_models_exist("policy")

        # Verify feedback initial values safetensors file
        self.verify_feedback_initial_values({
            "policy/history": torch.linspace(-1.0, 1.0, 12).reshape(1, 3, 4),
        })


class TestStateTensorErrors(LEAPPFunctionalTestBase):
    """Tests for error handling in state tensor API."""

    def test_state_tensor_without_input_tensors_raises(self):
        """Test that state_tensors raises error if input_tensors wasn't called first."""
        annotate.start(name=self.TEST_GRAPH_NAME)

        with self.assertRaises(Exception):
            # Should fail because no node "policy" exists yet
            annotate.state_tensors("policy", {"state": torch.zeros(3)})

        annotate.stop()

    def test_update_state_unknown_name_raises(self):
        """Test that update_state raises error for unregistered state name."""
        annotate.start(name=self.TEST_GRAPH_NAME)

        # Create node and state
        annotate.input_tensors("policy", {"obs": torch.zeros(3)})
        annotate.state_tensors("policy", {"registered_state": torch.zeros(3)})

        with self.assertRaises(Exception):
            # Should fail because "unknown_state" wasn't registered
            annotate.update_state("policy", {"unknown_state": torch.zeros(3)})

        annotate.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
