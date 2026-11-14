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
"""Tests for state tensor API (state_tensors and update_state)."""

import itertools
import os
import re
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn
import leapp
from leapp import InferenceManager
from leapp.buffer_tracker import BufferTracker
from leapp.leapp import _MANAGER as annotate
from leapp.leapp_graph.datatypes import TracedTensor, TracedNpArray
from .base import LEAPPFunctionalTestBase


def _assert_plain(test, tensor):
    """A plain tensor, not a TracedTensor or a view of one."""
    test.assertIs(type(tensor), torch.Tensor)
    test.assertNotIsInstance(tensor._base, TracedTensor)


class TestStateTensors(LEAPPFunctionalTestBase):
    """Tests for the state_tensors API which handles tensors that are both inputs and outputs."""

    def test_state_api_return_shapes_single_multi_and_reentry(self):
        """state_tensors/update_state should return single tensor for one key, tuple for many keys."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        obs0 = annotate.input_tensors(
            "policy", {"observation": torch.tensor([1.0, 2.0, 3.0])}
        )

        # First trace: state_tensors returns TracedTensor (single) and tuple[TracedTensor] (multi)
        single_state = annotate.state_tensors(
            "policy", {"single": torch.tensor([0.1, 0.2, 0.3])}
        )
        self.assertIsInstance(single_state, TracedTensor)

        multi_state = annotate.state_tensors(
            "policy",
            {
                "multi_a": torch.tensor([1.0, 1.0, 1.0]),
                "multi_b": torch.tensor([2.0, 2.0, 2.0]),
            },
        )
        self.assertIsInstance(multi_state, tuple)
        self.assertEqual(len(multi_state), 2)
        self.assertTrue(all(isinstance(s, TracedTensor) for s in multi_state))

        new_single = single_state + obs0
        returned_single = annotate.update_state("policy", {"single": new_single})
        self.assertIs(returned_single, new_single)

        new_multi_a = multi_state[0] + 1.0
        new_multi_b = multi_state[1] + 2.0
        returned_multi = annotate.update_state(
            "policy",
            {"multi_a": new_multi_a, "multi_b": new_multi_b},
        )
        self.assertIsInstance(returned_multi, tuple)
        self.assertEqual(len(returned_multi), 2)
        self.assertIs(returned_multi[0], new_multi_a)
        self.assertIs(returned_multi[1], new_multi_b)

        annotate.output_tensors(
            "policy",
            {"action": obs0 + single_state + multi_state[0] + multi_state[1]},
            export_with="jit",
        )
        leapp.stop()

        # Passthrough (outside tracing): preserve single-vs-tuple return shape with raw tensors
        single_passthrough = annotate.state_tensors(
            "policy", {"single": torch.tensor([0.0, 0.0, 0.0])}
        )
        self.assertIsInstance(single_passthrough, torch.Tensor)

        multi_passthrough = annotate.state_tensors(
            "policy",
            {
                "multi_a": torch.tensor([0.0, 0.0, 0.0]),
                "multi_b": torch.tensor([0.0, 0.0, 0.0]),
            },
        )
        self.assertIsInstance(multi_passthrough, tuple)
        self.assertEqual(len(multi_passthrough), 2)
        self.assertTrue(all(isinstance(s, torch.Tensor) for s in multi_passthrough))

        upd_single_passthrough = torch.tensor([9.0, 9.0, 9.0])
        returned_single_passthrough = annotate.update_state(
            "policy", {"single": upd_single_passthrough}
        )
        self.assertIs(returned_single_passthrough, upd_single_passthrough)

        upd_multi_a = torch.tensor([8.0, 8.0, 8.0])
        upd_multi_b = torch.tensor([7.0, 7.0, 7.0])
        returned_multi_passthrough = annotate.update_state(
            "policy", {"multi_a": upd_multi_a, "multi_b": upd_multi_b}
        )
        self.assertIsInstance(returned_multi_passthrough, tuple)
        self.assertEqual(len(returned_multi_passthrough), 2)
        self.assertIs(returned_multi_passthrough[0], upd_multi_a)
        self.assertIs(returned_multi_passthrough[1], upd_multi_b)

    def test_state_tensor_basic(self):
        """Test basic state tensor: input -> computation -> updated state output."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        obs_value = torch.tensor([1.0, 2.0, 3.0])
        # Create regular input
        obs = annotate.input_tensors("policy", {"observation": obs_value})

        # Create state tensor (both input and output) — non-trivial initial value
        initial_counter = torch.tensor([5.0, -3.0, 0.5])
        counter = annotate.state_tensors("policy", {"counter": initial_counter})

        # Use state in computation
        new_counter = counter + obs

        # Update state with new value
        returned_state = annotate.update_state("policy", {"counter": new_counter})
        self.assertIs(returned_state, new_counter)

        # Regular output
        action = obs * 2.0
        annotate.output_tensors("policy", {"action": action}, export_with="jit")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Verify structure: 1 dangling input (observation), 1 dangling output (action),
        # 1 feedback connection for the recurrent state
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )
        self.verify_all_models_exist("policy")

        # Verify model execution
        model_info = self.inspect_torchscript_model("policy")
        # Should have 2 inputs and 2 outputs
        self.assertEqual(len(model_info["inputs"]), 3)  # self + 2 tensors
        self.assertEqual(len(model_info["outputs"]), 2)

        # Verify feedback initial values safetensors file
        self.verify_feedback_initial_values(
            {
                "policy/counter": initial_counter,
            }
        )
        self.verify_safetensors_matches_feedback(annotate)
        self.verify_inference_manager(
            source_inputs={"policy/observation": obs_value},
            source_outputs={"policy/action": obs_value * 2.0},
        )

    def test_state_tensor_without_update_state_becomes_regular_input(self):
        """State tensors without update_state() should not create feedback."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        obs_value = torch.tensor([1.0, 2.0, 3.0])
        initial_counter = torch.tensor([5.0, -3.0, 0.5])

        obs = annotate.input_tensors("policy", {"observation": obs_value})
        counter = annotate.state_tensors("policy", {"counter": initial_counter})

        action = obs + counter
        annotate.output_tensors("policy", {"action": action}, export_with="onnx")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=2,
            outputs=1,
            internal_connections=0,
            feedback_connections=0,
        )
        self.verify_all_models_exist("policy")
        self.verify_safetensors_matches_feedback(annotate)
        self.verify_inference_manager(
            source_inputs={
                "policy/observation": obs_value,
                "policy/counter": initial_counter,
            },
            source_outputs={"policy/action": obs_value + initial_counter},
        )

    def test_state_tensor_history_buffer(self):
        """Test state tensor for observation history buffer (shift and append pattern)."""
        history_length = 3
        obs_dim = 4

        obs_value = torch.tensor([1.0, 2.0, 3.0, 4.0])

        leapp.start(name=self.TEST_GRAPH_NAME)

        # Current observation
        current_obs = annotate.input_tensors(
            "obs_processor", {"current_obs": obs_value}
        )

        # History buffer as state [batch, history_len, obs_dim] — non-trivial initial value
        initial_history = torch.arange(
            history_length * obs_dim, dtype=torch.float32
        ).reshape(1, history_length, obs_dim)
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

        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Verify structure: 1 dangling input, 1 dangling output,
        # 1 feedback connection (observation_history)
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )
        self.verify_all_models_exist("obs_processor")

        # Verify feedback initial values safetensors file
        self.verify_feedback_initial_values(
            {
                "obs_processor/observation_history": initial_history,
            }
        )
        self.verify_safetensors_matches_feedback(annotate)

        # Compute expected output for InferenceManager (first iteration uses initial_history)
        expected_history = torch.cat(
            [initial_history[:, 1:, :], obs_value.unsqueeze(0).unsqueeze(1)], dim=1
        )
        expected_flat = expected_history.reshape(1, -1)
        self.verify_inference_manager(
            source_inputs={"obs_processor/current_obs": obs_value},
            source_outputs={"obs_processor/flat_history": expected_flat},
        )

    def test_state_tensor_multiple_states(self):
        """Test multiple state tensors in a single node."""
        obs_value = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)

        # Input
        obs = annotate.input_tensors("policy", {"observation": obs_value})

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

        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Verify structure: 1 dangling input (observation), 1 dangling output (normalized),
        # 3 feedback connections (running_mean, running_var, step_count)
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=3,
        )
        self.verify_all_models_exist("policy")

        # Verify feedback initial values safetensors file
        self.verify_feedback_initial_values(
            {
                "policy/running_mean": initial_mean,
                "policy/running_var": initial_var,
                "policy/step_count": initial_count,
            }
        )
        self.verify_safetensors_matches_feedback(annotate)

        # Compute expected output for first iteration (uses initial state values)
        exp_mean = initial_mean * 0.9 + obs_value * 0.1
        exp_var = initial_var * 0.9 + (obs_value - exp_mean) ** 2 * 0.1
        exp_normalized = (obs_value - exp_mean) / (exp_var.sqrt() + 1e-8)
        self.verify_inference_manager(
            source_inputs={"policy/observation": obs_value},
            source_outputs={"policy/normalized": exp_normalized},
        )

    def test_state_tensor_multi_step_simulation(self):
        """Test state tensors with multiple simulation steps (like real usage)."""
        initial_history = torch.linspace(-1.0, 1.0, 12).reshape(1, 3, 4)
        obs_value = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)

        # Simulate multiple steps — non-trivial initial history
        history = initial_history.clone()

        for step in range(5):
            # Current observation (same value each step for reproducibility)
            current_obs = annotate.input_tensors("policy", {"current_obs": obs_value})

            # History as state
            history_state = annotate.state_tensors("policy", {"history": history})

            # Update history
            new_history = torch.cat(
                [history_state[:, 1:, :], current_obs.unsqueeze(1)], dim=1
            )
            returned_history = annotate.update_state("policy", {"history": new_history})
            self.assertIs(returned_history, new_history)

            # Compute action from history
            flat = new_history.reshape(1, -1)
            action = flat.mean(dim=1, keepdim=True)
            annotate.output_tensors("policy", {"action": action}, export_with="jit")

            # Update for next step (only affects non-traced execution)
            history = (
                new_history.tensor if hasattr(new_history, "tensor") else new_history
            )

        leapp.stop()
        leapp.compile_graph(visualize=False, validate=True)

        # Should have traced just one iteration
        # 1 dangling input (current_obs), 1 dangling output (action),
        # 1 feedback connection (history)
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )
        self.verify_all_models_exist("policy")

        # Verify feedback initial values safetensors file
        self.verify_feedback_initial_values(
            {
                "policy/history": initial_history,
            }
        )
        self.verify_safetensors_matches_feedback(annotate)

        # Compute expected output (first iteration uses initial_history)
        expected_new_history = torch.cat(
            [initial_history[:, 1:, :], obs_value.unsqueeze(1)], dim=1
        )
        expected_action = expected_new_history.reshape(1, -1).mean(dim=1, keepdim=True)
        self.verify_inference_manager(
            source_inputs={"policy/current_obs": obs_value},
            source_outputs={"policy/action": expected_action},
        )

    def test_state_reentry_reusing_prior_state_output_tensor_preserves_feedback_source(
        self,
    ):
        """Reusing the previous state output tensor should succeed because state feedback shares one source."""
        initial_last_action = torch.tensor([0.0, 0.0, 0.0])
        first_obs_value = torch.tensor([1.0, 2.0, 3.0])
        second_obs_value = torch.tensor([4.0, 5.0, 6.0])
        last_action_value = initial_last_action

        leapp.start(name=self.TEST_GRAPH_NAME)

        for step_idx, obs_value in enumerate((first_obs_value, second_obs_value)):
            obs = annotate.input_tensors("policy", {"obs": obs_value})
            last_action = annotate.state_tensors(
                "policy", {"last_action": last_action_value}
            )
            updated_state = obs + last_action
            returned_last_action = annotate.update_state(
                "policy", {"last_action": updated_state}
            )

            if step_idx == 0:
                self.assertIs(returned_last_action, updated_state)
            else:
                self.assertIsInstance(returned_last_action, torch.Tensor)

            action = obs - last_action
            annotate.output_tensors("policy", {"action": action}, export_with="onnx")

            if step_idx == 0:
                self.assertEqual("last_action", returned_last_action.output_port)
                self.assertIs(
                    annotate.nodes["policy"], returned_last_action.context_obj
                )

            last_action_value = returned_last_action

        leapp.stop()
        leapp.compile_graph(visualize=False, validate=True)

        # Should trace one node with one dangling input/output and one feedback edge
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )
        self.verify_feedback_initial_values(
            {
                "policy/last_action_in": initial_last_action,
            }
        )
        self.verify_inference_manager(
            source_inputs={"policy/obs": first_obs_value},
            source_outputs={"policy/action": first_obs_value - initial_last_action},
        )

    def test_state_update_also_published_as_regular_output(self):
        """One value may be both a state update and a regular output.

        The declared state pairing owns the feedback edge, so it stays on the
        state port. The value's own identity is the regular output, which is
        what a downstream node connects through.
        """
        initial_state = torch.tensor([0.0, 0.0, 0.0])
        obs_value = torch.tensor([1.0, 2.0, 3.0])

        leapp.start(name=self.TEST_GRAPH_NAME)

        state_value = initial_state
        for _ in range(2):
            obs = annotate.input_tensors("policy", {"obs": obs_value})
            state = annotate.state_tensors("policy", {"acc": state_value})
            updated = obs + state
            state_value = annotate.update_state("policy", {"acc": updated})
            annotate.output_tensors("policy", {"action": updated}, export_with="onnx")

            consumed = annotate.input_tensors("sink", {"action": updated})
            annotate.output_tensors(
                "sink", {"scaled": consumed * 2.0}, export_with="onnx"
            )

        leapp.stop()
        leapp.compile_graph(visualize=False, validate=True)

        policy_ports = [output.port for output in annotate.nodes["policy"].outputs]
        self.assertCountEqual(["acc", "action"], policy_ports)
        # Feedback stays on the declared state port, while the sink connects
        # through the regular output, so both ports carry a real edge. The
        # exported names carry the _in/_out suffixes the backend adds for the
        # overlapping state name; the ports underneath are unchanged.
        data_flow = {
            source: list(targets)
            for source, targets in annotate.detected_pipeline["data_flow"].items()
        }
        feedback_flow = {
            source: list(targets)
            for source, targets in annotate.detected_pipeline["feedback_flow"].items()
        }
        self.assertEqual({"policy/action": ["sink/action"]}, data_flow)
        self.assertEqual({"policy/acc_out": ["policy/acc_in"]}, feedback_flow)
        self.verify_num_connections(
            annotate,
            nodes=2,
            inputs=1,
            outputs=1,
            internal_connections=1,
            feedback_connections=1,
        )
        self.verify_feedback_initial_values({"policy/acc_in": initial_state})
        self.verify_inference_manager(
            source_inputs={"policy/obs": obs_value},
            source_outputs={"sink/scaled": (obs_value + initial_state) * 2.0},
        )


class TestStateTensorErrors(LEAPPFunctionalTestBase):
    """Tests for error handling in state tensor API."""

    def test_state_tensor_without_input_tensors_raises(self):
        """Test that state_tensors raises error if input_tensors wasn't called first."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        with self.assertRaises(Exception):
            # Should fail because no node "policy" exists yet
            annotate.state_tensors("policy", {"state": torch.zeros(3)})

        leapp.stop()

    def test_nested_state_tensor_raises(self):
        """Nested state payloads should be rejected with a clear error."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        annotate.input_tensors("policy", {"obs": torch.zeros(3)})

        with self.assertRaises(TypeError) as exc:
            annotate.state_tensors(
                "policy", {"state_payload": {"inner": torch.zeros(3)}}
            )

        self.assertIn("does not support nested state structures", str(exc.exception))
        leapp.stop()

    def test_update_state_unknown_name_raises(self):
        """Test that update_state raises error for unregistered state name."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        # Create node and state
        annotate.input_tensors("policy", {"obs": torch.zeros(3)})
        annotate.state_tensors("policy", {"registered_state": torch.zeros(3)})

        with self.assertRaises(Exception):
            # Should fail because "unknown_state" wasn't registered
            annotate.update_state("policy", {"unknown_state": torch.zeros(3)})

        leapp.stop()

    def test_nested_update_state_raises(self):
        """Nested update_state payloads should be rejected with a clear error."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        annotate.input_tensors("policy", {"obs": torch.zeros(3)})
        annotate.state_tensors("policy", {"state": torch.zeros(3)})

        with self.assertRaises(TypeError) as exc:
            annotate.update_state("policy", {"state": {"inner": torch.zeros(3)}})

        self.assertIn("does not support nested state structures", str(exc.exception))
        leapp.stop()

    def test_update_state_non_traced_tensor_raises_clean_error(self):
        """Raw tensors passed to update_state during first trace should fail cleanly."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.zeros(3)})
        state = annotate.state_tensors("policy", {"counter": torch.zeros(3)})

        with self.assertRaises(Exception) as exc:
            annotate.update_state("policy", {"counter": torch.ones(3)})

        msg = str(exc.exception)
        self.assertIn(
            "update_state() for node 'policy' received non-traced tensors", msg
        )
        self.assertNotIn("proxy", msg)

        annotate.update_state("policy", {"counter": state + obs})
        annotate.output_tensors("policy", {"action": obs + state}, export_with="jit")
        leapp.stop()


# ── Test models ──────────────────────────────────────────────────────────────


class GRUModel(nn.Module):
    """GRU with a single hidden state buffer that gets reassigned."""

    def __init__(self, obs_dim=4, hidden_dim=8, action_dim=3):
        super().__init__()
        self.gru = nn.GRU(obs_dim, hidden_dim, num_layers=1, batch_first=False)
        self.head = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("h_state", torch.zeros(1, 1, hidden_dim))

    def forward(self, obs):
        gru_out, h_out = self.gru(obs.unsqueeze(0), self.h_state)
        self.h_state = h_out
        return self.head(gru_out.squeeze(0))


class LSTMModel(nn.Module):
    """LSTM with two hidden state buffers (h and c) that get reassigned."""

    def __init__(self, obs_dim=4, hidden_dim=8, action_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, num_layers=1, batch_first=False)
        self.head = nn.Linear(hidden_dim, action_dim)
        self.register_buffer("h_state", torch.zeros(1, 1, hidden_dim))
        self.register_buffer("c_state", torch.zeros(1, 1, hidden_dim))

    def forward(self, obs):
        out, (h_out, c_out) = self.lstm(obs.unsqueeze(0), (self.h_state, self.c_state))
        self.h_state = h_out
        self.c_state = c_out
        return self.head(out.squeeze(0))


class PartialMutationModel(nn.Module):
    """Model with 3 buffers: 2 mutated, 1 constant (not mutated)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("running_mean", torch.zeros(1, 4))
        self.register_buffer("step_count", torch.tensor([0.0]))
        self.register_buffer("const_mask", torch.ones(1, 4))  # never mutated

    def forward(self, obs):
        self.running_mean = self.running_mean * 0.9 + obs * 0.1
        self.step_count = self.step_count + 1.0
        # const_mask is read but NOT reassigned
        masked = obs * self.const_mask
        return self.linear(masked + self.running_mean)


class NoMutationModel(nn.Module):
    """Model with buffers that are read but never reassigned."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))
        self.register_buffer("bias", torch.zeros(4))

    def forward(self, obs):
        return self.linear(obs * self.scale + self.bias)


class NoBufferModel(nn.Module):
    """Model with no registered buffers."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, obs):
        return self.linear(obs)


class NestedBufferModel(nn.Module):
    """Model with buffer in a nested submodule."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.ELU())
        self.rnn = GRUModel(obs_dim=8, hidden_dim=8, action_dim=4)

    def forward(self, obs):
        features = self.encoder(obs)
        return self.rnn(features)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestBufferTracking(LEAPPFunctionalTestBase):
    """Tests for annotate.module() auto-detection of stateful buffers."""

    def test_single_buffer_gru(self):
        """GRU with 1 mutated buffer -> 1 feedback connection."""
        model = GRUModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors(
            "policy", {"action": action}, export_with="onnx-torchscript"
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # 1 input (obs), 1 output (action), 1 feedback (h_state)
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )
        self.verify_all_models_exist("policy")
        self.verify_safetensors_matches_feedback(annotate)
        self.verify_feedback_initial_values(
            {
                "policy/h_state_in": torch.zeros(1, 1, 8),
            }
        )

    def test_lstm_two_buffers(self):
        """LSTM with 2 mutated buffers -> 2 feedback connections."""
        model = LSTMModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors(
            "policy", {"action": action}, export_with="onnx-torchscript"
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # 1 input (obs), 1 output (action), 2 feedback (h_state, c_state)
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=2,
        )
        self.verify_all_models_exist("policy")
        self.verify_safetensors_matches_feedback(annotate)
        self.verify_feedback_initial_values(
            {
                "policy/h_state_in": torch.zeros(1, 1, 8),
                "policy/c_state_in": torch.zeros(1, 1, 8),
            }
        )

    def test_partial_mutation(self):
        """3 buffers, 2 mutated -> 2 feedback, 1 regular input (const_mask)."""
        model = PartialMutationModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # 1 dangling input (obs), 1 dangling output (action),
        # 2 feedback (running_mean, step_count)
        # const_mask is baked as a constant (not a dynamic input)
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=2,
        )
        self.verify_all_models_exist("policy")
        self.verify_safetensors_matches_feedback(annotate)
        self.verify_feedback_initial_values(
            {
                "policy/running_mean": torch.zeros(1, 4),
                "policy/step_count": torch.tensor([0.0]),
            }
        )

    def test_no_buffers_mutated(self):
        """All buffers read but not reassigned -> 0 feedback, all regular inputs."""
        model = NoMutationModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # 1 dangling input (obs), 1 dangling output, 0 feedback
        # scale and bias are baked as constants (not dynamic inputs)
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=0,
        )
        self.verify_all_models_exist("policy")
        self.verify_safetensors_matches_feedback(annotate)

    def test_no_buffers(self):
        """Model with no registered buffers -> module() is a no-op."""
        model = NoBufferModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=0,
        )
        self.verify_all_models_exist("policy")
        self.verify_safetensors_matches_feedback(annotate)

    def test_buffer_names_filter(self):
        """Only specified buffers are tracked; others ignored."""
        model = LSTMModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        # Only track h_state, ignore c_state
        annotate.module("policy", model, buffer_names=["h_state"])
        action = model(obs_traced)

        annotate.output_tensors(
            "policy", {"action": action}, export_with="onnx-torchscript"
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # 1 input (obs), 1 output (action), 1 feedback (h_state only)
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )
        self.verify_all_models_exist("policy")

    def test_model_restored_after_tracing(self):
        """Verify model buffers are properly re-registered after tracing."""
        model = GRUModel()
        model.eval()
        obs = torch.randn(1, 4)

        # Record original buffer names
        original_buffers = dict(model.named_buffers())

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors(
            "policy", {"action": action}, export_with="onnx-torchscript"
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # After tracing, all buffers should be restored
        restored_buffers = dict(model.named_buffers())
        self.assertEqual(
            set(original_buffers.keys()),
            set(restored_buffers.keys()),
            "Buffer names should be restored after tracing",
        )

        # Restored values should be proper tensors (not TracedTensors)
        for name, buf in restored_buffers.items():
            self.assertIs(
                type(buf),
                torch.Tensor,
                f"Buffer '{name}' should be a torch.Tensor after restore",
            )
            self.assertNotIsInstance(buf._base, TracedTensor)

    def test_model_exports_after_tracing(self):
        """After a session, the model keeps its buffers and a module reading
        them exports."""

        class Normalizer(nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.ones(4))
                # A view of a parameter, so the gradient flows through it.
                self.register_buffer("mean", self.scale.view(1, 4))
                self.register_buffer("steps", torch.zeros(1))

            def forward(self, obs):
                self.steps = self.steps + 1
                return obs - self.mean

        model = Normalizer().eval()
        mean = model.mean
        # Another module holds the same tensor and keeps it.
        history = nn.Module()
        history.register_buffer("steps", model.steps)
        obs = torch.ones(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})
        annotate.module("policy", model)
        annotate.output_tensors(
            "policy", {"result": model(obs_traced)}, export_with="jit"
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Not assertIs: its failure message would print a leftover TracedTensor,
        # whose repr does not return.
        self.assertTrue(model.mean is mean)
        torch.testing.assert_close(model.steps, torch.ones(1))

        class Reader(nn.Module):
            # Reads every buffer without reassigning one, since torch 2.7 cannot
            # export a module that reassigns a buffer.
            def __init__(self, model, history):
                super().__init__()
                self.model = model
                self.history = history

            def forward(self, obs):
                model = self.model
                return obs - model.mean + model.steps + self.history.steps

        # A new input, since tracing promoted obs itself. Export before running
        # the model, which would replace the restored steps buffer.
        fresh = torch.ones(1, 4)
        reader = Reader(model, history)
        program = torch.export.export(reader, (fresh,))
        torch.testing.assert_close(program.module()(fresh), reader(fresh))
        model(fresh).sum().backward()
        self.assertIsNotNone(model.scale.grad)

    def test_model_exports_after_a_capture_loop(self):
        """After a session, a module reading the buffers exports also when a
        later pass left a returned buffer traced for the next node, or a view
        of an input, and when a buffer left out of buffer_names is the user's
        declared state."""

        class History(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("hist", torch.zeros(3, 4))

            def forward(self, obs):
                self.hist = torch.cat([self.hist[1:], obs])
                return self.hist

        class Acc(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(1, 4))

            def forward(self, obs):
                return self.h + obs

        class Viewer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(1, 4))

            def forward(self, obs):
                y = obs + self.h
                self.h = obs.view(1, 4)
                return y

        class Reader(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, obs):
                return obs + next(self.model.buffers()).sum(0, keepdim=True)

        for model in (History().eval(), Acc().eval(), Viewer().eval()):
            with self.subTest(model=type(model).__name__):
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    for _ in range(2):
                        obs = annotate.input_tensors(
                            "policy", {"obs": torch.ones(1, 4)}
                        )
                        if isinstance(model, Acc):
                            h = annotate.state_tensors("policy", {"h": model.h})
                            annotate.module("policy", model, buffer_names=[])
                            y = model(obs)
                            annotate.update_state("policy", {"h": h + obs})
                        else:
                            annotate.module("policy", model)
                            y = model(obs)
                        out = annotate.output_tensors(
                            "policy", {"y": y}, export_with="jit"
                        )
                        z = annotate.input_tensors("consumer", {"z": out})
                        annotate.output_tensors(
                            "consumer", {"w": z * 2}, export_with="jit"
                        )
                finally:
                    leapp.stop()
                self.assertTrue(all(type(b) is torch.Tensor for b in model.buffers()))
                fresh = torch.ones(1, 4)
                reader = Reader(model)
                program = torch.export.export(reader, (fresh,))
                torch.testing.assert_close(program.module()(fresh), reader(fresh))

    def test_buffer_reassigned_to_its_old_value(self):
        """A forward that stores a buffer's own value, or moves one buffer's value
        into another, exports and leaves the new values on the model."""

        class ToDevice(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.full((1, 4), 0.5))

            def forward(self, obs):
                self.mean = self.mean.to(obs.device)
                return obs - self.mean

        class Delay(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("last", torch.zeros(1, 4))
                self.register_buffer("prev", torch.zeros(1, 4))

            def forward(self, obs):
                self.prev = self.last
                self.last = obs * 10
                return obs + self.prev

        obs = torch.ones(1, 4)
        # (model, the buffers it should hold after one step)
        cases = [
            (ToDevice, {"mean": torch.full((1, 4), 0.5)}),
            (Delay, {"last": obs * 10, "prev": torch.zeros(1, 4)}),
        ]
        for model_type, expected in cases:
            for export_with in ("jit", "exported-program"):
                with self.subTest(model=model_type.__name__, export_with=export_with):
                    model = model_type().eval()
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    obs_traced = annotate.input_tensors("policy", {"obs": obs.clone()})
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"result": model(obs_traced)}, export_with=export_with
                    )
                    leapp.stop()
                    leapp.compile_graph(visualize=False)

                    for name, value in expected.items():
                        buffer = getattr(model, name)
                        _assert_plain(self, buffer)
                        self.assertNotIsInstance(buffer._base, TracedTensor)
                        torch.testing.assert_close(buffer, value)

    def test_buffer_read_by_one_node_and_declared_by_another(self):
        """A buffer one node bakes as a constant can be a later node's input."""

        class Offset(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("ref", torch.ones(3))

            def forward(self, z):
                return z + self.ref

        model = Offset().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        z = annotate.input_tensors("b", {"z": torch.ones(3)})
        annotate.module("b", model)
        annotate.output_tensors("b", {"o": model(z)}, export_with="exported-program")
        ref = annotate.input_tensors("c", {"ref": model.ref})
        annotate.output_tensors("c", {"w": ref * 2}, export_with="exported-program")
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"b": True, "c": True}
        )

    def test_buffer_old_value_returned_to_another_node(self):
        """A buffer's old value still feeds a later node, when the forward returns
        it, also under the name of the state that holds it, and when only the
        state holds it and the caller kept the buffer's tensor."""

        class OldOutput(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.full((1, 4), 3.0))

            def forward(self, obs):
                old = self.h
                self.h = obs + 1
                return old

        class Delay(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("last", torch.full((1, 4), 3.0))
                self.register_buffer("prev", torch.zeros(1, 4))

            def forward(self, obs):
                self.prev = self.last
                self.last = obs + 1
                return self.prev

        for model_type, name in ((OldOutput, "old"), (Delay, "prev")):
            with self.subTest(model=model_type.__name__):
                model = model_type().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.ones(1, 4)})
                annotate.module("policy", model)
                old = annotate.output_tensors(
                    "policy", {name: model(obs)}, export_with="jit"
                )
                c_in = annotate.input_tensors("c", {"x": old})
                annotate.output_tensors("c", {"y": c_in * 2}, export_with="jit")
                leapp.stop()
                leapp.compile_graph(visualize=False)

                data_flow = annotate.detected_pipeline["data_flow"]
                self.assertIn("c/x", list(data_flow.get(f"policy/{name}", [])))

        with self.subTest(model="Delay, kept by the caller"):
            model = Delay().eval()
            saved = model.last
            leapp.start(name=self.TEST_GRAPH_NAME)
            obs = annotate.input_tensors("policy", {"obs": torch.ones(1, 4)})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"y": model(obs) * 2}, export_with="jit")
            c_in = annotate.input_tensors("c", {"x": saved})
            annotate.output_tensors("c", {"y": c_in * 2}, export_with="jit")
            leapp.stop()
            leapp.compile_graph(visualize=False)

            data_flow = annotate.detected_pipeline["data_flow"]
            self.assertIn("c/x", list(data_flow.get("policy/prev", [])))

    def test_buffer_declared_after_module_feeds_a_later_node(self):
        """A buffer's tensor declared with input_tensors() after module() and moved
        into another buffer still feeds a later node that declares it, and a
        read-only one stays traced, like every declared input, while after the
        session the buffer holds a plain copy of it."""

        class Delay(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("last", torch.full((3,), 3.0))
                self.register_buffer("prev", torch.zeros(3))

            def forward(self, x):
                self.prev = self.last
                self.last = x + 1
                return x + self.prev

        model = Delay().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": torch.ones(3)})
        annotate.module("policy", model)
        prior = model.last + model.prev
        saved = annotate.input_tensors("policy", {"declared_last": model.last})
        annotate.output_tensors("policy", {"y": model(x) + prior}, export_with="jit")
        v = annotate.input_tensors("consumer", {"v": saved})
        annotate.output_tensors("consumer", {"z": v * 2}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        data_flow = annotate.detected_pipeline["data_flow"]
        self.assertIn("consumer/v", list(data_flow.get("policy/prev", [])))

        class Read(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.full((3,), 0.5))

            def forward(self, x):
                return x - self.mean

        model = Read().eval()
        mean = model.mean
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": torch.ones(3)})
        annotate.module("policy", model)
        annotate.input_tensors("policy", {"declared_mean": model.mean})
        annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        leapp.stop()
        self.assertIsInstance(mean, TracedTensor)
        _assert_plain(self, model.mean)
        torch.testing.assert_close(model.mean, mean.tensor)

    def test_non_persistent_buffer_stays_out_of_state_dict(self):
        """A non-persistent buffer the forward reassigns stays out of the
        model's state_dict after capture, so the state_dict still loads."""

        class Hidden(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.ones(1))
                self.register_buffer("h", torch.zeros(1), persistent=False)

            def forward(self, x):
                self.h = self.h + x
                return self.h * self.p

        model = Hidden().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": torch.ones(1)})
        annotate.module("policy", model)
        annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        leapp.stop()
        self.assertEqual(list(model.state_dict()), ["p"])
        Hidden().load_state_dict(model.state_dict())

    def test_subclass_buffer_kept_after_tracing(self):
        """A read-only buffer that is a tensor subclass is the same object after
        capture, also when it is declared as the node's state first."""

        class Tagged(torch.Tensor):
            pass

        class Offset(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "mean", torch.full((1, 4), 0.5).as_subclass(Tagged)
                )

            def forward(self, obs):
                return obs - self.mean

        for declared in (False, True):
            with self.subTest(declared=declared):
                model = Offset().eval()
                mean = model.mean
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.ones(1, 4)})
                if declared:
                    annotate.state_tensors("policy", {"mean": model.mean})
                annotate.module("policy", model)
                annotate.output_tensors(
                    "policy", {"result": model(obs)}, export_with="jit"
                )
                leapp.stop()
                self.assertTrue(model.mean is mean)
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )

    def test_untracked_buffers_on_a_copied_memory_keep_their_values(self):
        """A buffer left out of buffer_names on the memory of an input with
        autograd history that a tracked buffer is set to keeps its values
        after the session, also a lazily negated view, which the copy of that
        memory cannot carry, also one that requires gradients, a plain view of
        an input that is itself negated, one that is its own autograd leaf,
        and a tensor subclass, which keeps its type, while a plain buffer on
        those bytes through another storage views the copy, so it follows a
        later write. One the copy cannot hold in its layout keeps its values:
        unaligned, running past the copied memory, larger than it, or lazily
        conjugated."""

        class Holder(nn.Module):
            def __init__(self, other, h=None):
                super().__init__()
                self.register_buffer("h", torch.zeros(2) if h is None else h)
                self.register_buffer("n", other)

            def forward(self, x):
                self.h = x
                return x * 2

        class Tagged(torch.Tensor):
            pass

        for kind, expected in (
            ("negated", -2.0),
            ("negated with gradient", -2.0),
            ("plain view of a negated input", -2.0),
            ("leaf", 2.0),
            ("subclass", 2.0),
        ):
            with self.subTest(kind=kind):
                leaf = torch.full((2,), 2.0, requires_grad=True)
                if kind == "plain view of a negated input":
                    leaf = torch._neg_view(-leaf.detach()).detach().requires_grad_()
                    other = torch._neg_view(leaf)
                elif kind == "negated with gradient":
                    other = torch._neg_view(leaf)
                elif kind == "negated":
                    other = torch._neg_view(leaf.detach())
                elif kind == "subclass":
                    other = leaf.detach().as_subclass(Tagged)
                else:
                    other = leaf.detach().requires_grad_()
                model = Holder(other).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": leaf})
                    annotate.module("policy", model, buffer_names=["h"])
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                torch.testing.assert_close(model.n, torch.full((2,), expected))
                self.assertEqual(
                    model.n.requires_grad, kind not in ("negated", "subclass")
                )
                self.assertEqual(type(model.n) is Tagged, kind == "subclass")

        array = np.full(2, 2.0, dtype=np.float32)
        leaf = torch.from_numpy(array).requires_grad_()
        model = Holder(torch.from_numpy(array)).eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": leaf})
            annotate.module("policy", model, buffer_names=["h"])
            annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        finally:
            leapp.stop()
        with torch.no_grad():
            model.h.add_(3)
        torch.testing.assert_close(model.n, torch.full((2,), 5.0))

        for kind in ("unaligned", "past the end", "larger", "conjugated"):
            with self.subTest(kind=kind):
                array = np.arange(1, 9, dtype=np.float32)
                if kind == "larger":
                    array = array[:3].astype(np.float16)
                memory = memoryview(array).cast("B")
                leaf = torch.from_numpy(array[:4]).requires_grad_()
                if kind == "unaligned":
                    other = torch.frombuffer(
                        memory, dtype=torch.float32, count=2, offset=2
                    )
                elif kind == "past the end":
                    other = torch.from_numpy(array[2:])
                elif kind == "larger":
                    other = torch.frombuffer(memory, dtype=torch.float32, count=1)
                else:
                    other = torch.from_numpy(array[:4]).view(torch.complex64).conj()
                values = other.resolve_conj().clone()
                model = Holder(other, torch.zeros_like(leaf.detach())).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": leaf})
                    annotate.module("policy", model, buffer_names=["h"])
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                torch.testing.assert_close(model.n.resolve_conj(), values)

    def test_buffers_set_to_independent_leaves_on_one_memory(self):
        """Two buffers set to independent autograd leaves on the same memory
        keep their own gradients after the session."""

        class Pair(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.zeros(2))
                self.register_buffer("b", torch.zeros(1, 2))

            def forward(self, left, right):
                self.a = left
                self.b = right
                return left * 2

        left = torch.tensor([2.0, 4.0], requires_grad=True)
        right = left.view(1, 2).detach().requires_grad_()
        model = Pair().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            inputs = annotate.input_tensors("policy", {"left": left, "right": right})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"y": model(*inputs)}, export_with="jit")
        finally:
            leapp.stop()
        # The inputs stay traced, and .grad of one reads a new alias.
        left.clear_tracing_state()
        right.clear_tracing_state()
        model.b.sum().backward()
        self.assertIsNone(left.grad)
        torch.testing.assert_close(right.grad, torch.ones(1, 2))

    def test_buffer_gradient_kept_after_tracing(self):
        """A buffer that views a parameter and is declared with input_tensors()
        after module(), or reassigned to an input that views a parameter, holds
        a plain view of the parameter after the session, also one that ends
        without gradients, so it keeps the gradient and sees a later update of
        the parameter, and restoring it records nothing into the graph, also
        one that views it negated or in another dtype, and one made without
        gradients gets no gradient connection."""

        class Read(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.ones(2))
                self.register_buffer("mean", self.p.view(2))

            def forward(self, x):
                return x - self.mean

        model = Read()
        mean = model.mean
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": torch.ones(2)})
        annotate.module("policy", model)
        annotate.input_tensors("policy", {"declared_mean": model.mean})
        annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        with torch.no_grad():
            leapp.stop()
        self.assertIsInstance(mean, TracedTensor)
        _assert_plain(self, model.mean)
        model.mean.sum().backward()
        torch.testing.assert_close(model.p.grad, torch.ones(2))
        with torch.no_grad():
            model.p.add_(10)
        torch.testing.assert_close(model.mean, torch.full((2,), 11.0))

        class Negated(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.ones(2))
                self.register_buffer("mean", torch._neg_view(self.p))

            def forward(self, x):
                return x - self.mean

        model = Negated()
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": torch.ones(2)})
        annotate.module("policy", model)
        annotate.input_tensors("policy", {"declared_mean": model.mean})
        annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        leapp.stop()
        _assert_plain(self, model.mean)
        torch.testing.assert_close(model.mean, torch.full((2,), -1.0))
        model.mean.sum().backward()
        torch.testing.assert_close(model.p.grad, torch.full((2,), -1.0))
        with torch.no_grad():
            model.p.add_(10)
        torch.testing.assert_close(model.mean, torch.full((2,), -11.0))

        class Detached(Read):
            def __init__(self):
                super().__init__()
                with torch.no_grad():
                    self.mean = self.p.view(2)

        model = Detached()
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": torch.ones(2)})
        annotate.module("policy", model)
        annotate.input_tensors("policy", {"declared_mean": model.mean})
        annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        leapp.stop()
        _assert_plain(self, model.mean)
        model.mean.sum().backward()
        self.assertIsNone(model.p.grad)

        class Complex(nn.Module):
            def __init__(self):
                super().__init__()
                pair = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                self.c = nn.Parameter(torch.view_as_complex(pair))
                self.register_buffer("mean", torch.view_as_real(self.c))

            def forward(self, x):
                return x - self.mean.sum(0)

        model = Complex()
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": torch.ones(2)})
        annotate.module("policy", model)
        annotate.input_tensors("policy", {"declared_mean": model.mean})
        annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        leapp.stop()
        _assert_plain(self, model.mean)
        torch.testing.assert_close(model.mean, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        model.mean.sum().backward()
        self.assertIsNotNone(model.c.grad)

        class Memory(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("prev", torch.zeros(2))

            def forward(self, x):
                self.prev = x
                return x

        model = Memory()
        p = nn.Parameter(torch.ones(2))
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": p.view(2)})
        annotate.module("policy", model)
        annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        self.assertEqual(list(annotate.nodes["policy"].graph.nodes)[-1].op, "output")
        leapp.stop()
        _assert_plain(self, model.prev)
        model.prev.sum().backward()
        torch.testing.assert_close(p.grad, torch.ones(2))
        with torch.no_grad():
            p.add_(10)
        torch.testing.assert_close(model.prev, torch.full((2,), 11.0))

    def test_published_buffer_tensor_plain_after_the_session(self):
        """A buffer's tensor the node outputs, or moves into another buffer that
        becomes a state, is plain again after the session, so another module
        that holds the same tensor exports."""

        class Step(nn.Module):
            def __init__(self, h, move):
                super().__init__()
                self.register_buffer("h", h)
                self.register_buffer("prev", torch.zeros(1, 4))
                self.move = move

            def forward(self, x):
                old = self.h
                self.h = old + x
                if self.move:
                    self.prev = old
                    return self.h
                return self.h, old

        class Other(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

            def forward(self, x):
                return x * self.h

        for move in (False, True):
            with self.subTest(move=move):
                shared = torch.ones(1, 4)
                model, other = Step(shared, move).eval(), Other(shared).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    obs = annotate.input_tensors("policy", {"obs": torch.ones(1, 4)})
                    annotate.module("policy", model)
                    result = model(obs)
                    outputs = {"y": result} if move else dict(zip(("y", "old"), result))
                    annotate.output_tensors("policy", outputs, export_with="jit")
                finally:
                    leapp.stop()
                _assert_plain(self, shared)
                torch.export.export(other, (torch.ones(1, 4),))

    def test_untracked_view_of_a_tracked_buffer_exports(self):
        """A buffer left out of buffer_names that views a tracked one is a
        constant of the graph whose base is no longer traced, so a node that
        reads it next to an in-place op exports."""

        class Views(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.tensor([1.0, 2.0, 3.0, 4.0]))
                self.register_buffer("k", self.h[:2])

            def forward(self, x):
                y = x + self.h[:2]
                y.add_(self.k)
                return y

        for export_with in ("jit", "exported-program"):
            with self.subTest(export_with=export_with):
                model = Views().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    annotate.module("policy", model, buffer_names=["h"])
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with=export_with
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )

    def test_buffer_holding_an_output_with_gradient_read_by_a_later_node(self):
        """A buffer set to an input with autograd history that the node
        outputs keeps its memory while the session runs, so a later node that
        declares that output and reads the buffer connects to it."""

        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def encode(self, x):
                self.h = x
                return x

            def act(self, latent):
                return self.h * 3

        model = Shared().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors(
                "producer", {"x": torch.ones(2, requires_grad=True)}
            )
            annotate.module("producer", model)
            z = annotate.output_tensors(
                "producer", {"z": model.encode(x)}, export_with="jit"
            )
            latent = annotate.input_tensors("consumer", {"latent": z})
            annotate.module("consumer", model)
            annotate.output_tensors(
                "consumer", {"y": model.act(latent)}, export_with="jit"
            )
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {"producer": True, "consumer": True},
        )
        self.assertEqual(
            annotate.detected_pipeline["data_flow"],
            {"producer/z": ["consumer/latent"]},
        )
        _assert_plain(self, model.h)

    def test_buffer_with_an_input_on_its_bytes_negated(self):
        """An input declared on a buffer's bytes with lazy negation reads
        other values, so it is not the buffer's input: under another name the
        buffer is baked with its own values, and under the buffer's name, as an
        input or a state, it raises as a name clash."""

        class Offset(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

            def forward(self, x):
                return x + self.h

        for name in ("negated", "h", "state h"):
            with self.subTest(name=name):
                pair = torch.view_as_complex(torch.tensor([[0.0, 2.0], [0.0, 3.0]]))
                model = Offset(pair.imag).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    if name == "state h":
                        x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                        annotate.state_tensors("policy", {"h": pair.conj().imag})
                    else:
                        x = annotate.input_tensors(
                            "policy", {"x": torch.ones(2), name: pair.conj().imag}
                        )[0]
                    if name != "negated":
                        with self.assertRaisesRegex(ValueError, "other inputs"):
                            annotate.module("policy", model)
                        continue
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(annotate.detected_pipeline["inputs"]["policy"], ["x"])

    def test_buffer_reads_the_input_on_its_bytes_with_its_lazy_negation(self):
        """A read-only buffer reads the input declared on its bytes with its own
        lazy negation, also when another input declared first reads those
        bytes without it, so the export takes new values of that input."""

        class Offset(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

            def forward(self, x):
                return x + self.h

        pair = torch.view_as_complex(torch.tensor([[0.0, 2.0], [0.0, 3.0]]))
        model = Offset(pair.conj().imag).eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x, negated = annotate.input_tensors(
                "policy", {"x": pair.imag, "negated": pair.conj().imag}
            )
            annotate.module("policy", model)
            annotate.output_tensors(
                "policy", {"y": model(x) + negated}, export_with="jit"
            )
        finally:
            leapp.stop()
        node = annotate.nodes["policy"]
        values = {"x": torch.tensor([5.0, 7.0]), "negated": torch.tensor([-5.0, -7.0])}
        torch.testing.assert_close(
            node.m(*(values[d.name] for d in node.inputs)), torch.tensor([-5.0, -7.0])
        )

    def test_buffer_with_an_input_on_its_bytes_negated_changed_in_place(self):
        """A buffer that an input declared on its bytes with lazy negation
        covers raises when the forward changes it in place, also when the
        recorded write leaves the values."""

        class Write(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

            def forward(self, x):
                self.h.mul_(1.0)
                return x + self.h

        pair = torch.view_as_complex(torch.tensor([[0.0, 2.0], [0.0, 3.0]]))
        model = Write(pair.imag).eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors(
                "policy", {"x": torch.ones(2), "negated": pair.conj().imag}
            )[0]
            annotate.module("policy", model)
            y = model(x)
            with self.assertRaisesRegex(RuntimeError, "changed in place"):
                annotate.output_tensors("policy", {"y": y}, export_with="jit")
        finally:
            leapp.stop()
        _assert_plain(self, model.h)

    def test_buffer_on_the_memory_of_a_numpy_input_shares_its_writes(self):
        """A buffer made with torch.from_numpy() reads the input a NumPy array
        on the same memory declares, so a write through the buffer reaches a
        read through the array."""
        memory = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        model = nn.Module()
        model.register_buffer("h", torch.from_numpy(memory))
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            array = annotate.input_tensors("policy", {"array": memory})
            annotate.module("policy", model)
            model.h.add_(1)
            annotate.output_tensors("policy", {"y": array * 2}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )

    def test_buffer_written_through_a_declared_input_raises(self):
        """A buffer whose memory an input declared in another layout describes
        raises when the forward writes through that input, also when the
        write leaves the capture sample as it was, since the constant would
        lose it, and also when the input starts between the bytes of a strided
        buffer's elements, as torch.frombuffer() allows."""

        class Offset(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

            def forward(self, x):
                x.mul_(2)
                return x.flatten() + self.h

        memory = bytearray(24)
        h = torch.frombuffer(memory, dtype=torch.float32, count=4)[::2]
        cases = {
            "view": (torch.zeros(2), lambda h: h.view(1, 2)),
            "unaligned": (
                h,
                lambda _: torch.frombuffer(
                    memory, dtype=torch.float32, count=1, offset=7
                ),
            ),
        }
        for layout, (buffer, make_input) in cases.items():
            with self.subTest(layout=layout):
                model = Offset(buffer).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": make_input(model.h)})
                    annotate.module("policy", model)
                    y = model(x)
                    with self.assertRaisesRegex(
                        RuntimeError, "Buffer 'h'.*changed in place"
                    ):
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()

    def test_buffer_written_without_the_graph_then_reassigned_raises(self):
        """A buffer the forward writes through .data or out=, which the graph
        does not record, and then reassigns raises when the write moved its
        value, since the feedback would lose the write."""

        class Step(nn.Module):
            def __init__(self, write):
                super().__init__()
                self.register_buffer("h", torch.ones(2))
                self.write = write

            def forward(self, x):
                if self.write == "data":
                    self.h.data.add_(1)
                else:
                    torch.add(self.h, 1, out=self.h.data)
                self.h = self.h + x
                return self.h

        for write in ("data", "out"):
            with self.subTest(write=write):
                model = Step(write).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    annotate.module("policy", model)
                    y = model(x)
                    with self.assertRaisesRegex(
                        RuntimeError, "Buffer 'h'.*changed in place"
                    ):
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()

    def test_inputs_on_one_memory_read_negated_stay_apart(self):
        """Two inputs on the same bytes, one read with lazy negation, are two
        graph inputs, since they hold different values, so the second can also
        be an output, also when the graph writes another input in place."""
        pair = torch.view_as_complex(torch.tensor([[0.0, 2.0], [0.0, 3.0]]))
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            im, negated, z = annotate.input_tensors(
                "policy",
                {"im": pair.imag, "negated": pair.conj().imag, "z": torch.ones(2)},
            )
            z.add_(1)
            annotate.output_tensors(
                "policy", {"y": im + negated + z, "negated": negated}, export_with="jit"
            )
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )
        self.assertEqual(
            annotate.detected_pipeline["inputs"]["policy"], ["im", "negated", "z"]
        )

    def test_inputs_on_one_memory_read_negated_written_raises(self):
        """A write the graph records on bytes two inputs read with different
        lazy negation raises, since the export would not carry it from one to
        the other, also when they have different shapes."""
        pair = torch.view_as_complex(torch.tensor([[0.0, 2.0]]))
        for shape in ((1,), (1, 1)):
            with self.subTest(shape=shape):
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    im, negated = annotate.input_tensors(
                        "policy",
                        {"im": pair.imag, "negated": pair.conj().imag.view(shape)},
                    )
                    im.add_(1)
                    with self.assertRaisesRegex(ValueError, "lazy negation"):
                        annotate.output_tensors(
                            "policy", {"y": im * negated}, export_with="jit"
                        )
                finally:
                    leapp.stop()

    def test_buffer_read_only_by_discarded_work_is_not_copied(self):
        """A buffer the forward reads only for a result nothing uses is not
        copied for the export, since pruning drops that read."""

        class Discard(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.ones(4))

            def forward(self, x):
                self.discarded = self.h.sum()
                return x + 1

        model = Discard().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(4)})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        finally:
            leapp.stop()
        constant = annotate.nodes["policy"].tracer.root._buffer_h
        self.assertEqual(
            constant.untyped_storage().data_ptr(),
            model.h.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )

    def test_buffer_declared_as_an_input_before_module_rejected(self):
        """module() rejects a buffer whose tensor is already an input of the
        node, under another name or its own, or a state under another name,
        since the user's input would lose its reader. Left out of buffer_names,
        it is read through that input. The error names the state name to
        declare a nested buffer under, and after the session the rejected
        buffer is plain, also without another module() call."""

        class History(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("obs_hist", torch.zeros(3))
                self.register_buffer("scale", torch.full((3,), 2.0))

            def forward(self, x):
                return (x + self.obs_hist) * self.scale

        for name in ("obs", "obs_hist", "state"):
            with self.subTest(name=name):
                model = History().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    if name == "state":
                        x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                        annotate.state_tensors("policy", {name: model.obs_hist})
                    else:
                        x, _ = annotate.input_tensors(
                            "policy", {"x": torch.ones(3), name: model.obs_hist}
                        )
                    with self.assertRaisesRegex(ValueError, r"\['obs_hist'\]"):
                        annotate.module("policy", model)
                    annotate.module("policy", model, buffer_names=["scale"])
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="exported-program"
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(
                    annotate.detected_pipeline["inputs"]["policy"], ["x", name]
                )

        model = nn.Module()
        model.enc = History().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            annotate.input_tensors(
                "policy", {"x": torch.ones(3), "obs": model.enc.obs_hist}
            )
            with self.assertRaisesRegex(ValueError, r"state names \['enc_obs_hist'\]"):
                annotate.module("policy", model)
        finally:
            leapp.stop()
        _assert_plain(self, model.enc.obs_hist)

    def test_failed_node_keeps_a_declared_state_traced(self):
        """A tensor declared with state_tensors() before module() stays as the
        user made it when the node fails to compile or the session ends before
        output_tensors(), also one the forward only reads and that is never
        updated, while the buffer holds a plain copy, so the next session
        captures the model again with the same calls."""

        class Goal(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("goal", torch.ones(1, 4))

            def forward(self, obs):
                return obs + self.goal

        failures = ("unknown exporter", "stop() before output_tensors()")
        models = ((GRUModel, "h_state"), (Goal, "goal"))
        for failure, (model_type, name) in itertools.product(failures, models):
            with self.subTest(failure=failure, model=model_type.__name__):
                model = model_type().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                    h = annotate.state_tensors("policy", {name: getattr(model, name)})
                    annotate.module("policy", model)
                    action = model(obs)
                    if failure == "unknown exporter":
                        with self.assertRaisesRegex(Exception, "Unexpected backend"):
                            annotate.output_tensors(
                                "policy", {"action": action}, export_with="jitt"
                            )
                finally:
                    leapp.stop()
                self.assertIsInstance(h, TracedTensor)
                _assert_plain(self, getattr(model, name))
                self.assertNotIsInstance(getattr(model, name)._base, TracedTensor)
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                    annotate.state_tensors("policy", {name: getattr(model, name)})
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"action": model(obs)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )

    def test_buffer_declared_as_an_input_after_module_kept_for_a_retry(self):
        """A buffer's tensor declared with input_tensors() after module() stays
        traced when output_tensors() fails, so the repeated call compiles."""

        class Read(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.ones(2))

            def forward(self, x, h):
                return x + h

        model = Read().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(2)})
            annotate.module("policy", model)
            h = annotate.input_tensors("policy", {"h_input": model.h})
            y = model(x, h)
            with self.assertRaisesRegex(Exception, "Unexpected backend"):
                annotate.output_tensors("policy", {"y": y}, export_with="jitt")
            self.assertIsInstance(h, TracedTensor)
            annotate.output_tensors("policy", {"y": y}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )

    def test_buffer_written_with_its_own_value_baked(self):
        """A buffer the forward writes in place with the value it already
        holds, as some kernels do with running statistics, is baked like a
        read-only one, since only its version counter moved, also when it
        holds a NaN, and when a write the graph does not record moves it only
        by rounding, as InstanceNorm in eval mode does on torch 2.7, or by
        one step of a buffer's own lower precision, also when
        the graph then writes in place to a sum it went into, as
        ReLU(inplace=True) after InstanceNorm does."""

        class Same(nn.Module):
            def __init__(self, write):
                super().__init__()
                dtype = torch.bfloat16 if write == "bfloat16 rounding" else None
                self.register_buffer(
                    "h", torch.tensor([3.0, float("nan")], dtype=dtype)
                )
                self.write = write

            def forward(self, x):
                if self.write == "copy":
                    self.h.copy_(self.h)
                elif self.write == "bfloat16 rounding":
                    # One bfloat16 step.
                    self.h.data.mul_(1.004)
                else:
                    # About 8 units in the last place, as torch 2.7 InstanceNorm
                    # can drift with a large batch.
                    self.h.data.mul_(1 + 8 * torch.finfo(torch.float32).eps)
                y = x + torch.nan_to_num(self.h)
                if self.write == "rounding, then into the sum":
                    y.add_(1)
                return y

        for write in (
            "copy",
            "rounding",
            "bfloat16 rounding",
            "rounding, then into the sum",
        ):
            with self.subTest(write=write):
                model = Same(write).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(annotate.detected_pipeline["inputs"]["policy"], ["x"])

    def test_buffer_set_to_a_negative_view_keeps_its_values(self):
        """A buffer the forward sets to a lazily negated view holds the same
        values once it is plain, after the node compiles, also when the node
        outputs it, and after the session ends without it, and so does an
        untracked complex buffer set to a lazily conjugated view."""

        class Negative(nn.Module):
            def __init__(self, returned):
                super().__init__()
                self.register_buffer("h", torch.ones(2))
                self.returned = returned

            def forward(self, x):
                self.h = torch._neg_view(x + 1)
                return self.h if self.returned else self.h * 2

        for compiled, returned in ((False, False), (True, False), (True, True)):
            with self.subTest(compiled=compiled, returned=returned):
                model = Negative(returned).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    annotate.module("policy", model)
                    y = model(x)
                    if compiled:
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                _assert_plain(self, model.h)
                torch.testing.assert_close(model.h, torch.full((2,), -2.0))

        class Conjugate(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.zeros(2, dtype=torch.complex64))

            def forward(self, x):
                self.cache = torch.view_as_complex(x.data.view(2, 2)).conj()
                return x + 1

        model = Conjugate().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.arange(1.0, 5.0)})
            annotate.module("policy", model, buffer_names=[])
            annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        finally:
            leapp.stop()
        _assert_plain(self, model.cache)
        torch.testing.assert_close(model.cache, torch.tensor([1 - 2j, 3 - 4j]))

    def test_module_called_again_after_a_failed_node(self):
        """module() called again after output_tensors() failed raises and keeps
        the buffer tensors the failed call kept traced, so the advised repeated
        call compiles."""

        class Delay(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("last", torch.zeros(3))
                self.register_buffer("prev", torch.zeros(3))

            def forward(self, x):
                self.prev = self.last
                self.last = x * 10
                return x + self.prev

        model = Delay().eval()
        last = model.last
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            annotate.module("policy", model)
            y = model(x)
            with self.assertRaisesRegex(Exception, "Unexpected backend"):
                annotate.output_tensors("policy", {"y": y}, export_with="jitt")
            self.assertIsInstance(last, TracedTensor)
            with self.assertRaisesRegex(
                ValueError, "failed after its buffer checks, repeat the same call"
            ):
                annotate.module("policy", model)
            annotate.output_tensors("policy", {"y": y}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )
        _assert_plain(self, model.last)

    def test_state_declared_again_after_a_failed_node(self):
        """After output_tensors() fails, declaring a reassigned buffer's state
        again with its original tensor and repeating the call keeps the
        feedback, since the node still lists that state."""

        class Acc(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                self.h = self.h + x
                return self.h * 2

        model = Acc().eval()
        h0 = model.h
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.full((2,), 3.0)})
            annotate.module("policy", model)
            y = model(x)
            with self.assertRaisesRegex(Exception, "static output"):
                annotate.output_tensors(
                    "policy", {"y": y}, static_outputs={"s": y * 2}, export_with="jit"
                )
            h = annotate.state_tensors("policy", {"h": h0})
            self.assertIsInstance(h, TracedTensor)
            annotate.update_state("policy", {"h": h + x})
            annotate.output_tensors("policy", {"y": y}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )
        self.assertEqual(
            annotate.detected_pipeline["feedback_flow"], {"policy/h": ["policy/h"]}
        )
        _assert_plain(self, h0)

    def test_buffer_holding_a_value_the_node_computed_raises(self):
        """A buffer set to a value the node computed before module() and only
        read by the forward raises, by assignment or in place, as a reset like
        model.h.mul_(1 - done), also when the forward reads it through an in-place operation on another tensor, or
        on a view of one made with keyword arguments, also of an operator,
        and when the user feeds it back with update_state(), since a constant
        or the start of the feedback would
        freeze that value. One the forward reads only
        for a result no output uses does not, nor a constant made with
        annotate.register_buffer()."""

        class Offset(nn.Module):
            def __init__(self, read_in_place=False):
                super().__init__()
                self.register_buffer("h", torch.zeros(3))
                self.read_in_place = read_in_place

            def forward(self, x):
                if self.read_in_place == "keyword view":
                    y = x * 2
                    torch.reshape(input=y, shape=(1, 3)).add_(self.h)
                    return y
                if self.read_in_place == "operator keyword view":
                    y = x * 2
                    torch.ops.aten.view.default(self=y, size=[1, 3]).add_(self.h)
                    return y
                if self.read_in_place:
                    y = x * 2
                    y += self.h
                    return y
                return x + self.h

        class Unread(Offset):
            def forward(self, x):
                self.discarded = self.h + 1
                return x * 2

        with self.subTest(read=False):
            model = Unread().eval()
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                model.h = x * 2
                annotate.module("policy", model)
                annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
            finally:
                leapp.stop()
            self.assertEqual(
                leapp.compile_graph(visualize=False, validate=True), {"policy": True}
            )

        with self.subTest(how="register_buffer"):
            model = Offset().eval()
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                model.h = annotate.register_buffer("policy", {"h": torch.ones(3)})
                annotate.module("policy", model)
                annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
            finally:
                leapp.stop()
            self.assertEqual(
                leapp.compile_graph(visualize=False, validate=True), {"policy": True}
            )

        for how in (
            "assign",
            "copy_",
            "mul_",
            "read in place",
            "keyword view",
            "operator keyword view",
            "update_state",
        ):
            with self.subTest(how=how):
                read = how if how.endswith("view") else how == "read in place"
                model = Offset(read_in_place=read).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                    if how == "copy_":
                        model.h.copy_(x * 2)
                    elif how == "mul_":
                        model.h.mul_(x)
                    else:
                        model.h = x * 2
                    annotate.module("policy", model)
                    y = model(x)
                    if how == "update_state":
                        annotate.update_state("policy", {"h": y * 1})
                    with self.assertRaisesRegex(RuntimeError, "Buffer 'h'.*computed"):
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()

    def test_buffer_declared_as_a_state_and_not_updated_stays_an_input(self):
        """A buffer declared with state_tensors() under its own name and never
        updated stays a regular input when module() also tracks it, also when
        the forward changes it in place, since it is not baked."""

        class Goal(nn.Module):
            def __init__(self, in_place):
                super().__init__()
                self.register_buffer("goal", torch.ones(3))
                self.in_place = in_place

            def forward(self, x):
                if self.in_place:
                    self.goal.add_(x)
                return x + self.goal

        for in_place in (False, True):
            with self.subTest(in_place=in_place):
                model = Goal(in_place).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                    annotate.state_tensors("policy", {"goal": model.goal})
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(
                    annotate.detected_pipeline["inputs"]["policy"], ["x", "goal"]
                )

    def test_output_tensors_repeated_after_a_capture_error_with_in_place_op(self):
        """A failed output_tensors() that stops at a capture error in a graph
        with an in-place op changed nothing, so it can be repeated with an
        output that does not depend on the failure."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": np.zeros(3, dtype=np.float32)})
            x += 1
            failed = np.spacing(x)
            with self.assertRaisesRegex(NotImplementedError, "failed capture"):
                annotate.output_tensors("policy", {"y": failed}, export_with="jit")
            annotate.output_tensors("policy", {"y": x - 1}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )

    def test_in_place_op_on_a_complex_numpy_array(self):
        """An in-place op on a complex NumPy array captures, although the
        write checks cannot describe its memory."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": np.arange(8, dtype=np.float32)})
            spectrum = np.fft.rfft(x)
            spectrum *= 2
            y = np.fft.irfft(spectrum, n=8).astype(np.float32)
            annotate.output_tensors("policy", {"y": y}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )

    def test_output_tensors_repeated_with_other_outputs_raises(self):
        """A repeated output_tensors() call after a failure raises when its
        outputs differ from the failed call's, since the graph keeps the
        outputs that call built, also when that call failed before building
        the graph, since the buffers were judged by its outputs, also with
        other tensors under the same names, also in a node without
        annotate.module(), under another name, with another state update or
        another static output value, and captures with the same ones, also
        with an equal static output."""

        class Read(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.full((3,), 0.5))

            def forward(self, x):
                return x - self.mean

        model = Read().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            annotate.module("policy", model)
            y = model(x)
            with self.assertRaisesRegex(Exception, "Unexpected backend"):
                annotate.output_tensors("policy", {"y": y}, export_with="jitt")
            with self.assertRaisesRegex(RuntimeError, "only with the same outputs"):
                annotate.output_tensors("policy", {"y": y + 1}, export_with="jit")
        finally:
            leapp.stop()

        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            annotate.module("policy", model)
            y = model(x)
            with self.assertRaisesRegex(Exception, "static output"):
                annotate.output_tensors(
                    "policy", {"y": x * 2}, static_outputs={"s": y}, export_with="jit"
                )
            with self.assertRaisesRegex(RuntimeError, "only with the same outputs"):
                annotate.output_tensors("policy", {"y": x * 2}, export_with="jit")
            with self.assertRaisesRegex(RuntimeError, "only with the same outputs"):
                annotate.output_tensors("policy", {"z": y}, export_with="jit")
        finally:
            leapp.stop()

        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            h = annotate.state_tensors("policy", {"h": torch.zeros(3)})
            annotate.update_state("policy", {"h": h + x})
            y = x * 2
            with self.assertRaisesRegex(Exception, "Unexpected backend"):
                annotate.output_tensors("policy", {"y": y}, export_with="jitt")
            for outputs in ({"y": x * 3}, {"z": y}):
                with self.assertRaisesRegex(RuntimeError, "only with the same outputs"):
                    annotate.output_tensors("policy", outputs, export_with="jit")
            annotate.update_state("policy", {"h": h + 2 * x})
            with self.assertRaisesRegex(RuntimeError, "only with the same outputs"):
                annotate.output_tensors("policy", {"y": y}, export_with="jit")
        finally:
            leapp.stop()

        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            y, static = x * 2, {"s": torch.ones(3)}
            with self.assertRaisesRegex(Exception, "Unexpected backend"):
                annotate.output_tensors(
                    "policy", {"y": y}, static_outputs=static, export_with="jitt"
                )
            with self.assertRaisesRegex(RuntimeError, "only with the same outputs"):
                annotate.output_tensors(
                    "policy",
                    {"y": y},
                    static_outputs={"s": torch.zeros(3)},
                    export_with="jit",
                )
            annotate.output_tensors(
                "policy",
                {"y": y},
                static_outputs={"s": torch.ones(3)},
                export_with="jit",
            )
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )

    def test_buffer_reset_with_a_traced_value_around_module(self):
        """A buffer the forward reassigns and that was reset before module()
        with a value the node computed raises, since the feedback would start
        from that value every step, while a reset after module() is part of
        the graph, so the buffer is feedback."""

        class Acc(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.ones(3))

            def forward(self, x):
                self.h = self.h + x
                return self.h * 2

        for when in ("before", "after"):
            with self.subTest(when=when):
                model = Acc().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                    done = annotate.input_tensors("policy", {"done": torch.ones(3)})
                    if when == "before":
                        model.h = model.h * (1 - done)
                    annotate.module("policy", model)
                    if when == "after":
                        model.h = model.h * (1 - done)
                    y = model(x)
                    if when == "before":
                        with self.assertRaisesRegex(
                            RuntimeError, "Buffer 'h'.*computed"
                        ):
                            annotate.output_tensors(
                                "policy", {"y": y}, export_with="jit"
                            )
                    else:
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                _assert_plain(self, model.h)
                if when == "after":
                    self.assertEqual(
                        leapp.compile_graph(visualize=False, validate=True),
                        {"policy": True},
                    )
                    self.assertEqual(
                        annotate.detected_pipeline["feedback_flow"],
                        {"policy/h": ["policy/h"]},
                    )

    def test_buffer_registration_kept_after_tracing(self):
        """A Parameter registered as a buffer is still a buffer after capture."""

        class Offset(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("b", nn.Parameter(torch.ones(2)))

            def forward(self, x):
                return x + self.b

        model = Offset().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": torch.ones(2)})
        annotate.module("policy", model)
        annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        leapp.stop()
        self.assertEqual(list(dict(model.named_buffers())), ["b"])
        self.assertEqual(list(model.parameters()), [])

    def test_buffers_restored_when_output_tensors_fails(self):
        """An output_tensors() that fails while compiling the node leaves the
        model's buffers plain. One that fails while collecting them keeps them
        tracked until leapp.stop() or the next leapp.start() makes them plain."""

        class Accumulator(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(3))
                self.register_buffer("k", torch.ones(3))

            def forward(self, x):
                self.h = self.h + x
                return self.h * self.k

        class Grow(Accumulator):
            def forward(self, x):
                self.h = torch.cat([self.h, x])
                return x * self.k

        # (model, static outputs, the error output_tensors() raises, whether the
        # buffers are still tracked after it)
        cases = [
            # Rejected after the node's buffers were collected.
            (Accumulator, lambda y: {"s": y * 2}, "static output 's'", False),
            # Rejected while they are collected, since the state changes shape.
            (Grow, lambda y: None, "expected torch.Size", True),
        ]
        for (model_type, static_outputs, error, tracked), end in itertools.product(
            cases, ("stop", "start")
        ):
            with self.subTest(model=model_type.__name__, end=end):
                model = model_type().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                annotate.module("policy", model)
                y = model(x)
                with self.assertRaisesRegex(Exception, error):
                    annotate.output_tensors(
                        "policy",
                        {"y": y},
                        static_outputs=static_outputs(y),
                        export_with="jit",
                    )
                self.assertEqual(isinstance(model.k, TracedTensor), tracked)
                if end == "start":
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    _assert_plain(self, model.h)
                    _assert_plain(self, model.k)
                leapp.stop()
                _assert_plain(self, model.h)
                _assert_plain(self, model.k)

    def test_output_tensors_retried_after_a_failure(self):
        """An output_tensors() that fails before or while compiling the node can
        be called again for the same node, and the buffer still becomes
        feedback state, also with an output that views a read-only buffer,
        while a repeated call after the failed one rewrote the graph's in-place
        operations raises a clear error."""

        class Accumulator(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                self.h = self.h + x
                return self.h * 2

        class Delay(Accumulator):
            def forward(self, x):
                previous = self.h
                self.h = x * 2
                return previous

        class ToDevice(Accumulator):
            def forward(self, x):
                self.h = self.h.to(x.device)
                return x - self.h

        # (model, outputs of the failing call, its exporter, its error)
        cases = [
            (Accumulator, lambda y: {"y": torch.ones(2)}, "jit", "non-traced"),
            (Accumulator, lambda y: {"y": y}, "jitt", "Unexpected backend"),
            (Delay, lambda y: {"y": y}, "jitt", "Unexpected backend"),
            (ToDevice, lambda y: {"y": y}, "jitt", "Unexpected backend"),
        ]
        for model_type, outputs, export_with, error in cases:
            with self.subTest(model=model_type.__name__, export_with=export_with):
                model = model_type().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                annotate.module("policy", model)
                y = model(x)
                with self.assertRaisesRegex(Exception, error):
                    annotate.output_tensors(
                        "policy", outputs(y), export_with=export_with
                    )
                annotate.output_tensors("policy", {"y": y}, export_with="jit")
                leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(
                    annotate.detected_pipeline["feedback_flow"],
                    {"policy/h": ["policy/h"]},
                )
                _assert_plain(self, model.h)

        class InPlace(Accumulator):
            def forward(self, x):
                self.h = nn.functional.relu(self.h + x, inplace=True)
                return self.h * 2

        model = InPlace().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(2)})
            annotate.module("policy", model)
            y = model(x)
            with self.assertRaisesRegex(Exception, "Unexpected backend"):
                annotate.output_tensors("policy", {"y": y}, export_with="jitt")
            with self.assertRaisesRegex(RuntimeError, "cannot be repeated"):
                annotate.output_tensors("policy", {"y": y}, export_with="jit")
        finally:
            leapp.stop()
        _assert_plain(self, model.h)

        class ViewOut(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.arange(2.0))

            def forward(self, x):
                return x + 1, self.mean.unsqueeze(0).flatten()

        model = ViewOut().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(2)})
            annotate.module("policy", model)
            y, v = model(x)
            with self.assertRaisesRegex(Exception, "Unexpected backend"):
                annotate.output_tensors("policy", {"y": y, "v": v}, export_with="jitt")
            annotate.output_tensors("policy", {"y": y, "v": v}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )
        _assert_plain(self, model.mean)

    def test_output_tensors_retried_after_a_state_shape_error(self):
        """After a reassigned buffer got another shape, a repeated
        output_tensors() with the outputs of a re-run forward raises, since
        the buffers hold the failed forward's values, and the same call
        raises again. The buffers are plain after leapp.stop()."""

        class Pair(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))
                self.register_buffer("k", torch.zeros(2))

            def forward(self, x, grow):
                self.h = self.h + x
                self.k = self.k.repeat(2) if grow else self.k + x
                return self.h + self.k[:2]

        model = Pair().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(2)})
            annotate.module("policy", model)
            k = model.k
            y = model(x, grow=True)
            with self.assertRaisesRegex(Exception, "update_state for 'k' has shape"):
                annotate.output_tensors("policy", {"y": y}, export_with="jit")
            with self.assertRaisesRegex(Exception, "update_state for 'k' has shape"):
                annotate.output_tensors("policy", {"y": y}, export_with="jit")
            model.k = k
            with self.assertRaisesRegex(RuntimeError, "only with the same outputs"):
                annotate.output_tensors(
                    "policy", {"y": model(x, grow=False)}, export_with="jit"
                )
        finally:
            leapp.stop()
        for buf in model.buffers():
            _assert_plain(self, buf)

    def test_model_captured_again_after_a_failed_node(self):
        """After leapp.stop(), a node that did not compile keeps none of the
        buffers module() tracked traced, also one passed as an output, so the
        next capture with the same model works, and a repeated call while
        another node captures the same model leaves that node's buffer
        traced. Another node reading a buffer that holds the failed node's
        state raises, as it does once that node compiled."""

        class Read(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.full((3,), 0.5))
                self.register_buffer("h", torch.zeros(1))

            def forward(self, x, grow=False):
                if grow:
                    self.h = torch.cat([self.h, x[:1]])
                return x - self.mean

        def capture(failure=None):
            leapp.start(name=self.TEST_GRAPH_NAME)
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            annotate.module("policy", model)
            y = model(x, grow=failure == "state changes shape")
            outputs = {"y": y}
            if failure == "buffer's tensor as an output":
                outputs["h"] = model.h
                model.h = y[:1]
            if failure != "stop() before output_tensors()":
                annotate.output_tensors(
                    "policy",
                    outputs,
                    export_with="jit"
                    if failure in (None, "state changes shape")
                    else "jitt",
                )

        errors = {
            "unknown exporter": (Exception, "Unexpected backend"),
            "state changes shape": (Exception, "must match input state shape"),
            "buffer's tensor as an output": (Exception, "Unexpected backend"),
        }
        for failure in (*errors, "stop() before output_tensors()"):
            with self.subTest(failure=failure):
                model = Read().eval()
                mean, h = model.mean, model.h
                try:
                    if failure == "stop() before output_tensors()":
                        capture(failure)
                    else:
                        with self.assertRaisesRegex(*errors[failure]):
                            capture(failure)
                finally:
                    leapp.stop()
                self.assertTrue(model.mean is mean)
                _assert_plain(self, model.mean)
                _assert_plain(self, model.h)
                _assert_plain(self, h)
                capture()
                leapp.stop()

        with self.subTest(failure="unknown exporter, then another node"):
            model = Read().eval()
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                annotate.module("policy", model)
                y = model(x)
                with self.assertRaisesRegex(Exception, "Unexpected backend"):
                    annotate.output_tensors("policy", {"y": y}, export_with="jitt")
                x = annotate.input_tensors("other", {"x": torch.ones(3)})
                annotate.module("other", model)
                annotate.output_tensors("other", {"y": model(x)}, export_with="jit")
            finally:
                leapp.stop()
            _assert_plain(self, model.mean)

        class Accumulator(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(3))

            def forward(self, x):
                self.h = self.h + x
                return self.h * 2

        with self.subTest(failure="unknown exporter, repeated while another is open"):
            model = Accumulator().eval()
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                annotate.module("policy", model)
                y = model(x)
                with self.assertRaisesRegex(Exception, "Unexpected backend"):
                    annotate.output_tensors("policy", {"y": y}, export_with="jitt")
                x = annotate.input_tensors("other", {"x": torch.ones(3)})
                annotate.module("other", model)
                y_other = model(x)
                annotate.output_tensors("policy", {"y": y}, export_with="jit")
                with self.assertRaisesRegex(RuntimeError, "holds a value the graph"):
                    annotate.output_tensors("other", {"y": y_other}, export_with="jit")
            finally:
                leapp.stop()

    def test_buffer_holding_another_nodes_output(self):
        """A buffer that holds an earlier node's output and the forward reads
        raises, also when the forward then reassigns it, since the export would
        freeze a value that changes every step, or start the feedback from it,
        and after the session the output stays traced with its output port,
        while the buffer holds a plain tensor on its memory. Set to that output again in
        a later session without that node, it belongs to no node, so it is
        baked, also by a second node, and plain once the node compiles."""

        class Offset(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("b", b)

            def forward(self, x):
                return x + self.b

        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            p_in = annotate.input_tensors("producer", {"x": torch.ones(3)})
            out = annotate.output_tensors(
                "producer", {"o": p_in * 2}, export_with="jit"
            )
            port = out.output_port
            model = Offset(out).eval()
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            annotate.module("policy", model)
            y = model(x)
            with self.assertRaisesRegex(RuntimeError, "Buffer 'b'.*another node"):
                annotate.output_tensors("policy", {"y": y}, export_with="jit")
        finally:
            leapp.stop()
        self.assertIsInstance(out, TracedTensor)
        self.assertIs(out.output_port, port)
        _assert_plain(self, model.b)
        torch.testing.assert_close(model.b, out.tensor)
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            model.b = out
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            annotate.module("policy", model)
            y = annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
            _assert_plain(self, model.b)
            x = annotate.input_tensors("again", {"x": y})
            annotate.module("again", model)
            annotate.output_tensors("again", {"y": model(x)}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {"policy": True, "again": True},
        )
        self.assertEqual(annotate.detected_pipeline["inputs"]["policy"], ["x"])
        self.assertIsInstance(out, TracedTensor)

    def test_buffer_reading_another_nodes_output_as_an_input_reassigned_raises(
        self,
    ):
        """A recurrent model in two nodes, the second declaring the first one's
        output that the buffer holds as its input, raises when the second
        forward reassigns the buffer, also when update_state() updates it,
        since the feedback would replace that input, or changes it in place,
        since that would change the first node's output in eager only. Without
        that input, a second forward that sets the buffer without reading it
        captures. The input may also take the buffer's name."""

        class Recurrent(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                self.h = torch.tanh(x + self.h)
                return self.h

        class Overwrite(Recurrent):
            def forward(self, x):
                if self.overwrite == "set":
                    self.h = x * 2
                    return x + 1
                if self.overwrite == "add":
                    self.h.add_(1)
                    return x + self.h
                return super().forward(x)

        cases = (
            "reassigned",
            "same name",
            "updated",
            "overwritten",
            "changed in place",
        )
        for case in cases:
            with self.subTest(case=case):
                model = Overwrite().eval()
                model.overwrite = None
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("a", {"obs": torch.ones(2)})
                    annotate.module("a", model)
                    y = annotate.output_tensors("a", {"y": model(x)}, export_with="jit")
                    if case == "overwritten":
                        z = annotate.input_tensors("b", {"x": torch.ones(2)})
                    elif case == "same name":
                        z = annotate.input_tensors("b", {"h": y})
                    else:
                        z = annotate.input_tensors("b", {"y": y})
                    annotate.module("b", model)
                    model.overwrite = {
                        "overwritten": "set",
                        "changed in place": "add",
                    }.get(case)
                    w = model(z)
                    if case == "updated":
                        annotate.update_state("b", {"h": w * 3})
                    if case == "overwritten":
                        annotate.output_tensors("b", {"w": w}, export_with="jit")
                        continue
                    error = (
                        "changed in place"
                        if case == "changed in place"
                        else "input 'h'"
                        if case == "same name"
                        else "input 'y'"
                    )
                    with self.assertRaisesRegex(RuntimeError, f"Buffer 'h'.*{error}"):
                        annotate.output_tensors("b", {"w": w}, export_with="jit")
                finally:
                    leapp.stop()
                _assert_plain(self, model.h)

    def test_buffer_set_to_an_alias_of_its_own_declared_state(self):
        """A buffer set to a view, a detached alias, or .data of the state the
        user declared under its name before module() is that state, so the
        forward's reassignment is its feedback."""

        class Step(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.ones(2))

            def forward(self, x):
                self.h = x + self.h
                return self.h * 2

        makes = {
            "view": lambda h, model: h.view_as(model.h),
            "detach": lambda h, model: h.detach(),
            "data": lambda h, model: h.data,
        }
        for kind, make in makes.items():
            with self.subTest(kind=kind):
                model = Step().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    h = annotate.state_tensors("policy", {"h": model.h})
                    model.h = make(h, model)
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(
                    annotate.detected_pipeline["feedback_flow"],
                    {"policy/h": ["policy/h"]},
                )

    def test_model_outputting_its_state_captured_again_in_a_later_session(self):
        """A model whose forward outputs its reassigned buffer captures again in
        a later session, since what a session records about the values nodes
        set into buffers ends with it."""

        class Accumulate(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                self.h = self.h + x
                return self.h

        model = Accumulate().eval()
        for _ in range(2):
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                annotate.module("policy", model)
                annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
            finally:
                leapp.stop()
            self.assertEqual(
                leapp.compile_graph(visualize=False, validate=True),
                {"policy": True},
            )

    def test_buffer_on_other_bytes_of_an_earlier_output_baked(self):
        """A buffer on the same memory as a value an earlier node set into a
        buffer and output, but on other bytes, also between its elements or in
        other columns, is a constant, so a later node that reads it bakes it.
        One in another dtype is compared byte by byte, so it is a constant
        between the output's elements and the output's value within them, and
        a contiguous view of the output in another layout is its value."""

        class Encode(nn.Module):
            def __init__(self, split):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))
                self.split = split

            def forward(self, x):
                if self.split == "interleaved":
                    whole = torch.stack([x + 1, torch.full_like(x, 3)], dim=1)
                    self.h, self.tail = whole.flatten()[0::2], whole.flatten()[1::2]
                elif self.split == "columns":
                    whole = torch.stack([x + 1, x * 0 + 3, x * 0 + 4], dim=1)
                    self.h, self.tail = whole[:, 0], whole[:, 1:]
                elif self.split == "blocks":
                    whole = torch.stack([x + 1, x * 0 + 3, x * 0 + 4, x * 0 + 5], dim=1)
                    self.h, self.tail = whole[:, 0], whole[:, 1:]
                else:
                    whole = torch.cat([x + 1, torch.full_like(x, 3)])
                    self.h, self.tail = whole[:2], whole[2:]
                return self.h

        class Read(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("b", b)

            def forward(self, x):
                return x + self.b.sum()

        for split in ("halves", "interleaved", "columns", "blocks"):
            with self.subTest(split=split):
                encode = Encode(split).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("enc", {"x": torch.ones(2)})
                    annotate.module("enc", encode)
                    annotate.output_tensors("enc", {"y": encode(x)}, export_with="jit")
                    read = Read(encode.tail.tensor).eval()
                    x = annotate.input_tensors("read", {"x": torch.ones(2)})
                    annotate.module("read", read)
                    annotate.output_tensors("read", {"y": read(x)}, export_with="jit")
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"enc": True, "read": True},
                )

        class Halves(nn.Module):
            def __init__(self, first):
                super().__init__()
                self.register_buffer("h", torch.zeros(4))
                self.first = first

            def forward(self, x):
                self.whole = torch.cat([x + 1, x + 2, x + 3, x + 4])
                self.h = self.whole[self.first :: 2]
                return self.h

        # A contiguous view in another layout covers the output's bytes too.
        for make in (lambda h: h.view(1, 2), lambda h: h.flatten()[1:]):
            encode = Encode("halves").eval()
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                x = annotate.input_tensors("enc", {"x": torch.ones(2)})
                annotate.module("enc", encode)
                annotate.output_tensors("enc", {"y": encode(x)}, export_with="jit")
                read = Read(make(encode.h)).eval()
                x = annotate.input_tensors("read", {"x": torch.ones(2)})
                annotate.module("read", read)
                y = read(x)
                with self.assertRaisesRegex(RuntimeError, "holds a value"):
                    annotate.output_tensors("read", {"y": y}, export_with="jit")
            finally:
                leapp.stop()

        # The int16 view covers the second half of the even float32 elements.
        for first, held in ((1, False), (0, True)):
            with self.subTest(first=first):
                encode = Halves(first).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("enc", {"x": torch.ones(2)})
                    annotate.module("enc", encode)
                    annotate.output_tensors("enc", {"y": encode(x)}, export_with="jit")
                    read = Read(encode.whole.tensor.view(torch.int16)[1::4]).eval()
                    x = annotate.input_tensors("read", {"x": torch.ones(2)})
                    annotate.module("read", read)
                    y = read(x)
                    if held:
                        with self.assertRaisesRegex(RuntimeError, "holds a value"):
                            annotate.output_tensors("read", {"y": y}, export_with="jit")
                        continue
                    annotate.output_tensors("read", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"enc": True, "read": True},
                )

    def test_buffer_on_an_earlier_output_declared_as_its_own_input(self):
        """A buffer on the memory of a value an earlier node set into a buffer,
        which a later node declares as an input under the buffer's name before
        module(), stays that input, since the forward reads the input."""

        class Produce(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                self.h = x * 2
                return x + 1

        class Goal(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

            def forward(self, x):
                return x + self.h

        produce = Produce().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("enc", {"x": torch.ones(2)})
            annotate.module("enc", produce)
            annotate.output_tensors("enc", {"y": produce(x)}, export_with="jit")
            goal = Goal(produce.h).eval()
            x = annotate.input_tensors(
                "read", {"x": torch.ones(2), "h": goal.h.view(2)}
            )[0]
            annotate.module("read", goal)
            annotate.output_tensors("read", {"y": goal(x)}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {"enc": True, "read": True},
        )
        self.assertEqual(annotate.detected_pipeline["inputs"]["read"], ["x", "h"])

    def test_buffer_declared_as_a_state_holding_another_nodes_output(self):
        """A buffer holding an earlier node's output that the user declared
        with state_tensors() before module() stays that input, so reading it
        raises nothing and the export validates."""

        class Offset(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("b", b)

            def forward(self, x):
                return x + self.b

        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            p_in = annotate.input_tensors("producer", {"x": torch.ones(3)})
            out = annotate.output_tensors(
                "producer", {"o": p_in * 2}, export_with="jit"
            )
            model = Offset(out).eval()
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            annotate.state_tensors("policy", {"b": model.b})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {"producer": True, "policy": True},
        )

    def test_buffer_with_a_renamed_graph_input_baked(self):
        """A read-only buffer whose graph input FX renames, as for a buffer
        named max or one of nn.Sequential's numbered submodules, is baked."""

        class Clip(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("max", torch.ones(2))

            def forward(self, x):
                return torch.minimum(x, self.max)

        models = (Clip, lambda: nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)))
        for make, export_with in itertools.product(models, ("jit", "exported-program")):
            with self.subTest(model=make().__class__.__name__, export_with=export_with):
                model = make().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.randn(1, 2)})
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with=export_with
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(annotate.detected_pipeline["inputs"]["policy"], ["x"])

    def test_buffer_written_by_another_node_keeps_its_output(self):
        """A buffer that another node writes into and outputs while this node is
        open stays that node's output after this node compiles, so a later node
        still connects to it."""

        class Holder(nn.Module):
            def __init__(self, tensor):
                super().__init__()
                self.register_buffer("shared", tensor)

            def forward(self, x):
                return x * 2 + self.shared

        original = torch.zeros(3)
        model = Holder(original)
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("reader", {"x": original.view(3)})
            annotate.module("reader", model)
            y = model(x)
            z = annotate.input_tensors("producer", {"z": torch.ones(3)})
            original.copy_(z * 3)
            annotate.output_tensors("producer", {"out": original}, export_with="jit")
            annotate.output_tensors("reader", {"y": y}, export_with="jit")
            q = annotate.input_tensors("consumer", {"q": original})
            annotate.output_tensors("consumer", {"r": q * 2}, export_with="jit")
        finally:
            leapp.stop()
        leapp.compile_graph(visualize=False, validate=True)
        self.assertIn(
            "consumer/q",
            annotate.detected_pipeline["data_flow"].get("producer/out", []),
        )

    def test_buffer_holding_another_nodes_unpublished_value_baked(self):
        """A buffer that holds a value another node computed but did not output
        is baked, since there is no output port to connect to."""

        class Offset(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("b", b)

            def forward(self, x):
                return x + self.b

        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            p_in = annotate.input_tensors("producer", {"x": torch.ones(3)})
            hidden = p_in * 2
            annotate.output_tensors("producer", {"o": hidden + 1}, export_with="jit")
            model = Offset(hidden).eval()
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {"producer": True, "policy": True},
        )
        self.assertEqual(annotate.detected_pipeline["inputs"]["policy"], ["x"])

    def test_buffer_holding_another_nodes_output_changed_in_place_raises(self):
        """A buffer that holds an earlier node's output and the forward changes
        in place raises, also when the forward then reassigns it or does not
        read it, or the user declared it as a state, since the export would not
        change that output. One left from
        an earlier session belongs to no node, so a write and then a
        reassignment is accepted, and a write it does not read raises, as for a
        plain buffer."""

        class Bump(nn.Module):
            def __init__(self, b, reassign, read=True):
                super().__init__()
                self.register_buffer("b", b)
                self.reassign = reassign
                self.read = read

            def forward(self, x):
                old = self.b.add_(1)
                if self.reassign:
                    self.b = x * 3
                return x + old if self.read else x * 2

        cases = (
            (False, True, False),
            (True, True, False),
            (False, False, False),
            (False, True, True),
        )
        for reassign, read, declared in cases:
            with self.subTest(reassign=reassign, read=read, declared=declared):
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    p_in = annotate.input_tensors("producer", {"x": torch.ones(3)})
                    out = annotate.output_tensors(
                        "producer", {"o": p_in * 2}, export_with="jit"
                    )
                    model = Bump(out, reassign, read).eval()
                    x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                    if declared:
                        annotate.state_tensors("policy", {"b": model.b})
                    annotate.module("policy", model)
                    y = model(x)
                    with self.assertRaisesRegex(
                        RuntimeError, "Buffer 'b'.*changed in place"
                    ):
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()

        with self.subTest(session="earlier"):
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                p_in = annotate.input_tensors("producer", {"x": torch.ones(3)})
                out = annotate.output_tensors(
                    "producer", {"o": p_in * 2}, export_with="jit"
                )
            finally:
                leapp.stop()
            for reassign, read in ((True, True), (False, False)):
                model = Bump(out, reassign, read).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                    annotate.module("policy", model)
                    y = model(x)
                    if reassign:
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                        continue
                    with self.assertRaisesRegex(RuntimeError, "changed in place"):
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()

    def test_buffer_on_a_published_state_of_another_node_raises(self):
        """A buffer that views a state another node declared, changed in place
        and published raises, since that node's output changes every step, also
        through another storage on the same bytes."""

        class Holder(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

        for wrap in ("view", "dlpack"):
            with self.subTest(wrap=wrap):
                memory = torch.zeros(2)
                alias = memory.view(2) if wrap == "view" else torch.from_dlpack(memory)
                producer, consumer = Holder(memory), Holder(alias)
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("producer", {"x": torch.ones(2)})
                    annotate.state_tensors("producer", {"h": producer.h})
                    annotate.module("producer", producer)
                    producer.h.copy_(x)
                    annotate.update_state("producer", {"h": producer.h})
                    annotate.output_tensors(
                        "producer", {"y": producer.h}, export_with="jit"
                    )
                    y = annotate.input_tensors("consumer", {"x": torch.ones(2)})
                    annotate.module("consumer", consumer)
                    with self.assertRaisesRegex(RuntimeError, "another node's output"):
                        annotate.output_tensors(
                            "consumer", {"y": y + consumer.h}, export_with="jit"
                        )
                finally:
                    leapp.stop()

    def test_later_node_reads_a_buffer_another_node_updated_raises(self):
        """A later node raises when it reads a buffer that changes with another
        node's state every step: a view of that node's state, which its forward
        changes in place, and a buffer its forward reassigned while
        update_state() feeds back another value."""

        class Acc(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(3))

            def forward(self, x):
                if self.in_place:
                    y = x + self.h
                    self.h.mul_(0.5).add_(x)
                    return y
                self.h = self.h + x
                return self.h * 2

        class Reader(nn.Module):
            def __init__(self, acc, h):
                super().__init__()
                self.acc = acc
                self.register_buffer("v", h)

            def forward(self, z):
                return z * (self.v if self.acc.in_place else self.acc.h)

        for in_place in (True, False):
            with self.subTest(in_place=in_place):
                acc = Acc().eval()
                acc.in_place = in_place
                reader = Reader(acc, acc.h.view(3)).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("a", {"x": torch.ones(3)})
                    h = annotate.state_tensors("a", {"h": acc.h})
                    annotate.module("a", acc)
                    y = acc(x)
                    annotate.update_state("a", {"h": acc.h if in_place else h + x})
                    annotate.output_tensors("a", {"y": y}, export_with="jit")
                    z = annotate.input_tensors("b", {"z": torch.ones(3)})
                    annotate.module("b", reader)
                    with self.assertRaisesRegex(RuntimeError, "another node's output"):
                        annotate.output_tensors(
                            "b", {"w": reader(z)}, export_with="jit"
                        )
                finally:
                    leapp.stop()

    def test_buffers_with_the_same_state_name_rejected(self):
        """module() rejects buffers whose names become the same state name, since
        they would share one traced tensor, and leaves them plain."""

        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("b", torch.ones(2))

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a_b", torch.full((2,), 2.0))
                self.a = Inner()

        model = Outer().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            annotate.input_tensors("policy", {"x": torch.ones(2)})
            with self.assertRaisesRegex(ValueError, r"\['a\.b', 'a_b'\]"):
                annotate.module("policy", model)
        finally:
            leapp.stop()
        for buf in model.buffers():
            _assert_plain(self, buf)

    def test_failed_node_keeps_another_open_nodes_input(self):
        """A node that fails to compile leaves a buffer another open node
        declared as its input traced, so that node still compiles."""

        class Shared(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

            def forward(self, x):
                return x * 2 + self.h

        h = torch.zeros(3)
        model = Shared(h).eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            # The buffer shares memory with a declared input, so module() leaves
            # it plain, and the other node can declare it.
            x = annotate.input_tensors("policy", {"x": h.view(3)})
            annotate.module("policy", model)
            held = annotate.input_tensors("other", {"h": h})
            y = model(x)
            with self.assertRaisesRegex(Exception, "Unexpected backend"):
                annotate.output_tensors("policy", {"y": y}, export_with="jitt")
            self.assertIsInstance(held, TracedTensor)
            annotate.output_tensors("other", {"o": held * 3}, export_with="jit")
        finally:
            leapp.stop()

    def test_buffer_set_by_another_open_node_stays_traced(self):
        """A buffer that another node, still capturing, sets to its value,
        while the user's update_state() decides the tracking node's state,
        keeps that value traced after the tracking node compiles, so the other
        node still connects to it."""

        class Offset(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(3))

            def forward(self, x):
                return x + self.h

        model = Offset().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(3)})
            annotate.module("policy", model)
            y = model(x)
            annotate.update_state("policy", {"h": y})
            z = annotate.input_tensors("other", {"z": torch.ones(3)})
            model.h = z * 2
            annotate.output_tensors("policy", {"y": y}, export_with="jit")
            self.assertIsInstance(model.h, TracedTensor)
            annotate.output_tensors("other", {"o": model.h * 3}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {"policy": True, "other": True},
        )
        self.assertEqual(annotate.detected_pipeline["inputs"]["other"], ["z"])

    def test_buffer_changed_in_place_raises(self):
        """A buffer the graph reads and the forward changes in place, also
        through .data, a view, or another buffer on the same memory, also in
        another layout and when the write leaves the values, or resizes it, or
        does not read it, since a later node or step reads the buffer, or one
        whose graph input FX renames, also through .data, an integer one too, or with a write the graph records that
        leaves the capture sample as it was, also one the graph reads only in
        an in-place write to a view of an output, raises, naming the buffer as
        buffer_names does, again on every call, since a baked constant cannot
        carry the change. One the graph reads only through a declared input
        does not raise. The model's buffers, apart from a declared one, are
        plain afterwards."""

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(2))
                self.register_buffer("b", torch.zeros(2))

            def forward(self, x):
                self.a.add_(x)
                return x + self.a

        class Views(nn.Module):
            def __init__(self):
                super().__init__()
                memory = torch.zeros(3)
                self.register_buffer("a", memory[:2])
                self.register_buffer("b", memory[1:])

            def forward(self, x):
                y = x + self.b
                self.a.add_(x)
                self.a = self.a * 2
                return y

        class Data(Counter):
            def forward(self, x):
                self.a.data.mul_(0.9)
                return x + self.a

        class DataAlias(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.zeros(2))
                self.register_buffer("b", self.a.data)

            def forward(self, x):
                self.b.add_(1)
                return x + self.a

        class Nested(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Counter()

            def forward(self, x):
                return self.enc(x)

        class Max(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("max", torch.zeros(2))

            def forward(self, x):
                self.max.add_(x)
                return x + self.max

        class Masked(Counter):
            def forward(self, x):
                self.a[x < 0] = 0.0
                return x + self.a

        class AliasWrite(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))
                self.register_buffer("b", self.h.view(1, 2))

            def forward(self, x):
                self.h.mul_(1.0)
                self.h = self.h + 1
                return x + self.b

        class Resize(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(4))

            def forward(self, x):
                self.a.set_(torch.zeros(2))
                return x + self.a

        class ViewWrite(Counter):
            def forward(self, x):
                self.a[:1].mul_(1.0)
                return x + self.a

        class StridedViewWrite(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(8)[::4])

            def forward(self, x):
                self.a[1:].mul_(1.0)
                return x + self.a

        class StridedRowWrite(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(2, 8)[:, ::4])

            def forward(self, x):
                self.a[0, 1:].mul_(1.0)
                return x + self.a[0]

        class HalfData(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(2, dtype=torch.bfloat16))

            def forward(self, x):
                self.a.data.mul_(1.1)
                return x + self.a

        class ReadByViewWrite(Counter):
            def forward(self, x):
                self.a.add_(x)
                out = x.clone()
                out[:1].div_(self.a[:1])
                return out

        class Store(Counter):
            def forward(self, x):
                self.a.copy_(x)
                return x * 2

        class StoreData(Counter):
            def forward(self, x):
                self.a.data.mul_(0.5)
                return x * 2

        class MaxData(Max):
            def forward(self, x):
                self.max.data.add_(1.0)
                return x + self.max

        class DoubleData(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(2, dtype=torch.float64))

            def forward(self, x):
                self.a.data.mul_(1 + 1e-9)
                return x + self.a

        class IntData(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.zeros(2, dtype=torch.int64))

            def forward(self, x):
                self.a.data.add_(1)
                return x + self.a

        # (model, the buffer the error names, None when the capture is valid)
        cases = (
            (Counter, "a"),
            (Store, "a"),
            (StoreData, "a"),
            (Nested, "enc.a"),
            (Data, "a"),
            (DataAlias, "a"),
            (Views, "b"),
            (Max, "max"),
            (MaxData, "max"),
            (Masked, "a"),
            (ViewWrite, "a"),
            (StridedViewWrite, "a"),
            (StridedRowWrite, "a"),
            (ReadByViewWrite, "a"),
            (HalfData, "a"),
            (DoubleData, "a"),
            (IntData, "a"),
            (AliasWrite, "b"),
            (Resize, "a"),
            (Counter, None),
        )
        for model_type, name in cases:
            with self.subTest(model=model_type.__name__, name=name):
                model = model_type().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    annotate.module("policy", model)
                    if name is None:
                        annotate.input_tensors("policy", {"declared_a": model.a})
                    y = model(x)
                    if name is not None:
                        for _ in range(2):
                            with self.assertRaisesRegex(
                                RuntimeError,
                                f"Buffer '{re.escape(name)}'.*changed in place",
                            ):
                                annotate.output_tensors(
                                    "policy", {"y": y}, export_with="jit"
                                )
                    else:
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                if name is None:
                    self.assertEqual(
                        leapp.compile_graph(visualize=False, validate=True),
                        {"policy": True},
                    )
                    _assert_plain(self, model.b)
                else:
                    for buf in model.buffers():
                        _assert_plain(self, buf)

    def test_buffer_written_without_its_version_counter_raises(self):
        """A buffer the forward changes through a tensor on its bytes with
        another version counter, a NumPy view or a from_dlpack() alias
        declared as an input, raises, since the export would freeze it."""

        class NumpyWrite(nn.Module):
            def __init__(self, memory):
                super().__init__()
                self.register_buffer("h", memory)

            def forward(self, x):
                self.h.numpy()[0] += 1.0
                return x.flatten() + self.h

        class InputWrite(NumpyWrite):
            def forward(self, x):
                x.mul_(2)
                return x.flatten() + self.h

        for model_type in (NumpyWrite, InputWrite):
            with self.subTest(model=model_type.__name__):
                memory = torch.tensor([1.0, 2.0])
                model = model_type(memory).eval()
                inputs = memory if model_type is InputWrite else torch.ones(2)
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors(
                        "policy", {"x": torch.from_dlpack(inputs).view(1, 2)}
                    )
                    annotate.module("policy", model)
                    y = model(x)
                    with self.assertRaisesRegex(
                        RuntimeError, "Buffer 'h'.*changed in place"
                    ):
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                _assert_plain(self, model.h)

    def test_buffer_set_to_a_nested_tensor_is_not_called_shared(self):
        """A buffer the forward sets to a nested tensor, which has no layout to
        compare, raises the error about an untraced state, not one about
        buffers that share memory."""

        class Nest(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                self.h = torch.nested.nested_tensor([self.h + x, self.h])
                return x * 2

        model = Nest().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(2)})
            annotate.module("policy", model)
            y = model(x)
            with self.assertRaisesRegex(Exception, "non-traced tensors"):
                annotate.output_tensors("policy", {"y": y}, export_with="jit")
        finally:
            leapp.stop()

    def test_buffers_kept_plain_in_corner_cases(self):
        """A buffer that shares memory with a declared input, also written in
        place through the buffer, is the same plain tensor afterwards, and one
        the forward sets to None stays None. A buffer an in-place op the graph
        does not record changes before the forward reassigns it, which raises,
        and one the forward sets to a tensor it did not trace, are plain
        afterwards."""

        class Shared(nn.Module):
            def __init__(self, buf):
                super().__init__()
                self.register_buffer("h", buf)

            def forward(self, x):
                self.h.add_(1)
                return x * 2 + self.h

        class Dropper(Shared):
            def forward(self, x):
                out = x + self.h
                self.h = None
                return out

        class InPlace(Shared):
            def forward(self, x):
                self.h.t_()
                self.h = self.h * 2
                return x + 1

        class Untraced(Shared):
            def forward(self, x):
                self.h = torch.zeros(3)
                return x + 1

        for model_type in (Shared, Dropper, InPlace, Untraced):
            with self.subTest(model=model_type.__name__):
                # Two dimensions, so t_() changes the buffer.
                h = torch.zeros(1, 3) if model_type is InPlace else torch.zeros(3)
                model = model_type(h).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                # Shared declares a view of the buffer's own memory.
                x = annotate.input_tensors(
                    "policy",
                    {"x": h.view(3) if model_type is Shared else torch.ones(3)},
                )
                annotate.module("policy", model)
                y = model(x)
                try:
                    if model_type is Dropper:
                        with self.assertRaisesRegex(TypeError, "type None"):
                            annotate.output_tensors(
                                "policy", {"y": y}, export_with="jit"
                            )
                    elif model_type is Untraced:
                        with self.assertRaisesRegex(Exception, "non-traced"):
                            annotate.output_tensors(
                                "policy", {"y": y}, export_with="jit"
                            )
                    elif model_type is InPlace:
                        with self.assertRaisesRegex(RuntimeError, "changed in place"):
                            annotate.output_tensors(
                                "policy", {"y": y}, export_with="jit"
                            )
                    else:
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                _assert_plain(self, h)
                if model_type is Dropper:
                    self.assertIsNone(model.h)
                else:
                    _assert_plain(self, model.h)
                if model_type is Shared:
                    self.assertTrue(model.h is h)

    def test_buffers_baked_under_inference_mode(self):
        """A read-only buffer captured under torch.inference_mode() still exports,
        also when the model was built there, so the buffer is an inference
        tensor."""

        class Offset(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.full((1, 4), 0.5))

            def forward(self, obs):
                return obs - self.mean

        for export_with, built_there in itertools.product(
            ("jit", "exported-program"), (False, True)
        ):
            with self.subTest(export_with=export_with, built_there=built_there):
                with torch.inference_mode(built_there):
                    model = Offset().eval()
                with torch.inference_mode():
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    obs = annotate.input_tensors("policy", {"obs": torch.ones(1, 4)})
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"result": model(obs)}, export_with=export_with
                    )
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )

    def test_buffer_deleted_or_replaced_by_a_parameter_left_alone(self):
        """A tracked buffer the user deletes, or replaces with a parameter,
        during the session stays gone from the buffers at its end."""

        class Read(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                return x + self.h

        for change in ("deleted", "parameter"):
            with self.subTest(change=change):
                model = Read()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    annotate.input_tensors("policy", {"x": torch.ones(2)})
                    annotate.module("policy", model)
                    if change == "deleted":
                        del model.h
                    else:
                        model.h = nn.Parameter(torch.ones(2))
                finally:
                    leapp.stop()
                self.assertNotIn("h", model._buffers)
                self.assertEqual(change == "parameter", "h" in model._parameters)

    def test_failed_session_end_does_not_reach_the_next_session(self):
        """A failure while the session end makes buffers plain is not repeated
        by the next session, which captures again."""
        model = GRUModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", model)
        with mock.patch.object(
            BufferTracker, "restore", side_effect=RuntimeError("restore failed")
        ):
            with self.assertRaisesRegex(RuntimeError, "restore failed"):
                leapp.stop()
            leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.output_tensors("policy", {"y": obs * 2}, export_with="jit")
        finally:
            leapp.stop()

    def test_write_to_other_bytes_of_a_buffers_memory_is_not_a_change(self):
        """A recorded write to an input on other bytes of a buffer's memory,
        also through a view of an input that covers the buffer, or between the
        bytes of a strided buffer, leaves the buffer as it was,
        though they share one version counter, so the capture exports."""

        class Tail(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

        cases = {
            "disjoint input": (lambda m: m[4:], lambda m: m[:4], lambda x: x),
            "view of a wider input": (lambda m: m[4:], lambda m: m, lambda x: x[:4]),
            "between strided bytes": (
                lambda m: m.view(2, 4)[:, 1::2],
                lambda m: m,
                lambda x: x[2:3],
            ),
        }
        for case, (buffer, declared, written) in cases.items():
            with self.subTest(case=case):
                memory = torch.arange(8.0)
                model = Tail(buffer(memory)).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": declared(memory)})
                    annotate.module("policy", model)
                    written(x).add_(1)
                    annotate.output_tensors("policy", {"y": x * 2}, export_with="jit")
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )

    def test_inference_buffer_changed_in_place_raises(self):
        """A buffer built under torch.inference_mode(), which has no version
        counter, raises when the graph records an in-place write to it, also
        one that leaves the capture sample as it was, or one through .data
        moves it, and not for a write into a value computed from it, nor when
        the forward then reassigns it, which makes it feedback state."""

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(3))

            def forward(self, x):
                self.h.add_(x)
                return self.h * 2

        class Temporary(Counter):
            def forward(self, x):
                return (self.h + x).relu_()

        class Feedback(Counter):
            def forward(self, x):
                self.h.add_(x)
                self.h = self.h * 2
                return self.h

        class Data(Counter):
            def forward(self, x):
                self.h.data.add_(1)
                return x * self.h

        class Peak(Counter):
            def forward(self, x):
                self.h.copy_(torch.maximum(self.h, x - 2))
                return x + self.h

        for model_type in (Counter, Temporary, Feedback, Data, Peak):
            with self.subTest(model_type=model_type.__name__):
                with torch.inference_mode():
                    model = model_type().eval()
                # An in-place write to an inference tensor needs the mode.
                with torch.inference_mode(model_type is not Temporary):
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    try:
                        x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                        annotate.module("policy", model)
                        y = model(x)
                        if model_type in (Counter, Data, Peak):
                            with self.assertRaisesRegex(
                                RuntimeError, "changed in place"
                            ):
                                annotate.output_tensors(
                                    "policy", {"y": y}, export_with="jit"
                                )
                            continue
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                    finally:
                        leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )

    def test_non_mutated_buffers_baked_as_constants(self):
        """Non-mutated buffers should be baked into the model, not dynamic inputs.

        Verifies that the exported model produces correct output using the
        baked buffer values (scale=2, bias=1) without needing them as inputs,
        both through TorchScript and through torch.export, which cannot take
        a traced tensor as a constant.
        """

        class ScaleBiasModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.tensor([2.0, 2.0, 2.0, 2.0]))
                self.register_buffer("bias", torch.tensor([1.0, 1.0, 1.0, 1.0]))
                self.register_buffer("unused", torch.zeros(4))

            def forward(self, obs):
                return obs * self.scale + self.bias

        for export_with in ("jit", "exported-program", "onnx-dynamo"):
            with self.subTest(export_with=export_with):
                obs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
                # Expected: obs * 2.0 + 1.0 = [3.0, 5.0, 7.0, 9.0]
                expected = obs * 2.0 + 1.0
                model = ScaleBiasModel()
                model.eval()

                leapp.start(name=self.TEST_GRAPH_NAME)
                obs_traced = annotate.input_tensors("policy", {"obs": obs})

                annotate.module("policy", model)
                result = model(obs_traced)

                annotate.output_tensors(
                    "policy", {"result": result}, export_with=export_with
                )
                # The graph does not read the unused buffer, so no constant
                # holds a copy of it.
                root = annotate.nodes["policy"].tracer.root
                self.assertTrue(hasattr(root, "_buffer_scale"))
                self.assertFalse(hasattr(root, "_buffer_unused"))
                leapp.stop()
                # A change after capture does not reach the export.
                model.scale.add_(1.0)
                leapp.compile_graph(visualize=False)

                # Only 1 input (obs), scale and bias baked as constants
                self.verify_num_connections(
                    annotate,
                    nodes=1,
                    inputs=1,
                    outputs=1,
                    internal_connections=0,
                    feedback_connections=0,
                )

                # compile_graph() validated every export against the capture;
                # there is no helper to load the ONNX file on its own.
                verify = {
                    "jit": self.verify_single_torchscript_model_expected_value,
                    "exported-program": (
                        self.verify_single_exported_program_model_expected_value
                    ),
                }.get(export_with)
                if verify is not None:
                    verify(
                        inputs=[obs], expected_outputs=[expected], model_name="policy"
                    )

    def test_not_tracing_noop(self):
        """When not tracing, module() is a complete no-op."""
        model = GRUModel()
        model.eval()
        obs = torch.randn(1, 4)

        # Don't call leapp.start() — not tracing
        original_buffers = {n: b.clone() for n, b in model.named_buffers()}

        annotate.module("policy", model)
        action = model(obs)

        # Model should work normally and buffers should be untouched as buffers
        current_buffers = dict(model.named_buffers())
        self.assertEqual(set(original_buffers.keys()), set(current_buffers.keys()))

    def test_nested_submodule_buffers(self):
        """Buffers in nested submodules are detected correctly."""
        model = NestedBufferModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors(
            "policy", {"action": action}, export_with="onnx-torchscript"
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # GRU's h_state is at rnn.h_state (nested) -> 1 feedback
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )
        self.verify_all_models_exist("policy")


class TestBufferTrackingEdgeCases(LEAPPFunctionalTestBase):
    """Edge cases for annotate.module() — complex operations and tracing behavior."""

    def test_chained_operations_on_buffer(self):
        """Buffer used in a chain of operations: reshape, multiply, add."""

        class ChainedOpsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(1, 8))
                self.linear = nn.Linear(8, 4)

            def forward(self, obs):
                # Chain: buffer -> reshape -> multiply -> add -> reassign
                updated = self.state * 0.9 + obs.reshape(1, 8) * 0.1
                self.state = updated
                return self.linear(updated)

        model = ChainedOpsModel()
        model.eval()
        obs = torch.randn(1, 8)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )
        self.verify_all_models_exist("policy")

    def test_buffer_used_in_conditional_arithmetic(self):
        """Buffer combined with torch.where (element-wise conditional)."""

        class ConditionalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("counter", torch.zeros(1, 4))
                self.linear = nn.Linear(4, 4)

            def forward(self, obs):
                incremented = self.counter + 1.0
                # torch.where with TracedTensor operands
                self.counter = torch.where(obs > 0, incremented, self.counter)
                return self.linear(obs + self.counter)

        model = ConditionalModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )

    def test_buffer_with_torch_cat(self):
        """Buffer concatenated with input using torch.cat."""

        class CatModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("history", torch.zeros(1, 4))
                self.linear = nn.Linear(8, 4)

            def forward(self, obs):
                combined = torch.cat([self.history, obs], dim=1)
                self.history = obs  # shift: current obs becomes history
                return self.linear(combined)

        model = CatModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )

    def test_buffer_with_matmul(self):
        """Buffer used in matrix multiplication."""

        class MatmulModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight_state", torch.eye(4))

            def forward(self, obs):
                result = obs @ self.weight_state
                # Decay the weight state
                self.weight_state = self.weight_state * 0.99
                return result

        model = MatmulModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )

    def test_buffer_with_slicing(self):
        """Buffer read via slicing (getitem)."""

        class SlicingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("history", torch.zeros(3, 4))
                self.linear = nn.Linear(4, 4)

            def forward(self, obs):
                # Read last row, shift history, append new obs
                out = self.linear(self.history[-1:])
                new_history = torch.cat([self.history[1:], obs], dim=0)
                self.history = new_history
                return out

        model = SlicingModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )

    def test_buffer_with_different_dtypes(self):
        """Buffers with float32 and int64 dtypes coexist."""

        class MixedDtypeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("float_state", torch.zeros(1, 4))
                self.register_buffer("int_counter", torch.zeros(1, dtype=torch.int64))
                self.linear = nn.Linear(4, 4)

            def forward(self, obs):
                self.float_state = self.float_state + obs
                self.int_counter = self.int_counter + 1
                return self.linear(self.float_state)

        model = MixedDtypeModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=2,
        )

    def test_scalar_buffer(self):
        """Single scalar buffer (0-dimensional after squeeze, 1-element tensor)."""

        class ScalarBufferModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("step", torch.tensor([0.0]))
                self.linear = nn.Linear(4, 4)

            def forward(self, obs):
                self.step = self.step + 1.0
                return self.linear(obs) * (1.0 / (self.step + 1.0))

        model = ScalarBufferModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )

    def test_deeply_nested_submodule_buffer(self):
        """Buffer three levels deep: model.proj.rnn.h_state."""

        class DeepModel(nn.Module):
            def __init__(self):
                super().__init__()
                # NestedBufferModel expects obs_dim=4, so project 4->4 here
                self.proj = nn.Sequential(nn.Linear(4, 4), nn.ELU())
                self.rnn = NestedBufferModel()  # has encoder(4->8) + GRUModel(8) inside

            def forward(self, obs):
                features = self.proj(obs)
                return self.rnn(features)

        model = DeepModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors(
            "policy", {"action": action}, export_with="onnx-torchscript"
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # rnn.rnn.h_state is 3 levels deep -> still detected
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )

    def test_multiple_outputs_with_buffer(self):
        """Model returns multiple outputs alongside buffer mutation."""

        class MultiOutputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("hidden", torch.zeros(1, 4))
                self.head_a = nn.Linear(4, 3)
                self.head_b = nn.Linear(4, 2)

            def forward(self, obs):
                self.hidden = self.hidden * 0.9 + obs * 0.1
                return self.head_a(self.hidden), self.head_b(self.hidden)

        model = MultiOutputModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        out_a, out_b = model(obs_traced)

        annotate.output_tensors(
            "policy", {"action_a": out_a, "action_b": out_b}, export_with="jit"
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=2,
            internal_connections=0,
            feedback_connections=1,
        )

    def test_buffer_with_clamp(self):
        """Buffer updated with torch.clamp — common in RL for bounding state."""

        class ClampModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(1, 4))
                self.linear = nn.Linear(4, 4)

            def forward(self, obs):
                raw = self.state + obs
                self.state = torch.clamp(raw, -1.0, 1.0)
                return self.linear(self.state)

        model = ClampModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )


class TestBufferTrackingErrors(LEAPPFunctionalTestBase):
    """Error cases and validation for annotate.module()."""

    def test_module_before_input_tensors_raises(self):
        """module() called before input_tensors() should raise — node doesn't exist."""
        model = GRUModel()

        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            annotate.module("policy", model)
            leapp.stop()
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("module", str(e).lower())
        finally:
            # Ensure tracing is stopped even on exception
            try:
                leapp.stop()
            except Exception:
                pass

    def test_buffer_name_taken_by_an_input(self):
        """module() rejects a buffer whose name the node already uses for a
        regular input before it promotes any buffer, so all stay plain and a
        retry without it tracks the others."""

        class Pair(nn.Module):
            def __init__(self, name):
                super().__init__()
                self.register_buffer("a", torch.ones(2))
                self.register_buffer(name, torch.ones(2))

        # FX renames a graph input named max, so the check matches its target.
        for name in ("b", "max"):
            with self.subTest(name=name):
                model = Pair(name).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    annotate.input_tensors(
                        "policy", {"x": torch.ones(2), name: torch.zeros(2)}
                    )
                    with self.assertRaisesRegex(ValueError, rf"\['{name}'\]"):
                        annotate.module("policy", model)
                    for buf in model.buffers():
                        _assert_plain(self, buf)
                    annotate.module("policy", model, buffer_names=["a"])
                finally:
                    leapp.stop()

    def test_unsupported_buffer_rejected_before_any_is_tracked(self):
        """A module() call with a buffer no node input can take, a complex or a
        sparse one, raises before it tracks any buffer, so the call can be
        repeated in the same session without that buffer."""

        class Rotary(nn.Module):
            def __init__(self, extra):
                super().__init__()
                self.register_buffer("scale", torch.ones(2))
                self.register_buffer("extra", extra)

            def forward(self, x):
                return x * self.scale

        for extra in (
            torch.ones(2, dtype=torch.complex64),
            torch.ones(2).to_sparse(),
        ):
            with self.subTest(extra=f"{extra.dtype} {extra.layout}"):
                model = Rotary(extra).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    with self.assertRaisesRegex(ValueError, r"\['extra "):
                        annotate.module("policy", model)
                    _assert_plain(self, model.scale)
                    annotate.module("policy", model, buffer_names=["scale"])
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )

    def test_buffer_traced_by_another_open_node_rejected(self):
        """A buffer still traced by another node that has not called
        output_tensors() is rejected before any buffer is tracked."""

        class Pair(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("a", torch.ones(2))
                self.register_buffer("b", b)

        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            other = annotate.input_tensors("other", {"z": torch.ones(2)})
            model = Pair(other)
            annotate.input_tensors("policy", {"x": torch.ones(2)})
            with self.assertRaisesRegex(ValueError, "traced by node 'other'"):
                annotate.module("policy", model)
            _assert_plain(self, model.a)
            self.assertEqual([d.name for d in annotate.nodes["policy"].inputs], ["x"])
        finally:
            leapp.stop()

    def test_failed_module_call_keeps_another_nodes_output(self):
        """A module() call that fails part way leaves a buffer holding another
        node's output traced, so a later node still connects to that node."""

        class Picky(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("b", b)
                self.register_buffer("c", torch.ones(2))

            def __setattr__(self, name, value):
                if name == "c" and isinstance(value, TracedTensor):
                    raise TypeError("c must stay a plain tensor")
                super().__setattr__(name, value)

        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            p_in = annotate.input_tensors("producer", {"x": torch.ones(2)})
            out = annotate.output_tensors(
                "producer", {"o": p_in * 2}, export_with="jit"
            )
            x = annotate.input_tensors("policy", {"x": torch.ones(2)})
            with self.assertRaisesRegex(TypeError, "plain tensor"):
                annotate.module("policy", Picky(out))
            self.assertIsInstance(out, TracedTensor)
            annotate.output_tensors("policy", {"y": x * 2}, export_with="jit")
            c_in = annotate.input_tensors("later", {"x": out})
            annotate.output_tensors("later", {"y": c_in * 3}, export_with="jit")
        finally:
            leapp.stop()
        leapp.compile_graph(visualize=False)
        self.assertIn(
            "later/x", annotate.detected_pipeline["data_flow"].get("producer/o", [])
        )

    def test_module_called_twice_for_a_node_raises(self):
        """A second module() call for a node raises, since the node keeps one
        tracker, and the first call's buffers are still restored, also after a
        failed output_tensors() baked every buffer, since the repeated forward
        would not reach the graph, and in a second capture pass after a failed
        output_tensors() that left buffers tracked, which a new tracker would
        drop. A first call on a module without buffers tracks nothing, so it
        does not count."""
        model = GRUModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.module("policy", nn.Linear(4, 4))
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )

        model = GRUModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.module("policy", model)
            with self.assertRaisesRegex(ValueError, "already called"):
                annotate.module("policy", nn.Module())
        finally:
            leapp.stop()
        _assert_plain(self, model.h_state)

        model = NoMutationModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.module("policy", model)
            with self.assertRaisesRegex(Exception, "Unexpected backend"):
                annotate.output_tensors(
                    "policy", {"action": model(obs)}, export_with="jitt"
                )
            with self.assertRaisesRegex(ValueError, "already called"):
                annotate.module("policy", model)
        finally:
            leapp.stop()

        model = GRUModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.module("policy", model)
            with self.assertRaisesRegex(Exception, "Unexpected backend"):
                annotate.output_tensors(
                    "policy", {"action": model(obs)}, export_with="jitt"
                )
            annotate.nodes["policy"].reset_trace_state()
            with self.assertRaisesRegex(ValueError, "already called"):
                annotate.module("policy", model)
        finally:
            leapp.stop()
        for buf in model.buffers():
            _assert_plain(self, buf)

    def test_module_called_on_every_step(self):
        """A rollout loop that calls module() on every step captures, since a
        re-entry pass tracks no buffers."""
        model = NoMutationModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(3):
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )
        for buf in model.buffers():
            _assert_plain(self, buf)

    def test_module_called_on_a_second_capture_pass(self):
        """A second capture pass of a node, as a Warp node runs after its first
        pass, tracks a reassigned buffer again, though the node still lists the
        buffer's input from the first pass, and leaves it plain, also under
        inference mode, where the first pass left an inference tensor, and when
        a write the graph does not record leaves the second pass's value."""

        class Rewrite(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h_state", torch.ones(1, 4))

            def forward(self, x):
                self.h_state.data.copy_(self.h_state.data)
                self.h_state = self.h_state * 0.5 + x
                return self.h_state * 2

        for model_type, inference in (
            (GRUModel, False),
            (GRUModel, True),
            (Rewrite, False),
        ):
            with self.subTest(model=model_type.__name__, inference=inference):
                model = model_type().eval()
                with torch.inference_mode(inference):
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    for capture_pass in range(2):
                        if capture_pass:
                            annotate.nodes["policy"].reset_trace_state()
                        obs = annotate.input_tensors(
                            "policy", {"obs": torch.randn(1, 4)}
                        )
                        annotate.module("policy", model)
                        annotate.output_tensors(
                            "policy", {"action": model(obs)}, export_with="jit"
                        )
                    leapp.stop()
                _assert_plain(self, model.h_state)
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(
                    annotate.detected_pipeline["feedback_flow"],
                    {"policy/h_state": ["policy/h_state"]},
                )

    def test_module_called_on_the_first_capture_pass_only(self):
        """A second capture pass that does not call module() again captures,
        since the first pass did not fail, and a meta buffer the forward does
        not read is left alone."""

        class Scale(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.full((4,), 2.0))
                self.register_buffer("unused", torch.empty(4, device="meta"))

            def forward(self, x):
                return x * self.scale

        model = Scale().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            for capture_pass in range(2):
                if capture_pass:
                    annotate.nodes["policy"].reset_trace_state()
                x = annotate.input_tensors("policy", {"x": torch.ones(4)})
                if not capture_pass:
                    annotate.module("policy", model)
                annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )
        _assert_plain(self, model.scale)
        self.assertTrue(model.unused.is_meta)

    def test_buffer_declared_after_module_on_a_second_capture_pass(self):
        """A buffer declared with input_tensors() after module() on the first
        capture pass is not an input of the second pass, so module() tracks it
        again there, and it is plain once that pass compiles when the pass
        does not declare it."""

        class Offset(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("b", torch.ones(3))

            def forward(self, x):
                return x + self.b

        for declared_again in (True, False):
            with self.subTest(declared_again=declared_again):
                model = Offset().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    for capture_pass in range(2):
                        if capture_pass:
                            annotate.nodes["policy"].reset_trace_state()
                        x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                        annotate.module("policy", model)
                        if declared_again or not capture_pass:
                            annotate.input_tensors("policy", {"b_in": model.b})
                        annotate.output_tensors(
                            "policy", {"y": model(x)}, export_with="jit"
                        )
                    if not declared_again:
                        _assert_plain(self, model.b)
                finally:
                    leapp.stop()
                # The node keeps the first pass's inputs, so only the model is
                # checked when the second pass does not declare the buffer.
                if declared_again:
                    self.assertEqual(
                        leapp.compile_graph(visualize=False, validate=True),
                        {"policy": True},
                    )

    def test_module_call_refused_by_a_setattr_override_leaves_buffers_plain(self):
        """A module whose __setattr__ refuses a traced buffer, before or after
        storing it, makes module() fail part way, and the buffers it already
        promoted or replaced are their original tensors again right away, a
        non-persistent one still out of state_dict() and a Parameter still a
        buffer. Declaring one of them again in that node raises, since the node
        still lists it, also a tensor subclass, which has its own traced
        tensor."""

        class Tagged(torch.Tensor):
            pass

        class Picky(nn.Module):
            refused = None
            store_first = False

            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(2))
                self.register_buffer("n", torch.ones(2), persistent=False)
                self.register_buffer("p", nn.Parameter(torch.ones(2)))
                self.register_buffer("t", torch.ones(2).as_subclass(Tagged))
                self.register_buffer("b", torch.ones(2))

            def __setattr__(self, name, value):
                if name == self.refused and isinstance(value, TracedTensor):
                    if self.store_first:
                        super().__setattr__(name, value)
                    raise TypeError(f"{name} must stay as it is")
                super().__setattr__(name, value)

        cases = [*itertools.product(("t", "b"), (False, True), [None])]
        cases += [("t", False, ["t"]), ("t", True, ["t"])]
        for refused, store_first, names in cases:
            with self.subTest(refused=refused, store_first=store_first, names=names):
                model = Picky().eval()
                model.refused, model.store_first = refused, store_first
                t = model.t
                first = "a" if names is None else "t"
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    annotate.input_tensors("policy", {"x": torch.ones(2)})
                    with self.assertRaisesRegex(TypeError, "must stay as it is"):
                        annotate.module("policy", model, buffer_names=names)
                    _assert_plain(self, model.b)
                    with self.assertRaisesRegex(
                        RuntimeError, f"State tensor '{first}' .* left"
                    ):
                        annotate.module("policy", model, buffer_names=names)
                    with self.assertRaisesRegex(
                        RuntimeError, f"State tensor '{first}' .* left"
                    ):
                        annotate.state_tensors("policy", {first: getattr(model, first)})
                finally:
                    leapp.stop()
                _assert_plain(self, model.a)
                _assert_plain(self, model.n)
                self.assertNotIn("n", model.state_dict())
                self.assertIn("p", dict(model.named_buffers()))
                self.assertEqual(list(model.parameters()), [])
                self.assertIs(model.t, t)
                _assert_plain(self, model.b)

    def test_other_model_tracked_after_a_refused_module_call(self):
        """After module() fails part way because a __setattr__ refuses a traced
        buffer, another model tracked in that node captures."""

        class Picky(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(2))
                self.register_buffer("b", torch.ones(2))

            def __setattr__(self, name, value):
                if name == "b" and isinstance(value, TracedTensor):
                    raise TypeError("b must stay as it is")
                super().__setattr__(name, value)

        class Decay(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                self.h = self.h * 0.5 + x
                return self.h * 2

        model, other = Picky(), Decay().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(2)})
            with self.assertRaisesRegex(TypeError, "must stay as it is"):
                annotate.module("policy", model)
            annotate.module("policy", other)
            annotate.output_tensors("policy", {"y": other(x)}, export_with="jit")
        finally:
            leapp.stop()
        _assert_plain(self, model.a)
        _assert_plain(self, other.h)

    def test_empty_buffer_one_node_reassigns_read_by_another(self):
        """An empty buffer one node reassigns holds no values, so another node
        that reads it captures."""

        class Empty(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(0))

            def forward(self, x, write):
                if write:
                    self.h = self.h * 2
                return x + self.h.sum()

        model = Empty().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            for name, write in (("a", True), ("b", False)):
                x = annotate.input_tensors(name, {"x": torch.ones(2)})
                annotate.module(name, model)
                annotate.output_tensors(name, {"y": model(x, write)}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {"a": True, "b": True},
        )

    def test_empty_buffer_declared_as_its_own_state(self):
        """An empty buffer declared with state_tensors() under its own name is
        tracked, since it is the same tensor, though it has no memory layout,
        and an empty state of another shape under its name is rejected."""

        class Empty(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.empty(0, 2))

        model = Empty().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            annotate.input_tensors("policy", {"x": torch.ones(2)})
            h = annotate.state_tensors("policy", {"h": model.h})
            annotate.module("policy", model)
        finally:
            leapp.stop()
        self.assertIsInstance(h, TracedTensor)
        _assert_plain(self, model.h)

        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            annotate.input_tensors("policy", {"x": torch.ones(2)})
            annotate.state_tensors("policy", {"h": torch.empty(0, 3)})
            with self.assertRaisesRegex(ValueError, r"\['h'\]"):
                annotate.module("policy", model)
        finally:
            leapp.stop()

    def test_buffer_declared_as_a_state_and_updated_by_the_user(self):
        """A buffer declared with state_tensors() under its own name and updated
        with update_state() stays the user's feedback state when module() also
        tracks it, before or after module(), also when the forward changes it
        in place or reassigns it. The tensors the user declared stay traced, like
        every node input and output, while after the session the buffer holds a
        plain tensor."""

        class Reader(nn.Module):
            def __init__(self, write):
                super().__init__()
                self.register_buffer("h", torch.zeros(3))
                self.write = write

            def forward(self, x):
                y = x + self.h
                if self.write == "in_place":
                    self.h.mul_(0.5).add_(x)
                elif self.write == "reassign":
                    self.h = self.h + x
                return y

        for write, when in itertools.product(
            ("none", "in_place", "reassign"), ("before", "after")
        ):
            with self.subTest(write=write, when=when):
                model = Reader(write).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                    if when == "before":
                        annotate.state_tensors("policy", {"h": model.h})
                    annotate.module("policy", model)
                    if when == "after":
                        annotate.state_tensors("policy", {"h": model.h})
                    y = model(x)
                    state = model.h if write == "in_place" else y * 0.5
                    annotate.update_state("policy", {"h": state})
                    annotate.output_tensors("policy", {"y": y}, export_with="jit")
                    self.assertIs(
                        annotate.nodes["policy"]._state_tensors["h"]["output"], state
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(
                    annotate.detected_pipeline["feedback_flow"],
                    {"policy/h": ["policy/h"]},
                )
                _assert_plain(self, model.h)

    def test_buffer_declared_as_an_input_view_under_its_own_name(self):
        """A buffer declared with input_tensors() under its own name, through a
        view with the same layout, is tracked, since the buffer adopts that
        input."""

        class Acc(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                self.h = self.h + x
                return self.h * 2

        model = Acc().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors(
                "policy", {"x": torch.ones(2), "h": model.h.view(2)}
            )[0]
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )
        self.assertEqual(
            annotate.detected_pipeline["feedback_flow"], {"policy/h": ["policy/h"]}
        )

    def test_read_only_buffer_sharing_memory_with_a_declared_input_stays_an_input(
        self,
    ):
        """A read-only buffer whose memory an input declared through a view also
        describes, under the buffer's name before module() or another name
        after it, stays an input, since baking it would drop that input, with
        no warning that its state was never updated."""

        class Goal(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("goal", torch.full((2,), 5.0))

            def forward(self, x, target=None):
                return x + (self.goal if target is None else target)

        for when in ("before", "after"):
            with self.subTest(when=when):
                model = Goal().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    if when == "before":
                        x = annotate.input_tensors(
                            "policy", {"x": torch.ones(2), "goal": model.goal.view(2)}
                        )[0]
                        annotate.module("policy", model)
                        y = model(x)
                    else:
                        x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                        annotate.module("policy", model)
                        target = annotate.input_tensors(
                            "policy", {"target": model.goal.view(2)}
                        )
                        y = model(x, target)
                    with self.assertNoLogs("leapp", level="WARNING"):
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(
                    annotate.detected_pipeline["inputs"]["policy"], ["x", "goal"]
                )

    def test_buffers_sharing_memory_in_the_same_layout(self):
        """Two buffers on the same memory in the same layout, a view or the
        same tensor registered twice, share one graph input, so they are baked
        when the forward only reads them, are one state when it sets both to
        one new tensor, and raise when it reassigns only one, since that input
        cannot carry both values, with no warning while module() declares
        them. Both end plain, and set together, also as views of one new
        tensor, they still share its memory and version counter."""

        class Pair(nn.Module):
            def __init__(self, reassign, same):
                super().__init__()
                memory = torch.full((2,), 5.0)
                self.register_buffer("a", memory)
                self.register_buffer("b", memory if same else memory.view(2))
                self.reassign = reassign

            def forward(self, x):
                y = x + self.a
                if self.reassign == "one":
                    self.b = self.b + x
                elif self.reassign == "both":
                    self.a = self.b = self.a + x
                elif self.reassign == "views":
                    self.a = self.a + x
                    self.b = self.a.view(2)
                return y + self.b

        for reassign, same in itertools.product(
            (None, "one", "both", "views"), (False, True)
        ):
            with self.subTest(reassign=reassign, same=same):
                model = Pair(reassign, same).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    with self.assertNoLogs("leapp", level="WARNING"):
                        annotate.module("policy", model)
                    y = model(x)
                    if reassign == "one":
                        with self.assertRaisesRegex(
                            RuntimeError, r"\['a', 'b'\].*share memory"
                        ):
                            annotate.output_tensors(
                                "policy", {"y": y}, export_with="jit"
                            )
                    else:
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                if reassign != "one":
                    self.assertEqual(
                        leapp.compile_graph(visualize=False, validate=True),
                        {"policy": True},
                    )
                    self.assertEqual(
                        annotate.detected_pipeline["feedback_flow"],
                        {"policy/a": ["policy/a"]} if reassign else {},
                    )
                    if not reassign:
                        self.assertEqual(
                            annotate.detected_pipeline["inputs"]["policy"], ["x"]
                        )
                _assert_plain(self, model.a)
                _assert_plain(self, model.b)
                if reassign == "both":
                    self.assertIs(model.a, model.b)
                if reassign == "views":
                    version = model.b._version
                    model.a.zero_()
                    torch.testing.assert_close(model.b, torch.zeros(2))
                    self.assertGreater(model.b._version, version)

    def test_buffer_left_out_holding_a_tracked_tensor(self):
        """A buffer left out of buffer_names that holds the tensor of a tracked
        one reads its graph input, so reassigning the tracked one raises, while
        reading both captures. Swapping a tracked buffer with one left out
        raises too, and captures when both are tracked."""

        class Tied(nn.Module):
            def __init__(self, reassign):
                super().__init__()
                memory = torch.full((2,), 5.0)
                self.register_buffer("a", memory)
                self.sub = nn.Module()
                self.sub.register_buffer("b", memory)
                self.reassign = reassign

            def forward(self, x):
                y = x * self.sub.b + self.a
                if self.reassign:
                    self.a = self.a + 1
                return y

        for reassign in (False, True):
            with self.subTest(reassign=reassign):
                model = Tied(reassign).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    annotate.module("policy", model, buffer_names=["a"])
                    y = model(x)
                    if reassign:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            r"\['a', 'sub.b'\] .* left out of buffer_names",
                        ):
                            annotate.output_tensors(
                                "policy", {"y": y}, export_with="jit"
                            )
                        continue
                    annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )

        class Swap(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("p", torch.zeros(2))
                self.register_buffer("q", torch.ones(2))

            def forward(self, x):
                self.p, self.q = self.q, self.p
                return x + self.p

        for names in (["q"], ["p", "q"]):
            with self.subTest(buffer_names=names):
                model = Swap().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    annotate.module("policy", model, buffer_names=names)
                    y = model(x)
                    if names == ["q"]:
                        with self.assertRaisesRegex(
                            RuntimeError, "list every buffer the forward sets"
                        ):
                            annotate.output_tensors(
                                "policy", {"y": y}, export_with="jit"
                            )
                        continue
                    annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )

    def test_buffers_sharing_one_input_set_apart(self):
        """Two buffers on one memory in one layout that the forward does not
        reassign are not checked for different tensors, so transposing one in
        place raises as an in-place change, and two slots of one empty tensor set
        to different empty tensors of one shape are one state, since they hold
        no values, as is one set to a new tensor while the other is set to
        None, while empty ones of different shapes raise."""

        class Pair(nn.Module):
            def __init__(self, case):
                super().__init__()
                empty = case.startswith("empty")
                memory = torch.zeros(0) if empty else torch.arange(6.0).view(2, 3)
                self.register_buffer("a", memory)
                self.register_buffer("b", memory if empty else memory.view(2, 3))
                self.case = case

            def forward(self, x):
                if self.case == "transposed":
                    self.a.t_()
                elif self.case == "empty":
                    self.a, self.b = x[:0] + 1, x[:0] + 2
                elif self.case == "one cleared":
                    self.a, self.b = self.a + x.sum(), None
                else:
                    self.a, self.b = x[:0] + 1, torch.zeros(0, 3) + x[:1]
                return x * 2

        for case in ("transposed", "empty", "one cleared", "empty, other shapes"):
            with self.subTest(case=case):
                model = Pair(case).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    annotate.module("policy", model)
                    y = model(x)
                    if case == "transposed":
                        with self.assertRaisesRegex(RuntimeError, "changed in place"):
                            annotate.output_tensors(
                                "policy", {"y": y}, export_with="jit"
                            )
                        continue
                    if case == "empty, other shapes":
                        with self.assertRaisesRegex(RuntimeError, "share memory"):
                            annotate.output_tensors(
                                "policy", {"y": y}, export_with="jit"
                            )
                        continue
                    annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )

    def test_buffers_sharing_an_input_declared_under_another_name(self):
        """A buffer that an input declared under another name describes, alone
        or with a second buffer on its tensor or a view of it, also under the
        second buffer's name, is one state fed back to that input when the
        forward sets them to one new tensor."""

        class Pair(nn.Module):
            def __init__(self, alias):
                super().__init__()
                memory = torch.ones(2)
                self.register_buffer("h", memory)
                if alias:
                    self.register_buffer(
                        "alias", memory if alias == "same" else memory.view(2)
                    )

            def forward(self, x):
                self.h = self.h + x
                if hasattr(self, "alias"):
                    self.alias = self.h
                return self.h * 2

        for alias, name in itertools.product((None, "same", "view"), ("h_in", "alias")):
            with self.subTest(alias=alias, name=name):
                model = Pair(alias).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors(
                        "policy", {"x": torch.ones(2), name: model.h.view_as(model.h)}
                    )[0]
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(
                    annotate.detected_pipeline["feedback_flow"],
                    {"policy/h": [f"policy/{name}"]},
                )

    def test_buffers_keep_their_memory_relations_after_tracing(self):
        """Buffers the forward sets to views of one tensor, also of another
        dtype, of a buffer it keeps, or of a tensor the node outputs, also one
        with autograd history, still share its memory and version counter once
        they are plain, with that history for every view, and a buffer
        it keeps stays the same tensor, also for an input and a view of it,
        when two modules hold one state tensor, which stays one tensor, and for
        a buffer left out of buffer_names that views a copied one, and buffers
        set to separate empty tensors stay separate."""

        class Bits(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.zeros(2))
                self.register_buffer("b", torch.zeros(2, dtype=torch.int32))

            def forward(self, x):
                self.a = self.a + x
                self.b = self.a.view(torch.int32)
                return self.a * 2

        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(2, 2))
                self.register_buffer("b", torch.zeros(4))

            def forward(self, x):
                self.b = self.a.view(4)
                return x + self.b

        class Published(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.zeros(4))
                self.register_buffer("b", torch.zeros(4))

            def forward(self, x):
                self.a = self.a + x
                self.b = self.a.view(4)
                return self.a

        class Empty(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.empty(0))
                self.register_buffer("b", torch.empty(0))

            def forward(self, x):
                self.a = self.a + 1
                self.b = self.b + 2
                return x * 2

        cases = (
            (Bits, True, False),
            (Flat, True, True),
            (Published, True, False),
            (Empty, False, False),
        )
        for model_type, shared, keeps_a in cases:
            with self.subTest(model=model_type.__name__):
                model = model_type().eval()
                a = model.a
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(4)})
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"y": model(x[: model.b.numel()])}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                _assert_plain(self, model.a)
                _assert_plain(self, model.b)
                self.assertEqual(model.a is a, keeps_a)
                version = model.b._version
                model.a.zero_()
                self.assertEqual(model.b._version > version, shared)
                if shared:
                    self.assertFalse(model.b.any())

        class Gradient(nn.Module):
            def __init__(self, view):
                super().__init__()
                self.register_buffer("a", torch.zeros(2))
                self.register_buffer("b", view(torch.zeros(2)))
                self.view = view

            def forward(self, x):
                self.a = x
                self.b = self.view(x)
                return x * 2

        views = {"view": lambda t: t.view(2), "dtype": lambda t: t.view(torch.int32)}
        for kind, view in views.items():
            with self.subTest(model="Gradient", view=kind):
                model = Gradient(view)
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors(
                        "policy", {"x": torch.ones(2, requires_grad=True)}
                    )
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                _assert_plain(self, model.a)
                _assert_plain(self, model.b)
                with torch.no_grad():
                    model.a.add_(3)
                torch.testing.assert_close(model.b, view(model.a.detach()))

        class Slices(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.zeros(2))
                self.register_buffer("b", torch.zeros(2))

            def forward(self, left, right):
                self.a, self.b = left, right
                return left * 2

        with self.subTest(model="Slices"):
            model = Slices()
            root = torch.arange(1.0, 5.0, requires_grad=True)
            left, right = root[:2], root[2:]
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                x = annotate.input_tensors("producer", {"x": root})
                annotate.output_tensors("producer", {"y": x * 1}, export_with="jit")
                left, right = annotate.input_tensors(
                    "policy", {"left": left, "right": right}
                )
                annotate.module("policy", model)
                annotate.output_tensors(
                    "policy", {"y": model(left, right)}, export_with="jit"
                )
            finally:
                leapp.stop()
            model.b.sum().backward()
            root.clear_tracing_state()
            torch.testing.assert_close(root.grad, torch.tensor([0.0, 0.0, 1.0, 1.0]))
            self.assertEqual(
                model.a.untyped_storage().data_ptr(),
                model.b.untyped_storage().data_ptr(),
            )

        class Reader(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

            def forward(self, x):
                return x + self.h

        class Untracked(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.ones(2))
                self.register_buffer("b", self.a.view(2))

            def forward(self, x):
                return x + self.a + self.b

        class Input(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.zeros(2))
                self.register_buffer("b", torch.zeros(1, 2))

            def forward(self, x):
                self.a = x
                self.b = x.view(1, 2)
                return x * 2

        memory = torch.ones(2)
        pair = {"a": Reader(memory).eval(), "b": Reader(memory).eval()}
        model = Input().eval()
        untracked = Untracked().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            for name, reader in pair.items():
                x = annotate.input_tensors(name, {"x": torch.ones(2)})
                annotate.state_tensors(name, {"h": reader.h})
                annotate.module(name, reader)
                annotate.output_tensors(name, {"y": reader(x)}, export_with="jit")
            x = annotate.input_tensors("policy", {"x": torch.ones(2)})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
            x = annotate.input_tensors("untracked", {"x": torch.ones(2)})
            annotate.module("untracked", untracked, buffer_names=["a"])
            annotate.output_tensors(
                "untracked",
                {"y": untracked(x), "a": untracked.a.view(1, 2)},
                export_with="jit",
            )
        finally:
            leapp.stop()
        self.assertIs(pair["a"].h, pair["b"].h)
        pairs = ((pair["a"].h, pair["b"].h), (model.a, model.b))
        for a, b in (*pairs, (untracked.a, untracked.b)):
            version = b._version
            a.zero_()
            self.assertGreater(b._version, version)
            self.assertFalse(b.any())

    def test_buffers_sharing_memory_still_share_it_after_tracing(self):
        """Buffers that share memory still share it after the session, as in
        eager, when a node outputs a view of one of them: buffers of two
        modules that two nodes track, also two tensors made from one NumPy
        array, and when one holds a declared state: one left out of
        buffer_names made with .data, and a lazy negation or conjugation of it,
        while a sparse buffer is left alone. A buffer set to a later node's
        output that wrote in place to that view holds the written values."""

        class Reader(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.register_buffer("h", h)

            def forward(self, x):
                return x + self.h

        class Aliased(nn.Module):
            def __init__(self, make):
                super().__init__()
                self.register_buffer("a", torch.ones(4))
                self.register_buffer("b", make(self.a))
                self.register_buffer("s", torch.eye(2).to_sparse())

            def forward(self, x):
                return x + self.a

        for published, numpy in itertools.product(("first", "last"), (False, True)):
            with self.subTest(published=published, numpy=numpy):
                memory = np.ones(2, dtype=np.float32)
                tensor = torch.from_numpy(memory)
                pair = {
                    "a": Reader(torch.from_numpy(memory) if numpy else tensor).eval(),
                    "b": Reader(torch.from_numpy(memory) if numpy else tensor).eval(),
                }
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    for name, model in pair.items():
                        x = annotate.input_tensors(name, {"x": torch.ones(2)})
                        annotate.module(name, model)
                        outputs = {"y": model(x)}
                        if (name == "a") == (published == "first"):
                            outputs["h"] = model.h.view(1, 2)
                        annotate.output_tensors(name, outputs, export_with="jit")
                    self.assertEqual(pair["a"].h.data_ptr(), pair["b"].h.data_ptr())
                finally:
                    leapp.stop()
                pair["a"].h.zero_()
                self.assertFalse(pair["b"].h.any())

        makes = {
            "data": lambda a: a.data,
            "negation": lambda a: torch.view_as_complex(a.view(2, 2)).conj().imag,
            "conjugation": lambda a: torch.view_as_complex(a.view(2, 2)).conj(),
        }
        for kind, make in makes.items():
            with self.subTest(kind=kind):
                model = Aliased(make).eval()
                expected = make(torch.full((4,), 7.0))
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(4)})
                    annotate.state_tensors("policy", {"a": model.a})
                    annotate.module("policy", model, buffer_names=["a"])
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                _assert_plain(self, model.a)
                model.a.fill_(7)
                torch.testing.assert_close(model.b, expected)
                self.assertTrue(model.s.is_sparse)

        class Keep(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("b", torch.ones(2))

            def forward(self):
                return self.b.view(1, 2)

        model = Keep().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            annotate.input_tensors("a", {"x": torch.ones(2)})
            annotate.module("a", model)
            y = annotate.output_tensors("a", {"y": model()}, export_with="jit")
            z = annotate.input_tensors("c", {"z": y})
            z.add_(1)
            annotate.output_tensors("c", {"w": z}, export_with="jit")
            model.b = z.view(2)
        finally:
            leapp.stop()
        torch.testing.assert_close(model.b, torch.full((2,), 2.0))

    def test_model_tracked_by_two_nodes_keeps_its_buffers(self):
        """A model one node compiled and another tracks without reaching its
        outputs still holds its own read-only buffer after the session, so a
        module that holds the same tensor still shares it."""

        class Read(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.full((3,), 0.5))

            def forward(self, x):
                return x - self.mean

        model = Read().eval()
        mean = model.mean
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            for name in ("a", "b"):
                x = annotate.input_tensors(name, {"x": torch.ones(3)})
                annotate.module(name, model)
                y = model(x)
                if name == "a":
                    annotate.output_tensors(name, {"y": y}, export_with="jit")
        finally:
            leapp.stop()
        self.assertTrue(model.mean is mean)
        _assert_plain(self, mean)

    def test_buffer_holding_a_node_value_shares_its_memory(self):
        """A buffer the forward sets to the node's output, to a tensor a view of
        the output shares, or to an input of the node, also one whose memory
        ends in part of an element, and a buffer the forward keeps and outputs
        a view of, share their memory with that value once the node compiles,
        as in eager."""

        class Keep(nn.Module):
            def __init__(self, mode, h):
                super().__init__()
                self.register_buffer("h", h)
                self.mode = mode

            def forward(self, x):
                if self.mode == "kept view":
                    return self.h.view(1, 2)
                if self.mode.endswith("input"):
                    self.h = x
                    return x * 2
                self.h = self.h + x
                return self.h if self.mode == "output" else self.h.view(1, 2)

        odd = torch.arange(9, dtype=torch.uint8)[:8].view(torch.int32)
        for mode in ("output", "output view", "input", "odd input", "kept view"):
            with self.subTest(mode=mode):
                raw = odd if mode == "odd input" else torch.ones(2)
                model = Keep(mode, torch.zeros_like(raw)).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": raw})
                    annotate.module("policy", model)
                    y = annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                    read = x if mode.endswith("input") else y
                    _assert_plain(self, model.h)
                    model.h.add_(1)
                    torch.testing.assert_close(read.tensor.view(-1), model.h)
                finally:
                    leapp.stop()

    def test_later_node_reads_a_buffer_holding_an_earlier_output(self):
        """A model split into two nodes, where the first stores its output in a
        buffer and the second takes that output as an input and reads the
        buffer, exports a pipeline that reads the new value at every step,
        while the second node raises when it reads the buffer without taking
        that output as an input, also when the buffer holds the first node's
        state rather than its output, and after the first node's
        output_tensors() was repeated, since the export would bake it."""

        class Split(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("z", torch.zeros(2))

            def encode(self, x):
                self.z = torch.tanh(x)
                return self.z

            def act(self):
                return self.z * 3

        class State(Split):
            def encode(self, x):
                self.z = torch.tanh(x)
                return x * 2

        for retried in (False, True):
            with self.subTest(model="State", retried=retried):
                model = State().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("enc", {"x": torch.ones(2)})
                    annotate.module("enc", model)
                    y = model.encode(x)
                    if retried:
                        with self.assertRaisesRegex(Exception, "Unexpected backend"):
                            annotate.output_tensors("enc", {"y": y}, export_with="jitt")
                    annotate.output_tensors("enc", {"y": y}, export_with="jit")
                    annotate.input_tensors("act", {"x": torch.ones(2)})
                    annotate.module("act", model)
                    with self.assertRaisesRegex(
                        RuntimeError, "Buffer 'z'.*another node's output"
                    ):
                        annotate.output_tensors(
                            "act", {"a": model.act()}, export_with="jit"
                        )
                finally:
                    leapp.stop()

        for declared in (True, False):
            with self.subTest(declared=declared):
                model = Split().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("enc", {"x": torch.ones(2)})
                    annotate.module("enc", model)
                    z = annotate.output_tensors(
                        "enc", {"z": model.encode(x)}, export_with="jit"
                    )
                    if declared:
                        annotate.input_tensors("act", {"latent": z})
                    else:
                        annotate.input_tensors("act", {"x": torch.ones(2)})
                    annotate.module("act", model)
                    a = model.act()
                    if not declared:
                        with self.assertRaisesRegex(
                            RuntimeError, "Buffer 'z'.*another node's output"
                        ):
                            annotate.output_tensors("act", {"a": a}, export_with="jit")
                        continue
                    annotate.output_tensors("act", {"a": a}, export_with="jit")
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"enc": True, "act": True},
                )
                x = torch.tensor([0.5, -2.0])
                self.verify_inference_manager(
                    source_inputs={"enc/x": x},
                    source_outputs={"act/a": torch.tanh(x) * 3},
                )

    def test_export_follows_new_inputs_over_steps(self):
        """The exported pipeline matches eager over several steps with new
        inputs, so no buffer keeps its value from the capture: a buffer the
        forward reassigns is fed back, one it only reads is baked, and a later
        node reads the output of the first that it takes as an input."""

        class Policy(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))
                self.register_buffer("scale", torch.full((2,), 0.5))
                self.register_buffer("z", torch.zeros(2))

            def encode(self, x):
                self.h = torch.tanh(self.h + x * self.scale)
                self.z = self.h * 2
                return self.z

            def act(self, obs):
                return self.z * obs + self.scale

        model = Policy().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("enc", {"x": torch.ones(2)})
            annotate.module("enc", model)
            z = annotate.output_tensors(
                "enc", {"z": model.encode(x)}, export_with="jit"
            )
            _, obs = annotate.input_tensors("act", {"latent": z, "obs": torch.ones(2)})
            annotate.module("act", model)
            annotate.output_tensors("act", {"a": model.act(obs)}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True),
            {"enc": True, "act": True},
        )
        self.assertEqual(
            annotate.detected_pipeline["feedback_flow"], {"enc/h": ["enc/h"]}
        )
        eager = Policy().eval()
        manager = InferenceManager(
            os.path.join(self.TEST_GRAPH_NAME, f"{self.TEST_GRAPH_NAME}.yaml")
        )
        for step in range(3):
            x, obs = torch.tensor([step - 1.0, 2.0 * step]), torch.tensor([step, -1.0])
            eager.encode(x)
            torch.testing.assert_close(
                manager.run_policy({"enc/x": x, "act/obs": obs})["act/a"],
                eager.act(obs),
            )

    def test_buffer_set_to_an_input_keeps_how_it_was_made(self):
        """A buffer the forward sets to an input captured under inference_mode()
        or no_grad() holds a tensor made as the input was, and a buffer the
        forward keeps and outputs a view of holds a copy made as the buffer
        was, also when the session ends in that mode, so a reset in place and
        a backward still work after the session."""

        class Store(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                self.h = x
                return x * 2

        class KeptView(Store):
            def forward(self, x):
                return self.h.view(1, 2)

        for model_type, mode, grad in itertools.product(
            (Store, KeptView), (torch.inference_mode, torch.no_grad), (False, True)
        ):
            with self.subTest(model=model_type.__name__, mode=mode.__name__, grad=grad):
                model = model_type().eval()
                raw = torch.ones(2, requires_grad=grad)
                leapp.start(name=self.TEST_GRAPH_NAME)
                with mode():
                    try:
                        x = annotate.input_tensors("policy", {"x": raw})
                        annotate.module("policy", model)
                        annotate.output_tensors(
                            "policy", {"y": model(x)}, export_with="jit"
                        )
                    finally:
                        leapp.stop()
                self.assertFalse(model.h.is_inference())
                if model_type is Store:
                    self.assertEqual(model.h.requires_grad, raw.requires_grad)
                if model.h.requires_grad:
                    model.h.sum().backward()
                else:
                    model.h.zero_()

    def test_submodule_under_two_names_has_one_state(self):
        """A submodule stored under two names has one slot per buffer, so a
        buffer the forward reassigns is one feedback state, also when it is
        declared with state_tensors() under its first name."""

        class Accumulator(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))

            def forward(self, x):
                self.h = self.h + x
                return self.h * 2

        class Shared(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = Accumulator()
                self.right = self.left

            def forward(self, x):
                return self.left(x) + self.right(x)

        for declared in (False, True):
            with self.subTest(declared=declared):
                model = Shared().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    if declared:
                        annotate.state_tensors("policy", {"left_h": model.left.h})
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
                finally:
                    leapp.stop()
                self.assertEqual(
                    leapp.compile_graph(visualize=False, validate=True),
                    {"policy": True},
                )
                self.assertEqual(
                    annotate.detected_pipeline["feedback_flow"],
                    {"policy/left_h": ["policy/left_h"]},
                )
                if not declared:
                    _assert_plain(self, model.left.h)

    def test_buffer_adopting_a_declared_input_changed_in_place(self):
        """A buffer that took the input declared under its name before module()
        reads that input, so changing it in place does not raise, and a
        repeated output_tensors() after another buffer's in-place or state
        shape error raises that error again."""

        class Bump(nn.Module):
            def __init__(self, change):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))
                self.register_buffer("count", torch.zeros(2))
                self.change = change

            def forward(self, x):
                if self.change == "count":
                    self.count.add_(1)
                elif self.change == "shape":
                    self.count = self.count.repeat(2)
                    return x + self.h
                else:
                    self.h.add_(x)
                return x + self.h + self.count

        # (what the forward changes, the error of every output_tensors() call)
        cases = (
            ("h", None),
            ("count", "'count'.*changed in place"),
            ("shape", "update_state for 'count' has shape"),
        )
        for change, error in cases:
            with self.subTest(change=change):
                model = Bump(change).eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors(
                        "policy", {"x": torch.ones(2), "h": model.h.view(2)}
                    )[0]
                    annotate.module("policy", model)
                    y = model(x)
                    if error:
                        for _ in range(2):
                            with self.assertRaisesRegex(Exception, error):
                                annotate.output_tensors(
                                    "policy", {"y": y}, export_with="jit"
                                )
                    else:
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()
                if not error:
                    self.assertEqual(
                        leapp.compile_graph(visualize=False, validate=True),
                        {"policy": True},
                    )
                    self.assertEqual(
                        annotate.detected_pipeline["inputs"]["policy"], ["x", "h"]
                    )

    def test_buffer_holding_a_reentry_input(self):
        """A buffer set to a tensor declared on a re-entry pass, which has no
        graph value, is tracked as a plain buffer, whether another node or this
        node declared it before a second capture pass."""

        class Offset(nn.Module):
            def __init__(self, b):
                super().__init__()
                self.register_buffer("b", b)

            def forward(self, x):
                return x + self.b

        with self.subTest(node="other"):
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                for _ in range(2):
                    t = torch.zeros(3)
                    h = annotate.input_tensors("other", {"h": t})
                    annotate.output_tensors("other", {"o": h * 2}, export_with="jit")
                model = Offset(t).eval()
                x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                annotate.module("policy", model)
                annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
            finally:
                leapp.stop()
            self.assertEqual(
                leapp.compile_graph(visualize=False, validate=True),
                {"other": True, "policy": True},
            )

        with self.subTest(node="same"):
            model = Offset(torch.ones(3)).eval()
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                for step in range(3):
                    if step == 2:
                        annotate.nodes["policy"].reset_trace_state()
                    x = annotate.input_tensors("policy", {"x": torch.ones(3)})
                    annotate.module("policy", model)
                    fresh = torch.ones(3)
                    annotate.input_tensors(
                        "policy", {"b_in": fresh if step == 1 else model.b}
                    )
                    if step == 1:
                        model.b = fresh
                    annotate.output_tensors(
                        "policy", {"y": model(x)}, export_with="jit"
                    )
            finally:
                leapp.stop()
            self.assertEqual(
                leapp.compile_graph(visualize=False, validate=True), {"policy": True}
            )

    def test_buffer_read_only_for_a_state_changed_in_place_raises(self):
        """A buffer the graph reads only to compute a state, another buffer's or
        one the user updates, or to write into a buffer the user declared,
        counts as read, so changing it in place raises, and so does holding a
        value the node computed before module()."""

        class ViaBuffer(nn.Module):
            def __init__(self, write):
                super().__init__()
                self.register_buffer("h", torch.zeros(2))
                self.register_buffer("k", torch.ones(2))
                self.write = write

            def forward(self, x):
                self.h = self.h + self.k
                if self.write:
                    self.k.add_(1)
                return x * 2

        cases = itertools.product(("buffer", "user state"), ("in place", "computed"))
        for via, kind in cases:
            with self.subTest(via=via, kind=kind):
                leapp.start(name=self.TEST_GRAPH_NAME)
                try:
                    x = annotate.input_tensors("policy", {"x": torch.ones(2)})
                    model = ViaBuffer(kind == "in place").eval()
                    if kind == "computed":
                        model.k = x * 2
                    if via == "buffer":
                        annotate.module("policy", model, buffer_names=["h", "k"])
                        y = model(x)
                    else:
                        state = annotate.state_tensors("policy", {"s": torch.zeros(2)})
                        annotate.module("policy", model, buffer_names=["k"])
                        new_state = state + model.k
                        if kind == "in place":
                            model.k.add_(1)
                        annotate.update_state("policy", {"s": new_state})
                        y = x * 2
                    error = (
                        "changed in place" if kind == "in place" else "holds a value"
                    )
                    with self.assertRaisesRegex(RuntimeError, f"'k'.*{error}"):
                        annotate.output_tensors("policy", {"y": y}, export_with="jit")
                finally:
                    leapp.stop()

        class WriteDeclared(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("a", torch.zeros(2))
                self.register_buffer("k", torch.ones(2))

            def forward(self, x):
                self.a.add_(self.k)
                return x * 2

        model = WriteDeclared().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(2)})
            model.k = x * 2
            annotate.state_tensors("policy", {"a": model.a})
            annotate.module("policy", model)
            y = model(x)
            with self.assertRaisesRegex(RuntimeError, "'k'.*holds a value"):
                annotate.output_tensors("policy", {"y": y}, export_with="jit")
        finally:
            leapp.stop()

    def test_buffer_holding_a_computed_value_unread_captures(self):
        """A buffer holding a value the node computed before module() that the
        forward neither reads nor changes is not checked, since the export
        drops it."""

        class Skip(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("k", torch.ones(2))

            def forward(self, x):
                return x * 2

        model = Skip().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        try:
            x = annotate.input_tensors("policy", {"x": torch.ones(2)})
            model.k = x * 3
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"y": model(x)}, export_with="jit")
        finally:
            leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )

    def test_buffer_name_taken_by_a_state_tensor(self):
        """module() rejects a buffer whose name the node already uses for another
        state tensor, also a view of the buffer with another shape, since the
        forward would read that state instead of the buffer, and tracks it when
        that state is the buffer itself."""
        model = GRUModel().eval()
        for taken in (lambda: torch.ones(1, 8), lambda: model.h_state.view(8)):
            leapp.start(name=self.TEST_GRAPH_NAME)
            try:
                annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.state_tensors("policy", {"h_state": taken()})
                with self.assertRaisesRegex(ValueError, r"\['h_state'\]"):
                    annotate.module("policy", model)
            finally:
                leapp.stop()
            _assert_plain(self, model.h_state)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.state_tensors("policy", {"h_state": model.h_state})
        annotate.module("policy", model)
        annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        leapp.stop()
        self.assertEqual(
            leapp.compile_graph(visualize=False, validate=True), {"policy": True}
        )
        self.assertEqual(
            annotate.detected_pipeline["feedback_flow"],
            {"policy/h_state": ["policy/h_state"]},
        )

    def test_start_after_a_tracked_buffer_was_deleted(self):
        """A buffer deleted during the capture does not stop the next session
        from starting."""
        model = GRUModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", model)
        del model.h_state
        leapp.start(name=self.TEST_GRAPH_NAME)
        leapp.stop()
        self.assertNotIn("h_state", dict(model.named_buffers()))

    def test_stop_restores_buffers_without_output_tensors(self):
        """stop() safety net restores model buffers even if output_tensors() is never called."""
        model = GRUModel()
        model.eval()
        obs = torch.randn(1, 4)

        original_buffer_names = set(dict(model.named_buffers()).keys())

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})
        annotate.module("policy", model)
        action = model(obs_traced)

        # Intentionally skip output_tensors() — simulate user error or exception
        leapp.stop()

        # Buffers should still be restored by stop() safety net
        restored_buffer_names = set(dict(model.named_buffers()).keys())
        self.assertEqual(
            original_buffer_names,
            restored_buffer_names,
            "stop() should restore buffers even without output_tensors()",
        )

        # Verify they're real tensors, not TracedTensors
        from leapp.leapp_graph.datatypes import TracedTensor

        for name, buf in model.named_buffers():
            self.assertNotIsInstance(
                buf,
                TracedTensor,
                f"Buffer '{name}' should not be TracedTensor after stop()",
            )

    def test_untracked_buffer_set_with_a_tracked_one_shares_it_after_stop(self):
        """A buffer left out of buffer_names that the forward sets to the same
        new tensor as a tracked one holds the same plain tensor as that one
        after stop() without output_tensors()."""

        class Pair(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(3))
                self.register_buffer("k", torch.zeros(3))

            def forward(self, x):
                self.h = self.k = self.h + x
                return self.h * 2

        model = Pair().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("policy", {"x": torch.ones(3)})
        annotate.module("policy", model, buffer_names=["h"])
        model(x)
        leapp.stop()
        _assert_plain(self, model.h)
        self.assertIs(model.h, model.k)

    def test_model_functional_after_tracing(self):
        """Model produces correct output after tracing completes (buffers properly restored)."""
        model = GRUModel()
        model.eval()
        obs = torch.randn(1, 4)

        # Run once without tracing to get reference output
        with torch.no_grad():
            ref_output = model(obs.clone())

        # Reset model state
        model.h_state.zero_()

        # Trace
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})
        annotate.module("policy", model)
        action = model(obs_traced)
        annotate.output_tensors(
            "policy", {"action": action}, export_with="onnx-torchscript"
        )
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Reset state and run again — model should produce same result as pre-tracing
        model.h_state.zero_()
        with torch.no_grad():
            post_trace_output = model(obs.clone())

        self.assertTrue(
            torch.allclose(ref_output, post_trace_output, atol=1e-6),
            "Model should produce identical output before and after tracing",
        )

    def test_buffer_names_filter_nonexistent_name(self):
        """Filtering by a buffer name that doesn't exist -> no buffers tracked."""
        model = GRUModel()
        model.eval()
        obs = torch.randn(1, 4)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        # "nonexistent" doesn't match any buffer — tracker injects nothing
        annotate.module("policy", model, buffer_names=["nonexistent"])
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # No buffers were tracked -> 0 feedback
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=0,
        )


class TestNumpyStateAndBufferEquivalents(LEAPPFunctionalTestBase):
    """NumPy twins of the basic torch state_tensors and register_buffer tests.

    Same connectivity expectations as ``TestStateTensors.test_state_tensor_basic``
    and ``TestAnnotateTensor.test_annotate_register_buffer_with_inplace_assignment``,
    but with ``np.ndarray`` values and ``export_with=None``.
    """

    def test_state_tensor_basic(self):
        """Basic numpy state tensor: input -> computation -> updated state output."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        obs_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        obs = annotate.input_tensors("policy", {"observation": obs_value})

        initial_counter = np.array([5.0, -3.0, 0.5], dtype=np.float32)
        counter = annotate.state_tensors("policy", {"counter": initial_counter})

        new_counter = counter + obs
        returned_state = annotate.update_state("policy", {"counter": new_counter})
        self.assertIs(returned_state, new_counter)

        action = obs * 2.0
        annotate.output_tensors("policy", {"action": action}, export_with=None)

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=1,
        )
        self.verify_feedback_initial_values(
            {
                "policy/counter": torch.from_numpy(initial_counter),
            }
        )
        self.verify_safetensors_matches_feedback(annotate)

    def test_register_buffer_with_inplace_assignment(self):
        """register_buffer with numpy in-place assignment, matching the torch case."""

        class MockModule:
            def __init__(self):
                self.values = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            def run(self, traced_input):
                self.values = annotate.register_buffer(
                    "buffer_test", {"values": self.values}
                )
                self.values[:] = traced_input
                return self.values * 100.0

        module = MockModule()

        leapp.start(name=self.TEST_GRAPH_NAME)

        input_array = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        traced_input = annotate.input_tensors("buffer_test", {"input": input_array})

        result = module.run(traced_input)
        self.assertIsInstance(module.values, TracedNpArray)
        self.assertIs(module.values.context_obj, annotate.nodes["buffer_test"])

        annotate.output_tensors("buffer_test", {"result": result}, export_with=None)

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1, internal_connections=0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
