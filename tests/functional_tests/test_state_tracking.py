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

import contextlib
import functools
import os
import subprocess
import sys
import textwrap
import unittest
from unittest import mock
import numpy as np
import torch
from torch import nn
import leapp
from leapp import buffer_tracker
from leapp.leapp import _MANAGER as annotate
from leapp.leapp_graph.datatypes import TracedTensor, TracedNpArray
from .base import LEAPPFunctionalTestBase


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


def tied_to_capture(buf):
    """True for a TracedTensor or a view of one."""
    from leapp.leapp_graph.datatypes import is_traced_type

    return is_traced_type(buf) or (buf._base is not None and is_traced_type(buf._base))


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


class Scaler(nn.Module):
    """Model whose buffer has the same name as NoMutationModel's ``scale``."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.full((4,), 3.0))

    def forward(self, obs):
        return obs * self.scale


class Facade:
    """An object that is not an nn.Module and hands a model's buffers to module() as its own."""

    def __init__(self, model):
        self.model = model

    def named_buffers(self):
        return self.model.named_buffers()

    def register_buffer(self, name, tensor, persistent=True):
        self.model.register_buffer(name, tensor, persistent)

    def __call__(self, obs):
        return self.model(obs)


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

        # Restored values are plain tensors, not TracedTensors or views of them
        for name, buf in restored_buffers.items():
            self.assertFalse(tied_to_capture(buf), name)

    def test_buffers_plain_after_session(self):
        """After stop(), the model's buffers are plain tensors, not views of TracedTensors.

        torch.export of the model then works, and the buffers keep their memory
        and persistent flag. Also for a session that never reached
        output_tensors(), a session restarted with start(), a second session,
        and the model of an earlier module() call of the node, which is checked
        for plain buffers and run. A buffer that requires grad stays as
        before: autograd through it still reaches the parameter, and it stays
        tied to the capture.
        """

        def check_plain(model, keys):
            for name, buf in model.named_buffers(remove_duplicate=False):
                self.assertFalse(tied_to_capture(buf), name)
            self.assertEqual(sorted(model.state_dict()), keys)

        cases = (
            "traced",
            "earlier_module_call",
            "no_output_tensors",
            "restart",
            "restart_no_output_tensors",
            "twice",
            "buffer_names",
            "autograd_history",
        )
        for case in cases:
            with self.subTest(case=case):
                annotate.keep_buffers_tied = False
                model = NoMutationModel().eval()
                model.register_buffer("window", torch.ones(4), persistent=False)
                other = GRUModel(action_dim=4).eval()
                other.register_buffer("memory", torch.ones(4), persistent=False)
                grid = torch.empty_strided((4, 4), (1, 4))
                model.register_buffer("grid", grid.copy_(torch.arange(16.0).view(4, 4)))
                if case == "autograd_history":
                    model.register_buffer("pe", model.linear.weight.sum(0) * 2)
                    model.register_buffer("gain", torch.ones(4, requires_grad=True))
                keys = sorted(model.state_dict())
                other_keys = sorted(other.state_dict())
                untouched = model.bias
                pointers = {
                    n: b.data_ptr()
                    for n, b in model.named_buffers(remove_duplicate=False)
                }
                values = {
                    n: b.detach().clone()
                    for n, b in model.named_buffers(remove_duplicate=False)
                }
                for _ in range(2 if case == "twice" else 1):
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                    if case == "earlier_module_call":
                        annotate.module("policy", other)
                        obs = other(obs)
                    names = ["scale"] if case == "buffer_names" else None
                    annotate.module("policy", model, buffer_names=names)
                    action = model(obs)
                    if not case.endswith("no_output_tensors"):
                        annotate.output_tensors(
                            "policy", {"action": action}, export_with="jit"
                        )
                    if not case.startswith("restart"):
                        leapp.stop()
                if case.startswith("restart"):
                    # start() resets a session that was never stopped.
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    check_plain(model, keys)
                    obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"action": model(obs)}, export_with="jit"
                    )
                    leapp.stop()
                if case == "autograd_history":
                    (grad,) = torch.autograd.grad(model.pe.sum(), model.linear.weight)
                    self.assertTrue(torch.equal(grad, torch.full_like(grad, 2.0)))
                    self.assertTrue(model.gain.requires_grad)
                    torch.autograd.grad((model.gain * 3).sum(), model.gain)
                    # A buffer that requires grad stays tied to the capture, as before.
                    for name, buf in model.named_buffers():
                        self.assertEqual(tied_to_capture(buf), buf.requires_grad, name)
                    self.assertEqual(sorted(model.state_dict()), keys)
                    continue
                check_plain(model, keys)
                # The buffers keep their memory, strides and values.
                for name, buf in model.named_buffers(remove_duplicate=False):
                    self.assertEqual(buf.data_ptr(), pointers[name], name)
                    self.assertEqual(buf.stride(), values[name].stride(), name)
                    self.assertTrue(torch.equal(buf, values[name]), name)
                torch.export.export(model, (torch.randn(1, 4),))
                if case == "buffer_names":
                    # A buffer the session never tracked keeps its tensor object.
                    self.assertIs(model.bias, untouched)
                if case == "earlier_module_call":
                    check_plain(other, other_keys)
                    other(torch.randn(1, 4))

    def test_buffers_restored_for_a_finished_and_an_unfinished_node(self):
        """stop() restores the buffers of a finished node and of a node that never finished.

        One model goes into a finished node and then, with other buffers
        tracked, into a node that never reaches output_tensors(). After stop()
        its other buffers are plain, its buffer that requires grad is no
        TracedTensor and autograd through it works, and the next session
        captures the model.
        """
        model = NoMutationModel().eval()
        model.register_buffer("pe", model.linear.weight.sum(0) * 2)
        leapp.start(name=self.TEST_GRAPH_NAME)
        for node, finish in (("first", True), ("second", False)):
            obs = annotate.input_tensors(node, {"obs": torch.randn(1, 4)})
            annotate.module(
                node, model, buffer_names=["bias", "pe"] if finish else ["scale"]
            )
            action = model(obs)
            if finish:
                annotate.output_tensors(node, {"action": action}, export_with="jit")
        leapp.stop()

        for buf in (model.scale, model.bias):
            self.assertFalse(tied_to_capture(buf))
        self.assertNotIsInstance(model.pe, TracedTensor)
        self.assertNotIsInstance(model(torch.randn(1, 4)), TracedTensor)
        (grad,) = torch.autograd.grad(model.pe.sum(), model.linear.weight)
        self.assertTrue(torch.equal(grad, torch.full_like(grad, 2.0)))

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", model)
        annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        leapp.stop()

    def test_buffers_restored_with_model_overrides(self):
        """Models that override register_buffer() or __setattr__ work as before.

        A register_buffer() override is called with only the name and the
        tensor when buffers are put back, also one that takes more. Such code
        can change buffers unseen, so the buffers stay tied.
        """

        class LegacyModel(NoMutationModel):
            def register_buffer(self, name, tensor):
                self.__dict__.setdefault("registered", []).append(name)
                return super().register_buffer(name, tensor)

        class ForwardingModel(NoMutationModel):
            def register_buffer(self, name, tensor, *args, **kwargs):
                self.__dict__.setdefault("extra", []).append((args, kwargs))
                return super().register_buffer(name, tensor, *args, **kwargs)

        class WrappedLegacyModel(NoMutationModel):
            @functools.wraps(nn.Module.register_buffer)
            def register_buffer(self, name, tensor):
                return super().register_buffer(name, tensor)

            def __setattr__(self, name, value):
                # Writes existing buffers directly, so torch's own check, which
                # reads the wrapped signature, does not run.
                buffers = self.__dict__.get("_buffers")
                if buffers is not None and name in buffers:
                    buffers[name] = value
                else:
                    super().__setattr__(name, value)

        class InPlaceSetattrModel(NoMutationModel):
            def __setattr__(self, name, value):
                current = self.__dict__.get("_buffers", {}).get(name)
                if current is not None and current is not value:
                    with torch.no_grad():
                        current.copy_(value)
                else:
                    super().__setattr__(name, value)

        for model_class in (
            LegacyModel,
            ForwardingModel,
            WrappedLegacyModel,
            InPlaceSetattrModel,
        ):
            with self.subTest(model=model_class.__name__):
                annotate.keep_buffers_tied = False
                model = model_class().eval()
                keys = sorted(model.state_dict())
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", model)
                model.__dict__["registered"] = []
                model.__dict__["extra"] = []
                annotate.output_tensors(
                    "policy", {"action": model(obs)}, export_with="jit"
                )
                leapp.stop()

                for name, buf in model.named_buffers(remove_duplicate=False):
                    self.assertTrue(tied_to_capture(buf), name)
                self.assertEqual(sorted(model.state_dict()), keys)
                if model_class is LegacyModel:
                    self.assertIn("scale", model.registered)
                if model_class is ForwardingModel:
                    self.assertEqual(set(map(repr, model.extra)), {"((), {})"})

    def test_buffers_kept_in_registries_capture_again(self):
        """A model or hook that keeps its buffers through register_buffer() captures again.

        Its forward reads the buffer from its own registry. The same model then
        works in a second node and in a second session, as before, and after
        the session the registry holds the model's own buffer, still tied.
        """

        class RegistryModel(NoMutationModel):
            def register_buffer(self, name, tensor, persistent=True):
                self.__dict__.setdefault("registry", {})[name] = tensor
                return super().register_buffer(name, tensor, persistent)

            def forward(self, obs):
                return self.linear(obs * self.registry["scale"])

        registry = {}

        def hook(module, name, buffer):
            registry[(id(module), name)] = buffer

        class HookModel(NoMutationModel):
            def forward(self, obs):
                return self.linear(obs * registry[(id(self), "scale")])

        for model_class in (RegistryModel, HookModel):
            with self.subTest(model=model_class.__name__):
                annotate.keep_buffers_tied = False
                handle = None
                if model_class is HookModel:
                    handle = torch.nn.modules.module.register_module_buffer_registration_hook(
                        hook
                    )
                try:
                    model = model_class().eval()
                    # module() only in the first node of the first session.
                    for nodes in (("first", "second"), ("policy",)):
                        leapp.start(name=self.TEST_GRAPH_NAME)
                        for node in nodes:
                            obs = annotate.input_tensors(
                                node, {"obs": torch.randn(1, 4)}
                            )
                            if node == "first":
                                annotate.module(node, model)
                            annotate.output_tensors(
                                node, {"action": model(obs)}, export_with="jit"
                            )
                        leapp.stop()
                finally:
                    if handle is not None:
                        handle.remove()
                if model_class is RegistryModel:
                    self.assertIs(model.registry["scale"], model.scale)
                else:
                    self.assertIs(registry[(id(model), "scale")], model.scale)
                self.assertTrue(tied_to_capture(model.scale))

    def test_module_on_later_passes_adds_no_tracker(self):
        """module() on passes after the node is captured keeps one tracker per capture.

        Otherwise the list would grow by one tracker on every pass.
        """
        model = NoMutationModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(3):
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        self.assertEqual(len(annotate.buffer_trackers), 1)
        leapp.stop()
        self.assertFalse(tied_to_capture(model.scale))

    def test_non_mutated_buffers_baked_as_constants(self):
        """Non-mutated buffers should be baked into the model, not dynamic inputs.

        Verifies that the exported model produces correct output using the
        baked buffer values (scale=2, bias=1) without needing them as inputs.
        """

        class ScaleBiasModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.tensor([2.0, 2.0, 2.0, 2.0]))
                self.register_buffer("bias", torch.tensor([1.0, 1.0, 1.0, 1.0]))

            def forward(self, obs):
                return obs * self.scale + self.bias

        model = ScaleBiasModel()
        model.eval()
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        result = model(obs_traced)

        annotate.output_tensors("policy", {"result": result}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Only 1 input (obs) — scale and bias baked as constants
        self.verify_num_connections(
            annotate,
            nodes=1,
            inputs=1,
            outputs=1,
            internal_connections=0,
            feedback_connections=0,
        )

        # Load exported model and verify it produces correct values
        # Expected: obs * 2.0 + 1.0 = [3.0, 5.0, 7.0, 9.0]
        expected = obs * 2.0 + 1.0
        self.verify_single_torchscript_model_expected_value(
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

    def test_failed_buffer_restore_does_not_block_next_session(self):
        """A tracker that fails at stop() is logged and does not make a later start() fail."""
        from leapp.buffer_tracker import BufferTracker

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", NoMutationModel())
        with mock.patch.object(
            BufferTracker, "restore", side_effect=RuntimeError("restore failed")
        ):
            with self.assertLogs("leapp", level="ERROR") as logs:
                leapp.stop()
            self.assertIn("restore failed", "\n".join(logs.output))
            # The failing tracker is not tried again.
            self.assertEqual(annotate.buffer_trackers, [])
            leapp.start(name=self.TEST_GRAPH_NAME)

        for index in range(2):
            model = NoMutationModel().eval()
            if index:
                leapp.start(name=self.TEST_GRAPH_NAME)
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
            leapp.stop()
            self.assertFalse(tied_to_capture(model.scale))

    def test_failed_buffer_restore_does_not_skip_other_trackers(self):
        """stop() still releases the other trackers' buffers and removes LEAPP's patches."""
        models = [NoMutationModel().eval() for _ in range(2)]
        leapp.start(name=self.TEST_GRAPH_NAME)
        for node, model in zip(("first", "second"), models):
            obs = annotate.input_tensors(node, {"obs": torch.randn(1, 4)})
            annotate.module(node, model)
            model(obs)
        first = annotate.buffer_trackers[0]
        with (
            mock.patch.object(
                first, "restore", side_effect=RuntimeError("restore failed")
            ),
            self.assertLogs("leapp", level="ERROR"),
        ):
            leapp.stop()
        self.assertFalse(tied_to_capture(models[1].scale))
        self.assertFalse(annotate.patcher.installed)

    def test_session_end_keeps_a_buffer_a_later_node_replaced(self):
        """A buffer a later node deletes and replaces with a plain attribute stays replaced."""

        class ReplacingModel(NoMutationModel):
            replace = False

            def forward(self, obs):
                action = super().forward(obs)
                if self.replace:
                    del self.scale
                    self.scale = action[0]
                return action

        model = ReplacingModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        for node in ("first", "second"):
            model.replace = node == "second"
            obs = annotate.input_tensors(node, {"obs": torch.randn(1, 4)})
            if node == "first":
                annotate.module(node, model)
            annotate.output_tensors(node, {"action": model(obs)}, export_with="jit")
        with self.assertNoLogs("leapp", level="ERROR"):
            leapp.stop()
        self.assertNotIn("scale", model._buffers)
        self.assertFalse(tied_to_capture(model.bias))

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

        # Verify they're plain tensors, not TracedTensors or views of them
        for name, buf in model.named_buffers():
            self.assertFalse(tied_to_capture(buf), name)

    def test_wrapper_listing_another_models_buffers_before_it_is_plain(self):
        """An nn.Module that lists another model's buffers as its own, passed to module() before that model, logs no error."""

        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def named_buffers(self, *args, **kwargs):
                return self.model.named_buffers()

            def forward(self, obs):
                return self.model(obs)

        model = GRUModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", Wrapper(model))
        annotate.module("policy", model)
        annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        with self.assertNoLogs("leapp", level="ERROR"):
            leapp.stop()
        self.assertFalse(tied_to_capture(model.h_state))

    def test_stop_restores_buffers_of_non_module_model(self):
        """stop() restores the buffers of a model passed through an object that is not an nn.Module."""

        class Wrapper:
            def __init__(self, model):
                self.model = model

            def named_buffers(self):
                return self.model.named_buffers(prefix="model")

            def __call__(self, obs):
                return self.model(obs)

        model = NoMutationModel().eval()
        wrapper = Wrapper(model)
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", wrapper)
        wrapper(obs)
        leapp.stop()

        self.assertFalse(tied_to_capture(model.scale))
        self.assertNotIsInstance(model(torch.randn(1, 4)), TracedTensor)

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", wrapper)
        annotate.output_tensors("policy", {"action": wrapper(obs)}, export_with="jit")
        leapp.stop()

    def test_session_end_with_object_that_owns_buffers(self):
        """An object that is not an nn.Module but registers buffers itself works as before.

        Its buffers are put back as before but not made plain.
        """

        for finish in (True, False):
            with self.subTest(finish=finish):
                annotate.keep_buffers_tied = False
                model = NoMutationModel().eval()
                facade = Facade(model)
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", facade)
                action = facade(obs)
                if finish:
                    annotate.output_tensors(
                        "policy", {"action": action}, export_with="jit"
                    )
                with self.assertNoLogs("leapp", level="ERROR"):
                    leapp.stop()
                self.assertNotIsInstance(model(torch.randn(1, 4)), TracedTensor)
                self.assertTrue(tied_to_capture(model.scale))

    def test_object_that_forwards_buffers_keeps_the_session_tied(self):
        """A session that also tracks a model through an object that forwards its buffers makes no buffer plain.

        Such an object, an nn.Module or not, can write an older state back
        into the model, so the model's buffer stays tied, as before.
        """

        class ModuleFacade(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def named_buffers(self, *args, **kwargs):
                return self.model.named_buffers()

            def register_buffer(self, name, tensor, persistent=True):
                self.model.register_buffer(name, tensor, persistent)

            def forward(self, obs):
                return self.model(obs)

        class Forwarder:
            def __init__(self, model):
                self.__dict__["model"] = model

            def __getattr__(self, name):
                return getattr(self.__dict__["model"], name)

            def __call__(self, obs):
                return self.__dict__["model"](obs)

        for make in (Facade, ModuleFacade, Forwarder):
            with self.subTest(facade=make.__name__):
                annotate.keep_buffers_tied = False
                model = GRUModel().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("first", {"obs": torch.randn(1, 4)})
                annotate.module("first", model)
                annotate.output_tensors(
                    "first", {"action": model(obs)}, export_with="jit"
                )
                facade = make(model)
                obs = annotate.input_tensors("second", {"obs": torch.randn(1, 4)})
                annotate.module("second", facade)
                annotate.output_tensors(
                    "second", {"action": facade(obs)}, export_with="jit"
                )
                leapp.stop()
                self.assertTrue(tied_to_capture(model.h_state))

    def test_state_written_back_by_an_object_stays_tied_later(self):
        """An older state that an object which is not an nn.Module wrote back stays tied in a later session."""

        for untracked_node in (False, True):
            with self.subTest(untracked_node=untracked_node):
                annotate.keep_buffers_tied = False
                model = GRUModel().eval()
                facade = Facade(model)
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("first", {"obs": torch.randn(1, 4)})
                annotate.module("first", facade)
                annotate.output_tensors(
                    "first", {"action": facade(obs)}, export_with="jit"
                )
                leapp.stop()

                leapp.start(name=self.TEST_GRAPH_NAME)
                if untracked_node:
                    # The model runs in a node without module() first, so its
                    # state comes from that node.
                    obs = annotate.input_tensors("plain", {"obs": torch.randn(1, 4)})
                    annotate.output_tensors(
                        "plain", {"action": model(obs)}, export_with="jit"
                    )
                obs = annotate.input_tensors("again", {"obs": torch.randn(1, 4)})
                annotate.module("again", model)
                annotate.output_tensors(
                    "again", {"action": model(obs)}, export_with="jit"
                )
                leapp.stop()
                self.assertTrue(tied_to_capture(model.h_state))

    def test_session_end_keeps_forward_state_behind_an_object(self):
        """A buffer behind an object that is not an nn.Module keeps the state its forward assigned.

        The object still holds the tensor from before the forward pass. It is
        not put back when the object went into an earlier module() call of
        the node, or when start() resets the session.
        """

        def raw(tensor):
            return getattr(tensor, "tensor", tensor)

        for end in ("stop", "start"):
            with self.subTest(end=end):
                annotate.keep_buffers_tied = False
                model = GRUModel().eval()
                facade = Facade(model)
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", facade)
                action = facade(obs)
                assigned = raw(model.h_state).detach().clone()
                if end == "stop":
                    annotate.module("policy", NoBufferModel().eval())
                    annotate.output_tensors(
                        "policy", {"action": action}, export_with="jit"
                    )
                    with self.assertNoLogs("leapp", level="ERROR"):
                        leapp.stop()
                else:
                    leapp.start(name=self.TEST_GRAPH_NAME)
                self.assertTrue(torch.equal(raw(model.h_state), assigned))
                if end == "start":
                    leapp.stop()

    def test_buffer_replaced_by_an_attribute_in_the_capture_pass(self):
        """A forward that replaces a buffer with a plain attribute fails at output_tensors(), and stop() still ends the session."""

        class ReplacingModel(NoMutationModel):
            def forward(self, obs):
                action = super().forward(obs)
                del self.scale
                self.scale = action[0]
                return action

        model = ReplacingModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", model)
        action = model(obs)
        with self.assertRaises(KeyError):
            annotate.output_tensors("policy", {"action": action}, export_with="jit")
        with self.assertNoLogs("leapp", level="ERROR"):
            leapp.stop()

    def test_object_with_numpy_state(self):
        """An object that is not an nn.Module and keeps NumPy state captures as before."""

        class NumpyPolicy:
            def __init__(self):
                self.h = np.zeros((1, 4), dtype=np.float32)

            def named_buffers(self):
                return [("h", self.h)]

            def register_buffer(self, name, value):
                setattr(self, name, value)

            def __call__(self, obs):
                self.h = np.tanh(obs + self.h)
                return self.h

        policy = NumpyPolicy()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors(
            "policy", {"obs": np.full((1, 4), 0.1, dtype=np.float32)}
        )
        annotate.module("policy", policy)
        annotate.output_tensors("policy", {"action": policy(obs)}, export_with="jit")
        with self.assertNoLogs("leapp", level="ERROR"):
            leapp.stop()

    def test_object_forwarding_a_non_persistent_buffer(self):
        """An object that forwards attribute reads to its model works as before.

        Its register_buffer() takes no persistent argument and is called as
        before, also for a non-persistent buffer of the model.
        """

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.register_buffer("scale", torch.ones(4), persistent=False)

            def forward(self, obs):
                return self.linear(obs * self.scale)

        class Forwarder:
            def __init__(self, model):
                self.__dict__["model"] = model

            def __getattr__(self, name):
                return getattr(self.__dict__["model"], name)

            def register_buffer(self, name, tensor):
                self.__dict__.setdefault("calls", []).append(name)
                self.__dict__["model"].register_buffer(name, tensor)

            def __call__(self, obs):
                return self.__dict__["model"](obs)

        for finish in (True, False):
            with self.subTest(finish=finish):
                annotate.keep_buffers_tied = False
                model = Model().eval()
                forwarder = Forwarder(model)
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", forwarder)
                action = forwarder(obs)
                if finish:
                    annotate.output_tensors(
                        "policy", {"action": action}, export_with="jit"
                    )
                with self.assertNoLogs("leapp", level="ERROR"):
                    leapp.stop()
                self.assertEqual(forwarder.__dict__.get("calls"), ["scale"])

    def test_same_buffer_name_in_two_models_of_one_node(self):
        """Buffers stay tied in a session where two models of a node share a buffer name.

        Both buffers get one TracedTensor in the node, so the second model
        uses the first model's memory, and state the capture computes can be
        wrong. The buffers stay tied to the capture, also after a later
        session captures the second model alone, so torch.export of the
        models can fail instead of exporting those values.
        """

        for finish, view in ((True, False), (False, False), (True, True)):
            with self.subTest(finish=finish, view=view):
                annotate.keep_buffers_tied = False
                first, second = NoMutationModel().eval(), Scaler().eval()
                other = NoMutationModel().eval()
                if view:
                    # Same memory as the first model's buffer, as another tensor.
                    second.scale = first.scale[:]
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("other", {"obs": torch.randn(1, 4)})
                annotate.module("other", other)
                annotate.output_tensors(
                    "other", {"action": other(obs)}, export_with="jit"
                )
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", first)
                hidden = first(obs)
                annotate.module("policy", second)
                action = second(hidden)
                annotate.module("policy", NoBufferModel().eval())
                if finish:
                    annotate.output_tensors(
                        "policy", {"action": action}, export_with="jit"
                    )
                with self.assertNoLogs("leapp", level="ERROR"):
                    leapp.stop()
                for buf in (first.scale, first.bias, second.scale, other.scale):
                    self.assertTrue(tied_to_capture(buf))
                if not finish:
                    continue

                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("again", {"obs": torch.randn(1, 4)})
                annotate.module("again", second)
                annotate.output_tensors(
                    "again", {"action": second(obs)}, export_with="jit"
                )
                leapp.stop()
                self.assertTrue(tied_to_capture(second.scale))

    def test_recurrent_model_tracked_twice_in_a_node_stays_tied(self):
        """A recurrent model passed to module() twice in a node, with a forward between, stays tied.

        The second call gets the node's TracedTensor for the state the forward
        reassigned, which looks like another buffer with the same state name.
        """
        model = GRUModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", model)
        model(obs)
        annotate.module("policy", model)
        annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        leapp.stop()
        self.assertTrue(tied_to_capture(model.h_state))

    def test_state_handed_to_another_model_of_the_node_stays_tied(self):
        """A model given another model's state of the node, under the same buffer name, stays tied."""
        encoder, decoder = GRUModel().eval(), GRUModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", encoder)
        encoder(obs)
        decoder.h_state = encoder.h_state
        annotate.module("policy", decoder)
        action = decoder(obs)
        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()
        self.assertTrue(tied_to_capture(decoder.h_state))

    def test_same_buffer_name_then_capture_again_fails(self):
        """After a session with a shared state name, capturing the model again fails as before.

        The node never finished, and the session ends with start(), whether or
        not the clashing module() call is the node's latest, or with stop()
        where it is not.
        """

        for end, clash_is_latest in (
            ("start", True),
            ("start", False),
            ("stop", False),
        ):
            with self.subTest(end=end, clash_is_latest=clash_is_latest):
                annotate.keep_buffers_tied = False
                first, second = NoMutationModel().eval(), Scaler().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", first)
                hidden = first(obs)
                annotate.module("policy", second)
                second(hidden)
                if not clash_is_latest:
                    annotate.module("policy", NoBufferModel().eval())
                if end == "stop":
                    leapp.stop()
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("again", {"obs": torch.randn(1, 4)})
                with self.assertRaises(Exception):
                    annotate.module("again", second)
                leapp.stop()

    def test_buffer_assigned_by_the_user_after_output_tensors_is_kept(self):
        """A plain tensor the user assigns to a buffer after output_tensors() is kept by stop()."""
        model = NoMutationModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", model)
        annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        user_tensor = torch.full((4,), 2.0)
        model.scale = user_tensor
        leapp.stop()
        self.assertIs(model.scale, user_tensor)

    def test_buffer_a_node_reassigned_before_module_is_plain(self):
        """A buffer an earlier node of the session reassigned, then tracked by module(), is plain."""
        model = GRUModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        for node in ("first", "second"):
            obs = annotate.input_tensors(node, {"obs": torch.randn(1, 4)})
            if node == "second":
                annotate.module(node, model)
            annotate.output_tensors(node, {"action": model(obs)}, export_with="jit")
        leapp.stop()
        self.assertFalse(tied_to_capture(model.h_state))

    def test_shared_name_buffer_stays_tied_in_a_later_session(self):
        """A buffer that got another model's TracedTensor stays tied when a later session captures its model.

        The later session passes the model to module() once or twice, or
        first passes the buffer to state_tensors(), or stores what
        state_tensors() returns for it back into the model.
        """

        for finish, calls, before in (
            (True, 1, None),
            (False, 1, None),
            (True, 2, None),
            (True, 1, "promote"),
            (True, 1, "redeclare"),
        ):
            with self.subTest(finish=finish, calls=calls, before=before):
                annotate.keep_buffers_tied = False
                first, second = NoMutationModel().eval(), Scaler().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", first)
                hidden = first(obs)
                annotate.module("policy", second)
                action = second(hidden)
                if before == "redeclare":
                    annotate.module("policy", NoBufferModel().eval())
                if finish:
                    annotate.output_tensors(
                        "policy", {"action": action}, export_with="jit"
                    )
                leapp.stop()

                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("again", {"obs": torch.randn(1, 4)})
                if before == "promote":
                    annotate.state_tensors("again", {"scale": second.scale})
                elif before == "redeclare":
                    second.scale = annotate.state_tensors(
                        "again", {"scale": second.scale}
                    )
                for _ in range(calls):
                    annotate.module("again", second)
                annotate.output_tensors(
                    "again", {"action": second(obs)}, export_with="jit"
                )
                leapp.stop()
                self.assertTrue(tied_to_capture(second.scale))

    def test_recurrent_state_after_a_shared_name_stays_tied_later(self):
        """State a model computes in a later session from a clashed buffer stays tied."""
        actor, critic = GRUModel().eval(), GRUModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", actor)
        actor(obs)
        annotate.module("policy", critic)
        action = critic(obs)
        annotate.module("policy", NoBufferModel().eval())
        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()

        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("again", {"obs": torch.randn(1, 4)})
        annotate.module("again", critic)
        annotate.output_tensors("again", {"action": critic(obs)}, export_with="jit")
        leapp.stop()
        self.assertTrue(tied_to_capture(critic.h_state))

    def test_recurrent_state_that_required_grad_before_capture_is_plain(self):
        """State that required grad before the capture and that the forward replaced is plain, in every session."""
        model = GRUModel()
        model(torch.randn(1, 4))
        self.assertTrue(model.h_state.requires_grad)
        model.eval()
        for _ in range(2):
            leapp.start(name=self.TEST_GRAPH_NAME)
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
            leapp.stop()
            self.assertFalse(tied_to_capture(model.h_state))

    def test_every_session_after_a_clash_keeps_buffers_tied(self):
        """After a session with a shared state name, later sessions make no buffer plain.

        That covers a new model whose state comes from a clashed model, which
        the later session passes to module() or not, and a model that has
        nothing to do with the clash. A warning says so at each session end.
        """

        class Accumulator(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(1, 4))

            def forward(self, x):
                self.h = self.h + x
                return self.h

        for track in (True, False):
            with self.subTest(track=track):
                annotate.keep_buffers_tied = False
                model = NoMutationModel().eval()
                for session in ("alone", "clash", "after"):
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                    if session == "clash":
                        first, second = NoMutationModel().eval(), Scaler().eval()
                        annotate.module("policy", first)
                        annotate.module("policy", second)
                        action = second(first(obs))
                    elif session == "after":
                        recurrent = Accumulator().eval()
                        if track:
                            annotate.module("policy", second)
                        annotate.module("policy", recurrent)
                        annotate.module("policy", model)
                        action = model(recurrent(second(obs)))
                    else:
                        annotate.module("policy", model)
                        action = model(obs)
                    annotate.output_tensors(
                        "policy", {"action": action}, export_with="jit"
                    )
                    if session == "alone":
                        leapp.stop()
                    else:
                        with self.assertLogs("leapp", level="WARNING") as logs:
                            leapp.stop()
                        self.assertIn("rest of this process", "\n".join(logs.output))
                    tied = session == "after"
                    self.assertEqual(tied_to_capture(model.scale), tied, session)
                self.assertTrue(tied_to_capture(recurrent.h))

    def test_model_code_that_can_change_buffers_keeps_them_tied(self):
        """A __setattr__, __getattr__, property or register_buffer() override, or a buffer registration hook, keeps buffers tied.

        The __setattr__ writes another model's buffer, also one whose node has
        already put it back, or clamps; a __getattr__ or a property halves
        the buffer it reads; the register_buffer() override, also
        another model's method set on the model, writes a model that module()
        never saw; a hook changes the dtype, or writes another model's buffer
        while module() or output_tensors() runs; an override set after
        module() or output_tensors() writes another model's buffer; a
        named_buffers() lists another model's buffer under the model's own
        buffer name, also one registered as None; the model's buffer dict or
        non-persistent set, or a replaced nn.Module.register_buffer or
        __setattr__, also one replaced before LEAPP was imported, writes
        another model's buffer.
        """

        class Partner(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("g", torch.full((1, 4), 7.0))

            def forward(self, x):
                return x * self.g

        class Mirror(nn.Module):
            def __init__(self, partner):
                super().__init__()
                self.register_buffer("h", torch.full((1, 4), 2.0))
                object.__setattr__(self, "partner", partner)

            def __setattr__(self, name, value):
                if name == "h" and "partner" in self.__dict__:
                    self.partner.g = value
                super().__setattr__(name, value)

            def forward(self, x):
                return x + self.h

        class Clamp(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.full((1, 4), 5.0))

            def __setattr__(self, name, value):
                if name == "h":
                    value = value.clamp(-1.0, 1.0)
                super().__setattr__(name, value)

            def forward(self, x):
                return x + self.h

        class Plain(Clamp):
            __setattr__ = nn.Module.__setattr__

        class Halving(Plain):
            def __getattr__(self, name):
                value = super().__getattr__(name)
                return value * 0.5 if name == "h" else value

        class HalvingProperty(Plain):
            pass

        class Lister(Partner):
            def named_buffers(self, *args, **kwargs):
                yield "g", self.partner.g

        class NoneLister(Lister):
            def __init__(self):
                nn.Module.__init__(self)
                self.register_buffer("g", None)

            def forward(self, x):
                return x + 1

        class MirrorDict(dict):
            def __setitem__(self, name, value):
                mirror_hook(None, name, value)
                super().__setitem__(name, value)

        class MirrorSet(set):
            def discard(self, name):
                mirror_hook(None, name, model._buffers[name])
                super().discard(name)

        class RegisterMirror(Plain):
            def register_buffer(self, name, tensor, persistent=True):
                if name == "h" and "partner" in self.__dict__:
                    self.partner.g = tensor
                return super().register_buffer(name, tensor, persistent)

        def cast(module, name, buffer):
            return buffer.half()

        def mirror_hook(module, name, buffer):
            if name == "h":
                partner.g = buffer

        def mirror_register(name, tensor, persistent=True):
            mirror_hook(model, name, tensor)
            return nn.Module.register_buffer(model, name, tensor, persistent)

        cases = (
            "mirror",
            "restored",
            "clamp",
            "halving",
            "register",
            "bound",
            "hook",
            "early_hook",
            "late_hook",
            "late_override",
            "end_override",
            "lister",
            "lister_none",
            "buffer_dict",
            "persistent_set",
            "patched",
            "setattr_before",
            "property",
        )
        for case in cases:
            with self.subTest(case=case), contextlib.ExitStack() as patches:
                annotate.keep_buffers_tied = False
                partner = Partner().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                if case == "restored":
                    # The partner's node has already put its buffer back.
                    obs = annotate.input_tensors("first", {"obs": torch.ones(1, 4)})
                    annotate.module("first", partner)
                    annotate.output_tensors(
                        "first", {"action": partner(obs)}, export_with="jit"
                    )
                obs = annotate.input_tensors("policy", {"obs": torch.ones(1, 4)})
                if case in ("mirror", "restored"):
                    model = Mirror(partner).eval()
                if case == "mirror":
                    annotate.module("policy", partner)
                    annotate.module("policy", model)
                    action = model(partner(obs))
                elif case == "early_hook":
                    # Installed for module() only, then removed.
                    model = Plain().eval()
                    annotate.module("policy", partner)
                    hook = nn.modules.module.register_module_buffer_registration_hook(
                        mirror_hook
                    )
                    try:
                        annotate.module("policy", model)
                    finally:
                        hook.remove()
                    action = model(partner(obs))
                elif case in ("lister", "lister_none"):
                    model = (Lister if case == "lister" else NoneLister)().eval()
                    object.__setattr__(model, "partner", partner)
                    annotate.module("policy", partner)
                    annotate.module("policy", model)
                    action = model(partner(obs))
                elif case == "patched":
                    model = Plain().eval()
                    original = nn.Module.register_buffer

                    def patched(module, name, tensor, persistent=True):
                        mirror_hook(module, name, tensor)
                        return original(module, name, tensor, persistent)

                    patches.enter_context(
                        mock.patch.object(nn.Module, "register_buffer", patched)
                    )
                    annotate.module("policy", partner)
                    annotate.module("policy", model)
                    action = model(partner(obs))
                elif case == "setattr_before":
                    model = Plain().eval()

                    def mirror_setattr(module, name, value):
                        mirror_hook(module, name, value)
                        nn.Module.__setattr__(module, name, value)

                    # As if nn.Module.__setattr__ was replaced before LEAPP
                    # was imported.
                    patches.enter_context(
                        mock.patch.object(Plain, "__setattr__", mirror_setattr)
                    )
                    patches.enter_context(
                        mock.patch.object(
                            buffer_tracker,
                            "_TORCH_SETATTRS",
                            (*buffer_tracker._TORCH_SETATTRS, mirror_setattr),
                        )
                    )
                    annotate.module("policy", partner)
                    annotate.module("policy", model)
                    action = model(partner(obs))
                elif case in ("buffer_dict", "persistent_set"):
                    model = Plain().eval()
                    if case == "buffer_dict":
                        model.__dict__["_buffers"] = MirrorDict(model._buffers)
                    else:
                        model.__dict__["_non_persistent_buffers_set"] = MirrorSet()
                    annotate.module("policy", partner)
                    annotate.module("policy", model)
                    action = model(partner(obs))
                elif case in ("late_hook", "late_override", "end_override"):
                    model = Plain().eval()
                    annotate.module("policy", partner)
                    annotate.module("policy", model)
                    action = model(partner(obs))
                elif case == "bound":
                    model = Plain().eval()
                    model.register_buffer = partner.register_buffer
                    annotate.module("policy", model)
                    action = model(obs)
                elif case == "restored":
                    annotate.module("policy", model)
                    action = model(obs)
                else:
                    make = {
                        "clamp": Clamp,
                        "halving": Halving,
                        "property": HalvingProperty,
                        "register": RegisterMirror,
                        "hook": Plain,
                    }
                    model = make[case]().eval()
                    if case == "property":
                        # Set after register_buffer(), which refuses a name
                        # the class already has.
                        HalvingProperty.h = property(
                            lambda self: self._buffers["h"] * 0.5
                        )
                    object.__setattr__(model, "partner", partner)
                    annotate.module("policy", model)
                    action = model(obs)
                if case == "late_override":
                    model.register_buffer = mirror_register
                if case == "late_hook":
                    # Installed for output_tensors() only, then removed.
                    hook = nn.modules.module.register_module_buffer_registration_hook(
                        mirror_hook
                    )
                    try:
                        annotate.output_tensors(
                            "policy", {"action": action}, export_with="jit"
                        )
                    finally:
                        hook.remove()
                else:
                    annotate.output_tensors(
                        "policy", {"action": action}, export_with="jit"
                    )
                if case == "end_override":
                    model.register_buffer = mirror_register
                hook = None
                if case == "hook":
                    hook = nn.modules.module.register_module_buffer_registration_hook(
                        cast
                    )
                try:
                    leapp.stop()
                finally:
                    if hook is not None:
                        hook.remove()
                if case in ("clamp", "hook"):
                    check = model.h
                elif case in ("halving", "property"):
                    check = model._buffers["h"]
                elif case in ("lister", "lister_none"):
                    check = model.g
                else:
                    check = partner.h if case == "bound" else partner.g
                self.assertTrue(tied_to_capture(check))

    def test_explicit_state_on_every_pass_is_not_a_clash(self):
        """state_tensors() on every pass of a loop does not keep the session's buffers tied."""
        model = NoMutationModel().eval()
        history = torch.zeros(1, 4)
        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(2):
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            state = annotate.state_tensors("policy", {"history": history})
            annotate.module("policy", model)
            new_history = annotate.update_state("policy", {"history": state + obs})
            annotate.output_tensors(
                "policy", {"action": model(new_history)}, export_with="jit"
            )
            history = getattr(new_history, "tensor", new_history)
        leapp.stop()
        self.assertFalse(tied_to_capture(model.scale))

    def test_state_declared_after_module_under_a_buffer_name_stays_tied(self):
        """A buffer whose state name state_tensors() gets again for another tensor after module() stays tied."""
        model = NoMutationModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", model)
        annotate.state_tensors("policy", {"scale": torch.ones(4)})
        annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        leapp.stop()
        self.assertTrue(tied_to_capture(model.scale))

    def test_failed_module_call_with_a_shared_name_stays_tied_later(self):
        """A buffer that a failed module() call gave another model's TracedTensor stays tied in a later session."""

        class ListsParameter(nn.Module):
            """Lists a parameter as a buffer, so module() fails assigning it."""

            list_parameter = True

            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.full((4,), 3.0))
                self.zero = nn.Parameter(torch.zeros(4))

            def named_buffers(self, *args, **kwargs):
                yield "scale", self._buffers["scale"]
                if self.list_parameter:
                    yield "zero", self.zero

            def forward(self, obs):
                return obs * self.scale + self.zero

        first, second = NoMutationModel().eval(), ListsParameter().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", first)
        action = first(obs)
        with self.assertRaisesRegex(TypeError, "parameter"):
            annotate.module("policy", second)
        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        leapp.stop()

        second.list_parameter = False
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("again", {"obs": torch.randn(1, 4)})
        annotate.module("again", second)
        annotate.output_tensors("again", {"action": second(obs)}, export_with="jit")
        leapp.stop()
        self.assertTrue(tied_to_capture(second.scale))

    def test_state_name_repeated_in_one_module_call_stays_tied(self):
        """Buffers "enc.scale" and "enc_scale" of one model get one state name and stay tied."""

        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.ones(4))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = Enc()
                self.register_buffer("enc_scale", torch.full((4,), 5.0))

            def forward(self, obs):
                return obs * self.enc.scale + self.enc_scale

        model = Model().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", model)
        annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        leapp.stop()
        self.assertTrue(tied_to_capture(model.enc_scale))

    def test_same_model_twice_in_one_node_is_plain(self):
        """A model passed to module() twice in one node is not a shared state name."""
        for finish in (True, False):
            with self.subTest(finish=finish):
                annotate.keep_buffers_tied = False
                model = NoMutationModel().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", model)
                hidden = model(obs)
                annotate.module("policy", model)
                action = model(hidden)
                if finish:
                    annotate.output_tensors(
                        "policy", {"action": action}, export_with="jit"
                    )
                leapp.stop()
                self.assertFalse(tied_to_capture(model.scale))

    def test_refused_module_call_after_a_reset_leaves_the_model_usable(self):
        """A module() call that refuses a buffer still active in a reset node changes no buffer.

        After a start() reset, a buffer that requires grad, or one the forward
        assigned from None, still belongs to the discarded node. The next
        session refuses the model, as before, and the model still runs.
        """

        class GradModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.full((4,), 2.0))
                self.register_buffer("gain", torch.ones(4, requires_grad=True))

            def forward(self, x):
                return x * self.scale * self.gain

        class LazyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.full((4,), 2.0))
                self.register_buffer("h", None)

            def forward(self, x):
                self.h = x * self.scale
                return self.h

        for make in (GradModel, LazyModel):
            with self.subTest(model=make.__name__):
                annotate.keep_buffers_tied = False
                model = make().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.ones(1, 4)})
                annotate.module("policy", model)
                model(obs)
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("again", {"obs": torch.ones(1, 4)})
                with self.assertRaisesRegex(Exception, "different node"):
                    annotate.module("again", model)
                for name, buf in model.named_buffers():
                    self.assertNotEqual(getattr(buf, "context", None), "again", name)
                leapp.stop()
                model(torch.ones(1, 4))

    def test_buffer_of_a_reset_node_under_names_the_node_has_captures(self):
        """After a start() reset, a model whose buffer names the node already has is captured as before.

        Its buffer that requires grad still belongs to the discarded node, and
        the node reuses the names, so module() is not refused.
        """

        class GradModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.full((4,), 2.0))
                self.register_buffer("gain", torch.ones(4, requires_grad=True))

            def forward(self, x):
                return x * self.scale * self.gain

        first, second = GradModel(), GradModel()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.ones(4)})
        annotate.module("policy", first)
        first(obs)
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.full((4,), 3.0)})
        annotate.module("policy", second)
        hidden = second(obs)
        annotate.module("policy", first)
        annotate.output_tensors("policy", {"action": first(hidden)}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

    def test_refused_module_call_records_no_input_of_the_node(self):
        """A module() call refused for a buffer of a tracing node adds no input to the node and promotes no buffer.

        Two models share a tensor. module() for the second node is refused
        while the first node traces; after it finishes, module() is called
        again, and the second node exports and validates.
        """

        class Writer(nn.Module):
            def __init__(self, shared):
                super().__init__()
                self.register_buffer("s", shared)

            def forward(self, x):
                y = x * 2 + self.s
                self.s = self.s * 0.5 + x
                return y

        class Reader(nn.Module):
            def __init__(self, shared):
                super().__init__()
                self.register_buffer("a", torch.full((1, 4), 2.0))
                self.register_buffer("s", shared)

            def forward(self, x):
                y = x + self.a + self.s
                self.a = self.a * 0.5 + x
                self.s = self.s * 0.5 + x
                return y

        shared = torch.ones(1, 4)
        writer, reader = Writer(shared), Reader(shared)
        leapp.start(name=self.TEST_GRAPH_NAME)
        first = annotate.input_tensors("n1", {"obs": torch.randn(1, 4)})
        annotate.module("n1", writer)
        y1 = writer(first)
        second = annotate.input_tensors("n2", {"obs": torch.randn(1, 4)})
        with self.assertRaisesRegex(Exception, "different node"):
            annotate.module("n2", reader)
        inputs = [desc.name_str for desc in annotate.nodes["n2"].inputs]
        self.assertEqual(inputs, ["obs"])
        self.assertFalse(tied_to_capture(reader.a))
        annotate.output_tensors("n1", {"y": y1}, export_with="jit")
        annotate.module("n2", reader)
        annotate.output_tensors("n2", {"y": reader(second)}, export_with="jit")
        leapp.stop()
        leapp.compile_graph(visualize=False)

    def test_tensor_in_two_models_of_one_node_captures(self):
        """A tensor registered in two models of one node, under two names, exports and validates.

        Only a buffer active in another node is refused at module().
        """

        class Holder(nn.Module):
            def __init__(self, key, shared):
                super().__init__()
                self.key = key
                self.register_buffer(key, shared)

            def forward(self, x):
                return x + getattr(self, self.key)

        shared = torch.ones(1, 4)
        first, second = Holder("state", shared), Holder("memory", shared)
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
        annotate.module("policy", first)
        annotate.module("policy", second)
        annotate.output_tensors("policy", {"y": second(first(obs))}, export_with="jit")
        leapp.stop()
        self.assertEqual(leapp.compile_graph(visualize=False), {"policy": True})

    def test_buffer_that_requires_grad_stays_tied(self):
        """A buffer that requires grad is put back at the session end and stays tied, in any grad mode.

        Only a node's latest module() call puts it back, and it is never made
        plain. The model's other buffers are plain.
        """

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.zeros(4))

            def forward(self, x):
                return x + self.offset

        for mode in (torch.enable_grad, torch.no_grad, torch.inference_mode):
            with self.subTest(mode=mode.__name__):
                annotate.keep_buffers_tied = False
                model = NoMutationModel().eval()
                model.register_buffer("pe", model.linear.weight.sum(0) * 2)
                model.register_buffer("gain", torch.ones(4, requires_grad=True))
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", model)
                hidden = model(obs)
                annotate.module("policy", Head().eval())
                annotate.output_tensors("policy", {"action": hidden}, export_with="jit")
                obs = annotate.input_tensors("second", {"obs": torch.randn(1, 4)})
                annotate.module("second", model)
                model(obs)
                with mode():
                    leapp.stop()

                self.assertTrue(tied_to_capture(model.pe))
                self.assertTrue(tied_to_capture(model.gain))
                self.assertFalse(tied_to_capture(model.scale))

                model = NoMutationModel().eval()
                model.register_buffer("gain", torch.ones(4, requires_grad=True))
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", model)
                hidden = model(obs)
                annotate.module("policy", Head().eval())
                with mode():
                    leapp.stop()
                self.assertTrue(tied_to_capture(model.gain))

                model = NoMutationModel().eval()
                model.register_buffer("gain", torch.ones(4, requires_grad=True))
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", model)
                model(obs)
                with mode():
                    leapp.stop()
                self.assertEqual(model.gain.is_leaf, mode is not torch.enable_grad)

    def test_buffer_that_requires_grad_leaves_other_buffers_plain_later(self):
        """A buffer that requires grad stays tied, and the model's other buffers are still plain in a later session."""
        model = NoMutationModel().eval()
        model.register_buffer("gain", torch.ones(4, requires_grad=True))
        for _ in range(2):
            leapp.start(name=self.TEST_GRAPH_NAME)
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
            leapp.stop()
        self.assertTrue(tied_to_capture(model.gain))
        self.assertFalse(tied_to_capture(model.scale))

    def test_buffer_a_later_node_points_at_a_buffer_that_requires_grad(self):
        """A buffer that a later node sets to a view of a buffer that requires grad keeps requiring grad."""

        class PointingModel(NoMutationModel):
            point = False

            def forward(self, obs):
                if self.point:
                    self.h = self.pe.unsqueeze(0)
                return super().forward(obs)

        model = PointingModel().eval()
        model.register_buffer("pe", model.linear.weight.sum(0) * 2)
        model.register_buffer("h", torch.zeros(1, 4))
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("first", {"obs": torch.randn(1, 4)})
        annotate.module("first", model)
        annotate.output_tensors("first", {"action": model(obs)}, export_with="jit")
        model.point = True
        obs = annotate.input_tensors("second", {"obs": torch.randn(1, 4)})
        annotate.output_tensors("second", {"action": model(obs)}, export_with="jit")
        leapp.stop()
        self.assertTrue(model.h.requires_grad)

    def test_buffer_cleared_on_a_later_pass_stays_none(self):
        """A buffer a later pass sets to None stays None at stop()."""

        class ClearingModel(NoMutationModel):
            clear = False

            def forward(self, obs):
                action = super().forward(obs)
                if self.clear:
                    self.scale = None
                return action

        model = ClearingModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        for clear in (False, True):
            model.clear = clear
            obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {"action": model(obs)}, export_with="jit")
        with self.assertNoLogs("leapp", level="ERROR"):
            leapp.stop()
        self.assertIsNone(model.scale)
        self.assertFalse(tied_to_capture(model.bias))

    def test_buffer_reassigned_in_a_later_node_is_plain(self):
        """A buffer a later node assigns without its own module() call is plain after stop().

        The later node may also never reach output_tensors().
        """
        for finish in (True, False):
            with self.subTest(finish=finish):
                annotate.keep_buffers_tied = False
                model = GRUModel().eval()
                leapp.start(name=self.TEST_GRAPH_NAME)
                for node in ("first", "second"):
                    obs = annotate.input_tensors(node, {"obs": torch.randn(1, 4)})
                    if node == "first":
                        annotate.module(node, model)
                    action = model(obs)
                    if finish or node == "first":
                        annotate.output_tensors(
                            node, {"action": action}, export_with="jit"
                        )
                value = getattr(model.h_state, "tensor", model.h_state).detach().clone()
                leapp.stop()
                self.assertFalse(tied_to_capture(model.h_state))
                self.assertTrue(torch.equal(model.h_state, value))

    def test_inference_buffers_still_refuse_in_place_updates(self):
        """A buffer created under inference_mode is plain after stop() and still refuses in-place updates."""
        with torch.inference_mode():
            model = NoMutationModel().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        for node in ("first", "second"):
            obs = annotate.input_tensors(node, {"obs": torch.randn(1, 4)})
            annotate.module(node, model)
            action = model(obs)
            if node == "first":
                annotate.output_tensors(node, {"action": action}, export_with="jit")
        leapp.stop()
        self.assertFalse(tied_to_capture(model.scale))
        with self.assertRaises(RuntimeError):
            model.scale.add_(1)

    def test_buffer_reassigned_under_inference_mode_is_plain(self):
        """A buffer a later forward reassigns under inference_mode is plain after stop() and stays an inference tensor."""

        class Accumulator(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("h", torch.zeros(1, 4))

            def forward(self, x):
                self.h = self.h * 0.5 + x
                return self.h + 1

        model = Accumulator().eval()
        leapp.start(name=self.TEST_GRAPH_NAME)
        obs = annotate.input_tensors("first", {"obs": torch.ones(1, 4)})
        annotate.module("first", model)
        annotate.output_tensors("first", {"action": model(obs)}, export_with="jit")
        obs = annotate.input_tensors("second", {"obs": torch.ones(1, 4)})
        with torch.inference_mode():
            action = model(obs)
        annotate.output_tensors("second", {"action": action}, export_with="jit")
        leapp.stop()
        self.assertFalse(tied_to_capture(model.h))
        self.assertTrue(model.h.is_inference())

    def test_first_session_of_a_process_makes_buffers_plain(self):
        """In a new Python process, the first session makes buffers plain.

        The other tests reset the process-wide flag after each test, so only a
        new process shows the value it starts with.
        """
        script = textwrap.dedent(
            f"""
            import torch
            from torch import nn
            import leapp
            from leapp import annotate

            class Scaler(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("scale", torch.full((4,), 2.0))

                def forward(self, x):
                    return x * self.scale

            model = Scaler().eval()
            leapp.start(name="{self.TEST_GRAPH_NAME}")
            obs = annotate.input_tensors("policy", {{"obs": torch.ones(1, 4)}})
            annotate.module("policy", model)
            annotate.output_tensors("policy", {{"action": model(obs)}}, export_with="jit")
            leapp.stop()
            torch.export.export(model, (torch.ones(1, 4),))
            """
        )
        # The tree under test, wherever pytest runs from.
        root = os.path.dirname(os.path.dirname(os.path.abspath(leapp.__file__)))
        path = os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": path},
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_only_a_buffer_hook_from_torch_keeps_buffers_plain(self):
        """A buffer registration hook from torch itself does not keep buffers tied; others do.

        A failed torch.export leaves such a hook behind on torch 2.7. It only
        acts on the module given to that export. A hook from a package whose
        name only starts with "torch", with no module, or that only copies a
        torch function's module through functools.wraps keeps buffers tied.
        """
        from torch._functorch._aot_autograd.utils import (
            register_buffer_assignment_hook,
        )

        def keep(module, name, buffer):
            return None

        @functools.wraps(torch.nn.functional.relu)
        def wrapped(module, name, buffer):
            return None

        for source in ("torch", "torchfoo.hooks", None, "wrapped"):
            with self.subTest(source=source):
                annotate.keep_buffers_tied = False
                model = NoMutationModel().eval()
                if source == "torch":
                    hook = register_buffer_assignment_hook(nn.Module(), {})
                elif source == "wrapped":
                    hook = nn.modules.module.register_module_buffer_registration_hook(
                        wrapped
                    )
                else:
                    keep.__module__ = source
                    hook = nn.modules.module.register_module_buffer_registration_hook(
                        keep
                    )
                try:
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                    annotate.module("policy", model)
                    annotate.output_tensors(
                        "policy", {"action": model(obs)}, export_with="jit"
                    )
                    leapp.stop()
                finally:
                    hook.remove()
                self.assertEqual(tied_to_capture(model.scale), source != "torch")
                if source == "torch":
                    torch.export.export(model, (torch.randn(1, 4),))

    def test_scripted_models_capture(self):
        """A TorchScript model, a model with a TorchScript submodule, or a GRU subclass has plain buffers after the session."""

        class Norm(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("mean", torch.full((4,), 0.5))

            def forward(self, obs):
                return obs - self.mean

        class Policy(nn.Module):
            def __init__(self, norm):
                super().__init__()
                self.norm = norm
                self.linear = nn.Linear(4, 4)

            def forward(self, obs):
                return self.linear(self.norm(obs))

        class StatefulGRU(nn.GRU):
            def __init__(self):
                super().__init__(4, 4, batch_first=True)
                self.register_buffer("h", torch.zeros(1, 1, 4))

            def forward(self, obs):
                out, _ = super().forward(obs.unsqueeze(1), self.h)
                return out.squeeze(1)

        models = {
            "scripted_submodule": Policy(torch.jit.script(Norm())),
            "scripted": torch.jit.script(Policy(Norm())),
            "gru_subclass": StatefulGRU(),
        }
        for kind, model in models.items():
            with self.subTest(model=kind):
                annotate.keep_buffers_tied = False
                leapp.start(name=self.TEST_GRAPH_NAME)
                obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                annotate.module("policy", model.eval())
                annotate.output_tensors(
                    "policy", {"action": model(obs)}, export_with="jit"
                )
                leapp.stop()
                for name, buf in model.named_buffers():
                    self.assertFalse(tied_to_capture(buf), name)

    def test_buffers_kept_outside_buffers_dict(self):
        """A model that keeps its buffers in its own store works as before, also in a second session."""

        class StoreModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.register_buffer("scale", torch.full((4,), 2.0))

            def register_buffer(self, name, tensor, persistent=True):
                self.__dict__.setdefault("store", {})[name] = tensor
                object.__setattr__(self, name, tensor)

            def named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
                for name, buf in self.__dict__.get("store", {}).items():
                    yield (f"{prefix}.{name}" if prefix else name, buf)

            def forward(self, obs):
                return self.linear(obs * self.scale)

        for finish in (True, False):
            with self.subTest(finish=finish):
                annotate.keep_buffers_tied = False
                model = StoreModel().eval()
                for session in range(2):
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                    annotate.module("policy", model)
                    action = model(obs)
                    if finish or session:
                        annotate.output_tensors(
                            "policy", {"action": action}, export_with="jit"
                        )
                    leapp.stop()

    def test_session_end_keeps_buffer_changes_of_forward(self):
        """A forward that clears, deletes or replaces a buffer keeps that change at start() and stop()."""

        class ChangingModel(NoMutationModel):
            def __init__(self, change):
                super().__init__()
                self.change = change

            def forward(self, obs):
                action = super().forward(obs)
                if self.change == "none":
                    self.scale = None
                elif self.change == "delete":
                    del self.scale
                elif self.change == "parameter":
                    self.scale = nn.Parameter(torch.full((4,), 2.0))
                else:
                    self.scale = nn.Identity()
                return action

        for change in ("none", "delete", "parameter", "module"):
            for end in ("start", "stop"):
                with self.subTest(change=change, end=end):
                    annotate.keep_buffers_tied = False
                    model = ChangingModel(change)
                    eager = ChangingModel(change)
                    eager(torch.randn(1, 4))
                    leapp.start(name=self.TEST_GRAPH_NAME)
                    obs = annotate.input_tensors("policy", {"obs": torch.randn(1, 4)})
                    annotate.module("policy", model)
                    model(obs)
                    if end == "start":
                        leapp.start(name=self.TEST_GRAPH_NAME)
                    else:
                        leapp.stop()

                    self.assertIsNone(model._buffers.get("scale"))
                    self.assertFalse(tied_to_capture(model.bias))
                    self.assertEqual(
                        sorted(model.state_dict()), sorted(eager.state_dict())
                    )
                    if end == "start":
                        leapp.stop()

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
