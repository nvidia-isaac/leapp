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
"""Tests for automatic buffer tracking (annotate.module)."""

import unittest
import torch
import torch.nn as nn
from leapp import annotate
from .base import LEAPPFunctionalTestBase


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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action},
                                export_with="onnx-torchscript")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # 1 input (obs), 1 output (action), 1 feedback (h_state)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
        )
        self.verify_all_models_exist("policy")

    def test_lstm_two_buffers(self):
        """LSTM with 2 mutated buffers -> 2 feedback connections."""
        model = LSTMModel()
        model.eval()
        obs = torch.randn(1, 4)

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action},
                                export_with="onnx-torchscript")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # 1 input (obs), 1 output (action), 2 feedback (h_state, c_state)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=2
        )
        self.verify_all_models_exist("policy")

    def test_partial_mutation(self):
        """3 buffers, 2 mutated -> 2 feedback, 1 regular input (const_mask)."""
        model = PartialMutationModel()
        model.eval()
        obs = torch.randn(1, 4)

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # 1 dangling input (obs), 1 dangling output (action),
        # 2 feedback (running_mean, step_count)
        # const_mask is baked as a constant (not a dynamic input)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=2
        )
        self.verify_all_models_exist("policy")

    def test_no_buffers_mutated(self):
        """All buffers read but not reassigned -> 0 feedback, all regular inputs."""
        model = NoMutationModel()
        model.eval()
        obs = torch.randn(1, 4)

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # 1 dangling input (obs), 1 dangling output, 0 feedback
        # scale and bias are baked as constants (not dynamic inputs)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=0
        )
        self.verify_all_models_exist("policy")

    def test_no_buffers(self):
        """Model with no registered buffers -> module() is a no-op."""
        model = NoBufferModel()
        model.eval()
        obs = torch.randn(1, 4)

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=0
        )
        self.verify_all_models_exist("policy")

    def test_buffer_names_filter(self):
        """Only specified buffers are tracked; others ignored."""
        model = LSTMModel()
        model.eval()
        obs = torch.randn(1, 4)

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        # Only track h_state, ignore c_state
        annotate.module("policy", model, buffer_names=["h_state"])
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action},
                                export_with="onnx-torchscript")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # 1 input (obs), 1 output (action), 1 feedback (h_state only)
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
        )
        self.verify_all_models_exist("policy")

    def test_model_restored_after_tracing(self):
        """Verify model buffers are properly re-registered after tracing."""
        model = GRUModel()
        model.eval()
        obs = torch.randn(1, 4)

        # Record original buffer names
        original_buffers = dict(model.named_buffers())

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action},
                                export_with="onnx-torchscript")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # After tracing, all buffers should be restored
        restored_buffers = dict(model.named_buffers())
        self.assertEqual(set(original_buffers.keys()), set(restored_buffers.keys()),
                         "Buffer names should be restored after tracing")

        # Restored values should be proper tensors (not TracedTensors)
        for name, buf in restored_buffers.items():
            self.assertIsInstance(buf, torch.Tensor,
                                 f"Buffer '{name}' should be a torch.Tensor after restore")

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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        result = model(obs_traced)

        annotate.output_tensors("policy", {"result": result}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # Only 1 input (obs) — scale and bias baked as constants
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=0
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

        # Don't call annotate.start() — not tracing
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action},
                                export_with="onnx-torchscript")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # GRU's h_state is at rnn.h_state (nested) -> 1 feedback
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=2
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action},
                                export_with="onnx-torchscript")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # rnn.rnn.h_state is 3 levels deep -> still detected
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        out_a, out_b = model(obs_traced)

        annotate.output_tensors("policy", {"action_a": out_a, "action_b": out_b},
                                export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=2,
            internal_connections=0, feedback_connections=1
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

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        annotate.module("policy", model)
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=1
        )


class TestBufferTrackingErrors(LEAPPFunctionalTestBase):
    """Error cases and validation for annotate.module()."""

    def test_module_before_input_tensors_raises(self):
        """module() called before input_tensors() should raise — node doesn't exist."""
        model = GRUModel()

        annotate.start(name=self.TEST_GRAPH_NAME)
        try:
            annotate.module("policy", model)
            annotate.stop()
            self.fail("Expected an exception")
        except Exception as e:
            self.assertIn("module", str(e).lower())
        finally:
            # Ensure tracing is stopped even on exception
            try:
                annotate.stop()
            except Exception:
                pass

    def test_stop_restores_buffers_without_output_tensors(self):
        """stop() safety net restores model buffers even if output_tensors() is never called."""
        model = GRUModel()
        model.eval()
        obs = torch.randn(1, 4)

        original_buffer_names = set(dict(model.named_buffers()).keys())

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})
        annotate.module("policy", model)
        action = model(obs_traced)

        # Intentionally skip output_tensors() — simulate user error or exception
        annotate.stop()

        # Buffers should still be restored by stop() safety net
        restored_buffer_names = set(dict(model.named_buffers()).keys())
        self.assertEqual(original_buffer_names, restored_buffer_names,
                         "stop() should restore buffers even without output_tensors()")

        # Verify they're real tensors, not TracedTensors
        from leapp.leapp_graph.datatypes import TracedTensor
        for name, buf in model.named_buffers():
            self.assertNotIsInstance(buf, TracedTensor,
                                    f"Buffer '{name}' should not be TracedTensor after stop()")

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
        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})
        annotate.module("policy", model)
        action = model(obs_traced)
        annotate.output_tensors("policy", {"action": action},
                                export_with="onnx-torchscript")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # Reset state and run again — model should produce same result as pre-tracing
        model.h_state.zero_()
        with torch.no_grad():
            post_trace_output = model(obs.clone())

        self.assertTrue(
            torch.allclose(ref_output, post_trace_output, atol=1e-6),
            "Model should produce identical output before and after tracing")

    def test_buffer_names_filter_nonexistent_name(self):
        """Filtering by a buffer name that doesn't exist -> no buffers tracked."""
        model = GRUModel()
        model.eval()
        obs = torch.randn(1, 4)

        annotate.start(name=self.TEST_GRAPH_NAME)
        obs_traced = annotate.input_tensors("policy", {"obs": obs})

        # "nonexistent" doesn't match any buffer — tracker injects nothing
        annotate.module("policy", model, buffer_names=["nonexistent"])
        action = model(obs_traced)

        annotate.output_tensors("policy", {"action": action}, export_with="jit")
        annotate.stop()
        annotate.compile_graph(visualize=False)

        # No buffers were tracked -> 0 feedback
        self.verify_num_connections(
            annotate, nodes=1, inputs=1, outputs=1,
            internal_connections=0, feedback_connections=0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
