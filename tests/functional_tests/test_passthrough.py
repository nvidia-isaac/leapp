import unittest
import torch
from leapp import annotate
from .base import LEAPPFunctionalTestBase
from leapp.leapp_graph.datatypes import TracedTensor


class TestPassthrough(LEAPPFunctionalTestBase):

    def test_method_passthrough(self):
        seen_traced_inputs = []

        @annotate.method(export_with="jit")
        def policy(x: torch.Tensor):
            seen_traced_inputs.append(isinstance(x, TracedTensor))
            y = x * 2.0 + 1.0
            return y

        x = torch.tensor([1.0, 2.0, 3.0])
        y = policy(x)

        self.assertEqual(len(seen_traced_inputs), 1)
        self.assertFalse(seen_traced_inputs[0], "Input should remain a raw tensor outside tracing")
        self.assertNotIsInstance(y, TracedTensor)
        self.assertFalse(hasattr(x, "leapp_tag"))
        self.assertFalse(hasattr(y, "leapp_tag"))
        self.assertEqual(len(annotate.nodes), 0, "No nodes should be created outside start/stop")

    def test_input_output_tensor_passthrough(self):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([0.5, 0.5, 0.5])

        traced_a, traced_b = annotate.input_tensors("node_a", {"a": a, "b": b})
        self.assertIs(traced_a, a)
        self.assertIs(traced_b, b)
        self.assertNotIsInstance(traced_a, TracedTensor)
        self.assertNotIsInstance(traced_b, TracedTensor)

        out = traced_a + traced_b
        annotate.output_tensors("node_a", {"out": out}, export_with="jit")

        self.assertFalse(hasattr(traced_a, "leapp_tag"))
        self.assertFalse(hasattr(traced_b, "leapp_tag"))
        self.assertFalse(hasattr(out, "leapp_tag"))
        self.assertEqual(len(annotate.nodes), 0, "No nodes should be created outside start/stop")

    def test_state_tensor_passthrough(self):
        obs = torch.tensor([1.0, 2.0, 3.0])
        state = torch.tensor([0.1, 0.2, 0.3])

        passthrough_state = annotate.state_tensors("node_state", {"state": state})
        self.assertIs(passthrough_state, state)
        self.assertNotIsInstance(passthrough_state, TracedTensor)

        new_state = passthrough_state + obs
        annotate.update_state("node_state", {"state": new_state})

        self.assertFalse(hasattr(state, "leapp_tag"))
        self.assertFalse(hasattr(passthrough_state, "leapp_tag"))
        self.assertFalse(hasattr(new_state, "leapp_tag"))
        self.assertEqual(len(annotate.nodes), 0, "No nodes should be created outside start/stop")

    def test_module_passthrough(self):
        class TinyStatefulModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("running_state", torch.zeros(4))

            def forward(self, x):
                return x + self.running_state

        model = TinyStatefulModel()
        original_buffers = dict(model.named_buffers())
        original_buffer_ids = {name: id(buf) for name, buf in original_buffers.items()}

        # Should no-op outside tracing, even without creating a LEAPP node.
        annotate.module("policy", model)

        current_buffers = dict(model.named_buffers())
        self.assertEqual(set(original_buffers.keys()), set(current_buffers.keys()))
        for name, buf in current_buffers.items():
            self.assertNotIsInstance(buf, TracedTensor)
            self.assertEqual(
                id(buf), original_buffer_ids[name],
                f"Buffer '{name}' should not be replaced outside tracing"
            )
        self.assertEqual(len(annotate.nodes), 0, "No nodes should be created outside start/stop")


class TestDryrun(LEAPPFunctionalTestBase):

    