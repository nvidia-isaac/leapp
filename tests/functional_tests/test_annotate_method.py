
import unittest
import torch
from leapp import annotate
from leapp.utils import inspect_torchscript_model
from .base import LEAPPFunctionalTestBase
import os


class TestAnnotateMethod(LEAPPFunctionalTestBase):
    def test_annotate_method(self):
        """tests the basic situation of using the annotate.method decorator"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA

        @annotate.method(export_with="torch")
        def funcC(inputB: torch.Tensor):
            return inputB+5.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        for i in range(10):
            outputA = funcA(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
            outputC = funcC(outputA)
        annotate.stop()

        self.assertEqual(len(annotate.nodes), 2)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 1)
        self.assertEqual(len(annotate.nodes[funcC.__name__].inputs), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].outputs), 1)
        self.assertEqual(len(annotate.nodes[funcC.__name__].outputs), 1)
        self.assertEqual(
            annotate.nodes[funcA.__name__].inputs[0].name, "inputA")
        self.assertEqual(
            annotate.nodes[funcC.__name__].inputs[0].name, "inputB")

    def test_annotate_method_with_kwargs_and_default_value(self):
        """tests the situation where the function has and default values"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)):
            return inputA

        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA = funcA()
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 0)
        model = torch.jit.load(os.path.join(
            self.TEST_GRAPH_NAME, funcA.__name__+".pt"))

        expected_output = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        # check if using default
        self.assertTrue(torch.allclose(model(), expected_output))
        # Or get structured data
        model_info = inspect_torchscript_model(model)
        self.assertEqual(len(model_info['inputs']), 2)
        self.assertEqual(len(model_info['outputs']), 1)

    def test_annotate_method_ignoring_default_values(self):
        """tests the situation where we pass in a value overriding the default"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)):
            return inputA

        annotate.start(name=self.TEST_GRAPH_NAME)
        outputA = funcA(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))
        annotate.stop()
        annotate.compile_graph(visualize=False)
        self.assertEqual(len(annotate.nodes), 1)
        self.assertEqual(len(annotate.nodes[funcA.__name__].inputs), 1)
        model = torch.jit.load(os.path.join(
            self.TEST_GRAPH_NAME, funcA.__name__+".pt"))
        model_info = inspect_torchscript_model(model)
        self.assertEqual(len(model_info['inputs']), 2)
        self.assertEqual(len(model_info['outputs']), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
