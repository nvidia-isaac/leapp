import unittest
import torch
from leapp import annotate
import os
import shutil


class TestUnsupportedFail(unittest.TestCase):
    """Unit tests to see if unsupported io is properly handled"""
    TEST_GRAPH_NAME = "test_graph"

    def test_same_variable_used_twice(self):

        @annotate.method()
        def funcA(inputA: torch.Tensor):
            return inputA

        @annotate.method()
        def funcB(inputB: torch.Tensor, inputC: torch.Tensor):
            return inputB+inputC

        annotate.start(name=self.TEST_GRAPH_NAME)
        retvalA = funcA(torch.tensor([1, 2, 3]))
        funcB(retvalA, retvalA)
        annotate.stop()

        try:
            annotate.compile_graph()
        except Exception as e:
            self.assertEqual(
                str(e), "Error: unsupported use of sending the same tensor multiple times to the same node")
            return
        self.fail("Expected an exception")

    def test_function_name_overlap(self):
        node_name = "funcA"

        @annotate.method(node_name=node_name)
        def func(inputA: torch.Tensor):
            return inputA

        @annotate.method(node_name=node_name)
        def func_copy(inputA: torch.Tensor):
            return inputA

        annotate.start(name=self.TEST_GRAPH_NAME)

        try:
            return_value = func(torch.tensor([1, 2, 3]))
            func_copy(return_value)
        except Exception as e:
            annotate.stop()
            self.assertEqual(
                str(
                    e), f"Error when attempting to set up new trace for {node_name}. \n"
                f"ExportManager is already tracing {node_name}")
            return

        annotate.stop()

    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.TEST_GRAPH_NAME):
            shutil.rmtree(self.TEST_GRAPH_NAME)


if __name__ == '__main__':
    unittest.main(verbosity=2)
