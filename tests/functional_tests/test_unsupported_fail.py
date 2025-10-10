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

    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.TEST_GRAPH_NAME):
            shutil.rmtree(self.TEST_GRAPH_NAME)


if __name__ == '__main__':
    unittest.main(verbosity=2)
