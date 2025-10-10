import unittest
import torch
from leapp import annotate
import os
import shutil


class TestExportSituation(unittest.TestCase):
    """Unit tests to see if export situation is properly handled"""
    TEST_GRAPH_NAME = "test_graph"

    def test_export_nnModule_function(self):
        linear = torch.nn.Linear(3, 3)

        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            output = linear(inputA)
            return output

        annotate.start(name=self.TEST_GRAPH_NAME)
        funcA(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        annotate.stop()
        annotate.compile_graph()

    # def test_export_nnModule(self):
    #     class moduleA(torch.nn.Module):
    #         def __init__(self):
    #             super(moduleA, self).__init__()
    #             self.linear = torch.nn.Linear(3, 3)

    #         def forward(self, inputA: torch.Tensor):
    #             return self.linear(inputA)

    #     annotate.start(name=self.TEST_GRAPH_NAME)
    #     moduleA()(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
    #     annotate.stop()
    #     # annotate.compile_graph()

    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.TEST_GRAPH_NAME):
            shutil.rmtree(self.TEST_GRAPH_NAME)


if __name__ == '__main__':
    unittest.main(verbosity=2)
#
