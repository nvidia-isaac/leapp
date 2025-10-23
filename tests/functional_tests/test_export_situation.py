import unittest
from .base import LEAPPFunctionalTestBase
import torch
from leapp import annotate


class TestExportSituation(LEAPPFunctionalTestBase):
    """
    Unit tests to see if export situation is properly handled

    These tests test for things that are put inside of the code
    snippet that we want to support

    """

    def test_export_nnModule_function(self):
        linear = torch.nn.Linear(3, 3)

        @annotate.method(export_with="torch", environment_constants=['linear'])
        def funcA(inputA: torch.Tensor):
            output = linear(inputA)
            return output

        annotate.start(name=self.TEST_GRAPH_NAME)
        funcA(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        annotate.stop()
        annotate.compile_graph(visualize=False)

    def test_export_nnModule(self):
        class moduleA(torch.nn.Module):
            def __init__(self):
                super(moduleA, self).__init__()
                self.linear = torch.nn.Linear(3, 3)

            @annotate.method(export_with="torch")
            def forward(self, inputA: torch.Tensor):
                return self.linear(inputA)

        annotate.start(name=self.TEST_GRAPH_NAME)
        moduleA()(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
        annotate.stop()
        annotate.compile_graph(visualize=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
