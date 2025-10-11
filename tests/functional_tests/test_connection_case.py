
import unittest
import torch
from leapp import annotate
import os
import shutil


class TestConnectionCase(unittest.TestCase):
    """Unit tests to see if connections between nodes are properly handled"""
    TEST_GRAPH_NAME = "test_graph"

    def verify_num_connections(self, nodes=None, inputs=None, outputs=None,
                               internal_connections=None, feedback_connections=None):
        if nodes is not None:
            self.assertEqual(nodes, len(annotate.detected_nodes),
                             "Number of nodes do not match")
        if inputs is not None:
            self.assertEqual(inputs, len(annotate.detected_pipeline['dangling_inputs']),
                             "Number of inputs do not match")
        if outputs is not None:
            self.assertEqual(outputs, len(
                annotate.detected_pipeline['dangling_outputs']), "Number of outputs do not match")
        if internal_connections is not None:
            self.assertEqual(internal_connections, len(
                annotate.detected_pipeline['data_flow']), "Number of internal connections do not match")
        if feedback_connections is not None:
            pass  # TODO: implement this

    def test_multiple_runs_of_same_graph(self):
        """tests the situation where the same graph is run multiple times"""
        @annotate.method(export_with="torch")
        def funcA(inputA: torch.Tensor):
            return inputA

        @annotate.method(export_with="torch")
        def funcC(inputB: torch.Tensor):
            return inputB+5.0

        annotate.start(name=self.TEST_GRAPH_NAME)
        for i in range(10):
            outputA = funcA(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
            with annotate.block(node_name="blockA", inputs=["outputA"],
                                outputs=["outputB"], export_with="torch"):
                outputB = outputA*2.
            outputC = funcC(outputB)
        annotate.stop()
        annotate.compile_graph()
        self.verify_num_connections(nodes=3, inputs=1, outputs=1,
                                    internal_connections=2)

    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.TEST_GRAPH_NAME):
            shutil.rmtree(self.TEST_GRAPH_NAME)


if __name__ == '__main__':
    unittest.main(verbosity=2)
