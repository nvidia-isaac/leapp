import unittest
import os
import shutil


class LEAPPFunctionalTestBase(unittest.TestCase):
    def setUp(self):
        self.TEST_GRAPH_NAME = "test_graph"

    def tearDown(self):
        if os.path.exists(self.TEST_GRAPH_NAME):
            shutil.rmtree(self.TEST_GRAPH_NAME)

    def verify_num_connections(self, leapp_annotation, nodes=None, inputs=None, outputs=None,
                               internal_connections=None, feedback_connections=None):
        if nodes is not None:
            self.assertEqual(nodes, len(leapp_annotation.detected_nodes),
                             "Number of nodes do not match")
        if inputs is not None:
            total_inputs = sum(
                [len(graph_inputs) for graph_inputs in leapp_annotation.detected_pipeline['dangling_inputs'].values()])
            self.assertEqual(inputs, total_inputs,
                             "Number of inputs do not match")
        if outputs is not None:
            total_outputs = sum(
                [len(graph_outputs) for graph_outputs in leapp_annotation.detected_pipeline['dangling_outputs'].values()])
            self.assertEqual(outputs, total_outputs,
                             "Number of outputs do not match")
        if internal_connections is not None:
            self.assertEqual(internal_connections, len(
                leapp_annotation.detected_pipeline['data_flow']), "Number of internal connections do not match")
        if feedback_connections is not None:
            self.assertEqual(feedback_connections, len(
                leapp_annotation.detected_pipeline['feedback_flow']), "Number of feedback connections do not match")
