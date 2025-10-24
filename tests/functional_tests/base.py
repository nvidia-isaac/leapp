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
