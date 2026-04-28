#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch
import leapp
from leapp.leapp import _MANAGER as annotate
from .base import LEAPPFunctionalTestBase


class TestConnectionCase(LEAPPFunctionalTestBase):

    def test_multiple_runs_of_same_graph(self):
        """tests the situation where the same graph is run multiple times"""
        @annotate.method(export_with="jit")
        def funcA(inputA: torch.Tensor):
            return inputA

        @annotate.method(export_with="jit")
        def funcC(inputB: torch.Tensor):
            return inputB+5.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        for i in range(10):
            outputA = funcA(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
            outputA = annotate.input_tensors("blockA", {"outputA": outputA})
            outputB = outputA*2.
            annotate.output_tensors("blockA", {"outputB": outputB}, export_with="jit")
            outputC = funcC(outputB)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_num_connections(annotate, nodes=3, inputs=1, outputs=1,
                                    internal_connections=2)

    def test_feedback_connections(self):
        """tests the situation where the graph has feedback connections"""
        @annotate.method()
        def funcA(inputA: torch.Tensor, loop_back: torch.Tensor):
            return inputA + loop_back

        @annotate.method()
        def funcB(inputB: torch.Tensor):
            return inputB

        @annotate.method()
        def funcC(inputC: torch.Tensor, loop_back: torch.Tensor):
            return inputC + loop_back

        @annotate.method()
        def funcD(inputD: torch.Tensor):
            outputD = inputD.clone()
            return outputD

        leapp.start(name=self.TEST_GRAPH_NAME, verbose=False)
        feedback_input = torch.tensor([0.0, 0.0, 0.0])
        for i in range(2):
            out_funcA = funcA(torch.tensor([1.0, 2.0, 3.0]), feedback_input)
            out_funcB = funcB(out_funcA)
            out_funcC = funcC(out_funcB, feedback_input)
            out_funcD = funcD(out_funcC)
            feedback_input = out_funcC

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_num_connections(annotate, nodes=4, inputs=1, outputs=1,
                                    internal_connections=3, feedback_connections=1)

    def test_interleaved_traced_nodes_keep_forward_execution_order(self):
        """Interleaved traced nodes should classify completed dependencies as forward flow."""
        trace_seed = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        trace_external = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)

        leapp.start(name=self.TEST_GRAPH_NAME)

        seed = annotate.input_tensors("node_a", {"seed": trace_seed})

        external_input = annotate.input_tensors(
            "node_b", {"external_input": trace_external}
        )
        from_b = external_input + 5.0
        annotate.output_tensors("node_b", {"b_out": from_b}, export_with="jit")

        from_b = annotate.input_tensors("node_a", {"from_b": from_b})
        annotate.output_tensors(
            "node_a", {"final_output": seed + from_b}, export_with="jit"
        )

        leapp.stop()
        leapp.compile_graph(visualize=False)

        data_flow = {
            source: list(targets)
            for source, targets in annotate.detected_pipeline["data_flow"].items()
        }
        feedback_flow = {
            source: list(targets)
            for source, targets in annotate.detected_pipeline["feedback_flow"].items()
        }

        self.assertEqual({"node_b/b_out": ["node_a/from_b"]}, data_flow)
        self.assertEqual({}, feedback_flow)
        self.verify_num_connections(
            annotate,
            nodes=2,
            inputs=2,
            outputs=1,
            internal_connections=1,
            feedback_connections=0,
        )
        self.verify_all_models_exist("node_a", "node_b")
        self.verify_safetensors_matches_feedback(annotate)

        runtime_seed = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
        runtime_external = torch.tensor([7.0, 8.0, 9.0], dtype=torch.float32)
        expected_output = runtime_seed + (runtime_external + 5.0)

        self.verify_inference_manager(
            source_inputs={
                "node_a/seed": runtime_seed,
                "node_b/external_input": runtime_external,
            },
            source_outputs={"node_a/final_output": expected_output},
        )

    def test_tensor_tag_presistence(self):
        @annotate.method()
        def funcA(inputA: torch.Tensor):
            return inputA, inputA + 1

        @annotate.method()
        def funcB(inputB: torch.Tensor):
            return inputB

        @annotate.method()
        def funcC(inputC: torch.Tensor):
            return inputC

        @annotate.method()
        def funcD(inputD: torch.Tensor):
            return inputD

        @annotate.method()
        def funcE(inputE: torch.Tensor, inputE2: torch.Tensor):
            return inputE

        leapp.start(name="test_graph")
        out_funcA1, out_funcA2 = funcA(torch.tensor([1.0, 2.0, 3.0]))
        out_funcB = funcB(out_funcA1.clone())
        out_funcC = funcC(out_funcA2.detach())
        out_funcD = funcD(out_funcC.contiguous())
        # not testing .cuda() because CI machine does not have a GPU
        funcE(out_funcD.cpu(), out_funcB)

        leapp.stop()
        leapp.compile_graph()

        self.verify_num_connections(annotate, nodes=5, inputs=1, outputs=2,
                                    internal_connections=4, feedback_connections=0)

    def test_mirror_leapp_tags_with_inplace_assignment(self):
        """Test mirror_leapp_tags with in-place assignment operations between nodes"""
        @annotate.method(export_with="jit")
        def funcA(inputA: torch.Tensor):
            return inputA * 2.0

        @annotate.method(export_with="jit")
        def funcB(inputB: torch.Tensor):
            return inputB + 1.0

        @annotate.method(export_with="jit")
        def funcC(inputC: torch.Tensor):
            return inputC * 3.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        out_funcA = funcA(torch.tensor([1.0, 2.0, 3.0]))
        
        # Simulate in-place assignment to a preallocated buffer BETWEEN nodes
        buffer = torch.zeros_like(out_funcA)
        buffer[:] = out_funcA
        # Mirror tags to maintain graph connections
        annotate.mirror_leapp_tags(out_funcA, buffer)
        
        out_funcB = funcB(buffer)
        out_funcC = funcC(out_funcB)
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Should have proper connections: funcA -> funcB -> funcC
        self.verify_num_connections(annotate, nodes=3, inputs=1, outputs=1,
                                    internal_connections=2)

    def test_mirror_leapp_tags_with_preallocated_buffer(self):
        """Test mirror_leapp_tags with a class that uses preallocated buffers between nodes"""
        class DataProcessor:
            def __init__(self):
                self._buffer = torch.zeros(3)

            def copy_to_buffer(self, data):
                """Copy data to buffer and mirror tags (outside of annotated nodes)"""
                self._buffer[:] = data
                annotate.mirror_leapp_tags(data, self._buffer)
                return self._buffer

            @annotate.method(export_with="jit")
            def process(self, input_data: torch.Tensor):
                # Process the buffered data
                result = input_data * 2.0
                return result

        @annotate.method(export_with="jit")
        def upstream_node(inputA: torch.Tensor):
            return inputA + 1.0

        @annotate.method(export_with="jit")
        def downstream_node(inputB: torch.Tensor):
            return inputB + 3.0

        processor = DataProcessor()
        leapp.start(name=self.TEST_GRAPH_NAME)
        upstream_output = upstream_node(torch.tensor([1.0, 2.0, 3.0]))
        
        # Copy to buffer and mirror tags BETWEEN nodes
        buffered_data = processor.copy_to_buffer(upstream_output)
        
        processed = processor.process(buffered_data)
        final_output = downstream_node(processed)
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Should have proper connections through all three nodes
        self.verify_num_connections(annotate, nodes=3, inputs=1, outputs=1,
                                    internal_connections=2)

    def test_mirror_leapp_tags_multiple_buffers(self):
        """Test mirror_leapp_tags with multiple buffers between nodes"""
        @annotate.method(export_with="jit")
        def funcA(inputA: torch.Tensor):
            return inputA, inputA + 1.0

        @annotate.method(export_with="jit")
        def funcB(inputB1: torch.Tensor, inputB2: torch.Tensor):
            # Process the inputs
            return inputB1 * 2.0, inputB2 * 3.0

        @annotate.method(export_with="jit")
        def funcC(inputC1: torch.Tensor, inputC2: torch.Tensor):
            return inputC1 + inputC2

        leapp.start(name=self.TEST_GRAPH_NAME)
        out_A1, out_A2 = funcA(torch.tensor([1.0, 2.0, 3.0]))
        
        # Create two separate buffers BETWEEN nodes
        buffer1 = torch.zeros_like(out_A1)
        buffer2 = torch.zeros_like(out_A2)
        
        # In-place copy
        buffer1[:] = out_A1
        buffer2[:] = out_A2
        
        # Mirror tags for both
        annotate.mirror_leapp_tags(out_A1, buffer1)
        annotate.mirror_leapp_tags(out_A2, buffer2)
        
        out_B1, out_B2 = funcB(buffer1, buffer2)
        final_output = funcC(out_B1, out_B2)
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Verify connections: funcA (2 outputs) -> funcB (2 inputs, 2 outputs) -> funcC (2 inputs)
        self.verify_num_connections(annotate, nodes=3, inputs=1, outputs=1,
                                    internal_connections=4)

    def test_mirror_leapp_tags_preserves_tag_through_chain(self):
        """Test that mirror_leapp_tags preserves tags through a long chain of buffer operations"""
        @annotate.method(export_with="jit")
        def source_node(inputA: torch.Tensor):
            return inputA * 2.0

        @annotate.method(export_with="jit")
        def process_node1(inputB: torch.Tensor):
            return inputB + 1.0

        @annotate.method(export_with="jit")
        def process_node2(inputC: torch.Tensor):
            return inputC * 3.0

        @annotate.method(export_with="jit")
        def sink_node(inputD: torch.Tensor):
            return inputD

        leapp.start(name=self.TEST_GRAPH_NAME)
        out1 = source_node(torch.tensor([1.0, 2.0, 3.0]))
        
        # Buffer between nodes 1 and 2
        buffer1 = torch.empty_like(out1)
        buffer1[:] = out1
        annotate.mirror_leapp_tags(out1, buffer1)
        
        out2 = process_node1(buffer1)
        
        # Buffer between nodes 2 and 3
        buffer2 = torch.empty_like(out2)
        buffer2[:] = out2
        annotate.mirror_leapp_tags(out2, buffer2)
        
        out3 = process_node2(buffer2)
        final = sink_node(out3)
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Should maintain proper connections through all nodes
        self.verify_num_connections(annotate, nodes=4, inputs=1, outputs=1,
                                    internal_connections=3)

    def test_mirror_leapp_tags_noop_outside_tracing(self):
        """Test that mirror_leapp_tags safely no-ops outside of tracing"""
        source = torch.tensor([1.0, 2.0, 3.0])
        target = torch.zeros(3)
        target[:] = source
        
        # Should not raise an error when called outside tracing
        annotate.mirror_leapp_tags(source, target)
        
        # Verify data is still correct
        self.assertTrue(torch.allclose(source, target))


if __name__ == '__main__':
    unittest.main(verbosity=2)
