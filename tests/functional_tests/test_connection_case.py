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
import warp as wp
import leapp
from leapp.leapp import _MANAGER as annotate
from .base import LEAPPFunctionalTestBase
from tests.warp_support import WarpTestCase


@wp.kernel
def _warp_add_arrays_kernel(
    src1: wp.array(dtype=wp.float32),
    src2: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dst[i] = src1[i] + src2[i]


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


class TestWarpConnectionCase(WarpTestCase, LEAPPFunctionalTestBase):
    """Warp equivalents of the torch connection/tag tests in TestConnectionCase."""

    def _warp_add_arrays(self, src1, src2):
        dst = wp.empty_like(src1)
        wp.launch(
            _warp_add_arrays_kernel,
            dim=src1.size,
            inputs=[src1, src2],
            outputs=[dst],
        )
        return dst

    def _run_copy_node(self, node_name, input_name, output_name, src):
        dst = None
        for _ in range(2):
            src_traced = annotate.input_tensors(node_name, {input_name: src})
            with annotate.warp_op(node_name):
                dst = wp.empty_like(src_traced)
                wp.copy(dst, src_traced)
            dst = annotate.output_tensors(
                node_name, {output_name: dst}, export_with="onnx"
            )
        return dst

    def _run_add_scalar_node(self, node_name, input_name, output_name, src, value):
        dst = None
        for _ in range(2):
            src_traced = annotate.input_tensors(node_name, {input_name: src})
            with annotate.warp_op(node_name):
                dst = wp.empty_like(src_traced)
                wp.launch(
                    self.kernels.add_scalar,
                    dim=src_traced.size,
                    inputs=[src_traced, wp.float32(value)],
                    outputs=[dst],
                )
            dst = annotate.output_tensors(
                node_name, {output_name: dst}, export_with="onnx"
            )
        return dst

    def _pipeline_views(self):
        return (
            {
                source: list(targets)
                for source, targets in annotate.detected_pipeline["data_flow"].items()
            },
            {
                source: list(targets)
                for source, targets in annotate.detected_pipeline["feedback_flow"].items()
            },
        )

    def test_warp_nodes_chain_via_output_tags(self):
        """Two warp nodes connected by a tagged wp.array output -> input edge."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        arr1 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        arr2 = self._run_copy_node("node_a", "arr1", "arr2", arr1)

        self.assertTrue(hasattr(arr2, "leapp_tag"))
        self.assertEqual(arr2.leapp_tag, "node_a/arr2/")

        arr3 = self._run_copy_node("node_b", "arr2", "arr3", arr2)

        leapp.stop()
        leapp.compile_graph(visualize=False)

        data_flow, feedback_flow = self._pipeline_views()
        self.assertEqual({"node_a/arr2": ["node_b/arr2"]}, data_flow)
        self.assertEqual({}, feedback_flow)
        self.verify_num_connections(
            annotate,
            nodes=2,
            inputs=1,
            outputs=1,
            internal_connections=1,
        )
        self.verify_all_models_exist("node_a", "node_b")

    def test_warp_node_requires_second_execution_before_compile(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        arr = annotate.input_tensors(
            "node_a",
            {"in_a": wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)},
        )
        with annotate.warp_op("node_a"):
            out = wp.empty_like(arr)
            wp.copy(out, arr)
        annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")
        leapp.stop()

        with self.assertRaisesRegex(Exception, "executed a second time"):
            leapp.compile_graph(visualize=False)

    def test_multiple_explicit_warp_segments_in_one_node(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)

        for _ in range(2):
            arr = annotate.input_tensors("node_a", {"in_a": source})
            with annotate.warp_op("node_a"):
                copied = wp.empty_like(arr)
                wp.copy(copied, arr)
            with annotate.warp_op("node_a"):
                out = wp.empty_like(copied)
                wp.launch(
                    self.kernels.add_scalar,
                    dim=copied.size,
                    inputs=[copied, wp.float32(2.0)],
                    outputs=[out],
                )
            annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_num_connections(annotate, nodes=1, inputs=1, outputs=1)
        self.verify_all_models_exist("node_a")

    def test_multiple_runs_of_same_graph(self):
        """Same warp graph traced repeatedly across iterations."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(10):
            arr = self._run_copy_node(
                "warp_a", "arr", "out", wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
            )
            arr = self._run_copy_node("warp_b", "arr", "out", arr)
            arr = self._run_copy_node("warp_c", "arr", "out", arr)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate,
            nodes=3,
            inputs=1,
            outputs=1,
            internal_connections=2,
        )

    def test_feedback_connections(self):
        """Warp nodes with a feedback edge across trace iterations."""
        leapp.start(name=self.TEST_GRAPH_NAME, verbose=False)
        loop_back = wp.array([0.0, 0.0, 0.0], dtype=wp.float32, device=self.DEVICE)
        external = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        for _ in range(2):
            in_a, loop = annotate.input_tensors(
                "node_a", {"inputA": external, "loop_back": loop_back}
            )
            with annotate.warp_op("node_a"):
                out_a = self._warp_add_arrays(in_a, loop)
            out_a = annotate.output_tensors("node_a", {"out_a": out_a}, export_with="onnx")

            out_b = self._run_copy_node("node_b", "in_b", "out_b", out_a)

            in_c, loop = annotate.input_tensors(
                "node_c", {"inputC": out_b, "loop_back": loop_back}
            )
            with annotate.warp_op("node_c"):
                out_c = self._warp_add_arrays(in_c, loop)
            out_c = annotate.output_tensors("node_c", {"out_c": out_c}, export_with="onnx")

            out_d = self._run_copy_node("node_d", "in_d", "out_d", out_c)
            loop_back = out_c

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate,
            nodes=4,
            inputs=1,
            outputs=1,
            internal_connections=3,
            feedback_connections=1,
        )

    def test_interleaved_traced_nodes_keep_forward_execution_order(self):
        """Completed warp node output consumed by a later node is forward flow."""
        seed = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        external = wp.array([10.0, 20.0, 30.0], dtype=wp.float32, device=self.DEVICE)

        leapp.start(name=self.TEST_GRAPH_NAME)

        seed = annotate.input_tensors("node_a", {"seed": seed})

        from_b = self._run_add_scalar_node(
            "node_b", "external_input", "b_out", external, 5.0
        )

        from_b = annotate.input_tensors("node_a", {"from_b": from_b})
        final_output = wp.to_torch(seed) + wp.to_torch(from_b)
        annotate.output_tensors(
            "node_a", {"final_output": final_output}, export_with="onnx"
        )

        leapp.stop()
        leapp.compile_graph(visualize=False)

        data_flow, feedback_flow = self._pipeline_views()
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

        runtime_seed = torch.tensor([3.0, 4.0, 5.0], device=self.DEVICE)
        runtime_external = torch.tensor([7.0, 8.0, 9.0], device=self.DEVICE)
        expected_output = runtime_seed + (runtime_external + 5.0)
        self.verify_inference_manager(
            source_inputs={
                "node_a/seed": runtime_seed,
                "node_b/external_input": runtime_external,
            },
            source_outputs={"node_a/final_output": expected_output},
        )

    def test_warp_tag_persistence_through_operations(self):
        """Tags survive chained warp nodes and buffer hand-offs."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        arr = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        out_a1 = out_a2 = None
        for _ in range(2):
            in_a = annotate.input_tensors("node_a", {"in_a": arr})
            with annotate.warp_op("node_a"):
                out_a1 = wp.empty_like(in_a)
                wp.copy(out_a1, in_a)
                out_a2 = wp.empty_like(in_a)
                wp.launch(
                    self.kernels.add_scalar,
                    dim=in_a.size,
                    inputs=[in_a, wp.float32(1.0)],
                    outputs=[out_a2],
                )
            out_a1, out_a2 = annotate.output_tensors(
                "node_a", {"out_a1": out_a1, "out_a2": out_a2}, export_with="onnx"
            )

        buffer_b = wp.empty_like(out_a1)
        wp.copy(buffer_b, out_a1)
        annotate.mirror_leapp_tags(out_a1, buffer_b)
        out_b = self._run_copy_node("node_b", "in_b", "out_b", buffer_b)

        buffer_c = wp.empty_like(out_a2)
        wp.copy(buffer_c, out_a2)
        annotate.mirror_leapp_tags(out_a2, buffer_c)
        out_c = self._run_copy_node("node_c", "in_c", "out_c", buffer_c)

        out_d = self._run_copy_node("node_d", "in_d", "out_d", out_c)
        for _ in range(2):
            in_e1, in_e2 = annotate.input_tensors(
                "node_e", {"in_e1": out_d, "in_e2": out_b}
            )
            with annotate.warp_op("node_e"):
                final = wp.empty_like(in_e1)
                wp.copy(final, in_e1)
            annotate.output_tensors("node_e", {"final": final}, export_with="onnx")

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate,
            nodes=5,
            inputs=1,
            outputs=2,
            internal_connections=4,
            feedback_connections=0,
        )

    def test_mirror_leapp_tags_with_inplace_assignment(self):
        """mirror_leapp_tags with in-place wp.copy between warp nodes."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        out_a = self._run_copy_node("node_a", "in_a", "out_a", wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE))
        buffer = wp.empty_like(out_a)
        wp.copy(buffer, out_a)
        annotate.mirror_leapp_tags(out_a, buffer)

        out_b = self._run_copy_node("node_b", "in_b", "out_b", buffer)
        out_c = self._run_copy_node("node_c", "in_c", "out_c", out_b)

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate,
            nodes=3,
            inputs=1,
            outputs=1,
            internal_connections=2,
        )

    def test_mirror_leapp_tags_with_preallocated_buffer(self):
        """mirror_leapp_tags with a reusable buffer object between warp nodes."""

        class WarpBuffer:
            def __init__(self):
                self._buffer = wp.zeros(3, dtype=wp.float32, device=TestWarpConnectionCase.DEVICE)

            def copy_to_buffer(self, data):
                wp.copy(self._buffer, data)
                annotate.mirror_leapp_tags(data, self._buffer)
                return self._buffer

        processor = WarpBuffer()
        leapp.start(name=self.TEST_GRAPH_NAME)

        upstream = self._run_copy_node(
            "upstream", "in_a", "out_a", wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        )
        buffered = processor.copy_to_buffer(upstream)
        processed = self._run_copy_node("process", "in_b", "out_b", buffered)
        self._run_copy_node("downstream", "in_c", "out_c", processed)

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate,
            nodes=3,
            inputs=1,
            outputs=1,
            internal_connections=2,
        )

    def test_mirror_leapp_tags_multiple_buffers(self):
        """mirror_leapp_tags with multiple wp.array buffers between warp nodes."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        arr = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        out_a1 = out_a2 = None
        for _ in range(2):
            in_a = annotate.input_tensors("node_a", {"in_a": arr})
            with annotate.warp_op("node_a"):
                out_a1 = wp.empty_like(in_a)
                wp.copy(out_a1, in_a)
                out_a2 = wp.empty_like(in_a)
                wp.launch(
                    self.kernels.add_scalar,
                    dim=in_a.size,
                    inputs=[in_a, wp.float32(1.0)],
                    outputs=[out_a2],
                )
            out_a1, out_a2 = annotate.output_tensors(
                "node_a", {"out_a1": out_a1, "out_a2": out_a2}, export_with="onnx"
            )

        buffer1 = wp.empty_like(out_a1)
        buffer2 = wp.empty_like(out_a2)
        wp.copy(buffer1, out_a1)
        wp.copy(buffer2, out_a2)
        annotate.mirror_leapp_tags(out_a1, buffer1)
        annotate.mirror_leapp_tags(out_a2, buffer2)

        out_b1 = out_b2 = None
        for _ in range(2):
            in_b1, in_b2 = annotate.input_tensors(
                "node_b", {"in_b1": buffer1, "in_b2": buffer2}
            )
            with annotate.warp_op("node_b"):
                out_b1 = wp.empty_like(in_b1)
                wp.copy(out_b1, in_b1)
                out_b2 = wp.empty_like(in_b2)
                wp.copy(out_b2, in_b2)
            out_b1, out_b2 = annotate.output_tensors(
                "node_b", {"out_b1": out_b1, "out_b2": out_b2}, export_with="onnx"
            )

        for _ in range(2):
            in_c1, in_c2 = annotate.input_tensors(
                "node_c", {"in_c1": out_b1, "in_c2": out_b2}
            )
            with annotate.warp_op("node_c"):
                final = self._warp_add_arrays(in_c1, in_c2)
            annotate.output_tensors("node_c", {"final": final}, export_with="onnx")

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate,
            nodes=3,
            inputs=1,
            outputs=1,
            internal_connections=4,
        )

    def test_mirror_leapp_tags_preserves_tag_through_chain(self):
        """mirror_leapp_tags preserves tags through a long warp node chain."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        out1 = self._run_copy_node(
            "source", "in_a", "out_a", wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        )

        buffer1 = wp.empty_like(out1)
        wp.copy(buffer1, out1)
        annotate.mirror_leapp_tags(out1, buffer1)
        out2 = self._run_copy_node("process1", "in_b", "out_b", buffer1)

        buffer2 = wp.empty_like(out2)
        wp.copy(buffer2, out2)
        annotate.mirror_leapp_tags(out2, buffer2)
        out3 = self._run_copy_node("process2", "in_c", "out_c", buffer2)

        self._run_copy_node("sink", "in_d", "out_d", out3)

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_num_connections(
            annotate,
            nodes=4,
            inputs=1,
            outputs=1,
            internal_connections=3,
        )

    def test_mirror_leapp_tags_noop_outside_tracing(self):
        """mirror_leapp_tags safely no-ops outside of tracing."""
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        target = wp.zeros(3, dtype=wp.float32, device=self.DEVICE)
        wp.copy(target, source)

        annotate.mirror_leapp_tags(source, target)

        self.assertTrue(
            torch.allclose(wp.to_torch(source), wp.to_torch(target))
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
