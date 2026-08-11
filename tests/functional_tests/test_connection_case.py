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

import numpy as np
import torch
import warp as wp

import leapp
from leapp.leapp import _MANAGER as annotate
from tests.warp_support import WarpTestCase

from .base import LEAPPFunctionalTestBase


@wp.kernel
def _warp_add_arrays_kernel(
    src1: wp.array(dtype=wp.float32),
    src2: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dst[i] = src1[i] + src2[i]


class ConnectivityTestBase(LEAPPFunctionalTestBase):
    """Shared assertions for backend-specific connectivity tests."""

    def verify_connectivity(
        self,
        *,
        nodes,
        internal_connections=0,
        inputs=1,
        outputs=1,
        feedback_connections=0,
    ):
        self.verify_num_connections(
            annotate,
            nodes=nodes,
            inputs=inputs,
            outputs=outputs,
            internal_connections=internal_connections,
            feedback_connections=feedback_connections,
        )

    def pipeline_views(self):
        return (
            {
                source: list(targets)
                for source, targets in annotate.detected_pipeline["data_flow"].items()
            },
            {
                source: list(targets)
                for source, targets in annotate.detected_pipeline[
                    "feedback_flow"
                ].items()
            },
        )


class TestConnectionCase(ConnectivityTestBase):

    def test_multiple_runs_of_same_graph(self):
        """tests the situation where the same graph is run multiple times"""
        @annotate.method(export_with="jit")
        def funcA(inputA: torch.Tensor):
            return inputA

        @annotate.method(export_with="jit")
        def funcC(inputB: torch.Tensor):
            return inputB+5.0

        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(10):
            outputA = funcA(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))
            outputA = annotate.input_tensors("blockA", {"outputA": outputA})
            outputB = outputA*2.
            annotate.output_tensors("blockA", {"outputB": outputB}, export_with="jit")
            funcC(outputB)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_connectivity(nodes=3, internal_connections=2)

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
        for _ in range(2):
            out_funcA = funcA(torch.tensor([1.0, 2.0, 3.0]), feedback_input)
            out_funcB = funcB(out_funcA)
            out_funcC = funcC(out_funcB, feedback_input)
            funcD(out_funcC)
            feedback_input = out_funcC

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_connectivity(
            nodes=4, internal_connections=3, feedback_connections=1)

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
        self.verify_connectivity(nodes=2, inputs=2, internal_connections=1)
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

    def test_mirror_leapp_tags_with_inplace_assignment(self):
        """mirror_leapp_tags with a single in-place buffer between nodes."""
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
        buffer = torch.zeros_like(out_funcA)
        buffer[:] = out_funcA
        annotate.mirror_leapp_tags(out_funcA, buffer)
        out_funcB = funcB(buffer)
        funcC(out_funcB)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_connectivity(nodes=3, internal_connections=2)

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
        
        # Transfer traced state for both
        annotate.mirror_leapp_tags(out_A1, buffer1)
        annotate.mirror_leapp_tags(out_A2, buffer2)
        
        out_B1, out_B2 = funcB(buffer1, buffer2)
        funcC(out_B1, out_B2)
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Verify connections: funcA (2 outputs) -> funcB (2 inputs, 2 outputs) -> funcC (2 inputs)
        self.verify_connectivity(nodes=3, internal_connections=4)

    def test_mirror_leapp_tags_preserves_output_port_through_chain(self):
        """Test that mirror_leapp_tags preserves output ports through a long chain of buffer operations"""
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
        sink_node(out3)
        leapp.stop()
        leapp.compile_graph(visualize=False)

        # Should maintain proper connections through all nodes
        self.verify_connectivity(nodes=4, internal_connections=3)

    def test_mirror_leapp_tags_noop_outside_tracing(self):
        """Test that mirror_leapp_tags safely no-ops outside of tracing"""
        source = torch.tensor([1.0, 2.0, 3.0])
        target = torch.zeros(3)
        target[:] = source
        
        # Should not raise an error when called outside tracing
        annotate.mirror_leapp_tags(source, target)
        
        # Verify data is still correct
        self.assertTrue(torch.allclose(source, target))


class TestEquivalentCopyKeepsOutputPort(LEAPPFunctionalTestBase):
    """Specifications for copies between nodes that should keep the output port.

    A finished output carries its producing node and output port, which is what
    builds the graph edge. An equivalent copy of that value holds the same data
    and so should present the same port, letting the next node connect without
    ``mirror_leapp_tags``. Every case here currently loses the port, which is the
    work Phase 3 takes on.
    """

    def _finished_output(self, value):
        """Trace one node that publishes ``value * 2`` and return its output."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        traced = annotate.input_tensors("producer", {"x": value})
        output = traced * 2.0
        annotate.output_tensors("producer", {"y": output}, export_with=None)
        return output

    @unittest.expectedFailure
    def test_expected_fail_torch_equivalent_ops_keep_output_port(self):
        """``clone``/``detach``/``contiguous``/``cpu`` on a finished torch output."""
        output = self._finished_output(torch.tensor([1.0, 2.0, 3.0]))
        # .cuda() is left out because the CI machine has no GPU.
        for op in ("clone", "detach", "contiguous", "cpu"):
            with self.subTest(op=op):
                self.assertEqual("y", getattr(output, op)().output_port)

    @unittest.expectedFailure
    def test_expected_fail_torch_raw_buffer_assignment_keeps_output_port(self):
        """Full-slice assignment into a raw preallocated torch buffer."""
        output = self._finished_output(torch.tensor([1.0, 2.0, 3.0]))
        buffer = torch.zeros_like(output)
        buffer[:] = output
        self.assertEqual("y", buffer.output_port)

    @unittest.expectedFailure
    def test_expected_fail_numpy_allocating_copy_keeps_output_port(self):
        """An allocating numpy copy of a finished numpy output."""
        output = self._finished_output(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertEqual("y", np.copy(output).output_port)

    @unittest.expectedFailure
    def test_expected_fail_numpy_traced_buffer_assignment_keeps_output_port(self):
        """Full-slice assignment into a numpy buffer that is already traced.

        A raw ``np.ndarray`` destination cannot be class-swapped, so only a
        destination that already carries traced state can take the port over.
        """
        output = self._finished_output(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        buffer = annotate.mirror_leapp_tags(
            output, np.zeros_like(np.asarray(output)))
        buffer[:] = output
        self.assertEqual("y", buffer.output_port)

    @unittest.expectedFailure
    def test_expected_fail_cross_backend_copy_keeps_output_port(self):
        """A finished numpy output handed to torch without changing its values."""
        output = self._finished_output(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertEqual("y", torch.as_tensor(np.asarray(output)).output_port)


class TestNumpyConnectionCase(ConnectivityTestBase):
    """NumPy equivalents of the torch connection/traced-state tests.

    NumPy callers must assign the return value of ``mirror_leapp_tags``: a raw
    ``np.ndarray`` target becomes a zero-copy ``TracedNpArray`` view rather than
    being class-swapped in place.
    """

    def _run_scale_node(self, node_name, input_name, output_name, src, scale):
        traced = annotate.input_tensors(node_name, {input_name: src})
        return annotate.output_tensors(
            node_name, {output_name: traced * scale}, export_with=None)

    def _run_add_node(self, node_name, input_name, output_name, src, value):
        traced = annotate.input_tensors(node_name, {input_name: src})
        return annotate.output_tensors(
            node_name, {output_name: traced + value}, export_with=None)

    def test_numpy_nodes_chain_via_output_ports(self):
        """Two numpy nodes connected by a published ndarray output -> input edge."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out_a = self._run_scale_node("node_a", "arr", "out_a", arr, 2.0)

        self.assertEqual(out_a.output_port, "out_a")
        self.assertIs(out_a.context_obj, annotate.nodes["node_a"])

        self._run_add_node("node_b", "out_a", "out_b", out_a, 1.0)

        leapp.stop()
        leapp.compile_graph(visualize=False)

        data_flow, feedback_flow = self.pipeline_views()
        self.assertEqual({"node_a/out_a": ["node_b/out_a"]}, data_flow)
        self.assertEqual({}, feedback_flow)
        self.verify_connectivity(nodes=2, internal_connections=1)

    def test_multiple_runs_of_same_graph(self):
        """Same numpy graph traced repeatedly across iterations."""
        leapp.start(name=self.TEST_GRAPH_NAME)
        for _ in range(10):
            arr = self._run_scale_node(
                "numpy_a", "arr", "out",
                np.array([1.0, 2.0, 3.0], dtype=np.float32), 2.0)
            arr = self._run_add_node("numpy_b", "arr", "out", arr, 1.0)
            arr = self._run_scale_node("numpy_c", "arr", "out", arr, 3.0)
        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_connectivity(nodes=3, internal_connections=2)

    def test_feedback_connections(self):
        """Numpy nodes with a feedback edge across trace iterations."""
        leapp.start(name=self.TEST_GRAPH_NAME, verbose=False)
        loop_back = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        external = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        for _ in range(2):
            in_a, loop = annotate.input_tensors(
                "node_a", {"inputA": external, "loop_back": loop_back})
            out_a = annotate.output_tensors(
                "node_a", {"out_a": in_a + loop}, export_with=None)

            out_b = self._run_scale_node("node_b", "in_b", "out_b", out_a, 1.0)

            in_c, loop = annotate.input_tensors(
                "node_c", {"inputC": out_b, "loop_back": loop_back})
            out_c = annotate.output_tensors(
                "node_c", {"out_c": in_c + loop}, export_with=None)

            self._run_scale_node("node_d", "in_d", "out_d", out_c, 1.0)
            loop_back = out_c

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_connectivity(
            nodes=4, internal_connections=3, feedback_connections=1)

    def test_mirror_leapp_tags_with_inplace_assignment(self):
        """mirror_leapp_tags with a single in-place buffer between numpy nodes."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        out_a = self._run_scale_node(
            "node_a", "in_a", "out_a",
            np.array([1.0, 2.0, 3.0], dtype=np.float32), 2.0)

        buffer = np.zeros_like(np.asarray(out_a))
        buffer[:] = out_a
        buffer = annotate.mirror_leapp_tags(out_a, buffer)

        out_b = self._run_add_node("node_b", "in_b", "out_b", buffer, 1.0)
        self._run_scale_node("node_c", "in_c", "out_c", out_b, 3.0)

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_connectivity(nodes=3, internal_connections=2)

    def test_mirror_leapp_tags_preserves_output_port_through_chain(self):
        """mirror_leapp_tags preserves output ports through a long numpy node chain."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        out1 = self._run_scale_node(
            "source", "in_a", "out_a",
            np.array([1.0, 2.0, 3.0], dtype=np.float32), 2.0)

        buffer1 = np.empty_like(np.asarray(out1))
        buffer1[:] = out1
        buffer1 = annotate.mirror_leapp_tags(out1, buffer1)
        out2 = self._run_add_node("process1", "in_b", "out_b", buffer1, 1.0)

        buffer2 = np.empty_like(np.asarray(out2))
        buffer2[:] = out2
        buffer2 = annotate.mirror_leapp_tags(out2, buffer2)
        out3 = self._run_scale_node("process2", "in_c", "out_c", buffer2, 3.0)

        self._run_scale_node("sink", "in_d", "out_d", out3, 1.0)

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_connectivity(nodes=4, internal_connections=3)

class TestWarpConnectionCase(WarpTestCase, ConnectivityTestBase):
    """Warp equivalents of the torch connection/traced-state tests in TestConnectionCase."""

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

    @unittest.expectedFailure
    def test_expected_fail_warp_full_copy_keeps_output_port(self):
        """A full ``wp.copy`` of a finished warp output should keep its port.

        Losing the port here disconnects the next warp node, which is the work
        Phase 3 takes on.
        """
        leapp.start(name=self.TEST_GRAPH_NAME)
        arr1 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        arr2 = self._run_copy_node("node_a", "arr1", "arr2", arr1)

        copied = wp.empty_like(arr2)
        wp.copy(copied, arr2)
        self.assertEqual("arr2", copied.output_port)

    def test_warp_nodes_chain_via_output_ports(self):
        """Two warp nodes connected by a published wp.array output -> input edge."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        arr1 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        arr2 = self._run_copy_node("node_a", "arr1", "arr2", arr1)

        self.assertEqual(arr2.output_port, "arr2")
        self.assertIs(arr2.context_obj, annotate.nodes["node_a"])

        self._run_copy_node("node_b", "arr2", "arr3", arr2)

        leapp.stop()
        leapp.compile_graph(visualize=False)

        data_flow, feedback_flow = self.pipeline_views()
        self.assertEqual({"node_a/arr2": ["node_b/arr2"]}, data_flow)
        self.assertEqual({}, feedback_flow)
        self.verify_connectivity(nodes=2, internal_connections=1)
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
        self.verify_connectivity(nodes=1)
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
        self.verify_connectivity(nodes=3, internal_connections=2)

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

            self._run_copy_node("node_d", "in_d", "out_d", out_c)
            loop_back = out_c

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_connectivity(
            nodes=4, internal_connections=3, feedback_connections=1)

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

        data_flow, feedback_flow = self.pipeline_views()
        self.assertEqual({"node_b/b_out": ["node_a/from_b"]}, data_flow)
        self.assertEqual({}, feedback_flow)
        self.verify_connectivity(nodes=2, inputs=2, internal_connections=1)
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

    def test_warp_output_port_persistence_through_operations(self):
        """Output ports survive chained warp nodes and buffer hand-offs."""
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
            in_e1, _in_e2 = annotate.input_tensors(
                "node_e", {"in_e1": out_d, "in_e2": out_b}
            )
            with annotate.warp_op("node_e"):
                final = wp.empty_like(in_e1)
                wp.copy(final, in_e1)
            annotate.output_tensors("node_e", {"final": final}, export_with="onnx")

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_connectivity(
            nodes=5, outputs=2, internal_connections=4)

    def test_mirror_leapp_tags_with_inplace_assignment(self):
        """mirror_leapp_tags with a single in-place wp.copy between warp nodes."""
        leapp.start(name=self.TEST_GRAPH_NAME)

        out_a = self._run_copy_node(
            "node_a", "in_a", "out_a",
            wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE),
        )
        buffer = wp.empty_like(out_a)
        wp.copy(buffer, out_a)
        annotate.mirror_leapp_tags(out_a, buffer)

        out_b = self._run_copy_node("node_b", "in_b", "out_b", buffer)
        self._run_copy_node("node_c", "in_c", "out_c", out_b)

        leapp.stop()
        leapp.compile_graph(visualize=False)
        self.verify_connectivity(nodes=3, internal_connections=2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
