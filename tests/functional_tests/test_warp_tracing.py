#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import contextlib
import unittest

import leapp
import numpy as np
import torch
import warp as wp
from leapp.leapp import _MANAGER as annotate


from .base import LEAPPFunctionalTestBase
from tests.warp_support import WarpTestCase



class TestWarpOp(WarpTestCase, LEAPPFunctionalTestBase):
    def _run_torch_to_warp_to_torch_node(self, device):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = torch.tensor([1.0, 2.0, 3.0], device=device)

        for _ in range(2):
            tensor = annotate.input_tensors("node_a", {"in_a": source})
            tensor = tensor * 2.0
            array = wp.from_torch(tensor)
            with annotate.warp_op("node_a", device=device):
                added = wp.empty_like(array)
                wp.launch(
                    self.kernels.add_scalar,
                    dim=array.size,
                    inputs=[array, wp.float32(2.0)],
                    outputs=[added],
                    device=device,
                )
            out = wp.to_torch(added) * 2.0 + 1.0
            annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        return node

    def _run_warp_to_torch_to_warp_node(self, device):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=device)

        for _ in range(2):
            array = annotate.input_tensors("node_a", {"in_a": source})
            with annotate.warp_op("node_a", device=device):
                added = wp.empty_like(array)
                wp.launch(
                    self.kernels.add_scalar,
                    dim=array.size,
                    inputs=[array, wp.float32(2.0)],
                    outputs=[added],
                    device=device,
                )
            tensor = wp.to_torch(added) * 2.0 + 1.0
            converted = wp.from_torch(tensor)
            with annotate.warp_op("node_a", device=device):
                out = wp.empty_like(converted)
                wp.copy(out, converted)
            annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        return node

    def test_single_warp_op_two_pass_compiles(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)

        for _ in range(2):
            arr = annotate.input_tensors("node_a", {"in_a": source})
            with annotate.warp_op("node_a"):
                out = wp.empty_like(arr)
                wp.copy(out, arr)
            annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(len(node.warp_segments), 1)
        self.assertEqual(node.warp_segments[0].runner_name, "warp_segment_0")
        self.assertIsNotNone(node.warp_segments[0].apic_graph)
        self.verify_all_models_exist("node_a")

    def test_multiple_warp_ops_in_one_node_capture_in_discovery_order(self):
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

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(
            [segment.runner_name for segment in node.warp_segments],
            ["warp_segment_0", "warp_segment_1"],
        )
        self.assertTrue(all(segment.apic_graph is not None for segment in node.warp_segments))
        self.verify_all_models_exist("node_a")

    def test_single_warp_op_can_produce_multiple_outputs(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)

        for _ in range(2):
            arr = annotate.input_tensors("node_a", {"in_a": source})
            with annotate.warp_op("node_a"):
                out_a = wp.empty_like(arr)
                wp.copy(out_a, arr)
                out_b = wp.empty_like(arr)
                wp.launch(
                    self.kernels.add_scalar,
                    dim=arr.size,
                    inputs=[arr, wp.float32(1.0)],
                    outputs=[out_b],
                )
            annotate.output_tensors(
                "node_a",
                {"out_a": out_a, "out_b": out_b},
                export_with="onnx",
            )

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=2)
        self.assertEqual(len(node.warp_segments), 1)
        self.verify_all_models_exist("node_a")

    def test_cuda_torch_to_warp_to_torch_in_one_node(self):
        node = self._run_torch_to_warp_to_torch_node(self.DEVICE)
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(len(node.warp_segments), 1)
        self.verify_all_models_exist("node_a")

    def test_cuda_warp_to_torch_to_warp_in_one_node(self):
        node = self._run_warp_to_torch_to_warp_node(self.DEVICE)
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(len(node.warp_segments), 2)
        self.verify_all_models_exist("node_a")

    def test_cpu_torch_to_warp_to_torch_fails_validation(self):
        node = self._run_torch_to_warp_to_torch_node("cpu")

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(len(node.warp_segments), 1)
        with self.assertRaisesRegex(Exception, "Model validation failed"):
            leapp.compile_graph(visualize=False)

    def test_cpu_warp_to_torch_to_warp_fails_validation(self):
        node = self._run_warp_to_torch_to_warp_node("cpu")

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(len(node.warp_segments), 2)
        with self.assertRaisesRegex(Exception, "Model validation failed"):
            leapp.compile_graph(visualize=False)

    def test_numpy_to_warp_to_numpy_in_one_node(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        for _ in range(2):
            traced = annotate.input_tensors("node_a", {"in_a": source})
            array = wp.from_numpy(traced * 2.0)
            added = wp.empty_like(array)
            with annotate.warp_op("node_a"):
                wp.launch(
                    self.kernels.add_scalar,
                    dim=array.size,
                    inputs=[array, wp.float32(2.0)],
                    outputs=[added],
                )
            out = added.numpy() * 2.0 + 1.0
            annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(len(node.warp_segments), 2)
        self.verify_all_models_exist("node_a")

    def test_numpy_to_warp_to_numpy_with_capture_allocation_in_one_node(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        for _ in range(2):
            traced = annotate.input_tensors("node_a", {"in_a": source})
            array = wp.from_numpy(traced * 2.0)
            with annotate.warp_op("node_a"):
                added = wp.empty_like(array)
                wp.launch(
                    self.kernels.add_scalar,
                    dim=array.size,
                    inputs=[array, wp.float32(2.0)],
                    outputs=[added],
                )
            out = added.numpy() * 2.0 + 1.0
            annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(len(node.warp_segments), 1)
        self.verify_all_models_exist("node_a")

    def test_multiple_numpy_origin_warp_segments_allocate_inside_one_node(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        for _ in range(2):
            current = annotate.input_tensors("node_a", {"in_a": source})
            for segment_index in range(4):
                array = wp.from_numpy(current + float(segment_index))
                with annotate.warp_op("node_a"):
                    current_warp = wp.empty_like(array)
                    wp.launch(
                        self.kernels.add_scalar,
                        dim=array.size,
                        inputs=[array, wp.float32(segment_index + 1)],
                        outputs=[current_warp],
                    )
                current = current_warp.numpy()
            annotate.output_tensors("node_a", {"out_a": current}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(len(node.warp_segments), 4)
        self.verify_all_models_exist("node_a")

    def test_warp_to_numpy_to_warp_in_one_node(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)

        for _ in range(2):
            array = annotate.input_tensors("node_a", {"in_a": source})
            copied = wp.empty_like(array)
            with annotate.warp_op("node_a"):
                wp.copy(copied, array)
            converted = wp.from_numpy(copied.numpy() * 2.0 + 1.0)
            out = wp.empty_like(converted)
            with annotate.warp_op("node_a"):
                wp.launch(
                    self.kernels.add_scalar,
                    dim=converted.size,
                    inputs=[converted, wp.float32(2.0)],
                    outputs=[out],
                )
            annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(len(node.warp_segments), 4)
        self.verify_all_models_exist("node_a")

    def test_compile_fails_when_second_warp_pass_never_runs(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        arr = annotate.input_tensors("node_a", {"in_a": source})
        with annotate.warp_op("node_a"):
            out = wp.empty_like(arr)
            wp.copy(out, arr)
        annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")
        node = annotate.get_nodes()["node_a"]
        self.verify_node_io(node, inputs=1, outputs=1)
        leapp.stop()

        with self.assertRaisesRegex(Exception, "executed a second time"):
            leapp.compile_graph(visualize=False)

    def test_second_pass_with_extra_warp_op_fails(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)

        for include_extra in (False, True):
            arr = annotate.input_tensors("node_a", {"in_a": source})
            with annotate.warp_op("node_a"):
                out = wp.empty_like(arr)
                wp.copy(out, arr)
            if include_extra:
                with self.assertRaisesRegex(RuntimeError, "more regions than discovery"):
                    with annotate.warp_op("node_a"):
                        scaled = wp.empty_like(out)
                        wp.launch(
                            self.kernels.add_scalar,
                            dim=out.size,
                            inputs=[out, wp.float32(2.0)],
                            outputs=[scaled],
                        )
                        out = scaled
            else:
                annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")
        node = annotate.get_nodes()["node_a"]
        self.verify_node_io(node, inputs=1, outputs=1)
        leapp.stop()

    def test_second_pass_with_divergent_warp_calls_fails(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)

        for use_launch in (False, True):
            arr = annotate.input_tensors("node_a", {"in_a": source})
            error_ctx = (
                self.assertRaisesRegex(RuntimeError, "diverged")
                if use_launch
                else contextlib.nullcontext()
            )
            with error_ctx:
                with annotate.warp_op("node_a"):
                    out = wp.empty_like(arr)
                    if use_launch:
                        wp.launch(
                            self.kernels.add_scalar,
                            dim=arr.size,
                            inputs=[arr, wp.float32(2.0)],
                            outputs=[out],
                        )
                    else:
                        wp.copy(out, arr)
            if not use_launch:
                annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        self.verify_node_io(node, inputs=1, outputs=1)
        self.assertEqual(node.warp_segments[0].call_qualnames[-1], "warp.copy")
        leapp.stop()

    def test_second_pass_missing_warp_op_leaves_node_pending(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)

        for include_second in (True, False):
            arr = annotate.input_tensors("node_a", {"in_a": source})
            with annotate.warp_op("node_a"):
                copied = wp.empty_like(arr)
                wp.copy(copied, arr)
            out = copied
            if include_second:
                with annotate.warp_op("node_a"):
                    out = wp.empty_like(copied)
                    wp.launch(
                        self.kernels.add_scalar,
                        dim=copied.size,
                        inputs=[copied, wp.float32(2.0)],
                        outputs=[out],
                    )
            annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")
        node = annotate.get_nodes()["node_a"]
        self.verify_node_io(node, inputs=1, outputs=1)
        leapp.stop()

        with self.assertRaisesRegex(Exception, "executed a second time"):
            leapp.compile_graph(visualize=False)


class TestWarpAutomaticSegmentDetection(WarpTestCase, LEAPPFunctionalTestBase):
    NODE_NAME = "node_a"

    def _run_single_node_operation(self, operation):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array(
            [1.0, 2.0, 3.0],
            dtype=wp.float32,
            device=self.DEVICE,
        )

        for _ in range(2):
            values = annotate.input_tensors(self.NODE_NAME, {"in_a": source})
            output = operation(values)
            annotate.output_tensors(
                self.NODE_NAME,
                {"out_a": output},
                export_with="onnx",
            )

        node = annotate.get_nodes()[self.NODE_NAME]
        leapp.stop()
        return node

    def test_sync_device_boundary_creates_two_segments(self):
        def operation(values):
            values = self._launch_add(values, 1.0)
            wp.synchronize_device(values.device)
            return self._launch_add(values, 2.0)

        node = self._run_single_node_operation(operation)
        leapp.compile_graph(visualize=False)
        self._assert_compiled_segments(node, 2)

    def test_torch_roundtrip_boundary_creates_two_segments(self):
        def operation(values):
            values = self._launch_add(values, 1.0)
            values = self._torch_roundtrip(values, 2.0)
            return self._launch_add(values, 3.0)

        node = self._run_single_node_operation(operation)
        leapp.compile_graph(visualize=False)
        self._assert_compiled_segments(node, 2)

    def test_numpy_roundtrip_boundary_creates_two_segments(self):
        def operation(values):
            values = self._launch_add(values, 1.0)
            values = self._numpy_roundtrip(values, 2.0)
            return self._launch_add(values, 3.0)

        node = self._run_single_node_operation(operation)
        leapp.compile_graph(visualize=False)
        self._assert_compiled_segments(node, 2)

    def test_warp_copy_stays_in_one_segment(self):
        def operation(values):
            values = self._launch_add(values, 1.0)
            copied = wp.empty_like(values)
            wp.copy(copied, values)
            return self._launch_add(copied, 2.0)

        node = self._run_single_node_operation(operation)
        leapp.compile_graph(visualize=False)
        self._assert_compiled_segments(node, 1)

    def test_mixed_sync_and_torch_roundtrip_creates_three_segments(self):
        def operation(values):
            values = self._launch_add(values, 1.0)
            wp.synchronize_device(values.device)
            values = self._launch_add(values, 2.0)
            values = self._torch_roundtrip(values, 3.0)
            return self._launch_add(values, 4.0)

        node = self._run_single_node_operation(operation)
        leapp.compile_graph(visualize=False)
        self._assert_compiled_segments(node, 3)

    def test_loop_boundary_and_torch_roundtrip_creates_three_segments(self):
        def operation(values):
            for index in range(2):
                values = self._launch_add(values, float(index + 1))
                if index == 0:
                    wp.synchronize_device(values.device)
            values = self._torch_roundtrip(values, 3.0)
            return self._launch_add(values, 4.0)

        node = self._run_single_node_operation(operation)
        leapp.compile_graph(visualize=False)
        self._assert_compiled_segments(node, 3)

    def test_manual_warp_op_between_automatic_segments_creates_three_segments(self):
        def operation(values):
            values = self._launch_add(values, 1.0)
            wp.synchronize_device(values.device)
            values = self._launch_manual_add(values, 2.0)
            values = self._torch_roundtrip(values, 3.0)
            return self._launch_add(values, 4.0)

        node = self._run_single_node_operation(operation)
        leapp.compile_graph(visualize=False)
        self._assert_compiled_segments(node, 3)

    def test_manual_warp_op_before_automatic_segments_creates_three_segments(self):
        def operation(values):
            values = self._launch_manual_add(values, 1.0)
            values = self._torch_roundtrip(values, 2.0)
            values = self._launch_add(values, 3.0)
            wp.synchronize_device(values.device)
            return self._launch_add(values, 4.0)

        node = self._run_single_node_operation(operation)

        leapp.compile_graph(visualize=False)
        self._assert_compiled_segments(node, 3)


if __name__ == "__main__":
    unittest.main()
