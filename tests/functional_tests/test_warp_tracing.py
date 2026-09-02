#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import contextlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import leapp
import numpy as np
import torch
import warp as wp
from leapp.leapp import _MANAGER as annotate
from leapp.leapp_graph.custom_operator_registry.warp_operator.metadata import (
    build_runtime_metadata,
)
from leapp.leapp_graph.datatypes.warp.warp_segment import WarpTensorRef
from leapp.utils.tensor_description import TensorDescription


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

    def test_persistent_warp_state_buffer_survives_both_passes(self):
        """A Warp buffer reused across steps is re-declared on the second pass.

        Declared state is the only way a Warp array can legally span both
        passes: its carrier points at the graph discovery built, and the
        capture pass replaces that graph.
        """
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        state = wp.zeros(3, dtype=wp.float32, device=self.DEVICE)

        for _ in range(2):
            arr = annotate.input_tensors("node_a", {"in_a": source})
            carried = annotate.state_tensors("node_a", {"state": state})
            with annotate.warp_op("node_a"):
                wp.launch(
                    self.kernels.increment_in_place,
                    dim=carried.size,
                    inputs=[carried],
                    device=self.DEVICE,
                )
                out = wp.empty_like(arr)
                wp.copy(out, arr)
            annotate.update_state("node_a", {"state": carried})
            annotate.output_tensors("node_a", {"out_a": out}, export_with="onnx")

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=2, outputs=2)
        self.assertEqual(len(node.warp_segments), 1)
        self.assertIsNotNone(node.warp_segments[0].apic_graph)
        self.verify_all_models_exist("node_a")

    def test_persistent_torch_buffer_survives_both_passes(self):
        """Plain torch buffers written inside a Warp node shed stale provenance.

        Torch promotes a plain destination in place, so a buffer reused across
        steps reaches the capture pass still holding the graph discovery built.
        ``copy_`` and ``__setitem__`` are the writes that promote, so both have
        to shed that provenance and re-promote against the new graph.
        """
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        copied = torch.zeros(3, device=self.DEVICE)
        assigned = torch.zeros(3, device=self.DEVICE)

        for _ in range(2):
            arr = annotate.input_tensors("node_a", {"in_a": source})
            with annotate.warp_op("node_a"):
                out = wp.empty_like(arr)
                wp.copy(out, arr)
            tensor = wp.to_torch(out)
            copied.copy_(tensor)
            assigned[:] = tensor
            annotate.output_tensors(
                "node_a",
                {"out_a": copied * 2.0, "out_b": assigned + 1.0},
                export_with="onnx",
            )

        node = annotate.get_nodes()["node_a"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=2)
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


class TestWarpCompoundType(WarpTestCase, LEAPPFunctionalTestBase):
    def test_compound_storage_layout_is_derived_from_warp_dtype(self):
        vec5 = wp.types.vector(length=5, dtype=wp.float32)
        cases = (
            (wp.float32, (), (2, 4)),
            (wp.vec3, (3,), (2, 4, 3)),
            (wp.quat, (4,), (2, 4, 4)),
            (wp.mat33, (3, 3), (2, 4, 3, 3)),
            (vec5, (5,), (2, 4, 5)),
        )

        for dtype, component_shape, storage_shape in cases:
            with self.subTest(dtype=dtype):
                value = wp.zeros((2, 4), dtype=dtype, device="cpu")
                ref = WarpTensorRef.from_value("value", value)

                self.assertEqual(ref.shape, (2, 4))
                self.assertEqual(ref.component_shape, component_shape)
                self.assertEqual(ref.storage_shape, storage_shape)
                self.assertEqual(ref.storage_dtype, "float32")

    def test_runtime_metadata_contains_storage_and_logical_layouts(self):
        value = wp.zeros((2, 4), dtype=wp.vec3, device="cpu")
        ref = WarpTensorRef.from_value("positions", value)

        metadata = build_runtime_metadata(
            segment=None,
            input_refs=[ref],
            output_refs=[ref],
            output_shapes=[list(ref.storage_shape)],
            output_dtypes=[ref.storage_dtype],
            output_mask=[True],
        )

        self.assertNotIn("schema_version", metadata)
        for spec in (metadata["inputs"][0], metadata["outputs"][0]):
            self.assertEqual(spec["dtype"], "float32")
            self.assertEqual(spec["shape"], [2, 4, 3])
            self.assertEqual(spec["num_bytes"], 2 * 4 * 3 * 4)
            self.assertEqual(spec["warp_dtype"], "vec3f")
            self.assertEqual(spec["warp_shape"], [2, 4])
            self.assertEqual(spec["component_shape"], [3])

    def test_tensor_description_uses_expanded_torch_storage_layout(self):
        value = wp.zeros((2, 4), dtype=wp.vec3, device="cpu")

        description = TensorDescription("positions", value)

        self.assertEqual(description.dtype, "float32")
        self.assertEqual(tuple(description.shape), (2, 4, 3))
        self.assertEqual(tuple(description.value.shape), (2, 4, 3))

    def test_compound_warp_array_is_accepted_by_input_tensors(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        value = wp.zeros((2, 4), dtype=wp.vec3, device="cpu")

        try:
            traced = annotate.input_tensors("node_a", {"positions": value})
            node = annotate.get_nodes()["node_a"]

            self.assertEqual(traced.shape, (2, 4))
            self.assertEqual(traced.dtype, wp.vec3)
            self.assertEqual(node.inputs[0].dtype, "float32")
            self.assertEqual(tuple(node.inputs[0].shape), (2, 4, 3))

            annotate.output_tensors(
                "node_a",
                {"positions": traced},
                export_with=None,
            )
        finally:
            leapp.stop()

    def test_compound_warp_array_crosses_annotation_boundaries(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = wp.zeros((2, 4), dtype=wp.vec3, device=self.DEVICE)

        for _ in range(2):
            array = annotate.input_tensors("node_a", {"positions": source})
            self.assertEqual(array.shape, (2, 4))
            self.assertEqual(array.dtype, wp.vec3)

            with annotate.warp_op("node_a"):
                output = wp.empty_like(array)
                wp.copy(output, array)

            annotate.output_tensors(
                "node_a",
                {"positions_next": output},
                export_with="onnx",
            )

        node = annotate.get_nodes()["node_a"]
        segment = node.warp_segments[0]
        input_ref = next(iter(segment.input_refs.values()))
        output_ref = next(iter(segment.output_refs.values()))

        self.assertEqual(tuple(node.inputs[0].shape), (2, 4, 3))
        self.assertEqual(node.inputs[0].dtype, "float32")
        self.assertEqual(tuple(node.outputs[0].shape), (2, 4, 3))
        self.assertEqual(node.outputs[0].dtype, "float32")
        self.assertEqual(input_ref.storage_shape, (2, 4, 3))
        self.assertEqual(output_ref.storage_shape, (2, 4, 3))
        self.assertEqual(input_ref.storage_dtype, "float32")
        self.assertEqual(output_ref.storage_dtype, "float32")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.verify_all_models_exist("node_a")

    def test_multiple_compound_warp_arrays_cross_annotation_boundaries(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        positions_source = wp.zeros((2, 4), dtype=wp.vec3, device=self.DEVICE)
        rotations_source = wp.zeros((2, 4), dtype=wp.quat, device=self.DEVICE)
        frames_source = wp.zeros((2, 4), dtype=wp.mat33, device=self.DEVICE)

        try:
            positions = annotate.input_tensors(
                "node_a", {"positions": positions_source}
            )
            rotations = annotate.input_tensors(
                "node_a", {"rotations": rotations_source}
            )
            frames = annotate.input_tensors("node_a", {"frames": frames_source})

            annotate.output_tensors(
                "node_a",
                {
                    "positions_next": positions,
                    "rotations_next": rotations,
                    "frames_next": frames,
                },
                export_with=None,
            )
        finally:
            leapp.stop()

        node = annotate.get_nodes()["node_a"]

        self.assertEqual(
            [tuple(description.shape) for description in node.inputs],
            [(2, 4, 3), (2, 4, 4), (2, 4, 3, 3)],
        )
        self.assertEqual(
            [tuple(description.shape) for description in node.outputs],
            [(2, 4, 3), (2, 4, 4), (2, 4, 3, 3)],
        )
        self.assertEqual(
            [description.dtype for description in node.inputs],
            ["float32", "float32", "float32"],
        )
        self.assertEqual(
            [description.dtype for description in node.outputs],
            ["float32", "float32", "float32"],
        )
        self.verify_node_io(node, inputs=3, outputs=3)

    def test_torch_input_can_be_viewed_as_compound_warp_dtype(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source = torch.zeros((2, 4, 3), dtype=torch.float32, device=self.DEVICE)

        for _ in range(2):
            tensor = annotate.input_tensors("node_a", {"positions": source})
            array = wp.from_torch(tensor, dtype=wp.vec3)

            with annotate.warp_op("node_a"):
                output = wp.empty_like(array)
                wp.copy(output, array)

            annotate.output_tensors(
                "node_a",
                {"positions_next": output},
                export_with="onnx",
            )

        node = annotate.get_nodes()["node_a"]
        output_ref = next(iter(node.warp_segments[0].output_refs.values()))
        self.assertEqual(output_ref.storage_shape, (2, 4, 3))
        self.assertEqual(output_ref.storage_dtype, "float32")

        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=1, outputs=1)
        self.verify_all_models_exist("node_a")



class TestWarpInference(WarpTestCase, LEAPPFunctionalTestBase):
    NODE_NAME = "node_a"

    def test_pt2_runner_executes_distinct_warp_segments(self):
        def operation(values):
            values = self._launch_add(values, 1.0)
            values = self._torch_roundtrip(values, 2.0)
            return self._launch_add(values, 3.0)

        node = self._run_single_node_operation(operation, export_with="pt2")
        leapp.compile_graph(visualize=False)
        self._assert_compiled_segments(node, 2)
        self.verify_inference_manager(
            source_inputs={
                f"{self.NODE_NAME}/in_a": torch.tensor(
                    [1.0, 2.0, 3.0], device=self.DEVICE
                )
            },
            source_outputs={
                f"{self.NODE_NAME}/out_a": torch.tensor(
                    [7.0, 8.0, 9.0], device=self.DEVICE
                )
            },
        )

    def test_pt2_runner_requires_custom_op_library(self):
        self._run_single_node_operation(
            lambda values: self._launch_add(values, 1.0),
            export_with="pt2",
        )
        with tempfile.TemporaryDirectory() as cache_dir, mock.patch.dict(
            os.environ, {"XDG_CACHE_HOME": cache_dir}
        ):
            os.environ.pop("LEAPP_WARP_PT2_CUSTOM_OP_LIBRARY")
            with self.assertRaisesRegex(
                FileNotFoundError, "leapp-build-warp-runtime"
            ):
                leapp.compile_graph(visualize=False)

    def test_onnx_runner_requires_custom_op_library(self):
        from leapp import InferenceManager

        self._run_single_node_operation(
            lambda values: self._launch_add(values, 1.0),
        )
        leapp.compile_graph(visualize=False)
        with tempfile.TemporaryDirectory() as cache_dir, mock.patch.dict(
            os.environ, {"XDG_CACHE_HOME": cache_dir}
        ):
            os.environ.pop("LEAPP_WARP_ONNX_CUSTOM_OP_LIBRARY")
            manager = InferenceManager(
                f"{self.TEST_GRAPH_NAME}/{self.TEST_GRAPH_NAME}.yaml"
            )
            with self.assertRaisesRegex(
                FileNotFoundError, "leapp-build-warp-runtime"
            ):
                manager.run_policy(manager.get_mock_input())

    def test_one_array_read_by_three_launches_is_one_segment_input(self):
        """A buffer several launches read is one runner argument, not several.

        Inputs are recorded per launch, so this only holds because a launch
        skips an allocation the segment already stands for. The first launch's
        post-call walk registers every array it touched, which is what the
        later launches recognize.
        """
        def operation(values):
            first = self._launch_add(values, 1.0)
            second = self._launch_add(values, 2.0)
            output = wp.empty_like(values)
            wp.launch(
                self.kernels.average_three,
                dim=values.size,
                inputs=[first, second, values],
                outputs=[output],
                device=values.device,
            )
            return output

        node = self._run_single_node_operation(operation)
        leapp.compile_graph(visualize=False)
        self._assert_compiled_segments(node, 1)

        segment = node.warp_segments[0]
        self.assertEqual(
            len(segment.input_refs), 1,
            "one buffer read by three launches produced "
            f"{len(segment.input_refs)} segment inputs")
        self.verify_inference_manager(
            source_inputs={f"{self.NODE_NAME}/in_a": torch.tensor(
                [1.0, 2.0, 3.0], device=self.DEVICE)},
            source_outputs={f"{self.NODE_NAME}/out_a": torch.tensor(
                [2.0, 3.0, 4.0], device=self.DEVICE)},
        )


class TestWarpAutomaticSegmentDetection(WarpTestCase, LEAPPFunctionalTestBase):
    NODE_NAME = "node_a"

    def test_same_node_interleaved_warp_arrays_stay_in_one_segment(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source1 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        source2 = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=self.DEVICE)

        for _ in range(2):
            array1 = annotate.input_tensors("node1", {"a": source1})
            array2 = annotate.input_tensors("node1", {"b": source2})

            array1 = self._launch_add(array1, 1.0)
            array2 = self._launch_add(array2, 1.0)
            array1 = self._launch_add(array1, 1.0)

            annotate.output_tensors(
                "node1",
                {"a": array1, "b": array2},
                export_with="onnx",
            )

        node = annotate.get_nodes()["node1"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node.has_pending_warp_segments)
        self.verify_node_io(node, inputs=2, outputs=2)
        self.assertEqual(len(node.warp_segments), 1)
        self.assertIsNotNone(node.warp_segments[0].apic_graph)
        self.verify_all_models_exist("node1")

    def test_interleaved_warp_nodes_split_on_context_switch(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source1 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        source2 = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=self.DEVICE)

        for _ in range(2):
            node1_array = annotate.input_tensors("node1", {"a": source1})
            node2_array = annotate.input_tensors("node2", {"b": source2})

            node1_array = self._launch_add(node1_array, 1.0)
            node2_array = self._launch_add(node2_array, 1.0)
            node1_array = self._launch_add(node1_array, 1.0)

            annotate.output_tensors("node1", {"a": node1_array}, export_with="onnx")
            annotate.output_tensors("node2", {"b": node2_array}, export_with="onnx")

        node1 = annotate.get_nodes()["node1"]
        node2 = annotate.get_nodes()["node2"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node1.has_pending_warp_segments)
        self.assertFalse(node2.has_pending_warp_segments)
        self.verify_node_io(node1, inputs=1, outputs=1)
        self.verify_node_io(node2, inputs=1, outputs=1)
        self.assertEqual(len(node1.warp_segments), 2)
        self.assertEqual(len(node2.warp_segments), 1)
        self.assertTrue(
            all(segment.apic_graph is not None for segment in node1.warp_segments)
        )
        self.assertTrue(
            all(segment.apic_graph is not None for segment in node2.warp_segments)
        )
        self.verify_all_models_exist("node1", "node2")

    def test_probe_style_interleaved_warp_nodes_split_on_context_switch(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source1 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        source2 = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=self.DEVICE)

        for _ in range(2):
            node1_array = annotate.input_tensors("node1", {"a": source1})
            node2_array = annotate.input_tensors("node2", {"b": source2})

            node1_array = self._launch_increment_in_place(node1_array)
            node2_array = self._launch_increment_in_place(node2_array)
            node1_array = self._launch_increment_in_place(node1_array)

            annotate.output_tensors("node1", {"a": node1_array}, export_with="onnx")
            annotate.output_tensors("node2", {"b": node2_array}, export_with="onnx")

        node1 = annotate.get_nodes()["node1"]
        node2 = annotate.get_nodes()["node2"]
        leapp.stop()
        leapp.compile_graph(visualize=False)

        self.assertFalse(node1.has_pending_warp_segments)
        self.assertFalse(node2.has_pending_warp_segments)
        self.verify_node_io(node1, inputs=1, outputs=1)
        self.verify_node_io(node2, inputs=1, outputs=1)
        self.assertEqual(len(node1.warp_segments), 2)
        self.assertEqual(len(node2.warp_segments), 1)
        self.verify_all_models_exist("node1", "node2")

    def test_single_warp_call_with_arrays_from_multiple_nodes_still_fails(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source1 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        source2 = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=self.DEVICE)
        output = wp.empty_like(source1)

        array1 = annotate.input_tensors("node1", {"a": source1})
        array2 = annotate.input_tensors("node2", {"b": source2})

        with self.assertRaisesRegex(ValueError, "different LEAPP trace contexts"):
            wp.launch(
                self.kernels.average_three,
                dim=array1.size,
                inputs=[array1, array2, array1],
                outputs=[output],
                device=array1.device,
            )
        leapp.stop()

    def test_context_switch_inside_explicit_warp_op_still_fails(self):
        leapp.start(name=self.TEST_GRAPH_NAME)
        source1 = wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device=self.DEVICE)
        source2 = wp.array([4.0, 5.0, 6.0], dtype=wp.float32, device=self.DEVICE)
        node1_array = annotate.input_tensors("node1", {"a": source1})
        node2_array = annotate.input_tensors("node2", {"b": source2})

        try:
            with self.assertRaisesRegex(RuntimeError, "active WarpOp is protected"):
                with annotate.warp_op("node1", device=self.DEVICE):
                    self._launch_add(node1_array, 1.0)
                    self._launch_add(node2_array, 1.0)
        finally:
            leapp.stop()

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


class TestGlobalPatchingAliasScan(WarpTestCase, LEAPPFunctionalTestBase):
    """Global patching reads attributes off every loaded module's namespace.

    Repointing an alias such as ``from warp import launch`` means asking each
    already-imported name whether Warp defined it, so the scan touches objects
    belonging to unrelated third-party libraries.
    """

    def test_scan_tolerates_a_lazy_import_placeholder(self):
        """A stub for a missing optional dependency must not abort ``leapp.start``.

        Libraries such as trimesh, which urdfpy imports, park an object in their
        namespace that answers every attribute with the ImportError it stands
        for. Asking it for a defining module raises ``ModuleNotFoundError``,
        which ``getattr``'s default does not absorb because the default only
        covers ``AttributeError``.
        """
        class ExceptionWrapper:
            def __init__(self, exception):
                object.__setattr__(self, "_exception", exception)

            def __getattribute__(self, name):
                raise object.__getattribute__(self, "_exception")

            def __call__(self, *args, **kwargs):
                raise object.__getattribute__(self, "_exception")

        module = types.ModuleType("leapp_test_absent_dependency")
        module.mesh_loader = ExceptionWrapper(
            ModuleNotFoundError("No module named 'networkx'"))
        sys.modules[module.__name__] = module
        self.addCleanup(sys.modules.pop, module.__name__, None)

        leapp.start(name=self.TEST_GRAPH_NAME, global_patching=True)
        leapp.stop()


if __name__ == "__main__":
    unittest.main()
