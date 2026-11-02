#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Sizes of data-dependent tensors stay symbolic in exported graphs.

``torch.nonzero``, boolean masks and similar ops return tensors whose size
depends on input values. Reading such a size (``.shape``, ``.size()``,
``numel()``, or ``torch.tensor(x.shape)``) must not bake the traced value into
the export: a later input with a different count would fail. ``len()`` always
returns a plain int in Python and cannot stay symbolic.
"""

import torch

import leapp
from leapp import InferenceManager, TensorSemantics
from leapp.leapp import _MANAGER as annotate
from leapp.leapp_graph.datatypes import TracedInt
from tests.functional_tests.base import LEAPPFunctionalTestBase

# Two positives, then three: the traced count differs between samples.
SAMPLES = [torch.tensor([[1.0, -1.0, 2.0]]), torch.tensor([[1.0, 3.0, 2.0]])]
UNSEEN = torch.tensor([[-1.0, -2.0, 5.0]])  # one positive


def _shape_index(x):
    idx = torch.nonzero(x > 0)
    picked = x[idx[:, 0], idx[:, 1]]
    return (picked.reshape(idx.shape[0], 1) * 2).sum(dim=0, keepdim=True)


def _size_arithmetic(x):
    idx = torch.nonzero(x > 0)
    n = idx.size(0)
    # Integer arithmetic on traced sizes is recorded too.
    picked = x[idx[:, 0], idx[:, 1]].reshape(idx.numel() // 2, (n + 1) - n)
    return picked.sum(dim=0, keepdim=True) / idx.shape[0]


def _boolean_mask(x):
    selected = x[x > 0]
    return selected.reshape(1, selected.shape[0]).sum(dim=-1, keepdim=True)


def _shape_tensor(x):
    idx = torch.nonzero(x > 0)
    count = torch.tensor(idx.shape)[0:1].to(x.dtype)
    return count.reshape(1, 1)


class TestDataDependentShapes(LEAPPFunctionalTestBase):
    def _export_and_check(self, fn):
        leapp.start(self.TEST_GRAPH_NAME, max_cached_io=len(SAMPLES))
        for sample in SAMPLES:
            x = annotate.input_tensors("node", [TensorSemantics("x", sample)])
            annotate.output_tensors(
                "node",
                [TensorSemantics("y", fn(x))],
                export_with="onnx-torchscript",
                backend_params={"opset_version": 17},
            )
        leapp.stop()
        # Validation replays every cached sample, so a baked count fails here.
        leapp.compile_graph(visualize=False, validate=True, strict=True)

        manager = InferenceManager(
            f"{self.TEST_GRAPH_NAME}/{self.TEST_GRAPH_NAME}.yaml"
        )
        for value in (*SAMPLES, UNSEEN):
            actual = torch.as_tensor(manager.run_policy({"node/x": value})["node/y"])
            torch.testing.assert_close(actual.float(), fn(value).float())

    def test_reshape_with_nonzero_count(self):
        self._export_and_check(_shape_index)

    def test_size_numel_and_arithmetic(self):
        self._export_and_check(_size_arithmetic)

    def test_boolean_mask_count(self):
        self._export_and_check(_boolean_mask)

    def test_tensor_built_from_shape(self):
        self._export_and_check(_shape_tensor)

    def test_static_shapes_stay_plain_ints(self):
        leapp.start(self.TEST_GRAPH_NAME)
        x = annotate.input_tensors("node", [TensorSemantics("x", torch.zeros(2, 3))])
        self.assertNotIsInstance(x.shape[0], TracedInt)
        self.assertNotIsInstance(x.size(1), TracedInt)
        idx = torch.nonzero(x >= 0)
        self.assertIsInstance(idx.shape[0], TracedInt)
        leapp.stop()
