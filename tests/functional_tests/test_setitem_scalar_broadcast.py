#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Assigning one value to several positions exports correctly.

``axis = torch.zeros_like(q[..., 0:3]); axis[..., 2] = 1`` is lowered to a flat
``index_put``. PyTorch broadcasts a one-element source over the selected
positions, but ``torch.onnx.export``'s constant folding does not: when every
input of the assignment is constant it fails with "shape '[n]' is invalid for
input of size 1". With traced indices, the number of selected positions
depends on the input values and differs between samples.
"""

import torch

import leapp
from leapp import InferenceManager, TensorSemantics
from leapp.leapp import _MANAGER as annotate

from tests.functional_tests.base import LEAPPFunctionalTestBase


# Seeded so the number of positive entries differs between every input.
_GENERATOR = torch.Generator().manual_seed(0)
SAMPLES = [torch.randn(1, 5, 4, generator=_GENERATOR) for _ in range(2)]
UNSEEN = torch.randn(1, 5, 4, generator=_GENERATOR)


def _constant_axis(q):
    # The heading-axis pattern from ProtoMotions' rotation helpers.
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1
    return q[..., 0:3] + axis


def _traced_scalar_value(q):
    out = torch.zeros_like(q)
    out[..., 1] = q[0, 0, 0]
    return out + q


def _constant_at_mask(q):
    out = q.clone()
    out[q > 0] = 1.0
    return out


def _traced_scalar_at_mask(q):
    out = torch.zeros_like(q)
    out[q > 0] = q[0, 0, 0]
    return out + q


class TestSetitemScalarBroadcast(LEAPPFunctionalTestBase):
    def _export_and_check(self, fn):
        leapp.start(self.TEST_GRAPH_NAME, max_cached_io=len(SAMPLES))
        for sample in SAMPLES:
            q = annotate.input_tensors("node", [TensorSemantics("q", sample)])
            annotate.output_tensors(
                "node",
                [TensorSemantics("y", fn(q))],
                export_with="onnx-torchscript",
                backend_params={"opset_version": 17},
            )
        leapp.stop()
        leapp.compile_graph(visualize=False, validate=True, strict=True)

        manager = InferenceManager(
            f"{self.TEST_GRAPH_NAME}/{self.TEST_GRAPH_NAME}.yaml"
        )
        for value in (*SAMPLES, UNSEEN):
            actual = torch.as_tensor(manager.run_policy({"node/q": value})["node/y"])
            torch.testing.assert_close(actual.float(), fn(value).float())

    def test_constant_assigned_to_column(self):
        self._export_and_check(_constant_axis)

    def test_traced_scalar_assigned_to_column(self):
        self._export_and_check(_traced_scalar_value)

    def test_constant_assigned_through_traced_mask(self):
        self._export_and_check(_constant_at_mask)

    def test_traced_scalar_assigned_through_traced_mask(self):
        self._export_and_check(_traced_scalar_at_mask)
