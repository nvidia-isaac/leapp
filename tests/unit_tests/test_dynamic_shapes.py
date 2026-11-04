#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Capture and ownership of variable input dimensions."""

import unittest

import numpy as np
import torch

from leapp import Axis, TensorSemantics, Variable
from leapp.leapp_graph.datatypes import TracedInt
from leapp.leapp_graph.traced_node import TracedTensorNode


class TestDynamicShapes(unittest.TestCase):
    def test_reshape_uses_runtime_shape(self):
        for library in (torch, np):
            with self.subTest(library=library.__name__):
                sample = library.arange(16).reshape(2, 8)
                ctx = TracedTensorNode(name="pairs", node_index=0)
                semantics = TensorSemantics(
                    "x",
                    sample,
                    axes=[
                        None,
                        Axis(variable=Variable("length", min=4, max=32, multiple_of=2)),
                    ],
                )
                x = ctx.create_input(sample, name="x", semantics=semantics)
                n = x.size(1) if library is torch else x.shape[1]
                pairs = n // 2
                self.assertIs(type(n), TracedInt)
                self.assertIs(type(pairs), TracedInt)
                self.assertEqual(int(pairs), 4)
                ctx.compile_trace({"y": x.reshape(2, pairs, 2)})
                fresh = torch.arange(24).reshape(2, 12)
                torch.testing.assert_close(ctx.m(fresh), fresh.reshape(2, 6, 2))

    def test_inactive_shape_values_are_constants(self):
        for replaced in (False, True):
            with self.subTest(replaced=replaced):
                old = TracedTensorNode(name="old", node_index=0)
                source = old.create_input(torch.tensor([1.0, -1.0, 2.0]), name="x")
                n = torch.nonzero(source > 0).shape[0]
                if replaced:
                    old.reset_trace_state()
                else:
                    old._model_captured = True
                ctx = TracedTensorNode(name="numpy", node_index=1)
                sample = np.ones(8, dtype=np.float32)
                semantics = TensorSemantics(
                    "x", sample, axes=[Axis(variable=Variable("length", min=4, max=32))]
                )
                x = ctx.create_input(sample, name="x", semantics=semantics)
                ctx.compile_trace({"y": x + n})
                torch.testing.assert_close(
                    ctx.m(torch.ones(12)), torch.full((12,), 3.0)
                )

    def test_shape_dependent_out_mutations_are_rejected(self):
        ctx = TracedTensorNode(name="mutation", node_index=0)
        sample = torch.ones(2, 8)
        x = ctx.create_input(
            sample,
            name="x",
            semantics=TensorSemantics(
                "x",
                sample,
                axes=[None, Axis(variable=Variable("length", min=4, max=32))],
            ),
        )
        with self.assertRaisesRegex(ValueError, "Shape-dependent out="):
            torch.add(x, x.size(1) - 8, out=x)
        torch.testing.assert_close(x.tensor, torch.ones(2, 8))
