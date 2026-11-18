#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Capture and ownership of variable input dimensions."""

import unittest
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch

from leapp import Axis, TensorSemantics, Variable
from leapp.backends.onnx_export_backend import ONNXDynamoExportBackend
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


@pytest.mark.parametrize(
    "shape,error",
    [
        pytest.param([2, "renamed_symbol"], None, id="renamed-symbol"),
        pytest.param([None, None], None, id="unknown-dimensions"),
        pytest.param(None, None, id="unknown-rank"),
        pytest.param([2, 8], "dimension 1 is fixed", id="frozen-variable"),
        pytest.param(
            [3, "length"], "dimension 0 is fixed", id="wrong-static-dimension"
        ),
        pytest.param([2], "rank contradicts", id="wrong-rank"),
    ],
)
def test_onnx_metadata_agrees_with_shape_contract(shape, error):
    contract = {
        "shape": [2, "length"],
        "dimension_ranges": {"length": {"min": 4, "max": 32}},
    }
    node = SimpleNamespace(
        name="metadata",
        inputs=[SimpleNamespace(name="x", shape_contract=contract)],
        outputs=[SimpleNamespace(name="y", shape_contract=contract)],
    )
    model = onnx.helper.make_model(
        onnx.helper.make_graph(
            [],
            "metadata",
            [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, shape)],
            [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, shape)],
        )
    )
    backend = ONNXDynamoExportBackend(node)
    if error is None:
        backend._validate_shape_contracts(model)
    else:
        with pytest.raises(ValueError, match=error):
            backend._validate_shape_contracts(model)


def test_generated_shapes_follow_axis_constraints():
    from leapp.utils.validation_inputs import sample_validation_shapes

    bounds = {"length": {"min": 4, "max": 12, "multiple_of": 2}}
    contracts = {
        "ids": {"shape": [2, "length"], "dimension_ranges": bounds},
        "mask": {"shape": [2, "length"], "dimension_ranges": bounds},
        "image": {
            "shape": ["height", 3],
            "dimension_ranges": {"height": {"min": 2, "max": 6}},
        },
        "bias": {"shape": [3]},
    }
    shapes, omitted = sample_validation_shapes(
        contracts,
        {"ids": (2, 8), "mask": (2, 8), "image": (4, 3), "bias": (3,)},
        {name: 4 for name in contracts},
        max_samples=20,
    )
    assert not omitted
    assert any(s["ids"][1] != 8 and s["image"] == (4, 3) for s in shapes)
    assert any(s["ids"][1] == 8 and s["image"] != (4, 3) for s in shapes)
    for sample in shapes:
        assert sample["ids"] == sample["mask"]
        assert sample["ids"][1] in range(4, 13, 2)
        assert 2 <= sample["image"][0] <= 6
        assert sample["bias"] == (3,)


def test_shape_sampling_is_reproducible_and_bounded():
    from leapp.utils.validation_inputs import sample_validation_shapes

    contracts = {
        "x": {
            "shape": ["length"],
            "dimension_ranges": {"length": {"min": 0, "max": 1000}},
        }
    }
    arguments = (contracts, {"x": (8,)}, {"x": 4})
    first = sample_validation_shapes(
        *arguments, seed=7, max_samples=2, max_input_bytes=2048
    )
    assert first == sample_validation_shapes(
        *arguments, seed=7, max_samples=2, max_input_bytes=2048
    )
    shapes, omitted = first
    assert len(shapes) == 2
    assert shapes[0] == {"x": (0,)}
    assert len({s["x"] for s in shapes}) == len(shapes)
    assert all(s["x"][0] * 4 <= 2048 for s in shapes)
    assert any("memory budget" in message for message in omitted)
    assert any("sample budget" in message for message in omitted)


@pytest.mark.parametrize("library", [torch, np], ids=["torch", "numpy"])
@pytest.mark.parametrize("dtype", ["float32", "int64", "bool", "complex64"])
def test_default_generator_preserves_input_metadata(library, dtype):
    from leapp.utils.validation_inputs import generate_validation_inputs

    samples = {
        "x": library.ones((2, 8), dtype=getattr(library, dtype)),
        "bias": library.ones((2, 1), dtype=getattr(library, dtype)),
    }
    shapes = {"x": (2, 12), "bias": (2, 1)}
    first = generate_validation_inputs(samples, shapes, seed=7)
    second = generate_validation_inputs(samples, shapes, seed=7)
    for name, value in first.items():
        assert type(value) is type(samples[name])
        assert tuple(value.shape) == shapes[name]
        assert value.dtype == samples[name].dtype
        torch.testing.assert_close(
            torch.as_tensor(value), torch.as_tensor(second[name])
        )
    torch.testing.assert_close(
        torch.as_tensor(first["bias"]), torch.as_tensor(samples["bias"])
    )
    assert first["bias"] is not samples["bias"]
    if library is torch:
        assert first["x"].device == samples["x"].device
