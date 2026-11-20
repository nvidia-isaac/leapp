#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Variable dimension contracts, declarations, loading, and capture."""

import dataclasses
import unittest
from types import SimpleNamespace

import onnx
import pytest
import torch
import yaml

from leapp import Axis, InferenceManager, TensorSemantics, Variable
from leapp.backends.onnx_export_backend import ONNXDynamoExportBackend
from leapp.leapp_graph.datatypes import TracedInt
from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.utils.shape_contracts import (
    build_declared_shape_contract,
    declares_variable_axis,
    evaluate_dimension_expression,
    get_shape_contract_version,
    parse_shape_contract,
    serialize_shape_axes,
    validate_shape_contract,
)
from leapp.utils.validation_inputs import (
    generate_validation_inputs,
    sample_validation_shapes,
)


class TestDynamicShapes(unittest.TestCase):
    def test_inactive_shape_values_are_constants(self):
        """A traced size from a finished or replaced node is recorded as a constant."""
        for replaced in (False, True):
            with self.subTest(replaced=replaced):
                old = TracedTensorNode(name="old", node_index=0)
                source = old.create_input(torch.tensor([1.0, -1.0, 2.0]), name="x")
                n = torch.nonzero(source > 0).shape[0]
                if replaced:
                    old.reset_trace_state()
                else:
                    old._model_captured = True
                ctx = TracedTensorNode(name="consumer", node_index=1)
                sample = torch.ones(8)
                semantics = TensorSemantics(
                    "x", sample, axes=[Axis(variable=Variable("length", min=4, max=32))]
                )
                x = ctx.create_input(sample, name="x", semantics=semantics)
                ctx.compile_trace({"y": x + n})
                torch.testing.assert_close(
                    ctx.m(torch.ones(12)), torch.full((12,), 3.0)
                )

    def test_shape_dependent_out_mutations_are_rejected(self):
        """An ``out=`` operation with a traced size raises and changes nothing."""
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
    """A saved ONNX model may rename or omit a symbol but not fix or contradict it."""
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
    """Generated shapes keep shared symbols equal and stay within each domain."""
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
    """Sampling is seeded and reports sizes dropped by the sample and memory budgets."""
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


@pytest.mark.parametrize("dtype", ["float32", "int64", "bool", "complex64"])
def test_default_generator_preserves_input_metadata(dtype):
    """Generated inputs keep type, dtype and device; an unchanged shape keeps a copy."""
    samples = {
        "x": torch.ones((2, 8), dtype=getattr(torch, dtype)),
        "bias": torch.ones((2, 1), dtype=getattr(torch, dtype)),
    }
    shapes = {"x": (2, 12), "bias": (2, 1)}
    first = generate_validation_inputs(samples, shapes, seed=7)
    second = generate_validation_inputs(samples, shapes, seed=7)
    for name, value in first.items():
        assert type(value) is type(samples[name])
        assert tuple(value.shape) == shapes[name]
        assert value.dtype == samples[name].dtype
        torch.testing.assert_close(value, second[name])
    torch.testing.assert_close(first["bias"], samples["bias"])
    assert first["bias"] is not samples["bias"]
    assert first["x"].device == samples["x"].device


# =============================================================================
# Shape contracts
# =============================================================================

_LENGTH = {"min": 4, "max": 32, "multiple_of": 2}
_PLAIN = {"shape": [2, "length"], "dimension_ranges": {"length": _LENGTH}}
# [2, length // 2]
_HALVED = {
    "shape": [2, "pairs"],
    "dimension_ranges": {"pairs": {"min": 2, "max": 16}, "length": _LENGTH},
    "dimension_expressions": {
        "pairs": {"formula": {"op": "floordiv", "args": ["length", 2]}}
    },
}
# [2, 2 * n + 1]
_GROWN = {
    "shape": [2, "grown"],
    "dimension_ranges": {"grown": {"min": 9, "max": 65}, "n": {"min": 4, "max": 32}},
    "dimension_expressions": {"grown": {"terms": {"n": 2}, "offset": 1}},
}


@pytest.mark.parametrize(
    "contract,shape,bindings,expected",
    [
        pytest.param(_PLAIN, (2, 8), {}, {"length": 8}, id="variable"),
        pytest.param({"shape": [None, 3]}, (7, 3), {}, {}, id="unknown-dimension"),
        pytest.param(
            _HALVED, (2, 4), {"length": 8}, {"length": 8, "pairs": 4}, id="formula"
        ),
        pytest.param(_GROWN, (2, 13), {}, {"n": 6, "grown": 13}, id="affine-root"),
    ],
)
def test_shape_contract_accepts_admitted_shapes(contract, shape, bindings, expected):
    """An admitted shape returns the size bound to each symbol, including roots."""
    assert validate_shape_contract(contract, shape, dict(bindings)) == expected


@pytest.mark.parametrize(
    "contract,shape,bindings,error",
    [
        pytest.param(_PLAIN, (2, 8, 1), {}, "expected rank 2, got 3", id="rank"),
        pytest.param(
            _PLAIN, (3, 8), {}, "expected fixed dimension 2, got 3", id="fixed"
        ),
        pytest.param(
            _PLAIN, (2, 2), {}, r"length must be in \[4, 32\], got 2", id="min"
        ),
        pytest.param(
            _PLAIN, (2, 34), {}, r"length must be in \[4, 32\], got 34", id="max"
        ),
        pytest.param(_PLAIN, (2, 7), {}, "length must be a multiple of 2", id="step"),
        pytest.param(
            _PLAIN,
            (2, 8),
            {"length": 12},
            "shared dimension length is 12, got 8",
            id="shared-symbol",
        ),
        pytest.param(
            _HALVED,
            (2, 5),
            {"length": 8},
            "derived dimension pairs must be 4, got 5",
            id="formula-mismatch",
        ),
        pytest.param(
            _HALVED, (2, 4), {}, "cannot resolve shape formula roots", id="formula-root"
        ),
        pytest.param(
            _GROWN, (2, 12), {}, "size 12 violates derived dimension", id="affine-size"
        ),
        pytest.param(
            _GROWN,
            (2, 13),
            {"n": 5},
            "derived dimension grown must be 11, got 13",
            id="affine-mismatch",
        ),
    ],
)
def test_shape_contract_rejects_other_shapes(contract, shape, bindings, error):
    """Each violation of rank, fixed size, bounds, divisor or derived size is named."""
    with pytest.raises(ValueError, match=error):
        validate_shape_contract(contract, shape, dict(bindings), name="x")


@pytest.mark.parametrize(
    "contract,capture_shape",
    [
        pytest.param(_PLAIN, [2, 8], id="last-axis"),
        pytest.param(
            {"shape": ["length", 3], "dimension_ranges": {"length": _LENGTH}},
            [8, 3],
            id="first-axis",
        ),
        pytest.param(
            {"shape": [2, "length", 3], "dimension_ranges": {"length": _LENGTH}},
            [2, 8, 3],
            id="middle-axis",
        ),
        pytest.param(_HALVED, [2, 4], id="formula"),
        pytest.param(_GROWN, [2, 17], id="affine"),
    ],
)
def test_shape_contract_survives_yaml_axes(contract, capture_shape):
    """A contract written to YAML ``axes`` and parsed back is unchanged."""
    axes = serialize_shape_axes(SimpleNamespace(shape_contract=contract), [])
    assert len(axes) == len(capture_shape)
    assert parse_shape_contract({"shape": capture_shape, "axes": axes}) == contract


def test_serialized_axes_keep_declared_metadata():
    """Serialization adds ``variable`` beside declared metadata and copies the list."""
    declared = [{"kind": "system"}, {"kind": "custom"}]
    axes = serialize_shape_axes(SimpleNamespace(shape_contract=_PLAIN), declared)
    assert axes[0] == {"kind": "system"}
    assert axes[1] == {"kind": "custom", "variable": {"symbol": "length", **_LENGTH}}
    assert declared == [{"kind": "system"}, {"kind": "custom"}]
    static = SimpleNamespace(shape_contract={"shape": [2, 8]})
    assert serialize_shape_axes(static, []) == []


@pytest.mark.parametrize(
    "expression,expected",
    [
        pytest.param({"terms": {"n": 2, "m": 1}, "offset": 1}, 14, id="affine"),
        pytest.param({"formula": {"op": "add", "args": ["n", 3]}}, 8, id="add"),
        pytest.param({"formula": {"op": "mul", "args": ["n", 2]}}, 10, id="mul"),
        pytest.param({"formula": {"op": "floordiv", "args": ["n", 2]}}, 2, id="div"),
        pytest.param({"formula": {"op": "mod", "args": ["n", 2]}}, 1, id="mod"),
        pytest.param(
            {
                "formula": {
                    "op": "add",
                    "args": [
                        {"op": "floordiv", "args": ["n", 2]},
                        {"op": "mod", "args": ["n", 2]},
                    ],
                }
            },
            3,
            id="nested",
        ),
    ],
)
def test_dimension_expressions_evaluate(expression, expected):
    """Affine expressions and each formula operation use integer arithmetic."""
    assert evaluate_dimension_expression(expression, {"n": 5, "m": 3}) == expected


@pytest.mark.parametrize(
    "contract,version",
    [
        pytest.param({"shape": [2, 8]}, 1, id="static"),
        pytest.param(
            {"shape": [2, "n"], "dimension_ranges": {"n": {"min": 4, "max": 32}}},
            1,
            id="bounds-only",
        ),
        pytest.param(_GROWN, 1, id="affine"),
        pytest.param(_PLAIN, 2, id="multiple-of"),
        pytest.param(
            {
                "shape": [2, "pairs"],
                "dimension_ranges": {
                    "pairs": {"min": 2, "max": 16},
                    "n": {"min": 4, "max": 32},
                },
                "dimension_expressions": {
                    "pairs": {"formula": {"op": "floordiv", "args": ["n", 2]}}
                },
            },
            2,
            id="formula",
        ),
    ],
)
def test_shape_contract_version_follows_used_features(contract, version):
    """Only ``multiple_of`` and formulas need contract version 2."""
    assert get_shape_contract_version([{"shape": [3]}, contract]) == version


# =============================================================================
# Declarations
# =============================================================================


def test_variable_defaults_and_identity():
    """``Variable`` has the documented defaults, compares by value and is immutable."""
    assert Variable("n").to_dict() == {"symbol": "n", "min": 1, "max": 2**31 - 1}
    assert Variable("n", min=4, max=32, multiple_of=2).to_dict() == {
        "symbol": "n",
        **_LENGTH,
    }
    assert Variable("n", min=4, max=32) == Variable("n", min=4, max=32)
    assert Variable("n", min=4, max=32) != Variable("n", min=4, max=64)
    with pytest.raises(dataclasses.FrozenInstanceError):
        Variable("n").max = 8


@pytest.mark.parametrize(
    "arguments,error",
    [
        pytest.param({"name": "9lives"}, "must be an identifier", id="identifier"),
        pytest.param({"name": 3}, "must be an identifier", id="non-string-name"),
        pytest.param({"name": "__leapp_x"}, "are reserved", id="reserved-prefix"),
        pytest.param({"name": "n", "min": -1}, "bounds must satisfy", id="negative"),
        pytest.param({"name": "n", "min": True}, "bounds must satisfy", id="bool-min"),
        pytest.param(
            {"name": "n", "min": 8, "max": 8}, "bounds must satisfy", id="one-size"
        ),
        pytest.param(
            {"name": "n", "max": 2**31}, "bounds must satisfy", id="too-large"
        ),
        pytest.param(
            {"name": "n", "multiple_of": 0}, "positive integer", id="zero-multiple"
        ),
        pytest.param(
            {"name": "n", "multiple_of": 1.5}, "positive integer", id="float-multiple"
        ),
        pytest.param(
            {"name": "n", "min": 4, "max": 7, "multiple_of": 4},
            "at least two admitted sizes",
            id="one-admitted-multiple",
        ),
    ],
)
def test_variable_rejects_invalid_domains(arguments, error):
    """Invalid names, bounds and divisors are rejected at construction."""
    with pytest.raises(ValueError, match=error):
        Variable(**arguments)


def test_axis_variable_values():
    """``Axis.variable`` takes a ``Variable`` or a bool and excludes ``names``."""
    length = Variable("length", min=4, max=32)
    assert Axis(variable=length).to_dict() == {"variable": length.to_dict()}
    assert Axis(variable=True).to_dict() == {"variable": True}
    assert Axis("element", names=["a", "b"], variable=False).to_dict() == {
        "kind": "element",
        "names": ["a", "b"],
        "variable": False,
    }
    assert "variable" not in Axis("element").to_dict()
    with pytest.raises(TypeError, match="Axis.variable must be"):
        Axis(variable="length")
    for variable in (length, True):
        with pytest.raises(ValueError, match="fixed exhaustive names"):
            Axis("element", names=["a", "b"], variable=variable)


def test_axis_shorthand_declares_independent_dimensions():
    """``Axis(variable=True)`` gets its own generated symbol with the default bounds."""

    def declared(name, variable):
        return SimpleNamespace(
            name=name,
            shape=(2, 8),
            semantics=SimpleNamespace(axes=[None, Axis(variable=variable)]),
        )

    left = build_declared_shape_contract(declared("left", True), "node")
    right = build_declared_shape_contract(declared("right", True), "node")
    (symbol,) = left["dimension_ranges"]
    assert left["shape"] == [2, symbol]
    assert symbol.startswith("__leapp_axis_")
    assert left["dimension_ranges"][symbol] == {"min": 1, "max": 2**31 - 1}
    assert right["shape"][1] != symbol
    assert build_declared_shape_contract(declared("x", False), "node") == {
        "shape": [2, 8]
    }


# =============================================================================
# Loading
# =============================================================================


@pytest.mark.parametrize(
    "contract_version,variable,error",
    [
        pytest.param(
            3,
            {"symbol": "n", "min": 4, "max": 32},
            "Unsupported shape contract version 3",
            id="unknown-contract-version",
        ),
        pytest.param(
            1,
            {"symbol": "n", "min": 4, "max": 32, "multiple_of": 2},
            "require shape contract version 2",
            id="multiple-of-under-version-1",
        ),
        pytest.param(
            1,
            {
                "symbol": "pairs",
                "min": 2,
                "max": 16,
                "expression": {"formula": {"op": "floordiv", "args": ["n", 2]}},
                "roots": {"n": {"min": 4, "max": 32}},
            },
            "require shape contract version 2",
            id="formula-under-version-1",
        ),
        pytest.param(
            None,
            {"symbol": "n", "min": 4, "max": 32},
            "require shape contract version 1 or 2",
            id="missing-contract-version",
        ),
    ],
)
def test_loader_rejects_inconsistent_contract_versions(
    tmp_path, contract_version, variable, error
):
    """A bundle fails to load when its contract version does not cover its axes."""
    system = {}
    if contract_version is not None:
        system["shape contract version"] = contract_version
    port = {
        "name": "x",
        "dtype": "float32",
        "shape": [2, 8],
        "type": "tensor",
        "axes": [None, {"variable": variable}],
    }
    bundle = {
        "models": {"node": {"inputs": [port], "outputs": [], "parameters": {}}},
        "pipeline": {},
        "system information": system,
    }
    path = tmp_path / "bundle.yaml"
    path.write_text(yaml.safe_dump(bundle))
    with pytest.raises(ValueError, match=error):
        InferenceManager(str(path))


@pytest.mark.parametrize(
    "axes,expected",
    [
        pytest.param(None, False, id="no-axes"),
        pytest.param([None, {"kind": "element"}], False, id="metadata-only"),
        pytest.param([{"variable": False}], False, id="declared-fixed"),
        pytest.param([None, {"variable": {"symbol": "n"}}], True, id="variable"),
    ],
)
def test_only_variable_mappings_require_a_shape_contract(axes, expected):
    """Only an axis with a ``variable`` mapping marks a serialized port as variable."""
    description = {"name": "x", "shape": [2, 8]}
    if axes is not None:
        description["axes"] = axes
    assert declares_variable_axis(description) is expected


# =============================================================================
# Capture
# =============================================================================


def _split(x):
    return torch.split(x, x.shape[1] // 2, dim=1)


def _topk(x):
    return torch.topk(x, x.shape[1] // 2, dim=1)


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda x: x.reshape(2, x.size(1) // 2, 2), id="reshape"),
        pytest.param(lambda x: (x + 1).reshape(-1).reshape(x.size()), id="full-size"),
        pytest.param(lambda x: x.view(2, x.shape[1] // 2, 2), id="view"),
        pytest.param(
            lambda x: x.sum(dim=1, keepdim=True).expand(2, x.shape[1]), id="expand"
        ),
        pytest.param(_split, id="split"),
        pytest.param(lambda x: x.split(x.shape[1] // 2, dim=1), id="method-split"),
        pytest.param(_topk, id="topk"),
    ],
)
def test_shape_consumers_use_runtime_shape(operation):
    """An operation given a traced size follows the input length in FX replay."""
    sample = torch.arange(16.0).reshape(2, 8)
    ctx = TracedTensorNode(name="consumer", node_index=0)
    semantics = TensorSemantics(
        "x",
        sample,
        axes=[None, Axis(variable=Variable("length", min=4, max=32, multiple_of=2))],
    )
    outputs = operation(ctx.create_input(sample, name="x", semantics=semantics))
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    ctx.compile_trace({f"y{index}": output for index, output in enumerate(outputs)})

    fresh = torch.arange(24.0).flip(0).reshape(2, 12)
    actual = ctx.m(fresh)
    expected = operation(fresh)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    assert len(actual) == len(expected)
    for exported, reference in zip(actual, expected):
        torch.testing.assert_close(exported, reference)


def test_shape_consumer_accepts_parameter_operands():
    """A shape-dependent operation records ``nn.Parameter`` operands as constants."""
    weight = torch.nn.Parameter(torch.full((8,), 2.0))
    bias = torch.nn.Parameter(torch.ones(8))

    def normalize(x):
        return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias)

    sample = torch.arange(16.0).reshape(2, 8)
    ctx = TracedTensorNode(name="normalize", node_index=0)
    semantics = TensorSemantics(
        "x", sample, axes=[Axis(variable=Variable("batch", min=1, max=8)), None]
    )
    ctx.compile_trace(
        {"y": normalize(ctx.create_input(sample, name="x", semantics=semantics))}
    )
    fresh = torch.arange(40.0).flip(0).reshape(5, 8)
    torch.testing.assert_close(ctx.m(fresh), normalize(fresh))


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda picked, count: picked.reshape(count, 1), id="reshape"),
        pytest.param(lambda picked, count: picked.view(count, 1), id="view"),
        pytest.param(
            lambda picked, count: torch.reshape(picked, (count, 1)), id="torch-reshape"
        ),
        pytest.param(lambda picked, count: picked[:count], id="slice"),
    ],
)
def test_static_node_keeps_data_dependent_size_after_shape_consumer(operation):
    """A node without variable axes still traces sizes that depend on values."""
    sample = torch.tensor([[1.0, -1.0, 2.0], [3.0, 0.0, -4.0]])  # three positives
    ctx = TracedTensorNode(name="static", node_index=0)
    x = ctx.create_input(sample, name="x")
    index = torch.nonzero(x > 0)
    count = index.shape[0]
    result = operation(x[index[:, 0], index[:, 1]], count)
    assert type(count) is TracedInt
    assert type(result.shape[0]) is TracedInt
    ctx.compile_trace({"y": x.sum().reshape(1) * 0 + result.shape[0]})

    fresh = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, -6.0]])  # five positives
    torch.testing.assert_close(ctx.m(fresh), torch.tensor([5.0]))
