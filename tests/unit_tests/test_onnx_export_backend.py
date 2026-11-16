#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort
import onnxscript
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxscript import ir
from onnxscript.rewriter import _matcher, pattern

from leapp.backends._onnxscript_compat import patch_onnxscript_matcher
from leapp.backends.onnx_export_backend import ONNXDynamoExportBackend


@pytest.fixture
def stock_matcher(monkeypatch):
    """Keep each test's process-wide matcher change isolated."""
    method = _matcher.SimplePatternMatcher.match
    if getattr(method, "_leapp_root_guard", False):
        method = method.__wrapped__
    monkeypatch.setattr(_matcher.SimplePatternMatcher, "match", method)
    return method


def test_compile_materializes_once_and_saves_repaired_model(stock_matcher):
    """Avoid repeated weight serialization while preserving pruning and repairs."""
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Slice", ["x", "zero", "one"], ["sliced"]),
                helper.make_node("Gather", ["x", "zero"], ["picked"], axis=0),
            ],
            "shared_scalar",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])],
            [
                helper.make_tensor_value_info("sliced", TensorProto.FLOAT, [1]),
                helper.make_tensor_value_info("picked", TensorProto.FLOAT, []),
            ],
            [
                numpy_helper.from_array(np.array(0, dtype=np.int64), name="zero"),
                numpy_helper.from_array(np.array(1, dtype=np.int64), name="one"),
            ],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
    )
    accesses = []

    class Program:
        @property
        def model_proto(self):
            # Real ONNXProgram rebuilds a proto on every access.
            accesses.append(None)
            return onnx.ModelProto.FromString(model.SerializeToString())

    context = SimpleNamespace(
        name="shared_scalar",
        inputs=[
            SimpleNamespace(name=name, value=torch.ones(3)) for name in ("x", "unused")
        ],
        outputs=[SimpleNamespace(name=name) for name in ("sliced", "picked")],
        graph=torch.fx.Graph(),
        trimmed_inputs=set(),
    )
    backend = ONNXDynamoExportBackend(context)

    def export(*args, **kwargs):
        # The normal backend must activate the guard before PyTorch optimizes.
        assert _matcher.SimplePatternMatcher.match._leapp_root_guard
        assert kwargs["optimize"] is True
        return Program()

    with (
        mock.patch("torch.onnx.export", side_effect=ValueError("export failed")),
        pytest.raises(ValueError, match="export failed"),
    ):
        backend.compile(torch.nn.Identity())
    installed = _matcher.SimplePatternMatcher.match
    with mock.patch("torch.onnx.export", side_effect=export):
        backend.compile(torch.nn.Identity())
    assert _matcher.SimplePatternMatcher.match is installed

    # Count is the performance regression contract: each read copies all weights.
    assert len(accesses) == 1
    assert [item.name for item in context.inputs] == ["x"]
    assert context.trimmed_inputs == {"unused"}
    path = Path(backend.compiled_model._source_dir) / "shared_scalar.onnx"
    saved = onnx.load(path)
    onnx.checker.check_model(saved, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), options, providers=["CPUExecutionProvider"]
    )
    for values in ([2.0, 3.0, 4.0], [-5.0, 0.0, 7.0]):
        x = np.array(values, dtype=np.float32)
        sliced, picked = session.run(None, {"x": x})
        np.testing.assert_array_equal(sliced, x[:1])
        np.testing.assert_array_equal(picked, x[0])


@pytest.mark.parametrize(
    "domain,op_type,to", [("", "Relu", 1), ("custom", "Cast", 1), ("", "Cast", 10)]
)
def test_matcher_rejects_incompatible_roots_without_graph_scan(
    stock_matcher, domain, op_type, to
):
    def two_outputs(op, x):
        return op.Cast(x, to=1), op.Neg(x)

    # Existing rule instances must pick up the patch too.
    matcher = pattern.Pattern(two_outputs)
    patch_onnxscript_matcher()
    patched = _matcher.SimplePatternMatcher.match
    patch_onnxscript_matcher()
    assert _matcher.SimplePatternMatcher.match is patched
    x = ir.Value(name="x")
    root = ir.Node(domain, op_type, [x], {"to": ir.AttrInt64("to", to)})
    other = ir.Node("", "Neg", [x])
    graph = ir.Graph([x], [root.outputs[0], other.outputs[0]], nodes=[root, other])
    model = ir.Model(graph, ir_version=10)
    # A scan per incompatible root is the quadratic-work regression.
    with mock.patch.object(
        ir.Graph, "__iter__", side_effect=AssertionError("scanned graph")
    ):
        assert not matcher.match(model, graph, root)

    # Reuse the matcher after rejection: no stale bindings or reversed outputs.
    valid = ir.from_onnx_text(
        """
        <ir_version: 7, opset_import: [ "" : 17]>
        agraph (float[2] x) => (float[2] cast, float[2] neg)
        {
            neg = Neg(x)
            cast = Cast<to=1>(x)
        }
        """
    )
    result = matcher.match(valid, valid.graph, valid.graph.node(1))
    assert result
    assert [value.name for value in result.outputs] == ["cast", "neg"]
    assert result.bindings["x"] is valid.graph.inputs[0]


def test_matcher_preserves_shared_node_removal_policy(stock_matcher):
    def two_outputs(op, x):
        shared = op.Abs(x)
        return op.Neg(shared), op.Relu(shared)

    model = ir.from_onnx_text(
        """
        <ir_version: 7, opset_import: [ "" : 17]>
        agraph (float[2] x) => (float[2] neg, float[2] relu, float[2] extra)
        {
            shared = Abs(x)
            extra = Exp(shared)
            relu = Relu(shared)
            neg = Neg(shared)
        }
        """
    )
    matcher = pattern.Pattern(two_outputs)
    patch_onnxscript_matcher()
    assert not matcher.match(model, model.graph, model.graph.node(3))
    result = matcher.match(
        model, model.graph, model.graph.node(3), check_nodes_are_removable=False
    )
    assert result
    assert [value.name for value in result.outputs] == ["neg", "relu"]


def test_matcher_rejects_unqualified_version(stock_matcher, monkeypatch):
    monkeypatch.setattr(onnxscript, "__version__", "0.7.2")
    with pytest.raises(RuntimeError, match="requires onnxscript==0.7.1"):
        patch_onnxscript_matcher()
    assert _matcher.SimplePatternMatcher.match is stock_matcher


def test_matcher_concurrent_activation_wraps_once(stock_matcher):
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: patch_onnxscript_matcher(), range(16)))
    assert _matcher.SimplePatternMatcher.match.__wrapped__ is stock_matcher
