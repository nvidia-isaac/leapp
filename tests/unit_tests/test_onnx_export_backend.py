#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnxscript import ir
from onnxscript.rewriter import _matcher, pattern

from leapp.backends import _onnxscript_compat
from leapp.backends.onnx_export_backend import ONNXDynamoExportBackend


@pytest.mark.parametrize(
    "form,source_device,target_device,dtype",
    [
        ("method", "cpu", "cpu", torch.float32),
        (torch.Tensor.type_as, "cpu", "cpu", torch.float32),
        (torch.ops.aten.type_as, "cpu", "cpu", torch.float32),
        (torch.ops.aten.type_as.default, "cpu", "cpu", torch.float32),
        (torch.ops.aten.type_as.default, "cpu", "cpu", torch.bfloat16),
        (torch.ops.aten.type_as.default, "cuda", "cuda", torch.float32),
        (torch.ops.aten.type_as.default, "cuda", "cuda", torch.bfloat16),
        (torch.ops.aten.type_as.default, "cpu", "cuda", torch.bfloat16),
        (torch.ops.aten.type_as.default, "cuda", "cpu", torch.float32),
    ],
)
def test_type_as_canonicalization_preserves_alias_and_conversion(
    form, source_device, target_device, dtype
):
    """Cover FX spellings separately from dtype/device conversion semantics."""
    if "cuda" in (source_device, target_device) and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    root = torch.nn.Module()
    root.register_buffer(
        "constant", torch.arange(12.0, device=source_device).reshape(3, 4)
    )
    graph = torch.fx.Graph()
    other = graph.placeholder("other")
    source = graph.get_attr("constant")
    if form == "method":
        converted = graph.call_method("type_as", (source, other))
    elif form is torch.Tensor.type_as:
        converted = graph.call_function(form, (source, other))
    else:
        converted = graph.call_function(form, (), {"self": source, "other": other})
    graph.output(converted)
    module = torch.fx.GraphModule(root, graph)
    original_code = module.code
    prepared = ONNXDynamoExportBackend._canonicalize_type_as(module)
    # The reference shape must not change the converted constant's shape.
    reference = torch.zeros(2, 3, 4, dtype=dtype, device=target_device)
    expected, actual = module(reference), prepared(reference)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.device == expected.device
    assert actual.shape == (3, 4)
    assert (
        actual.device == root.constant.device
        and actual.data_ptr() == root.constant.data_ptr()
    ) == (
        expected.device == root.constant.device
        and expected.data_ptr() == root.constant.data_ptr()
    )
    assert module.code == original_code


def test_type_as_canonicalization_leaves_dynamic_sources_unchanged():
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    other = graph.placeholder("other")
    graph.output(graph.call_method("type_as", (source, other)))
    module = torch.fx.GraphModule(torch.nn.Module(), graph)
    assert ONNXDynamoExportBackend._canonicalize_type_as(module) is module


@pytest.fixture
def stock_matcher(monkeypatch):
    method = _matcher.SimplePatternMatcher.match
    if getattr(method, "_leapp_root_guard", False):
        method = method.__wrapped__
    monkeypatch.setattr(_matcher.SimplePatternMatcher, "match", method)
    monkeypatch.setattr(_onnxscript_compat, "_patch_disabled", False)
    return method


@pytest.fixture
def multi_output_match():
    def two_outputs(op, x):
        return op.Cast(x, to=1), op.Neg(x)

    model = ir.from_onnx_text(
        """<ir_version: 9, opset_import: [ "" : 18]>
        graph (float[2] x) => (float[2] cast, float[2] neg) {
            neg = Neg(x)
            cast = Cast<to=1>(x)
        }"""
    )
    return pattern.Pattern(two_outputs), model


@pytest.mark.parametrize(
    "domain,op_type,to", [("", "Relu", 1), ("custom", "Cast", 1), ("", "Cast", 10)]
)
def test_matcher_rejects_impossible_roots_without_scanning(
    stock_matcher, multi_output_match, domain, op_type, to
):
    matcher, model = multi_output_match
    _onnxscript_compat.patch_onnxscript_matcher()
    installed = _matcher.SimplePatternMatcher.match
    _onnxscript_compat.patch_onnxscript_matcher()
    assert _matcher.SimplePatternMatcher.match is installed
    x = ir.Value(name="x")
    root = ir.Node(domain, op_type, [x], {"to": ir.AttrInt64("to", to)})
    other = ir.Node("", "Neg", [x])
    graph = ir.Graph([x], [root.outputs[0], other.outputs[0]], nodes=[root, other])
    with mock.patch.object(ir.Graph, "__iter__", side_effect=AssertionError("scan")):
        assert not matcher.match(ir.Model(graph, ir_version=9), graph, root)
    # A rejection must leave the same matcher usable, with output order intact.
    result = matcher.match(model, model.graph, model.graph.node(1))
    assert result
    assert [value.name for value in result.outputs] == ["cast", "neg"]
    assert not _onnxscript_compat._patch_disabled


@pytest.mark.parametrize("failure", ["partial_match", "new_keyword"])
def test_matcher_guard_failure_restores_stock_and_retries(
    stock_matcher, multi_output_match, monkeypatch, failure
):
    matcher, model = multi_output_match
    target = matcher._matcher.pattern.output_nodes[0]
    original_matches = target.matches
    failed = False

    def fail_once(*args):
        nonlocal failed
        result = original_matches(*args)
        if not failed:
            failed = True
            raise AttributeError("guard API changed after partial matching")
        return result

    def stock(self, *args, **kwargs):
        # Simulate an upstream matcher accepting a newly added keyword.
        kwargs.pop("future_keyword", None)
        return stock_matcher(self, *args, **kwargs)

    monkeypatch.setattr(_matcher.SimplePatternMatcher, "match", stock)
    if failure == "partial_match":
        monkeypatch.setattr(target, "matches", fail_once)
    kwargs = {"future_keyword": True} if failure == "new_keyword" else {}
    with mock.patch.object(_onnxscript_compat, "_get_logger") as logger:
        _onnxscript_compat.patch_onnxscript_matcher()
        cached = matcher._matcher.match
        result = cached(model, model.graph, model.graph.node(1), **kwargs)
        assert result
        assert [value.name for value in result.outputs] == ["cast", "neg"]
        assert _matcher.SimplePatternMatcher.match is stock
        assert _onnxscript_compat._patch_disabled
        # Later exports and cached bound methods must stay on the stock path.
        _onnxscript_compat.patch_onnxscript_matcher()
        assert _matcher.SimplePatternMatcher.match is stock
        assert cached(model, model.graph, model.graph.node(1), **kwargs)
        logger.return_value.info.assert_called_once()


@pytest.mark.parametrize("guard_fails", [False, True])
def test_matcher_does_not_retry_original_errors(
    stock_matcher, multi_output_match, monkeypatch, guard_fails
):
    matcher, model = multi_output_match
    error = ValueError("native matcher failed")
    original = mock.Mock(side_effect=error)

    def stock(self, *args, **kwargs):
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_matcher.SimplePatternMatcher, "match", stock)
    if guard_fails:
        monkeypatch.setattr(
            matcher._matcher.pattern.output_nodes[0],
            "matches",
            mock.Mock(side_effect=AttributeError("guard API changed")),
        )
    _onnxscript_compat.patch_onnxscript_matcher()
    with pytest.raises(ValueError) as caught:
        matcher.match(model, model.graph, model.graph.node(1))
    assert caught.value is error
    original.assert_called_once()
    assert _onnxscript_compat._patch_disabled is guard_fails


def test_compile_materializes_once_and_saves_repaired_model():
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
    with (
        mock.patch("torch.onnx.export", return_value=Program()),
        mock.patch("leapp.backends.onnx_export_backend.patch_onnxscript_matcher"),
    ):
        backend.compile(torch.nn.Identity())

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
