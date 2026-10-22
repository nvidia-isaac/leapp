#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Changed-input replay of mutations through independently held aliases."""

import numpy as np
import onnxruntime as ort
import pytest
import torch
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

from leapp.leapp_graph.datatypes.patching import TracingPatcher
from leapp.leapp_graph.traced_node import TracedTensorNode


def nested(x, delta):
    leaf = x[1:10][1:5:2]
    leaf += delta
    return x, leaf, x[3:8]


def base_to_view(x, delta):
    view = x.reshape(3, 4)
    x += delta
    return view, x


def siblings(x, delta):
    left, right = x[1:7], x[3:9]
    left += delta
    x *= 2
    right *= delta
    return x, left, right


def transpose(x, delta):
    view = x.reshape(3, 4).T
    view[1:3] += delta
    return x, view


def base_assignment(x, delta):
    view = x[1:7]
    x[2:3] = delta
    return view, x


def view_assignment(x, delta):
    view = x[1:7][1:4]
    view[:] = delta
    return x, view


def copy_different_layout(x, delta):
    view = x.reshape(3, 4).T
    view[:] = (x + delta).reshape(4, 3)
    return x, view, view.reshape(12)


def independent_copy(x, delta):
    copied = x.copy() if isinstance(x, np.ndarray) else x.clone()
    copied += delta
    return x, copied


def copying_reshape(x, delta):
    transposed = x.reshape(3, 4).T
    copied = transposed.reshape(12)
    copied += delta
    return x, copied


def astype_copy(x, delta):
    copied = x.astype(np.float32)
    copied += delta
    return x, copied


def returned_flat_alias(x, delta):
    alias = x.reshape(3, 4).reshape(12)
    alias += delta
    return x, alias


def select(x, delta):
    selected = x.reshape(3, 4)[1]
    selected += delta
    return x, selected


def squeeze(x, delta):
    view = x.reshape(1, 3, 4).squeeze(0)
    view += delta
    return x, view


def diagonal(x, delta):
    view = x.reshape(3, 4).diagonal()
    view.add_(delta)
    return x, view


def unbind(x, delta):
    rows = x.reshape(3, 4).unbind()
    rows[1].add_(delta)
    return x, *rows


def numpy_to_torch(x, delta):
    alias = torch.from_numpy(x)
    alias += torch.from_numpy(delta)
    return x, alias


def metadata_mutation(x, delta):
    alias = x.detach()
    alias.unsqueeze_(0)
    alias += delta
    return x, alias


CASES = [
    nested,
    base_to_view,
    siblings,
    transpose,
    base_assignment,
    view_assignment,
    copy_different_layout,
    independent_copy,
    copying_reshape,
    returned_flat_alias,
    select,
    squeeze,
]
VARIANTS = [(0, 1), (100, 1), (0, 7), (-20, -3)]


def native(value, library):
    return value.numpy().copy() if library == "numpy" else value.clone()


def capture(function, library):
    patcher = TracingPatcher()
    patcher.install()
    try:
        node = TracedTensorNode("views")
        x = node.create_input(native(torch.arange(12.0), library), "x")
        delta = node.create_input(native(torch.tensor([1.0]), library), "delta")
        outputs = function(x, delta)
        node.compile_trace({f"out{i}": value for i, value in enumerate(outputs)})
        return node.m
    finally:
        patcher.uninstall()


def check_replay(model, function, library):
    for offset, amount in VARIANTS:
        x = torch.arange(12.0) + offset
        delta = torch.tensor([float(amount)])
        expected = function(native(x, library), native(delta, library))
        given_x, given_delta = x.clone(), delta.clone()
        actual = model(given_x, given_delta)
        for got, reference in zip(actual, expected):
            if isinstance(reference, np.ndarray):
                reference = torch.from_numpy(reference.copy())
            torch.testing.assert_close(got, reference, rtol=0, atol=0)
        torch.testing.assert_close(given_x, x, rtol=0, atol=0)
        torch.testing.assert_close(given_delta, delta, rtol=0, atol=0)


@pytest.mark.parametrize("library", ["torch", "numpy"])
@pytest.mark.parametrize("function", CASES, ids=lambda fn: fn.__name__)
def test_view_mutation_replay(function, library):
    check_replay(capture(function, library), function, library)


@pytest.mark.parametrize(
    "function,library",
    [
        (astype_copy, "numpy"),
        (metadata_mutation, "torch"),
        (numpy_to_torch, "numpy"),
        (diagonal, "torch"),
        (unbind, "torch"),
    ],
)
def test_copy_and_metadata_contracts(function, library):
    check_replay(capture(function, library), function, library)


@pytest.mark.parametrize("force_native", [False, True])
def test_tensor_slice_bounds_keep_runtime_mutations(force_native, monkeypatch):
    if force_native:

        def unsupported_indices(*args, **kwargs):
            raise GuardOnDataDependentSymNode("Exercise native mutation fallback")

        monkeypatch.setattr(torch.export, "export", unsupported_indices)
    patcher = TracingPatcher()
    patcher.install()
    try:
        node = TracedTensorNode("dynamic_views")
        x = node.create_input(torch.arange(12.0).reshape(2, 6), "x")
        start = node.create_input(torch.tensor(1), "start")
        stop = node.create_input(torch.tensor(4), "stop")
        node.create_input(torch.zeros(2), "unused")
        view = x[:, start:stop]
        view += 3
        node.compile_trace({"base": x, "view": view})
    finally:
        patcher.uninstall()
    assert [description.name for description in node.inputs] == ["x", "start", "stop"]
    for model in (node.m, torch.jit.script(node.m)):
        for left, right in ((2, 6), (0, 0), (1, 2)):
            given = torch.arange(20.0, 32.0).reshape(2, 6)
            expected = given.clone()
            expected[:, left:right] += 3
            base, view = model(given, torch.tensor(left), torch.tensor(right))
            torch.testing.assert_close(base, expected)
            torch.testing.assert_close(view, expected[:, left:right])
            torch.testing.assert_close(given, torch.arange(20.0, 32.0).reshape(2, 6))


@pytest.mark.parametrize("library", ["torch", "numpy"])
@pytest.mark.parametrize("backend", ["jit", "pt2", "onnx"])
def test_returned_alias_artifact_reload(tmp_path, library, backend):
    model = capture(nested, library)
    inputs = (torch.arange(12.0), torch.tensor([1.0]))
    if backend == "jit":
        path = tmp_path / "views.pt"
        torch.jit.save(torch.jit.script(model), path)
        loaded = torch.jit.load(path)
    elif backend == "pt2":
        path = tmp_path / "views.pt2"
        torch.export.save(torch.export.export(model, inputs), path)
        loaded = torch.export.load(path).module()
    else:
        path = tmp_path / "views.onnx"
        torch.onnx.export(model, inputs, path, dynamo=True)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        names = [value.name for value in session.get_inputs()]

        def loaded(x, delta):
            values = session.run(None, dict(zip(names, [x.numpy(), delta.numpy()])))
            return tuple(torch.from_numpy(value) for value in values)

    check_replay(loaded, nested, library)


@pytest.mark.parametrize("library", ["torch", "numpy"])
def test_boolean_update_cardinality_onnx_reload(tmp_path, library):
    patcher = TracingPatcher()
    patcher.install()
    try:
        node = TracedTensorNode("masked_views")
        x = node.create_input(native(torch.arange(12.0).reshape(3, 4), library), "x")
        mask = node.create_input(
            native(torch.tensor([True, False, True, False]), library), "mask"
        )
        delta = node.create_input(native(torch.tensor([2.0]), library), "delta")
        x[:, mask] *= delta
        node.compile_trace({"base": x, "selected": x[:, mask]})
    finally:
        patcher.uninstall()
    path = tmp_path / "masked.onnx"
    torch.onnx.export(
        node.m,
        (
            torch.arange(12.0).reshape(3, 4),
            torch.tensor([True, False, True, False]),
            torch.tensor([2.0]),
        ),
        path,
        dynamo=True,
        input_names=["x", "mask", "delta"],
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for offset, amount in VARIANTS:
        for cardinality in (0, 1, 2, 4):
            given = torch.arange(12.0).reshape(3, 4) + offset
            mask = torch.arange(4) < cardinality
            expected = given.clone()
            expected[:, mask] *= amount
            actual = session.run(
                None,
                {
                    "x": given.numpy(),
                    "mask": mask.numpy(),
                    "delta": np.array([amount], dtype=np.float32),
                },
            )
            torch.testing.assert_close(torch.from_numpy(actual[0]), expected)
            torch.testing.assert_close(torch.from_numpy(actual[1]), expected[:, mask])
            torch.testing.assert_close(given, torch.arange(12.0).reshape(3, 4) + offset)


@pytest.mark.parametrize("library", ["torch", "numpy"])
@pytest.mark.parametrize("reset", [False, True])
def test_retained_alias_reattaches_after_declaration_reentry(library, reset):
    node = TracedTensorNode("reentry")
    x = node.create_input(native(torch.arange(12.0), library), "x")
    alias = x[:] if library == "torch" else x.view(type(x))
    if reset:
        node.reset_trace_state()
    x = node.create_input(x, "x" if reset else "replacement")
    alias += 3
    node.compile_trace({"alias": alias * 2, "base": x})
    fresh = torch.arange(12.0) + 40
    given = fresh.clone()
    actual = node.m(given)
    torch.testing.assert_close(actual[0], (fresh + 3) * 2)
    torch.testing.assert_close(actual[1], fresh + 3)
    torch.testing.assert_close(given, fresh)
