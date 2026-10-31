# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operation-level port preservation and conversion identity contracts."""

from unittest import mock

import numpy as np
import pytest
import torch

from leapp.export_manager import ExportManager
from leapp.leapp_graph.datatypes.patching import TracingPatcher
from leapp.leapp_graph.traced_node import TracedTensorNode


@pytest.fixture(autouse=True)
def tracing_patches():
    patcher = TracingPatcher()
    patcher.install()
    try:
        yield
    finally:
        patcher.uninstall()


def finished_output(library="torch", name="producer", port="y", offset=0):
    node = TracedTensorNode(name)
    value = torch.arange(1.0, 4.0) + offset
    if library == "numpy":
        value = value.numpy().copy()
    output = node.create_input(value, "x") * 2
    node.compile_trace({port: output})
    node.publish_output_port(output, port)
    return output


@pytest.mark.parametrize(
    "library,operation",
    [
        pytest.param("torch", lambda x: x.clone(), id="torch_clone"),
        pytest.param("torch", lambda x: x.detach(), id="torch_detach"),
        pytest.param("torch", lambda x: x.contiguous(), id="torch_contiguous"),
        pytest.param("torch", lambda x: x.cpu(), id="torch_cpu"),
        pytest.param("torch", lambda x: x.to(torch.float32), id="torch_noop_to"),
        pytest.param("numpy", np.copy, id="numpy_copy_function"),
        pytest.param("numpy", lambda x: x.copy(), id="numpy_copy_method"),
        pytest.param("numpy", np.asanyarray, id="numpy_asanyarray"),
        pytest.param(
            "numpy", lambda x: torch.as_tensor(np.asarray(x)), id="numpy_to_torch"
        ),
        pytest.param("torch", lambda x: x.numpy(), id="torch_to_numpy"),
    ],
)
def test_equivalent_copy_preserves_output_port(library, operation):
    output = finished_output(library)
    copied = operation(output)
    assert copied.output_port == "y"
    assert copied.context_obj is output.context_obj


@pytest.mark.parametrize("operation", ["slice", "copy"])
def test_plain_torch_full_copy_preserves_output_port(operation):
    output = finished_output()
    buffer = torch.zeros(3)
    if operation == "slice":
        buffer[:] = output
    else:
        buffer.copy_(output)
    assert buffer.output_port == "y"
    assert buffer.context_obj is output.context_obj


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(lambda x: x + 1, id="arithmetic"),
        pytest.param(lambda x: x[:2], id="slice"),
        pytest.param(lambda x: x.reshape(3, 1), id="reshape"),
        pytest.param(lambda x: x.to(torch.float64), id="dtype_cast"),
    ],
)
def test_transform_drops_output_port(operation):
    assert getattr(operation(finished_output()), "output_port", None) is None


def test_partial_mutation_drops_output_port():
    output = finished_output()
    output[0] = 9
    assert output.output_port is None


def test_numpy_full_copy_adopts_new_source_port():
    first = finished_output("numpy", "first", "first_output")
    second = finished_output("numpy", "second", "second_output", offset=3)
    buffer = np.copy(first)
    assert buffer.output_port == "first_output"
    buffer[:] = second
    assert buffer.output_port == "second_output"
    assert buffer.context_obj is second.context_obj


def test_plain_numpy_assignment_requires_explicit_mirroring():
    output = finished_output("numpy")
    buffer = np.zeros_like(np.asarray(output))
    buffer[:] = output
    assert type(buffer) is np.ndarray
    with mock.patch.object(ExportManager, "_interpret_graph", True):
        mirrored = ExportManager().mirror_leapp_tags(output, buffer)
    assert mirrored.output_port == "y"
    assert mirrored.context_obj is output.context_obj


@pytest.mark.parametrize(
    "library,operation,same_proxy",
    [
        pytest.param("torch", lambda x: x.numpy(), True, id="torch_to_numpy"),
        pytest.param("numpy", lambda x: x.view(type(x)), True, id="same_layout_view"),
        pytest.param("numpy", lambda x: x[:2], False, id="narrowing_slice"),
        pytest.param("numpy", lambda x: np.add(x, 1), False, id="allocating_ufunc"),
    ],
)
def test_conversion_proxy_identity(library, operation, same_proxy):
    node = TracedTensorNode("conversion")
    value = torch.arange(3.0)
    if library == "numpy":
        value = value.numpy().copy()
    source = node.create_input(value, "x")
    assert (operation(source).proxy is source.proxy) == same_proxy
