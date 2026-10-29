# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU discovery regressions for Torch consumers of pending Warp outputs."""

from unittest import mock

import pytest
import torch
import warp as wp

from leapp.export_manager import ExportManager
from leapp.leapp_graph.datatypes import as_traced
from leapp.leapp_graph.datatypes.warp.patching import WarpPatchBackend
from leapp.leapp_graph.datatypes.warp.session import WarpTraceSession
from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.warp_op import WarpOp


def test_no_active_segment_skips_call_stack_collection():
    backend = WarpPatchBackend()
    backend._session = WarpTraceSession()
    with mock.patch(
        "leapp.leapp_graph.datatypes.warp.patching.get_caller_stack_identity"
    ) as collect_stack:
        backend.close_warp_segment()
    collect_stack.assert_not_called()


@pytest.fixture
def pending_output():
    wp.init()
    node = TracedTensorNode("boundary")
    tensor = node.create_input(torch.arange(6.0).reshape(2, 3), "values")
    array = as_traced(
        wp.from_torch(tensor.tensor), "values", node, view=tensor.proxy_view
    )
    backend = WarpPatchBackend()
    session = backend._session = WarpTraceSession()
    with mock.patch.object(ExportManager().patcher, "warp", backend):
        operation = WarpOp(node, session=session, capture=False)
        operation.begin(None)
        segment = operation.segment
        segment.add_input_ref(array)
        segment.add_output_ref(array)
        segment.add_event({"kind": "warp_call", "qualname": "warp.launch"})
        # Discovery only needs the boundary references. Simulate a kernel's
        # eager write without requiring CUDA/APIC on the unit-test machine.
        tensor.tensor.add_(7)
        yield node, tensor, segment, session
        session.close_warp_segment(requester=session.owner_token)


@pytest.mark.parametrize(
    "consume",
    [
        lambda x: x.reshape(3, 2),
        lambda x: x.view(3, 2),
        lambda x: x.transpose(0, 1),
        lambda x: x[:, 1:],
        lambda x: x + 2,
        lambda x: torch.cat([x, x], dim=0),
        lambda x: x.detach(),
        lambda x: x[:],
        lambda x: x.contiguous(),
    ],
    ids=[
        "reshape",
        "view",
        "transpose",
        "slice",
        "add",
        "nested_args",
        "detach",
        "full_slice",
        "contiguous",
    ],
)
def test_torch_consumer_uses_finalized_warp_output(pending_output, consume):
    node, tensor, segment, session = pending_output
    result = consume(tensor)

    assert session.active_segment is None
    assert segment.marker_proxy is not None
    # A no-op alias shares the finalized proxy; other consumers depend on it.
    # Eager values alone cannot detect a stale graph reference.
    assert tensor.proxy.node in [result.proxy.node, *result.proxy.node.all_input_nodes]
    assert segment.marker_proxy.node in tensor.proxy.node.all_input_nodes
    node.graph.lint()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda x: x.copy_(torch.zeros(2, 3)),
        lambda x: x.__setitem__((slice(None), 0), 0.0),
    ],
    ids=["copy", "setitem"],
)
def test_torch_write_closes_segment_before_mutating_storage(pending_output, mutate):
    _, tensor, _, session = pending_output
    operation = session.active_warp_op
    terminate = operation.terminate

    def check_before_close(*args, **kwargs):
        torch.testing.assert_close(tensor.tensor, torch.arange(6.0).reshape(2, 3) + 7)
        return terminate(*args, **kwargs)

    with mock.patch.object(
        operation, "terminate", side_effect=check_before_close
    ) as close:
        mutate(tensor)
    close.assert_called_once()


def test_internal_torch_work_does_not_close_paused_segment(pending_output):
    _, tensor, segment, session = pending_output
    with session.pause():
        tensor.reshape(3, 2)
    assert session.active_segment is segment


def test_torch_boundary_respects_explicit_owner(pending_output):
    _, tensor, segment, session = pending_output
    session.owner_token = object()
    tensor.reshape(3, 2)
    assert session.active_segment is segment
