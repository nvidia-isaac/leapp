#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Runtime integer shape expressions for traced tensor operations."""

import operator
from contextvars import ContextVar
from functools import wraps

from leapp.leapp_graph.datatypes.primitives.traced_int import TracedInt, graph_sizes
from leapp.leapp_graph.datatypes.traced_data import TracedData
from leapp.utils.logging import _get_logger
from leapp.utils.shape_contracts import has_variable_inputs

_DYNAMIC_SHAPE_TRACING_SUPPRESSED = ContextVar(
    "leapp_dynamic_shape_tracing_suppressed", default=False
)


def suppress_dynamic_shape_tracing(function):
    """Run ``function`` with variable-axis size reads returning plain integers."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _DYNAMIC_SHAPE_TRACING_SUPPRESSED.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            _DYNAMIC_SHAPE_TRACING_SUPPRESSED.reset(token)

    return wrapped


def enabled(tensor):
    """Whether variable-axis size reads on ``tensor`` return traced sizes."""
    return (
        not _DYNAMIC_SHAPE_TRACING_SUPPRESSED.get()
        and tensor.is_tracing
        and not tensor.describes_replaced_graph()
        and has_variable_inputs(tensor.context_obj)
    )


@suppress_dynamic_shape_tracing
def dimension(tensor, axis):
    """Return ``tensor.size(axis)`` as a traced integer recorded in the graph."""
    axis = operator.index(axis)
    value = tensor.tensor.size(axis)
    proxy = tensor.context_obj.tracer.create_proxy(
        "call_method", "size", (tensor.proxy, axis), {}
    )
    return TracedInt(value, proxy.node.name, tensor.context_obj, proxy)


@suppress_dynamic_shape_tracing
def numel(tensor):
    """Return ``tensor.numel()`` as a traced integer recorded in the graph."""
    proxy = tensor.context_obj.tracer.create_proxy(
        "call_method", "numel", (tensor.proxy,), {}
    )
    return TracedInt(tensor.tensor.numel(), proxy.node.name, tensor.context_obj, proxy)


@suppress_dynamic_shape_tracing
def trace_factory(function, args, kwargs, *, shape):
    """Run a tensor factory and record it with its traced sizes.

    ``shape`` is a recording size among the arguments and identifies the node.
    """
    from leapp.leapp_graph.datatypes import as_traced
    from leapp.leapp_graph.datatypes.torch.traced_tensor import TracedTensor

    node = shape.context_obj
    if kwargs.get("out") is not None:
        _get_logger().fatal(
            f"{function.__name__}() with a traced size does not support out=",
            error_type=ValueError,
        )
    # A torch.Size cannot hold the proxies that replace its sizes below.
    args, kwargs = graph_sizes((args, kwargs))

    def leaf(item, proxy):
        if isinstance(item, TracedData):
            if item.context_obj is not node:
                _get_logger().fatal(
                    f"{function.__name__}() in node {node.name} takes a size or "
                    f"tensor of node {item.context}",
                    error_type=ValueError,
                )
            if proxy:
                return item.proxy
            return int(item) if isinstance(item, TracedInt) else item.tensor
        return item

    value = function(
        *TracedData._map_structure(args, lambda x: leaf(x, False)),
        **TracedData._map_structure(kwargs, lambda x: leaf(x, False)),
    )
    proxy = node.tracer.create_proxy(
        "call_function",
        function,
        TracedData._map_structure(args, lambda x: leaf(x, True)),
        TracedData._map_structure(kwargs, lambda x: leaf(x, True)),
    )
    traced = as_traced(value, proxy.node.name, node, proxy)
    # Size reads in a static node return traced sizes only for flagged tensors.
    TracedTensor._mark_data_dependent_shape(traced)
    return traced
