# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime integer shape expressions for traced tensor operations."""

from contextvars import ContextVar
from functools import wraps
import operator

import torch

from leapp.leapp_graph.datatypes.primitives.traced_int import TracedInt, graph_value
from leapp.leapp_graph.datatypes.traced_data import TracedData
from leapp.utils.shape_contracts import has_variable_inputs

_DYNAMIC_SHAPE_TRACING_SUPPRESSED = ContextVar(
    "leapp_dynamic_shape_tracing_suppressed", default=False
)


def suppress_dynamic_shape_tracing(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _DYNAMIC_SHAPE_TRACING_SUPPRESSED.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            _DYNAMIC_SHAPE_TRACING_SUPPRESSED.reset(token)

    return wrapped


def enabled(tensor):
    return (
        not _DYNAMIC_SHAPE_TRACING_SUPPRESSED.get()
        and tensor.is_tracing
        and not tensor.describes_replaced_graph()
        and has_variable_inputs(tensor.context_obj)
    )


@suppress_dynamic_shape_tracing
def dimension(tensor, axis):
    axis = operator.index(axis)
    value = tensor.tensor.size(axis)
    proxy = tensor.context_obj.tracer.create_proxy(
        "call_method", "size", (tensor.proxy, axis), {}
    )
    return TracedInt(value, proxy.node.name, tensor.context_obj, proxy)


@suppress_dynamic_shape_tracing
def numel(tensor):
    proxy = tensor.context_obj.tracer.create_proxy(
        "call_method", "numel", (tensor.proxy,), {}
    )
    return TracedInt(tensor.tensor.numel(), proxy.node.name, tensor.context_obj, proxy)


def _convert_shape_value(value, owner, proxy):
    if isinstance(value, TracedInt):
        if value.is_recording and value.context_obj is not owner.context_obj:
            raise ValueError("Shape expressions belong to different nodes")
        return graph_value(value) if proxy else int(value)
    return value


@suppress_dynamic_shape_tracing
def trace_call(function, args, kwargs, *, shape, method=False):
    from leapp.leapp_graph.datatypes.torch.traced_tensor import TracedTensor

    owner = shape
    name = function if method else getattr(function, "__name__", "")
    if kwargs.get("out") is not None:
        raise ValueError("Shape-dependent out= operations are unsupported")
    if name.endswith("_") and not name.startswith("__"):
        target = getattr(torch.Tensor, name) if method else function
        handled, result = TracedTensor._handle_functional_inplace(target, args, kwargs)
        if handled:
            return result
        raise ValueError(f"Unsupported shape-dependent mutation: {name}")

    def leaf(item, proxy):
        if isinstance(item, TracedInt):
            return _convert_shape_value(item, owner, proxy)
        if isinstance(item, TracedData):
            if item.context_obj is not owner.context_obj:
                raise ValueError(
                    "Shape consumers and tensors belong to different nodes"
                )
            return item.proxy if proxy else item.tensor
        return item

    real_args = TracedData._map_structure(args, lambda x: leaf(x, False))
    real_kwargs = TracedData._map_structure(kwargs, lambda x: leaf(x, False))
    if method:
        value = getattr(real_args[0], function)(*real_args[1:], **real_kwargs)
    else:
        value = function(*real_args, **real_kwargs)
    # Only elide a view when the argument expressions prove identity for every
    # admitted shape. Equal sample layouts alone are not sufficient.
    if name in {"reshape", "view"} and len(args) == 2:
        source, dimensions = args
        if isinstance(source, TracedData) and isinstance(dimensions, (tuple, list)):
            if len(dimensions) == source.tensor.ndim and all(
                isinstance(d, TracedInt)
                and d.proxy.node.op == "call_method"
                and d.proxy.node.target == "size"
                and d.proxy.node.args == (source.proxy.node, i)
                for i, d in enumerate(dimensions)
            ):
                alias = source._new_alias(value)
                TracedTensor._mark_data_dependent_shape(alias)
                return alias
    if not isinstance(value, (torch.Tensor, tuple, list)):
        raise ValueError(
            f"Unsupported shape-dependent operation {name or function!r} in node "
            f"{owner.context}: expected tensor outputs, got {type(value).__name__}"
        )
    proxy_args = TracedData._map_structure(args, lambda x: leaf(x, True))
    proxy_kwargs = TracedData._map_structure(kwargs, lambda x: leaf(x, True))
    tracer = owner.context_obj.tracer
    proxy = tracer.create_proxy(
        "call_method" if method else "call_function", function, proxy_args, proxy_kwargs
    )
    from leapp.leapp_graph.datatypes import as_traced

    def wrap(output, output_proxy):
        traced = as_traced(
            output, output_proxy.node.name, owner.context_obj, output_proxy
        )
        # Size reads in a static node return traced sizes only for flagged
        # tensors; callers that bypass ``__torch_function__`` skip its flag.
        TracedTensor._mark_data_dependent_shape(traced)
        return traced

    if isinstance(value, torch.Tensor):
        return wrap(value, proxy)
    # Multi-output operations such as ``torch.split`` index the recorded call.
    return type(value)(
        [
            wrap(
                output,
                tracer.create_proxy(
                    "call_function", operator.getitem, (proxy, index), {}
                ),
            )
            if isinstance(output, torch.Tensor)
            else output
            for index, output in enumerate(value)
        ]
    )
