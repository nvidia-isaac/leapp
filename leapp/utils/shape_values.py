# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime integer shape expressions for traced tensor operations."""

from contextvars import ContextVar
from functools import wraps
import operator

import numpy as np
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
            if item.is_recording and item.context_obj is not owner.context_obj:
                raise ValueError("Shape expressions belong to different nodes")
            return graph_value(item) if proxy else int(item)
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
                return source._new_alias(value)
    proxy_args = TracedData._map_structure(args, lambda x: leaf(x, True))
    proxy_kwargs = TracedData._map_structure(kwargs, lambda x: leaf(x, True))
    proxy = owner.context_obj.tracer.create_proxy(
        "call_method" if method else "call_function", function, proxy_args, proxy_kwargs
    )
    if not isinstance(value, torch.Tensor):
        raise ValueError("Unsupported non-tensor shape consumer")
    from leapp.leapp_graph.datatypes import as_traced

    return as_traced(value, proxy.node.name, owner.context_obj, proxy)


@suppress_dynamic_shape_tracing
def trace_numpy_factory(function, args, kwargs, shape):
    """Execute NumPy factories on concrete sizes and record Torch equivalents."""
    from leapp.leapp_graph.datatypes.numpy.traced_np_array import TracedNpArray

    owner = shape

    def convert(value, proxy):
        if isinstance(value, TracedInt):
            if value.is_recording and value.context_obj is not owner.context_obj:
                raise ValueError("Shape expressions belong to different nodes")
            return graph_value(value) if proxy else int(value)
        return value

    real_args = TracedData._map_structure(args, lambda v: convert(v, False))
    real_kwargs = TracedData._map_structure(kwargs, lambda v: convert(v, False))
    result = function(*real_args, **real_kwargs)
    proxy_args = TracedData._map_structure(args, lambda v: convert(v, True))
    proxy_kwargs = TracedData._map_structure(kwargs, lambda v: convert(v, True))
    try:
        if set(kwargs) - {"dtype", "shape", "start", "stop", "step", "fill_value"}:
            raise TypeError("Unsupported NumPy factory keyword")
        # NumPy and Torch have different factory defaults; retain NumPy's dtype.
        proxy_kwargs["dtype"] = torch.from_numpy(np.empty((), dtype=result.dtype)).dtype
        if "shape" in proxy_kwargs:
            proxy_kwargs["size"] = proxy_kwargs.pop("shape")
        if "stop" in proxy_kwargs:
            proxy_kwargs["end"] = proxy_kwargs.pop("stop")
        if function.__name__ in {"zeros", "ones", "full"}:
            data_arguments = 2 if function.__name__ == "full" else 1
            # Optional positional dtype is represented by the result dtype above.
            if len(proxy_args) > data_arguments + 1:
                raise TypeError("Unsupported NumPy factory positional arguments")
            proxy_args = proxy_args[:data_arguments]
            if proxy_args and not isinstance(proxy_args[0], (tuple, list)):
                proxy_args = ((proxy_args[0],),) + proxy_args[1:]
        elif len(proxy_args) == 4:
            proxy_args = proxy_args[:3]  # arange(start, stop, step, dtype)
        if "size" in proxy_kwargs and not isinstance(
            proxy_kwargs["size"], (tuple, list)
        ):
            proxy_kwargs["size"] = (proxy_kwargs["size"],)
        proxy = owner.context_obj.tracer.create_proxy(
            "call_function", getattr(torch, function.__name__), proxy_args, proxy_kwargs
        )
    except (TypeError, ValueError, NotImplementedError) as exc:
        from leapp.leapp_graph.datatypes.numpy.aten_adapter import (
            record_numpy_capture_failure,
        )

        message = f"NumPy {function.__name__} in node {owner.context}: {exc}"
        return record_numpy_capture_failure(owner, result, args, kwargs, message)
    return TracedNpArray(result, proxy.node.name, owner.context_obj, proxy)
