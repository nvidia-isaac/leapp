# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture NumPy operations through PyTorch's private compatibility layer.

A small override registry handles compatibility gaps; other calls use PyTorch's
NumPy implementation. Each call is captured independently, checked against real NumPy, then
inlined as ATen into the owning node. No torch._numpy objects survive capture.
Capture-sample parity is a guard, not a guarantee for every runtime input.
"""

import inspect
import operator
from functools import lru_cache

import numpy as np
import torch
import torch._numpy as tnp
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import map_arg

from ..traced_data import TracedData
from .overrides import NUMPY_OVERRIDES


@lru_cache(maxsize=1)
def _implementations():
    mappings = {}
    # Match actual function identity within known namespaces, not just __name__.
    # Random/stateful functions are intentionally outside this adapter.
    for numpy_module, torch_module in (
        (np, tnp),
        (np.linalg, tnp.linalg),
        (np.fft, tnp.fft),
    ):
        for name, implementation in vars(torch_module).items():
            original = getattr(numpy_module, name, None)
            if (
                not name.startswith("_")
                and callable(original)
                and callable(implementation)
            ):
                mappings[original] = implementation
    return mappings


def capture_numpy_call(source, func, args, kwargs):
    """Return traced outputs, or fail before adding an invalid lowering."""
    label = f"NumPy {func.__module__}.{func.__name__} in node {source.context}"
    implementation = NUMPY_OVERRIDES.get(func, _implementations().get(func))
    if implementation is None:
        raise NotImplementedError(f"{label}: no torch._numpy lowering")
    # Bind the compatibility signature so positional out/overwrite_input are
    # rejected too, before either implementation can mutate user storage.
    bound = inspect.signature(implementation).bind(*args, **kwargs)
    if bound.arguments.get("out") is not None or bound.arguments.get(
        "overwrite_input", False
    ):
        raise NotImplementedError(
            f"{label}: out mutation is unsupported by the ATen adapter"
        )
    # A NumPy call may itself be a mutator (copyto, put, etc.). Never
    # execute an unvalidated implementation against the caller's storage.
    native_args, native_kwargs = TracedData._map_structure(
        TracedData.unwrap_traced_data((args, kwargs)),
        lambda x: x.copy() if isinstance(x, np.ndarray) else x,
    )
    expected = func(*native_args, **native_kwargs)
    # Dtype metadata is static under the existing LEAPP input contract.
    if func is np.result_type:
        return expected

    inputs = []
    TracedData._map_structure(
        (args, kwargs), lambda x: inputs.append(x) if isinstance(x, TracedData) else x
    )
    examples = tuple(torch.from_numpy(np.array(x.data, copy=True)) for x in inputs)

    def lower(*tensors):
        values = iter(tensors)

        def convert(x):
            if isinstance(x, TracedData):
                return tnp.asarray(next(values))
            if isinstance(x, np.dtype) or (
                isinstance(x, type) and issubclass(x, np.generic)
            ):
                return tnp.dtype(np.dtype(x).name)
            if isinstance(x, (np.ndarray, np.generic)):
                return tnp.asarray(torch.from_numpy(np.array(x, copy=True)))
            return x

        converted_args, converted_kwargs = TracedData._map_structure(
            (args, kwargs), convert
        )
        result = implementation(*converted_args, **converted_kwargs)
        return TracedData._map_structure(
            result, lambda x: x.tensor if isinstance(x, tnp.ndarray) else x
        )

    try:
        captured = make_fx(lower)(*examples)
        # Composite ATen kernels (e.g. isclose) introduce temporary mutations
        # during decomposition. Functionalize the decomposed graph itself.
        captured = make_fx(torch.func.functionalize(captured))(*examples)
        actual = captured(*examples)
        actual_leaves, expected_leaves = [], []
        TracedData._map_structure(actual, actual_leaves.append)
        TracedData._map_structure(expected, expected_leaves.append)
        if len(actual_leaves) != len(expected_leaves):
            raise ValueError("output structure differs from NumPy")
        for value, reference in zip(actual_leaves, expected_leaves):
            if not isinstance(value, torch.Tensor):
                raise ValueError("non-tensor output would lose runtime dependence")
            reference = torch.from_numpy(np.array(reference, copy=True))
            torch.testing.assert_close(value, reference, equal_nan=True)
        for node in captured.graph.nodes:
            if node.op not in ("placeholder", "get_attr", "call_function", "output"):
                raise ValueError(f"unsupported graph node: {node.op}")
            if node.op == "call_function" and node.target is not operator.getitem:
                if not isinstance(node.target, torch._ops.OpOverload):
                    raise ValueError(f"non-ATen operation: {node.target}")
                if node.target._schema.is_mutable:
                    raise ValueError(f"mutable operation: {node.target}")
    except Exception as exc:
        raise NotImplementedError(
            f"{label}: ATen capture/parity check failed: {exc}"
        ) from exc

    # Once the lowering is known to be functional, execute NumPy on the real
    # operands so eager view/storage behavior is retained (e.g. broadcast_to).
    expected = func(
        *TracedData.unwrap_traced_data(args), **TracedData.unwrap_traced_data(kwargs)
    )
    tracer = source._context.tracer
    bindings = {}
    proxies = iter(x.proxy for x in inputs)
    for node in captured.graph.nodes:
        if node.op == "placeholder":
            bindings[node] = next(proxies)
        elif node.op == "get_attr":
            value = captured
            for part in node.target.split("."):
                value = getattr(value, part)
            bindings[node] = (
                value.detach().clone() if isinstance(value, torch.Tensor) else value
            )
        elif node.op == "call_function":
            # TorchScript resolves overload packets, but not .default/.Tensor
            # attributes. ATen's dispatcher selects the captured signature.
            target = (
                node.target.overloadpacket
                if isinstance(node.target, torch._ops.OpOverload)
                else node.target
            )
            if node.target is torch.ops.aten.lift_fresh_copy.default:
                # Constants need a copy, but legacy ONNX has no lift lowering.
                target = torch.clone
            bindings[node] = tracer.create_proxy(
                node.op,
                target,
                map_arg(node.args, bindings.__getitem__),
                map_arg(node.kwargs, bindings.__getitem__),
            )
        elif node.op == "output":
            output = map_arg(node.args[0], bindings.__getitem__)

    output_leaves = []
    TracedData._map_structure(output, output_leaves.append)
    output_proxies = iter(output_leaves)
    return TracedData._map_structure(
        expected, lambda x: source._new(np.asarray(x), next(output_proxies))
    )
