# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture NumPy operations through PyTorch's private compatibility layer.

A small override registry handles compatibility gaps; other calls use PyTorch's
NumPy implementation. Each call is captured independently and structurally checked,
then inlined as ATen into the owning node. No torch._numpy objects survive capture.
Numerical validation happens on the exported node outputs.
"""

import inspect
import operator
from functools import lru_cache
from typing import Union

import numpy as np
import torch
import torch._numpy as tnp
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import map_arg

from ..primitives.traced_int import TracedInt
from leapp.utils.shape_contracts import has_variable_inputs
from leapp.utils.logging import _get_logger

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


class _NumpyCaptureError(Exception):
    """A valid NumPy call whose tensor implementation could not be captured."""


class _NumpyMutationError(NotImplementedError):
    """An unsupported mutation cannot be treated as an unused pure branch."""


def _failed_numpy_capture(message, *inputs):
    """Dependency marker; live markers are rejected before graph export."""
    raise NotImplementedError(message)


def capture_numpy_call(source, func, args, kwargs):
    """Log recoverable translation errors and retain their output dependencies."""
    label = f"NumPy {func.__module__}.{func.__name__} in node {source.context}"
    implementation = NUMPY_OVERRIDES.get(func, _implementations().get(func))
    if func is np.result_type:
        return func(
            *TracedData.unwrap_traced_data(args),
            **TracedData.unwrap_traced_data(kwargs),
        )

    try:
        if implementation is None:
            try:
                native_bound = inspect.signature(func).bind(*args, **kwargs).arguments
            except (TypeError, ValueError):
                native_bound = kwargs
            positional_out = isinstance(func, np.ufunc) and any(
                value is not None for value in args[func.nin : func.nin + func.nout]
            )
            if (
                native_bound.get("out") is not None
                or native_bound.get("overwrite_input", False)
                or (func is np.nan_to_num and not native_bound.get("copy", True))
                or positional_out
            ):
                raise NotImplementedError(f"{label}: untraced mutation is unsupported")
            raise _NumpyCaptureError(f"{label}: no torch._numpy lowering")
        try:
            bound = inspect.signature(implementation).bind(*args, **kwargs)
        except TypeError as exc:
            raise _NumpyCaptureError(f"{label}: unsupported signature: {exc}") from exc
        if (
            bound.arguments.get("out") is not None
            or bound.arguments.get("overwrite_input", False)
            or (func is np.nan_to_num and not bound.arguments.get("copy", True))
        ):
            raise NotImplementedError(
                f"{label}: mutation is unsupported by the ATen adapter"
            )
        return _capture_numpy_call(source, func, args, kwargs, implementation, label)
    except _NumpyCaptureError as exc:
        # A failed lowering cannot establish that the native call is functional.
        # Probe only this error path on copies, protecting caller storage from
        # unsupported mutators such as copyto/put (which return None). Native
        # NumPy errors propagate before a recoverable capture error is logged.
        native_args, native_kwargs = TracedData._map_structure(
            TracedData.unwrap_traced_data((args, kwargs)),
            lambda x: x.copy() if isinstance(x, np.ndarray) else x,
        )
        if func(*native_args, **native_kwargs) is None:
            raise NotImplementedError(f"{label}: untraced mutation is unsupported")
        expected = func(
            *TracedData.unwrap_traced_data(args),
            **TracedData.unwrap_traced_data(kwargs),
        )
        return record_numpy_capture_failure(source, expected, args, kwargs, str(exc))


def record_numpy_capture_failure(source, expected, args, kwargs, message):
    from .traced_np_array import TracedNpArray

    _get_logger().error(
        f"{message}. Continuing eager execution; export requires this operation "
        "to be independent of the declared outputs."
    )
    inputs = []
    TracedData._map_structure(
        (args, kwargs),
        lambda x: inputs.append(x.proxy) if isinstance(x, TracedData) else x,
    )

    def wrap(value):
        proxy = source.context_obj.tracer.create_proxy(
            "call_function", _failed_numpy_capture, (message, *inputs), {}
        )
        proxy.node.meta["leapp_capture_error"] = message
        return TracedNpArray(
            np.asarray(value), proxy.node.name, source.context_obj, proxy
        )

    return TracedData._map_structure(expected, wrap)


def _shape_tensor(
    value: Union[int, torch.Tensor], dtype: torch.dtype, device: torch.device
):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.scalar_tensor(value, dtype=dtype, device=device)


def _capture_numpy_call(source, func, args, kwargs, implementation, label):
    inputs = []
    TracedData._map_structure(
        (args, kwargs),
        lambda x: inputs.append(x) if isinstance(x, TracedData) else x,
    )

    scalar_dtype = None
    if isinstance(func, np.ufunc) and any(isinstance(x, TracedInt) for x in inputs):
        dtype = np.result_type(*TracedData.unwrap_traced_data(args[: func.nin]))
        scalar_dtype = torch.from_numpy(np.empty((), dtype=dtype)).dtype

    def lower(*tensors):
        values = iter(tensors)

        def convert(x):
            if isinstance(x, TracedInt):
                value = next(values)
                # Ufunc scalar promotion otherwise converts symbolic integers
                # to capture-time Python values. Shape arguments stay integers.
                if isinstance(func, np.ufunc):
                    return tnp.asarray(torch.scalar_tensor(value, dtype=scalar_dtype))
                return value
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
        examples = tuple(
            int(x)
            if isinstance(x, TracedInt)
            else torch.from_numpy(np.array(x.data, copy=True))
            for x in inputs
        )
        mode = "symbolic" if has_variable_inputs(source.context_obj) else "real"
        captured = make_fx(lower, tracing_mode=mode)(*examples)
        # Composite ATen kernels (e.g. isclose) introduce temporary mutations
        # during decomposition. Functionalize the decomposed graph itself.
        captured = make_fx(torch.func.functionalize(captured), tracing_mode=mode)(
            *examples
        )
        for node in captured.graph.nodes:
            if (
                isinstance(node.target, torch._ops.OpOverload)
                and node.target._schema.is_mutable
            ):
                raise _NumpyMutationError(f"{label}: mutable operation: {node.target}")
        output_nodes = []
        output_node = next(node for node in captured.graph.nodes if node.op == "output")
        TracedData._map_structure(output_node.args[0], output_nodes.append)
        if not output_nodes or any(
            not isinstance(node, torch.fx.Node)
            or not isinstance(node.meta.get("val"), torch.Tensor)
            for node in output_nodes
        ):
            raise ValueError("non-tensor output would lose runtime dependence")
        for node in captured.graph.nodes:
            if node.op not in ("placeholder", "get_attr", "call_function", "output"):
                raise ValueError(f"unsupported graph node: {node.op}")
            if node.op == "call_function" and node.target is not operator.getitem:
                if not isinstance(node.target, torch._ops.OpOverload) and not (
                    mode == "symbolic"
                    and node.target
                    in (
                        operator.add,
                        operator.sub,
                        operator.mul,
                        operator.floordiv,
                        operator.mod,
                    )
                ):
                    raise ValueError(f"non-ATen operation: {node.target}")

    except _NumpyMutationError:
        raise
    except Exception as exc:
        raise _NumpyCaptureError(f"{label}: ATen capture failed: {exc}") from exc

    # Once the lowering is known to be functional, execute NumPy on the real
    # operands so eager view/storage behavior is retained (e.g. broadcast_to).
    expected = func(
        *TracedData.unwrap_traced_data(args), **TracedData.unwrap_traced_data(kwargs)
    )
    expected_leaves = []
    TracedData._map_structure(expected, expected_leaves.append)
    if len(output_nodes) != len(expected_leaves):
        raise _NumpyCaptureError(f"{label}: output structure differs from NumPy")
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
            call_kwargs = map_arg(node.kwargs, bindings.__getitem__)
            if (
                mode == "symbolic"
                and node.target is torch.ops.aten.scalar_tensor.default
            ):
                # JIT tracing represents runtime sizes as scalar tensors.
                target = _shape_tensor
                call_kwargs = {
                    k: v for k, v in call_kwargs.items() if k in ("dtype", "device")
                }
            operation = node.op
            if mode == "symbolic":
                if node.target in (
                    torch.ops.aten.view.default,
                    torch.ops.aten.reshape.default,
                ):
                    target = torch.reshape
                elif node.target is torch.ops.aten.sym_size.int:
                    operation, target = "call_method", "size"
                elif node.target is torch.ops.aten.sym_numel.default:
                    operation, target = "call_method", "numel"
            bindings[node] = tracer.create_proxy(
                operation,
                target,
                map_arg(node.args, bindings.__getitem__),
                call_kwargs,
            )
        elif node.op == "output":
            output = map_arg(node.args[0], bindings.__getitem__)

    output_leaves = []
    TracedData._map_structure(output, output_leaves.append)
    output_proxies = iter(output_leaves)
    return TracedData._map_structure(
        expected, lambda x: source._new(np.asarray(x), next(output_proxies))
    )
