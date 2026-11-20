# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate bounded shape samples and replay annotated functions for validation."""

import copy
import inspect
import math
import random
from contextlib import ExitStack, contextmanager

import numpy as np
import torch

from leapp.backends.export_backend import prepare_tensors_for_export
from leapp.leapp_graph.datatypes import TracedData, to_export_torch_tensor
from leapp.utils.logging import _get_logger
from leapp.utils.semantics import TensorSemantics
from leapp.utils.tensor_description import flatten_io_structure
from leapp.utils.shape_contracts import (
    evaluate_dimension_expression,
    get_dimension_expression,
    get_shape_contract,
    has_variable_axis_declaration,
    has_variable_inputs,
    iter_root_dimension_ranges,
    validate_shape_contract,
)


def sample_validation_shapes(
    contracts,
    sample_shapes,
    item_sizes,
    *,
    seed=0,
    max_samples=5,
    max_input_bytes=64 * 1024 * 1024,
):
    """Return distinct admitted shapes and reasons for omitted candidate sizes."""
    bindings = {}
    ranges = {}
    for name, contract in contracts.items():
        validate_shape_contract(contract, sample_shapes[name], bindings, name=name)
        for symbol, bounds in iter_root_dimension_ranges(contract):
            if symbol in ranges and ranges[symbol] != bounds:
                raise ValueError(f"Conflicting bounds for {symbol}")
            ranges[symbol] = bounds
    rng = random.Random(seed)
    seen = {tuple(sample_shapes.values())}
    samples, omitted = [], []
    for symbol, bounds in ranges.items():
        stride = bounds.get("multiple_of", 1)
        lower = (bounds["min"] + stride - 1) // stride
        upper = bounds["max"] // stride
        sizes = [lower, upper, (lower + upper) // 2, rng.randint(lower, upper)]
        if lower <= 1 <= upper:
            sizes.insert(2, 1)
        for size in sizes:
            values = {**bindings, symbol: size * stride}
            shapes = {
                name: tuple(
                    evaluate_dimension_expression(
                        get_dimension_expression(dim, contract), values
                    )
                    if isinstance(dim, str)
                    else sample_shapes[name][axis]
                    for axis, dim in enumerate(contract["shape"])
                )
                for name, contract in contracts.items()
            }
            key = tuple(shapes.values())
            if key in seen:
                continue
            seen.add(key)
            checked = {}
            for name, contract in contracts.items():
                validate_shape_contract(contract, shapes[name], checked, name=name)
            byte_count = sum(
                math.prod(shape) * item_sizes[name] for name, shape in shapes.items()
            )
            if byte_count > max_input_bytes:
                omitted.append(f"{shapes}: input memory budget exceeded")
            elif len(samples) >= max_samples:
                omitted.append(f"{shapes}: sample budget exceeded")
            else:
                samples.append(shapes)
    return samples, omitted


def _is_plain_config(value):
    """Whether ``value`` is a scalar or a container holding only scalars."""
    if isinstance(value, (list, tuple)):
        return all(_is_plain_config(item) for item in value)
    if isinstance(value, dict):
        return all(_is_plain_config(item) for pair in value.items() for item in pair)
    return isinstance(value, (int, float, bool, str, type(None)))


def _copy_input(value):
    value = TracedData.unwrap_traced_data(value)
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (list, tuple, dict)) and _is_plain_config(value):
        # The function may edit its configuration; every call gets its own.
        return copy.deepcopy(value)
    return value


def generate_validation_inputs(samples, shapes, seed):
    """Generate numeric inputs, retaining captured values when shapes do not change."""
    generated = {}
    for index, (name, sample) in enumerate(samples.items()):
        shape = shapes[name]
        if tuple(sample.shape) == shape:
            value = _copy_input(sample)
        else:
            generator = torch.Generator(device=sample.device).manual_seed(seed + index)
            options = {
                "dtype": sample.dtype,
                "device": sample.device,
                "generator": generator,
            }
            if sample.is_floating_point() or sample.is_complex():
                value = torch.rand(shape, **options)
            else:
                value = torch.randint(0, 2, shape, **options)
        generated[name] = value
    return generated


@contextmanager
def _validation_random_state(samples, seed):
    devices = sorted(
        {
            v.device.index
            for v in samples.values()
            if isinstance(v, torch.Tensor) and v.is_cuda
        }
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class MethodValidation:
    """Keep a reference callable and its argument bindings until validation.

    Arguments are stored under the names ``annotate.method`` gives their ports:
    the parameter name, ``arg_<i>`` for the i-th ``*args`` value, and the
    keyword for a ``**kwargs`` value.
    """

    def __init__(self, function, bound, factory, descriptions):
        self.function = function
        self.factory = factory
        self.descriptions = descriptions
        self.signature = bound.signature
        self.variable_arguments = {
            name
            for name, desc in descriptions.items()
            if has_variable_axis_declaration(desc)
            or (desc.has_source and has_variable_inputs(desc.source_node))
        }
        self.grad_enabled = torch.is_grad_enabled()
        self.inference_mode = torch.is_inference_mode_enabled()
        self.autocast = {
            device: (
                torch.is_autocast_enabled(device),
                torch.get_autocast_dtype(device),
            )
            for device in ("cpu", "cuda")
        }
        self.arguments = {}
        self.argument_groups = {}
        self.positional_count = 0
        self.keyword_names = []
        self.ambiguous_name = None

        def keep(name, value):
            if name in self.arguments:
                self.ambiguous_name = name
            if isinstance(value, TensorSemantics):
                value = value.ref
            if name in descriptions:
                self.argument_groups.setdefault(id(value), []).append(name)
            value = TracedData.unwrap_traced_data(value)
            self.arguments[name] = _copy_input(value)

        for name, value in bound.arguments.items():
            kind = self.signature.parameters[name].kind
            if name in ("self", "cls"):
                self.arguments[name] = value
            elif kind is inspect.Parameter.VAR_POSITIONAL:
                self.positional_count = len(value)
                for index, item in enumerate(value):
                    keep(f"arg_{index}", item)
            elif kind is inspect.Parameter.VAR_KEYWORD:
                self.keyword_names = list(value)
                for keyword, item in value.items():
                    keep(keyword, item)
            else:
                keep(name, value)

    def _bind(self, arguments):
        """Rebuild the captured call from arguments stored under port names."""
        bound = {}
        for name, parameter in self.signature.parameters.items():
            if parameter.kind is parameter.VAR_POSITIONAL:
                bound[name] = tuple(
                    arguments[f"arg_{index}"] for index in range(self.positional_count)
                )
            elif parameter.kind is parameter.VAR_KEYWORD:
                bound[name] = {key: arguments[key] for key in self.keyword_names}
            elif name in arguments:
                bound[name] = arguments[name]
        return inspect.BoundArguments(self.signature, bound)

    def _replay_obstacle(self, node):
        """Why the captured call cannot be replayed on generated inputs, or None."""
        if self.ambiguous_name is not None:
            return (
                f"generated validation cannot tell apart two arguments named "
                f"{self.ambiguous_name!r}"
            )
        for name, value in self.arguments.items():
            if name in ("self", "cls"):
                continue
            if not isinstance(value, torch.Tensor) and not _is_plain_config(value):
                return (
                    f"generated validation does not support argument {name!r}; "
                    "pass tensors, scalars, or containers of scalars"
                )
        if set(node.inputs) - set(self.descriptions.values()):
            return "generated validation requires all node inputs to be function parameters"
        return None

    def validate(self, node, rtol, atol):
        for names in self.argument_groups.values():
            retained = next(
                (
                    self.descriptions[name]
                    for name in names
                    if self.descriptions[name] in node.inputs
                ),
                None,
            )
            if retained is not None:
                for name in names:
                    self.descriptions[name] = retained
        samples = {name: self.arguments[name] for name in self.descriptions}
        if any(
            self.descriptions[name] not in node.inputs
            for name in self.variable_arguments
        ):
            raise ValueError(
                "Cannot validate a variable input removed from the exported graph"
            )
        contracts = {
            name: get_shape_contract(desc) for name, desc in self.descriptions.items()
        }
        # A static contract admits no other shape, so nothing is replayed.
        if not any(contract.get("dimension_ranges") for contract in contracts.values()):
            return True
        obstacle = self._replay_obstacle(node)
        if obstacle is not None:
            # A requested callback must run; automatic generation is best effort.
            if self.factory is not None:
                raise ValueError(f"{node.name}: {obstacle}")
            _get_logger().warning(
                f"{node.name}: skipped generated shape validation: {obstacle}. "
                "Only the captured samples were compared; pass "
                "validation_inputs=False to disable generated validation."
            )
            return True
        shapes, omitted = sample_validation_shapes(
            contracts,
            {name: tuple(value.shape) for name, value in samples.items()},
            {name: value.element_size() for name, value in samples.items()},
        )
        for reason in omitted:
            _get_logger().warning(f"{node.name}: generated validation omitted {reason}")
        if not shapes:
            raise ValueError(
                "No additional dynamic shapes could be validated; provide bounded variable axes within the input memory budget"
            )
        node.export_backend.load(node.model_path, node.sha256sum)
        for seed, target_shapes in enumerate(shapes):
            label = f"generated sample {seed}, shapes={target_shapes}"
            try:
                with _validation_random_state(samples, seed):
                    factory = self.factory or generate_validation_inputs
                    try:
                        generated = factory(
                            {name: _copy_input(v) for name, v in samples.items()},
                            dict(target_shapes),
                            seed,
                        )
                    except Exception as exc:
                        raise ValueError(
                            "Validation could not complete: input generation failed; "
                            f"provide validation_inputs for domain-specific data. {exc}"
                        ) from exc
                self._validate_inputs(generated, samples, target_shapes, contracts)
                self._bind_aliased_inputs(generated)
                expected = self._reference_outputs(generated, samples, seed)
                by_port = {
                    desc.name: generated[name]
                    for name, desc in self.descriptions.items()
                }
                inputs = prepare_tensors_for_export(
                    [to_export_torch_tensor(by_port[desc.name]) for desc in node.inputs]
                )
                inputs = [
                    value.to(node.export_backend.runtime_device) for value in inputs
                ]
                with _validation_random_state(dict(enumerate(inputs)), seed):
                    if node._sample_mismatches(label, inputs, expected, rtol, atol):
                        raise ValueError(
                            "Exported outputs differ from the original function"
                        )
            except Exception as exc:
                raise ValueError(f"{node.name} {label}: {exc}") from exc
        _get_logger().info(
            f"{node.name}: checked {len(shapes)} generated shape samples; this does not certify untested inputs"
        )
        return True

    @staticmethod
    def _validate_inputs(generated, samples, target_shapes, contracts):
        if not isinstance(generated, dict) or generated.keys() != samples.keys():
            raise ValueError(
                f"validation_inputs must return exactly these tensor parameters: {list(samples)}"
            )
        bindings = {}
        for name, value in generated.items():
            sample = samples[name]
            if not isinstance(value, torch.Tensor) or isinstance(value, TracedData):
                raise TypeError(
                    f"validation_inputs returned an invalid tensor type for {name}"
                )
            if tuple(value.shape) != target_shapes[name] or value.dtype != sample.dtype:
                raise ValueError(
                    f"validation_inputs returned the wrong shape or dtype for {name}"
                )
            if value.device != sample.device:
                raise ValueError(
                    f"validation_inputs returned the wrong device for {name}"
                )
            validate_shape_contract(contracts[name], value.shape, bindings, name=name)

    def _bind_aliased_inputs(self, generated):
        by_port = {}
        for name, desc in self.descriptions.items():
            value = generated[name]
            if desc in by_port:
                shared = by_port[desc]
                if self.factory is not None:
                    if not torch.equal(value, shared):
                        raise ValueError(
                            f"validation_inputs returned conflicting values for aliased parameter {name}"
                        )
                generated[name] = shared
            else:
                by_port[desc] = value

    def _reference_outputs(self, generated, samples, seed):
        arguments = {name: _copy_input(value) for name, value in self.arguments.items()}
        copies = {}
        for name, value in generated.items():
            if id(value) not in copies:
                copies[id(value)] = _copy_input(value)
            arguments[name] = copies[id(value)]
        bound = self._bind(arguments)
        with _validation_random_state(samples, seed), ExitStack() as context:
            context.enter_context(torch.inference_mode(self.inference_mode))
            context.enter_context(torch.set_grad_enabled(self.grad_enabled))
            for device, (enabled, dtype) in self.autocast.items():
                context.enter_context(
                    torch.autocast(device, enabled=enabled, dtype=dtype)
                )
            try:
                expected = self.function(*bound.args, **bound.kwargs)
            except Exception as exc:
                raise ValueError(
                    "Validation could not complete: Original function could not evaluate "
                    "generated inputs; provide validation_inputs for domain-specific data. "
                    f"{exc}"
                ) from exc
        # Capture registers nested returns through the same flattening, so the
        # reference outputs follow the node's port order.
        expected = list(flatten_io_structure(expected, "").values())
        expected = prepare_tensors_for_export(
            [to_export_torch_tensor(value) for value in expected]
        )
        return expected
