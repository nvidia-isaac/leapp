# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text boundaries for ordinary LEAPP FX capture, export and validation."""

import copy

import numpy as np
import torch

from leapp.leapp_graph.datatypes import as_traced, promote_in_place
from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.utils.shape_contracts import (
    build_declared_shape_contract,
    get_shape_contract,
    inherit_shape_contract,
    validate_reentry_variable_axes,
    validate_shape_contract,
)
from . import FORMAT, StringCaptureError
from .capture import ACTIVE, Patches, TracedString, finish_capture, plain
from .utf8 import pack, unpack


def values(payload):
    if (
        type(payload) is not dict
        or not payload
        or any(type(k) is not str for k in payload)
    ):
        raise StringCaptureError(
            "String boundaries require a nonempty dictionary of named values"
        )
    return dict(payload)


def interface(payload):
    """Compare captures with their first owned sample, without another schema."""
    result = []
    for name, value in payload.items():
        if type(value) is str:
            result.append((name, str))
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            kind = torch.Tensor if isinstance(value, torch.Tensor) else np.ndarray
            result.append((name, kind, value.dtype, value.ndim))
        else:
            raise StringCaptureError("String inputs require scalar strings or tensors")
    return result


def snapshot(payload):
    """Own flat boundary samples without retaining mutable application buffers."""
    return {
        name: value.detach().cpu().clone()
        if isinstance(value, torch.Tensor)
        else value.copy()
        if isinstance(value, np.ndarray)
        else value
        for name, value in payload.items()
    }


class StringNode(TracedTensorNode):
    def __init__(self, name):
        super().__init__(name)
        self._has_data_dependent_shapes = True
        self._validation_exact_discrete = True
        self.utf8_layout = {}
        self.samples = []
        self.tokenizers = {}
        self._text_inputs, self._text_events, self._text_token = {}, [], None
        self._patches = self._active = None
        self.region_open = False
        self.setup_backend("onnx", {})

    def open(self, payload):
        """Start an observation. Only the first one traces and patches CPython.

        Later observations run eagerly, like tensor-node re-entry, and are
        checked against the exported model by final validation.
        """
        if self.region_open:
            raise StringCaptureError(f"Finish String region {self.name!r} first")
        raw = plain(values(payload))
        current = interface(raw)
        if not self.captured:
            self.begin(payload)
            self.region_open = True
            try:
                self._patches = Patches(self, self.tokenizers)
                self._patches.install()
                self._active = ACTIVE.set(self)
            except BaseException:
                self.close()
                raise
            return dict(self._text_inputs)
        if current != interface(self.samples[0]["inputs"]):
            raise StringCaptureError(
                f"String node {self.name!r} changed its input schema"
            )
        self.samples.append({"inputs": snapshot(raw)})
        self.region_open = True
        return {
            name: value if isinstance(value, str) else self.create_input(value, name)
            for name, value in raw.items()
        }

    def close(self):
        if self._active is not None:
            ACTIVE.reset(self._active)
            self._active = None
        if self._patches is not None:
            self._patches.restore()
            self._patches = None
        self.end()
        self.region_open = False

    def begin(self, payload):
        """Create traced inputs for the first capture."""
        raw = plain(payload)
        interface(raw)
        self.samples.append({"inputs": snapshot(raw)})
        self._text_token, self._text_error, self._text_events = object(), None, []
        self._text_inputs = {}
        # FX purity checks during output pruning need the owning module.
        self.tracer.graph.owning_module = self.tracer.root
        try:
            for name, value in payload.items():
                if isinstance(value, str):
                    proxy = self.tracer.create_proxy("placeholder", name, (), {})
                    value = TracedString(plain(value), self, proxy)
                else:
                    value = self.create_input(value, name)
                self._text_inputs[name] = value
        except BaseException:
            self.end()
            raise

    def end(self):
        self._text_inputs.clear()
        self._text_events.clear()
        self._text_token = None

    def create_input(self, data, name, semantics=None):
        if self.captured:
            self.validate_input_and_update_sources(name, name, data)
            return as_traced(data, name, self, None)
        return super().create_input(data, name, semantics)

    def _prepare_text(self, required, capacity, outputs):
        # Materialize only live modules. Unsupported discarded operations and
        # callbacks never need to load resources or execute their lowering.
        prepared = set()
        for operation in required:
            if factory := operation.meta.pop("leapp_text_factory", None):
                if operation.target not in prepared:
                    constructor, arguments = factory
                    self.tracer.root.add_module(
                        operation.target, constructor(*arguments)
                    )
                    prepared.add(operation.target)
        text = {
            name for name, value in self._text_inputs.items() if isinstance(value, str)
        }
        if text & set(outputs):
            raise StringCaptureError("String input and output names must be distinct")
        for name, value in self._text_inputs.items():
            if not isinstance(value, str):
                continue
            logical = value.proxy.node
            self.utf8_layout[name] = capacity
            # Converting even dead logical inputs allows the normal pruning pass
            # to own all input removal and its TensorDescription bookkeeping.
            with self.graph.inserting_before(logical):
                physical = self.create_input(
                    torch.from_numpy(pack(plain(value), capacity)), name
                ).proxy.node
                converted = self.graph.call_function(unpack, (physical, True))
            logical.replace_all_uses_with(converted)
            self.graph.erase_node(logical)
        first_operation = next(
            (n for n in self.graph.nodes if n.op != "placeholder"), None
        )
        placeholders = [n for n in self.graph.nodes if n.op == "placeholder"]
        if first_operation is not None:
            for placeholder in placeholders:
                first_operation.prepend(placeholder)
        order = {node.target: index for index, node in enumerate(placeholders)}
        self.inputs.sort(key=lambda port: order[port.name])
        self.input_formats = list(self.inputs)
        self.graph.owning_module = self.tracer.root

    def finish(
        self, payload, metadata, export_with=None, backend_params=None, dry_run=False
    ):
        """Publish one observation; compile on the first. Always closes the region."""
        try:
            if not self.region_open:
                raise StringCaptureError(
                    f"Mismatched String output boundary: {self.name}"
                )
            values(payload)
            export_with = export_with or "onnx"
            parameters = dict(backend_params or {})
            capacity = parameters.get("utf8_capacity", 4096)
            if export_with not in ("onnx", "onnx-dynamo", "pt2", "exported-program"):
                raise StringCaptureError("String nodes support ONNX and PT2 export")
            if (
                set(parameters) - {"utf8_capacity"}
                or type(capacity) is not int
                or capacity <= 0
            ):
                raise StringCaptureError(
                    "String backend_params require a positive utf8_capacity"
                )
            backend = None if dry_run else export_with
            return self._publish(payload, metadata or {}, backend, parameters)
        finally:
            self.close()

    def _publish(self, payload, metadata, backend, parameters):
        capacity = parameters.get("utf8_capacity", 4096)
        required = None if self.captured else finish_capture(self, payload)
        if self.captured and any(
            known != capacity for known in self.utf8_layout.values()
        ):
            raise StringCaptureError("UTF-8 capacity cannot change across captures")
        raw = plain(payload)
        if any(
            not isinstance(value, (torch.Tensor, np.ndarray)) for value in raw.values()
        ):
            raise StringCaptureError("String graph outputs must be tensors")
        self.samples[-1]["outputs"] = snapshot(raw)
        if self.captured:
            validate_reentry_variable_axes(metadata, self.outputs)
            if interface(raw) != interface(self.samples[0]["outputs"]):
                raise StringCaptureError(
                    f"String node {self.name!r} changed its output schema"
                )
            published, bindings = {}, {}
            for port in self.outputs:
                if port.semantics is not None:
                    validate_shape_contract(
                        get_shape_contract(port),
                        raw[port.name].shape,
                        bindings,
                        name=f"{self.name}/{port.name}",
                    )
                value = promote_in_place(raw[port.name], port.name, self, None)
                self.publish_output_port(value, port.name)
                published[port.name] = value
            return published
        self._prepare_text(required, capacity, payload)
        result = self.compile_trace(
            payload,
            backend=backend,
            backend_params=parameters,
            semantics_map=metadata,
        )
        for port in self.outputs:
            port.shape_contract = build_declared_shape_contract(port, self.name)
        ordered = (result,) if len(payload) == 1 else result
        return dict(zip(payload, ordered))

    def encode_inputs(self, payload):
        return tuple(
            torch.from_numpy(pack(payload[port.name], self.utf8_layout[port.name]))
            if port.name in self.utf8_layout
            else torch.as_tensor(payload[port.name])
            for port in self.inputs
        )

    def _build_validation_examples(self):
        # Text capture has always validated every observation, independently of
        # the bounded tensor cache. Preserve that policy using the common runner.
        return [
            (
                index,
                self.encode_inputs(sample["inputs"]),
                tuple(
                    torch.as_tensor(sample["outputs"][port.name])
                    for port in self.outputs
                ),
            )
            for index, sample in enumerate(self.samples)
        ]

    def compile_model(self):
        for port in self.inputs:
            port.shape_contract = build_declared_shape_contract(port, self.name)
            if port.has_source:
                source = next(
                    p for p in port.source_node.outputs if p.port == port.port
                )
                inherit_shape_contract(source, port)
        super().compile_model()

    @property
    def text_definition(self):
        """Name the retained packed UTF-8 ports that accept Python strings."""
        physical = {port.name for port in self.inputs}
        return {
            "format": FORMAT,
            "text_inputs": [name for name in self.utf8_layout if name in physical],
        }

    def get_description(self):
        """Publish the model's physical tensor interface using ordinary YAML fields."""
        if self.export_backend.compiled_model is None:
            raise StringCaptureError(
                "String YAML descriptions require model compilation"
            )
        program = getattr(self.export_backend, "exported_program", None)
        if program is not None:
            by_name = {node.name: node for node in program.graph.nodes}
            shapes = [
                by_name[name].meta["val"].shape
                for name in program.graph_signature.user_outputs
            ]
        else:
            model = self.export_backend._get_onnx_model(
                self.export_backend.compiled_model
            )
            shapes = [
                [
                    dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                    for dim in port.type.tensor_type.shape.dim
                ]
                for port in model.graph.output
            ]
        outputs = []
        for port, shape in zip(self.outputs, shapes):
            description = copy.copy(port)
            contract = copy.deepcopy(get_shape_contract(port))
            # Unannotated tokenizer widths are intrinsically data dependent.
            # Express that with existing variable-axis metadata rather than a
            # string-specific schema or a misleading fixed capture width.
            if port.semantics is None:
                for index, dim in enumerate(shape):
                    if isinstance(dim, int):
                        continue
                    dimension = str(dim) or f"{port.name}_{index}"
                    symbol = f"__leapp_text_{self.name.encode().hex()}_{dimension.encode().hex()}"
                    contract["shape"][index] = symbol
                    contract.setdefault("dimension_ranges", {})[symbol] = {
                        "min": 0,
                        "max": 2**31 - 1,
                    }
            description.shape_contract = contract
            outputs.append(description.dict())
        result = super().get_description()
        if program is None:
            descriptions = {port.name: port.dict() for port in self.inputs}
            result["inputs"] = [descriptions[port.name] for port in model.graph.input]
        result["outputs"] = outputs
        return result
