#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import collections.abc
from dataclasses import dataclass, fields
from typing import Optional, Any, Dict, Tuple, List

import torch

# Dtype mapping lives in a dependency-free module; re-exported here (explicit
# ``as`` form) for the existing import sites
# (``from leapp.utils.tensor_description import ...``).
from leapp.utils.dtype import (
    map_to_torch_dtype as map_to_torch_dtype,
    warp_dtype_to_torch_name as warp_dtype_to_torch_name,
    dtype_to_name as dtype_to_name,
    value_to_name_and_shape as value_to_name_and_shape,
    DtypeCodec as DtypeCodec,
    register_dtype_codec as register_dtype_codec,
)

from leapp.utils.logging import _get_logger
from leapp.leapp_graph.datatypes import (
    TracedData,
    is_tracable_tensor_type,
    to_export_torch_tensor,
)
from leapp.utils.utils import safe_deepcopy

# Preserve existing imports while semantics owns the annotation classes.
from leapp.utils.semantics import (
    Axis as Axis,
    TemporalAxis as TemporalAxis,
    TensorSemantics as TensorSemantics,
    TEMPORAL_AXIS_SENTINEL as TEMPORAL_AXIS_SENTINEL,
)
from leapp.utils.yaml_utils import (
    CompactYamlList as CompactYamlList,
    CompactYamlDict as CompactYamlDict,
)


@dataclass
class GraphConfigs:
    """User-facing graph-level metadata for YAML serialization."""

    frequency: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.frequency is not None and self.frequency <= 0:
            _get_logger().fatal(
                "GraphConfigs frequency must be positive when provided",
                error_type=ValueError,
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return non-None graph config fields with extra flattened."""
        result = {}
        for f in fields(self):
            if f.name == "extra":
                continue
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        if self.extra:
            result.update(self.extra)
        return result


class TensorDescription:
    """Describes a tensor input/output in the computational graph.

    Composes TensorSemantics for semantic metadata (kind, element_names, etc.).
    Access semantic fields directly via properties (td.kind, td.element_names)
    or in bulk via get_semantics()/set_semantics().
    """

    def __init__(
        self, name: str, value: Any, semantics: Optional[TensorSemantics] = None
    ):
        # Capture where the value came from before unwrapping loses that state.
        source_node = None
        port = None
        if isinstance(value, TracedData):
            source_node = value.context_obj
            port = value.output_port

        # unwrap TracedData to the underlying tensor
        if isinstance(value, TracedData):
            value = value.tensor

        dtype, shape = TensorDescription.get_shape_and_dtype(value)

        self._aliases = [name]
        # Capture-only Python identifier, assigned once when registered as an
        # FX input. Public names and aliases remain unchanged.
        self.fx_name = None
        # Capture-only layout facts must precede safe_deepcopy's dense snapshot.
        tensor = to_export_torch_tensor(value)
        self.strides = None
        self.is_dense = True
        if tensor.layout == torch.strided and not tensor.is_nested:
            self.strides = tuple(tensor.stride())
            if tensor.numel() != 0:
                expected_stride = 1
                for stride, size in sorted(
                    (stride, size)
                    for stride, size in zip(self.strides, shape)
                    if size > 1
                ):
                    if stride != expected_stride:
                        self.is_dense = False
                        break
                    expected_stride *= size
        self.value = safe_deepcopy(value)
        self.dtype = dtype
        self.shape = CompactYamlList(shape)
        self.type = "tensor"
        self.source_node = source_node
        # The port on ``source_node`` that identifies this data. An input takes
        # it from the value it received; an output is its own source, so its
        # node stamps both at registration. Unlike ``name``, which export
        # backends may rewrite, a port is set once and never changes.
        self.port = port
        self.cached_values = []
        self.init_semantics(semantics)

    @property
    def has_source(self) -> bool:
        """Whether this description resolves to a specific node output."""
        return self.source_node is not None and self.port is not None

    # --- Semantic field access ---

    def init_semantics(self, semantics: Optional[TensorSemantics]):
        self.semantics = semantics
        if semantics is not None and semantics.name is not None:
            self.change_name(semantics.name)

    def get_semantics(self) -> Dict[str, Any]:
        """Return all non-None semantic fields as a dict."""
        if self.semantics is None:
            return {}
        return self.semantics.to_dict()

    def update_semantics(self, values: Dict[str, Any]):
        """Set semantic fields from a dict."""
        if self.semantics is not None:
            semantics = self.semantics
        else:
            semantics = TensorSemantics(name=self.name, ref=self.value)
        semantics.update(values)
        self.init_semantics(semantics)

    # --- Core methods ---

    @staticmethod
    def get_shape_and_dtype(value) -> Tuple[str, tuple]:
        """Extract dtype string and shape from a registered backend value."""
        dtype, shape = value_to_name_and_shape(value)

        return dtype, shape

    def dict(self, ignore_value=True) -> Dict[str, Any]:
        """Convert to a dictionary for YAML serialization."""
        result = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": self.shape,
            "type": self.type,
        }
        if self.value is not None and not ignore_value:
            result["value"] = self.value
        # Merge in all non-None semantic fields
        for key, value in self.get_semantics().items():
            if hasattr(value, "value"):  # Enum → serialize to its .value string
                result[key] = value.value
            else:
                result[key] = value
        return result

    def change_name(self, new_name: str):
        """Change the name of the tensor description."""
        self.name = new_name
        if self.semantics is not None:
            self.semantics.name = new_name

    @property
    def name(self) -> str:
        """Return the canonical name."""
        return self._aliases[0]

    @name.setter
    def name(self, new_name: str):
        """Rename the canonical name while preserving aliases."""
        self._aliases[0] = new_name

    @property
    def aliases(self) -> List[str]:
        """Return every declared name for this tensor."""
        return self._aliases

    @property
    def name_str(self) -> str:
        """Return just the name as a string."""
        return self.name


@dataclass
class ParameterFormat:
    name: str
    formatting: Any


def validate_connection_compatibility(
    source_name, source_shape, source_dtype, target_name, target_shape, target_dtype
):
    """Validate shape and dtype compatibility for a single pipeline edge."""
    if list(source_shape) != list(target_shape):
        _get_logger().fatal(
            f"Shape mismatch in pipeline connection: "
            f"{source_name} {tuple(source_shape)} -> "
            f"{target_name} {tuple(target_shape)}",
            error_type=ValueError,
        )

    if target_dtype != source_dtype:
        _get_logger().fatal(
            f"Dtype mismatch in pipeline connection: "
            f"{source_name} {source_dtype} -> "
            f"{target_name} {target_dtype}",
            error_type=ValueError,
        )


def verify_data_exact_match(source_data, target_data):
    # Check if both are the same type (allow dict-like and list-like substitutions)
    source_is_list_like = (
        isinstance(source_data, collections.abc.Sequence)
        and not isinstance(source_data, (str, bytes, torch.Tensor))
        and not is_tracable_tensor_type(source_data)
    )
    target_is_list_like = (
        isinstance(target_data, collections.abc.Sequence)
        and not isinstance(target_data, (str, bytes, torch.Tensor))
        and not is_tracable_tensor_type(target_data)
    )
    source_is_dict_like = isinstance(source_data, collections.abc.Mapping)
    target_is_dict_like = isinstance(target_data, collections.abc.Mapping)

    # If one is list-like and the other is not, they don't match
    if source_is_list_like != target_is_list_like:
        return False
    # If one is dict-like and the other is not, they don't match
    if source_is_dict_like != target_is_dict_like:
        return False

    if is_tracable_tensor_type(source_data):
        # conversion to torch tensor to utilize torch equality
        if not is_tracable_tensor_type(target_data):
            return False
        source_data = to_export_torch_tensor(source_data)
        target_data = to_export_torch_tensor(target_data)

        # identity comparison
        if source_data.shape != target_data.shape:
            return False
        if source_data.dtype != target_data.dtype:
            return False
        if source_data.device != target_data.device:
            return False
        if not torch.equal(source_data, target_data):
            return False

    elif source_is_list_like:
        if len(source_data) != len(target_data):
            return False
        for source_item, target_item in zip(source_data, target_data):
            if not verify_data_exact_match(source_item, target_item):
                return False
    elif source_is_dict_like:
        if set(source_data.keys()) != set(target_data.keys()):
            return False
        for key, source_item in source_data.items():
            if not verify_data_exact_match(source_item, target_data[key]):
                return False
    else:
        if source_data != target_data:
            return False

    return True


def flatten_io_structure(data, name_str):
    flat_data = {}
    if isinstance(data, collections.abc.Sequence) and not isinstance(
        data, (str, bytes, torch.Tensor)
    ):
        for idx, item in enumerate(data):
            child_name = f"{name_str}_{idx}" if name_str else str(idx)
            flat_data.update(flatten_io_structure(item, child_name))
    elif isinstance(data, collections.abc.Mapping):
        for key, value in data.items():
            child_name = f"{name_str}_{key}" if name_str else key
            flat_data.update(flatten_io_structure(value, child_name))
    elif is_tracable_tensor_type(data):
        flat_data[name_str] = data

    return flat_data


def unwrap_tensor_semantics(data):
    """Unwrap TensorSemantics object(s) into a dict mapping names to tensors.

    Accepts a single TensorSemantics or a list of TensorSemantics.
    All items must be TensorSemantics — mixing with raw tensors is not allowed.

    Uses each item's .name as the dict key and .ref as the tensor value.
    Collects the TensorSemantics objects for later attachment to TensorDescriptions.

    Args:
        data: A single TensorSemantics or a list of TensorSemantics.

    Returns:
        tuple: (dict[str, tensor], dict[str, TensorSemantics])
    """
    if isinstance(data, TensorSemantics):
        data = [data]

    if not isinstance(data, (list, tuple)):
        _get_logger().fatal(
            f"unwrap_tensor_semantics expects a TensorSemantics or list of TensorSemantics, "
            f"got {type(data).__name__}",
            error_type=TypeError,
        )

    if not all(isinstance(item, TensorSemantics) for item in data):
        bad_types = {
            type(item).__name__
            for item in data
            if not isinstance(item, TensorSemantics)
        }
        _get_logger().fatal(
            f"All items must be TensorSemantics when using semantic inputs. "
            f"Found non-TensorSemantics types: {bad_types}",
            error_type=TypeError,
        )

    # Check for duplicate names
    names = [sem.name for sem in data]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        _get_logger().fatal(
            f"Duplicate TensorSemantics names: {duplicates}. "
            f"Each TensorSemantics must have a unique name.",
            error_type=ValueError,
        )

    tensors = {}
    semantics_map = {}
    for sem in data:
        tensors[sem.name] = sem.ref
        semantics_map[sem.name] = sem

    return tensors, semantics_map


def describe_io_helper(data, name_str):
    data_description = []
    io_format = {}
    if isinstance(data, collections.abc.Sequence) and not isinstance(
        data, (str, bytes, torch.Tensor)
    ):
        if not isinstance(data, list):
            type_name = type(data).__name__
            _get_logger().warning(
                f"Input/Output '{name_str}' has list-like type '{type_name}' which will be "
                f"treated as 'list'. Ensure the runtime can handle this substitution."
            )

        io_format = []
        for idx, item in enumerate(data):
            child_name = f"{name_str}_{idx}" if name_str else str(idx)
            child_description, child_format = describe_io_helper(item, child_name)
            if child_format is not None:
                io_format.append(child_format)
                data_description.extend(child_description)
        return data_description, io_format
    elif isinstance(data, collections.abc.Mapping):
        if not isinstance(data, dict):
            type_name = type(data).__name__
            _get_logger().warning(
                f"Input/Output '{name_str}' has dict-like type '{type_name}' which will be "
                f"treated as 'dict'. Ensure the runtime can handle this substitution."
            )

        io_format = {}
        for k, v in data.items():
            child_name = f"{name_str}_{k}" if name_str else k
            child_description, child_format = describe_io_helper(v, child_name)
            if child_format is not None:
                io_format[k] = child_format
                data_description.extend(child_description)
    # elif isinstance(data, torch.Tensor):
    elif is_tracable_tensor_type(data):
        tensor_desc = TensorDescription(name_str, data)

        # Return as a list containing the dataclass (for now, keep compatibility)
        data_description = [tensor_desc]
        io_format = tensor_desc

    else:
        return [], None

    return data_description, io_format


def describe_io(name, raw_name, data):
    data_description, io_format = describe_io_helper(data, name)
    parameter_description = ParameterFormat(name=raw_name, formatting=io_format)
    return data_description, parameter_description


def _resolve_tensor_descriptions(io_format, extractor):
    """
    Recursively resolve TensorDescription objects using a custom extractor function.

    This function traverses through nested structures (lists, dicts, etc.) and
    applies the extractor function to any TensorDescription objects found.

    Args:
        io_format: The object to recursively process (can be TensorDescription, ParameterFormat, list, dict, or any other type)
        extractor: A function that takes a TensorDescription and returns the desired value

    Returns:
        The processed object with TensorDescription objects replaced by extracted values
    """
    if isinstance(io_format, ParameterFormat):
        # For ParameterFormat, apply resolution to the formatting attribute
        return _resolve_tensor_descriptions(io_format.formatting, extractor)
    elif isinstance(io_format, TensorDescription):
        return extractor(io_format)
    elif isinstance(io_format, list):
        return [_resolve_tensor_descriptions(item, extractor) for item in io_format]
    elif isinstance(io_format, dict):
        return {
            key: _resolve_tensor_descriptions(value, extractor)
            for key, value in io_format.items()
        }
    else:
        return io_format


def resolve_tensor_descriptions_to_names(io_format):
    """
    Recursively resolve TensorDescription objects to their string names.

    Args:
        io_format: The object to recursively process (can be ParameterFormat, TensorDescription, list, dict, etc.)

    Returns:
        The processed object with TensorDescription objects replaced by their string names.
        If a ParameterFormat is passed, returns the resolved formatting from within it.
    """
    return _resolve_tensor_descriptions(io_format, lambda td: td.name_str)
