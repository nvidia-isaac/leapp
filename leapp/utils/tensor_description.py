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
from dataclasses import dataclass, field, fields
from typing import Optional, Any, Dict, Tuple, List

import torch
import yaml

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
    TRACABLE_BASE_TYPES,
)
from leapp.utils.utils import safe_deepcopy
from leapp.utils.enums import InputKindEnum, OutputKindEnum


class CompactYamlList(list):
    """Custom list class for tensor shapes that enables compact YAML formatting
    This class will behave exactly like a list. The only difference is that it will be
    formatted in a more compact way in the yaml file.

    """
    pass


class CompactYamlDict(dict):
    """Custom dict class for tensor shapes that enables compact YAML formatting
    This class will behave exactly like a dict. The only difference is that it will be
    formatted in a more compact way in the yaml file.
    """
    pass


# Register custom YAML representers so these types are always serialized in flow style
yaml.add_representer(
    CompactYamlList,
    lambda dumper, data: dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True))
yaml.add_representer(
    CompactYamlDict,
    lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True))


TEMPORAL_AXIS_SENTINEL = "__temporal_axis__"


@dataclass(frozen=True)
class TemporalAxis:
    """Marks an element_names axis as temporal with a fixed period in ms."""

    period_ms: float

    def __post_init__(self):
        if self.period_ms <= 0:
            _get_logger().fatal(
                "TemporalAxis period_ms must be positive",
                error_type=ValueError,
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


@dataclass
class TensorSemantics:
    """User-facing semantic metadata for a tensor.

    Add new semantic fields here — they automatically propagate to
    TensorDescription, YAML serialization, and the unwrap/apply flow.

    Convention:
        - Internal fields (ref, name) are listed in _INTERNAL_FIELDS and excluded from serialization.
        - Public fields (kind, element_names, ...) are semantic data that gets serialized to YAML.
    """

    # Fields excluded from serialization (internal use only)
    _INTERNAL_FIELDS = frozenset({'ref', 'name'})

    name: str = None
    ref: Any = None

    # Semantic fields
    kind: Optional[InputKindEnum | OutputKindEnum | str] = None
    element_names: Optional[List] = None # deprecated
    temporal_period_ms: Optional[float] = field(default=None, init=False)
    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        '''error checking, auto conditioning'''
        existing_period_ms = self.temporal_period_ms
        self.temporal_period_ms = None
        if not is_tracable_tensor_type(self.ref): # this checks for both base types and traced types
            _get_logger().fatal(
                f"TensorSemantics 'ref' must be a traceable tensor type "
                f"accepted types are: {TRACABLE_BASE_TYPES}"
                f"got {type(self.ref).__name__}",
                error_type=TypeError,
            )
        if self.element_names is not None:
            self.element_names, detected_period_ms = self._normalize_element_names(
                self.element_names, allow_temporal_sentinel=existing_period_ms is not None)
            self.temporal_period_ms = detected_period_ms or existing_period_ms


    def to_dict(self) -> Dict[str, Any]:
        """Return all non-None semantic fields, excluding internal fields.

        Extra fields are flattened directly into the result dict rather than
        nested under an 'extra' key.
        """
        result = {}
        for f in fields(self):
            if f.name in self._INTERNAL_FIELDS or f.name == 'extra':
                continue
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        if self.extra:
            result.update(self.extra)
        return result

    def update(self, values: Dict[str, Any]):
        """Set semantic fields from a dict.

        Known dataclass fields are set directly. Unknown keys are stored
        in the ``extra`` dict so they still appear in the serialized YAML.
        """
        valid_fields = {f.name for f in fields(self) if f.init} - self._INTERNAL_FIELDS
        for key, value in values.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                if self.extra is None:
                    self.extra = {}
                self.extra[key] = value
        self.__post_init__()

    @staticmethod
    def _normalize_element_names(element_names, allow_temporal_sentinel=False):
        """Normalize element_names and extract temporal axis metadata."""
        if isinstance(element_names, str):
            if element_names == TEMPORAL_AXIS_SENTINEL:
                if allow_temporal_sentinel:
                    return CompactYamlList([element_names]), None
                _get_logger().fatal(
                    f"{TEMPORAL_AXIS_SENTINEL!r} is reserved for TemporalAxis",
                    error_type=ValueError,
                )
            return CompactYamlList([CompactYamlList([element_names])]), None

        if not isinstance(element_names, list):
            return element_names, None

        temporal_period_ms = None
        normalized = CompactYamlList()
        has_axis_descriptors = False
        has_temporal_axis = False

        for item in element_names:
            if isinstance(item, TemporalAxis):
                if has_temporal_axis:
                    _get_logger().fatal(
                        "element_names can contain at most one TemporalAxis",
                        error_type=ValueError,
                    )
                has_axis_descriptors = True
                has_temporal_axis = True
                temporal_period_ms = item.period_ms
                normalized.append(TEMPORAL_AXIS_SENTINEL)
            elif isinstance(item, str):
                if item == TEMPORAL_AXIS_SENTINEL:
                    if not allow_temporal_sentinel:
                        _get_logger().fatal(
                            f"{TEMPORAL_AXIS_SENTINEL!r} is reserved for TemporalAxis",
                            error_type=ValueError,
                        )
                    has_axis_descriptors = True
                    normalized.append(TEMPORAL_AXIS_SENTINEL)
                else:
                    normalized.append(item)
            elif isinstance(item, list):
                if any(isinstance(child, TemporalAxis) for child in item):
                    _get_logger().fatal(
                        "TemporalAxis must be an axis item, not nested in a list",
                        error_type=ValueError,
                    )
                if any(child == TEMPORAL_AXIS_SENTINEL for child in item):
                    _get_logger().fatal(
                        f"{TEMPORAL_AXIS_SENTINEL!r} must be a bare axis item, not nested in a list",
                        error_type=ValueError,
                    )
                has_axis_descriptors = True
                normalized.append(CompactYamlList(item))
            elif item is None:
                has_axis_descriptors = True
                normalized.append(None)
            else:
                _get_logger().warning(
                    "element_names has mixed types, expected names or axis descriptors")
                return element_names, temporal_period_ms

        if has_axis_descriptors:
            return normalized, temporal_period_ms

        return CompactYamlList([CompactYamlList(normalized)]), None


class TensorDescription:
    """Describes a tensor input/output in the computational graph.

    Composes TensorSemantics for semantic metadata (kind, element_names, etc.).
    Access semantic fields directly via properties (td.kind, td.element_names)
    or in bulk via get_semantics()/set_semantics().
    """

    def __init__(self, name: str, value: Any,
                 semantics: Optional[TensorSemantics] = None):
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
            "type": self.type
        }
        if self.value is not None and not ignore_value:
            result["value"] = self.value
        # Merge in all non-None semantic fields
        for key, value in self.get_semantics().items():
            if hasattr(value, 'value'):  # Enum → serialize to its .value string
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

    def get_format_string(self, data: Any) -> Tuple[bool, str]:

        if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes, torch.Tensor)):
            # Warn if not a native list
            if not isinstance(data, list):
                type_name = type(data).__name__
                _get_logger().debug(
                    f"Parameter '{self.name}' has list-like type '{type_name}' which will be "
                    f"treated as 'list' in the generated function signature. This may cause "
                    f"issues if the exact type is required.")

            if len(data) == 0:
                return True, "list"  # Empty list, can't determine child type

            item_types_valid = len(set(type(item) for item in data)) == 1
            if not item_types_valid:
                return False, "INCONSISTENT LIST ITEM TYPES"
            else:
                success, child = self.get_format_string(data[0])
                if not success:
                    return False, "LIST CHILD FORMAT STRING NOT VALID"
                else:
                    return True, f"List[{child}]"

        elif isinstance(data, collections.abc.Mapping):
            # Warn if not a native dict
            if not isinstance(data, dict):
                type_name = type(data).__name__
                _get_logger().debug(
                    f"Parameter '{self.name}' has dict-like type '{type_name}' which will be "
                    f"treated as 'dict' in the generated function signature. This may cause "
                    f"issues if the exact type is required.")

            if len(data) == 0:
                return True, "dict"  # Empty dict, can't determine child type

            key_types = set(type(key) for key in data.keys())
            if len(key_types) > 1 or key_types != {str}:
                return False, "INCONSISTENT DICT KEY TYPES"
            value_types = set(type(value) for value in data.values())
            if len(value_types) > 1:
                return False, "INCONSISTENT DICT VALUE TYPES"

            success, child = self.get_format_string(
                list(data.values())[0])
            if not success:
                return False, "DICT CHILD FORMAT STRING NOT VALID"
            else:
                return True, f"Dict[str, {child}]"

        elif isinstance(data, TensorDescription):
            return True, type(data.value).__name__

        else:
            return False, "UNEXPECTED TYPE"

    @property
    def name_str(self) -> str:
        """Return just the name as a string. 
        will do some modifications to me the vairable compliant with python"""
        return self.name.replace(".", "_")

    @property
    def name_raw(self) -> str:
        return self.name

    @property
    def dtype(self) -> str:
        """Return the dtype as a string."""
        success, dtype = self.get_format_string(self.formatting)
        if not success:
            return None
        return dtype

    @property
    def valid(self):
        a = self.get_format_string(self.formatting)
        if not a:
            return False
        return True

    @property
    # TODO: use this to replace reference to the tensor descriptions.
    def tensor_expr_in_order(self) -> list[str]:
        """
        Generate a list of tensor expressions in the order of the nested structure.
        """
        def _generate_expr(format_item) -> list[str]:
            if isinstance(format_item, ParameterFormat):
                return _generate_expr(format_item.formatting)
            elif isinstance(format_item, TensorDescription):
                return [format_item.name_str]
            elif isinstance(format_item, list):
                return [_generate_expr(item) for item in format_item]
            elif isinstance(format_item, dict):
                return [_generate_expr(value) for value in format_item.values()]
            else:
                return []
        return _generate_expr(self.formatting)

    @property
    def packed_tensor_expr(self) -> str:
        """
        Generate a packing assignment expression for this parameter.

        Converts flat tensor inputs into the expected nested structure.
        Handles nested structures (lists, dicts).

        Returns:
            Assignment string like "inputA = [inputA_0, inputA_1]". 
            Empty string if no packing needed.
        """
        def _generate_expr(format_item) -> str:
            if isinstance(format_item, ParameterFormat):
                return _generate_expr(format_item.formatting)
            elif isinstance(format_item, TensorDescription):
                return format_item.name_str
            elif isinstance(format_item, list):
                elements = [_generate_expr(item) for item in format_item]
                return "[" + ", ".join(elements) + "]"
            elif isinstance(format_item, dict):
                items = [
                    f'"{k}": {_generate_expr(v)}' for k, v in format_item.items()]
                return "{" + ", ".join(items) + "}"
            else:
                return "None"

        reconstruction = _generate_expr(self.formatting)

        # Skip trivial assignments where name == expression
        if self.name_str == reconstruction:
            return ""

        return f"{self.name_str} = {reconstruction}"

    @property
    def unpacked_tensor_expr(self) -> str:
        """
        Generate unpacking assignment expressions that extract flat tensors from
        a nested structure, using accessor paths.

        This is the inverse of packed_tensor_expr.

        Handles:
        - List unpacking (e.g., "inputA_0 = inputA[0]", "inputA_1 = inputA[1]")
        - Dict unpacking (e.g., 'state_pose = state["pose"]')
        - Nested structures (e.g., 'nested_0_0 = nested[0][0]')

        Returns:
            Newline-separated assignment strings. Empty string if no unpacking needed.
        """
        def _generate_unpacking(format_item, accessor_path: str, assignments: list):
            if isinstance(format_item, ParameterFormat):
                _generate_unpacking(format_item.formatting,
                                    accessor_path, assignments)
            elif isinstance(format_item, TensorDescription):
                tensor_name = format_item.name_str
                if tensor_name != accessor_path:
                    assignments.append(f"{tensor_name} = {accessor_path}")
            elif isinstance(format_item, list):
                for idx, item in enumerate(format_item):
                    child_path = f"{accessor_path}[{idx}]"
                    _generate_unpacking(item, child_path, assignments)
            elif isinstance(format_item, dict):
                for key, value in format_item.items():
                    child_path = f'{accessor_path}["{key}"]'
                    _generate_unpacking(value, child_path, assignments)

        assignments = []
        _generate_unpacking(self.formatting, self.name_str, assignments)
        return "\n".join(assignments)

def validate_connection_compatibility(source_name, source_shape, source_dtype,
                                      target_name, target_shape, target_dtype):
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
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes, torch.Tensor)):
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
        bad_types = {type(item).__name__ for item in data if not isinstance(item, TensorSemantics)}
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
    if isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes, torch.Tensor)):
        if not isinstance(data, list):
            type_name = type(data).__name__
            _get_logger().warning(
                f"Input/Output '{name_str}' has list-like type '{type_name}' which will be "
                f"treated as 'list'. Ensure the runtime can handle this substitution.")

        io_format = []
        for idx, item in enumerate(data):
            child_name = f"{name_str}_{idx}" if name_str else str(idx)
            child_description, child_format = describe_io_helper(
                item, child_name)
            if child_format is not None:
                io_format.append(child_format)
                data_description.extend(child_description)
        return data_description, io_format
    elif isinstance(data, collections.abc.Mapping):
        if not isinstance(data, dict):
            type_name = type(data).__name__
            _get_logger().warning(
                f"Input/Output '{name_str}' has dict-like type '{type_name}' which will be "
                f"treated as 'dict'. Ensure the runtime can handle this substitution.")

        io_format = {}
        for k, v in data.items():
            child_name = f"{name_str}_{k}" if name_str else k
            child_description, child_format = describe_io_helper(
                v, child_name)
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
    data_description, io_format = describe_io_helper(
        data, name)
    parameter_description = ParameterFormat(
        name=raw_name, formatting=io_format)
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
        return {key: _resolve_tensor_descriptions(value, extractor) for key, value in io_format.items()}
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


def resolve_tensor_descriptions_to_values(io_format):
    """
    Recursively resolve TensorDescription objects to their values.

    Args:
        io_format: The object to recursively process (can be ParameterFormat, TensorDescription, list, dict, etc.)

    Returns:
        The processed object with TensorDescription objects replaced by their values.
        If a ParameterFormat is passed, returns the resolved formatting from within it.
    """
    return _resolve_tensor_descriptions(io_format, lambda td: td.value)
