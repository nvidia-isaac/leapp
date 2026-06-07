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

import sys
import functools
import inspect
import os
import torch

from leapp.utils.logging import _get_logger
from leapp.leapp_graph.leapp_node import LeappNode
from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.function_decorator_node import FunctionDecoratorNode
from leapp.utils.tracing_lock import TracingLock
from leapp.leapp_graph.datatypes import (
    TracedTensor,
    is_traced_type,
    is_tracable_tensor_type,
)
from leapp.leapp_graph.datatypes.global_patching import warn_if_script_functions_in_scope
from leapp.utils.tensor_description import TensorSemantics
from leapp.utils.tensor_description import (verify_data_exact_match,
                                             flatten_io_structure,
                                             unwrap_tensor_semantics)
from leapp.utils.caller_identity import (get_caller_stack_identity,
                                         caller_identity_has_same_anchor,
                                         format_caller_identity)
from leapp.utils.utils import (get_relative_path,
                               mirror_all_tensor_tags,
                               extract_return_names,
                               frame_to_namespace)


class ExportManager:
    _instance = None
    _initialized = False  # True after singleton __init__ completes
    # True between start() and stop() - enables graph interpretation
    _interpret_graph = False

    #########################################################
    # initialization
    #########################################################

    @property
    def config_path(self):
        return os.path.join(self.SAVE_PATH, f"{self.GRAPH_NAME}.yaml")

    def __new__(cls, *args, **kwargs):
        """Singleton implementation - only one instance allowed."""
        if cls._instance is None:
            cls._instance = super(ExportManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ExportManager._initialized:
            # graph settings
            self.GRAPH_NAME = "my_graph"
            self.SAVE_PATH = None
            self.dry_run = False
            self.non_traced = set()
            self._patches_applied = False

            # tracetime variables
            self.nodes = {}
            self._next_completed_node_index = 0
            self._region_segmenters = {}

            ExportManager._initialized = True



    #########################################################
    # state accessors (for runtime API)
    #########################################################
    def set_graph_name(self, name: str):
        self.GRAPH_NAME = name

    def get_graph_name(self):
        return self.GRAPH_NAME

    def set_save_path(self, save_path: str):
        self.SAVE_PATH = save_path

    def get_save_path(self):
        return self.SAVE_PATH

    def ensure_save_path_exists(self):
        if self.SAVE_PATH is not None and not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)

    def configure_logger(self, verbose=False):
        _get_logger().configure(self.SAVE_PATH, verbose=verbose)

    def set_dry_run_and_non_traced(self, dry_run: bool, non_traced):
        self.set_dry_run(dry_run)
        if isinstance(non_traced, str):
            non_traced = [non_traced]
        self.non_traced = set(non_traced)

    def set_dry_run(self, dry_run: bool):
        self.dry_run = dry_run

    def is_dry_run(self, name: str = None):
        if name is None:
            return self.dry_run
        else:
            return self.dry_run or name in self.non_traced

    def set_max_cached_io(self, max_cached_io: int):
        self._max_cached_io = max_cached_io - 1

    def reset_nodes(self):
        self.nodes = {}
        self._next_completed_node_index = 0
        self._region_segmenters = {}

    def get_nodes(self):
        return self.nodes

    @classmethod
    def set_interpret_graph(cls, is_enabled: bool):
        cls._interpret_graph = is_enabled

    @classmethod
    def is_interpret_graph_enabled(cls):
        return cls._interpret_graph

    def set_patches_applied(self, is_applied: bool):
        self._patches_applied = is_applied

    def is_numpy_patches_applied(self):
        return self._patches_applied

    def reset_tracing_lock(self):
        TracingLock().reset()

    def restore_pending_buffer_trackers(self):
        for node in self.nodes.values():
            if hasattr(node, '_buffer_tracker') and node._buffer_tracker is not None:
                node._buffer_tracker.restore()

    def set_detected_graph(self, models: dict, pipeline: dict):
        self.detected_nodes = models['models']
        self.detected_pipeline = pipeline['pipeline']

    #########################################################
    # node setup
    #########################################################
    def _assign_completion_index(self, node: LeappNode):
        """Assign execution order when a node completes its initial trace."""
        if node.node_index != LeappNode.UNSET_NODE_INDEX:
            return
        node.node_index = self._next_completed_node_index
        self._next_completed_node_index += 1

    def _rename_node(self, old, new):
        if new in self.nodes:
            raise Exception(f"cannot rename node '{old}' to existing '{new}'")
        node = self.nodes.pop(old)
        node.name = new
        self.nodes[new] = node

    def _assign_index(self, node):
        self._assign_completion_index(node)

    def _default_torch_backend(self):
        return "onnx-torchscript"

    def _resolve_open_node_name(self, node_name):
        """Map a region base-name to its currently-open segment node (identity for
        torch-only / non-segmented regions)."""
        seg = self._region_segmenters.get(node_name)
        return seg.open_node.name if seg is not None else node_name

    def validate_nodes_ready_for_compile(self):
        incomplete_nodes = [
            name for name, node in self.nodes.items()
            if node.node_index == LeappNode.UNSET_NODE_INDEX
        ]
        if incomplete_nodes:
            incomplete_nodes.sort()
            formatted = ", ".join(incomplete_nodes)
            raise Exception(
                "The following nodes were created but never completed: "
                f"{formatted}. Did you forget to call output_tensors() "
                "or finish the annotated function?"
            )

    def _setup_new_node(self, name, node_class: LeappNode, **kwargs):
        if name in self.nodes:
            raise Exception(
                f"Error: node '{name}' already exists. "
                f"Cannot create a new node with the same name.")

        if self.is_dry_run(name):
            kwargs['export_with'] = None
            kwargs.setdefault('backend_params', {})

        if node_class is FunctionDecoratorNode:
            # kept for backward compatibility we still suppport old style initialization
            # TODO: this needs to be removed in the future
            node = node_class(name,
                          backend=kwargs.get("export_with", None),
                          backend_params=kwargs.get("backend_params", None),
                          inputs=kwargs.get("inputs", None),
                          outputs=kwargs.get("outputs", None),
                          environment_constants=kwargs.get(
                              "environment_constants", None),
                          register_buffers=kwargs.get("register_buffers", None),
                          dry_run=self.is_dry_run(name))
        else:
            node = node_class(name, dry_run=self.is_dry_run(name), **kwargs)

        node._max_cached_io = self._max_cached_io
        self.nodes[name] = node
        return node

    @staticmethod
    def _passthrough_dict_values(tensors: dict):
        """Unwrap a single-entry dict to its value, or a multi-entry dict to a tuple."""
        values = list(tensors.values())
        return values[0] if len(values) == 1 else tuple(values)

    @staticmethod
    def _normalize_named_tensor_payload(api_name: str, node_name: str, tensors):
        metadata = {}
        if isinstance(tensors, (TensorSemantics, list, tuple)) and (
            isinstance(tensors, TensorSemantics) or
            any(isinstance(t, TensorSemantics) for t in tensors)
        ):
            tensors, metadata = unwrap_tensor_semantics(tensors)

        if is_tracable_tensor_type(tensors):
            raise TypeError(
                f"{api_name}() for node '{node_name}' does not accept a bare tensor. "
                "Pass a dict of named tensors or a TensorSemantics/list of TensorSemantics."
            )

        if isinstance(tensors, dict):
            return tensors, metadata

        raise TypeError(
            f"{api_name}() for node '{node_name}' expects either a dict of named tensors "
            f"or a TensorSemantics/list of TensorSemantics. Received {type(tensors).__name__}."
        )

    #########################################################
    # annotation APIs
    #########################################################
    def input_tensors(self, node_name: str, tensors):
        if TracingLock().is_active:
            _get_logger().error(
                "Cannot call input_tensors() while a _method()-traced function "
                "is executing. Mixing active contexts is not allowed.")
            raise Exception("Mixing active contexts is not allowed")

        tensors, metadata = self._normalize_named_tensor_payload(
            "input_tensors", node_name, tensors)

        if not ExportManager._interpret_graph:
            return self._passthrough_dict_values(tensors)

        # create the node if it doesn't exist
        if node_name in self.nodes:
            traced_tensors_node = self.nodes[node_name]
        else:
            traced_tensors_node = self._setup_new_node(
                node_name, TracedTensorNode)

        _caller_identity = get_caller_stack_identity()

        # if the node is not tracing, we validate the inputs only and return the raw tensors
        # the node is not tracing if it is already compiled.
        if not traced_tensors_node.is_tracing:
            matching_origin = next(
                (
                    identity for identity in traced_tensors_node._caller_identities
                    if caller_identity_has_same_anchor(identity, _caller_identity)
                ),
                None,
            )
            if matching_origin is None:
                raise Exception(
                    f"Error: node '{node_name}' is being called from a different annotation origin "
                    f"than the first trace. Cannot reuse a node name from a different call site.\n"
                    f"New call site:\n{format_caller_identity(_caller_identity)}")
            if _caller_identity not in traced_tensors_node._caller_identities:
                _get_logger().warning(
                    f"Warning: node '{node_name}' is being re-entered from a new caller context, "
                    f"but the normalized annotation origin matches a previously seen site. "
                    f"Allowing re-entry.\n"
                    f"Original origin:\n{format_caller_identity(matching_origin)}\n"
                    f"New call site:\n{format_caller_identity(_caller_identity)}")
                traced_tensors_node._caller_identities.add(_caller_identity)
            traced_tensors_node.reentry_validate_inputs(tensors)
            return self._passthrough_dict_values(tensors)

        traced_tensors_node._caller_identities.add(_caller_identity)

        # Register a warp segmenter for this region the first time it opens on the tracing
        # path. For torch-only regions the segmenter is inert (open_node stays the original
        # node, open_kind stays "torch", no rename happens). For regions with warp crossings
        # the patched wp.from_torch will call into the segmenter to split the region.
        if (node_name not in self._region_segmenters
                and getattr(self, "_warp_bridge_state", None)):
            from leapp.warp_bridge import RegionSegmenter, set_active_segmenter
            seg = RegionSegmenter(self, region=node_name, first_node=self.nodes[node_name])
            self._region_segmenters[node_name] = seg
            set_active_segmenter(seg)

        # Warn if pre-compiled ScriptFunctions are visible in the caller's scope
        # this is only a best effort warning, catching the error and ignoring if fault
        try:
            warn_if_script_functions_in_scope()
        except Exception:
            # ignore errors from warn_if_script_functions_in_scope
            pass

        # we need to handle input tensors more carefully than outputs because
        # we need to ensure the inputs are returned in the original structure
        traced_tensors = []
        for tensor_name, tensor in tensors.items():
            traced_tensor = traced_tensors_node.create_input(
                tensor, tensor_name, semantics=metadata.get(tensor_name))
            traced_tensors.append(traced_tensor)

        # if node is tracing we return the traced tensors
        return traced_tensors[0] if len(traced_tensors) == 1 else tuple(traced_tensors)

    def output_tensors(self, node_name: str, tensors, static_outputs=None, **kwargs):
        tensors, metadata = self._normalize_named_tensor_payload(
            "output_tensors", node_name, tensors)

        if not ExportManager._interpret_graph:
            return self._passthrough_dict_values(tensors)
 

        resolved_name = self._resolve_open_node_name(node_name)
        if resolved_name in self.nodes:
            traced_tensors_node = self.nodes[resolved_name]
        else:
            raise Exception(
                f"output_tensors() called for node '{node_name}' but input_tensors() was never called for it. "
                "Call annotate.input_tensors() before annotate.output_tensors() for the same node name.")

        # process outputs
        flattened_tensors = flatten_io_structure(tensors, '')

        static_outputs_metadata = {}
        normalized_static_outputs = None
        if static_outputs is not None:
            normalized_static_outputs, static_outputs_metadata = self._normalize_named_tensor_payload(
                "output_tensors static_outputs", node_name, static_outputs)

        if not traced_tensors_node.is_tracing:
            flattened_static = {}
            if normalized_static_outputs is not None:
                flattened_static = flatten_io_structure(normalized_static_outputs, '')
            traced_tensors_node.reentry_validate_and_tag_outputs(
                flattened_tensors, flattened_static)
            return self._passthrough_dict_values(tensors)

        # Warn if pre-compiled ScriptFunctions are visible in the caller's scope
        warn_if_script_functions_in_scope()

        if not getattr(traced_tensors_node, 'dry_run', False):
            self._validate_initial_traced_payload(
                "output_tensors", node_name, traced_tensors_node, flattened_tensors)

        # process static outputs (constant tensors that should be returned but aren't derived from inputs)
        flattened_static_outputs = None
        if normalized_static_outputs is not None:
            flattened_static_outputs = flatten_io_structure(normalized_static_outputs, '')

        export_with = None if self.is_dry_run(node_name) else kwargs.get("export_with", None)
        traced_tensors_node.compile_trace(flattened_tensors,
                                          backend=export_with,
                                          backend_params=kwargs.get("backend_params", {}),
                                          static_tensors=flattened_static_outputs,
                                          semantics_map=metadata,
                                          static_semantics_map=static_outputs_metadata)
        self._assign_completion_index(traced_tensors_node)

        return self._passthrough_dict_values(tensors)

    def register_buffer(self, node_name: str, tensors):
        """Register tensors as persistent buffers for a traced node.

        The tensors become part of the compiled module's state and persist
        across forward calls. The returned TracedData can be used in traced
        computations, and modifications (via in-place ops) will be retained.

        Args:
            node_name: Name of the TracedTensorNode to register the buffers with
            tensors: A single tensor, a list/tuple of tensors, or a dict
                mapping buffer names to tensors. Names are auto-generated
                when not provided.

        Returns:
            Single TracedData if one buffer, or tuple of TracedData if multiple.

        Example:
            ```python
            # With explicit names (dict):
            self.values, self.state = annotate.register_buffer('node', {
                'values': self.values, 'state': self.state
            })

            # Without names (single tensor):
            self.values = annotate.register_buffer('node', self.values)

            # Without names (list):
            self.values, self.state = annotate.register_buffer(
                'node', [self.values, self.state]
            )
            ```
        """
        if not ExportManager._interpret_graph:
            if isinstance(tensors, torch.Tensor):
                return tensors
            if isinstance(tensors, dict):
                values = list(tensors.values())
                return values[0] if len(values) == 1 else tuple(values)
            return tensors[0] if len(tensors) == 1 else tuple(tensors)

        if node_name not in self.nodes:
            msg = (
                f"register_buffer() called for node '{node_name}' but node not found. "
                "Call annotate.input_tensors() first to create the node.")
            _get_logger().error(msg)
            raise Exception(msg)

        # Normalize input to a dict with auto-generated names if needed
        tensors, was_single = self._normalize_buffer_input(node_name, tensors)

        traced_node = self.nodes[node_name]

        if not isinstance(traced_node, TracedTensorNode):
            msg = (
                f"register_buffer() is not supported for node '{node_name}' — "
                "it was created with the legacy method annotation.")
            _get_logger().error(msg)
            raise Exception(msg)

        if not traced_node.is_tracing:
            values = list(tensors.values())
            return values[0] if was_single else tuple(values)

        # Validate and wrap static tensors while preserving nested structure
        result = traced_node.create_static_tensors(tensors)
        values = list(result.values())
        return values[0] if was_single else tuple(values)

    def _normalize_buffer_input(self, node_name, tensors):
        """Normalize tensors arg into (dict, was_single).

        Returns:
            (dict mapping names to tensors, bool indicating single-tensor input)
        """
        if isinstance(tensors, dict):
            return tensors, len(tensors) == 1

        is_single = isinstance(tensors, torch.Tensor)
        items = [tensors] if is_single else list(tensors)

        node = self.nodes.get(node_name)
        start_idx = node._next_buffer_idx if node is not None else 0
        named = {}
        for i, t in enumerate(items):
            named[f"buffer_{start_idx + i}"] = t
        if node is not None:
            node._next_buffer_idx = start_idx + len(items)

        return named, is_single

    @staticmethod
    def _validate_flat_state_payload(api_name: str, node_name: str, tensors):
        for state_name, value in tensors.items():
            if not is_tracable_tensor_type(value):
                raise TypeError(
                    f"{api_name}() for node '{node_name}' does not support nested state structures. "
                    f"State '{state_name}' has unsupported top-level type {type(value).__name__}. "
                    "Please either explicitly list out each state as its own named tensor "
                    "or use input_tensors() and rely on LEAPP feedback detection."
                )

    @staticmethod
    def _validate_initial_traced_payload(api_name: str, node_name: str,
                                         traced_tensors_node: TracedTensorNode,
                                         flattened_tensors: dict) -> None:
        """Validate first-trace payloads for APIs that require active TracedTensors.

        This helper is only for the initial tracing path, before a node has been
        compiled. It enforces two invariants:
        1. Every provided value is a traced tensor, not a raw torch/numpy value.
        2. Every traced tensor belongs to the same active node context as
           ``traced_tensors_node``.

        Matching these checks keeps user-facing errors consistent for
        ``output_tensors()`` and ``update_state()`` and avoids later internal FX
        failures when raw tensors slip into graph outputs/state updates.
        """
        instances = set(is_traced_type(tensor)
                        for tensor in flattened_tensors.values())

        if not all(instances):
            types = set(type(tensor).__name__
                        for tensor in flattened_tensors.values())
            msg = (
                f"{api_name}() for node '{node_name}' received non-traced tensors: {types}\n"
                "Possible causes:\n"
                "1. You did not use the traced tensors returned by leapp functions — "
                "make sure to replace your original tensors with the return value of annotation functions.\n"
                "2. An operation in your computation broke tracing (e.g. converting to numpy and back).\n"
                f"3. You passed raw tensors instead of the traced ones to {api_name}().")
            _get_logger().error(msg)
            raise Exception(msg)

        context_names = {
            tensor.context for tensor in flattened_tensors.values()
        }
        if not (len(context_names) == 1 and next(iter(context_names)) == traced_tensors_node.name):
            msg = (
                f"{api_name}() for node '{node_name}' received tensors that belong to a different node: "
                f"{context_names}. Make sure you are passing tensors derived from "
                f"annotate.input_tensors('{node_name}', ...) to annotate.{api_name}('{node_name}', ...).")
            _get_logger().error(msg)
            raise Exception(msg)

    def state_tensors(self, node_name: str, tensors: dict[str, torch.Tensor]) -> TracedTensor | tuple[TracedTensor, ...]:
        """Register state tensors (both inputs AND outputs) for a traced node.

        Call input_tensors() first to create the node. Use update_state() to set output values.
        """
        if not ExportManager._interpret_graph:
            self._validate_flat_state_payload("state_tensors", node_name, tensors)
            return self._passthrough_dict_values(tensors)

 

        if node_name not in self.nodes:
            msg = (
                f"state_tensors() called for node '{node_name}' but node not found. "
                "Call annotate.input_tensors() first to create the node.")
            _get_logger().error(msg)
            raise Exception(msg)

        traced_node = self.nodes[node_name]

        if not isinstance(traced_node, TracedTensorNode):
            msg = (
                f"state_tensors() is not supported for node '{node_name}' — "
                "it was created with the legacy method annotation.")
            _get_logger().error(msg)
            raise Exception(msg)

        self._validate_flat_state_payload("state_tensors", node_name, tensors)

        if not traced_node.is_tracing:
            traced_node.reentry_validate_inputs(tensors)
            return self._passthrough_dict_values(tensors)

        # Create state tensors (input placeholders that will also be outputs)
        state_dict = traced_node.create_state_tensors(tensors)
        values = list(state_dict.values())
        return values[0] if len(values) == 1 else tuple(values)

    def update_state(self, node_name: str, tensors: dict[str, TracedTensor]) -> TracedTensor | tuple[TracedTensor, ...]:
        """Set output values for state tensors and return passthrough values."""
        if not ExportManager._interpret_graph:
            self._validate_flat_state_payload("update_state", node_name, tensors)
            return self._passthrough_dict_values(tensors)

        if node_name not in self.nodes:
            msg = (
                f"update_state() called for node '{node_name}' but node not found. "
                "Call annotate.input_tensors() first to create the node.")
            _get_logger().error(msg)
            raise Exception(msg)

        traced_node = self.nodes[node_name]

        if not isinstance(traced_node, TracedTensorNode):
            msg = (
                f"update_state() is not supported for node '{node_name}' — "
                "it was created with the legacy method annotation.")
            _get_logger().error(msg)
            raise Exception(msg)

        self._validate_flat_state_payload("update_state", node_name, tensors)

        if not traced_node.is_tracing:
            traced_node.reentry_validate_state_update(tensors)
            return self._passthrough_dict_values(tensors)

        if not getattr(traced_node, 'dry_run', False):
            self._validate_initial_traced_payload(
                "update_state", node_name, traced_node, tensors)
        traced_node.update_state_tensors(tensors)
        return self._passthrough_dict_values(tensors)

    def module(self, node_name: str, model: torch.nn.Module,
               buffer_names: list[str] | None = None) -> None:
        """Register a module for automatic stateful buffer tracking.

        Replaces registered buffers with TracedTensor inputs so the forward
        pass is traced through them.  When ``output_tensors()`` triggers
        ``compile_trace()``, mutations are auto-detected: reassigned buffers
        become state outputs with feedback connections, non-mutated buffers are
        baked as constants.  Model buffers are restored afterwards.

        Must be called after ``input_tensors()`` creates the node and before
        the model's forward pass.

        Args:
            node_name: Name of the TracedTensorNode (must already exist).
            model: The ``nn.Module`` whose buffers to track.
            buffer_names: Optional list of buffer names to track (dotted names
                like ``"h_state"`` or ``"encoder.running_mean"``).
                If ``None``, all registered buffers are tracked.

        Example::

            leapp.start("graph", save_path=output_dir)
            obs_traced = annotate.input_tensors("policy", {"obs": obs})

            annotate.module("policy", model)
            action = model(obs_traced)

            annotate.output_tensors("policy", {"action": action},
                                    export_with="onnx-torchscript")
            leapp.stop()
            leapp.compile_graph()

        Note:
            Detects *reassignment* (``self.h = h_out``), not in-place mutation
            (``self.h.copy_(h_out)``). Use ``state_tensors()``/``update_state()``
            for in-place patterns.
        """
        if not ExportManager._interpret_graph:
            return

        if node_name not in self.nodes:
            msg = f"Error: module() called for node '{node_name}' but node not found. Call input_tensors() first to create the node."
            _get_logger().error(msg)
            raise Exception(msg)

        from leapp.buffer_tracker import BufferTracker
        tracker = BufferTracker(model, node_name, self, buffer_names=buffer_names)
        tracker.inject()

        self.nodes[node_name]._buffer_tracker = tracker


    def method(self, **params):
        def decorator(func):

            name = params.get("node_name", func.__name__)
            export_with = params.get("export_with", None)
            backend_params = params.get("backend_params", {})
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not ExportManager._interpret_graph:
                    return func(*args, **kwargs)
                
                # ~~~~~~~~~~~~~~~~~~~ ensure node exists ~~~~~~~~~~~~~~~~~~~~~~~~ #
                if name not in self.nodes:
                    self._setup_new_node(name, TracedTensorNode)
                
                # ~~~~~~~~~~~~~~~~~~~ set up inputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                
                params_list = list(sig.parameters.items())
                new_args = []
                new_kwargs = {}
                for i, arg in enumerate(args):
                    param_name, param = params_list[i]
                    if i == 0 and param_name in ('self', 'cls'):
                        new_args.append(arg)
                        continue
                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        for j, a in enumerate(args[i:]):
                            new_args.append(self.input_tensors(name, {f"arg_{j}": a}))
                        break
                    new_args.append(self.input_tensors(name, {param_name: arg}))
                
                for key, value in kwargs.items():
                    new_kwargs[key] = self.input_tensors(name, {key: value})
                
                # ~~~~~~~~~~~~~~~~~~~ register default kwargs as buffers ~~~~~~~~ #
                for param_name, param_value in bound_args.arguments.items():
                    if param_name in ('self', 'cls'):
                        continue
                    param = sig.parameters[param_name]
                    was_provided = (
                        param_name in kwargs or
                        (param.kind != inspect.Parameter.VAR_POSITIONAL and
                        list(sig.parameters.keys()).index(param_name) < len(args))
                    )
                    if not was_provided and is_tracable_tensor_type(param_value):
                        traced = self.register_buffer(name, {param_name: param_value})
                        new_kwargs[param_name] = traced
                
                # ~~~~~~~~~~~~~~~~~~~ run the function ~~~~~~~~~~~~~~~~~~~~~~~~~ #
                result = func(*new_args, **new_kwargs)
                # ~~~~~~~~~~~~~~~~~~~ set up outputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                return_names = extract_return_names(func)

                if result is None:
                    raise Exception(f"Error: annotated method {name} returned None, but LEAPP expects a return value")
                elif isinstance(result, tuple):
                    if len(return_names) != len(result):
                        _get_logger().error(
                            f"Fatal: annotated method {name} returned {len(result)} values, "
                            f"but LEAPP detected the following return names {return_names} from source")
                    output_dict = {return_names[i]: result[i] for i in range(len(result))}
                    self.output_tensors(
                        name,
                        output_dict,
                        export_with=export_with,
                        backend_params=backend_params,
                    )
                else:
                    self.output_tensors(
                        name,
                        {return_names[0]: result},
                        export_with=export_with,
                        backend_params=backend_params,
                    )
                # ~~~~~~~~~~~~~~~~~~~ set up outputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                return result
            return wrapper
        return decorator

    def _method(self, **params):
        """Legacy decorator for tracing functions via sys.settrace + ModuleBuilder.

        This uses the original source-code-capture approach (FunctionDecoratorNode)
        and is kept for use cases where the newer TracedTensorNode-based method()
        does not cover all patterns. Not advertised in the public API.
        """
        def decorator(func):

            if "node_name" in params:
                name = params["node_name"]
            else:
                name = func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not ExportManager._interpret_graph:
                    return func(*args, **kwargs)

                if name in self.nodes:
                    new_node = False
                    node_context = self.nodes[name]
                else:
                    new_node = True
                    node_context = self._setup_new_node(
                        name, FunctionDecoratorNode, **params)

                caller_namespace = frame_to_namespace(sys._getframe(1))

                if new_node:
                    _get_logger().info(f"****Tracing started for {name}****")
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    node_context.inspect_function_inputs(func, args, kwargs)
                    input_namespace = {
                        **caller_namespace, **bound_args.arguments}
                    if hasattr(func, '__self__'):
                        input_namespace['self'] = func.__self__
                    node_context.capture_inputs_from_namespace(input_namespace)
                    node_context.compile_trace(func)
                    node_context.snapshot_buffer_values(input_namespace)

                    trace_fn = node_context.create_trace_function(
                        os.path.basename(__file__), entry_hook=None)
                else:
                    node_context.validate_function_boundaries(func)
                    node_context.validate_function_inputs(func, args, kwargs)
                    trace_fn = node_context.create_trace_function(
                        os.path.basename(__file__), entry_hook=None)

                sys._getframe(1).f_trace = trace_fn

                TracingLock().acquire()
                sys.settrace(trace_fn)

                try:
                    result = func(*args, **kwargs)
                finally:
                    sys.settrace(None)
                    TracingLock().release()

                if new_node:
                    _get_logger().info(
                        f"****Tracing stopped for {node_context.name}****\n\n")
                    node_context.inspect_function_outputs(func, result)
                    if node_context.output_namespace is not None:
                        node_context.capture_outputs_from_namespace(
                            node_context.output_namespace)
                    self._assign_completion_index(node_context)
                else:
                    node_context.validate_function_outputs(func, result)
                    if node_context.output_namespace is not None:
                        node_context.validate_outputs_from_namespace(
                            node_context.output_namespace)
                    node_context.increment_cache_idx()

                return result
            return wrapper
        return decorator

    #########################################################
    # export flow control
    #########################################################

    def mirror_leapp_tags(self, source, target):
        if not ExportManager._interpret_graph:
            return
        try:
            if not verify_data_exact_match(source, target):
                _get_logger().error(
                    f"Error: source and target do not match: {source} != {target}")
                raise Exception("Error: source and target do not match")
            mirror_all_tensor_tags(source, target)
        except Exception as e:
            _get_logger().error(f"Unexpected error mirroring LEAPP tags: {e}")
            raise Exception(
                f"Error: unexpected error mirroring LEAPP tags: {e}")

    def get_io_descriptions(self):
        _get_logger().section(
            f"Compiling graph parameters for {len(self.nodes)} nodes")
        models = {"models": {}}
        nodes_to_describe = sorted(
            self.nodes.values(), key=lambda x: x.node_index)
        for node in nodes_to_describe:
            _get_logger().info(f"Compiling parameters for {node.name}")
            description = node.get_description()
            if 'parameters' in description and 'model_path' in description['parameters']:
                # Convert model path to be relative to YAML file location
                model_path = description['parameters']['model_path']
                if model_path:
                    description['parameters']['model_path'] = get_relative_path(
                        model_path, self.SAVE_PATH)
            models["models"][node.name] = description
        return models

    def compile_models(self):
        _get_logger().section(f"Discovered {len(self.nodes)} nodes")
        for node_context in self.nodes.values():
            _get_logger().section(f"Compiling {node_context.name}")
            node_context.compile_model()
            _get_logger().info("Success\n")

    def save_models(self):
        if self.SAVE_PATH is None:
            raise Exception(
                "Error: No save path provided, please provide a save path to export the graph")
        _get_logger().section(
            f"Saving {len(self.nodes)} models to {self.SAVE_PATH}")
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        for node_context in self.nodes.values():
            _get_logger().info(
                f"Saving {node_context.name}")
            node_context.save_model(self.SAVE_PATH)
            _get_logger().info("Success\n")

    def validate_all_models(self, rtol: float = 1e-3, atol: float = 1e-5, strict: bool = True):
        """Validate all exported models by comparing computed outputs against captured outputs.

        This method runs each compiled model with its captured input values and verifies
        that the outputs match the captured output values within the specified tolerances.

        Args:
            rtol (float): Relative tolerance for torch.allclose comparison. Default: 1e-3
            atol (float): Absolute tolerance for torch.allclose comparison. Default: 1e-5

        Returns:
            dict: A dictionary mapping node names to validation results:
                - True: All outputs matched within tolerance
                - False: At least one output did not match
                - Exception object: If model execution failed

        Raises:
            Exception: If called before compile_models() has been run.

        Example:
            >>> leapp.start(name="my_graph")
            >>> # ... run your code ...
            >>> leapp.stop()
            >>> leapp.compile_graph()
            >>> results = annotate.validate_all_models()
            >>> assert all(results.values()), "Some models failed validation"
        """

        if not self.nodes:
            _get_logger().error("No nodes to validate")
            return {}

        results = {}
        error_hints = []

        _get_logger().section(f"Validating {len(self.nodes)} models")

        for node_name, node in self.nodes.items():

            _get_logger().info(f"Validating {node_name}...")

            try:
                passed, error_hint = node.validate_compiled_model(
                    rtol=rtol, atol=atol)
                results[node_name] = passed
                if error_hint is not None:
                    error_hints.append(f"{node_name}: {error_hint}")

            except Exception as e:
                _get_logger().error(
                    f"{node_name}: Validation failed with exception: {e}")
                results[node_name] = e

            finally:
                # Free GPU memory after validating each model
                # Delete the compiled model (validation is the last step)
                node.delete_compiled_model()
                # Clear CUDA cache to ensure GPU memory is released
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Print summary
        _get_logger().section("Validation Summary")
        passed = sum(1 for v in results.values() if v is True)
        failed = sum(1 for v in results.values() if v is False)
        errors = sum(1 for v in results.values() if isinstance(v, Exception))

        _get_logger().info(f"  Passed: {passed}/{len(results)}")
        if failed > 0:
            _get_logger().warning(f"  Failed: {failed}/{len(results)}")
        if errors > 0:
            _get_logger().error(f"  Errors: {errors}/{len(results)}")

        if strict and (failed > 0 or errors > 0):
            failed_nodes = [name for name,
                            v in results.items() if v is not True]
            message = (
                f"Model validation failed for {len(failed_nodes)} node(s): {failed_nodes}"
            )
            if error_hints:
                message += "\n\n" + "\n".join(error_hints)
            raise Exception(message)

        return results