#
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import functools
import inspect
import yaml
import os
import torch

from leapp._logging import _get_logger
from leapp.leapp_graph.leapp_graph import LeappGraph
from leapp.leapp_graph.leapp_node import LeappNode
from leapp.leapp_graph.traced_node import TracedTensorNode
from leapp.leapp_graph.datatypes import (
    TracedTensor,
    is_traced_type,
    apply_traced_tensor_patches,
    remove_traced_tensor_patches,
    is_tracable_tensor_type,
)
from leapp.leapp_graph.datatypes.global_patching import warn_if_script_functions_in_scope
from leapp.utils.enums import MergeCfgEnum
from leapp.utils.tensor_description import TensorSemantics
from leapp.utils.tensor_description import (verify_data_exact_match,
                                             flatten_io_structure,
                                             unwrap_tensor_semantics,
                                             apply_semantic_metadata)
from leapp.utils.utils import (get_relative_path,
                               get_system_info,
                               mirror_all_tensor_tags,
                               extract_return_names)


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
            self._numpy_patches_applied = False

            # tracetime variables
            self.nodes = {}

            ExportManager._initialized = True

    #########################################################
    # no-op context manager (used when block() is called outside tracing)
    #########################################################
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    #########################################################
    # flow control
    #########################################################
    def start(self, name, save_path=".", verbose=False, dry_run=False, patch_numpy=True):
        """Initialize and start LEAPP graph interpretation.

        This method prepares the export manager for tracing by setting up the graph name,
        creating the save directory, configuring the logger, and enabling graph interpretation.
        If graph interpretation is already active, it will reset the current graph state.

        Args:
            name (str): The name of the graph to be created. This will be used as the 
                directory name where graph artifacts are saved.
            save_path (str, optional): The base directory path where the graph directory 
                will be created. Defaults to "." (current directory).
            verbose (bool, optional): If True, enables verbose logging output. 
                Defaults to False.
            dry_run (bool, optional): If True, enables dry run mode which skips model
                compilation and export. Defaults to False.
            patch_numpy (bool, optional): If True, applies patches to torch.from_numpy,
                torch.as_tensor, and torch.tensor to enable TracedTensor compatibility
                with numpy operations. Defaults to True.

        Returns:
            None

        Note:
            - The full save path will be: {save_path}/{name}/
            - Calling start() while interpretation is already active will reset the graph. This is discouraged.
        """
        self.GRAPH_NAME = name
        self.SAVE_PATH = os.path.join(save_path, self.GRAPH_NAME)
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        _get_logger().configure(self.SAVE_PATH, verbose=verbose)
        if ExportManager._interpret_graph:
            _get_logger().warning("LEAPP graph interpretation is already enabled, "
                                  "calling start() again will reset the graph")
            _get_logger().warning("Resetting graph...")
        if dry_run:
            _get_logger().info("Starting dry run mode")
        self.dry_run = dry_run
        self.nodes = {}
        ExportManager._interpret_graph = True
        # Apply patches for torch functions that bypass __torch_function__
        self._numpy_patches_applied = patch_numpy
        if patch_numpy:
            apply_traced_tensor_patches()

    def stop(self):
        """Stop LEAPP graph interpretation and disable tracing.

        This method disables graph interpretation mode that was previously enabled by start().
        It performs safety checks to ensure that no active tracing is in progress and that
        graph interpretation is currently enabled before stopping.

        Args:
            None

        Returns:
            None

        Raises:
            Exception: If ExportManager is currently in the middle of tracing a node.
            Exception: If graph interpretation is not currently enabled.

        Note:
            - This method should only be called after start() has been called.
            - Ensure all active tracing operations are completed before calling stop().
        """
        if not ExportManager._interpret_graph:
            raise Exception("ExportManager graph interpretation is disabled")
        ExportManager._interpret_graph = False
        # Restore model buffers for any pending buffer trackers
        for node in self.nodes.values():
            if hasattr(node, '_buffer_tracker') and node._buffer_tracker is not None:
                node._buffer_tracker.restore()
        # Remove patches to restore original torch function behavior
        if self._numpy_patches_applied:
            remove_traced_tensor_patches()
            self._numpy_patches_applied = False

    #########################################################
    # node setup
    #########################################################
    def get_node_index(self, name):
        if name in self.nodes.keys():
            # retracing inherits the node index of the original node
            node_index = self.nodes[name].node_index
        else:
            node_index = len(self.nodes)
        return node_index

    def _setup_new_node(self, name, node_class: LeappNode, **kwargs):
        node_index = self.get_node_index(name)

        if self.dry_run:
            kwargs['export_with'] = None
            kwargs['backend_params'] = {}

        node = node_class(name, node_index,
                          backend=kwargs.get("export_with", None),
                          backend_params=kwargs.get("backend_params", None),
                          inputs=kwargs.get("inputs", None),
                          outputs=kwargs.get("outputs", None),
                          environment_constants=kwargs.get(
                              "environment_constants", None),
                          register_buffers=kwargs.get("register_buffers", None))

        return node, name

    #########################################################
    # annotation APIs
    #########################################################
    def input_tensors(self, node_name: str, tensors):
        # If TensorSemantics are passed (single or list), unwrap to dict
        metadata = {}
        if isinstance(tensors, (TensorSemantics, list)) and (
            isinstance(tensors, TensorSemantics) or
            any(isinstance(t, TensorSemantics) for t in tensors)
        ):
            tensors, metadata = unwrap_tensor_semantics(tensors)

        if not ExportManager._interpret_graph:
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)
 

        # create the node if it doesn't exist
        if node_name in self.nodes.keys():
            traced_tensors_node = self.nodes[node_name]
        else:
            traced_tensors_node, node_name = self._setup_new_node(
                node_name, TracedTensorNode)
            self.nodes[node_name] = traced_tensors_node

        # TODO: this is still confusing. we need to make it more explicit.
        tensors_changed = False
        if not isinstance(tensors, dict):
            tensors_changed = True
            tensors = {'tensor': tensors}

        # reason this is convoluted is to mirror the scheme in output_tensors
        if tensors_changed:
            _get_logger().warning(f"Warning: no tensor name provided for input_tensors call in node {node_name}\n"
                                  "Assuming default tensor name")

        # if the node is not tracing, we validate the inputs only and return the raw tensors
        # the node is not tracing if it is already compiled.
        if not traced_tensors_node.is_tracing:
            for tensor_name, tensor in tensors.items():
                traced_tensors_node.validate_input_and_update_tags(
                    tensor_name, tensor_name, tensor)
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

        # Warn if pre-compiled ScriptFunctions are visible in the caller's scope
        warn_if_script_functions_in_scope()

        # we need to handle input tensors more carefully than outputs because
        # we need to ensure the inputs are returned in the original structure
        traced_tensors = []
        for tensor_name, tensor in tensors.items():
            traced_tensor = traced_tensors_node.create_input(
                tensor, tensor_name)
            traced_tensors.append(traced_tensor)

        # Apply semantic metadata from TensorDescription wrappers
        if metadata:
            apply_semantic_metadata(traced_tensors_node, metadata)

        # if node is tracing we return the traced tensors
        return traced_tensors[0] if len(traced_tensors) == 1 else tuple(traced_tensors)

    def output_tensors(self, node_name: str, tensors, static_outputs=None, **kwargs):
        # If TensorSemantics are passed (single or list), unwrap to dict
        metadata = {}
        if isinstance(tensors, (TensorSemantics, list)) and (
            isinstance(tensors, TensorSemantics) or
            any(isinstance(t, TensorSemantics) for t in tensors)
        ):
            tensors, metadata = unwrap_tensor_semantics(tensors)

        if not ExportManager._interpret_graph:
            return
 

        if node_name in self.nodes.keys():
            traced_tensors_node = self.nodes[node_name]
        else:
            _get_logger().error(
                f"Error: output tensors called for node {node_name} but not registered to the ExportManager")
            raise Exception(
                "Error: exception detected in output_tensors declaration")

        # process outputs
        tensors_changed = False
        if not isinstance(tensors, dict):
            tensors_changed = True
            tensors = {'tensor': tensors}

        flattened_tensors = flatten_io_structure(tensors, '')

        if not traced_tensors_node.is_tracing:
            # tag regardless of tracing status
            for tensor_name, tensor in flattened_tensors.items():
                traced_tensors_node.tag_data(tensor, tensor_name)
            if static_outputs is not None:
                if not isinstance(static_outputs, dict):
                    static_outputs = {'static_output': static_outputs}
                for tensor_name, tensor in flatten_io_structure(static_outputs, '').items():
                    traced_tensors_node.tag_data(tensor, tensor_name)
            return

        if tensors_changed:
            _get_logger().warning(f"Warning: no tensor name provided for output_tensors call in node {node_name}\n"
                                  "Assuming default tensor name")

        # Warn if pre-compiled ScriptFunctions are visible in the caller's scope
        warn_if_script_functions_in_scope()

        instances = set(is_traced_type(tensor) for tensor in flattened_tensors.values())

        if not all(instances):
            types = set(type(tensor) for tensor in flattened_tensors.values())
            _get_logger().error(
                f"Error: in output_tensors call for the node {node_name} detected the following"
                f" types when expected all outputs to be TracedData: {types}\n"
                "**This could happen if you are not using TracedData in your computations.**\n"
                "Please verify if you are using the returned wrapped tensors from input_tensors() to "
                "correctly trace your computations.")
            raise Exception(
                "Error: exception detected in output_tensors declaration")

        context_names = set(
            [tensor.context for tensor in flattened_tensors.values()])
        # Check that all tensors come from exactly one context matching the node name
        if not (len(context_names) == 1 and next(iter(context_names)) == traced_tensors_node.name):
            _get_logger().error(
                f"Error: expected all context names to match the node name: {traced_tensors_node.name}"
                f" but detected the following context names: {context_names}")
            raise Exception(
                "Error: exception detected in output_tensors declaration")

        # process static outputs (constant tensors that should be returned but aren't derived from inputs)
        flattened_static_outputs = None
        if static_outputs is not None:
            static_outputs_changed = False
            if not isinstance(static_outputs, dict):
                static_outputs_changed = True
                static_outputs = {'static_output': static_outputs}

            flattened_static_outputs = flatten_io_structure(static_outputs, '')

            if static_outputs_changed:
                _get_logger().warning(f"Warning: no tensor name provided for static_outputs in node {node_name}\n"
                                      "Assuming default tensor name")

        export_with = None if self.dry_run else kwargs.get("export_with", None)
        traced_tensors_node.compile_trace(flattened_tensors,
                                          backend=export_with,
                                          backend_params=kwargs.get("backend_params", {}),
                                          static_tensors=flattened_static_outputs)

        # Apply semantic metadata from TensorDescription wrappers
        if metadata:
            apply_semantic_metadata(traced_tensors_node, metadata)

    def register_buffer(self, node_name: str, tensors: dict):
        """Register tensors as persistent buffers for a traced node.

        The tensors become part of the compiled module's state and persist
        across forward calls. The returned TracedData can be used in traced
        computations, and modifications (via in-place ops) will be retained.

        Args:
            node_name: Name of the TracedTensorNode to register the buffers with
            tensors: Dictionary mapping buffer names to tensors

        Returns:
            Single TracedData if one buffer, or tuple of TracedData if multiple.

        Example:
            ```python
            class Module:
                def __init__(self):
                    self.values = torch.tensor([1, 2, 3])
                    self.state = torch.tensor([0, 0, 0])

                def run(self, input):
                    # Make tensors participate in tracing
                    self.values, self.state = annotate.register_buffer('my_node', {
                        'values': self.values,
                        'state': self.state
                    })

                    self.values[:] = input  # This assignment is now traced
                    return self.values * 100
            ```
        """
        if not ExportManager._interpret_graph:
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

 

        if node_name not in self.nodes:
            _get_logger().error(
                f"Error: register_buffer called for node '{node_name}' but node not found. "
                "Call input_tensors() first to create the node.")
            raise Exception("Error: exception detected in register_buffer")

        traced_node = self.nodes[node_name]

        if not isinstance(traced_node, TracedTensorNode):
            _get_logger().error(
                f"Error: register_buffer only works with TracedTensorNode, "
                f"but '{node_name}' is a {type(traced_node).__name__}")
            raise Exception("Error: exception detected in register_buffer")

        if not traced_node.is_tracing:
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

        # Flatten, validate, and wrap using create_static_tensors
        flattened = flatten_io_structure(tensors, '')
        result = traced_node.create_static_tensors(flattened)
        values = list(result.values())
        return values[0] if len(values) == 1 else tuple(values)

    def state_tensors(self, node_name: str, tensors: dict[str, torch.Tensor]) -> TracedTensor | tuple[TracedTensor, ...]:
        """Register state tensors (both inputs AND outputs) for a traced node.

        Call input_tensors() first to create the node. Use update_state() to set output values.
        """
        if not ExportManager._interpret_graph:
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

 

        if node_name not in self.nodes:
            _get_logger().error(
                f"Error: state_tensors called for node '{node_name}' but node not found. "
                "Call input_tensors() first to create the node.")
            raise Exception("Error: exception detected in state_tensors")

        traced_node = self.nodes[node_name]

        if not isinstance(traced_node, TracedTensorNode):
            _get_logger().error(
                f"Error: state_tensors only works with TracedTensorNode, "
                f"but '{node_name}' is a {type(traced_node).__name__}")
            raise Exception("Error: exception detected in state_tensors")

        if not traced_node.is_tracing:
            values = list(tensors.values())
            return values[0] if len(values) == 1 else tuple(values)

        # Create state tensors (input placeholders that will also be outputs)
        state_dict = traced_node.create_state_tensors(tensors)
        values = list(state_dict.values())
        return values[0] if len(values) == 1 else tuple(values)

    def update_state(self, node_name: str, tensors: dict[str, TracedTensor]) -> None:
        """Set output values for state tensors. If not called, state passes through unchanged."""
        if not ExportManager._interpret_graph:
            return  # No-op when not tracing

 

        if node_name not in self.nodes:
            _get_logger().error(
                f"Error: update_state called for node '{node_name}' but node not found.")
            raise Exception("Error: exception detected in update_state")

        traced_node = self.nodes[node_name]

        if not isinstance(traced_node, TracedTensorNode):
            _get_logger().error(
                f"Error: update_state only works with TracedTensorNode, "
                f"but '{node_name}' is a {type(traced_node).__name__}")
            raise Exception("Error: exception detected in update_state")

        if not traced_node.is_tracing:
            return

        traced_node.update_state_tensors(tensors)

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

            annotate.start("graph", save_path=output_dir)
            obs_traced = annotate.input_tensors("policy", {"obs": obs})

            annotate.module("policy", model)
            action = model(obs_traced)

            annotate.output_tensors("policy", {"action": action},
                                    export_with="onnx-torchscript")
            annotate.stop()
            annotate.compile_graph()

        Note:
            Detects *reassignment* (``self.h = h_out``), not in-place mutation
            (``self.h.copy_(h_out)``). Use ``state_tensors()``/``update_state()``
            for in-place patterns.
        """
        if not ExportManager._interpret_graph:
            return

        if node_name not in self.nodes:
            _get_logger().error(
                f"Error: module() called for node '{node_name}' but node not found. "
                "Call input_tensors() first to create the node.")
            raise Exception("Error: exception detected in module")

        from leapp.buffer_tracker import BufferTracker
        tracker = BufferTracker(model, node_name, self, buffer_names=buffer_names)
        tracker.inject()

        self.nodes[node_name]._buffer_tracker = tracker


    def method(self, **params):
        def decorator(func):
            if "node_name" in params:
                name = params["node_name"]
            else:
                name = func.__name__
            
            export_with = params.get("export_with", None)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not ExportManager._interpret_graph:
                    return func(*args, **kwargs)
                
                # ~~~~~~~~~~~~~~~~~~~ ensure node exists ~~~~~~~~~~~~~~~~~~~~~~~~ #
                if name not in self.nodes:
                    node, _ = self._setup_new_node(name, TracedTensorNode)
                    self.nodes[name] = node
                
                is_first_trace = self.nodes[name].is_tracing

                def wrap_if_tracable(value, tensor_name, display_name=None):
                    if display_name is None:
                        display_name = tensor_name
                    if is_tracable_tensor_type(value):
                        return self.input_tensors(name, {tensor_name: value})
                    if is_first_trace:
                        _get_logger().warning(
                            f"Detected non-tracable dynamic input (type={type(value).__name__}) "
                            f"for parameter '{display_name}' in method '{name}'. "
                            f"This value will be passed through as a constant.")
                    return value
                
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
                            new_args.append(wrap_if_tracable(a, f"arg_{j}", display_name=f"*args[{j}]"))
                        break
                    new_args.append(wrap_if_tracable(arg, param_name))
                
                for key, value in kwargs.items():
                    new_kwargs[key] = wrap_if_tracable(value, key)
                
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
                    self.output_tensors(name, output_dict, export_with=export_with)
                else:
                    self.output_tensors(name, {return_names[0]: result}, export_with=export_with)
                # ~~~~~~~~~~~~~~~~~~~ set up outputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
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

    #########################################################
    # graph compilation
    #########################################################
    def compile_graph(self, visualize=True, verbose=None, merge_nodes: MergeCfgEnum = MergeCfgEnum.NO_MERGE,
                      validate: bool = True, rtol: float = 1e-3, atol: float = 1e-5, strict=True):
        """Compile and save the computational graph from traced nodes.

        This method performs the complete pipeline of compiling traced nodes into exportable
        models, building graph connections, saving models to disk, and generating a YAML 
        description of the entire computational graph. It also optionally creates a 
        visualization of the graph structure.

        Args:
            visualize (bool, optional): If True, generates a visual representation of the 
                graph structure and saves it to the output directory. The visualization 
                will be created even if an error occurs during the process. 
                Defaults to True.
            merge_nodes (MergeCfgEnum, optional): Strategy for merging nodes in the graph.
                Options from MergeCfgEnum include:
                - NO_MERGE: Keep all nodes separate (default)
                - MERGE_ALL: Merge all possible nodes
                - MERGE_SEQUENTIAL: Merge only sequentially connected nodes (not available yet)
                Defaults to MergeCfgEnum.NO_MERGE.

        Returns:
            None

        Generated Artifacts:
            - Compiles all traced models using the configured backend
            - Saves compiled models to {SAVE_PATH}/ directory
            - Creates {GRAPH_NAME}.yaml file with complete graph description
            - Generates visualization files if visualize=True
            - Updates self.detected_nodes and self.detected_pipeline attributes
            - Prints graph statistics to the logger

        Note:
            - Must be called after tracing is complete and stop() has been called.
            - The YAML file contains model descriptions, pipeline connections, and system info.
            - Graph statistics include node count, dangling inputs/outputs, and edge counts.
            - Visualization errors are logged but don't stop the compilation process.
        """
        # compile models first before input name reconciliation
        if verbose is not None:
            _get_logger().set_verbose(verbose)

        if not self.dry_run:        
            self.compile_models()

        # builds the graph connections. this may change input and output names
        if not isinstance(merge_nodes, MergeCfgEnum):
            raise Exception(
                f"Error: merge_nodes must be an instance of MergeCfgEnum, got {type(merge_nodes)}")
        graph = LeappGraph(self.nodes, self.GRAPH_NAME)
        graph.merge_nodes(merge_nodes)
        pipeline = graph.get_full_pipeline_description()

        inital_value_filename = None
        if not self.dry_run:
            # cache the initial values for feedback inputs
            inital_value_filename = graph.save_feedback_initial_values(self.SAVE_PATH, self.GRAPH_NAME)


        if inital_value_filename is not None:
            pipeline['pipeline']['initial_values'] = inital_value_filename

        if not self.dry_run:
            self.save_models()

        models = self.get_io_descriptions()

        if visualize:
            try:
                graph.visualize(self.SAVE_PATH, self.GRAPH_NAME)
            except Exception as e:
                _get_logger().error(f"Error visualizing graph: {e}")

        internal_connections, total_edges = graph.get_graph_statistics()

        # Print graph statistics
        _get_logger().section("Graph Statistics")
        _get_logger().info(f"- Computation nodes: {len(self.nodes)}")
        _get_logger().info(f"- Dangling inputs: {len(graph.graph_inputs)}")
        _get_logger().info(f"- Dangling outputs: {len(graph.graph_outputs)}")
        _get_logger().info(f"- Internal connections: {internal_connections}")
        _get_logger().info(f"- Total edges: {total_edges}")

        system_info = get_system_info()
        with open(os.path.join(self.SAVE_PATH, f"{self.GRAPH_NAME}.yaml"), "w") as f:
            yaml.dump(models, f, sort_keys=False)
            f.write("\n")  # Add a newline separator
            yaml.dump(pipeline, f)
            f.write("\n")
            yaml.dump(system_info, f)
            f.write("\n")

        # store the models and pipeline as part of the object
        # this will do a rewrite each time
        # this is **ONLY USED FOR TESTING**
        self.detected_nodes = models['models']
        self.detected_pipeline = pipeline['pipeline']

        # validate all the models in the compute graph
        if validate and not self.dry_run:
            return self.validate_all_models(rtol=rtol, atol=atol, strict=strict)

        return True

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
            >>> annotate.start(name="my_graph")
            >>> # ... run your code ...
            >>> annotate.stop()
            >>> annotate.compile_graph()
            >>> results = annotate.validate_all_models()
            >>> assert all(results.values()), "Some models failed validation"
        """

        if not self.nodes:
            _get_logger().error("No nodes to validate")
            return {}

        results = {}

        _get_logger().section(f"Validating {len(self.nodes)} models")

        for node_name, node in self.nodes.items():

            _get_logger().info(f"Validating {node_name}...")

            try:
                # Check that model has been compiled
                if node.compiled_model is None:
                    _get_logger().warning(f"Model {node_name} does not have a compiled model. "
                                          "Skipping validation.")
                    # model wasn't provided but we will skip validation
                    results[node_name] = True
                    continue

                # 1. Extract input values from TensorDescriptions
                # The compiled model expects flat tensor inputs in order
                input_values = [
                    tensor_desc.value for tensor_desc in node.inputs]

                # 3. Run the compiled model
                with torch.no_grad():
                    exported_outputs = node.compiled_model(*input_values)

                # 4. Normalize outputs to tuple for consistent handling
                if not isinstance(exported_outputs, tuple):
                    exported_outputs = (exported_outputs,)

                # 5. Extract source code output values
                source_outputs = tuple(
                    tensor_desc.value for tensor_desc in node.outputs)

                # 6. Validate output count matches
                if len(exported_outputs) != len(source_outputs):
                    _get_logger().error(
                        f"{node_name}: Output count mismatch - "
                        f"got {len(exported_outputs)}, expected {len(source_outputs)}")
                    results[node_name] = True  # the model wasn't provided
                    continue

                # 7. Compare each output tensor
                all_match = True
                for idx, (exported, source) in enumerate(zip(exported_outputs, source_outputs)):
                    output_name = node.outputs[idx].name if idx < len(
                        node.outputs) else f"output_{idx}"

                    # Ensure tensors are on the same device for comparison
                    if exported.device != source.device:
                        exported = exported.to(source.device)

                    # Check for NaN/Inf values
                    exported_nan = torch.isnan(exported).sum().item()
                    exported_inf = torch.isinf(exported).sum().item()
                    source_nan = torch.isnan(source).sum().item()
                    source_inf = torch.isinf(source).sum().item()
                    
                    if exported_nan > 0 or exported_inf > 0 or source_nan > 0 or source_inf > 0:
                        all_match = False
                        num_elements = exported.numel()
                        _get_logger().error(
                            f"{node_name}/{output_name}: NaN/Inf detected!")
                        if exported_nan > 0:
                            _get_logger().error(
                                f"  Exported has {exported_nan}/{num_elements} NaN values ({100*exported_nan/num_elements:.3f}%)")
                        if exported_inf > 0:
                            _get_logger().error(
                                f"  Exported has {exported_inf}/{num_elements} Inf values ({100*exported_inf/num_elements:.3f}%)")
                        if source_nan > 0:
                            _get_logger().warning(
                                f"  Source has {source_nan}/{num_elements} NaN values ({100*source_nan/num_elements:.3f}%)")
                        if source_inf > 0:
                            _get_logger().warning(
                                f"  Source has {source_inf}/{num_elements} Inf values ({100*source_inf/num_elements:.3f}%)")
                        continue

                    if not torch.allclose(exported, source, rtol=rtol, atol=atol):
                        all_match = False
                        diff = (exported - source).abs()
                        diff_flat = diff.flatten().float()
                        
                        # Basic stats
                        max_diff = diff.max().item()
                        mean_diff = diff.mean().item()
                        
                        # Percentile differences
                        percentiles = torch.tensor([0.50, 0.75, 0.90, 0.99, 0.995], device=diff_flat.device)
                        pct_values = torch.quantile(diff_flat, percentiles)
                        p50, p75, p90, p99, p995 = pct_values.tolist()
                        
                        # Tensor value ranges
                        source_min, source_max = source.min().item(), source.max().item()
                        exported_min, exported_max = exported.min().item(), exported.max().item()
                        
                        log_path = _get_logger().path
                        _get_logger().error(
                            f"{node_name}/{output_name}: Mismatch detected (rtol={rtol}, atol={atol}). Please check {log_path} for more details.")
                        _get_logger().info(
                            f"  Source shape: {source.shape}, dtype: {source.dtype}")
                        _get_logger().info(
                            f"  Exported shape: {exported.shape}, dtype: {exported.dtype}")
                        _get_logger().info(
                            f"  Source range:   [{source_min:.6e}, {source_max:.6e}]")
                        _get_logger().info(
                            f"  Exported range: [{exported_min:.6e}, {exported_max:.6e}]")
                        _get_logger().info(
                            f"  Diff stats: max={max_diff:.6e}, mean={mean_diff:.6e}")
                        _get_logger().info(
                            f"  Diff percentiles: p50={p50:.6e}, p75={p75:.6e}, p90={p90:.6e}, p99={p99:.6e}, p995={p995:.6e}")

                results[node_name] = all_match

                if all_match:
                    _get_logger().info(f"  ✓ {node_name} passed validation")

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
            raise Exception(
                f"Model validation failed for {len(failed_nodes)} node(s): {failed_nodes}"
            )

        return results