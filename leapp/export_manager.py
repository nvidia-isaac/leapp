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

import sys
import functools
import inspect
import yaml
import os
import time
from .node_context import NodeContext
from .utils import verify_data_exact_match, CompactYamlList, CompactYamlDict, get_relative_path


class ExportManager:
    _instance = None
    _initialized = False
    _is_tracing = False

    #########################################################
    # initialization
    #########################################################

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
            self.TAG_IO = True

            # tracetime settings
            self.intepret_graph = False

            # tracetime variables
            self.current_node_name = None
            self.nodes = {}

            ExportManager._initialized = True

    def start(self, name, save_path=".", tag_io=True):
        if self.intepret_graph:
            print("\033[1;33mWARNING: LEAPP graph interpretation is already enabled, \033[0m"
                  "\033[1;33mcalling start() again will reset the graph\033[0m")
            time.sleep(0.5)
            print("\033[1;33mResetting graph...\033[0m")
            self._is_tracing = False
            self.current_node_name = None
            time.sleep(0.5)
        self.nodes = {}
        self.intepret_graph = True
        self.GRAPH_NAME = name
        self.SAVE_PATH = os.path.join(save_path, self.GRAPH_NAME)
        self.TAG_IO = tag_io

    def stop(self):
        if ExportManager._is_tracing:
            raise Exception("ExportManager is currently tracing")
        if not self.intepret_graph:
            raise Exception("ExportManager graph interpretation is disabled")
        self.intepret_graph = False

    #########################################################
    # Tracing
    #########################################################

    def _trace_code_snippet(self, frame, event, arg):
        """Enhanced trace function that captures the line range of with block execution."""
        if not ExportManager._is_tracing or self.current_node_name is None:
            return self._trace_code_snippet

        # Skip tracing our own file
        code = frame.f_code
        if code.co_filename.split('/')[-1] == __file__.split('/')[-1]:
            return self._trace_code_snippet

        # Capture line events to determine the range of executed code
        if event == 'line':
            node_context = self.nodes[self.current_node_name]
            # Only track lines from the same file as the first line
            if node_context.executed_lines['filename'] == code.co_filename and node_context.executed_lines['function_name'] == code.co_name:
                node_context.executed_lines['lines'].add(frame.f_lineno)
                node_context.executed_lines['min_line'] = min(
                    node_context.executed_lines['min_line'], frame.f_lineno)
                node_context.snapshot_buffer_values(frame)
                node_context.executed_lines['max_line'] = max(
                    node_context.executed_lines['max_line'], frame.f_lineno)

        return self._trace_code_snippet

    def _trace_function(self, frame, event, arg):
        """Trace function that captures the function frame when called."""
        if not ExportManager._is_tracing or self.current_node_name is None:
            return self._trace_function

        if event == 'call':
            node_context = self.nodes[self.current_node_name]
            code = frame.f_code
            # Skip tracing our own file
            if code.co_filename.split('/')[-1] == __file__.split('/')[-1]:
                return self._trace_function

            # Save frame if function name matches current node name
            if (code.co_filename == node_context.executed_lines['filename'] and
                    node_context.executed_lines['min_line'] <= frame.f_lineno <= node_context.executed_lines['max_line']):
                # if code.co_name == self.current_node_name and node_context.executed_lines['filename'] == code.co_filename:
                node_context = self.nodes[self.current_node_name]
                if node_context.input_frame is None:
                    node_context.input_frame = frame  # we will only store input frame once
                    # store buffer values upon entering the function
                    node_context.snapshot_buffer_values(frame)
                # Keep on updating output frame
                node_context.output_frame = frame

        return self._trace_function

    def _setup_new_node_context(self, name, from_function, **kwargs):
        if self.current_node_name is not None:
            raise Exception(
                f"Error when attempting to set up new trace for {name}. \n"
                f"ExportManager is already tracing {self.current_node_name}")

        if name not in self.nodes:
            node_context = NodeContext(name, from_function,
                                       backend=kwargs.get("export_with", None),
                                       backend_params=kwargs.get(
                                           "backend_params", None),
                                       use_trace=kwargs.get(
                                           "use_trace", False),
                                       inputs=kwargs.get("inputs", None),
                                       outputs=kwargs.get("outputs", None),
                                       environment_constants=kwargs.get(
                                           "environment_constants", None),
                                       register_buffers=kwargs.get(
                                           "register_buffers", None),
                                       tag_io=self.TAG_IO,
                                       enable_fp16=kwargs.get(
                                           "enable_fp16", False),
                                       enable_cuda_graphs=kwargs.get("enable_cuda_graphs", False))
            # store the node context
            self.nodes[name] = node_context
            return True
        else:
            return False

    def _start_tracing(self, frame, trace_function):
        if self.current_node_name is None:
            raise Exception("Error: No node context found")

        if ExportManager._is_tracing:
            raise Exception("ExportManager is already tracing")
        ExportManager._is_tracing = True
        """Start the trace function."""
        frame.f_trace = trace_function
        sys.settrace(trace_function)

    def _stop_tracing(self, node_context):
        if ExportManager._is_tracing:
            """Stop the trace function."""
            sys.settrace(None)
            node_context.compile_trace()
            ExportManager._is_tracing = False
            self.current_node_name = None

    def block(self, node_name, **kwargs):
        """Create a context manager for tracing a block of code in the computational graph.

        Args:
            node_name: Name of the node to trace
            **kwargs: Additional parameters for node configuration

        Returns:
            Self for use as a context manager
        """
        if not self.intepret_graph:
            return self
        if self._setup_new_node_context(node_name, from_function=False, **kwargs):
            self.current_node_name = node_name
        return self

    # Keep export for backward compatibility
    def export(self, **params):
        """Deprecated: Use method() instead."""
        return self.method(**params)

    def __enter__(self):
        if not self.intepret_graph or self.current_node_name is None:
            return

        caller_frame = sys._getframe(1)  # Get the caller's frame
        """Enter context manager - called when entering 'with' block."""
        node_context = self.nodes[self.current_node_name]

        node_context.capture_inputs_from_frame(caller_frame)

        print(f"****Tracing started for {self.current_node_name}****")
        # CRITICAL: Set up local tracing for the caller frame

        if hasattr(caller_frame, 'f_trace_lines'):
            caller_frame.f_trace_lines = True
        node_context.executed_lines['filename'] = caller_frame.f_code.co_filename
        node_context.executed_lines['function_name'] = caller_frame.f_code.co_name
        node_context.executed_lines['min_line'] = caller_frame.f_lineno
        node_context.executed_lines['max_line'] = caller_frame.f_lineno

        self._start_tracing(caller_frame, self._trace_code_snippet)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.intepret_graph or self.current_node_name is None:
            return

        node_context = self.nodes[self.current_node_name]

        self._stop_tracing(node_context)

        node_context.capture_outputs_from_frame(sys._getframe(1))
        print(
            f"****Tracing stopped for {node_context.name}****\n\n")

    def method(self, **params):
        def decorator(func):

            if "node_name" in params:
                name = params["node_name"]
            else:
                name = func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.intepret_graph:
                    return func(*args, **kwargs)

                if not self._setup_new_node_context(name, from_function=True, **params):
                    return func(*args, **kwargs)

                print(f"****Tracing started for {name}****")
                node_context = self.nodes[name]
                node_context.inspect_function_inputs(
                    func, args, kwargs)
                self.current_node_name = name

                func_code = func.__code__
                node_context.executed_lines['filename'] = func_code.co_filename
                node_context.executed_lines['function_name'] = func_code.co_name

                # Get the line range of the function
                func_lines, start_line = inspect.getsourcelines(func)
                node_context.executed_lines['min_line'] = start_line
                node_context.executed_lines['max_line'] = start_line + \
                    len(func_lines) - 1

                # Initialize the lines set with all function lines
                node_context.executed_lines['lines'] = set(
                    range(start_line, start_line + len(func_lines)))

                self._start_tracing(sys._getframe(1), self._trace_function)

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    raise e
                finally:
                    self._stop_tracing(node_context)
                    print(
                        f"****Tracing stopped for {node_context.name}****\n\n")

                node_context.inspect_function_outputs(func, result)
                return result
            return wrapper
        return decorator

    #########################################################
    # Graph compilation
    #########################################################

    def connect_untagged_graph_connections(self):
        raise Exception("Untagged graph connections are not supported for now")
        print()
        print("\033[1mProcessing Node Connections Using Value Equivalence\033[0m")
        connections = {}

        for source_node in self.nodes.values():
            for target_node in self.nodes.values():
                for out_idx, output in enumerate(source_node.outputs):
                    output_value = output.value  # Access TensorDescription.value attribute
                    for in_idx, input in enumerate(target_node.inputs):
                        input_value = input.value  # Access TensorDescription.value attribute
                        if verify_data_exact_match(output_value,
                                                   input_value):
                            source_data = source_node.name + \
                                '/' + output.name_str  # Use name_str property
                            if source_data not in connections:
                                connections[source_data] = {
                                    'source': {'node': source_node, 'idx': out_idx},
                                    'targets': []
                                }
                            connections[source_data]['targets'].append(
                                {'node': target_node, 'idx': in_idx})
                            break

        total_connections = sum([len(connection['targets'])
                                for connection in connections.values()])
        print(
            f"\033[92mDiscovered {total_connections} internal connections\033[0m")
        print()
        return connections

    def connect_tagged_graph_connections(self):
        print()
        print("\033[1mProcessing Node Connections Using Tagged Values\033[0m")
        connections = {}
        for node in self.nodes.values():
            # first check if any duplicate tags. duplicates are not suppported
            tags = [input.tag for input in node.inputs if input.tag is not None]
            duplicates = set([tag for tag in tags if tags.count(tag) > 1])
            if duplicates:
                for duplicate in duplicates:
                    print(
                        f"found duplicate input with the tag {duplicate} in node {node.name}")
                raise Exception(
                    "Error: unsupported use of sending the same tensor multiple times to the same node")

            for in_idx, input in enumerate(node.inputs):
                if input.tag is None:  # case where the input is dangling
                    pass
                else:
                    source_node_name = input.tag.split('/')[0]
                    print(f"source node name: {source_node_name}")

                    source_node = self.nodes[source_node_name]
                    source_node_output_ports = [
                        output.tag for output in source_node.outputs]
                    if input.tag not in source_node_output_ports:
                        raise Exception(
                            f"Error: {source_node_name} does not produce tag {input.tag}")

                    out_idx = source_node_output_ports.index(input.tag)

                    if input.tag not in connections:
                        connections[input.tag] = {
                            'source': {'node': source_node, 'idx': out_idx},
                            'targets': []
                        }
                    connections[input.tag]['targets'].append(
                        {'node': node, 'idx': in_idx})

        return connections

    def reconcile_io_names(self, connections):
        names_changed = True
        print()
        print("\033[1mReconciling internal i/o names\033[0m")
        for connection in connections.values():
            source = connection['source']
            targets = connection['targets']

            # Use name_str property for TensorDescription objects
            target_names = [target['node'].inputs[target['idx']].name_str
                            for target in targets]
            desired_target_name = target_names[0]
            if not all(name == desired_target_name for name in target_names):
                names_changed = True
                for target in targets:
                    target['node'].change_input_name(
                        target['node'].inputs[target['idx']].name_str, desired_target_name)

            if not source['node'].outputs[source['idx']].name_str == desired_target_name:
                names_changed = True
                source['node'].change_output_name(
                    source['node'].outputs[source['idx']].name_str, desired_target_name)
        if names_changed:
            print("\033[93;1mWARNING: i/o names changed, this process edits the node specifications,"
                  " and may produce unexpected behavior. Please check the graph for correctness. \n"
                  "if this is not desired behavior, please make sure to match io names in the source code \033[0m")
        else:
            print("\033[92mno names changed\033[0m")

        print()

    def compile_graph_io(self, connections):
        # any inputs and outputs that are not connected to any nodes are outside connections
        graph_inputs = []
        graph_outputs = []

        print()
        print("\033[1mDiscovering graph inputs and outputs\033[0m")

        # Collect all target connections (destinations)
        all_targets = []
        for targets_list in connections.values():
            all_targets.extend(targets_list)

        for node in self.nodes.values():
            # An input is dangling if it's not the target of any internal connection
            for input_desc in node.inputs:
                input_name = input_desc.name_str  # Use name_str property
                node_input = node.name + '/' + input_name
                if node_input not in all_targets:
                    graph_inputs.append(node_input)

            # An output is dangling if it's not the source of any internal connection
            for output_desc in node.outputs:
                output_name = output_desc.name_str  # Use name_str property
                node_output = node.name + '/' + output_name
                if node_output not in connections:
                    graph_outputs.append(node_output)

        print(f"\033[92mDiscovered {len(graph_inputs)} graph inputs\033[0m")
        print(f"\033[92mDiscovered {len(graph_outputs)} graph outputs\033[0m")
        print()
        return graph_inputs, graph_outputs

    def process_graph_connections(self):
        if self.TAG_IO:
            connections = self.connect_tagged_graph_connections()
        else:
            connections = self.connect_untagged_graph_connections()

        self.reconcile_io_names(connections)

        processed_connections = {}
        for connection in connections.values():
            source = connection['source']
            targets = connection['targets']
            source_port = source['node'].name + '/' + \
                source['node'].outputs[source['idx']].name_str
            target_ports = CompactYamlList()
            for target in targets:
                target_ports.append(target['node'].name + '/' +
                                    target['node'].inputs[target['idx']].name_str)

            processed_connections[source_port] = target_ports

        graph_inputs, graph_outputs = self.compile_graph_io(
            processed_connections)

        return processed_connections, graph_inputs, graph_outputs

    def compile_models(self):
        if self.SAVE_PATH is None:
            raise Exception(
                "Error: No save path provided, please provide a save path to export the graph")
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)

        print(f"\n\033[1mDiscovered {len(self.nodes)} nodes\033[0m\n")
        for node_context in self.nodes.values():
            print(f"\033[1mCompiling {node_context.name}\033[0m")
            node_context.export_model(self.SAVE_PATH)
            print("\033[92mSuccess\033[0m\n")
        print()

    def get_io_descriptions(self):
        print(
            f"\n\033[1mCompiling graph parameters for {len(self.nodes)} nodes\033[0m")
        models = {"models": {}}
        for node in self.nodes.values():
            print(f"Compiling parameters for {node.name}")
            description = node.get_description(
                ['inputs', 'outputs', 'parameters', 'input_format', 'output_format'])
            if 'parameters' in description and 'model_path' in description['parameters']:
                # Convert model path to be relative to YAML file location
                model_path = description['parameters']['model_path']
                if model_path:
                    description['parameters']['model_path'] = get_relative_path(
                        model_path, self.SAVE_PATH)
            models["models"][node.name] = description
        print()
        return models

    def compile_graph(self, visualize=True):
        # compile models first before input name reconciliation
        self.compile_models()
        # then process graph connections. this process may change input and output names
        connections, dangling_inputs, dangling_outputs = self.process_graph_connections()
        # then get io description, this finalizes the input and output names
        models = self.get_io_descriptions()

        if visualize:
            try:
                from .graph_gui import visualize_graph
                visualize_graph(self.nodes, connections,
                                dangling_inputs, dangling_outputs, self.SAVE_PATH, self.GRAPH_NAME)
            except Exception as e:
                print(f"Error visualizing graph: {e}")

        internal_connections = sum([len(connection)
                                   for connection in connections.values()])
        # Print graph statistics
        print(f"\nGraph Statistics:")
        print(f"- Computation nodes: {len(self.nodes)}")
        print(f"- Dangling inputs: {len(dangling_inputs)}")
        print(f"- Dangling outputs: {len(dangling_outputs)}")
        print(f"- Internal connections: {internal_connections}")
        total_edges = internal_connections + \
            len(dangling_inputs) + len(dangling_outputs)
        print(f"- Total edges: {total_edges}")

        # Set up custom YAML representers before writing any YAML
        def represent_shape_list(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

        def represent_shape_dict(dumper, data):
            return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)

        # Register the custom representers
        yaml.add_representer(CompactYamlList, represent_shape_list)
        yaml.add_representer(CompactYamlDict, represent_shape_dict)
        # Convert dangling inputs and outputs from "node_name/field" format to {"node_name": [field1, field2]} format
        dangling_inputs_dict = {}
        for inp in dangling_inputs:
            if '/' in inp:
                node_name, field_name = inp.split('/', 1)
                if node_name not in dangling_inputs_dict:
                    dangling_inputs_dict[node_name] = CompactYamlList()
                dangling_inputs_dict[node_name].append(field_name)

        dangling_outputs_dict = {}
        for out in dangling_outputs:
            if '/' in out:
                node_name, field_name = out.split('/', 1)
                if node_name not in dangling_outputs_dict:
                    dangling_outputs_dict[node_name] = CompactYamlList()
                dangling_outputs_dict[node_name].append(field_name)

        pipeline = {'pipeline': {'data_flow': connections,
                                 'dangling_inputs': dangling_inputs_dict,
                                 'dangling_outputs': dangling_outputs_dict}}
        with open(os.path.join(self.SAVE_PATH, f"{self.GRAPH_NAME}.yaml"), "w") as f:
            yaml.dump(models, f)
            f.write("\n")  # Add a newline separator
            yaml.dump(pipeline, f)
            f.write("\n")
