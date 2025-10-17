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
from .utils import CompactYamlList, CompactYamlDict, get_relative_path, get_system_info
from .logging import LeappLogger
from .leapp_graph import LeappGraph


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

            # tracetime settings
            self.intepret_graph = False

            # tracetime variables
            self.current_node_name = None
            self.node_candidate = None
            self.nodes = {}

            # logging
            self.logger = LeappLogger(self)

            # Set up custom YAML representers before writing any YAML
            def represent_shape_list(dumper, data):
                return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

            def represent_shape_dict(dumper, data):
                return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)

            # Register the custom representers
            yaml.add_representer(CompactYamlList, represent_shape_list)
            yaml.add_representer(CompactYamlDict, represent_shape_dict)

            ExportManager._initialized = True

    def start(self, name, save_path=".", verbose=False):
        self.GRAPH_NAME = name
        self.SAVE_PATH = os.path.join(save_path, self.GRAPH_NAME)
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        self.logger.configure_logger(self.SAVE_PATH, verbose=verbose)
        if self.intepret_graph:
            self.logger.warning("LEAPP graph interpretation is already enabled, "
                                "calling start() again will reset the graph")
            time.sleep(0.5)
            self.logger.warning("Resetting graph...")
            self._is_tracing = False
            self.current_node_name = None
            self.node_candidate = None
            time.sleep(0.5)
        self.nodes = {}
        self.intepret_graph = True

    def stop(self):
        if ExportManager._is_tracing:
            raise Exception("ExportManager is currently tracing")
            ExportManager._is_tracing = False
        if not self.intepret_graph:
            raise Exception("ExportManager graph interpretation is disabled")
        self.intepret_graph = False

    #########################################################
    # node context setup
    #########################################################
    def _setup_new_node_context(self, name, from_function, **kwargs):
        if self.current_node_name is not None or self.node_candidate is not None:
            self.current_node_name = name
            if self.node_candidate.name is not None:
                name = self.node_candidate.name
            raise Exception(
                f"Error when attempting to set up new trace for {name}. \n"
                f"ExportManager is already tracing {self.current_node_name}")
        if name in self.nodes.keys():
            # retracing inherits the node index of the original node
            node_index = self.nodes[name].node_index
        else:
            node_index = len(self.nodes)
        self.node_candidate = NodeContext(name, node_index, from_function,
                                          logger=self.logger,
                                          backend=kwargs.get(
                                              "export_with", None),
                                          backend_params=kwargs.get(
                                              "backend_params", None),
                                          use_trace=kwargs.get(
                                              "use_trace", False),
                                          inputs=kwargs.get(
                                              "inputs", None),
                                          outputs=kwargs.get(
                                              "outputs", None),
                                          environment_constants=kwargs.get(
                                              "environment_constants", None),
                                          register_buffers=kwargs.get(
                                              "register_buffers", None),
                                          enable_fp16=kwargs.get(
                                              "enable_fp16", False),
                                          enable_cuda_graphs=kwargs.get("enable_cuda_graphs", False))
        self.current_node_name = name

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
            # Only track lines from the same file as the first line
            if self.node_candidate.executed_lines['filename'] == code.co_filename and self.node_candidate.executed_lines['function_name'] == code.co_name:
                self.node_candidate.executed_lines['lines'].add(frame.f_lineno)
                self.node_candidate.executed_lines['min_line'] = min(
                    self.node_candidate.executed_lines['min_line'], frame.f_lineno)
                self.node_candidate.snapshot_buffer_values(frame)
                self.node_candidate.executed_lines['max_line'] = max(
                    self.node_candidate.executed_lines['max_line'], frame.f_lineno)

        return self._trace_code_snippet

    def _trace_function(self, frame, event, arg):
        """Trace function that captures the function frame when called."""
        if not ExportManager._is_tracing or self.current_node_name is None:
            return self._trace_function

        if event == 'call':
            code = frame.f_code
            # Skip tracing our own file
            if code.co_filename.split('/')[-1] == __file__.split('/')[-1]:
                return self._trace_function

            # Save frame if function name matches current node name
            if (code.co_filename == self.node_candidate.executed_lines['filename'] and
                    self.node_candidate.executed_lines['min_line'] <= frame.f_lineno <= self.node_candidate.executed_lines['max_line']):
                # if code.co_name == self.current_node_name and node_context.executed_lines['filename'] == code.co_filename:
                if self.node_candidate.input_frame is None:
                    self.node_candidate.input_frame = frame  # we will only store input frame once
                    # store buffer values upon entering the function
                    self.node_candidate.snapshot_buffer_values(frame)
                # Keep on updating output frame
                self.node_candidate.output_frame = frame

        return self._trace_function

    def _start_tracing(self, frame, trace_function):
        if self.current_node_name is None:
            raise Exception("Error: No node context found")

        if ExportManager._is_tracing:
            raise Exception("ExportManager is already tracing")
        ExportManager._is_tracing = True
        """Start the trace function."""
        frame.f_trace = trace_function
        sys.settrace(trace_function)

    def _stop_tracing(self, node_name, node_context):
        if ExportManager._is_tracing:
            """Stop the trace function."""
            sys.settrace(None)

            try:
                if node_name in self.nodes.keys():
                    # we have seen this node before, we need to check if all values match.
                    original_node_data = {
                        k: v for k, v in self.nodes[node_name].executed_lines.items() if k != 'source_code'}
                    new_node_data = {
                        k: v for k, v in node_context.executed_lines.items() if k != 'source_code'}

                    if original_node_data != new_node_data:
                        raise Exception(
                            f"Error: {node_name} seen twice but detected lines do not match\n"
                            f"Original node data: {original_node_data}\n"
                            f"New node data: {new_node_data}")

                self.nodes[node_name] = node_context
                self.nodes[node_name].compile_trace()

            finally:
                # Always reset tracing state, even if an exception occurred
                ExportManager._is_tracing = False
                # reset the current node name and node candidate
                self.current_node_name = None
                self.node_candidate = None

    #########################################################
    # annotation APIs
    #########################################################
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
        self._setup_new_node_context(node_name, from_function=False, **kwargs)
        return self

    def __enter__(self):
        """Enter context manager - called when entering 'with' block."""
        if not self.intepret_graph:
            return
        if self.current_node_name is None or self.node_candidate is None:
            raise Exception(
                "Unexpected error when setting up new node context, current_node_name or node_candidate is None")

        caller_frame = sys._getframe(1)  # Get the caller's frame

        self.node_candidate.capture_inputs_from_frame(caller_frame)

        self.logger.info(
            f"****Tracing started for {self.current_node_name}****")
        # CRITICAL: Set up local tracing for the caller frame

        if hasattr(caller_frame, 'f_trace_lines'):
            caller_frame.f_trace_lines = True
        self.node_candidate.executed_lines['filename'] = caller_frame.f_code.co_filename
        self.node_candidate.executed_lines['function_name'] = caller_frame.f_code.co_name
        self.node_candidate.executed_lines['min_line'] = caller_frame.f_lineno
        self.node_candidate.executed_lines['max_line'] = caller_frame.f_lineno

        self._start_tracing(caller_frame, self._trace_code_snippet)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.intepret_graph:
            return

        if self.current_node_name is None or self.node_candidate is None:
            raise Exception(
                "Unexpected error when completing tracing, current_node_name or node_candidate is None")

        name = self.current_node_name

        self._stop_tracing(self.current_node_name, self.node_candidate)

        node_context = self.nodes[name]

        node_context.capture_outputs_from_frame(sys._getframe(1))
        self.logger.info(
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

                self._setup_new_node_context(
                    name, from_function=True, **params)
                if self.current_node_name is None or self.node_candidate is None:
                    raise Exception(
                        "Unexpected error when setting up new node context, current_node_name or node_candidate is None")

                self.logger.info(f"****Tracing started for {name}****")
                self.node_candidate.inspect_function_inputs(
                    func, args, kwargs)

                func_code = func.__code__
                self.node_candidate.executed_lines['filename'] = func_code.co_filename
                self.node_candidate.executed_lines['function_name'] = func_code.co_name

                # Get the line range of the function
                func_lines, start_line = inspect.getsourcelines(func)
                self.node_candidate.executed_lines['min_line'] = start_line
                self.node_candidate.executed_lines['max_line'] = start_line + \
                    len(func_lines) - 1

                # Initialize the lines set with all function lines
                self.node_candidate.executed_lines['lines'] = set(
                    range(start_line, start_line + len(func_lines)))

                self._start_tracing(sys._getframe(1), self._trace_function)

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    raise e
                finally:
                    self._stop_tracing(
                        self.current_node_name, self.node_candidate)
                    if name not in self.nodes.keys():
                        raise Exception(
                            f"Error: expected node {name} to be in nodes to be in nodes dictionary")
                    node_context = self.nodes[name]
                    self.logger.info(
                        f"****Tracing stopped for {node_context.name}****\n\n")

                node_context.inspect_function_outputs(func, result)
                return result
            return wrapper
        return decorator

    def get_io_descriptions(self):
        self.logger.section(
            f"Compiling graph parameters for {len(self.nodes)} nodes")
        models = {"models": {}}
        for node in self.nodes.values():
            self.logger.info(f"Compiling parameters for {node.name}")
            description = node.get_description(
                ['inputs', 'outputs', 'parameters', 'input_format', 'output_format'])
            if 'parameters' in description and 'model_path' in description['parameters']:
                # Convert model path to be relative to YAML file location
                model_path = description['parameters']['model_path']
                if model_path:
                    description['parameters']['model_path'] = get_relative_path(
                        model_path, self.SAVE_PATH)
            models["models"][node.name] = description
        return models

    def compile_models(self):
        if self.SAVE_PATH is None:
            raise Exception(
                "Error: No save path provided, please provide a save path to export the graph")
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)

        self.logger.section(f"Discovered {len(self.nodes)} nodes")
        for node_context in self.nodes.values():
            self.logger.section(f"Compiling {node_context.name}")
            node_context.export_model(self.SAVE_PATH)
            self.logger.info("Success\n")

    #########################################################
    # graph compilation
    #########################################################
    def compile_graph(self, visualize=True):
        # compile models first before input name reconciliation
        self.compile_models()

        # builds the graph connections. this may change input and output names
        graph = LeappGraph(self.logger, self.nodes)
        pipeline = graph.get_full_pipeline_description()

        models = self.get_io_descriptions()

        if visualize:
            try:
                graph.visualize(self.SAVE_PATH, self.GRAPH_NAME)
            except Exception as e:
                self.logger.error(f"Error visualizing graph: {e}")

        internal_connections, total_edges = graph.get_graph_statistics()

        # Print graph statistics
        self.logger.section("Graph Statistics")
        self.logger.info(f"- Computation nodes: {len(self.nodes)}")
        self.logger.info(f"- Dangling inputs: {len(graph.graph_inputs)}")
        self.logger.info(f"- Dangling outputs: {len(graph.graph_outputs)}")
        self.logger.info(f"- Internal connections: {internal_connections}")
        self.logger.info(f"- Total edges: {total_edges}")

        system_info = get_system_info()
        with open(os.path.join(self.SAVE_PATH, f"{self.GRAPH_NAME}.yaml"), "w") as f:
            yaml.dump(models, f)
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
