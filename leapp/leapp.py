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

"""Public LEAPP API surface split between runtime and annotation APIs."""

from __future__ import annotations

import os
import yaml

from .utils.logging import _get_logger
from .leapp_graph.datatypes import apply_traced_data_patches, remove_traced_data_patches
from .leapp_graph.leapp_graph import LeappGraph
from .utils.tracing_lock import TracingLock
from .utils.utils import get_system_info
from .export_manager import ExportManager


_MANAGER = ExportManager()


def start(name, save_path=".", verbose=False, dry_run=False, non_traced=None, max_cached_io=5, global_patching=True):
    """Initialize and start LEAPP graph interpretation.

    ``name`` may be a bare graph name (``"my_graph"``), a relative path
    (``"foo/bar"``), or an absolute path (``"/tmp/my_graph"``). LEAPP always
    uses the trailing path component as the graph name for emitted artifacts
    (``<graph_name>.yaml``, ``<graph_name>.svg``, ``<graph_name>.png``,
    ``<graph_name>_initial_values.safetensors``)
    and resolves the final output directory as ``save_path / dirname(name) / basename(name)``.
    An absolute ``name`` overrides ``save_path`` (mirroring ``os.path.join`` semantics).
    """
    if not isinstance(name, str):
        raise TypeError(f"leapp.start(name=...) must be a string, got {type(name).__name__}")
    normalized_name = os.path.normpath(os.path.expanduser(name))
    if normalized_name in ("", ".", os.sep):
        raise ValueError(
            f"leapp.start(name={name!r}) does not contain a usable graph name; "
            "provide a non-empty basename such as 'my_graph' or '/tmp/my_graph'."
        )
    name_dir, graph_name = os.path.split(normalized_name)
    if not graph_name:
        raise ValueError(
            f"leapp.start(name={name!r}) resolved to an empty graph name basename; "
            "provide a name whose basename is non-empty (e.g. 'my_graph')."
        )

    manager = _MANAGER
    manager.set_graph_name(graph_name)
    manager.set_save_path(os.path.join(save_path, name_dir, graph_name))
    manager.ensure_save_path_exists()
    manager.configure_logger(verbose=verbose)

    if ExportManager.is_interpret_graph_enabled():
        _get_logger().warning("LEAPP graph interpretation is already enabled, "
                              "calling start() again will reset the graph")
        _get_logger().warning("Resetting graph...")
        manager.reset_tracing_lock()

    if dry_run:
        _get_logger().info("Starting dry run mode")

    if non_traced is None:
        non_traced = []
    manager.set_dry_run_and_non_traced(dry_run, non_traced)
    manager.set_max_cached_io(max_cached_io)
    manager.reset_nodes()
    ExportManager.set_interpret_graph(True)

    # Apply patches for torch functions that bypass __torch_function__
    if global_patching:
        apply_traced_data_patches()
    manager.set_patches_applied(global_patching)


def stop():
    """Stop LEAPP graph interpretation and disable tracing."""
    manager = _MANAGER

    if TracingLock().is_active:
        raise Exception("leapp.stop() was called while a traced function is still executing")
    if not ExportManager.is_interpret_graph_enabled():
        raise Exception("leapp.stop() called with no active tracing session — did you call leapp.start()?")

    ExportManager.set_interpret_graph(False)
    manager.restore_pending_buffer_trackers()

    # Remove patches to restore original torch function behavior
    if manager.is_numpy_patches_applied():
        remove_traced_data_patches()
        manager.set_patches_applied(False)


def compile_graph(visualize=True, verbose=None, validate=True, dry_run=False, rtol=1e-3, atol=1e-5, strict=True):
    """Compile and save the computational graph from traced nodes.

    When ``visualize`` is ``True`` on Python 3.11 or later, LEAPP writes
    static ``.svg`` and ``.png`` graph artifacts to the graph output
    directory. Earlier Python versions emit a warning and skip visualization.
    """
    manager = _MANAGER

    # Enforce lifecycle: compile only after tracing is stopped.
    if ExportManager.is_interpret_graph_enabled():
        raise Exception("LEAPP graph interpretation is enabled. Call leapp.stop() before leapp.compile_graph().")

    if verbose is not None:
        _get_logger().set_verbose(verbose)

    if dry_run:
        manager.set_dry_run_and_non_traced(dry_run, [])

    manager.validate_nodes_ready_for_compile()

    if not manager.is_dry_run():
        manager.compile_models()

    graph = LeappGraph(manager.get_nodes(), manager.get_graph_name())
    pipeline = graph.get_full_pipeline_description()

    initial_value_filename = None
    if not manager.is_dry_run():
        initial_value_filename = graph.save_feedback_initial_values(
            manager.get_save_path(), manager.get_graph_name())

    if initial_value_filename is not None:
        pipeline['pipeline']['initial_values'] = initial_value_filename

    if not manager.is_dry_run():
        manager.save_models()

    models = manager.get_io_descriptions()

    if visualize:
        graph.visualize(manager.get_save_path(), manager.get_graph_name())

    internal_connections, total_edges = graph.get_graph_statistics()

    _get_logger().section("Graph Statistics")
    _get_logger().info(f"- Computation nodes: {len(manager.get_nodes())}")
    _get_logger().info(f"- Dangling inputs: {len(graph.graph_inputs)}")
    _get_logger().info(f"- Dangling outputs: {len(graph.graph_outputs)}")
    _get_logger().info(f"- Internal connections: {internal_connections}")
    _get_logger().info(f"- Total edges: {total_edges}")

    system_info = get_system_info()
    with open(os.path.join(manager.get_save_path(), f"{manager.get_graph_name()}.yaml"), "w") as f:
        yaml.dump(models, f, sort_keys=False)
        f.write("\n")
        yaml.dump(pipeline, f)
        f.write("\n")
        yaml.dump(system_info, f)
        f.write("\n")

    if not manager.is_dry_run():
        _get_logger().log_export_artifact_locations(
            manager.get_save_path(),
            manager.get_graph_name(),
            manager.get_nodes().values(),
        )

    manager.set_detected_graph(models, pipeline)

    if validate and not manager.is_dry_run():
        return manager.validate_all_models(rtol=rtol, atol=atol, strict=strict)

    return True


class AnnotateAPI:
    """Annotation-only facade over ExportManager."""

    _ALLOWED_APIS = {
        "input_tensors",
        "output_tensors",
        "method",
        "_method",
        "state_tensors",
        "update_state",
        "register_buffer",
        "module",
        "mirror_leapp_tags",
    }

    def __getattr__(self, name):
        if name in self._ALLOWED_APIS:
            return getattr(_MANAGER, name)
        raise AttributeError(
            f"leapp.annotate has no attribute '{name}'. "
        )


annotate = AnnotateAPI()
