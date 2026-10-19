#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Normalize captured mutation without exporting caller-input side effects."""

import torch
from torch import fx
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

from leapp.utils.logging import _get_logger


def scriptable_sym_ite(condition: bool, if_true: int, if_false: int) -> int:
    """Keep symbolic slice arithmetic exportable and TorchScript-compatible."""
    if torch.jit.is_scripting():
        return if_true if condition else if_false
    return torch.sym_ite(condition, if_true, if_false)


def functionalize_graph(graph, root, inputs):
    """Import the functional user-output graph, retaining the recording graph ID.

    Use the ExportedProgram signature instead of ``module()``: the latter adds
    input-copyback operations to reproduce eager caller mutation. LEAPP exposes
    updates through declared outputs/state, so only USER_OUTPUT values belong
    in its artifact. This also handles an output which is the updated input.
    """
    # Drop orphan metadata reads and diagnostics, but never prune a recorded
    # mutation by data dependencies before its alias effects are normalized.
    graph.eliminate_dead_code(
        is_impure_node=lambda node: (
            node.is_impure() or node.meta.get("leapp_mutation", False)
        )
    )
    original_inputs = [node.target for node in graph.nodes if node.op == "placeholder"]
    raw = fx.GraphModule(root, graph)
    with torch.inference_mode(False):
        samples = tuple(value.detach().clone() for value in inputs)
    try:
        program = torch.export.export(raw, samples, strict=False).run_decompositions({})
    except GuardOnDataDependentSymNode:
        # Older exporters cannot normalize tensor-valued slice bounds. Retain
        # ordered native mutations for FX/TorchScript, cloning inputs/constants
        # per call to keep LEAPP's functional external contract. Never replace
        # runtime indices with their capture-time values.
        constants = dict(raw.named_buffers()) | dict(raw.named_parameters())
        for node in list(graph.nodes):
            if node.op == "placeholder" or (
                node.op == "get_attr" and node.target in constants
            ):
                with graph.inserting_after(node):
                    clone = graph.call_function(torch.clone, (node,))
                node.replace_all_uses_with(clone)
                clone.args = (node,)
        graph.preserve_mutations = True
        _get_logger().warning(
            "PyTorch could not functionalize data-dependent indices; retaining "
            "native mutations on local copies. FX/TorchScript remain available; "
            "PT2/ONNX export is subject to the exporter's index restrictions."
        )
        return raw
    normalized = program.graph_module.graph
    signature = program.graph_signature
    input_specs = iter(signature.input_specs)
    user_inputs = iter(original_inputs)
    mapping = {}
    module = torch.nn.Module()

    # Keep this graph object's identity: published carriers still identify the
    # captured pass through their old proxy's graph. A later capture reset must
    # still invalidate them, whereas compilation must preserve boundary ports.
    for node in reversed(list(graph.nodes)):
        graph.erase_node(node)

    for node in normalized.nodes:
        if node.op == "placeholder":
            spec = next(input_specs)
            if spec.kind == InputKind.USER_INPUT:
                mapping[node] = graph.placeholder(next(user_inputs))
            else:
                if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
                    value = program.state_dict[spec.target]
                elif spec.kind in (InputKind.CONSTANT_TENSOR, InputKind.CUSTOM_OBJ):
                    value = program.constants[spec.target]
                else:
                    raise NotImplementedError(
                        f"LEAPP cannot functionalize input kind {spec.kind}"
                    )
                name = f"_leapp_lifted_{len(mapping)}"
                if isinstance(value, torch.Tensor):
                    module.register_buffer(name, value.detach().clone())
                else:
                    setattr(module, name, value)
                mapping[node] = graph.get_attr(name)
        elif node.op == "output":
            values = [
                mapping[value]
                for value, spec in zip(node.args[0], signature.output_specs)
                if spec.kind == OutputKind.USER_OUTPUT
            ]
            graph.output(values[0] if len(values) == 1 else tuple(values))
        else:
            mapping[node] = graph.node_copy(node, mapping.__getitem__)

    return module
