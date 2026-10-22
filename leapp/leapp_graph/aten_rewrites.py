#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Functionalization and ATen graph rewrites shared by LEAPP export backends."""

from typing import Optional

import torch
import torch.fx as fx
import torch.nn.functional as F
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

from leapp.leapp_graph.custom_operator_registry import warp_operator
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

    The caller applies LEAPP's pruning and capture-error checks first,
    preserving mutations until their alias effects have been normalized.
    """
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
            "native mutations on local copies. TorchScript remains available; "
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


def _normalize_slice_bound(bound: int, size: int) -> int:
    bound = bound + size if bound < 0 else bound
    if bound < 0:
        return 0
    return size if bound > size else bound


def _slice_with_narrow(
    tensor: torch.Tensor,
    dim: int,
    start: Optional[int],
    end: Optional[int],
    copy: bool,
) -> torch.Tensor:
    """Lower a unit-step slice while preserving runtime bounds and copy semantics."""
    # The legacy tracer represents size() as a Tensor; script the integer math
    # so tracing retains runtime sizes rather than freezing Python integers.
    if not torch.jit.is_scripting() and torch.jit.is_tracing():
        return torch.jit.script(_slice_with_narrow)(tensor, dim, start, end, copy)
    size = tensor.size(dim)
    start = _normalize_slice_bound(0 if start is None else start, size)
    end = size if end is None else _normalize_slice_bound(end, size)
    length = max(end - start, 0)
    result = torch.narrow(tensor, dim, start, length)
    return result.clone() if copy else result


def _rewrite_as_function(node: fx.Node, target):
    node.target = target


def _rewrite_as_method(node: fx.Node, method_name: str):
    node.op = "call_method"
    node.target = method_name


def _rewrite_to_copy(node: fx.Node):
    _rewrite_as_method(node, "to")
    node.kwargs = {
        key: value
        for key, value in node.kwargs.items()
        if key not in ("layout", "pin_memory") and value is not None
    }
    node.kwargs = {**node.kwargs, "copy": True}


def _rewrite_interpolation(node: fx.Node, mode: str, has_align_corners=False):
    kwargs = {"size": node.args[1], "mode": mode}
    if has_align_corners:
        kwargs["align_corners"] = node.args[2]
    node.args = (node.args[0],)
    node.kwargs = kwargs
    node.target = F.interpolate


def _rewrite_identity_slice(node: fx.Node):
    node.replace_all_uses_with(node.args[0])


def _rewrite_slice(node: fx.Node, dim, start, end, copy: bool):
    node.target = _slice_with_narrow
    node.args = (node.args[0], dim, start, end, copy)
    node.kwargs = {}


def rewrite_aten_ops(graph: fx.Graph):
    """Select and apply ATen rewrites for export compatibility.

    Keep operation identification and applicability checks in this traversal;
    the helpers perform the selected graph edits. Public functions and tensor
    methods replace OpOverloads that TorchScript cannot resolve, while slice
    rewrites also avoid ONNX Slice nodes with MAX_INT ends.
    """
    rewritten = []
    nodes_to_erase = []
    warp_op = warp_operator.get_op()

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        if target is torch.sym_ite:
            _rewrite_as_function(node, scriptable_sym_ite)
            continue
        if not isinstance(target, torch._ops.OpOverload):
            continue
        if target.overloadpacket is warp_op:
            continue
        op_name = target.overloadpacket.__name__

        if (
            op_name == "_to_copy"
            and node.kwargs.get("layout") in (None, torch.strided)
            and not node.kwargs.get("pin_memory")
        ):
            _rewrite_to_copy(node)
            continue

        if op_name in ("view", "reshape", "contiguous", "to"):
            _rewrite_as_method(node, op_name)
            rewritten.append(op_name)
            continue

        if op_name in ("upsample_bilinear2d", "upsample_bicubic2d"):
            mode = "bilinear" if op_name == "upsample_bilinear2d" else "bicubic"
            _rewrite_interpolation(node, mode, has_align_corners=True)
            rewritten.append(op_name)
            continue

        if op_name == "upsample_nearest2d":
            _rewrite_interpolation(node, "nearest")
            rewritten.append(op_name)
            continue

        if op_name in ("slice", "slice_copy"):
            dim = node.args[1] if len(node.args) > 1 else 0
            start = node.args[2] if len(node.args) > 2 else 0
            end = node.args[3] if len(node.args) > 3 else (1 << 62)
            step = node.args[4] if len(node.args) > 4 else 1
            is_end_max = end is None or (isinstance(end, int) and end >= (1 << 62))

            if isinstance(start, fx.Node) or isinstance(end, fx.Node):
                _rewrite_as_function(node, target.overloadpacket)
                continue

            if op_name == "slice" and step == 1 and start in (None, 0) and is_end_max:
                _rewrite_identity_slice(node)
                nodes_to_erase.append(node)
                rewritten.append(op_name)
                continue

            if step == 1:
                _rewrite_slice(
                    node,
                    dim,
                    start,
                    None if is_end_max else end,
                    op_name == "slice_copy",
                )
            else:
                _rewrite_as_function(node, target.overloadpacket)
            rewritten.append(op_name)
            continue

        replacement = getattr(torch, op_name, None)
        if replacement is None or not callable(replacement):
            replacement = getattr(F, op_name, None)
        if replacement is not None and callable(replacement):
            _rewrite_as_function(node, replacement)
            rewritten.append(op_name)
        else:
            _rewrite_as_function(node, target.overloadpacket)

    # Defer erasure until traversal finishes, keeping graph iteration stable.
    for node in nodes_to_erase:
        graph.erase_node(node)

    if rewritten:
        _get_logger().debug(
            f"Rewrote {len(rewritten)} aten OpOverload node(s) to "
            f"high-level equivalents: {rewritten}"
        )
    graph.lint()
