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


def rewrite_aten_ops(graph: fx.Graph):
    """Replace low-level ``torch.ops.aten.*`` OpOverload targets with their
    high-level ``torch.*`` or ``torch.nn.functional.*`` equivalents.

    When a TorchScript model is decomposed via ``TS2EPConverter``, the
    resulting graph contains aten-level ops (e.g.
    ``torch.ops.aten.addmm.default``).  These work fine for tracing and
    ONNX export, but ``torch.jit.script`` cannot resolve them.  This
    pass rewrites them to the public API equivalents that TorchScript
    understands.
    """
    # -----------------------------------------------------------------
    # Ops that should become call_method nodes (tensor methods).
    # key = overloadpacket name, value = method name on the tensor.
    # -----------------------------------------------------------------
    _TO_METHOD = {
        "view": "view",
        "reshape": "reshape",
        "contiguous": "contiguous",
        "to": "to",
    }

    # -----------------------------------------------------------------
    # Ops with custom call_function replacements (different name or
    # needs arg rewriting).  Each value is (replacement_fn, arg_rewriter)
    # where arg_rewriter is None or a callable(args, kwargs)->(args, kwargs).
    # -----------------------------------------------------------------
    def _upsample_bilinear2d_args(args, kwargs):
        # aten sig: (input, output_size, align_corners)
        # F.interpolate sig: (input, size=, mode=, align_corners=)
        inp, size, align_corners = args[0], args[1], args[2]
        return (inp,), {
            "size": size,
            "mode": "bilinear",
            "align_corners": align_corners,
        }

    def _upsample_bicubic2d_args(args, kwargs):
        inp, size, align_corners = args[0], args[1], args[2]
        return (inp,), {
            "size": size,
            "mode": "bicubic",
            "align_corners": align_corners,
        }

    def _upsample_nearest2d_args(args, kwargs):
        inp, size = args[0], args[1]
        return (inp,), {"size": size, "mode": "nearest"}

    _CUSTOM_FN = {
        "upsample_bilinear2d": (F.interpolate, _upsample_bilinear2d_args),
        "upsample_bicubic2d": (F.interpolate, _upsample_bicubic2d_args),
        "upsample_nearest2d": (F.interpolate, _upsample_nearest2d_args),
    }

    # -----------------------------------------------------------------
    # Ops that can be handled by torch.narrow (slice with dim/start/end).
    # aten::slice.Tensor(self, dim, start, end, step=1)
    # Unit-step bounds must be normalized before lowering to torch.narrow.
    # Identity slices (start=0, end>=2^62) are replaced by the input.
    # -----------------------------------------------------------------
    _SLICE_OPS = {"slice", "slice_copy"}

    rewritten = []
    nodes_to_erase = []
    warp_op = warp_operator.get_op()

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        target = node.target
        if target is torch.sym_ite:
            node.target = scriptable_sym_ite
            continue
        if not isinstance(target, torch._ops.OpOverload):
            continue
        if target.overloadpacket is warp_op:
            continue
        op_name = target.overloadpacket.__name__

        if (
            op_name == "_to_copy"
            and node.kwargs.get("layout")
            in (
                None,
                torch.strided,
            )
            and not node.kwargs.get("pin_memory")
        ):
            node.op = "call_method"
            node.target = "to"
            node.kwargs = {
                key: value
                for key, value in node.kwargs.items()
                if key not in ("layout", "pin_memory") and value is not None
            }
            node.kwargs = {**node.kwargs, "copy": True}
            continue

        # --- Method conversion ---
        if op_name in _TO_METHOD:
            method_name = _TO_METHOD[op_name]
            node.op = "call_method"
            node.target = method_name
            # First arg is self; remaining stay as-is.
            rewritten.append(op_name)
            continue

        # --- Custom function replacement ---
        if op_name in _CUSTOM_FN:
            fn, arg_rewriter = _CUSTOM_FN[op_name]
            if arg_rewriter is not None:
                node.args, node.kwargs = arg_rewriter(node.args, node.kwargs)
            node.target = fn
            rewritten.append(op_name)
            continue

        # --- Slice handling ---
        # aten::slice.Tensor(self, dim, start, end, step=1)
        # Rewrite to torch.narrow for all backends:
        #   - jit-script can't resolve aten OpOverloads
        #   - onnx-dynamo produces bad ONNX Slice nodes for MAX_INT end
        # Slice bounds are normalized against the runtime dimension size.
        if op_name in _SLICE_OPS:
            tensor_arg = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else 0
            start = node.args[2] if len(node.args) > 2 else 0
            end = node.args[3] if len(node.args) > 3 else (1 << 62)
            step = node.args[4] if len(node.args) > 4 else 1
            is_end_max = end is None or (isinstance(end, int) and end >= (1 << 62))

            if isinstance(start, fx.Node) or isinstance(end, fx.Node):
                node.target = target.overloadpacket
                continue

            if op_name == "slice" and step == 1 and start in (None, 0) and is_end_max:
                # Identity slice — remove (no-op)
                node.replace_all_uses_with(tensor_arg)
                nodes_to_erase.append(node)
                rewritten.append(op_name)
                continue

            if step == 1:
                node.target = _slice_with_narrow
                node.args = (
                    tensor_arg,
                    dim,
                    start,
                    None if is_end_max else end,
                    op_name == "slice_copy",
                )
                node.kwargs = {}
            else:
                # TorchScript resolves overload packets, not .Tensor.
                node.target = target.overloadpacket
            rewritten.append(op_name)
            continue

        # --- Generic name-based lookup ---
        replacement = getattr(torch, op_name, None)
        if replacement is None or not callable(replacement):
            replacement = getattr(F, op_name, None)
        if replacement is not None and callable(replacement):
            node.target = replacement
            rewritten.append(op_name)
        else:
            node.target = target.overloadpacket

    # Erase identity-slice nodes (must be done after iteration)
    for node in nodes_to_erase:
        graph.erase_node(node)

    if rewritten:
        _get_logger().debug(
            f"Rewrote {len(rewritten)} aten OpOverload node(s) to "
            f"high-level equivalents: {rewritten}"
        )
    graph.lint()
