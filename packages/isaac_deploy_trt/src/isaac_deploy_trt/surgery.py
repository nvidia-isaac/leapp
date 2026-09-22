#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""ONNX graph surgery helpers (GraphSurgeon-style edit API).

Thin wrappers around ``onnx_graphsurgeon`` for common edits used when preparing
models for TensorRT: insert / remove / swap / rename nodes and tensors, and
replace a subgraph with a single node (e.g. a TRT plugin).

Requires the optional ``onnx-graphsurgeon`` package::

    pip install onnx-graphsurgeon
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import onnx
from onnx import ModelProto

import logging

GraphLike = Any  # onnx_graphsurgeon.Graph
NodeLike = Any  # onnx_graphsurgeon.Node
TensorLike = Any  # onnx_graphsurgeon.Tensor


def _gs():
    try:
        import onnx_graphsurgeon as gs
    except ImportError as exc:
        raise ImportError(
            "onnx-graphsurgeon is required for isaac_deploy_trt. "
            "Install with: pip install onnx-graphsurgeon"
        ) from exc
    return gs


def load_graph(model_or_path: Union[str, ModelProto]) -> GraphLike:
    """Load an ONNX model path or ``ModelProto`` into a GraphSurgeon graph."""
    gs = _gs()
    if isinstance(model_or_path, str):
        model = onnx.load(model_or_path, load_external_data=False)
    else:
        model = model_or_path
    return gs.import_onnx(model)


def export_model(graph: GraphLike, *, do_cleanup: bool = True) -> ModelProto:
    """Export a GraphSurgeon graph to an ONNX ``ModelProto``."""
    gs = _gs()
    if do_cleanup:
        graph.cleanup().toposort()
    return gs.export_onnx(graph)


def save_graph(
    graph: GraphLike,
    path: str,
    *,
    do_cleanup: bool = True,
) -> str:
    """Cleanup/toposort (optional) and save the graph to ``path``."""
    model = export_model(graph, do_cleanup=do_cleanup)
    onnx.save(model, path)
    return path


def cleanup_graph(graph: GraphLike) -> GraphLike:
    """Remove dangling nodes/tensors and restore topological order."""
    return graph.cleanup().toposort()


def tensor_map(graph: GraphLike) -> Dict[str, TensorLike]:
    """Return ``{tensor_name: tensor}`` for the graph."""
    return graph.tensors()


def find_nodes(
    graph: GraphLike,
    *,
    name: Optional[str] = None,
    op: Optional[str] = None,
    predicate: Optional[Callable[[NodeLike], bool]] = None,
) -> List[NodeLike]:
    """Find nodes by exact name, op type, and/or predicate."""
    matches: List[NodeLike] = []
    for node in graph.nodes:
        if name is not None and node.name != name:
            continue
        if op is not None and node.op != op:
            continue
        if predicate is not None and not predicate(node):
            continue
        matches.append(node)
    return matches


def get_node(graph: GraphLike, name: str) -> NodeLike:
    """Return the unique node with ``name`` or raise ``KeyError``."""
    nodes = find_nodes(graph, name=name)
    if not nodes:
        raise KeyError(f"No node named {name!r}")
    if len(nodes) > 1:
        raise KeyError(f"Multiple nodes named {name!r}")
    return nodes[0]


def insert_node(
    graph: GraphLike,
    op: str,
    inputs: Sequence[Union[str, TensorLike]],
    outputs: Optional[Sequence[Union[str, TensorLike]]] = None,
    *,
    attrs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
    output_dtype=None,
    output_shape=None,
) -> NodeLike:
    """Insert a node into ``graph``.

    ``inputs`` / ``outputs`` may be tensor names or GraphSurgeon tensors.
    If ``outputs`` is omitted, a single new variable tensor is created.
    """
    gs = _gs()
    tmap = graph.tensors()

    def _as_tensor(value: Union[str, TensorLike], *, create: bool) -> TensorLike:
        if not isinstance(value, str):
            return value
        if value in tmap:
            return tmap[value]
        if not create:
            raise KeyError(f"Tensor {value!r} not found in graph")
        return gs.Variable(
            name=value,
            dtype=output_dtype,
            shape=output_shape,
        )

    input_tensors = [_as_tensor(v, create=False) for v in inputs]
    if outputs is None:
        out_name = f"{name or op}_out"
        output_tensors = [
            gs.Variable(name=out_name, dtype=output_dtype, shape=output_shape)
        ]
    else:
        output_tensors = [_as_tensor(v, create=True) for v in outputs]

    node = gs.Node(
        op=op,
        name=name or "",
        attrs=dict(attrs or {}),
        inputs=list(input_tensors),
        outputs=list(output_tensors),
    )
    graph.nodes.append(node)
    return node


def remove_nodes(
    graph: GraphLike,
    *,
    names: Optional[Iterable[str]] = None,
    ops: Optional[Iterable[str]] = None,
    predicate: Optional[Callable[[NodeLike], bool]] = None,
    reconnect: bool = False,
) -> int:
    """Remove matching nodes from ``graph``.

    Args:
        names: Exact node names to remove.
        ops: Op types to remove.
        predicate: Extra filter; node is removed if it matches any provided
            selector and passes the predicate (if given).
        reconnect: If True, and a removed node is 1-in/1-out, rewire consumers
            of its output to its input (Identity-style bypass). Otherwise
            dangling edges are left for ``cleanup_graph``.

    Returns:
        Number of nodes removed.
    """
    name_set = set(names) if names is not None else None
    op_set = set(ops) if ops is not None else None

    def _should_remove(node: NodeLike) -> bool:
        matched = False
        if name_set is not None and node.name in name_set:
            matched = True
        if op_set is not None and node.op in op_set:
            matched = True
        if name_set is None and op_set is None:
            matched = True
        if not matched:
            return False
        if predicate is not None and not predicate(node):
            return False
        return True

    to_remove = [node for node in graph.nodes if _should_remove(node)]
    for node in to_remove:
        if reconnect and len(node.inputs) == 1 and len(node.outputs) == 1:
            inp = node.inputs[0]
            out = node.outputs[0]
            # Rewire consumers of ``out`` to use ``inp``.
            for consumer in list(out.outputs):
                consumer.inputs = [inp if t is out else t for t in consumer.inputs]
            # Update graph I/O if needed.
            graph.outputs = [inp if t is out else t for t in graph.outputs]
            out.inputs.clear()
            out.outputs.clear()
        # Detach node from its tensors.
        for tensor in list(node.inputs):
            if node in tensor.outputs:
                tensor.outputs.remove(node)
        for tensor in list(node.outputs):
            tensor.inputs.clear()
        node.inputs.clear()
        node.outputs.clear()
        graph.nodes.remove(node)

    return len(to_remove)


def swap_node_op(
    graph: GraphLike,
    node_name: str,
    new_op: str,
    *,
    attrs: Optional[Dict[str, Any]] = None,
    clear_attrs: bool = True,
    new_name: Optional[str] = None,
) -> NodeLike:
    """Change a node's operator type (and optionally attrs / name)."""
    node = get_node(graph, node_name)
    node.op = new_op
    if clear_attrs:
        node.attrs.clear()
    if attrs:
        node.attrs.update(attrs)
    if new_name is not None:
        node.name = new_name
    return node


def rename_node(graph: GraphLike, old_name: str, new_name: str) -> NodeLike:
    """Rename a node. Raises if ``new_name`` already exists."""
    if find_nodes(graph, name=new_name):
        raise ValueError(f"Node name {new_name!r} already exists")
    node = get_node(graph, old_name)
    node.name = new_name
    return node


def rename_tensor(
    graph: GraphLike,
    old_name: str,
    new_name: str,
    *,
    update_graph_io: bool = True,
) -> TensorLike:
    """Rename a tensor everywhere it appears.

    GraphSurgeon tensors are referenced by object identity on node edges, so
    renaming ``tensor.name`` is sufficient for producer/consumer links. Graph
    inputs/outputs are the same objects when ``update_graph_io`` is True.
    """
    tmap = graph.tensors()
    if old_name not in tmap:
        raise KeyError(f"Tensor {old_name!r} not found")
    if new_name in tmap and tmap[new_name] is not tmap[old_name]:
        raise ValueError(f"Tensor name {new_name!r} already exists")
    tensor = tmap[old_name]
    tensor.name = new_name
    if update_graph_io:
        # Inputs/outputs already hold the same tensor objects; name change is
        # enough. Keep the flag for API clarity / future-proofing.
        _ = graph.inputs, graph.outputs
    return tensor


def replace_subgraph(
    graph: GraphLike,
    inputs: Sequence[Union[str, TensorLike]],
    outputs: Sequence[Union[str, TensorLike]],
    op: str,
    *,
    attrs: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> NodeLike:
    """Replace the subgraph spanning ``inputs`` → ``outputs`` with one node.

    Mirrors the GraphSurgeon pattern used for TRT plugin substitution:

    1. Disconnect outputs of each input tensor.
    2. Disconnect inputs of each output tensor.
    3. Insert ``op`` with those inputs/outputs.
    4. Caller should ``cleanup_graph(graph)`` to drop the dangling subgraph.
    """
    gs = _gs()
    tmap = graph.tensors()

    def _resolve(value: Union[str, TensorLike]) -> TensorLike:
        if isinstance(value, str):
            if value not in tmap:
                raise KeyError(f"Tensor {value!r} not found")
            return tmap[value]
        return value

    input_tensors = [_resolve(v) for v in inputs]
    output_tensors = [_resolve(v) for v in outputs]

    for tensor in input_tensors:
        tensor.outputs.clear()
    for tensor in output_tensors:
        tensor.inputs.clear()

    node = gs.Node(
        op=op,
        name=name or "",
        attrs=dict(attrs or {}),
        inputs=list(input_tensors),
        outputs=list(output_tensors),
    )
    graph.nodes.append(node)
    logging.getLogger(__name__).info(
        f"Replaced subgraph with {op!r} "
        f"(inputs={[t.name for t in input_tensors]}, "
        f"outputs={[t.name for t in output_tensors]})"
    )
    return node


def replace_node_with_plugin(
    graph: GraphLike,
    node_name: str,
    plugin_op: str,
    *,
    attrs: Optional[Dict[str, Any]] = None,
    new_name: Optional[str] = None,
) -> NodeLike:
    """Replace a single node in-place with a TensorRT plugin op name.

    Keeps the same input/output tensors; only the op (and attrs) change.
    Equivalent to ``swap_node_op`` with a plugin-oriented name.
    """
    return swap_node_op(
        graph,
        node_name,
        plugin_op,
        attrs=attrs,
        clear_attrs=True,
        new_name=new_name,
    )


# ONNX TensorProto.UINT8 / INT32
_ONNX_UINT8 = 2
_ONNX_INT32 = 6

# nvinfer1::Dims::MAX_DIMS
TENSORRT_MAX_TENSOR_RANK = 8


def _is_uint8_dtype(dtype) -> bool:
    if dtype is None:
        return False
    if dtype == _ONNX_UINT8:
        return True
    name = getattr(dtype, "name", None)
    if name == "uint8":
        return True
    return str(dtype) in ("uint8", "<class 'numpy.uint8'>")


def rewrite_uint8_intermediate_casts(
    graph: GraphLike,
    *,
    target_onnx_dtype: int = _ONNX_INT32,
) -> List[str]:
    """Rewrite UINT8 ``Cast`` nodes used as intermediate tensors to INT32.

    TensorRT rejects UINT8 intermediates unless UINT8-quantization import is
    enabled (``legalUINT8``). Image preprocess graphs often cast FP32 → UINT8
    then back to FP32 around reshape/gather. INT32 is TRT-legal and matches
    values already clipped to ``[0, 255]``.

    Does not change graph inputs/outputs that are already UINT8.
    Returns names of rewritten Cast nodes.
    """
    import numpy as np

    np_dtype = {6: np.int32, 1: np.float32, 7: np.int64}.get(
        target_onnx_dtype, np.int32
    )
    rewritten: List[str] = []
    graph_io_ids = {id(t) for t in list(graph.inputs) + list(graph.outputs)}

    for node in list(graph.nodes):
        if node.op != "Cast":
            continue
        to_attr = node.attrs.get("to")
        try:
            to_val = int(to_attr)
        except (TypeError, ValueError):
            continue
        if to_val != _ONNX_UINT8:
            continue
        # Keep UINT8 if this Cast produces a graph I/O tensor.
        if any(id(t) in graph_io_ids for t in node.outputs):
            continue

        node.attrs["to"] = int(target_onnx_dtype)
        stack = list(node.outputs)
        seen: set[int] = set()
        while stack:
            tensor = stack.pop()
            tid = id(tensor)
            if tid in seen:
                continue
            seen.add(tid)
            if id(tensor) in graph_io_ids:
                continue
            if _is_uint8_dtype(tensor.dtype) or tensor.dtype is None:
                tensor.dtype = np_dtype
            for consumer in list(tensor.outputs):
                if consumer.op == "Cast":
                    continue
                stack.extend(consumer.outputs)
        rewritten.append(node.name or f"Cast_{id(node)}")

    if rewritten:
        logging.getLogger(__name__).info(
            f"Rewrote UINT8 intermediate Cast(s) to ONNX dtype "
            f"{target_onnx_dtype}: {rewritten}"
        )
    return rewritten


def disable_resize_antialias(graph: GraphLike) -> List[str]:
    """Set ``antialias=0`` on every Resize node.

    TensorRT's ONNX parser requires ``antialias == 0`` (``UNSUPPORTED_NODE_ATTR``).
    Bicubic Resize in GR00T ``preprocess_video`` is exported with ``antialias=1``.
    Clearing it makes the node legal; it can slightly increase aliasing vs
    the original PyTorch upsample.
    """
    rewritten: List[str] = []
    for node in graph.nodes:
        if node.op != "Resize":
            continue
        antialias = node.attrs.get("antialias", 0)
        try:
            value = int(antialias)
        except (TypeError, ValueError):
            continue
        if value == 0:
            continue
        node.attrs["antialias"] = 0
        rewritten.append(node.name or f"Resize_{id(node)}")
    if rewritten:
        logging.getLogger(__name__).info(f"Disabled Resize antialias on: {rewritten}")
    return rewritten


def _shifted_axis(axis: int, dropped: Sequence[int]) -> int:
    """Index of ``axis`` after the ``dropped`` axes are removed."""
    return axis - sum(1 for d in dropped if d < axis)


def _constant_values(tensor: TensorLike) -> Optional[List[int]]:
    gs = _gs()
    if not isinstance(tensor, gs.Constant):
        return None
    return [int(v) for v in tensor.values.reshape(-1)]


def _plan_axis_drop(
    tensor: TensorLike,
    dropped: Sequence[int],
    plan: List[Any],
    depth: int = 0,
) -> bool:
    """Collect Transpose perm updates needed to drop ``dropped`` axes of ``tensor``.

    Returns False if any consumer cannot be rewritten safely; callers must then
    discard ``plan`` instead of applying it.
    """
    if depth > TENSORRT_MAX_TENSOR_RANK:
        return False
    for consumer in tensor.outputs:
        if consumer.op == "Reshape":
            # A Reshape target shape is absolute, so removing extent-1 axes from
            # its input leaves it valid -- unless a 0 makes it copy input dims.
            if len(consumer.inputs) < 2:
                return False
            values = _constant_values(consumer.inputs[1])
            if values is None or 0 in values:
                return False
            continue
        if consumer.op != "Transpose":
            return False
        perm = consumer.attrs.get("perm")
        if perm is None:
            return False
        perm = [int(p) for p in perm]
        new_perm = [_shifted_axis(p, dropped) for p in perm if p not in dropped]
        out_dropped = [j for j, p in enumerate(perm) if p in dropped]
        plan.append((consumer, new_perm))
        for out in consumer.outputs:
            if not _plan_axis_drop(out, out_dropped, plan, depth + 1):
                return False
    return True


def reduce_reshape_rank_for_tensorrt(
    graph: GraphLike,
    max_rank: int = TENSORRT_MAX_TENSOR_RANK,
) -> List[str]:
    """Drop extent-1 axes from Reshape targets whose rank exceeds ``max_rank``.

    TensorRT tensors are limited to ``TENSORRT_MAX_TENSOR_RANK`` dimensions, so a
    Reshape to higher rank fails with ``INVALID_NODE``. Patch-shuffle graphs
    (Reshape → Transpose → Reshape, as in GR00T ``preprocess_video``) hit this
    with a leading ``1``. Removing extent-1 axes never changes row-major element
    order, so consuming Transpose perms are remapped and the result is
    numerically identical.

    Skips any Reshape that lacks enough extent-1 axes or feeds an op other than
    Transpose / constant-shape Reshape. Returns names of rewritten Reshape nodes.
    """
    import numpy as np

    gs = _gs()
    rewritten: List[str] = []

    for node in list(graph.nodes):
        if node.op != "Reshape" or len(node.inputs) < 2:
            continue
        shape_tensor = node.inputs[1]
        values = _constant_values(shape_tensor)
        if values is None:
            continue
        excess = len(values) - max_rank
        if excess <= 0:
            continue

        unit_axes = [i for i, v in enumerate(values) if v == 1]
        name = node.name or f"Reshape_{id(node)}"
        if len(unit_axes) < excess:
            logging.getLogger(__name__).warning(
                f"Cannot lower rank of {name}: target shape {values} needs "
                f"{excess} fewer axes but has {len(unit_axes)} extent-1 axes"
            )
            continue

        dropped = sorted(unit_axes[:excess])
        plan: List[Any] = []
        if not all(_plan_axis_drop(out, dropped, plan) for out in node.outputs):
            logging.getLogger(__name__).warning(
                f"Cannot lower rank of {name}: a consumer is not a Transpose or "
                "constant-shape Reshape"
            )
            continue

        keep = set(range(len(values))) - set(dropped)
        new_values = np.array(
            [v for i, v in enumerate(values) if i in keep],
            dtype=shape_tensor.values.dtype,
        )
        node.inputs[1] = gs.Constant(
            name=f"{shape_tensor.name}_rank{len(new_values)}", values=new_values
        )
        for transpose, new_perm in plan:
            transpose.attrs["perm"] = new_perm
        # Recorded shapes still carry the old rank; let shape inference redo them.
        for tensor in list(node.outputs) + [
            out for transpose, _ in plan for out in transpose.outputs
        ]:
            tensor.shape = None
        rewritten.append(name)

    if rewritten:
        logging.getLogger(__name__).info(
            f"Lowered Reshape rank to <= {max_rank} for TensorRT: {rewritten}"
        )
    return rewritten


def apply_tensorrt_rewrites(graph: GraphLike) -> List[str]:
    """Run all TensorRT GraphSurgeon rewrites. Returns names of edited nodes."""
    rewritten: List[str] = []
    rewritten.extend(rewrite_uint8_intermediate_casts(graph))
    rewritten.extend(disable_resize_antialias(graph))
    rewritten.extend(reduce_reshape_rank_for_tensorrt(graph))
    return rewritten


def graph_io_signature(model: ModelProto) -> Dict[str, List[tuple]]:
    """Return graph input/output (name, elem_type, dims) for alignment checks."""

    def _dims(info) -> tuple:
        return tuple(
            d.dim_value if d.dim_value else (d.dim_param or "?")
            for d in info.type.tensor_type.shape.dim
        )

    def _entries(infos) -> List[tuple]:
        return [
            (info.name, int(info.type.tensor_type.elem_type), _dims(info))
            for info in infos
        ]

    return {
        "inputs": _entries(model.graph.input),
        "outputs": _entries(model.graph.output),
    }


def rewrite_onnx_for_tensorrt(
    model_or_path: Union[str, ModelProto],
    output_path: Optional[str] = None,
) -> ModelProto:
    """Apply GraphSurgeon rewrites that unblock TensorRT parse for typical ONNX graphs.

    Currently: UINT8 intermediate Casts → INT32, Resize ``antialias`` → 0, and
    Reshape rank > 8 lowered by dropping extent-1 axes. Graph I/O names, dtypes,
    and shapes are left unchanged so downstream YAML / Triton bindings stay valid.
    """
    graph = load_graph(model_or_path)
    apply_tensorrt_rewrites(graph)
    if output_path is not None:
        save_graph(graph, output_path)
        return onnx.load(output_path)
    return export_model(graph)


def rewrite_uint8_intermediates_in_onnx(
    model_or_path: Union[str, ModelProto],
    output_path: Optional[str] = None,
) -> ModelProto:
    """Load an ONNX model, rewrite UINT8 intermediates, optionally save."""
    return rewrite_onnx_for_tensorrt(model_or_path, output_path=output_path)
