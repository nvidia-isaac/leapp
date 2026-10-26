#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Split ONNX graphs into TensorRT-supported and unsupported segments.

Provides catalog-based partitioning (no TensorRT install required) plus helpers
to extract and save contiguous subgraphs. Optionally uses TensorRT's ONNX
parser ``supports_model`` API when ``tensorrt`` is installed, similar to
``polygraphy inspect capability --with-partitioning``.

For insert / remove / swap / rename / subgraph-replace edits (GraphSurgeon),
see ``isaac_deploy_trt.surgery``. To rewrite an ONNX layer as a TensorRT
plugin node, see ``leapp.backends.tensorrt_plugin_replace``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import onnx
from onnx import ModelProto, TensorProto, ValueInfoProto, helper

from leapp.utils.logging import _get_logger


@dataclass(frozen=True)
class NodeSegment:
    """Inclusive-start / exclusive-end node index range in topological order."""

    start: int
    end: int
    supported: bool
    ops: Tuple[str, ...] = ()

    @property
    def num_nodes(self) -> int:
        return self.end - self.start


@dataclass
class GraphSplitResult:
    """Result of partitioning an ONNX model for TensorRT."""

    onnx_path: Optional[str]
    segments: List[NodeSegment] = field(default_factory=list)
    supported_paths: List[str] = field(default_factory=list)
    unsupported_paths: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def fully_supported(self) -> bool:
        return bool(self.segments) and all(seg.supported for seg in self.segments)

    @property
    def supported_segments(self) -> List[NodeSegment]:
        return [seg for seg in self.segments if seg.supported]

    @property
    def unsupported_segments(self) -> List[NodeSegment]:
        return [seg for seg in self.segments if not seg.supported]


def is_op_tensorrt_supported(
    op_type: str,
    *,
    known_supported: Optional[Set[str]] = None,
    unknown_as_supported: bool = False,
) -> bool:
    """Return whether an ONNX op is treated as TensorRT-supported.

    Args:
        op_type: ONNX operator type name.
        known_supported: Optional override set of supported op names.
        unknown_as_supported: If True, ops absent from the catalog are treated
            as supported. Default False (safer for TRT subgraph extraction).
    """
    from isaac_deploy_trt.tensorrt_onnx_ops import (
        TENSORRT_ONNX_OPS,
        supported_onnx_ops,
    )

    supported = known_supported if known_supported is not None else supported_onnx_ops()
    if op_type in supported:
        return True
    if op_type in TENSORRT_ONNX_OPS:
        return False
    return unknown_as_supported


def partition_nodes_by_tensorrt_support(
    nodes: Sequence,
    *,
    known_supported: Optional[Set[str]] = None,
    unknown_as_supported: bool = False,
) -> List[NodeSegment]:
    """Partition a node list into contiguous supported/unsupported segments.

    Consecutive nodes with the same support flag are merged into one segment.
    """
    if not nodes:
        return []

    segments: List[NodeSegment] = []
    start = 0
    current_supported = is_op_tensorrt_supported(
        nodes[0].op_type,
        known_supported=known_supported,
        unknown_as_supported=unknown_as_supported,
    )
    current_ops: List[str] = [nodes[0].op_type]

    for idx in range(1, len(nodes)):
        supported = is_op_tensorrt_supported(
            nodes[idx].op_type,
            known_supported=known_supported,
            unknown_as_supported=unknown_as_supported,
        )
        if supported == current_supported:
            current_ops.append(nodes[idx].op_type)
            continue
        segments.append(
            NodeSegment(
                start=start,
                end=idx,
                supported=current_supported,
                ops=tuple(current_ops),
            )
        )
        start = idx
        current_supported = supported
        current_ops = [nodes[idx].op_type]

    segments.append(
        NodeSegment(
            start=start,
            end=len(nodes),
            supported=current_supported,
            ops=tuple(current_ops),
        )
    )
    return segments


def _value_info_map(model: ModelProto) -> Dict[str, ValueInfoProto]:
    mapping: Dict[str, ValueInfoProto] = {}
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        mapping[vi.name] = vi
    return mapping


def _initializer_names(model: ModelProto) -> Set[str]:
    return {init.name for init in model.graph.initializer}


def _make_value_info(name: str, existing: Optional[ValueInfoProto] = None) -> ValueInfoProto:
    if existing is not None and existing.name == name:
        return existing
    # Fallback when shape/type metadata is unavailable.
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, None)


def extract_onnx_subgraph(
    model: ModelProto,
    start: int,
    end: int,
    *,
    graph_name: Optional[str] = None,
) -> ModelProto:
    """Extract ``model.graph.node[start:end]`` into a standalone ONNX model.

    Graph inputs are tensors consumed by the segment that are not produced
    inside it (excluding initializers). Graph outputs are tensors produced by
    the segment that are not consumed inside it, plus any original model
    outputs produced by the segment.
    """
    if start < 0 or end > len(model.graph.node) or start >= end:
        raise ValueError(
            f"Invalid node range [{start}, {end}) for graph with "
            f"{len(model.graph.node)} nodes"
        )

    nodes = list(model.graph.node[start:end])
    produced: Set[str] = {out for node in nodes for out in node.output if out}
    consumed: Set[str] = {inp for node in nodes for inp in node.input if inp}
    initializer_names = _initializer_names(model)
    original_outputs = {out.name for out in model.graph.output}

    input_names = sorted(name for name in consumed if name not in produced and name not in initializer_names)
    # Prefer tensors that escape the segment; always keep original graph outputs.
    internal_consumed = consumed & produced
    output_names = sorted(
        (produced - internal_consumed) | (produced & original_outputs)
    )
    if not output_names:
        # Degenerate segment that only feeds initializers-style sinks: expose
        # all produced tensors so the subgraph remains runnable.
        output_names = sorted(produced)

    vi_map = _value_info_map(model)
    graph_inputs = [_make_value_info(name, vi_map.get(name)) for name in input_names]
    graph_outputs = [_make_value_info(name, vi_map.get(name)) for name in output_names]

    used_initializers = [
        init for init in model.graph.initializer if init.name in consumed
    ]
    value_infos = [
        vi_map[name]
        for name in sorted(produced & set(vi_map) - set(input_names) - set(output_names))
    ]

    subgraph = helper.make_graph(
        nodes=nodes,
        name=graph_name or f"subgraph_nodes_{start}_{end - 1}",
        inputs=graph_inputs,
        outputs=graph_outputs,
        initializer=used_initializers,
        value_info=value_infos,
    )

    opset_imports = list(model.opset_import) or [helper.make_opsetid("", 17)]
    return helper.make_model(subgraph, opset_imports=opset_imports, ir_version=model.ir_version)


def save_segment_subgraphs(
    model: ModelProto,
    segments: Sequence[NodeSegment],
    output_dir: str,
    *,
    prefix: str = "",
    only_supported: Optional[bool] = None,
) -> Tuple[List[str], List[str]]:
    """Save segment subgraphs under ``output_dir``.

    Args:
        model: Parent ONNX model.
        segments: Partition segments.
        output_dir: Directory for written ``.onnx`` files.
        prefix: Optional filename prefix.
        only_supported: If True/False, only write that class of segments.
            ``None`` writes both.

    Returns:
        ``(supported_paths, unsupported_paths)``
    """
    os.makedirs(output_dir, exist_ok=True)
    supported_paths: List[str] = []
    unsupported_paths: List[str] = []

    for seg in segments:
        if only_supported is True and not seg.supported:
            continue
        if only_supported is False and seg.supported:
            continue

        kind = "supported" if seg.supported else "unsupported"
        filename = f"{prefix}{kind}_subgraph-nodes-{seg.start}-{seg.end - 1}.onnx"
        path = os.path.join(output_dir, filename)
        submodel = extract_onnx_subgraph(
            model,
            seg.start,
            seg.end,
            graph_name=f"{kind}_nodes_{seg.start}_{seg.end - 1}",
        )
        onnx.save(submodel, path)
        if seg.supported:
            supported_paths.append(path)
        else:
            unsupported_paths.append(path)

    return supported_paths, unsupported_paths


def split_onnx_by_tensorrt_support(
    onnx_path: str,
    *,
    output_dir: Optional[str] = None,
    known_supported: Optional[Iterable[str]] = None,
    unknown_as_supported: bool = False,
    save: bool = True,
) -> GraphSplitResult:
    """Split an ONNX model using the static TensorRT op catalog.

    This does not require the ``tensorrt`` Python package. Segments are formed
    from contiguous runs of supported vs unsupported/unknown nodes.
    """
    model = onnx.load(onnx_path, load_external_data=False)
    supported_set = set(known_supported) if known_supported is not None else None
    segments = partition_nodes_by_tensorrt_support(
        model.graph.node,
        known_supported=supported_set,
        unknown_as_supported=unknown_as_supported,
    )

    result = GraphSplitResult(onnx_path=onnx_path, segments=segments)
    if not save:
        return result

    out_dir = output_dir or os.path.join(
        os.path.dirname(os.path.abspath(onnx_path)) or ".",
        "tensorrt_graph_split",
    )
    supported_paths, unsupported_paths = save_segment_subgraphs(
        model, segments, out_dir
    )
    result.supported_paths = supported_paths
    result.unsupported_paths = unsupported_paths

    logger = _get_logger()
    logger.info(
        f"Split {onnx_path} into {len(segments)} segment(s) "
        f"({len(result.supported_segments)} supported, "
        f"{len(result.unsupported_segments)} unsupported) under {out_dir}"
    )
    return result


def split_onnx_with_tensorrt_parser(
    onnx_path: str,
    *,
    output_dir: Optional[str] = None,
    save: bool = True,
) -> GraphSplitResult:
    """Split an ONNX model using TensorRT ``OnnxParser.supports_model``.

    Requires the ``tensorrt`` package. Falls back to reporting an error in
    ``GraphSplitResult.errors`` when TensorRT is unavailable or the API is
    missing.
    """
    result = GraphSplitResult(onnx_path=onnx_path)
    try:
        import tensorrt as trt
    except ImportError as exc:
        result.errors.append(
            "tensorrt is not installed; cannot run parser-based graph split"
        )
        result.errors.append(str(exc))
        return result

    if not hasattr(trt.OnnxParser, "supports_model"):
        result.errors.append(
            "tensorrt.OnnxParser.supports_model is unavailable in this TensorRT version"
        )
        return result

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        model_bytes = f.read()
    fully_supported, nodelists = parser.supports_model(model_bytes, onnx_path)

    model = onnx.load(onnx_path, load_external_data=False)
    segments: List[NodeSegment] = []
    for node_indices, supported in nodelists:
        if not node_indices:
            continue
        start = int(node_indices[0])
        end = int(node_indices[-1]) + 1
        ops = tuple(model.graph.node[i].op_type for i in range(start, end))
        segments.append(
            NodeSegment(start=start, end=end, supported=bool(supported), ops=ops)
        )

    # If TRT reports full support but returns no partitions, treat the whole
    # graph as one supported segment.
    if fully_supported and not segments and model.graph.node:
        segments = [
            NodeSegment(
                start=0,
                end=len(model.graph.node),
                supported=True,
                ops=tuple(node.op_type for node in model.graph.node),
            )
        ]

    result.segments = segments
    if not save:
        return result

    out_dir = output_dir or os.path.join(
        os.path.dirname(os.path.abspath(onnx_path)) or ".",
        "tensorrt_graph_split_parser",
    )
    supported_paths, unsupported_paths = save_segment_subgraphs(
        model, segments, out_dir
    )
    result.supported_paths = supported_paths
    result.unsupported_paths = unsupported_paths
    return result


def replace_unsupported_nodes_with_identity(
    model: ModelProto,
    *,
    known_supported: Optional[Set[str]] = None,
    unknown_as_supported: bool = False,
) -> ModelProto:
    """Return a copy where unsupported single-input/single-output nodes become Identity.

    Useful as a coarse placeholder when prototyping TRT-only paths. Nodes that
    are not 1-in/1-out are left unchanged and should be handled by subgraph
    extraction instead.
    """
    cloned = ModelProto()
    cloned.CopyFrom(model)
    supported_set = known_supported

    for node in cloned.graph.node:
        if is_op_tensorrt_supported(
            node.op_type,
            known_supported=supported_set,
            unknown_as_supported=unknown_as_supported,
        ):
            continue
        if len(node.input) != 1 or len(node.output) != 1:
            continue
        del node.attribute[:]
        node.op_type = "Identity"
        node.name = f"{node.name}_as_identity" if node.name else "unsupported_as_identity"

    return cloned
