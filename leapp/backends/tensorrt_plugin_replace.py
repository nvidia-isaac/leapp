#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Replace ONNX layers / subgraphs with TensorRT plugin nodes.

TensorRT's ONNX parser looks up a registered plugin creator whose name matches
the ONNX node ``op_type``. These helpers rewrite the graph (via GraphSurgeon)
so a standard ONNX layer or tensor-bounded subgraph becomes that plugin node.

Requires ``isaac_deploy_trt`` (``pip install -e ./packages/isaac_deploy_trt``).

Example::

    from leapp.backends.tensorrt_plugin_replace import (
        TrtPluginSpec,
        replace_onnx_layer_with_trt_plugin,
    )

    replace_onnx_layer_with_trt_plugin(
        "model.onnx",
        layer_name="nms",
        plugin=TrtPluginSpec(
            name="BatchedNMS_TRT",
            attrs={"shareLocation": 1, "numClasses": 80},
            version="1",
            namespace="",
        ),
        output_path="model_trt_plugin.onnx",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from onnx import ModelProto

from leapp.utils.logging import _get_logger

GraphLike = Any
NodeLike = Any
TensorLike = Any


def _ogs():
    try:
        from isaac_deploy_trt import surgery as ogs
    except ImportError as exc:
        raise ImportError(
            "isaac_deploy_trt is required for TensorRT plugin replacement. "
            "Install with: pip install -e ./packages/isaac_deploy_trt"
        ) from exc
    return ogs


@dataclass
class TrtPluginSpec:
    """Description of a TensorRT plugin node to insert into ONNX.

    Attributes:
        name: Plugin / creator name. Written as the ONNX ``op_type``.
        attrs: Plugin attributes stored as ONNX node attributes.
        version: Optional plugin version; stored as ``plugin_version`` unless
            already present in ``attrs``.
        namespace: Optional plugin namespace; stored as ``plugin_namespace``
            unless already present in ``attrs``.
        node_name: Optional ONNX node name for the replacement node.
    """

    name: str
    attrs: Dict[str, Any] = field(default_factory=dict)
    version: Optional[str] = None
    namespace: Optional[str] = None
    node_name: Optional[str] = None

    def onnx_attrs(self) -> Dict[str, Any]:
        """Attrs dict written onto the ONNX plugin node."""
        attrs = dict(self.attrs)
        if self.version is not None:
            attrs.setdefault("plugin_version", self.version)
        if self.namespace is not None:
            attrs.setdefault("plugin_namespace", self.namespace)
        return attrs


def _normalize_plugin(plugin: Union[str, TrtPluginSpec]) -> TrtPluginSpec:
    if isinstance(plugin, TrtPluginSpec):
        return plugin
    return TrtPluginSpec(name=plugin)


def replace_layer_with_trt_plugin(
    graph: GraphLike,
    layer_name: str,
    plugin: Union[str, TrtPluginSpec],
    *,
    keep_node_name: bool = True,
) -> NodeLike:
    """Replace one ONNX layer (by node name) with a TRT plugin node in-place.

    Keeps the same input/output tensors; only ``op_type``, attrs, and optionally
    the node name change.
    """
    spec = _normalize_plugin(plugin)
    new_name = spec.node_name
    if new_name is None and keep_node_name:
        new_name = None  # leave existing name
    elif new_name is None and not keep_node_name:
        new_name = f"{layer_name}_{spec.name}"

    node = _ogs().replace_node_with_plugin(
        graph,
        layer_name,
        spec.name,
        attrs=spec.onnx_attrs(),
        new_name=new_name,
    )
    _get_logger().info(
        f"Replaced ONNX layer {layer_name!r} with TRT plugin {spec.name!r}"
    )
    return node


def replace_layers_by_op_with_trt_plugin(
    graph: GraphLike,
    onnx_op: str,
    plugin: Union[str, TrtPluginSpec],
    *,
    predicate=None,
) -> List[NodeLike]:
    """Replace every node with ONNX op ``onnx_op`` with the same TRT plugin.

    Args:
        graph: GraphSurgeon graph.
        onnx_op: Source ONNX operator type (e.g. ``"NonMaxSuppression"``).
        plugin: Target plugin spec or plugin name string.
        predicate: Optional ``callable(node) -> bool`` to filter matches.

    Returns:
        List of replaced plugin nodes.
    """
    spec = _normalize_plugin(plugin)
    matches = _ogs().find_nodes(graph, op=onnx_op, predicate=predicate)
    replaced: List[NodeLike] = []
    for node in matches:
        # Names can be empty; fall back to object identity via current name
        # after ensuring a stable name for lookup.
        if not node.name:
            node.name = f"{onnx_op}_{len(replaced)}"
        replaced.append(
            replace_layer_with_trt_plugin(graph, node.name, spec, keep_node_name=True)
        )
    _get_logger().info(
        f"Replaced {len(replaced)} ONNX '{onnx_op}' layer(s) with TRT plugin {spec.name!r}"
    )
    return replaced


def replace_subgraph_with_trt_plugin(
    graph: GraphLike,
    inputs: Sequence[Union[str, TensorLike]],
    outputs: Sequence[Union[str, TensorLike]],
    plugin: Union[str, TrtPluginSpec],
    *,
    cleanup: bool = True,
) -> NodeLike:
    """Replace the subgraph between ``inputs`` and ``outputs`` with one TRT plugin.

    This is the GraphSurgeon TRT-plugin pattern: disconnect the span, insert a
    plugin node with those I/O tensors, then cleanup dangling nodes.
    """
    spec = _normalize_plugin(plugin)
    node = _ogs().replace_subgraph(
        graph,
        inputs=inputs,
        outputs=outputs,
        op=spec.name,
        attrs=spec.onnx_attrs(),
        name=spec.node_name or "",
    )
    if cleanup:
        _ogs().cleanup_graph(graph)
    _get_logger().info(
        f"Replaced subgraph with TRT plugin {spec.name!r} "
        f"(inputs={list(inputs)}, outputs={list(outputs)})"
    )
    return node


def replace_onnx_layer_with_trt_plugin(
    model_or_path: Union[str, ModelProto],
    plugin: Union[str, TrtPluginSpec],
    *,
    layer_name: Optional[str] = None,
    onnx_op: Optional[str] = None,
    input_tensors: Optional[Sequence[str]] = None,
    output_tensors: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
) -> ModelProto:
    """High-level helper: load ONNX, replace layer(s)/subgraph with a TRT plugin.

    Exactly one replacement mode must be selected:

    - ``layer_name``: replace a single named node in-place.
    - ``onnx_op``: replace all nodes of that ONNX op type.
    - ``input_tensors`` + ``output_tensors``: replace that subgraph span.

    Args:
        model_or_path: ONNX path or ``ModelProto``.
        plugin: Plugin name or :class:`TrtPluginSpec`.
        layer_name: Node name to replace.
        onnx_op: Op type to replace everywhere.
        input_tensors: Subgraph input tensor names.
        output_tensors: Subgraph output tensor names.
        output_path: If set, write the modified model here.

    Returns:
        Modified ``ModelProto``.
    """
    modes = (
        int(layer_name is not None)
        + int(onnx_op is not None)
        + int(input_tensors is not None or output_tensors is not None)
    )
    if modes != 1:
        raise ValueError(
            "Specify exactly one of: layer_name, onnx_op, "
            "or input_tensors+output_tensors"
        )
    if (input_tensors is None) ^ (output_tensors is None):
        raise ValueError("input_tensors and output_tensors must be provided together")

    graph = _ogs().load_graph(model_or_path)
    spec = _normalize_plugin(plugin)

    if layer_name is not None:
        replace_layer_with_trt_plugin(graph, layer_name, spec)
    elif onnx_op is not None:
        replaced = replace_layers_by_op_with_trt_plugin(graph, onnx_op, spec)
        if not replaced:
            raise KeyError(f"No ONNX layers with op_type {onnx_op!r} found")
    else:
        assert input_tensors is not None and output_tensors is not None
        replace_subgraph_with_trt_plugin(
            graph, input_tensors, output_tensors, spec, cleanup=True
        )

    model = _ogs().export_model(graph, do_cleanup=True)
    if output_path is not None:
        import onnx

        onnx.save(model, output_path)
        _get_logger().info(f"Wrote TRT-plugin ONNX to {output_path}")
    return model
