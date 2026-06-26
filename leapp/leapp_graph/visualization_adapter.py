from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from leapp_visualization.model import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal, visual_id


def build_visual_graph(
    nodes: Mapping[str, Any],
    connections: Sequence[dict],
    feedback_connections: Sequence[dict],
    graph_inputs: Sequence[str],
    graph_outputs: Sequence[str],
) -> VisualGraph:
    ordered_nodes = [_build_node(node) for _, node in sorted(nodes.items(), key=_node_sort_key)]
    node_name_map = {node.title: node for node in ordered_nodes}

    terminals: list[VisualTerminal] = []
    edges: list[VisualEdge] = []

    for connection in connections:
        edges.extend(_build_internal_edges(connection, "forward"))

    for connection in feedback_connections:
        edges.extend(_build_internal_edges(connection, "feedback"))

    for sequence_index, terminal_edge in enumerate(graph_inputs):
        terminal, edge = _build_graph_input_terminal_edge(terminal_edge, node_name_map, sequence_index)
        terminals.append(terminal)
        edges.append(edge)

    for sequence_index, terminal_edge in enumerate(graph_outputs):
        terminal, edge = _build_graph_output_terminal_edge(terminal_edge, node_name_map, sequence_index)
        terminals.append(terminal)
        edges.append(edge)

    return VisualGraph(nodes=tuple(ordered_nodes), terminals=tuple(terminals), edges=tuple(edges))


def _build_internal_edges(connection: dict, kind: str) -> list[VisualEdge]:
    source = connection["source"]
    source_node = source["node"]
    source_desc = source_node.outputs[source["idx"]]
    source_port_id = _find_port_id(source_node, "output", source_desc.name_str)
    source_id = visual_id("node", source_node.name)

    edges: list[VisualEdge] = []
    for sequence_index, target in enumerate(connection["targets"]):
        target_node = target["node"]
        target_desc = target_node.inputs[target["idx"]]
        target_port_id = _find_port_id(target_node, "input", target_desc.name_str)
        target_id = visual_id("node", target_node.name)
        edges.append(
            VisualEdge(
                id=visual_id("edge", kind, source_id, source_desc.name_str, target_id, target_desc.name_str, sequence_index),
                kind=kind,
                source_id=source_id,
                target_id=target_id,
                source_port_id=source_port_id,
                target_port_id=target_port_id,
                label=source_desc.name_str,
            )
        )
    return edges


def _build_graph_input_terminal_edge(node_port: str, node_name_map: dict[str, VisualNode], sequence_index: int) -> tuple[VisualTerminal, VisualEdge]:
    node_name, port_name = node_port.split("/", 1)
    node = node_name_map[node_name]
    port_id = _find_port_id(node, "input", port_name)
    terminal_id = visual_id("terminal", "input", node_name, port_name)
    terminal = VisualTerminal(
        id=terminal_id,
        kind="graph_input",
        title=port_name,
        node_id=node.id,
        port_id=port_id,
    )
    edge = VisualEdge(
        id=visual_id("edge", "graph_input", terminal_id, port_name, node.id, port_name, sequence_index),
        kind="graph_input",
        source_id=terminal_id,
        target_id=node.id,
        source_port_id=None,
        target_port_id=port_id,
        label=port_name,
    )
    return terminal, edge


def _build_graph_output_terminal_edge(node_port: str, node_name_map: dict[str, VisualNode], sequence_index: int) -> tuple[VisualTerminal, VisualEdge]:
    node_name, port_name = node_port.split("/", 1)
    node = node_name_map[node_name]
    port_id = _find_port_id(node, "output", port_name)
    terminal_id = visual_id("terminal", "output", node_name, port_name)
    terminal = VisualTerminal(
        id=terminal_id,
        kind="graph_output",
        title=port_name,
        node_id=node.id,
        port_id=port_id,
    )
    edge = VisualEdge(
        id=visual_id("edge", "graph_output", node.id, port_name, terminal_id, port_name, sequence_index),
        kind="graph_output",
        source_id=node.id,
        target_id=terminal_id,
        source_port_id=port_id,
        target_port_id=None,
        label=port_name,
    )
    return terminal, edge


def _build_node(node: Any) -> VisualNode:
    backend = node.get_backend() if hasattr(node, "get_backend") else getattr(node, "backend", None)
    return VisualNode(
        id=visual_id("node", node.name),
        title=node.name,
        backend=backend,
        inputs=tuple(_build_port(node.name, "input", desc) for desc in node.inputs),
        outputs=tuple(_build_port(node.name, "output", desc) for desc in node.outputs),
    )


def _build_port(node_name: str, side: str, desc: Any) -> VisualPort:
    kind = _kind_to_string(desc)
    return VisualPort(
        id=visual_id("port", node_name, side, desc.name_str),
        node_id=visual_id("node", node_name),
        side=side,  # type: ignore[arg-type]
        name=desc.name_str,
        shape=_shape_to_tuple(desc.shape),
        dtype=str(desc.dtype),
        kind=kind,
    )


def _shape_to_tuple(shape: Any) -> tuple[str, ...]:
    if shape is None:
        return ()
    if isinstance(shape, tuple):
        return tuple(str(dim) for dim in shape)
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        return tuple(str(dim) for dim in shape)
    return (str(shape),)


def _kind_to_string(desc: Any) -> str | None:
    semantics = desc.get_semantics() if hasattr(desc, "get_semantics") else {}
    kind = semantics.get("kind")
    if kind is None:
        return None
    return kind.value if hasattr(kind, "value") else str(kind)


def _find_port_id(node: Any, side: str, port_name: str) -> str:
    ports = node.inputs if side == "input" else node.outputs
    node_name = getattr(node, "name", getattr(node, "title", None))
    for desc in ports:
        desc_name = getattr(desc, "name_str", getattr(desc, "name", None))
        if desc_name == port_name:
            return visual_id("port", node_name, side, desc_name)
    raise KeyError(f"Port '{port_name}' not found on node '{node_name}' for side '{side}'")


def _node_sort_key(item: tuple[str, Any]) -> tuple[int, str]:
    _, node = item
    return (getattr(node, "node_index", 0), getattr(node, "name", ""))
