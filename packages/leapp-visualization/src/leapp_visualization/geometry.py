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

from __future__ import annotations

from dataclasses import dataclass, replace

from .layout import LayoutResult, Point
from .model import EdgeKind, VisualGraph, VisualNode, VisualPort, VisualTerminal


_CANVAS_MARGIN = 96.0
_HEADER_HEIGHT = 34.0
_PORT_ROW_HEIGHT = 62.0
_NODE_PADDING = 14.0
_PORT_COLUMN_GAP = 40.0
_LAYER_GAP = 88.0
_LAYER_ITEM_GAP = 56.0
_MIN_NODE_WIDTH = 220.0
_MIN_TERMINAL_WIDTH = 100.0
_MAX_TERMINAL_WIDTH = 220.0
_TERMINAL_HEIGHT = 34.0
_TITLE_CHAR_WIDTH = 8.0
_PORT_PRIMARY_CHAR_WIDTH = 7.0
_PORT_DETAIL_CHAR_WIDTH = 6.2
_PORT_NAME_MAX_CHARS = 24
_PORT_DETAIL_MAX_CHARS = 32
_FEEDBACK_LANE_STEP = 34.0
_FEEDBACK_STUB = 24.0
_FEEDBACK_STUB_STEP = 8.0


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class NodeGeometry:
    id: str
    title: str
    backend: str | None
    rect: Rect
    header_rect: Rect


@dataclass(frozen=True)
class TerminalGeometry:
    id: str
    title: str
    kind: str
    rect: Rect
    anchor: Point


@dataclass(frozen=True)
class PortGeometry:
    id: str
    node_id: str
    side: str
    name: str
    detail: str
    kind: str | None
    rect: Rect
    anchor: Point
    full_label: str


@dataclass(frozen=True)
class EdgeGeometry:
    id: str
    kind: EdgeKind
    label: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class GraphGeometry:
    graph_name: str
    width: float
    height: float
    content_bounds: Rect
    nodes: dict[str, NodeGeometry]
    terminals: dict[str, TerminalGeometry]
    ports: dict[str, PortGeometry]
    edges: dict[str, EdgeGeometry]


@dataclass(frozen=True)
class _Size:
    width: float
    height: float


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def resolve_geometry(graph: VisualGraph, layout: LayoutResult, graph_name: str) -> GraphGeometry:
    visual_sizes = _measure_visual_sizes(graph)
    centers = _expand_layout_centers(layout.centers, visual_sizes)
    graph = _order_ports_for_geometry(graph, centers)
    node_geometries = {
        node.id: _build_node_geometry(node, centers[node.id])
        for node in graph.nodes
    }

    port_geometries: dict[str, PortGeometry] = {}
    for node in graph.nodes:
        port_geometries.update(_build_port_geometries(node, node_geometries[node.id]))

    terminal_geometries = {
        terminal.id: _build_terminal_geometry(terminal, centers[terminal.id], port_geometries[terminal.port_id])
        for terminal in graph.terminals
    }

    raw_content_bounds = _bounds_from_rects(
        [geometry.rect for geometry in node_geometries.values()]
        + [geometry.rect for geometry in terminal_geometries.values()]
    )

    feedback_edge_ids = tuple(sorted(edge.id for edge in graph.edges if edge.kind == "feedback"))
    feedback_lane_index = {edge_id: index for index, edge_id in enumerate(feedback_edge_ids)}

    edge_geometries = {
        edge.id: _build_edge_geometry(
            edge=edge,
            ports=port_geometries,
            terminals=terminal_geometries,
            content_bounds=raw_content_bounds,
            feedback_lane_index=feedback_lane_index,
        )
        for edge in graph.edges
    }

    raw_overall_bounds = _bounds_from_rects_and_points(
        rects=[raw_content_bounds],
        points=[point for edge in edge_geometries.values() for point in edge.points],
    )
    feedback_top_offset = max(0.0, raw_content_bounds.y - raw_overall_bounds.y)
    shift_x = _CANVAS_MARGIN - raw_overall_bounds.x
    shift_y = _CANVAS_MARGIN + feedback_top_offset - raw_content_bounds.y

    shifted_nodes = {
        node_id: NodeGeometry(
            id=geometry.id,
            title=geometry.title,
            backend=geometry.backend,
            rect=_shift_rect(geometry.rect, shift_x, shift_y),
            header_rect=_shift_rect(geometry.header_rect, shift_x, shift_y),
        )
        for node_id, geometry in node_geometries.items()
    }
    shifted_terminals = {
        terminal_id: TerminalGeometry(
            id=geometry.id,
            title=geometry.title,
            kind=geometry.kind,
            rect=_shift_rect(geometry.rect, shift_x, shift_y),
            anchor=_shift_point(geometry.anchor, shift_x, shift_y),
        )
        for terminal_id, geometry in terminal_geometries.items()
    }
    shifted_ports = {
        port_id: PortGeometry(
            id=geometry.id,
            node_id=geometry.node_id,
            side=geometry.side,
            name=geometry.name,
            detail=geometry.detail,
            kind=geometry.kind,
            rect=_shift_rect(geometry.rect, shift_x, shift_y),
            anchor=_shift_point(geometry.anchor, shift_x, shift_y),
            full_label=geometry.full_label,
        )
        for port_id, geometry in port_geometries.items()
    }
    shifted_edges = {
        edge_id: EdgeGeometry(
            id=geometry.id,
            kind=geometry.kind,
            label=geometry.label,
            points=tuple(_shift_point(point, shift_x, shift_y) for point in geometry.points),
        )
        for edge_id, geometry in edge_geometries.items()
    }

    content_bounds = _shift_rect(raw_content_bounds, shift_x, shift_y)
    overall_bounds = _bounds_from_rects_and_points(
        rects=[content_bounds],
        points=[point for edge in shifted_edges.values() for point in edge.points],
    )

    return GraphGeometry(
        graph_name=graph_name,
        width=overall_bounds.x + overall_bounds.width + _CANVAS_MARGIN,
        height=overall_bounds.y + overall_bounds.height + _CANVAS_MARGIN,
        content_bounds=content_bounds,
        nodes=shifted_nodes,
        terminals=shifted_terminals,
        ports=shifted_ports,
        edges=shifted_edges,
    )


def _order_ports_for_geometry(graph: VisualGraph, centers: dict[str, Point]) -> VisualGraph:
    original_port_indices = _collect_original_port_indices(graph.nodes)
    input_order_keys: dict[str, list[tuple[float, float]]] = {}

    for edge in graph.edges:
        if edge.kind not in ("forward", "feedback"):
            continue
        if edge.source_port_id is None or edge.target_port_id is None:
            continue
        source_key = _connected_port_order_key(edge.source_id, edge.source_port_id, centers, original_port_indices)
        input_order_keys.setdefault(edge.target_port_id, []).append(source_key)

    ordered_nodes = tuple(
        replace(
            node,
            inputs=_reorder_connected_ports(node.inputs, input_order_keys, original_port_indices),
        )
        for node in graph.nodes
    )
    return VisualGraph(nodes=ordered_nodes, terminals=graph.terminals, edges=graph.edges)


def _collect_original_port_indices(nodes: tuple[VisualNode, ...]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for node in nodes:
        for index, port in enumerate(node.inputs):
            indices[port.id] = index
        for index, port in enumerate(node.outputs):
            indices[port.id] = index
    return indices


def _connected_port_order_key(
    visual_id: str,
    port_id: str,
    centers: dict[str, Point],
    original_port_indices: dict[str, int],
) -> tuple[float, float]:
    center = centers[visual_id]
    return (center.y, float(original_port_indices[port_id]))


def _reorder_connected_ports(
    ports: tuple[VisualPort, ...],
    order_keys: dict[str, list[tuple[float, float]]],
    original_port_indices: dict[str, int],
) -> tuple[VisualPort, ...]:
    connected_slots = [index for index, port in enumerate(ports) if port.id in order_keys]
    if len(connected_slots) < 2:
        return ports

    connected_ports = [ports[index] for index in connected_slots]
    connected_ports.sort(
        key=lambda port: (
            *_average_order_key(order_keys[port.id]),
            original_port_indices[port.id],
        )
    )

    ordered_ports = list(ports)
    for slot, port in zip(connected_slots, connected_ports):
        ordered_ports[slot] = port
    return tuple(ordered_ports)


def _average_order_key(keys: list[tuple[float, float]]) -> tuple[float, float]:
    return (
        sum(key[0] for key in keys) / len(keys),
        sum(key[1] for key in keys) / len(keys),
    )


def _build_node_geometry(node: VisualNode, center: Point) -> NodeGeometry:
    size = _measure_node_size(node)
    rect = Rect(
        x=center.x - (size.width / 2.0),
        y=center.y - (size.height / 2.0),
        width=size.width,
        height=size.height,
    )
    return NodeGeometry(
        id=node.id,
        title=node.title,
        backend=node.backend,
        rect=rect,
        header_rect=Rect(rect.x, rect.y, rect.width, _HEADER_HEIGHT),
    )


def _build_terminal_geometry(terminal: VisualTerminal, center: Point, connected_port: PortGeometry) -> TerminalGeometry:
    width = _measure_terminal_width(terminal)
    rect = Rect(
        x=center.x - (width / 2.0),
        y=connected_port.anchor.y - (_TERMINAL_HEIGHT / 2.0),
        width=width,
        height=_TERMINAL_HEIGHT,
    )
    anchor_x = rect.x + rect.width if terminal.kind == "graph_input" else rect.x
    anchor = Point(anchor_x, rect.y + (rect.height / 2.0))
    return TerminalGeometry(
        id=terminal.id,
        title=terminal.title,
        kind=terminal.kind,
        rect=rect,
        anchor=anchor,
    )


def _build_port_geometries(node: VisualNode, node_geometry: NodeGeometry) -> dict[str, PortGeometry]:
    port_geometries: dict[str, PortGeometry] = {}
    input_slot_width = _measure_side_slot_width(node.inputs)
    output_slot_width = _measure_side_slot_width(node.outputs)
    row_count = max(len(node.inputs), len(node.outputs))
    first_row_y = node_geometry.rect.y + _HEADER_HEIGHT + _NODE_PADDING

    for index in range(row_count):
        row_y = first_row_y + (index * _PORT_ROW_HEIGHT)
        if index < len(node.inputs):
            geometry = _build_port_geometry(node.inputs[index], node_geometry, row_y, input_slot_width)
            port_geometries[geometry.id] = geometry
        if index < len(node.outputs):
            geometry = _build_port_geometry(node.outputs[index], node_geometry, row_y, output_slot_width)
            port_geometries[geometry.id] = geometry

    return port_geometries


def _build_port_geometry(port: VisualPort, node_geometry: NodeGeometry, row_y: float, slot_width: float) -> PortGeometry:
    name = truncate_text(port.name, _PORT_NAME_MAX_CHARS)
    detail = truncate_text(_port_detail_text(port), _PORT_DETAIL_MAX_CHARS)
    full_label = "\n".join(part for part in (port.name, _port_detail_text(port), port.kind) if part)
    rect_x = node_geometry.rect.x if port.side == "input" else node_geometry.rect.x + node_geometry.rect.width - slot_width
    rect = Rect(
        x=rect_x,
        y=row_y,
        width=slot_width,
        height=_PORT_ROW_HEIGHT,
    )
    anchor_x = node_geometry.rect.x if port.side == "input" else node_geometry.rect.x + node_geometry.rect.width
    anchor = Point(anchor_x, rect.y + (rect.height / 2.0))
    return PortGeometry(
        id=port.id,
        node_id=port.node_id,
        side=port.side,
        name=name,
        detail=detail,
        kind=port.kind,
        rect=rect,
        anchor=anchor,
        full_label=full_label,
    )


def _build_edge_geometry(
    edge,
    ports: dict[str, PortGeometry],
    terminals: dict[str, TerminalGeometry],
    content_bounds: Rect,
    feedback_lane_index: dict[str, int],
) -> EdgeGeometry:
    start = _edge_anchor(edge.source_id, edge.source_port_id, ports, terminals)
    end = _edge_anchor(edge.target_id, edge.target_port_id, ports, terminals)

    if edge.kind == "feedback":
        lane_index = feedback_lane_index[edge.id]
        lane_y = content_bounds.y - ((lane_index + 1) * _FEEDBACK_LANE_STEP)
        stub = _FEEDBACK_STUB + (lane_index * _FEEDBACK_STUB_STEP)
        points = (
            start,
            Point(start.x + stub, start.y),
            Point(start.x + stub, lane_y),
            Point(end.x - stub, lane_y),
            Point(end.x - stub, end.y),
            end,
        )
    else:
        control_x = start.x + ((end.x - start.x) / 2.0)
        points = (
            start,
            Point(control_x, start.y),
            Point(control_x, end.y),
            end,
        )

    return EdgeGeometry(
        id=edge.id,
        kind=edge.kind,
        label=edge.label,
        points=points,
    )


def _edge_anchor(
    visual_id: str,
    port_id: str | None,
    ports: dict[str, PortGeometry],
    terminals: dict[str, TerminalGeometry],
) -> Point:
    if port_id is not None:
        return ports[port_id].anchor
    return terminals[visual_id].anchor


def _measure_visual_sizes(graph: VisualGraph) -> dict[str, _Size]:
    sizes = {node.id: _measure_node_size(node) for node in graph.nodes}
    sizes.update(
        {
            terminal.id: _Size(_measure_terminal_width(terminal), _TERMINAL_HEIGHT)
            for terminal in graph.terminals
        }
    )
    return sizes


def _expand_layout_centers(centers: dict[str, Point], sizes: dict[str, _Size]) -> dict[str, Point]:
    if not centers:
        return {}

    layer_ids: dict[float, list[str]] = {}
    for visual_id, center in centers.items():
        layer_ids.setdefault(round(center.x, 3), []).append(visual_id)

    layer_x: dict[float, float] = {}
    previous_right: float | None = None
    for layer_key in sorted(layer_ids):
        half_width = max(sizes[visual_id].width / 2.0 for visual_id in layer_ids[layer_key])
        center_x = half_width if previous_right is None else previous_right + _LAYER_GAP + half_width
        layer_x[layer_key] = center_x
        previous_right = center_x + half_width

    adjusted: dict[str, Point] = {}
    for layer_key, visual_ids in layer_ids.items():
        previous_bottom: float | None = None
        for visual_id in sorted(visual_ids, key=lambda item: (centers[item].y, item)):
            half_height = sizes[visual_id].height / 2.0
            center_y = centers[visual_id].y
            if previous_bottom is not None and center_y - half_height < previous_bottom + _LAYER_ITEM_GAP:
                center_y = previous_bottom + _LAYER_ITEM_GAP + half_height
            adjusted[visual_id] = Point(layer_x[layer_key], center_y)
            previous_bottom = center_y + half_height

    return adjusted


def _measure_node_size(node: VisualNode) -> _Size:
    input_slot_width = _measure_side_slot_width(node.inputs)
    output_slot_width = _measure_side_slot_width(node.outputs)
    port_widths = [width for width in (input_slot_width, output_slot_width) if width > 0.0]
    if len(port_widths) == 2:
        port_width = input_slot_width + _PORT_COLUMN_GAP + output_slot_width
    elif port_widths:
        port_width = port_widths[0] + _NODE_PADDING
    else:
        port_width = 0.0

    width = max(_MIN_NODE_WIDTH, _measure_node_header_width(node), port_width)
    row_count = max(len(node.inputs), len(node.outputs))
    height = _HEADER_HEIGHT + _NODE_PADDING + (row_count * _PORT_ROW_HEIGHT) + _NODE_PADDING
    return _Size(width, height)


def _measure_node_header_width(node: VisualNode) -> float:
    title_width = len(node.title) * _TITLE_CHAR_WIDTH
    backend_width = len(node.backend or "") * _TITLE_CHAR_WIDTH
    backend_gap = _PORT_COLUMN_GAP if node.backend else 0.0
    return title_width + backend_gap + backend_width + (_NODE_PADDING * 2.0)


def _measure_terminal_width(terminal: VisualTerminal) -> float:
    return _clamp(
        len(terminal.title) * _TITLE_CHAR_WIDTH + (_NODE_PADDING * 2.0),
        _MIN_TERMINAL_WIDTH,
        _MAX_TERMINAL_WIDTH,
    )


def _measure_side_slot_width(ports: tuple[VisualPort, ...]) -> float:
    if not ports:
        return 0.0
    return max(_measure_port_content_width(port) for port in ports) + _NODE_PADDING


def _measure_port_content_width(port: VisualPort) -> float:
    widths = [
        len(truncate_text(port.name, _PORT_NAME_MAX_CHARS)) * _PORT_PRIMARY_CHAR_WIDTH,
        len(truncate_text(_port_detail_text(port), _PORT_DETAIL_MAX_CHARS)) * _PORT_DETAIL_CHAR_WIDTH,
    ]
    visible_kind = _visible_port_kind(port)
    if visible_kind:
        widths.append(len(visible_kind) * _PORT_PRIMARY_CHAR_WIDTH)
    return max(widths)


def _visible_port_kind(port: VisualPort | PortGeometry) -> str | None:
    if not port.kind:
        return None
    return truncate_text(port.kind, _PORT_NAME_MAX_CHARS)


def _port_detail_text(port: VisualPort) -> str:
    return f"{_shape_text(port.shape)} {port.dtype}"


def _shape_text(shape: tuple[str, ...]) -> str:
    if not shape:
        return "[]"
    return f"[{', '.join(shape)}]"


def _bounds_from_rects(rects: list[Rect]) -> Rect:
    if not rects:
        return Rect(0.0, 0.0, 0.0, 0.0)

    min_x = min(rect.x for rect in rects)
    min_y = min(rect.y for rect in rects)
    max_x = max(rect.x + rect.width for rect in rects)
    max_y = max(rect.y + rect.height for rect in rects)
    return Rect(min_x, min_y, max_x - min_x, max_y - min_y)


def _bounds_from_rects_and_points(rects: list[Rect], points: list[Point]) -> Rect:
    xs = [rect.x for rect in rects] + [rect.x + rect.width for rect in rects]
    ys = [rect.y for rect in rects] + [rect.y + rect.height for rect in rects]

    xs.extend(point.x for point in points)
    ys.extend(point.y for point in points)

    min_x = min(xs, default=0.0)
    max_x = max(xs, default=0.0)
    min_y = min(ys, default=0.0)
    max_y = max(ys, default=0.0)
    return Rect(min_x, min_y, max_x - min_x, max_y - min_y)


def _shift_rect(rect: Rect, dx: float, dy: float) -> Rect:
    return Rect(rect.x + dx, rect.y + dy, rect.width, rect.height)


def _shift_point(point: Point, dx: float, dy: float) -> Point:
    return Point(point.x + dx, point.y + dy)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
