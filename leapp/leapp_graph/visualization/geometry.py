from __future__ import annotations

from dataclasses import dataclass

from .layout import LayoutResult, Point
from .model import EdgeKind, VisualGraph, VisualNode, VisualPort, VisualTerminal


_CANVAS_MARGIN = 48.0
_HEADER_HEIGHT = 34.0
_PORT_ROW_HEIGHT = 46.0
_NODE_PADDING = 14.0
_PORT_GROUP_GAP = 10.0
_MIN_NODE_WIDTH = 220.0
_MAX_NODE_WIDTH = 420.0
_MIN_TERMINAL_WIDTH = 100.0
_MAX_TERMINAL_WIDTH = 220.0
_TERMINAL_HEIGHT = 34.0
_TITLE_CHAR_WIDTH = 8.0
_PORT_PRIMARY_CHAR_WIDTH = 7.0
_PORT_DETAIL_CHAR_WIDTH = 6.2
_PORT_NAME_MAX_CHARS = 24
_PORT_DETAIL_MAX_CHARS = 32
_FEEDBACK_LANE_STEP = 34.0


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


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def resolve_geometry(graph: VisualGraph, layout: LayoutResult, graph_name: str) -> GraphGeometry:
    node_geometries = {
        node.id: _build_node_geometry(node, layout.centers[node.id])
        for node in graph.nodes
    }
    terminal_geometries = {
        terminal.id: _build_terminal_geometry(terminal, layout.centers[terminal.id])
        for terminal in graph.terminals
    }

    port_geometries: dict[str, PortGeometry] = {}
    for node in graph.nodes:
        port_geometries.update(_build_port_geometries(node, node_geometries[node.id]))

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


def _build_node_geometry(node: VisualNode, center: Point) -> NodeGeometry:
    width = _measure_node_width(node)
    height = (
        _HEADER_HEIGHT
        + _NODE_PADDING
        + len(node.inputs) * _PORT_ROW_HEIGHT
        + _PORT_GROUP_GAP
        + len(node.outputs) * _PORT_ROW_HEIGHT
        + _NODE_PADDING
    )
    rect = Rect(
        x=center.x - (width / 2.0),
        y=center.y - (height / 2.0),
        width=width,
        height=height,
    )
    return NodeGeometry(
        id=node.id,
        title=node.title,
        backend=node.backend,
        rect=rect,
        header_rect=Rect(rect.x, rect.y, rect.width, _HEADER_HEIGHT),
    )


def _build_terminal_geometry(terminal: VisualTerminal, center: Point) -> TerminalGeometry:
    width = _clamp(
        len(terminal.title) * _TITLE_CHAR_WIDTH + (_NODE_PADDING * 2.0),
        _MIN_TERMINAL_WIDTH,
        _MAX_TERMINAL_WIDTH,
    )
    rect = Rect(
        x=center.x - (width / 2.0),
        y=center.y - (_TERMINAL_HEIGHT / 2.0),
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
    row_y = node_geometry.rect.y + _HEADER_HEIGHT + _NODE_PADDING

    for port in node.inputs:
        geometry = _build_port_geometry(port, node_geometry, row_y)
        port_geometries[port.id] = geometry
        row_y += _PORT_ROW_HEIGHT

    row_y += _PORT_GROUP_GAP

    for port in node.outputs:
        geometry = _build_port_geometry(port, node_geometry, row_y)
        port_geometries[port.id] = geometry
        row_y += _PORT_ROW_HEIGHT

    return port_geometries


def _build_port_geometry(port: VisualPort, node_geometry: NodeGeometry, row_y: float) -> PortGeometry:
    name = truncate_text(port.name, _PORT_NAME_MAX_CHARS)
    detail = truncate_text(_port_detail_text(port), _PORT_DETAIL_MAX_CHARS)
    full_label = "\n".join(part for part in (port.name, _port_detail_text(port), port.kind) if part)
    rect = Rect(
        x=node_geometry.rect.x,
        y=row_y,
        width=node_geometry.rect.width,
        height=_PORT_ROW_HEIGHT,
    )
    anchor_x = rect.x if port.side == "input" else rect.x + rect.width
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
        lane_mid_x = (
            content_bounds.x - _CANVAS_MARGIN
            if end.x <= start.x
            else content_bounds.x + content_bounds.width + _CANVAS_MARGIN
        )
        points = (
            start,
            Point(start.x, lane_y),
            Point(lane_mid_x, lane_y),
            Point(end.x, lane_y),
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


def _measure_node_width(node: VisualNode) -> float:
    width = len(node.title) * _TITLE_CHAR_WIDTH + (_NODE_PADDING * 2.0)

    for port in (*node.inputs, *node.outputs):
        name = truncate_text(port.name, _PORT_NAME_MAX_CHARS)
        detail = truncate_text(_port_detail_text(port), _PORT_DETAIL_MAX_CHARS)
        port_width = max(
            len(name) * _PORT_PRIMARY_CHAR_WIDTH,
            len(detail) * _PORT_DETAIL_CHAR_WIDTH,
        )
        width = max(width, port_width + (_NODE_PADDING * 2.0) + 36.0)

    return _clamp(width, _MIN_NODE_WIDTH, _MAX_NODE_WIDTH)


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
