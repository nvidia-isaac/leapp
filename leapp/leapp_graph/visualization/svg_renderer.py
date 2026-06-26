from __future__ import annotations

from html import escape

from .geometry import EdgeGeometry, GraphGeometry, Point, PortGeometry, Rect


COLORS = {
    "background": "#F7F8FA",
    "node": "#FFFFFF",
    "node_border": "#C9D2DE",
    "header": "#EDF2F7",
    "text": "#18212F",
    "secondary_text": "#5E6B7A",
    "forward_edge": "#566273",
    "feedback_edge": "#B42318",
    "graph_input": "#2F855A",
    "graph_output": "#C2410C",
    "torch": "#2563EB",
    "warp": "#76B900",
    "unknown_backend": "#667085",
}

_FONT_FAMILY = (
    'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
)


def render_svg(geometry: GraphGeometry) -> str:
    width = int(geometry.width)
    height = int(geometry.height)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        _render_defs(),
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{COLORS["background"]}" />',
    ]
    for edge in geometry.edges.values():
        parts.append(_render_edge(edge))
    for terminal in geometry.terminals.values():
        parts.append(
            _render_terminal(
                rect=terminal.rect,
                anchor=terminal.anchor,
                title=terminal.title,
                kind=terminal.kind,
                terminal_id=terminal.id,
            )
        )
    for node in geometry.nodes.values():
        parts.append(_render_node(geometry, node.id))
    parts.append("</svg>")
    return "".join(parts)


def write_svg(path: str, geometry: GraphGeometry) -> None:
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(render_svg(geometry))


def _render_defs() -> str:
    return (
        "<defs>"
        f'{_marker("forward-arrow", COLORS["forward_edge"])}'
        f'{_marker("feedback-arrow", COLORS["feedback_edge"])}'
        f'{_marker("graph-input-arrow", COLORS["graph_input"])}'
        f'{_marker("graph-output-arrow", COLORS["graph_output"])}'
        "</defs>"
    )


def _marker(marker_id: str, color: str) -> str:
    return (
        f'<marker id="{marker_id}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
        f'<path d="M 0 0 L 10 3.5 L 0 7 z" fill="{color}" />'
        "</marker>"
    )


def _render_edge(edge: EdgeGeometry) -> str:
    color = COLORS[_edge_color_name(edge.kind)]
    marker_id = _edge_marker_id(edge.kind)
    return (
        f'<path id="{escape(edge.id, quote=True)}" d="{_edge_path_data(edge)}" fill="none" stroke="{color}" stroke-width="2.5" '
        f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#{marker_id})" />'
    )


def _edge_path_data(edge: EdgeGeometry) -> str:
    if len(edge.points) == 4:
        start, control1, control2, end = edge.points
        return f"M {_fmt_point(start)} C {_fmt_point(control1)} {_fmt_point(control2)} {_fmt_point(end)}"
    if len(edge.points) == 5:
        return " ".join(
            [f"M {_fmt_point(edge.points[0])}"]
            + [f"L {_fmt_point(point)}" for point in edge.points[1:]]
        )
    raise ValueError(f"Unsupported edge point count for {edge.id}: {len(edge.points)}")


def _render_terminal(rect: Rect, anchor: Point, title: str, kind: str, terminal_id: str) -> str:
    color = COLORS[kind]
    anchor_on_right = anchor.x >= rect.x + rect.width
    text_x = rect.x + 12.0 if anchor_on_right else rect.x + rect.width - 12.0
    text_anchor = "start" if anchor_on_right else "end"
    return (
        f'<g id="{escape(terminal_id, quote=True)}">'
        f'<rect x="{_fmt(rect.x)}" y="{_fmt(rect.y)}" width="{_fmt(rect.width)}" height="{_fmt(rect.height)}" rx="14" fill="{color}" opacity="0.12" stroke="{color}" stroke-width="1.5" />'
        f'<text x="{_fmt(text_x)}" y="{_fmt(rect.y + rect.height / 2.0 + 5.0)}" fill="{color}" font-family="{escape(_FONT_FAMILY, quote=True)}" font-size="13" font-weight="600" text-anchor="{text_anchor}">{escape(title)}</text>'
        "</g>"
    )


def _render_node(geometry: GraphGeometry, node_id: str) -> str:
    node = geometry.nodes[node_id]
    port_ids = [port_id for port_id, port in geometry.ports.items() if port.node_id == node_id]
    port_ids.sort(key=lambda port_id: (geometry.ports[port_id].rect.y, geometry.ports[port_id].rect.x, port_id))
    parts = [
        f'<g id="{escape(node.id, quote=True)}">',
        f'<rect x="{_fmt(node.rect.x)}" y="{_fmt(node.rect.y)}" width="{_fmt(node.rect.width)}" height="{_fmt(node.rect.height)}" rx="8" fill="{COLORS["node"]}" stroke="{COLORS["node_border"]}" stroke-width="1.5" />',
        f'<rect x="{_fmt(node.header_rect.x)}" y="{_fmt(node.header_rect.y)}" width="{_fmt(node.header_rect.width)}" height="{_fmt(node.header_rect.height)}" rx="8" fill="{COLORS["header"]}" />',
        f'<rect x="{_fmt(node.header_rect.x)}" y="{_fmt(node.header_rect.y + node.header_rect.height - 8.0)}" width="{_fmt(node.header_rect.width)}" height="8" fill="{COLORS["header"]}" />',
        f'<text x="{_fmt(node.header_rect.x + 14.0)}" y="{_fmt(node.header_rect.y + 22.0)}" fill="{COLORS["text"]}" font-family="{escape(_FONT_FAMILY, quote=True)}" font-size="14" font-weight="700">{escape(node.title)}</text>',
    ]
    badge = _backend_badge(node)
    if badge:
        parts.append(badge)
    parts.extend(_render_port(geometry.ports[port_id]) for port_id in port_ids)
    parts.append("</g>")
    return "".join(parts)


def _render_port(port: PortGeometry) -> str:
    accent = COLORS.get(port.kind or "", COLORS["node_border"])
    line_x = port.rect.x if port.side == "input" else port.rect.x + port.rect.width - 1.5
    text_x = port.rect.x + 14.0 if port.side == "input" else port.rect.x + port.rect.width - 14.0
    text_anchor = "start" if port.side == "input" else "end"
    return (
        f'<g id="{escape(port.id, quote=True)}">'
        f"<title>{escape(port.full_label)}</title>"
        f'<rect x="{_fmt(port.rect.x)}" y="{_fmt(port.rect.y)}" width="{_fmt(port.rect.width)}" height="{_fmt(port.rect.height)}" fill="transparent" />'
        f'<circle cx="{_fmt(port.anchor.x)}" cy="{_fmt(port.anchor.y)}" r="4" fill="{accent}" />'
        f'<line x1="{_fmt(line_x)}" y1="{_fmt(port.rect.y + 6.0)}" x2="{_fmt(line_x)}" y2="{_fmt(port.rect.y + port.rect.height - 6.0)}" stroke="{accent}" stroke-width="3" stroke-linecap="round" />'
        f'<text x="{_fmt(text_x)}" y="{_fmt(port.rect.y + 18.0)}" fill="{COLORS["text"]}" font-family="{escape(_FONT_FAMILY, quote=True)}" font-size="13" font-weight="600" text-anchor="{text_anchor}">{escape(port.name)}</text>'
        f'<text x="{_fmt(text_x)}" y="{_fmt(port.rect.y + 34.0)}" fill="{COLORS["secondary_text"]}" font-family="{escape(_FONT_FAMILY, quote=True)}" font-size="12" text-anchor="{text_anchor}">{escape(port.detail)}</text>'
        f'{_render_port_kind(port, accent)}'
        "</g>"
    )


def _render_port_kind(port: PortGeometry, accent: str) -> str:
    if not port.kind:
        return ""
    return (
        f'<text x="{_fmt(port.rect.x + port.rect.width - 14.0)}" y="{_fmt(port.rect.y + 18.0)}" fill="{accent}" font-family="{escape(_FONT_FAMILY, quote=True)}" font-size="12" font-weight="600" text-anchor="end">{escape(port.kind)}</text>'
    )


def _backend_badge(node) -> str | None:
    if node.backend is None:
        return None
    if node.backend.startswith("torch"):
        color = COLORS["torch"]
    elif node.backend.startswith("warp"):
        color = COLORS["warp"]
    else:
        color = COLORS["unknown_backend"]
    return (
        f'<text x="{_fmt(node.header_rect.x + node.header_rect.width - 14.0)}" y="{_fmt(node.header_rect.y + 22.0)}" fill="{color}" font-family="{escape(_FONT_FAMILY, quote=True)}" font-size="12" font-weight="600" text-anchor="end">{escape(node.backend)}</text>'
    )


def _edge_color_name(kind: str) -> str:
    if kind == "feedback":
        return "feedback_edge"
    if kind == "graph_input":
        return "graph_input"
    if kind == "graph_output":
        return "graph_output"
    return "forward_edge"


def _edge_marker_id(kind: str) -> str:
    if kind == "feedback":
        return "feedback-arrow"
    if kind == "graph_input":
        return "graph-input-arrow"
    if kind == "graph_output":
        return "graph-output-arrow"
    return "forward-arrow"


def _fmt(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _fmt_point(point: Point) -> str:
    return f"{_fmt(point.x)} {_fmt(point.y)}"
