from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFont

from .geometry import EdgeGeometry, GraphGeometry, Point, Rect, _visible_port_kind
from .svg_renderer import COLORS

_SCALE_FACTOR = 2
_NODE_RADIUS = 6
_TERMINAL_RADIUS = 6
_NODE_PADDING = 14.0
_TERMINAL_PADDING = 12.0
_PORT_DOT_RADIUS = 4.0
_EDGE_WIDTH = 2.5
_TERMINAL_BORDER_WIDTH = 1.5
_NODE_BORDER_WIDTH = 1.5
_PORT_ACCENT_WIDTH = 3.0
_ARROW_LENGTH = 10.0
_ARROW_HALF_WIDTH = 3.5
_FONT_REGULAR_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def write_png(path: str, geometry: GraphGeometry) -> None:
    width = int(geometry.width)
    height = int(geometry.height)
    scaled_size = (width * _SCALE_FACTOR, height * _SCALE_FACTOR)
    image = Image.new("RGBA", scaled_size, _color("background"))
    draw = ImageDraw.Draw(image)
    fonts = _load_fonts()

    for edge in geometry.edges.values():
        _draw_edge(draw, edge)

    for terminal in geometry.terminals.values():
        _draw_terminal(draw, terminal.rect, terminal.anchor, terminal.title, terminal.kind, fonts)

    for node in geometry.nodes.values():
        _draw_node_card(draw, node.rect, node.header_rect)
        if node.backend:
            _draw_backend_badge(draw, node.header_rect, node.backend, fonts)
        _draw_text(
            draw,
            Point(node.header_rect.x + 14.0, node.header_rect.y + 22.0),
            node.title,
            fill=_color("text"),
            font=fonts["title"],
            anchor="ls",
        )

    for port in sorted(geometry.ports.values(), key=lambda port: (port.rect.y, port.rect.x, port.id)):
        _draw_port(draw, port, fonts)

    final_image = image.resize((width, height), Image.Resampling.LANCZOS)
    final_image.save(path, format="PNG")


def sample_cubic(points: tuple[Point, Point, Point, Point], samples: int = 24) -> list[Point]:
    start, control1, control2, end = points
    sampled: list[Point] = []
    for index in range(samples + 1):
        t = index / samples
        one_minus_t = 1.0 - t
        x = (
            (one_minus_t ** 3) * start.x
            + 3.0 * (one_minus_t ** 2) * t * control1.x
            + 3.0 * one_minus_t * (t ** 2) * control2.x
            + (t ** 3) * end.x
        )
        y = (
            (one_minus_t ** 3) * start.y
            + 3.0 * (one_minus_t ** 2) * t * control1.y
            + 3.0 * one_minus_t * (t ** 2) * control2.y
            + (t ** 3) * end.y
        )
        sampled.append(Point(x, y))
    return sampled


def _draw_edge(draw: ImageDraw.ImageDraw, edge: EdgeGeometry) -> None:
    color = _color(_edge_color_name(edge.kind))
    if edge.kind == "feedback":
        route = list(edge.points)
    else:
        route = sample_cubic(edge.points)  # type: ignore[arg-type]

    draw.line(_scaled_points(route), fill=color, width=_scaled_width(_EDGE_WIDTH), joint="curve")
    _draw_arrowhead(draw, route[-2], route[-1], color)


def _draw_arrowhead(draw: ImageDraw.ImageDraw, start: Point, end: Point, color: tuple[int, int, int, int]) -> None:
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.hypot(dx, dy)
    if length == 0:
        return

    unit_x = dx / length
    unit_y = dy / length
    perp_x = -unit_y
    perp_y = unit_x
    tip = _scaled_point(end)
    base_x = end.x - (unit_x * _ARROW_LENGTH)
    base_y = end.y - (unit_y * _ARROW_LENGTH)
    left = _scaled_point(Point(base_x + (perp_x * _ARROW_HALF_WIDTH), base_y + (perp_y * _ARROW_HALF_WIDTH)))
    right = _scaled_point(Point(base_x - (perp_x * _ARROW_HALF_WIDTH), base_y - (perp_y * _ARROW_HALF_WIDTH)))
    draw.polygon([left, tip, right], fill=color)


def _draw_terminal(
    draw: ImageDraw.ImageDraw,
    rect: Rect,
    anchor: Point,
    title: str,
    kind: str,
    fonts: dict[str, ImageFont.ImageFont | ImageFont.FreeTypeFont],
) -> None:
    color = _color(kind)
    _rounded_rect(draw, rect, fill=(color[0], color[1], color[2], 31), outline=color, width=_TERMINAL_BORDER_WIDTH, radius=_TERMINAL_RADIUS)
    anchor_on_right = anchor.x >= rect.x + rect.width
    text_x = rect.x + _TERMINAL_PADDING if anchor_on_right else rect.x + rect.width - _TERMINAL_PADDING
    _draw_text(
        draw,
        Point(text_x, rect.y + rect.height / 2.0 + 5.0),
        title,
        fill=color,
        font=fonts["label_bold"],
        anchor="ls" if anchor_on_right else "rs",
    )


def _draw_node_card(draw: ImageDraw.ImageDraw, rect: Rect, header_rect: Rect) -> None:
    _rounded_rect(draw, rect, fill=_color("node"), outline=_color("node_border"), width=_NODE_BORDER_WIDTH, radius=_NODE_RADIUS)
    _rounded_rect(draw, header_rect, fill=_color("header"), outline=None, width=0, radius=_NODE_RADIUS)
    bottom_rect = Rect(header_rect.x, header_rect.y + header_rect.height - 8.0, header_rect.width, 8.0)
    _scaled_rect(draw, bottom_rect, fill=_color("header"))


def _draw_backend_badge(
    draw: ImageDraw.ImageDraw,
    header_rect: Rect,
    backend: str,
    fonts: dict[str, ImageFont.ImageFont | ImageFont.FreeTypeFont],
) -> None:
    if backend.startswith("torch"):
        fill = _color("torch")
    elif backend.startswith("warp"):
        fill = _color("warp")
    else:
        fill = _color("unknown_backend")
    _draw_text(
        draw,
        Point(header_rect.x + header_rect.width - 14.0, header_rect.y + 22.0),
        backend,
        fill=fill,
        font=fonts["label_bold_small"],
        anchor="rs",
    )


def _draw_port(
    draw: ImageDraw.ImageDraw,
    port,
    fonts: dict[str, ImageFont.ImageFont | ImageFont.FreeTypeFont],
) -> None:
    accent = _accent_color(port.kind)
    line_x = port.rect.x if port.side == "input" else port.rect.x + port.rect.width - 1.5
    text_x = port.rect.x + _NODE_PADDING if port.side == "input" else port.rect.x + port.rect.width - _NODE_PADDING
    anchor = "ls" if port.side == "input" else "rs"

    draw.ellipse(_ellipse_bounds(port.anchor, _PORT_DOT_RADIUS), fill=accent)
    draw.line(
        _scaled_points(
            [
                Point(line_x, port.rect.y + 6.0),
                Point(line_x, port.rect.y + port.rect.height - 6.0),
            ]
        ),
        fill=accent,
        width=_scaled_width(_PORT_ACCENT_WIDTH),
    )
    _draw_text(draw, Point(text_x, port.rect.y + 18.0), port.name, fill=_color("text"), font=fonts["label_bold"], anchor=anchor)
    _draw_text(draw, Point(text_x, port.rect.y + 34.0), port.detail, fill=_color("secondary_text"), font=fonts["label"], anchor=anchor)

    visible_kind = _visible_port_kind(port)
    if visible_kind:
        kind_x = port.rect.x + _NODE_PADDING if port.side == "output" else port.rect.x + port.rect.width - _NODE_PADDING
        kind_anchor = "ls" if port.side == "output" else "rs"
        _draw_text(draw, Point(kind_x, port.rect.y + 18.0), visible_kind, fill=accent, font=fonts["label"], anchor=kind_anchor)


def _edge_color_name(kind: str) -> str:
    if kind == "feedback":
        return "feedback_edge"
    if kind == "graph_input":
        return "graph_input"
    if kind == "graph_output":
        return "graph_output"
    return "forward_edge"


def _load_fonts() -> dict[str, ImageFont.ImageFont | ImageFont.FreeTypeFont]:
    default_font = ImageFont.load_default()
    try:
        regular = ImageFont.truetype(_FONT_REGULAR_PATH, size=_scaled_width(13))
    except OSError:
        regular = default_font
    try:
        bold = ImageFont.truetype(_FONT_BOLD_PATH, size=_scaled_width(13))
    except OSError:
        bold = default_font
    try:
        title = ImageFont.truetype(_FONT_BOLD_PATH, size=_scaled_width(14))
    except OSError:
        title = bold
    try:
        badge = ImageFont.truetype(_FONT_BOLD_PATH, size=_scaled_width(12))
    except OSError:
        badge = default_font
    return {
        "label": regular,
        "label_bold": bold,
        "label_bold_small": badge,
        "title": title,
    }


def _draw_text(
    draw: ImageDraw.ImageDraw,
    point: Point,
    text: str,
    *,
    fill: tuple[int, int, int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    anchor: str,
) -> None:
    draw.text(_scaled_point(point), text, fill=fill, font=font, anchor=anchor)


def _rounded_rect(
    draw: ImageDraw.ImageDraw,
    rect: Rect,
    *,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int] | None,
    width: float,
    radius: float,
) -> None:
    draw.rounded_rectangle(
        _scaled_box(rect),
        radius=_scaled_width(radius),
        fill=fill,
        outline=outline,
        width=_scaled_width(width) if outline is not None else 0,
    )


def _scaled_rect(draw: ImageDraw.ImageDraw, rect: Rect, *, fill: tuple[int, int, int, int]) -> None:
    draw.rectangle(_scaled_box(rect), fill=fill)


def _ellipse_bounds(center: Point, radius: float) -> tuple[float, float, float, float]:
    return _scaled_box(Rect(center.x - radius, center.y - radius, radius * 2.0, radius * 2.0))


def _scaled_box(rect: Rect) -> tuple[int, int, int, int]:
    return (
        _scale_value(rect.x),
        _scale_value(rect.y),
        _scale_value(rect.x + rect.width),
        _scale_value(rect.y + rect.height),
    )


def _scaled_point(point: Point) -> tuple[int, int]:
    return (_scale_value(point.x), _scale_value(point.y))


def _scaled_points(points: list[Point]) -> list[tuple[int, int]]:
    return [_scaled_point(point) for point in points]


def _scaled_width(value: float) -> int:
    return max(1, int(round(value * _SCALE_FACTOR)))


def _scale_value(value: float) -> int:
    return int(round(value * _SCALE_FACTOR))


def _color(name: str) -> tuple[int, int, int, int]:
    return _hex_to_rgba(COLORS[name])


def _accent_color(name: str | None) -> tuple[int, int, int, int]:
    if not name:
        return _color("node_border")
    return _hex_to_rgba(COLORS.get(name, COLORS["node_border"]))


def _hex_to_rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
        alpha,
    )
