from pathlib import Path

from PIL import ImageFont

from leapp_visualization.geometry import (
    EdgeGeometry,
    GraphGeometry,
    NodeGeometry,
    PortGeometry,
    Rect,
    TerminalGeometry,
)
from leapp_visualization.layout import Point
from leapp_visualization import png_renderer
from leapp_visualization.png_renderer import sample_cubic, write_png


def _geometry() -> GraphGeometry:
    return GraphGeometry(
        graph_name="demo",
        width=500.0,
        height=260.0,
        content_bounds=Rect(48.0, 70.0, 404.0, 142.0),
        nodes={
            "node:policy": NodeGeometry(
                "node:policy",
                "policy",
                "onnx-dynamo",
                Rect(140.0, 90.0, 240.0, 120.0),
                Rect(140.0, 90.0, 240.0, 34.0),
            ),
        },
        terminals={
            "terminal:input:policy:obs": TerminalGeometry(
                "terminal:input:policy:obs",
                "obs",
                "graph_input",
                Rect(48.0, 126.0, 80.0, 28.0),
                Point(128.0, 140.0),
            ),
            "terminal:output:policy:action": TerminalGeometry(
                "terminal:output:policy:action",
                "action",
                "graph_output",
                Rect(384.0, 162.0, 96.0, 28.0),
                Point(384.0, 176.0),
            ),
        },
        ports={
            "port:policy:input:obs": PortGeometry(
                "port:policy:input:obs",
                "node:policy",
                "input",
                "obs",
                "[1, 12] float32",
                "state",
                Rect(140.0, 130.0, 110.0, 46.0),
                Point(140.0, 153.0),
                "obs\n[1, 12] float32\nstate",
            ),
            "port:policy:output:action": PortGeometry(
                "port:policy:output:action",
                "node:policy",
                "output",
                "action",
                "[1, 4] float32",
                "command",
                Rect(270.0, 130.0, 110.0, 46.0),
                Point(380.0, 153.0),
                "action\n[1, 4] float32\ncommand",
            ),
        },
        edges={
            "edge:input": EdgeGeometry(
                "edge:input",
                "graph_input",
                "obs",
                (Point(128.0, 140.0), Point(134.0, 140.0), Point(134.0, 153.0), Point(140.0, 153.0)),
            ),
            "edge:feedback": EdgeGeometry(
                "edge:feedback",
                "feedback",
                "state",
                (Point(380.0, 153.0), Point(410.0, 153.0), Point(410.0, 92.0), Point(110.0, 92.0), Point(110.0, 153.0)),
            ),
            "edge:output": EdgeGeometry(
                "edge:output",
                "graph_output",
                "action",
                (Point(380.0, 176.0), Point(382.0, 176.0), Point(382.0, 176.0), Point(384.0, 176.0)),
            ),
        },
    )


def test_sample_cubic_returns_requested_samples_with_endpoints():
    points = (
        Point(0.0, 0.0),
        Point(10.0, 0.0),
        Point(10.0, 20.0),
        Point(20.0, 20.0),
    )

    sampled = sample_cubic(points, samples=24)

    assert len(sampled) == 25
    assert sampled[0] == points[0]
    assert sampled[-1] == points[-1]


def test_draw_edge_samples_forward_and_graph_output_routes_for_arrowheads(monkeypatch):
    sampled_route = [Point(1.0, 1.0), Point(2.0, 2.0), Point(3.0, 3.0)]
    sample_calls = []
    arrow_calls = []
    line_calls = []

    class DrawStub:
        def line(self, points, *, fill, width, joint):
            line_calls.append((points, fill, width, joint))

    def fake_sample(points, samples=24):
        sample_calls.append((points, samples))
        return list(sampled_route)

    def fake_arrowhead(draw, start, end, color):
        arrow_calls.append((start, end, color))

    monkeypatch.setattr(png_renderer, "sample_cubic", fake_sample)
    monkeypatch.setattr(png_renderer, "_draw_arrowhead", fake_arrowhead)

    forward_edge = _geometry().edges["edge:input"]
    output_edge = _geometry().edges["edge:output"]
    draw = DrawStub()

    png_renderer._draw_edge(draw, forward_edge)
    png_renderer._draw_edge(draw, output_edge)

    assert sample_calls == [
        (forward_edge.points, 24),
        (output_edge.points, 24),
    ]
    assert len(line_calls) == 2
    assert arrow_calls == [
        (sampled_route[-2], sampled_route[-1], png_renderer._color("graph_input")),
        (sampled_route[-2], sampled_route[-1], png_renderer._color("graph_output")),
    ]


def test_draw_edge_renders_feedback_with_line_segments_and_raw_arrowhead(monkeypatch):
    arrow_calls = []
    line_calls = []

    class DrawStub:
        def line(self, points, *, fill, width, joint):
            line_calls.append((points, fill, width, joint))

    def fail_sample(*args, **kwargs):
        raise AssertionError("feedback edges should not use cubic sampling")

    def fake_arrowhead(draw, start, end, color):
        arrow_calls.append((start, end, color))

    edge = _geometry().edges["edge:feedback"]

    monkeypatch.setattr(png_renderer, "sample_cubic", fail_sample)
    monkeypatch.setattr(png_renderer, "_draw_arrowhead", fake_arrowhead)

    png_renderer._draw_edge(DrawStub(), edge)

    assert len(line_calls) == 1
    points, fill, width, joint = line_calls[0]
    assert len(points) == len(edge.points)
    assert fill == png_renderer._color("feedback_edge")
    assert width == png_renderer._scaled_width(2.5)
    assert joint == "curve"
    assert arrow_calls == [
        (edge.points[-2], edge.points[-1], png_renderer._color("feedback_edge")),
    ]


def test_write_png_renders_on_scaled_canvas_then_downsamples(monkeypatch, tmp_path: Path):
    calls = {"new": None, "resize": None, "save": None}

    class FinalImageStub:
        def save(self, path, format):
            calls["save"] = (path, format)

    class ImageStub:
        def resize(self, size, resample):
            calls["resize"] = (size, resample)
            return FinalImageStub()

    monkeypatch.setattr(
        png_renderer.Image,
        "new",
        lambda mode, size, color: calls.__setitem__("new", (mode, size, color)) or ImageStub(),
    )
    monkeypatch.setattr(png_renderer.ImageDraw, "Draw", lambda image: object())
    monkeypatch.setattr(png_renderer, "_load_fonts", lambda: {"label": object(), "label_bold": object(), "label_bold_small": object(), "title": object()})
    monkeypatch.setattr(png_renderer, "_draw_edge", lambda draw, edge: None)
    monkeypatch.setattr(png_renderer, "_draw_terminal", lambda *args, **kwargs: None)
    monkeypatch.setattr(png_renderer, "_draw_node_card", lambda *args, **kwargs: None)
    monkeypatch.setattr(png_renderer, "_draw_backend_badge", lambda *args, **kwargs: None)
    monkeypatch.setattr(png_renderer, "_draw_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(png_renderer, "_draw_port", lambda *args, **kwargs: None)

    path = tmp_path / "graph.png"

    write_png(str(path), _geometry())

    assert calls["new"] == ("RGBA", (1000, 520), png_renderer._color("background"))
    assert calls["resize"] == ((500, 260), png_renderer.Image.Resampling.LANCZOS)
    assert calls["save"] == (str(path), "PNG")


def test_write_png_passes_terminal_and_port_geometry_to_drawers(monkeypatch, tmp_path: Path):
    terminal_calls = []
    port_calls = []

    monkeypatch.setattr(png_renderer, "_load_fonts", lambda: {"label": object(), "label_bold": object(), "label_bold_small": object(), "title": object()})
    monkeypatch.setattr(png_renderer, "_draw_edge", lambda draw, edge: None)
    monkeypatch.setattr(png_renderer, "_draw_terminal", lambda draw, rect, anchor, title, kind, fonts: terminal_calls.append((rect, anchor, title, kind, fonts)))
    monkeypatch.setattr(png_renderer, "_draw_node_card", lambda *args, **kwargs: None)
    monkeypatch.setattr(png_renderer, "_draw_backend_badge", lambda *args, **kwargs: None)
    monkeypatch.setattr(png_renderer, "_draw_text", lambda *args, **kwargs: None)
    monkeypatch.setattr(png_renderer, "_draw_port", lambda draw, port, fonts: port_calls.append((port, fonts)))

    path = tmp_path / "graph.png"
    geometry = _geometry()

    write_png(str(path), geometry)

    assert terminal_calls == [
        (
            geometry.terminals["terminal:input:policy:obs"].rect,
            geometry.terminals["terminal:input:policy:obs"].anchor,
            "obs",
            "graph_input",
            terminal_calls[0][4],
        ),
        (
            geometry.terminals["terminal:output:policy:action"].rect,
            geometry.terminals["terminal:output:policy:action"].anchor,
            "action",
            "graph_output",
            terminal_calls[1][4],
        ),
    ]
    assert [port.id for port, _fonts in port_calls] == [
        "port:policy:input:obs",
        "port:policy:output:action",
    ]
    assert port_calls[0][0].full_label == "obs\n[1, 12] float32\nstate"
    assert port_calls[1][0].kind == "command"


def test_load_fonts_prefers_dejavu_paths_before_default_fallback(monkeypatch):
    calls = []
    default_font = ImageFont.load_default()

    def fake_truetype(path, size):
        calls.append(("truetype", path, size))
        raise OSError("missing test font")

    def fake_default():
        calls.append(("default",))
        return default_font

    monkeypatch.setattr(png_renderer.ImageFont, "truetype", fake_truetype)
    monkeypatch.setattr(png_renderer.ImageFont, "load_default", fake_default)

    fonts = png_renderer._load_fonts()

    assert calls[:4] == [
        ("truetype", png_renderer._FONT_REGULAR_PATH, png_renderer._scaled_width(13)),
        ("truetype", png_renderer._FONT_BOLD_PATH, png_renderer._scaled_width(13)),
        ("truetype", png_renderer._FONT_BOLD_PATH, png_renderer._scaled_width(14)),
        ("truetype", png_renderer._FONT_BOLD_PATH, png_renderer._scaled_width(12)),
    ]
    assert calls[4:] == [("default",)]
    assert fonts == {
        "label": default_font,
        "label_bold": default_font,
        "label_bold_small": default_font,
        "title": default_font,
    }


def test_write_png_creates_nonempty_raster(tmp_path: Path):
    path = tmp_path / "graph.png"

    write_png(str(path), _geometry())

    image = png_renderer.Image.open(path)
    assert image.size == (500, 260)
    assert image.getbbox() is not None
