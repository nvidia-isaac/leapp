import xml.etree.ElementTree as ET

from leapp.leapp_graph.visualization.geometry import EdgeGeometry, GraphGeometry, NodeGeometry, PortGeometry, Rect, TerminalGeometry
from leapp.leapp_graph.visualization.layout import Point
from leapp.leapp_graph.visualization.svg_renderer import COLORS, render_svg, write_svg


def _geometry():
    return GraphGeometry(
        graph_name="demo",
        width=500.0,
        height=260.0,
        content_bounds=Rect(48.0, 70.0, 404.0, 142.0),
        nodes={
            "node:policy": NodeGeometry("node:policy", "policy", "onnx-dynamo", Rect(140.0, 90.0, 240.0, 120.0), Rect(140.0, 90.0, 240.0, 34.0)),
        },
        terminals={
            "terminal:input:policy:obs": TerminalGeometry("terminal:input:policy:obs", "obs", "graph_input", Rect(48.0, 126.0, 80.0, 28.0), Point(128.0, 140.0)),
        },
        ports={
            "port:policy:input:obs": PortGeometry("port:policy:input:obs", "node:policy", "input", "obs", "[1, 12] float32", "state", Rect(140.0, 130.0, 110.0, 46.0), Point(140.0, 153.0), "obs\n[1, 12] float32\nstate"),
        },
        edges={
            "edge:input": EdgeGeometry("edge:input", "graph_input", "obs", (Point(128.0, 140.0), Point(134.0, 140.0), Point(134.0, 153.0), Point(140.0, 153.0))),
        },
    )


def test_render_svg_is_parseable_static_and_contains_port_details():
    svg = render_svg(_geometry())

    root = ET.fromstring(svg)
    assert root.tag.endswith("svg")
    assert root.attrib["viewBox"] == "0 0 500 260"
    assert "<script" not in svg.lower()
    assert "policy" in svg
    assert "obs" in svg
    assert "[1, 12] float32" in svg
    assert "state" in svg
    assert "marker-end" in svg


def test_render_svg_includes_port_title_and_renders_edges_before_nodes():
    svg = render_svg(_geometry())

    assert "<title>obs\n[1, 12] float32\nstate</title>" in svg
    assert svg.index('id="edge:input"') < svg.index('id="node:policy"')


def test_write_svg_writes_utf8_text(tmp_path):
    path = tmp_path / "graph.svg"

    write_svg(str(path), _geometry())

    assert path.read_text(encoding="utf-8").startswith("<svg")
    assert COLORS["warp"] == "#76B900"
