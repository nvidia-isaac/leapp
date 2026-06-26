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
            "terminal:output:policy:action": TerminalGeometry("terminal:output:policy:action", "action", "graph_output", Rect(384.0, 162.0, 96.0, 28.0), Point(384.0, 176.0)),
        },
        ports={
            "port:policy:input:obs": PortGeometry("port:policy:input:obs", "node:policy", "input", "obs", "[1, 12] float32", "state", Rect(140.0, 130.0, 110.0, 46.0), Point(140.0, 153.0), "obs\n[1, 12] float32\nstate"),
            "port:policy:output:action": PortGeometry("port:policy:output:action", "node:policy", "output", "action", "[1, 4] float32", "command", Rect(270.0, 130.0, 110.0, 46.0), Point(380.0, 153.0), "action\n[1, 4] float32\ncommand"),
        },
        edges={
            "edge:input": EdgeGeometry("edge:input", "graph_input", "obs", (Point(128.0, 140.0), Point(134.0, 140.0), Point(134.0, 153.0), Point(140.0, 153.0))),
            "edge:feedback": EdgeGeometry("edge:feedback", "feedback", "state", (Point(380.0, 153.0), Point(410.0, 153.0), Point(410.0, 92.0), Point(110.0, 92.0), Point(110.0, 153.0))),
            "edge:output": EdgeGeometry("edge:output", "graph_output", "action", (Point(380.0, 176.0), Point(382.0, 176.0), Point(382.0, 176.0), Point(384.0, 176.0))),
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


def test_render_svg_uses_fixed_size_arrow_markers():
    svg = render_svg(_geometry())

    assert 'markerUnits="userSpaceOnUse"' in svg
    assert 'markerUnits="strokeWidth"' not in svg
    assert 'refX="10"' in svg


def test_render_svg_includes_port_title_and_renders_edges_before_nodes():
    svg = render_svg(_geometry())

    assert "<title>obs\n[1, 12] float32\nstate</title>" in svg
    assert svg.index('id="edge:input"') < svg.index('id="node:policy"')


def test_render_svg_right_aligns_output_port_text_inside_output_row():
    svg = render_svg(_geometry())

    assert 'id="port:policy:output:action"' in svg
    assert 'text-anchor="end">action</text>' in svg
    assert 'text-anchor="end">[1, 4] float32</text>' in svg
    assert 'text-anchor="end">command</text>' in svg


def test_render_svg_places_port_kind_below_shape_detail():
    svg = render_svg(_geometry())
    root = ET.fromstring(svg)
    namespace = {"svg": "http://www.w3.org/2000/svg"}

    output_port = root.find('.//svg:g[@id="port:policy:output:action"]', namespace)
    assert output_port is not None

    text_positions = {
        element.text: (element.attrib["x"], element.attrib["y"])
        for element in output_port.findall("svg:text", namespace)
        if element.text in {"action", "[1, 4] float32", "command"}
    }

    assert float(text_positions["command"][1]) > float(text_positions["[1, 4] float32"][1])


def test_render_svg_serializes_five_point_feedback_edges_and_graph_output_terminal():
    svg = render_svg(_geometry())

    assert 'id="edge:feedback"' in svg
    assert 'd="M 380 153 L 410 153 L 410 92 L 110 92 L 110 153"' in svg
    assert 'marker-end="url(#feedback-arrow)"' in svg
    assert 'id="edge:output"' in svg
    assert 'marker-end="url(#graph-output-arrow)"' in svg
    assert 'id="terminal:output:policy:action"' in svg
    assert f'fill="{COLORS["graph_output"]}"' in svg


def test_write_svg_writes_utf8_text(tmp_path):
    path = tmp_path / "graph.svg"

    write_svg(str(path), _geometry())

    assert path.read_text(encoding="utf-8").startswith("<svg")
    assert COLORS["warp"] == "#76B900"
