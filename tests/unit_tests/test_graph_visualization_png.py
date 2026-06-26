from pathlib import Path

from PIL import Image

from leapp.leapp_graph.visualization.geometry import EdgeGeometry, GraphGeometry, NodeGeometry, PortGeometry, Rect
from leapp.leapp_graph.visualization.layout import Point
from leapp.leapp_graph.visualization.png_renderer import write_png


def test_write_png_creates_nonempty_raster(tmp_path: Path):
    path = tmp_path / "graph.png"
    geometry = GraphGeometry(
        graph_name="demo",
        width=360.0,
        height=220.0,
        content_bounds=Rect(48.0, 48.0, 264.0, 124.0),
        nodes={
            "node:policy": NodeGeometry("node:policy", "policy", "jit-script", Rect(90.0, 60.0, 220.0, 120.0), Rect(90.0, 60.0, 220.0, 34.0)),
        },
        terminals={},
        ports={
            "port:policy:output:action": PortGeometry("port:policy:output:action", "node:policy", "output", "action", "[1, 4] float32", "command", Rect(200.0, 118.0, 110.0, 46.0), Point(310.0, 141.0), "action\n[1, 4] float32\ncommand"),
        },
        edges={
            "edge:self": EdgeGeometry("edge:self", "forward", "action", (Point(310.0, 141.0), Point(330.0, 141.0), Point(330.0, 170.0), Point(310.0, 170.0))),
        },
    )

    write_png(str(path), geometry)

    image = Image.open(path)
    assert image.size == (360, 220)
    assert image.getbbox() is not None
