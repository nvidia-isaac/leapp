from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from leapp.leapp_graph.visualization.visualize import visualize_graph


@dataclass
class FakeTensorDescription:
    name_str: str
    dtype: str
    shape: tuple[int, ...]
    tag: str | None = None
    semantics: dict | None = None

    def get_semantics(self):
        return self.semantics or {}


@dataclass
class FakeNode:
    name: str
    inputs: list[FakeTensorDescription]
    outputs: list[FakeTensorDescription]
    backend: str | None = "jit-script"
    node_index: int = 0


def test_visualize_graph_writes_svg_and_png(tmp_path: Path):
    node = FakeNode(
        name="policy",
        inputs=[FakeTensorDescription("obs", "float32", (1, 12))],
        outputs=[FakeTensorDescription("action", "float32", (1, 4), semantics={"kind": "command"})],
        backend="jit-script",
    )

    visualize_graph(
        nodes={"policy": node},
        connections=[],
        feedback_connections=[],
        inputs=["policy/obs"],
        outputs=["policy/action"],
        save_path=str(tmp_path),
        graph_name="demo",
    )

    svg_path = tmp_path / "demo.svg"
    png_path = tmp_path / "demo.png"
    assert svg_path.exists()
    assert png_path.exists()
    assert "policy" in svg_path.read_text(encoding="utf-8")
    assert "command" in svg_path.read_text(encoding="utf-8")
    assert Image.open(png_path).size[0] > 0
