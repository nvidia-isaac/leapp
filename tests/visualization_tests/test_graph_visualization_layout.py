from leapp_visualization.layout import compute_layered_layout
from leapp_visualization.model import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal


def _node(name):
    return VisualNode(id=f"node:{name}", title=name, backend="jit-script")


def test_layered_layout_places_pipeline_left_to_right():
    graph = VisualGraph(
        nodes=(_node("a"), _node("b")),
        terminals=(
            VisualTerminal("terminal:input:a:x", "graph_input", "x", "node:a", "port:a:input:x"),
            VisualTerminal("terminal:output:b:y", "graph_output", "y", "node:b", "port:b:output:y"),
        ),
        edges=(
            VisualEdge("e0", "graph_input", "terminal:input:a:x", "node:a", None, "port:a:input:x", "x"),
            VisualEdge("e1", "forward", "node:a", "node:b", "port:a:output:y", "port:b:input:y", "y"),
            VisualEdge("e2", "graph_output", "node:b", "terminal:output:b:y", "port:b:output:y", None, "y"),
        ),
    )

    layout = compute_layered_layout(graph)

    assert layout.centers["terminal:input:a:x"].x < layout.centers["node:a"].x
    assert layout.centers["node:a"].x < layout.centers["node:b"].x
    assert layout.centers["node:b"].x < layout.centers["terminal:output:b:y"].x
    assert all(edge_id != "e_feedback" for edge_id in layout.forward_edge_ids)


def test_layered_layout_is_deterministic_for_same_graph():
    graph = VisualGraph(
        nodes=(_node("a"), _node("b"), _node("c")),
        terminals=(),
        edges=(
            VisualEdge("e0", "forward", "node:a", "node:c"),
            VisualEdge("e1", "forward", "node:b", "node:c"),
            VisualEdge("e_feedback", "feedback", "node:c", "node:a"),
        ),
    )

    first = compute_layered_layout(graph)
    second = compute_layered_layout(graph)

    assert first == second
    assert first.centers["node:a"].x < first.centers["node:c"].x
    assert first.centers["node:b"].x < first.centers["node:c"].x
    assert "e_feedback" not in first.forward_edge_ids


def test_layered_layout_handles_edgeless_graph_with_stable_grid():
    graph = VisualGraph(nodes=(_node("a"), _node("b")), terminals=(), edges=())

    layout = compute_layered_layout(graph)

    assert tuple(layout.centers) == ("node:a", "node:b")
    assert layout.centers["node:a"].x < layout.centers["node:b"].x


def test_layered_layout_sorts_encoded_edges_and_places_missing_ids(monkeypatch):
    import leapp_visualization.layout as layout_module

    captured: dict[str, object] = {}

    class FakeLayouts:
        def dot_layout(self, *, spacing):
            captured["spacing"] = spacing
            return self

        def to_dict(self):
            return {
                0: (10.0, 96.0),
                2: (30.0, 0.0),
            }

    def fake_from_edges(edge_pairs, **kwargs):
        captured["edge_pairs"] = edge_pairs
        captured["kwargs"] = kwargs
        return FakeLayouts()

    monkeypatch.setattr(layout_module, "from_edges", fake_from_edges)

    graph = VisualGraph(
        nodes=(_node("b"), _node("a"), _node("c")),
        terminals=(VisualTerminal("terminal:input:c:z", "graph_input", "z", "node:c", "port:c:input:z"),),
        edges=(
            VisualEdge("e1", "forward", "node:b", "node:c"),
            VisualEdge("e0", "forward", "node:a", "node:c"),
            VisualEdge("ignored", "feedback", "node:c", "node:a"),
        ),
    )

    layout = compute_layered_layout(graph)

    assert captured["edge_pairs"] == [(0, 2), (1, 2)]
    assert captured["kwargs"] == {
        "vertex_spacing": 96,
        "dummy_vertices": False,
        "crossing_minimization": "median",
        "check_layout": True,
    }
    assert captured["spacing"] == 96
    assert layout.centers["node:a"] == layout_module.Point(x=0.0, y=0.0)
    assert layout.centers["node:c"] == layout_module.Point(x=249.60000000000002, y=36.0)
    assert layout.centers["node:b"] == layout_module.Point(x=509.6, y=0.0)
    assert layout.centers["terminal:input:c:z"] == layout_module.Point(x=509.6, y=140.0)
