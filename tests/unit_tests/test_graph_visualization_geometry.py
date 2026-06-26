from leapp.leapp_graph.visualization import geometry as geometry_module
from leapp.leapp_graph.visualization.geometry import resolve_geometry
from leapp.leapp_graph.visualization.layout import LayoutResult, Point
from leapp.leapp_graph.visualization.model import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal


def test_geometry_places_input_and_output_ports_on_node_edges():
    node = VisualNode(
        id="node:policy",
        title="policy",
        backend="onnx-dynamo",
        inputs=(VisualPort("port:policy:input:obs", "node:policy", "input", "obs", ("1", "12"), "float32", "state"),),
        outputs=(VisualPort("port:policy:output:action", "node:policy", "output", "action", ("1", "4"), "float32", "command"),),
    )
    graph = VisualGraph(nodes=(node,), terminals=(), edges=())
    layout = LayoutResult(centers={"node:policy": Point(0.0, 0.0)}, forward_edge_ids=())

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    node_geometry = geometry.nodes["node:policy"]
    input_anchor = geometry.ports["port:policy:input:obs"].anchor
    output_anchor = geometry.ports["port:policy:output:action"].anchor

    assert input_anchor.x == node_geometry.rect.x
    assert output_anchor.x == node_geometry.rect.x + node_geometry.rect.width
    assert input_anchor.y < output_anchor.y
    assert geometry.width > node_geometry.rect.width
    assert geometry.height > node_geometry.rect.height


def test_geometry_routes_forward_and_feedback_edges_differently():
    a = VisualNode(
        id="node:a",
        title="a",
        backend=None,
        inputs=(VisualPort("port:a:input:state", "node:a", "input", "state", ("2",), "float32", None),),
        outputs=(VisualPort("port:a:output:y", "node:a", "output", "y", ("2",), "float32", None),),
    )
    b = VisualNode(
        id="node:b",
        title="b",
        backend=None,
        inputs=(VisualPort("port:b:input:y", "node:b", "input", "y", ("2",), "float32", None),),
        outputs=(VisualPort("port:b:output:state", "node:b", "output", "state", ("2",), "float32", None),),
    )
    graph = VisualGraph(
        nodes=(a, b),
        terminals=(),
        edges=(
            VisualEdge("edge:forward", "forward", "node:a", "node:b", "port:a:output:y", "port:b:input:y", "y"),
            VisualEdge("edge:feedback", "feedback", "node:b", "node:a", "port:b:output:state", "port:a:input:state", "state"),
        ),
    )
    layout = LayoutResult(centers={"node:a": Point(0.0, 0.0), "node:b": Point(300.0, 0.0)}, forward_edge_ids=("edge:forward",))

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    forward = geometry.edges["edge:forward"]
    feedback = geometry.edges["edge:feedback"]
    assert forward.kind == "forward"
    assert feedback.kind == "feedback"
    assert min(point.y for point in feedback.points) < geometry.content_bounds.y
    assert forward.points[0].x < forward.points[-1].x


def test_geometry_budgets_output_row_width_for_long_inline_kind_labels():
    separator_gap = 16.0
    long_kind = "semantic_feedback_reference_signal"
    node = VisualNode(
        id="node:policy",
        title="policy",
        backend=None,
        inputs=(),
        outputs=(
            VisualPort(
                "port:policy:output:action",
                "node:policy",
                "output",
                "action_output_label",
                ("1", "4"),
                "float32",
                long_kind,
            ),
        ),
    )
    graph = VisualGraph(nodes=(node,), terminals=(), edges=())
    layout = LayoutResult(centers={"node:policy": Point(0.0, 0.0)}, forward_edge_ids=())

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    visible_name = geometry_module.truncate_text(node.outputs[0].name, geometry_module._PORT_NAME_MAX_CHARS)
    visible_kind = geometry_module.truncate_text(long_kind, geometry_module._PORT_NAME_MAX_CHARS)
    visible_detail = geometry_module.truncate_text("[1, 4] float32", geometry_module._PORT_DETAIL_MAX_CHARS)
    required_primary_width = (
        len(visible_name) * geometry_module._PORT_PRIMARY_CHAR_WIDTH
        + separator_gap
        + len(visible_kind) * geometry_module._PORT_PRIMARY_CHAR_WIDTH
    )
    required_port_width = max(
        required_primary_width,
        len(visible_detail) * geometry_module._PORT_DETAIL_CHAR_WIDTH,
    )
    expected_node_width = required_port_width + (geometry_module._NODE_PADDING * 2.0) + 36.0

    assert geometry.nodes["node:policy"].rect.width >= expected_node_width
