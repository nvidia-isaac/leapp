#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from leapp_visualization import geometry as geometry_module
from leapp_visualization.geometry import resolve_geometry
from leapp_visualization.layout import LayoutResult, Point
from leapp_visualization.model import VisualEdge, VisualGraph, VisualNode, VisualPort, VisualTerminal


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
    assert input_anchor.y == output_anchor.y
    assert geometry.width > node_geometry.rect.width
    assert geometry.height > node_geometry.rect.height


def test_geometry_pairs_input_and_output_ports_by_row():
    node = VisualNode(
        id="node:policy",
        title="policy",
        backend="onnx",
        inputs=(
            VisualPort("port:policy:input:obs", "node:policy", "input", "obs", ("1", "12"), "float32", "state/joint/position"),
            VisualPort("port:policy:input:vel", "node:policy", "input", "vel", ("1", "12"), "float32", "state/joint/velocity"),
        ),
        outputs=(
            VisualPort("port:policy:output:action", "node:policy", "output", "action", ("1", "12"), "float32", "target/joint/position"),
            VisualPort("port:policy:output:effort", "node:policy", "output", "effort", ("1", "12"), "float32", "target/joint/effort"),
        ),
    )
    graph = VisualGraph(nodes=(node,), terminals=(), edges=())
    layout = LayoutResult(centers={"node:policy": Point(0.0, 0.0)}, forward_edge_ids=())

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    assert geometry.ports["port:policy:input:obs"].anchor.y == geometry.ports["port:policy:output:action"].anchor.y
    assert geometry.ports["port:policy:input:vel"].anchor.y == geometry.ports["port:policy:output:effort"].anchor.y


def test_geometry_keeps_two_sided_port_columns_from_overlapping():
    node = VisualNode(
        id="node:policy",
        title="policy",
        backend=None,
        inputs=(
            VisualPort(
                "port:policy:input:backbone_outputs_attention_mask",
                "node:policy",
                "input",
                "backbone_outputs_attention_mask",
                ("1", "101"),
                "float32",
                "state/joint/position",
            ),
        ),
        outputs=(
            VisualPort(
                "port:policy:output:converted_outputs_backbone_attention_mask",
                "node:policy",
                "output",
                "converted_outputs_backbone_attention_mask",
                ("1", "101"),
                "float32",
                "target/joint/position",
            ),
        ),
    )
    graph = VisualGraph(nodes=(node,), terminals=(), edges=())
    layout = LayoutResult(centers={"node:policy": Point(0.0, 0.0)}, forward_edge_ids=())

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    input_rect = geometry.ports["port:policy:input:backbone_outputs_attention_mask"].rect
    output_rect = geometry.ports["port:policy:output:converted_outputs_backbone_attention_mask"].rect
    assert input_rect.x + input_rect.width < output_rect.x


def test_geometry_expands_layer_spacing_to_prevent_node_overlap():
    left = VisualNode(
        id="node:left",
        title="left",
        backend=None,
        inputs=(),
        outputs=(
            VisualPort(
                "port:left:output:converted_outputs_backbone_attention_mask",
                "node:left",
                "output",
                "converted_outputs_backbone_attention_mask",
                ("1", "101"),
                "float32",
                "target/joint/position",
            ),
        ),
    )
    right = VisualNode(
        id="node:right",
        title="right",
        backend=None,
        inputs=(
            VisualPort(
                "port:right:input:backbone_outputs_attention_mask",
                "node:right",
                "input",
                "backbone_outputs_attention_mask",
                ("1", "101"),
                "float32",
                "state/joint/position",
            ),
        ),
        outputs=(),
    )
    graph = VisualGraph(nodes=(left, right), terminals=(), edges=())
    layout = LayoutResult(
        centers={
            "node:left": Point(0.0, 0.0),
            "node:right": Point(260.0, 0.0),
        },
        forward_edge_ids=(),
    )

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    left_rect = geometry.nodes["node:left"].rect
    right_rect = geometry.nodes["node:right"].rect
    assert left_rect.x + left_rect.width <= right_rect.x


def test_geometry_keeps_generous_canvas_margin_around_content():
    node = VisualNode(
        id="node:policy",
        title="policy",
        backend=None,
        inputs=(VisualPort("port:policy:input:obs", "node:policy", "input", "obs", ("1", "12"), "float32", None),),
        outputs=(VisualPort("port:policy:output:action", "node:policy", "output", "action", ("1", "12"), "float32", None),),
    )
    graph = VisualGraph(
        nodes=(node,),
        terminals=(
            VisualTerminal("terminal:input:policy:obs", "graph_input", "obs", "node:policy", "port:policy:input:obs"),
            VisualTerminal("terminal:output:policy:action", "graph_output", "action", "node:policy", "port:policy:output:action"),
        ),
        edges=(
            VisualEdge("edge:input", "graph_input", "terminal:input:policy:obs", "node:policy", None, "port:policy:input:obs", "obs"),
            VisualEdge("edge:output", "graph_output", "node:policy", "terminal:output:policy:action", "port:policy:output:action", None, "action"),
        ),
    )
    layout = LayoutResult(
        centers={
            "terminal:input:policy:obs": Point(0.0, 240.0),
            "node:policy": Point(260.0, 240.0),
            "terminal:output:policy:action": Point(520.0, 240.0),
        },
        forward_edge_ids=("edge:input", "edge:output"),
    )

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    content_right = geometry.content_bounds.x + geometry.content_bounds.width
    content_bottom = geometry.content_bounds.y + geometry.content_bounds.height
    assert geometry.content_bounds.x >= 80.0
    assert geometry.content_bounds.y >= 80.0
    assert geometry.width - content_right >= 80.0
    assert geometry.height - content_bottom >= 80.0


def test_geometry_aligns_terminal_pills_to_connected_port_rows():
    node = VisualNode(
        id="node:preprocess_state",
        title="preprocess_state",
        backend="onnx",
        inputs=(
            VisualPort("port:preprocess_state:input:left_leg", "node:preprocess_state", "input", "left_leg", ("1", "6"), "float32", None),
            VisualPort("port:preprocess_state:input:right_leg", "node:preprocess_state", "input", "right_leg", ("1", "6"), "float32", None),
        ),
        outputs=(
            VisualPort("port:preprocess_state:output:reference_0_left_leg", "node:preprocess_state", "output", "reference_0_left_leg", ("1", "6"), "float32", None),
            VisualPort("port:preprocess_state:output:reference_0_right_leg", "node:preprocess_state", "output", "reference_0_right_leg", ("1", "6"), "float32", None),
            VisualPort("port:preprocess_state:output:reference_0_waist", "node:preprocess_state", "output", "reference_0_waist", ("1", "3"), "float32", None),
        ),
    )
    graph = VisualGraph(
        nodes=(node,),
        terminals=(
            VisualTerminal("terminal:input:preprocess_state:left_leg", "graph_input", "left_leg", "node:preprocess_state", "port:preprocess_state:input:left_leg"),
            VisualTerminal("terminal:input:preprocess_state:right_leg", "graph_input", "right_leg", "node:preprocess_state", "port:preprocess_state:input:right_leg"),
            VisualTerminal("terminal:output:preprocess_state:reference_0_left_leg", "graph_output", "reference_0_left_leg", "node:preprocess_state", "port:preprocess_state:output:reference_0_left_leg"),
            VisualTerminal("terminal:output:preprocess_state:reference_0_right_leg", "graph_output", "reference_0_right_leg", "node:preprocess_state", "port:preprocess_state:output:reference_0_right_leg"),
            VisualTerminal("terminal:output:preprocess_state:reference_0_waist", "graph_output", "reference_0_waist", "node:preprocess_state", "port:preprocess_state:output:reference_0_waist"),
        ),
        edges=(),
    )
    layout = LayoutResult(
        centers={
            "terminal:input:preprocess_state:left_leg": Point(0.0, 240.0),
            "terminal:input:preprocess_state:right_leg": Point(0.0, 80.0),
            "node:preprocess_state": Point(260.0, 160.0),
            "terminal:output:preprocess_state:reference_0_left_leg": Point(520.0, 300.0),
            "terminal:output:preprocess_state:reference_0_right_leg": Point(520.0, 80.0),
            "terminal:output:preprocess_state:reference_0_waist": Point(520.0, 180.0),
        },
        forward_edge_ids=(),
    )

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    for terminal in graph.terminals:
        assert geometry.terminals[terminal.id].anchor.y == geometry.ports[terminal.port_id].anchor.y


def test_geometry_reorders_target_inputs_by_connected_source_output_order():
    source = VisualNode(
        id="node:preprocess_video",
        title="preprocess_video",
        backend="onnx",
        inputs=(),
        outputs=(
            VisualPort("port:preprocess_video:output:pixel_values", "node:preprocess_video", "output", "pixel_values", ("352", "1536"), "float32", None),
            VisualPort("port:preprocess_video:output:input_ids", "node:preprocess_video", "output", "input_ids", ("1", "101"), "int64", None),
            VisualPort("port:preprocess_video:output:attention_mask", "node:preprocess_video", "output", "attention_mask", ("1", "101"), "int64", None),
            VisualPort("port:preprocess_video:output:image_grid_thw", "node:preprocess_video", "output", "image_grid_thw", ("1", "3"), "int64", None),
            VisualPort("port:preprocess_video:output:embodiment_id", "node:preprocess_video", "output", "embodiment_id", ("1",), "int64", None),
        ),
    )
    target = VisualNode(
        id="node:action_head",
        title="action_head",
        backend="onnx",
        inputs=(
            VisualPort("port:action_head:input:action_inputs_embodiment_id", "node:action_head", "input", "action_inputs_embodiment_id", ("1",), "int64", None),
            VisualPort("port:action_head:input:action_inputs_image_grid_thw", "node:action_head", "input", "action_inputs_image_grid_thw", ("1", "3"), "int64", None),
            VisualPort("port:action_head:input:action_inputs_attention_mask", "node:action_head", "input", "action_inputs_attention_mask", ("1", "101"), "int64", None),
            VisualPort("port:action_head:input:action_inputs_input_ids", "node:action_head", "input", "action_inputs_input_ids", ("1", "101"), "int64", None),
            VisualPort("port:action_head:input:action_inputs_pixel_values", "node:action_head", "input", "action_inputs_pixel_values", ("352", "1536"), "float32", None),
        ),
        outputs=(),
    )
    graph = VisualGraph(
        nodes=(source, target),
        terminals=(),
        edges=(
            VisualEdge("edge:pixel", "forward", "node:preprocess_video", "node:action_head", "port:preprocess_video:output:pixel_values", "port:action_head:input:action_inputs_pixel_values", "pixel_values"),
            VisualEdge("edge:input_ids", "forward", "node:preprocess_video", "node:action_head", "port:preprocess_video:output:input_ids", "port:action_head:input:action_inputs_input_ids", "input_ids"),
            VisualEdge("edge:attention_mask", "forward", "node:preprocess_video", "node:action_head", "port:preprocess_video:output:attention_mask", "port:action_head:input:action_inputs_attention_mask", "attention_mask"),
            VisualEdge("edge:image_grid_thw", "forward", "node:preprocess_video", "node:action_head", "port:preprocess_video:output:image_grid_thw", "port:action_head:input:action_inputs_image_grid_thw", "image_grid_thw"),
            VisualEdge("edge:embodiment_id", "forward", "node:preprocess_video", "node:action_head", "port:preprocess_video:output:embodiment_id", "port:action_head:input:action_inputs_embodiment_id", "embodiment_id"),
        ),
    )
    layout = LayoutResult(
        centers={
            "node:preprocess_video": Point(0.0, 0.0),
            "node:action_head": Point(260.0, 0.0),
        },
        forward_edge_ids=tuple(edge.id for edge in graph.edges),
    )

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    target_order = [
        "port:action_head:input:action_inputs_pixel_values",
        "port:action_head:input:action_inputs_input_ids",
        "port:action_head:input:action_inputs_attention_mask",
        "port:action_head:input:action_inputs_image_grid_thw",
        "port:action_head:input:action_inputs_embodiment_id",
    ]
    target_y_positions = [geometry.ports[port_id].anchor.y for port_id in target_order]
    assert target_y_positions == sorted(target_y_positions)


def test_geometry_preserves_source_outputs_and_reorders_target_inputs_to_match():
    source = VisualNode(
        id="node:producer",
        title="producer",
        backend="onnx",
        inputs=(),
        outputs=(
            VisualPort("port:producer:output:c", "node:producer", "output", "c", ("1",), "float32", None),
            VisualPort("port:producer:output:b", "node:producer", "output", "b", ("1",), "float32", None),
            VisualPort("port:producer:output:a", "node:producer", "output", "a", ("1",), "float32", None),
        ),
    )
    target = VisualNode(
        id="node:consumer",
        title="consumer",
        backend="onnx",
        inputs=(
            VisualPort("port:consumer:input:a", "node:consumer", "input", "a", ("1",), "float32", None),
            VisualPort("port:consumer:input:b", "node:consumer", "input", "b", ("1",), "float32", None),
            VisualPort("port:consumer:input:c", "node:consumer", "input", "c", ("1",), "float32", None),
        ),
        outputs=(),
    )
    graph = VisualGraph(
        nodes=(source, target),
        terminals=(),
        edges=(
            VisualEdge("edge:a", "forward", "node:producer", "node:consumer", "port:producer:output:a", "port:consumer:input:a", "a"),
            VisualEdge("edge:b", "forward", "node:producer", "node:consumer", "port:producer:output:b", "port:consumer:input:b", "b"),
            VisualEdge("edge:c", "forward", "node:producer", "node:consumer", "port:producer:output:c", "port:consumer:input:c", "c"),
        ),
    )
    layout = LayoutResult(
        centers={
            "node:producer": Point(0.0, 0.0),
            "node:consumer": Point(260.0, 0.0),
        },
        forward_edge_ids=tuple(edge.id for edge in graph.edges),
    )

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    source_order = [
        "port:producer:output:c",
        "port:producer:output:b",
        "port:producer:output:a",
    ]
    source_y_positions = [geometry.ports[port_id].anchor.y for port_id in source_order]
    assert source_y_positions == sorted(source_y_positions)

    target_order = [
        "port:consumer:input:c",
        "port:consumer:input:b",
        "port:consumer:input:a",
    ]
    target_y_positions = [geometry.ports[port_id].anchor.y for port_id in target_order]
    assert target_y_positions == sorted(target_y_positions)


def test_geometry_reorders_ports_iteratively_for_fanout_targets():
    source = VisualNode(
        id="node:preprocess_video",
        title="preprocess_video",
        backend="onnx",
        inputs=(),
        outputs=(
            VisualPort("port:preprocess_video:output:pixel_values", "node:preprocess_video", "output", "pixel_values", ("1",), "float32", None),
            VisualPort("port:preprocess_video:output:input_ids", "node:preprocess_video", "output", "input_ids", ("1",), "int64", None),
            VisualPort("port:preprocess_video:output:attention_mask", "node:preprocess_video", "output", "attention_mask", ("1",), "int64", None),
        ),
    )
    first_target = VisualNode(
        id="node:backbone",
        title="backbone",
        backend="onnx",
        inputs=(
            VisualPort("port:backbone:input:attention_mask", "node:backbone", "input", "attention_mask", ("1",), "int64", None),
            VisualPort("port:backbone:input:input_ids", "node:backbone", "input", "input_ids", ("1",), "int64", None),
            VisualPort("port:backbone:input:pixel_values", "node:backbone", "input", "pixel_values", ("1",), "float32", None),
        ),
        outputs=(),
    )
    second_target = VisualNode(
        id="node:action_head",
        title="action_head",
        backend="onnx",
        inputs=(
            VisualPort("port:action_head:input:action_inputs_attention_mask", "node:action_head", "input", "action_inputs_attention_mask", ("1",), "int64", None),
            VisualPort("port:action_head:input:action_inputs_input_ids", "node:action_head", "input", "action_inputs_input_ids", ("1",), "int64", None),
            VisualPort("port:action_head:input:action_inputs_pixel_values", "node:action_head", "input", "action_inputs_pixel_values", ("1",), "float32", None),
        ),
        outputs=(),
    )
    graph = VisualGraph(
        nodes=(source, first_target, second_target),
        terminals=(),
        edges=(
            VisualEdge("edge:backbone_pixel", "forward", "node:preprocess_video", "node:backbone", "port:preprocess_video:output:pixel_values", "port:backbone:input:pixel_values", "pixel_values"),
            VisualEdge("edge:backbone_input_ids", "forward", "node:preprocess_video", "node:backbone", "port:preprocess_video:output:input_ids", "port:backbone:input:input_ids", "input_ids"),
            VisualEdge("edge:backbone_attention", "forward", "node:preprocess_video", "node:backbone", "port:preprocess_video:output:attention_mask", "port:backbone:input:attention_mask", "attention_mask"),
            VisualEdge("edge:action_pixel", "forward", "node:preprocess_video", "node:action_head", "port:preprocess_video:output:pixel_values", "port:action_head:input:action_inputs_pixel_values", "pixel_values"),
            VisualEdge("edge:action_input_ids", "forward", "node:preprocess_video", "node:action_head", "port:preprocess_video:output:input_ids", "port:action_head:input:action_inputs_input_ids", "input_ids"),
            VisualEdge("edge:action_attention", "forward", "node:preprocess_video", "node:action_head", "port:preprocess_video:output:attention_mask", "port:action_head:input:action_inputs_attention_mask", "attention_mask"),
        ),
    )
    layout = LayoutResult(
        centers={
            "node:preprocess_video": Point(0.0, 0.0),
            "node:backbone": Point(260.0, 0.0),
            "node:action_head": Point(260.0, 400.0),
        },
        forward_edge_ids=tuple(edge.id for edge in graph.edges),
    )

    geometry = resolve_geometry(graph, layout, graph_name="demo")

    source_order = [
        "port:preprocess_video:output:pixel_values",
        "port:preprocess_video:output:input_ids",
        "port:preprocess_video:output:attention_mask",
    ]
    source_y_positions = [geometry.ports[port_id].anchor.y for port_id in source_order]
    assert source_y_positions == sorted(source_y_positions)


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
    assert len(feedback.points) == 6
    assert feedback.points[1].x > feedback.points[0].x
    assert feedback.points[-2].x < feedback.points[-1].x
    assert feedback.points[2].y == feedback.points[3].y
    assert feedback.points[2].x > feedback.points[3].x
    assert forward.points[0].x < forward.points[-1].x


def test_geometry_budgets_output_row_width_for_kind_labels_on_their_own_row():
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

    visible_kind = geometry_module.truncate_text(long_kind, geometry_module._PORT_NAME_MAX_CHARS)
    output_rect = geometry.ports["port:policy:output:action"].rect
    expected_kind_width = len(visible_kind) * geometry_module._PORT_PRIMARY_CHAR_WIDTH

    assert output_rect.width >= expected_kind_width + geometry_module._NODE_PADDING
