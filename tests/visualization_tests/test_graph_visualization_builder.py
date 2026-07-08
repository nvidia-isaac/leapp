from dataclasses import dataclass

from leapp.leapp_graph.visualization_adapter import build_visual_graph


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


def test_build_visual_graph_preserves_ports_semantics_and_external_io():
    policy = FakeNode(
        name="policy",
        inputs=[FakeTensorDescription("obs", "float32", (1, 12), semantics={"kind": "state"})],
        outputs=[FakeTensorDescription("action", "float32", (1, 4), tag="policy/action", semantics={"kind": "command"})],
        backend="onnx-dynamo",
        node_index=0,
    )
    clamp = FakeNode(
        name="clamp",
        inputs=[FakeTensorDescription("raw_action", "float32", (1, 4), tag="policy/action")],
        outputs=[FakeTensorDescription("action", "float32", (1, 4), tag="clamp/action")],
        backend="jit-script",
        node_index=1,
    )
    connections = [
        {
            "source": {"node": policy, "idx": 0},
            "targets": [{"node": clamp, "idx": 0}],
        }
    ]

    graph = build_visual_graph(
        nodes={"policy": policy, "clamp": clamp},
        connections=connections,
        feedback_connections=[],
        graph_inputs=["policy/obs"],
        graph_outputs=["clamp/action"],
    )

    node_by_id = {node.id: node for node in graph.nodes}
    assert tuple(node_by_id) == ("node:policy", "node:clamp")
    assert node_by_id["node:policy"].inputs[0].name == "obs"
    assert node_by_id["node:policy"].inputs[0].shape == ("1", "12")
    assert node_by_id["node:policy"].inputs[0].dtype == "float32"
    assert node_by_id["node:policy"].inputs[0].kind == "state"
    assert node_by_id["node:policy"].outputs[0].kind == "command"
    assert node_by_id["node:policy"].backend == "onnx-dynamo"

    terminal_by_id = {terminal.id: terminal for terminal in graph.terminals}
    assert terminal_by_id["terminal:input:policy:obs"].kind == "graph_input"
    assert terminal_by_id["terminal:output:clamp:action"].kind == "graph_output"

    edge_kinds = [edge.kind for edge in graph.edges]
    assert edge_kinds == ["forward", "graph_input", "graph_output"]
    assert graph.edges[0].source_port_id == "port:policy:output:action"
    assert graph.edges[0].target_port_id == "port:clamp:input:raw_action"
    assert graph.edges[0].label == "action"


def test_build_visual_graph_marks_feedback_edges_without_adding_them_to_forward_flow():
    first = FakeNode(
        name="first",
        inputs=[FakeTensorDescription("state", "float32", (2,), tag="second/state_next")],
        outputs=[FakeTensorDescription("hidden", "float32", (2,), tag="first/hidden")],
        node_index=0,
    )
    second = FakeNode(
        name="second",
        inputs=[FakeTensorDescription("hidden", "float32", (2,), tag="first/hidden")],
        outputs=[FakeTensorDescription("state_next", "float32", (2,), tag="second/state_next")],
        node_index=1,
    )

    graph = build_visual_graph(
        nodes={"first": first, "second": second},
        connections=[{"source": {"node": first, "idx": 0}, "targets": [{"node": second, "idx": 0}]}],
        feedback_connections=[{"source": {"node": second, "idx": 0}, "targets": [{"node": first, "idx": 0}]}],
        graph_inputs=[],
        graph_outputs=[],
    )

    assert [edge.kind for edge in graph.edges] == ["forward", "feedback"]
    assert graph.edges[1].source_id == "node:second"
    assert graph.edges[1].target_id == "node:first"
