from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PortSide = Literal["input", "output"]
TerminalKind = Literal["graph_input", "graph_output"]
EdgeKind = Literal["forward", "feedback", "graph_input", "graph_output"]


@dataclass(frozen=True)
class VisualPort:
    id: str
    node_id: str
    side: PortSide
    name: str
    shape: tuple[str, ...]
    dtype: str
    kind: str | None = None


@dataclass(frozen=True)
class VisualNode:
    id: str
    title: str
    backend: str | None
    inputs: tuple[VisualPort, ...] = field(default_factory=tuple)
    outputs: tuple[VisualPort, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VisualTerminal:
    id: str
    kind: TerminalKind
    title: str
    node_id: str
    port_id: str


@dataclass(frozen=True)
class VisualEdge:
    id: str
    kind: EdgeKind
    source_id: str
    target_id: str
    source_port_id: str | None = None
    target_port_id: str | None = None
    label: str = ""


@dataclass(frozen=True)
class VisualGraph:
    nodes: tuple[VisualNode, ...]
    terminals: tuple[VisualTerminal, ...]
    edges: tuple[VisualEdge, ...]

    def visual_ids(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes) + tuple(terminal.id for terminal in self.terminals)


def visual_id(*parts: object) -> str:
    normalized = [str(part).replace("/", ":").replace(" ", "_") for part in parts]
    return ":".join(normalized)
