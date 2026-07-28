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
