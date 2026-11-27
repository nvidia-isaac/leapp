#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Value-independent semantics for captured node ports."""

from copy import copy, deepcopy
from .semantics import TensorSemantics


class NodeSemantics:
    """A node-scoped handle for named input and output semantic descriptions."""

    def __init__(self, name):
        self.name = name
        self._entries = {"inputs": {}, "outputs": {}}
        self._ports = {"inputs": {}, "outputs": {}}

    def set_semantics(
        self, *, inputs=None, outputs=None, states=None, allow_unregistered_ports=False
    ):
        """Atomically replace descriptions for supplied named ports.

        Names come from ``TensorSemantics.name``. Omitted fields are cleared;
        other ports are unchanged. No tensor reference is retained.
        """
        if not isinstance(allow_unregistered_ports, bool):
            raise TypeError("allow_unregistered_ports must be a boolean")
        if states is not None and not isinstance(states, (list, tuple)):
            raise TypeError("states must be a list of TensorSemantics")
        updates = {}
        for direction, descriptions in (("inputs", inputs), ("outputs", outputs)):
            if descriptions is None:
                descriptions = []
            if not isinstance(descriptions, (list, tuple)):
                raise TypeError(f"{direction} must be a list of TensorSemantics")
            descriptions = [*descriptions, *(states or [])]
            pending = {}
            for description in descriptions:
                if not isinstance(description, TensorSemantics):
                    raise TypeError(f"{direction} must contain only TensorSemantics")
                name = description.name
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("TensorSemantics requires a non-empty port name")
                if name in pending:
                    raise ValueError(f"Duplicate {direction} semantics for '{name}'")
                if not allow_unregistered_ports and name not in self._ports[direction]:
                    raise ValueError(
                        f"Invalid {direction[:-1]} port name {name!r} for node {self.name!r}. "
                        "To predefine semantics, pass allow_unregistered_ports=True."
                    )
                # Drop ref before copying: even a bound legacy annotation must
                # not keep a sample tensor (or a whole traced graph) alive here.
                snapshot = copy(description)
                snapshot.ref = None
                snapshot = deepcopy(snapshot)
                snapshot.name = name
                snapshot.__post_init__()
                if direction == "outputs" and snapshot.is_setpoint:
                    raise ValueError("is_setpoint=True applies only to inputs")
                if set(snapshot.to_dict()) & {
                    "name",
                    "dtype",
                    "shape",
                    "type",
                }:
                    raise ValueError("Semantic metadata cannot replace port structure")
                shape = self._ports[direction].get(name)
                if shape is not None:
                    snapshot.validate_shape(shape)
                pending[name] = snapshot
            updates[direction] = pending
        for direction, pending in updates.items():
            self._entries[direction].update(pending)

    def _bind(self, direction, description, semantics=None):
        name = getattr(description, "_semantic_name", description.name)
        if semantics is not None:
            named = copy(semantics)
            named.name = name
            named.validate_shape(description.shape)
            self.set_semantics(**{direction: [named]}, allow_unregistered_ports=True)
        else:
            existing = self._entries[direction].get(name)
            if existing is not None:
                existing.validate_shape(description.shape)
        self._ports[direction][name] = tuple(description.shape)
        description._semantic_store = self
        description._semantic_direction = direction
        description._semantic_name = name
        description._semantics = None

    def _validate_ports(self):
        """Reject metadata declarations that never acquired a tensor port."""
        for direction in ("inputs", "outputs"):
            unknown = set(self._entries[direction]) - set(self._ports[direction])
            if unknown:
                raise ValueError(
                    f"Unresolved {direction} semantics in '{self.name}': {sorted(unknown)}"
                )
