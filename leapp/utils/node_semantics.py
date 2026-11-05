#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Value-independent semantics shared by capture and loaded node ports."""

from copy import copy, deepcopy
import json
import os
import tempfile

import yaml

from .semantics import TensorSemantics


class NodeSemantics:
    """A node-scoped handle for named input and output semantic descriptions."""

    def __init__(self, name):
        self.name = name
        self._entries = {"inputs": {}, "outputs": {}}
        self._ports = {"inputs": {}, "outputs": {}}
        self._aliases = {"inputs": {}, "outputs": {}}
        self._sealed = False

    def set_semantics(self, *, inputs=None, outputs=None, states=None):
        """Atomically replace descriptions for supplied named ports.

        Names come from ``TensorSemantics.name``. Omitted fields are cleared;
        other ports are unchanged. No tensor reference is retained.
        """
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
                name = self._aliases[direction].get(name, name)
                if name in pending:
                    raise ValueError(f"Duplicate {direction} semantics for '{name}'")
                if self._sealed and name not in self._ports[direction]:
                    raise ValueError(
                        f"Unknown {direction} port '{name}' in '{self.name}'"
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
                    "aliases",
                }:
                    raise ValueError("Semantic metadata cannot replace port structure")
                shape = self._ports[direction].get(name)
                if shape is not None:
                    snapshot.validate_shape(shape)
                pending[name] = snapshot
            updates[direction] = pending
        for direction, pending in updates.items():
            self._entries[direction].update(pending)

    def _register_port(self, direction, name, shape, aliases=()):
        """Remember only a port's contract, never its tensor or description."""
        if isinstance(shape, str):
            shape = json.loads(shape)
        canonical = self._aliases[direction].get(name, name)
        names = set(aliases) | {name, canonical}
        for alias in names:
            previous = self._aliases[direction].get(alias)
            if previous is not None and previous != canonical:
                raise ValueError(f"Ambiguous {direction} port alias '{alias}'")
        declarations = [
            self._entries[direction][n] for n in names if n in self._entries[direction]
        ]
        if len(declarations) > 1:
            raise ValueError(f"Conflicting semantics for aliases of '{name}'")
        if declarations:
            declarations[0].validate_shape(shape)
        for alias in names:
            self._aliases[direction][alias] = canonical
        self._ports[direction][canonical] = tuple(shape)
        if declarations:
            for alias in names:
                self._entries[direction].pop(alias, None)
            self._entries[direction][canonical] = declarations[0]
        return canonical

    def _bind(self, direction, description, semantics=None):
        name = getattr(description, "_semantic_name", description.name)
        canonical = self._register_port(
            direction, name, description.shape, description.aliases
        )
        if semantics is not None:
            named = copy(semantics)
            named.name = canonical
            self.set_semantics(**{direction: [named]})
        description._semantic_store = self
        description._semantic_direction = direction
        description._semantic_name = canonical
        description._semantics = None

    def _seal(self, inputs, outputs):
        """Reject unresolved declarations once the executable interface is known."""
        for direction, descriptions in (("inputs", inputs), ("outputs", outputs)):
            active = set()
            for description in descriptions:
                self._bind(direction, description)
                active.add(description._semantic_name)
            unknown = set(self._entries[direction]) - set(self._ports[direction])
            if unknown:
                raise ValueError(
                    f"Unresolved {direction} semantics in '{self.name}': {sorted(unknown)}"
                )
            # Pruned inputs do not become executable ports because metadata was
            # declared for them. Retain no contract for later artifact edits.
            self._ports[direction] = {
                name: shape
                for name, shape in self._ports[direction].items()
                if name in active
            }
            self._entries[direction] = {
                name: value
                for name, value in self._entries[direction].items()
                if name in active
            }
        self._sealed = True

    def _load_ports(self, direction, descriptions):
        for description in descriptions:
            name = self._register_port(
                direction,
                description["name"],
                description["shape"],
                description.get("aliases", ()),
            )
            metadata = {
                key: value
                for key, value in description.items()
                if key not in _PORT_FIELDS
            }
            # Loaded fields are already normalized. Legacy extra fields may
            # resemble axes or spatial metadata; do not reinterpret them.
            self._entries[direction][name] = TensorSemantics(name=name, extra=metadata)

    def _update_descriptions(self, direction, descriptions):
        for description in descriptions:
            name = self._aliases[direction].get(
                description["name"], description["name"]
            )
            semantics = self._entries[direction].get(name)
            for key in set(description) - _PORT_FIELDS:
                del description[key]
            if semantics is not None:
                description.update(
                    {
                        key: value.value if hasattr(value, "value") else deepcopy(value)
                        for key, value in semantics.to_dict().items()
                    }
                )


_PORT_FIELDS = frozenset({"name", "dtype", "shape", "type", "aliases"})


def save_node_semantics(path, stores):
    """Atomically update YAML metadata, leaving models and pipeline untouched."""
    with open(path) as stream:
        description = yaml.safe_load(stream)
    for name, store in stores.items():
        if name not in description["models"]:
            continue
        for direction in ("inputs", "outputs"):
            store._update_descriptions(
                direction, description["models"][name][direction]
            )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=os.path.dirname(os.path.abspath(path)), delete=False
        ) as stream:
            temporary = stream.name
            yaml.dump(description, stream, sort_keys=False)
        os.chmod(temporary, os.stat(path).st_mode & 0o777)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
