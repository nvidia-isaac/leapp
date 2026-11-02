#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Traced integers for sizes that are only known at run time.

``torch.nonzero``, boolean masks, ``torch.unique`` and similar operations
return tensors whose size depends on their input values. Reading such a size
as a plain Python ``int`` would bake the traced value into the exported graph,
so a later input with a different count fails (for example in ``Reshape``).
:class:`TracedInt` is an ``int`` that is also a :class:`TracedData`: it carries
the FX proxy of the size query, so shapes built from it stay symbolic in the
exported graph, and LEAPP's generic structure traversal, unwrapping, proxy
extraction and context validation recognize it like any other traced value.
"""

import operator

import numpy as np
import torch
import yaml
from torch.fx.proxy import Proxy

from ..traced_data import TracedData


class TracedInt(int, TracedData):
    """An ``int`` that is also a graph value, such as a data-dependent size.

    It behaves like the integer it holds. While its node is tracing, integer
    arithmetic with other integers is recorded, so derived sizes stay symbolic
    too. Once the node stops tracing, or its graph has been replaced,
    arithmetic returns plain ``int`` results and records nothing.
    """

    def __new__(cls, value: int, name: str, context, proxy: Proxy):
        return super().__new__(cls, value)

    def __init__(self, value: int, name: str, context, proxy: Proxy):
        TracedData.__init__(self, int(value), name, context, proxy)

    # =========================================================================
    # TracedData interface
    # =========================================================================

    @property
    def tensor(self) -> torch.Tensor:
        """The value as a 0-d ``int64`` tensor."""
        return torch.tensor(int(self), dtype=torch.int64)

    @property
    def data(self) -> int:
        """The value as a plain ``int``."""
        return int(self)

    def get_dtype_name(self) -> str:
        return "int64"

    def _new(self, value: int, proxy: Proxy = None) -> "TracedInt":
        return TracedInt(value, self._name_from_proxy(proxy), self._context, proxy)

    @property
    def is_recording(self) -> bool:
        """Whether operations on this value are recorded into a live graph."""
        return self.is_tracing and not self.describes_replaced_graph()

    # =========================================================================
    # Arithmetic
    # =========================================================================

    def _record(self, op, lhs, rhs):
        """Return ``op(lhs, rhs)``, recorded if any operand is recording."""
        if not isinstance(lhs, int) or not isinstance(rhs, int):
            return NotImplemented
        value = op(int(lhs), int(rhs))
        anchor = find_recording_int((lhs, rhs))
        if anchor is None:
            return value
        # Fails when operands belong to different tracing nodes.
        anchor.validate_status((lhs, rhs))
        proxy = anchor._context.tracer.create_proxy(
            "call_function", op, (graph_value(lhs), graph_value(rhs)), {}
        )
        return anchor._new(value, proxy)

    def __add__(self, other):
        return self._record(operator.add, self, other)

    def __radd__(self, other):
        return self._record(operator.add, other, self)

    def __sub__(self, other):
        return self._record(operator.sub, self, other)

    def __rsub__(self, other):
        return self._record(operator.sub, other, self)

    def __mul__(self, other):
        return self._record(operator.mul, self, other)

    def __rmul__(self, other):
        return self._record(operator.mul, other, self)

    def __floordiv__(self, other):
        return self._record(operator.floordiv, self, other)

    def __rfloordiv__(self, other):
        return self._record(operator.floordiv, other, self)

    def __mod__(self, other):
        return self._record(operator.mod, self, other)

    def __rmod__(self, other):
        return self._record(operator.mod, other, self)

    def __neg__(self):
        return self._record(operator.sub, 0, self)

    # =========================================================================
    # Python protocols
    # =========================================================================

    # ``int`` is immutable, so a copy is the same graph value. Without these
    # hooks copying would rebuild the object through ``__new__`` without its
    # tracing arguments.
    def __copy__(self) -> "TracedInt":
        return self

    def __deepcopy__(self, memo) -> "TracedInt":
        return self

    def __reduce_ex__(self, protocol):
        # Serialize the value only; the tracer and graph are not picklable.
        return (int, (int(self),))

    def __array__(self, dtype=None, copy=None):
        return np.asarray(int(self), dtype=dtype)

    def __repr__(self) -> str:
        return f"TracedInt({int(self)})"

    def __str__(self) -> str:
        return str(int(self))


# YAML looks representers up by exact type, so a TracedInt would otherwise be
# rejected by ``safe_dump`` and written as a Python object by ``dump``.
for _representer in (yaml.representer.SafeRepresenter, yaml.representer.Representer):
    _representer.add_representer(
        TracedInt, yaml.representer.SafeRepresenter.represent_int
    )


def find_recording_int(obj):
    """Return the first recording :class:`TracedInt` in ``obj``, else None."""
    found = None

    def visit(item):
        nonlocal found
        if found is None and isinstance(item, TracedInt) and item.is_recording:
            found = item
        return item

    TracedData._map_structure(obj, visit)
    return found


def graph_value(value):
    """FX argument for ``value``.

    A recording :class:`TracedInt` contributes its proxy. One that stopped
    recording, because its node finished or replaced its graph, has no node in
    the graph being built and is used as the plain integer it holds.
    """
    if isinstance(value, TracedInt):
        return value.proxy if value.is_recording else int(value)
    return value
