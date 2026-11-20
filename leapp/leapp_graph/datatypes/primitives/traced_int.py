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
from leapp.utils.logging import _get_logger
from leapp.utils.shape_contracts import has_variable_inputs


class TracedInt(int, TracedData):
    """An ``int`` that is also a graph value, such as a data-dependent size.

    It behaves like the integer it holds. While its node is tracing, integer
    arithmetic with other integers is recorded, so derived sizes stay symbolic
    too. Once the node stops tracing, or its graph has been replaced,
    arithmetic returns plain ``int`` results and records nothing. A size read
    in a node with variable dimensions keeps its origin instead, so that
    recording it while another node is recording is rejected.
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
            # Keep the origin of a variable size, so that a node recording
            # the result later rejects it.
            origin = next((v for v in (lhs, rhs) if _is_variable_size(v)), None)
            return value if origin is None else origin._new(value, origin.proxy)
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

    def __fx_create_arg__(self, tracer):
        # FX calls this for every recorded argument. Without it FX stores an
        # ``int`` subclass as the constant it holds.
        return tracer.create_arg(graph_value(self))

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
        return value.proxy if value.is_recording else settled_value(value)
    return value


def _is_variable_size(value) -> bool:
    return (
        isinstance(value, TracedInt)
        and value.context_obj is not None
        and has_variable_inputs(value.context_obj)
    )


def settled_value(value: TracedInt) -> int:
    """Return the integer held by a traced integer that stopped recording.

    A size read in a node with variable dimensions changes at run time. While
    a node is recording, that node would store the captured value, so the use
    is rejected. With no node recording, nothing is exported.
    """
    if _is_variable_size(value):
        from leapp.export_manager import ExportManager

        for node in ExportManager().nodes.values():
            if node.is_tracing:
                _get_logger().fatal(
                    f"A size read in node '{value.context}' is used while node "
                    f"'{node.name}' is recording, which would store it there "
                    f"as the constant {int(value)}. Read the size from a tensor "
                    f"of node '{node.name}'",
                    error_type=ValueError,
                )
    return int(value)


def _contains_traced_int(obj) -> bool:
    found = False

    def visit(item):
        nonlocal found
        found = found or isinstance(item, TracedInt)
        return item

    TracedData._map_structure(obj, visit)
    return found


def graph_sizes(obj):
    """Prepare arguments holding traced integers for FX recording.

    FX records a size as a proxy, which ``torch.Size`` cannot hold, so sizes
    become tuples. A traced integer whose node stopped recording has no node in
    the graph being built and becomes the plain integer it holds.
    """
    if not _contains_traced_int(obj):
        return obj

    def convert(item):
        if isinstance(item, TracedInt):
            return item if item.is_recording else settled_value(item)
        if isinstance(item, torch.Size):
            return tuple(convert(size) for size in item)
        if isinstance(item, slice):
            return slice(convert(item.start), convert(item.stop), convert(item.step))
        if isinstance(item, (list, tuple)) and not hasattr(item, "_fields"):
            return type(item)(convert(value) for value in item)
        if isinstance(item, dict):
            return {name: convert(value) for name, value in item.items()}
        return item

    return convert(obj)
