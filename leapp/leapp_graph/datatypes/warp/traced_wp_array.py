"""Traced Warp array support."""

from typing import Any

import torch
from torch.fx.proxy import Proxy

import warp as wp

from ..traced_data import TracedData
from leapp.utils.dtype import DtypeCodec, register_dtype_codec
from leapp.utils.logging import _get_logger


# Register the Warp dtype codec so leapp core can convert warp dtypes to common
# name strings without importing warp itself. Mirrors the torch/numpy entries
# and only runs when this (warp-gated) module is imported.
register_dtype_codec(DtypeCodec(
    backend="warp",
    matches=lambda v: isinstance(v, wp.array),
    value_dtype=lambda v: v.dtype,
    dtype_to_name={
        wp.float16: "float16",
        wp.float32: "float32",
        wp.float64: "float64",
        wp.int8: "int8",
        wp.int16: "int16",
        wp.int32: "int32",
        wp.int64: "int64",
        wp.uint8: "uint8",
        wp.uint16: "uint16",
        wp.uint32: "uint32",
        wp.uint64: "uint64",
        wp.bool: "bool",
    },
))


class TracedWpArray(wp.array):
    """A ``wp.array`` subclass that can be class-swapped in place.

    Keep this as single inheritance from ``wp.array``. Directly inheriting from
    ``TracedData`` changes the object layout and breaks ``arr.__class__`` swaps
    from raw Warp arrays.
    """

    def __new__(cls, array, name, context, proxy):
        obj = wp.array.__new__(cls)
        from leapp.leapp_graph.datatypes.patching import get_warp_backend

        # Warp's array initializers are patched in place by the session backend;
        # pause detection so they populate ptr/shape/etc. on this half-built object
        # instead of normalizing ``self`` into a ``.data`` view (which needs ptr).
        warp_backend = get_warp_backend()
        init_kwargs = dict(
            dtype=array.dtype,
            shape=array.shape,
            ptr=array.ptr,
            device=array.device,
            copy=False,
        )
        if warp_backend is not None:
            with warp_backend.paused():
                wp.array.__init__(obj, **init_kwargs)
        else:
            wp.array.__init__(obj, **init_kwargs)
        obj._init_tracing_state(name, context, proxy)
        return obj

    def __init__(self, array: wp.array, name: str, context, proxy: Proxy):
        # ``__new__`` initializes Warp storage and tracing metadata.
        pass

    @classmethod
    def make_traced_in_place(
        cls, array: wp.array, name: str, context, proxy: Proxy
    ) -> "TracedWpArray":
        """Turn an existing raw ``wp.array`` into a traced array in place."""
        if type(array) is cls:
            array._init_tracing_state(name, context, proxy)
            return array
        if type(array) is not wp.array:
            _get_logger().fatal(
                "Can only class-swap exact raw wp.array instances into "
                f"{cls.__name__}; got {type(array).__name__}",
                error_type=TypeError,
            )
        array.__class__ = cls
        array._init_tracing_state(name, context, proxy)
        return array

    def _init_tracing_state(self, name: str, context, proxy: Proxy) -> None:
        self._name = name
        self._context = context
        self._proxy = proxy

    def rebind_tracing_proxy(self, name: str, context, proxy: Proxy) -> None:
        """Rebind this traced array to the canonical FX proxy for its value."""
        self._init_tracing_state(name, context, proxy)

    @property
    def warp_segment(self):
        return getattr(self, "_leapp_warp_segment", None)

    @warp_segment.setter
    def warp_segment(self, segment) -> None:
        self._leapp_warp_segment = segment

    @property
    def data(self) -> wp.array:
        """Return an exact ``wp.array`` aliasing this array's memory.

        Warp's launch-time checks compare ``type(value)`` against the concrete
        array class (e.g. ``type(value) is concrete_array_type(arg_type)`` in
        ``pack_arg``, ``type(arg) in array_types`` in ``infer_argument_types``),
        so ``wp.array`` subclasses are rejected. This returns a non-owning raw
        ``wp.array`` over the same pointer for Warp to consume, leaving this
        traced object untouched. Mirrors Warp's own ``flatten``/``reshape``/
        ``view`` aliasing, including the ``_ref`` back-pointer that keeps the
        owning allocation alive.
        """
        raw = wp.array(
            ptr=self.ptr,
            dtype=self.dtype,
            shape=self.shape,
            strides=self.strides,
            device=self.device,
            pinned=self.pinned,
            copy=False,
            grad=self.grad,
        )
        raw._ref = self
        return raw


    @property
    def tensor(self) -> torch.Tensor:
        from leapp.leapp_graph.datatypes.patching import get_warp_backend

        warp_backend = get_warp_backend()
        if warp_backend is not None:
            with warp_backend.paused():
                return wp.to_torch(self.data)
        return wp.to_torch(self.data)

    @property
    def proxy(self) -> Proxy:
        return self._proxy

    @property
    def name(self) -> str:
        return self._name

    @property
    def context(self) -> str:
        if self._context is None:
            return "untraced"
        return self._context.name

    @property
    def context_obj(self):
        return self._context

    @property
    def is_tracing(self) -> bool:
        # ``_context`` may be missing while the object is still being built
        # (``wp.array.__init__`` is patched and runs before tracing state is set).
        context = getattr(self, "_context", None)
        if context is None:
            return False
        return context.is_tracing

    def _new(self, value: Any, proxy: Proxy = None) -> "TracedWpArray":
        name = TracedData._name_from_proxy(proxy)
        return type(self)(value, name, self._context, proxy)

    def validate_status(self, args=None, kwargs=None) -> bool:
        return TracedData.validate_status(self, args=args, kwargs=kwargs)


###############################################################################
# Registration
# this allows the TracedWpArray to be viewed as a TracedData instance. This
# structure is required to be able to do class swapping in place.
# downside is we need to duplicate calls and forward to TracedData methods
# and don't get the stability of inheritance.
###############################################################################
TracedData.register(TracedWpArray)
