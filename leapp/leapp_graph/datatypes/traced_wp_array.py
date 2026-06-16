"""Traced Warp array support."""

from typing import Any

import torch
from torch.fx.proxy import Proxy

try:
    import warp as wp
except ImportError as exc:
    raise ImportError(
        "traced_wp_array requires warp-lang (pip install warp-lang)."
    ) from exc


from .traced_data import TracedData


class TracedWpArray(wp.array):
    """A ``wp.array`` subclass that can be class-swapped in place.

    Keep this as single inheritance from ``wp.array``. Directly inheriting from
    ``TracedData`` changes the object layout and breaks ``arr.__class__`` swaps
    from raw Warp arrays.
    """

    def __new__(cls, array: wp.array, name: str, context, proxy: Proxy):
        obj = wp.array.__new__(cls)
        wp.array.__init__(
            obj,
            dtype=array.dtype,
            shape=array.shape,
            ptr=array.ptr,
            device=array.device,
            copy=False,
        )
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
        if not isinstance(array, wp.array):
            raise TypeError(f"Expected wp.array, got {type(array).__name__}")
        if not isinstance(array, cls):
            if type(array) is not wp.array:
                raise TypeError(
                    "Can only class-swap exact raw wp.array instances into "
                    f"{cls.__name__}; got {type(array).__name__}"
                )
            array.__class__ = cls
        array._init_tracing_state(name, context, proxy)
        return array

    def _init_tracing_state(self, name: str, context, proxy: Proxy) -> None:
        self._name = name
        self._context = context
        self._proxy = proxy

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
        if self._context is None:
            return False
        return self._context.is_tracing

    @property
    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.numpy())

    @property
    def data(self) -> wp.array:
        return self

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
