"""Traced Warp array support."""

from typing import Any, Callable

import torch
from torch.fx.proxy import Proxy

import warp as wp

from ..proxy_view import (
    bind_new_view,
    bind_shared_view,
    may_adopt_view,
    update_view_proxy,
)
from ..traced_data import TracedData
from leapp.utils.dtype import DtypeCodec, register_dtype_codec
from leapp.utils.logging import _get_logger


_MAX_PARAM_SCAN_DEPTH = 16


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


class TracedWpArray(wp.array, TracedData):
    """A traced ``wp.array`` subclass that can be class-swapped in place.

    ``wp.array`` must remain the first base. That preserves its CPython object
    layout lineage so exact raw Warp arrays can be promoted with
    ``arr.__class__ = TracedWpArray``.
    """

    def __new__(cls, array, name, context, proxy=None, *, view=None):
        if view is not None and proxy is not None:
            _get_logger().fatal(
                "TracedWpArray accepts proxy= or view=, not both",
                error_type=ValueError,
            )
        obj = wp.array.__new__(cls)
        wp.array.__init__(
            obj,
            dtype=array.dtype,
            shape=array.shape,
            ptr=array.ptr,
            device=array.device,
            copy=False,
        )
        # Keep the source allocation alive for non-owning consumer aliases.
        obj._ref = array
        if view is not None:
            bind_shared_view(obj, name, context, view)
        else:
            bind_new_view(obj, name, context, proxy)
        return obj

    def __init__(self, array: wp.array, name: str, context, proxy=None, *, view=None):
        # ``__new__`` initializes Warp storage and tracing metadata.
        pass

    @classmethod
    def make_traced_in_place(
        cls, array: wp.array, name: str, context, proxy=None, *, view=None
    ) -> "TracedWpArray":
        """Turn an existing raw ``wp.array`` into a traced array in place.

        Pass ``proxy`` for a private root, or ``view`` to share an existing cell.
        """
        if view is not None and proxy is not None:
            _get_logger().fatal(
                "make_traced_in_place accepts proxy= or view=, not both",
                error_type=ValueError,
            )
        if type(array) is cls:
            if view is not None:
                bind_shared_view(array, name, context, view)
            else:
                bind_new_view(array, name, context, proxy)
            return array
        if type(array) is not wp.array:
            _get_logger().fatal(
                "Can only class-swap exact raw wp.array instances into "
                f"{cls.__name__}; got {type(array).__name__}",
                error_type=TypeError,
            )
        array.__class__ = cls
        if view is not None:
            bind_shared_view(array, name, context, view)
        else:
            bind_new_view(array, name, context, proxy)
        return array

    def rebind_tracing_proxy(self, name: str, context, proxy: Proxy) -> None:
        """Rebind this traced array to the canonical FX proxy for its value.

        This updates the shared view's proxy rather than binding a new view: the
        array still describes the same buffer, so every Torch or NumPy alias
        sharing its view has to follow the segment output too. Binding a new
        view here would leave those aliases on the pre-segment proxy while eager
        Warp had already overwritten the memory they read.
        """
        update_view_proxy(self, name, context, proxy)

    @classmethod
    def __warp_function__(
        cls,
        func: Callable,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Execute an intercepted Warp call with traced-value semantics."""
        from ..patching import get_warp_backend

        backend = get_warp_backend()
        if backend is None or not backend.installed:
            return NotImplemented
        kwargs = {} if kwargs is None else kwargs
        qualname = backend.function_qualname(func)

        if backend.is_boundary_function(func):
            handled, result = cls._handle_boundary_call(
                backend, func, args, kwargs
            )
            if handled:
                return result

        traced_inputs, call_args, call_kwargs = cls._normalize_and_collect(
            args, kwargs
        )
        trace_source = cls._select_trace_source(qualname, traced_inputs)
        segment = backend.resolve_or_begin_warp_segment(trace_source)

        if segment is not None:
            backend.record_segment_inputs(segment, qualname, traced_inputs)

        with backend.pause_context():
            result = func(*call_args, **call_kwargs)

        cls._process_post_call_arrays(
            segment, args, kwargs, result, trace_source
        )
        cls._carry_full_copy_port(backend, func, args, kwargs)
        return result

    @classmethod
    def _handle_boundary_call(
        cls,
        backend: Any,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[bool, Any]:
        from leapp.leapp_graph.datatypes import as_traced, is_tracable_tensor_type

        src = cls._find_single_traced_data(args, kwargs)
        if src is None:
            return False, None

        if backend.is_readback_boundary(func):
            backend.close_warp_segment()

        call_args = TracedData.unwrap_traced_data(args)
        call_kwargs = TracedData.unwrap_traced_data(kwargs)
        with backend.pause_context():
            raw = func(*call_args, **call_kwargs)

        if raw is src.data:
            return True, src
        if is_tracable_tensor_type(raw):
            if may_adopt_view(src, raw):
                view, proxy = src.proxy_view, None
            else:
                view, proxy = None, src.proxy
            return True, as_traced(
                raw, src.name, src.context_obj, proxy, view=view
            )
        return True, raw

    @staticmethod
    def _find_single_traced_data(
        args: tuple[Any, ...], kwargs: dict[str, Any] | None = None
    ) -> TracedData | None:
        values: list[TracedData] = []

        def collect(obj: Any) -> None:
            if isinstance(obj, TracedData):
                values.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    collect(value)
            elif isinstance(obj, (list, tuple, set, frozenset)):
                for value in obj:
                    collect(value)

        collect((args, kwargs or {}))
        if not values:
            return None

        if len({id(value.context_obj) for value in values}) > 1:
            _get_logger().fatal(
                "Warp boundary call received traced data from different LEAPP "
                "trace contexts. Mixing contexts is not supported.",
                error_type=ValueError,
            )
        return values[0]

    @classmethod
    def _normalize_and_collect(
        cls, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[list["TracedWpArray"], tuple[Any, ...], dict[str, Any]]:
        traced: list[TracedWpArray] = []
        call_args = cls._normalize_node(args, traced, depth=0)
        call_kwargs = cls._normalize_node(kwargs, traced, depth=0)
        return traced, call_args, call_kwargs

    @classmethod
    def _normalize_node(
        cls, obj: Any, traced: list["TracedWpArray"], *, depth: int
    ) -> Any:
        cls._validate_scan_depth(depth)

        if isinstance(obj, cls):
            traced.append(obj)
            return obj.data
        if isinstance(obj, dict):
            return {
                key: cls._normalize_node(value, traced, depth=depth + 1)
                for key, value in obj.items()
            }
        if isinstance(obj, (list, tuple, set, frozenset)):
            return type(obj)(
                cls._normalize_node(item, traced, depth=depth + 1)
                for item in obj
            )
        return obj

    @staticmethod
    def _validate_scan_depth(depth: int) -> None:
        if depth > _MAX_PARAM_SCAN_DEPTH:
            _get_logger().fatal(
                "When traversing a nested structure in a Warp function call, "
                f"exceeded LEAPP's max traversal depth ({_MAX_PARAM_SCAN_DEPTH}).",
                error_type=RuntimeError,
            )

    @staticmethod
    def _select_trace_source(
        qualname: str, traced_inputs: list["TracedWpArray"]
    ) -> "TracedWpArray | None":
        if not traced_inputs:
            return None

        source = traced_inputs[0]
        for candidate in traced_inputs[1:]:
            if candidate.context_obj is not source.context_obj:
                _get_logger().fatal(
                    f"{qualname} received traced Warp arrays from different LEAPP "
                    "trace contexts.",
                    error_type=ValueError,
                )
        return source

    @classmethod
    def _process_post_call_arrays(
        cls,
        segment: Any | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        trace_source: "TracedWpArray | None",
    ) -> None:
        seen: set[int] = set()
        for value in (args, kwargs, result):
            cls._process_post_call_node(
                value, segment, trace_source, seen, depth=0
            )

    @classmethod
    def _process_post_call_node(
        cls,
        obj: Any,
        segment: Any | None,
        trace_source: "TracedWpArray | None",
        seen: set[int],
        *,
        depth: int,
    ) -> None:
        from leapp.leapp_graph.datatypes import (
            as_traced,
            is_tracable_tensor_type,
            promote_in_place,
        )

        cls._validate_scan_depth(depth)
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if is_tracable_tensor_type(obj):
            if trace_source is not None and isinstance(obj, wp.array):
                owner = getattr(obj, "context_obj", None)
                published = getattr(obj, "output_port", None) is not None
                if published or (
                    owner is not None and owner is not trace_source.context_obj
                ):
                    traced_array = as_traced(
                        obj,
                        trace_source.name,
                        trace_source.context_obj,
                        trace_source.proxy,
                    )
                elif owner is not None:
                    traced_array = obj
                else:
                    traced_array = promote_in_place(
                        obj,
                        trace_source.name,
                        trace_source.context_obj,
                        trace_source.proxy,
                    )
                if segment is not None:
                    segment.add_output_ref(traced_array)
                    traced_array.warp_segment = segment
            return

        if isinstance(obj, dict):
            values = obj.values()
        elif isinstance(obj, (list, tuple, set, frozenset)):
            values = obj
        else:
            return
        for value in values:
            cls._process_post_call_node(
                value,
                segment,
                trace_source,
                seen,
                depth=depth + 1,
            )

    @staticmethod
    def _carry_full_copy_port(
        backend: Any,
        func: Callable,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if (
            not backend.is_full_copy_function(func)
            or len(args) < 2
            or any(args[2:])
            or kwargs.get("dest_offset")
            or kwargs.get("src_offset")
            or kwargs.get("count")
        ):
            return

        dest, src = args[0], args[1]
        if (
            getattr(src, "output_port", None) is None
            or not isinstance(dest, TracedWpArray)
            or dest.shape != src.shape
            or dest.dtype != src.dtype
        ):
            return
        src.preserve_port(dest)

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
        return wp.to_torch(self)

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
