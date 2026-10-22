"""Traced Warp array support."""

from typing import Any, Callable

import torch
from torch.fx.proxy import Proxy

import warp as wp

from ..proxy import (
    bind_proxy,
    layout_key,
    is_same_value,
    update_proxy,
    conversion_proxy,
)
from ..traced_data import TracedData
from leapp.utils.dtype import DtypeCodec, register_dtype_codec
from leapp.utils.logging import _get_logger


_MAX_PARAM_SCAN_DEPTH = 16


# Register the Warp dtype codec so leapp core can convert warp dtypes to common
# name strings without importing warp itself. Mirrors the torch/numpy entries
# and only runs when this (warp-gated) module is imported.
register_dtype_codec(
    DtypeCodec(
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
    )
)


class TracedWpArray(wp.array, TracedData):
    """A traced ``wp.array`` subclass that can be class-swapped in place.

    ``wp.array`` must remain the first base. That preserves its CPython object
    layout lineage so exact raw Warp arrays can be promoted with
    ``arr.__class__ = TracedWpArray``.
    """

    def __new__(cls, array, name, context, proxy=None):
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
        bind_proxy(obj, name, context, proxy)
        return obj

    def __init__(self, array: wp.array, name: str, context, proxy=None):
        # ``__new__`` initializes Warp storage and tracing metadata.
        pass

    @classmethod
    def make_traced_in_place(
        cls, array: wp.array, name: str, context, proxy=None
    ) -> "TracedWpArray":
        """Turn an existing raw ``wp.array`` into a traced array in place."""
        if type(array) is cls:
            bind_proxy(array, name, context, proxy)
            return array
        if type(array) is not wp.array:
            _get_logger().fatal(
                "Can only class-swap exact raw wp.array instances into "
                f"{cls.__name__}; got {type(array).__name__}",
                error_type=TypeError,
            )
        array.__class__ = cls
        bind_proxy(array, name, context, proxy)
        return array

    @classmethod
    def clear_tracing_state(cls, array: "TracedWpArray") -> wp.array:
        """Undo :meth:`make_traced_in_place`, returning a raw ``wp.array``.

        The proxy points at one graph, so a buffer that outlives
        the pass that promoted it has to shed that provenance before the next
        pass can describe it.
        """
        for attribute in (
            "_name",
            "_context",
            "_proxy",
            "_output_port",
            "_leapp_warp_segment",
        ):
            array.__dict__.pop(attribute, None)
        array.__class__ = wp.array
        return array

    def rebind_tracing_proxy(self, name: str, context, proxy: Proxy) -> None:
        """Record the runner's write into this array's stable FX handle.

        Existing Torch/NumPy aliases keep that handle; functionalization makes
        the runner result visible through all of their recorded views.
        """
        update_proxy(self, name, context, proxy)

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
            handled, result = cls._handle_boundary_call(backend, func, args, kwargs)
            if handled:
                return result

        traced_inputs, call_args, call_kwargs = cls._normalize_and_collect(args, kwargs)
        trace_source = cls._select_trace_source(qualname, traced_inputs)
        segment = backend.resolve_or_begin_warp_segment(trace_source)

        if segment is not None:
            backend.record_segment_inputs(segment, qualname, traced_inputs)

        with backend.pause_context():
            result = func(*call_args, **call_kwargs)

            # Allocation handles are capture bookkeeping, not Torch consumers
            # that should close an automatic Warp region.
            cls._process_post_call_arrays(
                func, segment, args, kwargs, result, trace_source
            )
        if segment is not None and getattr(func, "__name__", None) == "launch":
            outputs = kwargs.get("outputs", args[3] if len(args) > 3 else ())
            if any(getattr(array, "_tensor", None) is not None for array in outputs):
                backend.close_warp_segment()
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
            # Keep the caller's Torch object as the owner of the Warp view so
            # writes through that view can update the caller's tracing state.
            alias = getattr(raw, "_tensor", None)
            if (
                isinstance(src, torch.Tensor)
                and type(alias) is torch.Tensor
                and layout_key(alias) == layout_key(src)
            ):
                raw._tensor = src
            return True, as_traced(
                raw, src.name, src.context_obj, conversion_proxy(src, raw)
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
            if obj.describes_replaced_graph():
                cls.clear_tracing_state(obj)
                return obj
            traced.append(obj)
            return obj.data
        if isinstance(obj, dict):
            return {
                key: cls._normalize_node(value, traced, depth=depth + 1)
                for key, value in obj.items()
            }
        if type(obj) in (list, tuple, set, frozenset):
            return type(obj)(
                cls._normalize_node(item, traced, depth=depth + 1) for item in obj
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
        func: Callable,
        segment: Any | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        result: Any,
        trace_source: "TracedWpArray | None",
    ) -> None:
        writable_ids: set[int] = set()
        if getattr(func, "__name__", None) == "launch":
            for values in (
                kwargs.get("outputs", args[3] if len(args) > 3 else ()),
                kwargs.get("adj_outputs", args[5] if len(args) > 5 else ()),
            ):
                writable_ids.update(id(value) for value in values)
        elif getattr(func, "__name__", None) == "copy":
            dest = kwargs.get("dest", args[0] if args else None)
            writable_ids.add(id(dest))
        seen: set[int] = set()
        for value in (args, kwargs, result):
            cls._process_post_call_node(
                value, segment, trace_source, seen, writable_ids, depth=0
            )

    @classmethod
    def _process_post_call_node(
        cls,
        obj: Any,
        segment: Any | None,
        trace_source: "TracedWpArray | None",
        seen: set[int],
        writable_ids: set[int],
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
                if owner is trace_source.context_obj and not published:
                    traced_array = obj
                else:
                    proxy = cls._allocation_proxy(obj, trace_source)
                    if published or owner is not None:
                        traced_array = as_traced(
                            obj, trace_source.name, trace_source.context_obj, proxy
                        )
                    else:
                        traced_array = promote_in_place(
                            obj, trace_source.name, trace_source.context_obj, proxy
                        )
                torch_owner = getattr(obj, "_tensor", None)
                if isinstance(torch_owner, torch.Tensor) and obj_id in writable_ids:
                    if not is_same_value(traced_array, torch_owner):
                        _get_logger().fatal(
                            "Warp output aliases a Torch tensor with an incompatible layout; "
                            "cannot trace reads through the Torch alias.",
                            error_type=RuntimeError,
                        )
                    if not isinstance(torch_owner, TracedData):
                        promote_in_place(
                            torch_owner,
                            traced_array.name,
                            traced_array.context_obj,
                            None,
                        )
                    bind_proxy(
                        torch_owner,
                        traced_array.name,
                        traced_array.context_obj,
                        traced_array.proxy,
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
                writable_ids,
                depth=depth + 1,
            )

    @staticmethod
    def _allocation_proxy(array, source):
        """Give an unfinished Warp output its own stable graph allocation.

        Conversions made before segment close keep this handle. Publication is
        an ordinary copy_ from the runner output, normalized with other writes.
        """
        if is_same_value(source, array):
            return source.proxy
        value = wp.to_torch(array)
        return source.context_obj.tracer.create_proxy(
            "call_function",
            torch.empty,
            (tuple(value.shape),),
            {"dtype": value.dtype, "device": value.device},
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
