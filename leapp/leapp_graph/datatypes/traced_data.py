#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Base class for traced data types (tensors, arrays, etc.)."""

from abc import ABC, abstractmethod
import operator
from typing import Any, Set, Optional

from torch.fx.proxy import Proxy
from leapp.utils.logging import _get_logger
from leapp.utils.dtype import dtype_to_name

from .proxy_view import ProxyView

import torch


class TracedData(ABC):
    """Abstract base class for traced data types.
    
    This class provides the common infrastructure for tracing operations
    on different data types (torch.Tensor, numpy.ndarray, etc.) using torch.fx.
    
    Child classes implement type-specific operation interception:
    - TracedTensor: Uses __torch_function__ for torch operations
    - TracedArray (future): Uses __array_ufunc__/__array_function__ for numpy
    
    The FX graph always records torch operations, so child classes must
    map their native operations to torch equivalents for proxy recording.
    """
    
    def __init__(self, value: Any, name: str, context, proxy: Proxy):
        """Initialize a TracedData instance.
        
        Args:
            value: The underlying data (torch.Tensor, numpy.ndarray, etc.)
            name: Name for the data (used in ONNX export and graph)
            context: The TraceContext that owns this data
            proxy: The fx.Proxy for graph recording
        """
        self._value = value
        self._init_tracing_state(name, context, proxy)
    
    # =========================================================================
    # Common Properties
    # =========================================================================

    def _init_tracing_state(self, name: str, context, proxy: Proxy) -> None:
        """Initialize common tracing metadata for TracedData subclasses.

        This starts a new root ``ProxyView``. Use it for a fresh carrier, an
        out-of-place result, or a boundary rewrap. The two alternatives are
        :meth:`_replace_tracing_state` for a value mutated in place and
        :meth:`_adopt_tracing_state` for a second handle onto one value.
        """
        self._name = name
        self._context = context
        self._proxy_view = ProxyView(proxy)
        # A value only earns an output port by being registered as an output of
        # its own node, so entering a node always starts without one.
        self._output_port = None

    def _replace_tracing_state(self, name: str, context, proxy: Proxy) -> None:
        """Rebind this value's existing view after its data changed in place.

        The value keeps its identity and only its graph representation changes,
        so the view object is reused and every carrier sharing it observes the
        new proxy. Building a view here instead would strand those carriers on
        the old proxy while the memory they describe has already moved on.
        """
        self._name = name
        self._context = context
        self._proxy_view.proxy = proxy
        self._output_port = None

    def _adopt_tracing_state(self, source: "TracedData") -> None:
        """Share ``source``'s view so both carriers describe one value.

        For a result that occupies exactly ``source``'s memory and layout, both
        projections between the two are the identity, so they need no view edge
        and can hold one root between them. A mutation through either then
        replaces the single root both of them read.
        """
        self._name = source.name
        self._context = source.context_obj
        self._proxy_view = source._proxy_view
        self._output_port = None

    @staticmethod
    def _name_from_proxy(proxy: Proxy) -> str:
        """Return the conventional traced-data name for an operation result."""
        if proxy is not None:
            return str(proxy.node.name)
        return "untraced"
    
    @property
    def proxy(self) -> Proxy:
        """Get the fx.Proxy for graph recording."""
        return self._proxy_view.proxy
    
    @property
    def name(self) -> str:
        """Get the name of this traced data."""
        return self._name
    
    @property
    def context(self) -> str:
        """Get the name of the context that owns this data."""
        if self._context is None:
            return "untraced"
        return self._context.name
    
    @property
    def context_obj(self):
        """Get the TraceContext that owns this data."""
        return self._context

    @property
    def output_port(self) -> Optional[str]:
        """Name of this node's output that published this value, if any.

        Together with ``context_obj`` this is the complete connection identity
        of a node boundary value. ``None`` means the value was never registered
        as an output of its node.
        """
        return self._output_port

    @output_port.setter
    def output_port(self, port: Optional[str]) -> None:
        self._output_port = port
    
    @property
    def is_tracing(self) -> bool:
        """Get the tracing status of the context that owns this data."""
        if self._context is None:
            return False
        return self._context.is_tracing
    
    @property
    @abstractmethod
    def tensor(self) -> torch.Tensor:
        """Get the underlying data as a torch.Tensor."""
        pass

    @property
    @abstractmethod
    def data(self) -> Any:
        """Get the underlying data."""
        pass

    def get_dtype_name(self) -> str:
        """Common dtype name (e.g. "float32") of the underlying value.

        Delegates to the backend dtype-codec registry, so each backend's
        mapping lives with that backend rather than in leapp core.
        """
        return dtype_to_name(self.data.dtype)
    # =========================================================================
    # Abstract Methods - Must be implemented by child classes
    # =========================================================================
    
    @abstractmethod
    def _new(self, value: Any, proxy: Proxy = None) -> "TracedData":
        """Create a new TracedData of the same type in the same context.
        
        Args:
            value: The new underlying value
            proxy: The proxy for the new data (can be None if not tracing)
            
        Returns:
            A new TracedData instance of the same type
        """
        pass

    # =========================================================================
    # Common Static Methods
    # =========================================================================
    
    @staticmethod
    def find_traced_data(obj):
        """Find the first TracedData in a supported nested structure.
        
        Args:
            obj: Object to search (can be TracedData, list, tuple, dict, or other)
            
        Returns:
            The first TracedData found, or None if not found
        """
        found = None

        def visit(item):
            nonlocal found
            if found is None and isinstance(item, TracedData):
                found = item
            return item

        TracedData._map_structure(obj, visit)
        return found
    
    @staticmethod
    def _map_structure(obj, leaf_fn):
        """Recursively apply leaf_fn while preserving common containers."""
        if isinstance(obj, slice):
            return slice(
                TracedData._map_structure(obj.start, leaf_fn),
                TracedData._map_structure(obj.stop, leaf_fn),
                TracedData._map_structure(obj.step, leaf_fn),
            )
        if isinstance(obj, (list, tuple)):
            mapped = tuple(
                TracedData._map_structure(item, leaf_fn) for item in obj
            )
            if hasattr(obj, "_fields"):
                # namedtuples take their fields positionally. Plain sequences and
                # torch's structseq return types instead take a single sequence.
                return type(obj)(*mapped)
            return type(obj)(mapped)
        if isinstance(obj, dict):
            return {
                key: TracedData._map_structure(value, leaf_fn)
                for key, value in obj.items()
            }
        return leaf_fn(obj)

    @staticmethod
    def unwrap_traced_data(obj):
        """Recursively unwrap TracedData to get raw values.
        
        Args:
            obj: Object to unwrap (can be TracedData, list, tuple, dict, or other)
            
        Returns:
            The unwrapped value with all TracedData replaced by their raw values
        """
        return TracedData._map_structure(
            obj,
            lambda item: item.data if isinstance(item, TracedData) else item,
        )
    
    @staticmethod
    def find_all_contexts(obj, contexts: Optional[Set[str]] = None) -> Set[str]:
        """Recursively find all unique context names.
        
        Args:
            obj: Object to search
            contexts: Set to accumulate context names (created if None)
            
        Returns:
            Set of context names found
        """
        if contexts is None:
            contexts = set()

        def collect(item):
            if isinstance(item, TracedData) and item.is_tracing:
                contexts.add(item.context)
            return item

        TracedData._map_structure(obj, collect)
        return contexts
    
    @staticmethod
    def extract_proxy(obj):
        """Recursively extract proxies from TracedData objects.
        
        Args:
            obj: Object to extract proxies from
            
        Returns:
            Object with TracedData replaced by their proxies
        """
        return TracedData._map_structure(
            obj,
            lambda item: item.proxy if isinstance(item, TracedData) else item,
        )
    
    # =========================================================================
    # Common Validation
    # =========================================================================
    
    def validate_status(self, args=None, kwargs=None) -> bool:
        """Validate that this TracedData can be used in the current context.
        
        Checks:
        1. If not tracing, returns False (operation should not be recorded)
        2. If tracing inside a traced function, raises an error
        3. If mixing TracedData from different contexts, raises an error
        
        Args:
            args: Positional arguments to check for other TracedData
            kwargs: Keyword arguments to check for other TracedData
            
        Returns:
            True if tracing should proceed, False if not tracing
            
        Raises:
            Exception: If TracedData is used inside a traced function
            Exception: If TracedData from multiple contexts are mixed
        """
        if not self.is_tracing:
            return False

        contexts = set()
        if args is not None:
            for arg in args:
                contexts = TracedData.find_all_contexts(arg, contexts)
        if kwargs is not None:
            for kwarg in kwargs.values():
                contexts = TracedData.find_all_contexts(kwarg, contexts)
        
        if len(contexts) > 1:
            cls_name = self.__class__.__name__
            _get_logger().fatal(
                f"Error: detected multiple {cls_name} contexts: {contexts} inside of a traced function.\n"
                "\n"
                f"This happens when you mix multiple active {cls_name}s from different contexts "
                "inside of a traced function/block. solutions:\n"
                "\n"
                "1. call output_tensors() to finalize one of the nodes first\n"
                "2. combine both nodes into a single node by calling input_tensors() with the same node name",
                error_type=Exception)
        return True

    # =========================================================================
    # Transit State
    # =========================================================================

    # Subclasses override these to name the allocating / relocating ops whose
    # results still hold the published boundary values, plus the native type of
    # those results. Empty means no inactive-source function preserves a port.
    _EQUIVALENT_COPY_NAMES: frozenset = frozenset()
    _NATIVE_TYPE = None

    @classmethod
    def _is_equivalent_copy(cls, func, source, result, args) -> bool:
        """Return whether ``func`` produced a value-identical copy of ``source``.

        Recognition is by ``func.__name__`` so torch method descriptors and
        numpy module functions share one template. Subclasses supply the name
        set and native result type; method-level copies that never reach
        dispatch (e.g. ``TracedNpArray.copy``) call ``preserve_port`` directly.
        """
        return (
            cls._NATIVE_TYPE is not None
            and getattr(func, "__name__", "") in cls._EQUIVALENT_COPY_NAMES
            and isinstance(result, cls._NATIVE_TYPE)
            and bool(args)
            and args[0] is source
        )

    @staticmethod
    def _is_complete_slice(key) -> bool:
        """Return whether key is an open ``slice(None)`` without comparing tensors."""
        return (
            isinstance(key, slice)
            and key.start is None
            and key.stop is None
            and key.step is None
        )

    @staticmethod
    def _is_full_assignment_key(key) -> bool:
        """Return whether an index covers the complete destination."""
        if key is Ellipsis:
            return True
        if isinstance(key, slice):
            return TracedData._is_complete_slice(key)
        return (
            isinstance(key, tuple)
            and key
            and all(TracedData._is_complete_slice(item) for item in key)
        )

    def preserve_port(self, result):
        """Give ``result`` this carrier's published output port, if any.

        ``result`` may already be a traced carrier (e.g. after an explicit
        ``as_traced`` conversion) or a native value. A native value is promoted
        first. With no published port, ``result`` is returned unchanged so
        callers can always promote-then-preserve without a special case.
        """
        if self._output_port is None:
            return result

        if not isinstance(result, TracedData):
            from leapp.leapp_graph.datatypes import as_traced

            result = as_traced(result, self._name, self._context, self.proxy)
        result.output_port = self._output_port
        return result

    def overwrite_port(self, key, value) -> None:
        """Update this carrier's boundary identity after a write made in transit.

        A full overwrite from a published output leaves this carrier holding
        that output's data, so it adopts the whole boundary identity. Any other
        write makes the data differ from whatever this carrier published, so it
        drops its own port and stops being a usable edge source.
        """
        if (
            getattr(value, "output_port", None) is not None
            and self._is_full_assignment_key(key)
            and tuple(value.shape) == tuple(self.shape)
            and value.dtype == self.dtype
        ):
            self._replace_tracing_state(
                value.name, value.context_obj, value.proxy
            )
            self._output_port = value.output_port
            return
        self._output_port = None

    # =========================================================================
    # Common Magic Methods
    # =========================================================================
    
    def __len__(self) -> int:
        """Length operator."""
        return len(self._value)
    
    def __bool__(self) -> bool:
        """Boolean conversion for TracedNpArray.
        
        During tracing, this indicates tensor-dependent control flow which
        produces a silently incorrect graph. Logs an error to alert the user.
        """
        if self.is_tracing:
            _get_logger().error(
                f"Attempted to use {self.__class__.__name__} '{self._name}' from node '{self.context}' "
                f"in a boolean context (if/while/and/or/not) during tracing.\n"
                f"\n"
                f"This typically happens with array-value-dependent control flow:\n"
                f"  - if array.mean() > 0.9: return early   (Early Exit)\n"
                f"  - while error.sum() > eps: refine()      (Iterative Refinement)\n"
                f"  - if array.any(): ...                     (Conditional on values)\n"
                f"\n"
                f"The traced graph is a static DAG and cannot represent dynamic branches.\n"
                f"Only the branch taken during tracing would be recorded, producing a\n"
                f"silently incorrect graph for other inputs.\n"
                f"\n"
                f"Alternatives:\n"
                f"  1. Use torch.where(condition, true_val, false_val) for simple if/else\n"
                f"  2. Use fixed iteration counts instead of dynamic while loops\n"
                f"  3. Break the trace: call output_tensors() before the conditional,\n"
                f"     then start a new input_tensors() after it"
            )
        return bool(self.data)
    
    def __str__(self) -> str:
        """String representation."""
        return f"{type(self).__name__}({self._value})"
    
    def __repr__(self) -> str:
        """String representation."""
        return f"{type(self).__name__}({self._value})"
    
    def __format__(self, format_spec: str) -> str:
        """Format the TracedData by delegating to the underlying value."""
        return self._value.__format__(format_spec)

    # =========================================================================
    # Setitem Proxy Generation
    # =========================================================================
    
    def _register_setitem_tensor(self, value: torch.Tensor, prefix: str):
        """Register a tensor constant and return its get_attr proxy."""
        attr_base = f"{prefix}_{id(value)}"
        attr_name = attr_base
        suffix = 1
        while hasattr(self._context.tracer.root, attr_name):
            attr_name = f"{attr_base}_{suffix}"
            suffix += 1
        self._context.tracer.root.register_buffer(
            attr_name, value.clone().detach()
        )
        return self._context.tracer.create_proxy(
            "get_attr", attr_name, (), {}
        )

    @staticmethod
    def _tensor_index_key(key):
        """Replace traced indices with their torch graph-time values."""
        return TracedData._map_structure(
            key,
            lambda item: item.tensor if isinstance(item, TracedData) else item,
        )

    def _proxy_index_key(self, key):
        """Replace tensor indices with graph values or registered constants."""
        def convert(item):
            if isinstance(item, TracedData):
                return item.proxy
            if isinstance(item, torch.Tensor):
                return self._register_setitem_tensor(item, "_setitem_key")
            return item

        return TracedData._map_structure(key, convert)

    @staticmethod
    def _slice_needs_narrow(item):
        """Whether a slice carries tensor bounds that no Python slice can hold."""
        if not isinstance(item, slice):
            return False
        return any(
            isinstance(bound, (TracedData, torch.Tensor))
            for bound in (item.start, item.stop, item.step)
        )

    @staticmethod
    def _is_supported_index_key(key):
        """Return whether key can be lowered through the flat-index pipeline."""
        if isinstance(key, TracedData):
            return key.tensor.dtype in (torch.bool, torch.int32, torch.int64)
        if isinstance(key, torch.Tensor):
            return key.dtype in (torch.bool, torch.int32, torch.int64)
        if isinstance(key, (list, tuple)):
            if not all(TracedData._is_supported_index_key(item) for item in key):
                return False
            if any(TracedData._slice_needs_narrow(item) for item in key):
                # Tensor bounds are lowered per dimension, so the remaining key
                # must keep dimensions in place.
                return all(isinstance(item, (slice, int)) for item in key)
            return True
        if isinstance(key, slice):
            for bound in (key.start, key.stop, key.step):
                if bound is None or isinstance(bound, int):
                    continue
                if isinstance(bound, TracedData):
                    bound = bound.tensor
                if not isinstance(bound, torch.Tensor) or bound.ndim != 0:
                    return False
                try:
                    operator.index(bound)
                except TypeError:
                    return False
            if TracedData._slice_needs_narrow(key):
                # narrow() cannot express a stride.
                return not isinstance(
                    key.step, (TracedData, torch.Tensor)
                ) and key.step in (None, 1)
            return True
        return key is None or key is Ellipsis or isinstance(key, int)

    def _narrow_bound(self, bound, default, size):
        """Resolve one slice bound to an integer or a graph value."""
        if bound is None:
            return default
        if isinstance(bound, TracedData):
            return self._context.tracer.create_proxy(
                "call_method", "item", (bound.proxy,), {}
            )
        value = operator.index(bound)
        if value < 0:
            value += size
        return min(max(value, 0), size)

    def _lower_index_key(self, proxy, key):
        """Record the selection described by key against proxy.

        Slices with tensor bounds become ``narrow`` calls, because Python
        slicing needs concrete integers and so cannot hold a bound that is only
        known at runtime. Whatever is left of the key is recorded as one
        getitem. Runtime bounds are used as given, so out-of-range values fail
        the way ``narrow`` does rather than being clamped the way slicing would.
        """
        elements = key if isinstance(key, tuple) else (key,)
        shape = tuple(self.tensor.shape)
        residual = list(elements)
        narrowed = False
        for dim, item in enumerate(elements):
            if not self._slice_needs_narrow(item):
                continue
            start = self._narrow_bound(item.start, 0, shape[dim])
            stop = self._narrow_bound(item.stop, shape[dim], shape[dim])
            if isinstance(start, Proxy) or isinstance(stop, Proxy):
                length = self._context.tracer.create_proxy(
                    "call_function", operator.sub, (stop, start), {}
                )
            else:
                length = max(stop - start, 0)
            proxy = self._context.tracer.create_proxy(
                "call_function", torch.narrow, (proxy, dim, start, length), {}
            )
            residual[dim] = slice(None)
            narrowed = True

        if narrowed and all(item == slice(None) for item in residual):
            return proxy
        residual_key = tuple(residual) if isinstance(key, tuple) else residual[0]
        return self._context.tracer.create_proxy(
            "call_function", operator.getitem,
            (proxy, self._proxy_index_key(residual_key)), {}
        )

    def _create_getitem_proxy(self, key):
        """Create a functional getitem proxy for a supported index key."""
        if not self._is_supported_index_key(key):
            return None
        return self._lower_index_key(self.proxy, key)

    def _create_setitem_proxy(self, key, value_proxy, real_value=None):
        """Lower indexed assignment to one functional flat ``index_put``.

        Destination tracing, source tracing, and index kind are deliberately
        independent here. Static indices become constant flat positions;
        traced integer or boolean indices select positions inside the graph.
        """
        if not self._is_supported_index_key(key):
            return None

        destination = self.tensor
        tensor_shape = tuple(destination.shape)
        real_key = self._tensor_index_key(key)
        target_shape = tuple(destination[real_key].shape)

        flat_positions = torch.arange(
            destination.numel(), dtype=torch.long, device=destination.device
        ).reshape(tensor_shape)
        if TracedData.find_traced_data(key) is None:
            flat_indices = flat_positions[real_key].reshape(-1)
            indices_proxy = self._register_setitem_tensor(
                flat_indices, "_setitem_indices"
            )
        else:
            positions_proxy = self._register_setitem_tensor(
                flat_positions, "_setitem_positions"
            )
            selected_proxy = self._lower_index_key(positions_proxy, key)
            indices_proxy = self._context.tracer.create_proxy(
                "call_method", "reshape", (selected_proxy, (-1,)), {}
            )

        source_proxy = value_proxy
        if not isinstance(source_proxy, Proxy):
            source_tensor = torch.as_tensor(
                real_value if real_value is not None else source_proxy,
                dtype=destination.dtype,
                device=destination.device,
            )
            if source_tensor.numel() != 1 and tuple(source_tensor.shape) != target_shape:
                source_tensor = torch.broadcast_to(source_tensor, target_shape).clone()
            source_proxy = self._register_setitem_tensor(
                source_tensor, "_setitem_value"
            )
        else:
            source_numel = getattr(real_value, "numel", lambda: 1)()
            if source_numel != 1 and tuple(getattr(real_value, "shape", ())) != target_shape:
                source_proxy = self._context.tracer.create_proxy(
                    "call_function", torch.broadcast_to,
                    (source_proxy, target_shape), {}
                )

        destination_proxy = self.proxy
        source_proxy = self._context.tracer.create_proxy(
            "call_method", "to", (source_proxy, destination_proxy), {}
        )
        flat_source_proxy = self._context.tracer.create_proxy(
            "call_method", "reshape", (source_proxy, (-1,)), {}
        )
        flat_destination_proxy = self._context.tracer.create_proxy(
            "call_method", "reshape", (destination_proxy, (-1,)), {}
        )
        updated_proxy = self._context.tracer.create_proxy(
            "call_function", torch.index_put,
            (flat_destination_proxy, (indices_proxy,), flat_source_proxy), {}
        )
        return self._context.tracer.create_proxy(
            "call_method", "reshape", (updated_proxy, tensor_shape), {}
        )

    def _update_setitem_proxy(self, key, value_proxy, real_value=None):
        """Record a functional assignment and install its resulting proxy."""
        proxy_out = self._create_setitem_proxy(
            key, value_proxy, real_value=real_value
        )
        if proxy_out is None:
            return False
        self._proxy_view.proxy = proxy_out
        return True
