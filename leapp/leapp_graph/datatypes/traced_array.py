import operator

import numpy as np
from torch.fx.proxy import Proxy
from leapp._logging import _get_logger
from leapp.tracing_lock import TracingLock
from .numpy_compatibility import (
    AXIS_TO_DIM_FUNCTIONS,
    convert_numpy_arg_to_torch,
    get_torch_equivalent_ufunc,
    get_torch_equivalent_func,
)
from .global_patching import is_numpy_patching_enabled


class TracedArray:

    def __init__(self, tensor: np.ndarray, name: str, context, proxy: Proxy):
        """Initialize a TracedTensor.

        TracedTensors can only be created via TraceContext.create_input().

        Args:
            tensor: The actual numpy array to wrap
            name: Name for the tensor (used in ONNX export and graph)
            context: The TraceContext that owns this tensor
            proxy: The fx.Proxy for graph recording
        """
        self._tensor = tensor
        self._name = name
        self._context = context
        self._proxy = proxy
        self._global_tracing_lock = TracingLock()


    @property
    def tensor(self) -> np.ndarray:
        """Get the underlying numpy array."""
        return self._tensor

    @property
    def proxy(self) -> Proxy:
        """Get the fx.Proxy for graph recording."""
        return self._proxy

    @property
    def name(self) -> str:
        """Get the name of the tensor."""
        return self._name

    @property
    def context(self) -> str:
        """Get the name of the  that owns this tensor."""
        return self._context.name

    @property
    def context_obj(self):
        """Get the TracedTensorNode that owns this tensor."""
        return self._context

    @property
    def is_tracing(self) -> bool:
        """Get the tracing status of the TracedTensorNode that owns this tensor."""
        return self._context.is_tracing


    @property
    def __class__(self):
        """Return underlying tensor type to pass isinstance checks.
        
        This allows TracedTensor to pass isinstance(x, torch.Tensor) checks
        in external code, while still being identifiable as TracedTensor
        for our internal patches (isinstance checks both actual type and __class__).
        
        Note: Some C++ functions that check isinstance then access internal
        tensor structures may crash. These should be patched in global_patching.py.
        """
        return type(self._tensor)

    def _new(self, tensor: np.ndarray, proxy: Proxy = None) -> "TracedArray":
        """Create a new TracedTensor in the same context.

        Intermediate tensors get auto-generated names based on the operation.
        When not tracing, proxy can be None.
        """
        if proxy is not None:
            # Generate a name based on the proxy node's name
            intermediate_name = str(proxy.node.name)
        else:
            # When not tracing, use a placeholder name
            intermediate_name = "untraced"
        return TracedArray(tensor, intermediate_name, self._context, proxy)