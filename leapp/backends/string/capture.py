# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture string operations: traced carriers, interception, and CPython patches.

Only ordinary descriptors are replaced. Operators need traced value methods;
changing a type dictionary does not update CPython's special-method slots.
"""

from contextvars import ContextVar
from enum import Enum, IntEnum, IntFlag
import ctypes
import functools
import gc
import json
import operator
import re
import sys
import threading
import types

import numpy as np
import torch

from leapp.leapp_graph.datatypes import TracedData, as_traced
from leapp.leapp_graph.datatypes._attribute_patching import AttributePatchRegistry
from leapp.leapp_graph.traced_node import required_nodes
from leapp.utils.logging import _get_logger

from . import StringCaptureError
from .tokenizer import TokenizerCall
from .utf8 import TextOperations, unsupported

ACTIVE = ContextVar("leapp_string_node", default=None)
BUSY = ContextVar("leapp_string_patch_busy", default=False)


def plain(value):
    return TracedData._map_structure(
        value,
        lambda item: (
            item.data
            if isinstance(item, TracedData)
            else plain(item._value)
            if isinstance(item, UnloweredValue)
            else item
        ),
        map_keys=True,
    )


class TracedString(str, TracedData):
    def __new__(cls, value, node, proxy):
        return str.__new__(cls, value)

    def __init__(self, value, node, proxy):
        TracedData.__init__(self, value, self._name_from_proxy(proxy), node, proxy)
        self._capture_token = node._text_token

    @property
    def data(self):
        return self._value

    @property
    def tensor(self):
        """A packed sample; export capacity is chosen at the output boundary."""
        from .utf8 import pack

        return torch.from_numpy(pack(self.data, max(1, len(self.data.encode("utf-8")))))

    def get_dtype_name(self):
        return "uint8"

    def _new(self, value, proxy=None):
        result = type(self)(value, self.context_obj, proxy)
        result._capture_token = self._capture_token
        return result

    def _control(self, *args):
        self.context_obj._text_error = (
            "Input-dependent string control, indexing, or iteration is unsupported"
        )
        raise StringCaptureError(self.context_obj._text_error)

    __bool__ = __len__ = __iter__ = __getitem__ = __contains__ = _control
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _control
    __mul__ = __rmul__ = __mod__ = __format__ = _control
    __str__ = _control
    __hash__ = str.__hash__

    def __repr__(self):
        # CPython's %r/ascii formatting can consume repr's Unicode payload
        # without invoking our methods, even if repr returns a TracedString.
        if BUSY.get():
            return repr(self._value)
        return self._control()

    def _add(self, left, right):
        if BUSY.get():
            return operator.add(plain(left), plain(right))
        node = self.context_obj
        if ACTIVE.get() is not node:
            raise StringCaptureError("Declare a new input before reusing captured text")
        token = BUSY.set(True)
        try:
            return capture_method(node, "add", operator.add, (left, right), {})
        finally:
            BUSY.reset(token)

    def __add__(self, other):
        return self._add(self, other)

    def __radd__(self, other):
        return self._add(other, self)


class UnloweredValue:
    """Retain unsupported results until pruning; reject Python control/conversions."""

    def __init__(self, value, node, proxy):
        self._value, self.context_obj, self.proxy = value, node, proxy
        self._capture_token = node._text_token

    _control = TracedString._control
    __bool__ = __len__ = __iter__ = __contains__ = _control
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _control
    __int__ = __float__ = __index__ = __str__ = __repr__ = __format__ = _control
    __hash__ = __getattr__ = _control

    def __getitem__(self, key):
        node = self.context_obj
        if ACTIVE.get() is not node:
            raise StringCaptureError("Declare a new input before reusing captured text")
        proxy = reference(node, self)
        return wrap(
            node,
            self._value[plain(key)],
            record(
                node,
                "call_function",
                operator.getitem,
                (proxy, arguments(node, key)),
                {},
            ),
        )


def reference(node, value):
    if isinstance(value, (TracedString, UnloweredValue)):
        if (
            value.context_obj is not node
            or value._capture_token is not node._text_token
        ):
            raise StringCaptureError("Text belongs to another capture")
        return value.proxy
    if (
        isinstance(value, TracedData)
        and value.context_obj is node
        and node.is_tracing
        and value.proxy is not None
    ):
        return value.proxy
    return None


def observes(node, value):
    found = False

    def visit(item):
        nonlocal found
        found |= reference(node, item) is not None
        return item

    TracedData._map_structure(value, visit, map_keys=True)
    return found


def arguments(node, value):
    def convert(item):
        proxy = reference(node, item)
        if proxy is not None:
            return proxy
        if isinstance(item, (IntEnum, IntFlag)):
            return int(item)
        if isinstance(item, str) and isinstance(item, Enum):
            return str.__str__(item)
        if type(item) in (str, int, bool, type(None)):
            return item
        raise StringCaptureError(
            f"Untraced value in string capture: {type(item).__name__}"
        )

    return TracedData._map_structure(value, convert, map_keys=True)


def record(node, operation, target, args, kwargs, factory=None, resource=None):
    proxy = node.tracer.create_proxy(operation, target, args, kwargs)
    if factory is not None:
        proxy.node.meta["leapp_text_factory"] = factory
        proxy.node.meta["leapp_text_resource"] = resource
        if node.is_tracing and not hasattr(node.tracer.root, target):
            # Resources and callbacks are initialized only after output pruning.
            node.tracer.root.add_module(target, torch.nn.Module())
    node._text_events.append(proxy.node)
    return proxy


def wrap(node, value, proxy):
    if type(value) is str:
        return TracedString(value, node, proxy)
    if isinstance(value, dict) or hasattr(value, "items"):
        return {
            key: wrap(
                node,
                item,
                record(node, "call_function", operator.getitem, (proxy, key), {}),
            )
            for key, item in value.items()
        }
    if isinstance(value, (torch.Tensor, np.ndarray)):
        value = as_traced(
            value, proxy.node.name, node, proxy if node.is_tracing else None
        )
        if isinstance(value, torch.Tensor):
            value._data_dependent_shape = True
        return value
    return UnloweredValue(value, node, proxy)


def capture_method(node, name, implementation, args, kwargs):
    supported = name in {
        "add",
        "join",
        "lower",
        "upper",
        "strip",
        "lstrip",
        "rstrip",
        "replace",
    }
    return capture_call(
        node,
        name,
        implementation,
        args,
        kwargs,
        factory=(TextOperations, ()) if supported else None,
        prefix=(name,) if supported else (),
    )


def capture_call(node, name, implementation, args, kwargs, *, factory=None, prefix=()):
    positional, keywords = arguments(node, (args, kwargs))
    if factory is None:
        _get_logger().error(f"String call {name!r} has no tensor lowering")
        dependencies = []
        TracedData._map_structure(
            (positional, keywords),
            lambda item: (
                dependencies.append(item) if isinstance(item, torch.fx.Proxy) else item
            ),
            map_keys=True,
        )
        target, positional, keywords = unsupported, (name, *dependencies), {}
    else:
        target = "_leapp_text" if prefix else f"_leapp_text_{len(node._text_events)}"
        positional = (*prefix, *positional)
    result = implementation(*plain(args), **plain(kwargs))
    proxy = record(
        node,
        "call_module" if factory else "call_function",
        target,
        positional,
        keywords,
        factory,
        name,
    )
    if factory is None:
        proxy.node.meta["leapp_capture_error"] = (
            f"unsupported call: {name} has no tensor lowering"
        )
    return wrap(node, result, proxy)


def finish_capture(node, payload):
    """Reject failed text control and return the live dependencies of outputs."""
    if node._text_error:
        raise StringCaptureError(node._text_error)
    roots = []
    for value in payload.values():
        proxy = reference(node, value)
        if proxy is None:
            proxy = getattr(value, "proxy", None)
        if proxy is not None:
            roots.append(proxy.node)
    return required_nodes(node.tracer.graph, roots, preserve_mutations=True)


_CAPTURE_LOCK = threading.Lock()


class _Registry(AttributePatchRegistry):
    @staticmethod
    def _set(owner, name, value):
        if owner not in (str, re.Pattern):
            return setattr(owner, name, value)
        # CPython exposes the dictionary through a read-only mappingproxy.
        # Avoid casting PyTypeObject: its layout changes between versions.
        dictionaries = gc.get_referents(owner.__dict__)
        if len(dictionaries) != 1 or type(dictionaries[0]) is not dict:
            raise RuntimeError("Cannot access CPython's type dictionary")
        dictionaries[0][name] = value
        ctypes.pythonapi.PyType_Modified(ctypes.py_object(owner))


class Patches:
    def __init__(self, node, tokenizers):
        self.node = node
        self.tokenizers = tokenizers
        self._registry = _Registry()
        self._owns_lock = False

    def _replace(self, owner, name, wrapper):
        self._registry.install(owner, name, vars(owner)[name], wrapper)

    def install(self):
        if sys.implementation.name != "cpython":
            raise RuntimeError("String method patching requires CPython")
        if not _CAPTURE_LOCK.acquire(blocking=False):
            raise RuntimeError("Concurrent string capture sessions are unsupported")
        self._owns_lock = True
        try:
            for name, original in tuple(str.__dict__.items()):
                if not name.startswith("_") and isinstance(
                    original, types.MethodDescriptorType
                ):
                    self._replace(str, name, self._method(name, original))
            self._replace(re, "sub", self._function(re.sub, "re.sub"))
            # Bound receivers are configuration, not captured text arguments.
            self._replace(
                re.Pattern,
                "sub",
                self._function(re.Pattern.sub, "re.Pattern.sub", bound=True),
            )
            # Hook the encoder, so saved json.dumps references also reach us.
            self._replace(
                json.JSONEncoder,
                "encode",
                self._function(
                    json.JSONEncoder.encode,
                    "json.encoder.JSONEncoder.encode",
                    bound=True,
                ),
            )
            try:
                from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            except ModuleNotFoundError as error:
                if error.name != "transformers":
                    raise
            else:
                # Intercept loaded overrides too: an unsupported wrapper must
                # not bypass discovery and publish constant capture-time IDs.
                pending, seen = [PreTrainedTokenizerBase], set()
                while pending:
                    owner = pending.pop()
                    if owner in seen:
                        continue
                    seen.add(owner)
                    pending.extend(owner.__subclasses__())
                    if "__call__" in owner.__dict__:
                        self._replace(
                            owner, "__call__", self._tokenizer(owner.__call__)
                        )
        except BaseException:
            self.restore()
            raise

    def _intercept(self, original, capture, *, tracked_only=True):
        def wrapped(*args, **kwargs):
            node = ACTIVE.get()
            if BUSY.get() or node is not self.node:
                return original(*args, **kwargs)
            token = BUSY.set(True)
            try:
                if tracked_only and not observes(node, (args, kwargs)):
                    return original(*args, **kwargs)
                return capture(node, args, kwargs)
            finally:
                BUSY.reset(token)

        return wrapped

    def _method(self, name, original):
        return self._intercept(
            original,
            lambda node, args, kwargs: capture_method(
                node, name, original, args, kwargs
            ),
        )

    def _function(self, original, binding, *, bound=False):
        if getattr(original, "_leapp_function_patch", False):
            return original

        def capture(node, args, kwargs):
            call = original
            if bound:
                call, args = functools.partial(original, args[0]), args[1:]
            return capture_call(node, binding, call, args, kwargs)

        return self._intercept(original, capture)

    def _tokenizer(self, original):
        def capture(node, args, kwargs):
            receiver, args = args[0], args[1:]
            for label, (tokenizer, _) in self.tokenizers.items():
                if receiver is tokenizer:
                    break
            else:
                if not observes(node, (args, kwargs)):
                    return original(receiver, *args, **kwargs)
                from .tokenizer import discover

                label = discover(self.tokenizers, receiver)
            positional, keywords = arguments(node, (args, kwargs))
            result = original(receiver, *plain(args), **plain(kwargs))
            # HF has already applied per-call padding/truncation to its backend.
            # Freeze that exact configuration before a later call changes it.
            proxy = record(
                node,
                "call_module",
                f"_leapp_text_{len(node._text_events)}",
                positional,
                keywords,
                (
                    TokenizerCall,
                    (receiver.backend_tokenizer.to_str(), tuple(result), keywords),
                ),
                label,
            )
            return wrap(node, result, proxy)

        # Explicitly registered tokenizers also capture calls with constant text.
        wrapped = functools.wraps(original)(
            self._intercept(original, capture, tracked_only=False)
        )
        wrapped._leapp_tokenizer_original = original
        return wrapped

    def restore(self):
        self._registry.restore()
        if self._owns_lock:
            self._owns_lock = False
            _CAPTURE_LOCK.release()
