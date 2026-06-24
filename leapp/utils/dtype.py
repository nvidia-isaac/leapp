#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Backend-agnostic dtype mapping for LEAPP.

This module is intentionally free of *all* backend imports (torch, numpy, warp)
and of any ``leapp`` imports, so it can be imported from low-level datatype
modules without circular-import risk. Each backend registers a
:class:`DtypeCodec` from its own node-library module (``traced_tensor`` for
torch, ``traced_np_array`` for numpy, ``traced_wp_array`` for warp), describing
how to recognize its values and map its dtype objects to the common name
strings (e.g. ``"float32"``). This keeps each backend's dtype knowledge unified
with that backend's implementation.

The name -> ``torch.dtype`` direction (``map_to_torch_dtype``) is resolved
through the registered torch codec, so it too needs no torch import here.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple


# Canonical common dtype-name vocabulary shared across backends. Backend codecs
# map their dtype objects onto these names.
_VALID_DTYPE_NAMES = frozenset({
    "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool",
})


def map_to_torch_dtype(string):
    """Map a common name string (e.g. "float32") to a ``torch.dtype``.

    Resolved via the registered torch codec, so this module needs no torch
    import. The torch codec is registered when ``traced_tensor`` is imported,
    which always happens before this is called in a running LEAPP process.
    """
    for codec in _DTYPE_CODECS:
        if codec.backend == "torch":
            for dtype, name in codec.dtype_to_name.items():
                if name == string:
                    return dtype
            break
    raise ValueError(f"Unsupported string: {string}")


# =============================================================================
# Backend dtype-codec registry
# =============================================================================

@dataclass(frozen=True)
class DtypeCodec:
    # Backend label, e.g. "torch" / "numpy" / "warp".
    backend: str
    # True when ``value`` belongs to this backend.
    matches: Callable[[Any], bool]
    # Extract the backend dtype object from a backend value.
    value_dtype: Callable[[Any], Any]
    # Backend dtype object -> common name string.
    dtype_to_name: Dict[Any, str]


# constant value
_DTYPE_CODECS: List[DtypeCodec] = []


def register_dtype_codec(codec: DtypeCodec) -> None:
    """Register a backend dtype codec (idempotent per backend label)."""
    for existing in _DTYPE_CODECS:
        if existing.backend == codec.backend:
            return
    _DTYPE_CODECS.append(codec)


def dtype_to_name(dtype_obj) -> str:
    """Map a backend dtype object to its common name string (e.g. "float32")."""
    for codec in _DTYPE_CODECS:
        if dtype_obj in codec.dtype_to_name:
            return codec.dtype_to_name[dtype_obj]
    raise ValueError(f"Unsupported dtype: {dtype_obj!r}")


def value_to_name_and_shape(value) -> Tuple[str, tuple]:
    """Map a backend value to its (common dtype name, shape)."""
    for codec in _DTYPE_CODECS:
        if codec.matches(value):
            return codec.dtype_to_name[codec.value_dtype(value)], value.shape
    raise ValueError(f"Unsupported type: {type(value)}")


def warp_dtype_to_torch_name(dtype_obj=None, *, text: str | None = None) -> str:
    """Resolve a Warp dtype to a common torch dtype-name string (e.g. "float32").

    Resolution order:
    1. registry lookup of ``dtype_obj`` (the warp codec, when warp is present);
    2. the dtype object's ``__name__`` if it is a known torch dtype name
       (``wp.float32.__name__`` -> ``"float32"``);
    3. a recognizable token parsed from ``text`` (e.g. the stored
       ``"<class 'warp.types.float32'>"`` form).

    Raises ``ValueError`` if none resolve.
    """
    if dtype_obj is not None:
        try:
            return dtype_to_name(dtype_obj)
        except ValueError:
            pass

    name = getattr(dtype_obj, "__name__", None)
    if name in _VALID_DTYPE_NAMES:
        return name
    for token in reversed(re.findall(r"[A-Za-z][A-Za-z0-9_]*", text or "")):
        if token in _VALID_DTYPE_NAMES:
            return token

    raise ValueError(
        f"Cannot map Warp dtype "
        f"'{text if text is not None else dtype_obj}' to a torch dtype name"
    )
