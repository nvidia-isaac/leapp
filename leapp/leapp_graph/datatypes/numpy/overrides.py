# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NumPy semantic/export exceptions to the automatic torch._numpy lowering.

Implementations accept torch._numpy arrays and NumPy argument conventions and
return tensors or torch._numpy arrays. The shared adapter captures and validates
both overrides and upstream implementations through the same ATen path.
"""

import numpy as np
import torch
import torch._numpy as tnp
import torch.nn.functional as F


def _cbrt(x, *, dtype=None):
    # torch._numpy uses pow(x, 1/3), which produces NaN for negative inputs.
    x = tnp.asarray(x, dtype=dtype).tensor
    if not x.is_floating_point():
        x = x.to(torch.float64)
    return torch.copysign(torch.abs(x).pow(1 / 3), x)


def _pad(array, pad_width, mode="constant", *, constant_values=0):
    # Keep static widths out of tensor normalization: F.pad needs Python ints.
    if mode != "constant" or isinstance(pad_width, tnp.ndarray):
        raise NotImplementedError(
            "pad requires constant mode and static padding widths"
        )
    widths = np.broadcast_to(np.asarray(pad_width), (array.ndim, 2))
    if widths.dtype.kind not in "iu" or np.any(widths < 0):
        raise NotImplementedError("pad widths must be nonnegative integers")
    if not isinstance(constant_values, (int, float, bool)):
        raise NotImplementedError("pad requires a scalar constant_values")
    return F.pad(
        array.tensor, tuple(int(x) for x in widths[::-1].flat), value=constant_values
    )


def _nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    # Missing from torch._numpy. NumPy promotes integer inputs to float64.
    a = tnp.asarray(a, dtype=dtype).tensor
    if not a.is_floating_point() and not a.is_complex():
        a = a.to(torch.float64)
    if axis == ():
        return a
    return torch.nanmean(a, dim=axis, keepdim=keepdims)


def _sorted(a, axis, kind, order, stable):
    # Preserve LEAPP's static-K ONNX lowering. Stable/structured sorting needs
    # a separate implementation; do not silently ignore those NumPy options.
    if kind not in (None, "quicksort") or order is not None or stable:
        raise NotImplementedError("only default, unstructured sorting is supported")
    a = tnp.asarray(a).tensor
    if axis is None:
        a, axis = a.flatten(), -1
    return torch.topk(a, a.shape[axis], dim=axis, largest=False, sorted=True)


def _sort(a, axis=-1, kind=None, order=None, *, stable=None):
    return _sorted(a, axis, kind, order, stable).values


def _argsort(a, axis=-1, kind=None, order=None, *, stable=None):
    return _sorted(a, axis, kind, order, stable).indices


NUMPY_OVERRIDES = {
    np.cbrt: _cbrt,
    np.pad: _pad,
    np.nanmean: _nanmean,
    np.sort: _sort,
    np.argsort: _argsort,
}
