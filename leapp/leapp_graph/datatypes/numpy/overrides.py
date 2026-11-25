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


def _quantile(
    a,
    q,
    axis=None,
    out=None,
    overwrite_input=False,
    method="linear",
    keepdims=False,
    *,
    interpolation=None,
):
    if interpolation is not None:
        raise NotImplementedError("use quantile's method argument")
    a = tnp.asarray(a).tensor
    scalar_q = isinstance(q, (int, float))
    q = tnp.asarray(q).tensor
    shape, all_axes = a.shape, axis is None
    numpy2 = np.lib.NumpyVersion(np.__version__) >= "2.0.0"
    # NumPy 2 preserves floating input precision for Python scalar q.
    if numpy2 and scalar_q and a.is_floating_point():
        q = q.to(a.dtype)
    if axis is None:
        a, axis = a.flatten(), 0
    else:
        if isinstance(axis, tuple):
            if len(axis) != 1:
                raise NotImplementedError("quantile requires a single axis")
            axis = axis[0]
        if axis < 0:
            axis += a.ndim
    values = torch.sort(a, dim=axis).values.movedim(axis, 0)
    position = q * (values.shape[0] - 1)
    previous = position.floor() if position.is_floating_point() else position
    lower = previous.to(torch.int64)
    upper = (lower + 1).clamp(max=values.shape[0] - 1)

    def take(index):
        return torch.index_select(values, 0, index.flatten()).reshape(
            q.shape + values.shape[1:]
        )

    if method == "lower" or (method == "linear" and not q.is_floating_point()):
        result = take(lower)
    elif method == "higher":
        result = take(position.ceil().to(torch.int64))
    elif method == "nearest":
        result = take(position.round().to(torch.int64))
    elif method in ("linear", "midpoint"):
        # NumPy subtracts adjacent samples in the input dtype, then promotes
        # interpolation. Upcasting only the final result loses precision;
        # upcasting before subtraction changes NumPy's rounding/overflow.
        weight_dtype = q.dtype if numpy2 and q.is_floating_point() else torch.float64
        weight = position.to(weight_dtype) - lower.to(weight_dtype)
        if method == "midpoint":
            weight = (weight != 0).to(weight_dtype) * 0.5
        weight = weight.reshape(q.shape + (1,) * (values.ndim - 1))
        left, right = take(lower), take(upper)
        difference = right - left
        dtype = torch.promote_types(a.dtype, weight.dtype)
        left, right, difference = (x.to(dtype) for x in (left, right, difference))
        result = torch.where(
            weight >= 0.5,
            right - difference * (1 - weight),
            left + difference * weight,
        )
    else:
        raise NotImplementedError(f"unsupported quantile method: {method}")
    if a.is_floating_point():
        last = values[-1].to(result.dtype)
        result = torch.where(torch.isnan(last), last, result)
    if keepdims:
        kept_shape = (
            (1,) * len(shape)
            if all_axes
            else tuple(1 if i == axis else size for i, size in enumerate(shape))
        )
        result = result.reshape(q.shape + kept_shape)
    return result


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
    np.quantile: _quantile,
    np.sort: _sort,
    np.argsort: _argsort,
}
