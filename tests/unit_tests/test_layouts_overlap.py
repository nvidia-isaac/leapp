# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""layouts_overlap() against the bytes two views of one memory cover."""

import itertools

import numpy as np
import pytest
import torch

from leapp.leapp_graph.datatypes.proxy import layouts_overlap

MEMORY = np.zeros(64, dtype=np.uint8)

VIEWS = {
    "float32 strided": MEMORY[:32].view(np.float32)[::2],
    "float32 between elements": MEMORY[7:11].view(np.float32),
    "float16 reversed": MEMORY[:16].view(np.float16)[::-1][:3],
    "float64 column": MEMORY.view(np.float64).reshape(2, 4)[:, 1],
    "uint8 slice": MEMORY[4:8],
    "int32 record field": MEMORY[:25].view([("b", np.int32), ("a", np.uint8)])["b"],
    "int32 at byte 20": MEMORY[20:24].view(np.int32),
    "float32 transposed": MEMORY[16:48].view(np.float32).reshape(2, 4).T,
    "torch float32 strided": torch.from_numpy(MEMORY[32:64].view(np.float32))[1::3],
    "torch int16 2-D": torch.from_numpy(MEMORY[40:56].view(np.int16)).view(2, 4)[
        :, ::2
    ],
}


def covered(view):
    """The offsets into MEMORY of the bytes a view covers."""
    array = view.numpy() if isinstance(view, torch.Tensor) else view
    starts = np.array([array.ctypes.data - MEMORY.ctypes.data])
    for n, stride in zip(array.shape, array.strides):
        starts = (starts[:, None] + np.arange(n) * stride).reshape(-1)
    return {int(start) + byte for start in starts for byte in range(array.itemsize)}


@pytest.mark.parametrize("a,b", itertools.combinations(VIEWS, 2))
def test_layouts_overlap_matches_covered_bytes(a, b):
    expected = bool(covered(VIEWS[a]) & covered(VIEWS[b]))
    assert layouts_overlap(VIEWS[a], VIEWS[b]) == expected


def test_layouts_overlap_edge_cases():
    view = VIEWS["float32 strided"]
    assert layouts_overlap(view, view)
    assert not layouts_overlap(view, MEMORY[:0])
    on_cpu = ("cpu", "float32", (2,), (4,), 0)
    assert not layouts_overlap(on_cpu, ("cuda:0", *on_cpu[1:]))
    column = VIEWS["float64 column"]
    with torch.device("meta"):
        assert layouts_overlap(view, column) == bool(covered(view) & covered(column))
