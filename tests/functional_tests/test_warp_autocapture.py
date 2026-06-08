#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Non-invasive Warp capture: `with leapp.warp_node(...)` records plain wp.launch calls.

The user's Warp code (kernels, launches, arrays) is unchanged; only the surrounding context manager
is added. We assert the captured node round-trips: reload the emitted .wrp and reproduce eager.
GPU/warp-guarded.
"""
import os
import sys

import pytest

wp = pytest.importorskip("warp", reason="warp-lang not installed")
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="warp capture requires a CUDA GPU")

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

import leapp  # noqa: E402
from leapp.backends.warp_export_backend import WarpExportBackend  # noqa: E402


class _StubNode:
    name = "autocapture"


def _run_wrp(wn, **named_inputs):
    """Load the captured .wrp via the export backend and run it (single-output helper)."""
    be = WarpExportBackend(_StubNode(), wn.node["parameters"])
    be.load(wn.wrp_path, wn.node["parameters"]["sha256sum"])
    # inputs are positional in input_names order
    ordered = [named_inputs[i["name"]] for i in wn.node["inputs"]]
    out = be.compiled_model(*ordered)
    return out  # single output -> tensor

N = 8
SCALE = 2.0
BIAS = 1.0


@wp.kernel
def _affine(x: wp.array(dtype=wp.float32), s: wp.float32, b: wp.float32, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * s + b


@wp.kernel
def _relu(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.max(x[i], wp.float32(0.0))


def test_warp_node_autocapture_noninvasive(tmp_path):
    # ===== plain warp code; only the `with leapp.warp_node(...)` is added =====
    with leapp.warp_node("auto", save_path=str(tmp_path)) as wn:
        x = wp.zeros(N, dtype=wp.float32, device="cuda:0")
        y = wp.zeros(N, dtype=wp.float32, device="cuda:0")
        wp.launch(_affine, dim=N, inputs=[x, SCALE, BIAS], outputs=[y], device="cuda:0")
        wp.launch(_relu, dim=N, inputs=[y], outputs=[y], device="cuda:0")
    # =========================================================================

    # I/O auto-detected from buffer read/write order: x is read-first (input), y is write-last (output).
    assert wn.node["parameters"]["backend"] == "warp"
    assert [i["name"] for i in wn.node["inputs"]] == ["in0"]
    assert [o["name"] for o in wn.node["outputs"]] == ["out0"]
    assert os.path.exists(wn.wrp_path)

    # The emitted .wrp round-trips: reload and reproduce eager results.
    xt = torch.linspace(-1.0, 2.0, N, device="cuda", dtype=torch.float32)
    out = _run_wrp(wn, in0=xt)
    ref = torch.relu(xt * SCALE + BIAS)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5)
    assert int((out > 0).sum()) > 0 and int((out == 0).sum()) > 0  # non-trivial


def test_warp_node_chain_detects_single_boundary_io(tmp_path):
    """A 3-launch chain a->b->c exposes only the true boundary I/O (one in, one out)."""
    with leapp.warp_node("chain", save_path=str(tmp_path)) as wn:
        a = wp.zeros(N, dtype=wp.float32, device="cuda:0")
        b = wp.zeros(N, dtype=wp.float32, device="cuda:0")
        c = wp.zeros(N, dtype=wp.float32, device="cuda:0")
        wp.launch(_affine, dim=N, inputs=[a, 3.0, -1.0], outputs=[b], device="cuda:0")  # a -> b
        wp.launch(_affine, dim=N, inputs=[b, 0.5, 0.0], outputs=[c], device="cuda:0")   # b -> c
        wp.launch(_relu, dim=N, inputs=[c], outputs=[c], device="cuda:0")               # c -> c
    assert [i["name"] for i in wn.node["inputs"]] == ["in0"]    # only `a` (intermediate b not exposed)
    assert [o["name"] for o in wn.node["outputs"]] == ["out0"]  # only `c`
    at = torch.linspace(-1.0, 2.0, N, device="cuda", dtype=torch.float32)
    out = _run_wrp(wn, in0=at)
    ref = torch.relu((at * 3.0 - 1.0) * 0.5)
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-5)


def test_warp_node_empty_block_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no wp.launch"):
        with leapp.warp_node("empty", save_path=str(tmp_path)):
            pass  # no launches recorded
