#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pytest
torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
if not torch.cuda.is_available():
    pytest.skip("requires CUDA", allow_module_level=True)
import leapp
from leapp import annotate, InferenceManager

DEV = "cuda:0"
N = 6


@wp.kernel
def _norm_vec3(x: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    out[i] = wp.normalize(x[i])


def test_two_sequential_mixed_regions(tmp_path):
    """Two independent mixed regions in a single leapp.start()/stop() — verifies the segmenter
    is correctly reset between regions (the _ACTIVE segmenter cleared on each region's
    output_tensors call)."""
    wp.init()
    leapp.start("pg2", save_path=str(tmp_path))

    # Region A: obsA — torch scale -> warp normalize -> torch reshape
    gA = torch.randn(N, 3, device=DEV, dtype=torch.float32)
    gtA = annotate.input_tensors("obsA", {"g": gA})
    scaledA = gtA * 2.0
    aA = wp.from_torch(scaledA.contiguous().reshape(-1, 3), dtype=wp.vec3f)
    outA = wp.zeros(N, dtype=wp.vec3f, device=DEV)
    wp.launch(_norm_vec3, dim=N, inputs=[aA], outputs=[outA], device=DEV)
    dA = wp.to_torch(outA).reshape(N, 3)
    annotate.output_tensors("obsA", {"pg": dA}, export_with="onnx-torchscript")

    # Region B: obsB — independent torch scale -> warp normalize -> torch reshape
    gB = torch.randn(N, 3, device=DEV, dtype=torch.float32)
    gtB = annotate.input_tensors("obsB", {"g": gB})
    scaledB = gtB * 3.0
    aB = wp.from_torch(scaledB.contiguous().reshape(-1, 3), dtype=wp.vec3f)
    outB = wp.zeros(N, dtype=wp.vec3f, device=DEV)
    wp.launch(_norm_vec3, dim=N, inputs=[aB], outputs=[outB], device=DEV)
    dB = wp.to_torch(outB).reshape(N, 3)
    annotate.output_tensors("obsB", {"pg": dB}, export_with="onnx-torchscript")

    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)

    yaml_path = str(tmp_path / "pg2" / "pg2.yaml")
    im = InferenceManager(yaml_path)

    gA_in = torch.randn(N, 3, device=DEV, dtype=torch.float32)
    gB_in = torch.randn(N, 3, device=DEV, dtype=torch.float32)
    in_keyA = [k for k in im.inputs if "obsA" in k and k.endswith("/g")][0]
    in_keyB = [k for k in im.inputs if "obsB" in k and k.endswith("/g")][0]
    res = im({in_keyA: gA_in, in_keyB: gB_in})

    out_keyA = [k for k in res if "obsA" in k and k.endswith("/pg")][0]
    out_keyB = [k for k in res if "obsB" in k and k.endswith("/pg")][0]

    refA = torch.nn.functional.normalize(gA_in * 2.0, dim=1)
    refB = torch.nn.functional.normalize(gB_in * 3.0, dim=1)
    assert torch.allclose(res[out_keyA], refA, rtol=1e-4, atol=1e-5), \
        f"obsA output mismatch: max_err={( res[out_keyA] - refA).abs().max()}"
    assert torch.allclose(res[out_keyB], refB, rtol=1e-4, atol=1e-5), \
        f"obsB output mismatch: max_err={(res[out_keyB] - refB).abs().max()}"


def test_mixed_autosplit_roundtrips(tmp_path):
    wp.init()
    leapp.start("pg", save_path=str(tmp_path))
    g = torch.randn(N, 3, device=DEV, dtype=torch.float32)
    gt = annotate.input_tensors("obs", {"g": g})
    scaled = gt * 2.0                                   # torch segment (obs.01_torch)
    a = wp.from_torch(scaled.contiguous().reshape(-1, 3), dtype=wp.vec3f)   # bridge ->
    out = wp.zeros(N, dtype=wp.vec3f, device=DEV)
    wp.launch(_norm_vec3, dim=N, inputs=[a], outputs=[out], device=DEV)     # warp segment (obs.02_warp)
    d = wp.to_torch(out).reshape(N, 3)                  # bridge back -> torch (obs.03_torch)
    annotate.output_tensors("obs", {"pg": d}, export_with="onnx-torchscript")
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)

    yaml_path = str(tmp_path / "pg" / "pg.yaml")
    im = InferenceManager(yaml_path)
    g_in = torch.randn(N, 3, device=DEV, dtype=torch.float32)
    in_key = [k for k in im.inputs if k.endswith("/g")][0]
    res = im({in_key: g_in})
    out_key = [k for k in res if k.endswith("/pg")][0]
    ref = torch.nn.functional.normalize(g_in * 2.0, dim=1)
    assert torch.allclose(res[out_key], ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# T1: multiple wp.launch calls in ONE warp segment
# ---------------------------------------------------------------------------

@wp.kernel
def _affine_k(x: wp.array(dtype=wp.float32), s: wp.float32, b: wp.float32,
               out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * s + b


@wp.kernel
def _relu_k(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = wp.max(x[i], wp.float32(0.0))


def test_multi_launch_warp_segment(tmp_path):
    """Two wp.launch calls in one warp segment — record_launch called twice, all replayed in
    a single APIC capture producing one .wrp node."""
    wp.init()
    leapp.start("t1", save_path=str(tmp_path))

    # Input spans negative → positive so relu actually clamps some values
    g = torch.linspace(-1.0, 1.0, N, device=DEV, dtype=torch.float32)
    gt = annotate.input_tensors("obs", {"g": g})

    # First torch segment: trivial scale so the warp segment is non-trivial
    g0 = gt * 1.0

    # Warp segment: affine then relu via two separate wp.launch calls
    a = wp.from_torch(g0.contiguous(), dtype=wp.float32)
    tmp = wp.zeros(N, dtype=wp.float32, device=DEV)
    out = wp.zeros(N, dtype=wp.float32, device=DEV)
    wp.launch(_affine_k, dim=N, inputs=[a, 2.0, -0.5], outputs=[tmp], device=DEV)
    wp.launch(_relu_k, dim=N, inputs=[tmp], outputs=[out], device=DEV)
    d = wp.to_torch(out)

    annotate.output_tensors("obs", {"y": d}, export_with="onnx-torchscript")
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)

    yaml_path = str(tmp_path / "t1" / "t1.yaml")
    im = InferenceManager(yaml_path)

    g_in = torch.linspace(-1.0, 1.0, N, device=DEV, dtype=torch.float32)
    in_key = [k for k in im.inputs if k.endswith("/g")][0]
    res = im({in_key: g_in})
    out_key = [k for k in res if k.endswith("/y")][0]

    ref = torch.relu(g_in * 2.0 - 0.5)
    assert torch.allclose(res[out_key], ref, rtol=1e-4, atol=1e-5), \
        f"T1 mismatch: max_err={(res[out_key] - ref).abs().max()}"


# ---------------------------------------------------------------------------
# T2: torch → warp → torch → warp (two warp segments in one region)
# ---------------------------------------------------------------------------

def test_two_warp_segments_in_one_region(tmp_path):
    """torch→warp→torch→warp: two distinct warp segments separated by a torch op.
    Exercises the pending-state clear between segments (fix F8) and confirms
    multiple WarpRegionNodes per region compile end-to-end."""
    wp.init()
    leapp.start("t2", save_path=str(tmp_path))

    g = torch.randn(N, 3, device=DEV, dtype=torch.float32)
    gt = annotate.input_tensors("obs", {"g": g})

    # Warp segment 1: scale by 2 then normalize
    s1 = gt * 2.0
    a1 = wp.from_torch(s1.contiguous().reshape(-1, 3), dtype=wp.vec3f)
    o1 = wp.zeros(N, dtype=wp.vec3f, device=DEV)
    wp.launch(_norm_vec3, dim=N, inputs=[a1], outputs=[o1], device=DEV)
    d1 = wp.to_torch(o1).reshape(N, 3)

    # Torch op between the two warp segments
    s2 = d1 + 1.0

    # Warp segment 2: normalize again
    a2 = wp.from_torch(s2.contiguous().reshape(-1, 3), dtype=wp.vec3f)
    o2 = wp.zeros(N, dtype=wp.vec3f, device=DEV)
    wp.launch(_norm_vec3, dim=N, inputs=[a2], outputs=[o2], device=DEV)
    d2 = wp.to_torch(o2).reshape(N, 3)

    annotate.output_tensors("obs", {"pg": d2}, export_with="onnx-torchscript")
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)

    yaml_path = str(tmp_path / "t2" / "t2.yaml")
    im = InferenceManager(yaml_path)

    # Capture emitted node names for diagnostics (im.inputs is a list of key strings)
    node_names = list(im.inputs)

    g_in = torch.randn(N, 3, device=DEV, dtype=torch.float32)
    in_key = [k for k in im.inputs if k.endswith("/g")][0]
    res = im({in_key: g_in})
    out_key = [k for k in res if k.endswith("/pg")][0]

    import torch.nn.functional as F
    ref = F.normalize(F.normalize(g_in * 2.0, dim=1) + 1.0, dim=1)
    assert torch.allclose(res[out_key], ref, rtol=1e-4, atol=1e-5), \
        f"T2 mismatch: max_err={(res[out_key] - ref).abs().max()}; node_names={node_names}"


# ---------------------------------------------------------------------------
# T3: wp.from_torch without an explicit dtype (inferred-dtype branch)
# ---------------------------------------------------------------------------

def test_from_torch_no_explicit_dtype(tmp_path):
    """wp.from_torch called WITHOUT dtype=: exercises torch_dtype_to_warp_str inference
    (the 'else' branch in patched_from_torch)."""
    wp.init()
    leapp.start("t3", save_path=str(tmp_path))

    g = torch.randn(N, device=DEV, dtype=torch.float32)
    gt = annotate.input_tensors("obs", {"g": g})

    g0 = gt * 3.0
    # No dtype= argument: warp infers float32 from the tensor; leapp uses torch_dtype_to_warp_str
    a = wp.from_torch(g0.contiguous())
    out = wp.zeros(N, dtype=wp.float32, device=DEV)
    wp.launch(_relu_k, dim=N, inputs=[a], outputs=[out], device=DEV)
    d = wp.to_torch(out)

    annotate.output_tensors("obs", {"y": d}, export_with="onnx-torchscript")
    leapp.stop()
    leapp.compile_graph(visualize=False, validate=True)

    yaml_path = str(tmp_path / "t3" / "t3.yaml")
    im = InferenceManager(yaml_path)

    g_in = torch.randn(N, device=DEV, dtype=torch.float32)
    in_key = [k for k in im.inputs if k.endswith("/g")][0]
    res = im({in_key: g_in})
    out_key = [k for k in res if k.endswith("/y")][0]

    ref = torch.relu(g_in * 3.0)
    assert torch.allclose(res[out_key], ref, rtol=1e-4, atol=1e-5), \
        f"T3 mismatch: max_err={(res[out_key] - ref).abs().max()}"
