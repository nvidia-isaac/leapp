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
