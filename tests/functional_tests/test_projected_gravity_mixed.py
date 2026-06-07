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
