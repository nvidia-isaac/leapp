#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
torch = pytest.importorskip("torch")
wp = pytest.importorskip("warp")
if not torch.cuda.is_available():
    pytest.skip("warp .wrp requires CUDA", allow_module_level=True)

from leapp.backends.warp_export_backend import save_warp_node, WarpExportBackend

DEV = "cuda:0"


@wp.kernel
def _normalize3(x: wp.array(dtype=wp.vec3f), out: wp.array(dtype=wp.vec3f)):
    i = wp.tid()
    out[i] = wp.normalize(x[i])


def test_vec3f_input_roundtrips(tmp_path):
    wp.init()
    n = 5
    x = wp.zeros(n, dtype=wp.vec3f, device=DEV)
    out = wp.zeros(n, dtype=wp.vec3f, device=DEV)
    wp.load_module(device=DEV)
    with wp.ScopedCapture(device=DEV, force_module_load=True, apic=True) as cap:
        wp.launch(_normalize3, dim=n, inputs=[x], outputs=[out], device=DEV)

    node = save_warp_node(cap.graph, str(tmp_path), "normy",
                          inputs={"v": x}, outputs={"o": out})
    assert node["inputs"][0]["dtype"] == "float32"
    assert node["inputs"][0]["shape"] == [n, 3]
    assert node["inputs"][0]["warp_dtype"] == "vec3f"

    backend = WarpExportBackend(_NodeStub(), node["parameters"])
    backend.load(str(tmp_path / "normy.wrp"), node["parameters"]["sha256sum"])

    t = torch.randn(n, 3, dtype=torch.float32, device=DEV)
    got = backend.compiled_model(t)
    ref = torch.nn.functional.normalize(t, dim=1)
    assert got.shape == (n, 3)
    assert torch.allclose(got, ref, rtol=1e-4, atol=1e-5)


class _NodeStub:
    name = "normy"


from leapp.backends.warp_capture_core import replay_into_apic_capture, resolve_device  # noqa: E402


@wp.kernel
def _scale(x: wp.array(dtype=wp.float32), s: wp.float32, out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * s


def test_capture_core_replays(tmp_path):
    wp.init()
    n = 4
    x = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32, device=DEV)
    out = wp.zeros(n, dtype=wp.float32, device=DEV)
    records = [{"args": (_scale,), "kwargs": {"dim": n, "inputs": [x, 2.0],
                                              "outputs": [out], "device": DEV}}]
    assert resolve_device(wp, records, None) == DEV
    graph = replay_into_apic_capture(wp, records, device=DEV)
    node = save_warp_node(graph, str(tmp_path), "scaler", inputs={"x": x}, outputs={"o": out})
    assert node["parameters"]["backend"] == "warp"
    wp.capture_launch(graph)
    wp.synchronize_device(DEV)
    assert torch.allclose(wp.to_torch(out), torch.tensor([2.0, 4.0, 6.0, 8.0], device=DEV))
