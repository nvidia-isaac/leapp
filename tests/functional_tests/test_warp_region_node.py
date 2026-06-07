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


from leapp.leapp_graph.warp_region_node import WarpRegionNode  # noqa: E402
import json  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402


def test_region_node_compiles_and_validates(tmp_path):
    wp.init()
    n = 4
    x = wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32, device=DEV)
    out = wp.zeros(n, dtype=wp.float32, device=DEV)
    records = [{"args": (_scale,), "kwargs": {"dim": n, "inputs": [x, 2.0],
                                              "outputs": [out], "device": DEV}}]
    node = WarpRegionNode("seg.02_warp", device=DEV)
    src = torch.ones(n, dtype=torch.float32, device=DEV)
    src.leapp_tag = "seg.01_torch/h/"
    node.set_save_dir(str(tmp_path))
    node.set_io(records, inputs={"in0": (x, src)}, outputs={"out0": out})
    node.node_index = 1
    node.compile_model()
    node.save_model(str(tmp_path))
    assert node.inputs[0].tag == "seg.01_torch/h/"
    assert node.model_path is not None and node.sha256sum is not None
    got = node.compiled_model(torch.tensor([1., 2., 3., 4.], device=DEV))
    assert torch.allclose(got, torch.tensor([2., 4., 6., 8.], device=DEV))


@wp.kernel
def _mul2(x: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = x[i] * wp.float32(2.0)


def test_apic_name_fallback_for_old_artifacts(tmp_path):
    """Backward-compat: WarpExportBackend loads old .wrp files whose .warpmeta.json has no
    apic_input_names / apic_output_names keys.

    The current save_warp_node() writes APIC bindings as "_in_<port>" / "_out_<port>" and
    records them in the meta.  Old artifacts used PLAIN port names as APIC bindings and wrote
    no apic_* keys.  _WarpGraphCallable falls back via:
        self.apic_input_names = meta.get("apic_input_names", self.input_names)
        self.apic_output_names = meta.get("apic_output_names", self.output_names)
    so apic_* == plain port names, which matches the plain-name bindings in the old .wrp.

    This test simulates the old artifact by:
      1. Capturing the graph and saving the .wrp directly with plain port names ("x" / "y")
         as APIC bindings (bypassing save_warp_node).
      2. Writing a .warpmeta.json WITHOUT apic_input_names / apic_output_names.
      3. Loading via WarpExportBackend and asserting numeric correctness — this exercises the
         fallback branch in _WarpGraphCallable.__init__.
    """
    wp.init()
    n = 6
    x_arr = wp.zeros(n, dtype=wp.float32, device=DEV)
    y_arr = wp.zeros(n, dtype=wp.float32, device=DEV)
    wp.load_module(device=DEV)

    # Capture a simple multiply-by-2 kernel.
    with wp.ScopedCapture(device=DEV, force_module_load=True, apic=True) as cap:
        wp.launch(_mul2, dim=n, inputs=[x_arr], outputs=[y_arr], device=DEV)

    # --- Step 1: save the .wrp using PLAIN port names as APIC bindings (old behaviour) ---
    base = str(tmp_path / "oldnode")
    wp.capture_save(cap.graph, base, inputs={"x": x_arr}, outputs={"y": y_arr})
    wrp_path = base + ".wrp"

    def _sha256(p):
        with open(p, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _md5(p):
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    modules_dir = base + "_modules"
    modules_sha256 = {}
    if os.path.isdir(modules_dir):
        for fn in sorted(os.listdir(modules_dir)):
            fp = os.path.join(modules_dir, fn)
            if os.path.isfile(fp):
                modules_sha256[fn] = _sha256(fp)

    # --- Step 2: hand-write a warpmeta.json WITHOUT apic_input_names / apic_output_names ---
    # This mimics an old artifact produced before the prefix scheme was introduced.
    meta = {
        "inputs": ["x"],           # plain LEAPP port names
        "outputs": ["y"],
        # NO apic_input_names / apic_output_names — the fallback must supply them.
        "input_dtypes": ["float32"],
        "output_dtypes": ["float32"],
        "input_warp_dtypes": ["float32"],
        "output_warp_dtypes": ["float32"],
        "output_shapes": [[n]],
        "device_type": "cuda",
        "modules_sha256": modules_sha256,
    }
    meta_path = base + ".warpmeta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # --- Step 3: load via WarpExportBackend and run ---
    # The fallback in _WarpGraphCallable maps apic_input_names -> ["x"], apic_output_names -> ["y"],
    # which matches the plain-name APIC bindings written in step 1 — so get_param / set_param work.
    class _Stub:
        name = "oldnode"

    params = {
        "model_path": wrp_path,
        "md5sum": _md5(wrp_path),
        "sha256sum": _sha256(wrp_path),
        "backend": "warp",
    }
    backend = WarpExportBackend(_Stub(), params)
    backend.load(wrp_path, params["sha256sum"])

    x_in = torch.tensor([1., 2., 3., 4., 5., 6.], dtype=torch.float32, device=DEV)
    result = backend.compiled_model(x_in)
    expected = x_in * 2.0
    assert torch.allclose(result, expected), f"fallback gave {result}, expected {expected}"
