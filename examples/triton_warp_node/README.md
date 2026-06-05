<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Warp (APIC) as a Triton ensemble step — runtime prototype

Companion to the export-side `warp` node-kind (branch `lgulich/warp-node-prototype`). This shows
how a LEAPP **Warp peer node** runs in the **deployment runtime**, which in `isaac_ros_deploy`
(Runtime A — ros2 nodes) is a **Triton ensemble** built by `create_triton_model_repo.py`.

**Design (Approach A):** a warp node = one APIC `.wrp` (+ `<name>_modules/` + `.warpmeta.json`)
= one **Triton python-backend model** = one **ensemble step**. The ensemble step-builder is already
node-kind-agnostic, so the warp node drops in next to onnx/jit steps with no downstream change.
warp↔onnx tensors stay GPU-resident across the ensemble via the python backend's DLPack path
(`KIND_GPU` + `FORCE_CPU_ONLY_INPUT_TENSORS:"no"`).

## Files
| File | Role |
|---|---|
| `warp_apic_runtime.py` | Deploy-side core (no leapp dep): load `.wrp` + verify modules + replay on torch tensors. Fails loudly on dtype/size/checksum/device divergence. |
| `model.py` | Triton **python-backend** model: `wp.capture_load` in `initialize`; DLPack→`wp.from_torch`→`set_param`→`capture_launch`→DLPack in `execute`. |
| `affine_passthrough_model.py` | Trivial GPU python-backend model (`out = in*scale+bias`, DLPack) — stand-in for the ONNX neighbor steps in the dockerless multi-step demo. |
| `make_warp_model_repo.py` | Captures a coarse 2-kernel `.wrp` and lays out the Triton model repo (`warp_node/` + `ensemble/`). |
| `_triton_serve.py` | Dockerless helper: discovers the PyTriton-bundled `tritonserver` (no hardcoded paths) and serves a model repo for the demos. |
| `run_live_triton.py` | Demo: serve a single-node warp ensemble on a real (dockerless) tritonserver and infer over HTTP. Skips if pytriton absent. |
| `multistep_ensemble.py` | Demo: Part 1 runs the patched generator on a real `onnx→warp→onnx` graph and asserts the emitted ensemble (optional — needs the isaac_3 generator); Part 2 serves a live `python→warp→python` ensemble and asserts GPU-resident internal step→step handoffs. |
| `create_triton_model_repo.warp.patch` | The actual `isaac_ros_deploy` generator change (Runtime A): adds `backend == "warp"` dispatch + `_create_warp_model_dir` + `_generate_warp_model_config`. Apply in `isaac_ros_deploy_converters/`. |
| `../../tests/functional_tests/test_triton_warp_node.py` | **Canonical validation** (pytest): Tier A standalone `model.py` path via a `pb_utils` DLPack mock (no server) + loud-failure guards; Tier B live single-node Triton ensemble (auto-skips without pytriton). |

## Run
Canonical, portable validation is the pytest (needs warp + CUDA; the live tier auto-skips without `nvidia-pytriton`):
```bash
PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_triton_warp_node.py -q
```
Interactive demos (the live parts need an env with `nvidia-pytriton`):
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv nvidia-pytriton warp-lang torch numpy pyyaml onnx
.venv/bin/python examples/triton_warp_node/run_live_triton.py
.venv/bin/python examples/triton_warp_node/multistep_ensemble.py   # Part 1 also needs LEAPP_WARP_GENERATOR_DIR -> the patched generator
```

## Results
- **Standalone**: warp `model.py` GPU/DLPack path round-trips, `max_abs_err = 0.0`.
- **Live Triton ensemble**: the `.wrp` runs as a python-backend model in a real tritonserver
  ensemble served over HTTP, `max_abs_err = 0.0`.
- **Patched generator**: `create_triton_model_repo.py` (with the patch) emits a `warp_node` +
  ensemble that serves correctly live, `max_abs_err = 0.0`.
- **Multi-step (`multistep_ensemble.py`)**: the patched generator emits a correct `onnx→warp→onnx`
  ensemble (Part 1); and a live 3-step `python→warp→python` ensemble round-trips with
  `max_abs_err = 0.0` while keeping tensors **GPU-resident across both internal step→step edges**
  (`is_cpu=False` at the warp input and the post input) — Part 2. A *live onnxruntime-backend* run
  needs the full Triton container (the dockerless PyTriton bundle ships only the python backend);
  Part 1 validates the generated onnx config, Part 2 validates the live GPU-resident handoff.

## Two runtimes
- **Runtime A (ros2 nodes, Triton ensemble)** — covered here: the patch is the whole integration;
  the C++ Triton runner is unchanged (it just calls the ensemble).
- **Runtime B (ros2_control controller, single-model, NOT an ensemble)** — add a
  `WarpRunner : InferenceRunner` behind `InferenceRunner::create`, selected by
  `parameters.backend: warp`; load the `.wrp` in `warmup()` (off-RT) and replay the captured CUDA
  graph (`wp_apic_get_cuda_graph_exec` + `cudaGraphLaunch`) allocation-free in `run()`. (Not
  prototyped here — needs C++/CUDA against the Triton 2.60 ABI.)

## Hardening applied (after adversarial review)
- **H1 (CUDA stream ordering):** `model.py`/`warp_apic_runtime.py` and the export `_WarpGraphCallable`
  bind + `capture_launch(stream=…)` + read back on **torch's current stream** (via `wp.ScopedStream`),
  then sync only that stream — no producer→graph cross-stream race, no whole-device sync. *Still needs an
  N-instance concurrent-HTTP stress test before any concurrency-safe claim.*
- **H3 (dtype mismatch):** the generator validates warp-node I/O dtypes at **generation time** (clear error,
  not an opaque serve-time crash); runtime dtype map expanded (fp16/int/uint/bool) and unsupported dtypes raise.
- **I1 (batch guard):** `run_torch` rejects a wrong-element-count (batched) input loudly.
- **I2 (device id):** `model.py` binds to the instance's actual GPU (`model_instance_device_id`), not bare `cuda:0`.

## Production notes
- `.wrp` is **version-gated/experimental** — pin warp, treat `.wrp` as a re-captured build artifact; add a
  `warp_version`/`wrp_format_version` gate at load (not yet enforced).
- The python backend's env must have `torch`+`warp` (in the Triton container it does; the dockerless
  prototype injects them via `WARP_TRITON_SITE_PACKAGES`, a no-op in real deploy).
- **H2 (packaging — hard break if skipped):** the generator copies `model.py` + `warp_apic_runtime.py` from
  `_warp_templates/` next to the module. In the real `isaac_ros_deploy_converters` Bazel/wheel build these
  templates **must** be added to the target `srcs`/`data` (`glob(["_warp_templates/**"])`) + `package_data`/
  `MANIFEST.in`, and resolved via runfiles — otherwise every warp generation raises `FileNotFoundError`.
- **Now proven (`multistep_ensemble.py`):** the generator emits a correct onnx→warp→onnx ensemble, and
  a live 3-step ensemble keeps tensors GPU-resident across both internal step→step edges.
- **Still not proven:** a *live onnxruntime-backend* onnx→warp→onnx run (needs the full Triton container);
  batching; the Runtime B `WarpRunner`; Jetson/aarch64.
- For hard-RT, prefer the Runtime B `WarpRunner` (no GIL/IPC) or a custom C++ Triton backend; the
  python-backend `execute()` logic ports near-verbatim.
