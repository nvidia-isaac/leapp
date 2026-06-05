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
| `make_warp_model_repo.py` | Captures a coarse 2-kernel `.wrp` and lays out the Triton model repo (`warp_node/` + `ensemble/`). |
| `harness_standalone.py` | Validates `model.py.execute()` on a GPU tensor via a faithful `pb_utils` DLPack mock — **always runs** (no server). |
| `run_live_triton.py` | Serves the repo in a **real** (dockerless) tritonserver and infers the ensemble over HTTP. |
| `harness_pytriton.py` | Alt: bind via PyTriton's high-level API. |
| `create_triton_model_repo.warp.patch` | The actual `isaac_ros_deploy` generator change (Runtime A): adds `backend == "warp"` dispatch + `_create_warp_model_dir` + `_generate_warp_model_config`. Apply in `isaac_ros_deploy_converters/`. |

## Run (validated on RTX 6000 Ada, warp 1.14.0, Triton 2.x python backend)
```bash
uv venv --python 3.12 /tmp/leapp-warp/venv
uv pip install --python /tmp/leapp-warp/venv nvidia-pytriton warp-lang==1.14.0 torch numpy pyyaml onnx
/tmp/leapp-warp/venv/bin/python harness_standalone.py   # PASS, max_abs_err=0.0
/tmp/leapp-warp/venv/bin/python run_live_triton.py      # PASS — warp in a live Triton ensemble
```

## Results
- **Standalone**: warp `model.py` GPU/DLPack path round-trips, `max_abs_err = 0.0`.
- **Live Triton ensemble**: the `.wrp` runs as a python-backend model in a real tritonserver
  ensemble served over HTTP, `max_abs_err = 0.0`.
- **Patched generator**: `create_triton_model_repo.py` (with the patch) emits a `warp_node` +
  ensemble that serves correctly live, `max_abs_err = 0.0`.

## Two runtimes
- **Runtime A (ros2 nodes, Triton ensemble)** — covered here: the patch is the whole integration;
  the C++ Triton runner is unchanged (it just calls the ensemble).
- **Runtime B (ros2_control controller, single-model, NOT an ensemble)** — add a
  `WarpRunner : InferenceRunner` behind `InferenceRunner::create`, selected by
  `parameters.backend: warp`; load the `.wrp` in `warmup()` (off-RT) and replay the captured CUDA
  graph (`wp_apic_get_cuda_graph_exec` + `cudaGraphLaunch`) allocation-free in `run()`. (Not
  prototyped here — C++/CUDA + the Triton 2.60 ABI; design in `/tmp/leapp-warp/03-triton-runtime-map.md`.)

## Production notes
- `.wrp` is **version-gated/experimental** — pin warp, treat `.wrp` as a re-captured build artifact.
- The python backend's env must have `torch`+`warp` (in the Triton container it does; the dockerless
  prototype injects them via `WARP_TRITON_SITE_PACKAGES`, a no-op in real deploy).
- Package `model.py` + `warp_apic_runtime.py` as data with `isaac_ros_deploy_converters`.
- For hard-RT, prefer the Runtime B `WarpRunner` (no GIL/IPC) or a custom C++ Triton backend; the
  python-backend `execute()` logic ports near-verbatim.
