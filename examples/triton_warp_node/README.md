<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Warp (APIC) as a Triton ensemble step

Companion to the export-side `warp` node-kind. Shows how a LEAPP **Warp peer node** runs in the
**deployment runtime**, which in `isaac_ros_deploy` (Runtime A — ros2 nodes) is a **Triton
ensemble**.

> **LEAPP owns the runtime code.** The LEAPP-graph → Triton-repo generator and the Warp
> python-backend node template live in the package at **`leapp_runtimes/triton/`** (tested by
> `tests/functional_tests/test_triton_model_repo.py`). Downstream deployers (`isaac_ros_deploy`)
> import it instead of vendoring a copy:
> ```python
> from leapp_runtimes.triton.create_triton_model_repo import create_triton_model_repo
> create_triton_model_repo(Path("graph.yaml"), Path("model_repo"))
> ```
> This folder holds the **test helper** that exercises that owned code.

**Design (Approach A):** a warp node = one APIC `.wrp` (+ `<name>_modules/` + `.warpmeta.json`)
= one **Triton python-backend model** = one **ensemble step**. The ensemble step-builder is already
node-kind-agnostic, so the warp node drops in next to onnx/jit steps with no downstream change.
warp↔onnx tensors stay GPU-resident across the ensemble via the python backend's DLPack path
(`KIND_GPU` + `FORCE_CPU_ONLY_INPUT_TENSORS:"no"`).

## Files

**Owned by LEAPP** (`leapp_runtimes/triton/`, packaged + importable):
| File | Role |
|---|---|
| `leapp_runtimes/triton/create_triton_model_repo.py` | The generator: LEAPP YAML + per-node artifacts → Triton model repo + ensemble. ONNX/JIT nodes → standard backends; `backend: warp` → a python-backend node. |
| `leapp_runtimes/triton/warp_node/warp_apic_runtime.py` | Deploy-side core (torch+warp only): load `.wrp` + verify modules + replay on torch tensors. Fails loudly on dtype/size/checksum/device divergence. |
| `leapp_runtimes/triton/warp_node/model.py` | Triton **python-backend** model TEMPLATE (copied into each warp model dir): DLPack → `wp.from_torch` → `capture_launch` → DLPack. |

**Test helper** (this folder):
| File | Role |
|---|---|
| `make_warp_model_repo.py` | Captures a coarse 2-kernel `.wrp` and builds a Triton repo **via the owned generator** (`warp_node/` + `ensemble/`). Used as a test fixture by `test_triton_model_repo.py` and `test_triton_warp_node.py`. |

**Tests** (`tests/functional_tests/`):
| File | Role |
|---|---|
| `test_triton_model_repo.py` | Generator: ONNX-only ensemble, warp-node generation, `onnx→warp→onnx`, and the generation-time dtype guard. |
| `test_triton_warp_node.py` | Warp node runtime: Tier A standalone (`pb_utils` DLPack mock, no server) + loud-failure guards; Tier B live single-node Triton ensemble (auto-skips without pytriton). |

## Run
Canonical, portable validation is the pytest (needs warp + CUDA; the live tier auto-skips without `nvidia-pytriton`):
```bash
PYTHONPATH=$PWD python3.12 -m pytest tests/functional_tests/test_triton_model_repo.py tests/functional_tests/test_triton_warp_node.py -q
```

## Hardening applied (after adversarial review)
- **H1 (CUDA stream ordering):** `model.py`/`warp_apic_runtime.py` and the export `_WarpGraphCallable`
  bind + `capture_launch(stream=…)` + read back on **torch's current stream** (via `wp.ScopedStream`),
  then sync only that stream — no producer→graph cross-stream race, no whole-device sync.
- **H3 (dtype mismatch):** the generator validates warp-node I/O dtypes at **generation time** (clear error,
  not an opaque serve-time crash); runtime dtype map expanded (fp16/int/uint/bool) and unsupported dtypes raise.
- **I1 (batch guard):** `run_torch` rejects a wrong-element-count (batched) input loudly.
- **I2 (device id):** `model.py` binds to the instance's actual GPU (`model_instance_device_id`), not bare `cuda:0`.

## Production notes
- `.wrp` is **version-gated/experimental** — pin warp, treat `.wrp` as a re-captured build artifact; add a
  `warp_version`/`wrp_format_version` gate at load (not yet enforced).
- The python backend's env must have `torch`+`warp` (in the Triton container it does; the dev/test env
  injects them via `WARP_TRITON_SITE_PACKAGES`, a no-op in real deploy).
- **Packaging:** the generator copies `model.py` + `warp_apic_runtime.py` from
  `leapp_runtimes/triton/warp_node/`; these are shipped with LEAPP via `pyproject.toml`
  (`package-data` for `leapp_runtimes.triton.warp_node` + `zip-safe=false`), so a `pip install leapp`
  has them.
- **Still not proven:** a *live onnxruntime-backend* onnx→warp→onnx run (needs the full Triton container);
  batching; the Runtime B `WarpRunner`; Jetson/aarch64.
- For hard-RT, prefer the Runtime B `WarpRunner` (no GIL/IPC) or a custom C++ Triton backend; the
  python-backend `execute()` logic ports near-verbatim.
