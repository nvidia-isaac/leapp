# isaac_deploy_trt

Standalone GraphSurgeon toolkit that applies the same TensorRT ONNX rewrites
used in LEAPP, without depending on LEAPP export/tracing.

Given an ONNX file (or a directory of them), it:

1. Detects TensorRT-hostile patterns
2. Rewrites them in-graph
3. **Keeps graph input/output names, dtypes, and shapes** so Triton / DAG
   bindings do not need renaming

This repo **does not download models**. Point the CLI at ONNX you already have
(for GR00T Apple PnP, that is typically a Hugging Face LEAPP export dir).

It does **not** freeze GR00T backbone grid tensors or build `.plan` files.
Use ``merge-pipeline`` (below) to inline a LEAPP DAG for one Isaac
``TensorRTNode``. That is not a substitute for ``trtexec``.

## Rewrites (same as LEAPP)

| Pattern | Change |
|---|---|
| Intermediate `Cast` to UINT8 | Cast to INT32 instead |
| `Resize` with `antialias != 0` | Set `antialias=0` |
| `Reshape` to rank > 8 | Drop extent-1 axes; remap following `Transpose` perms |

Graphs without those patterns are **copied** (plus `.onnx.data` sidecars). Output
must be a different path than the input.

This tree lives under `packages/isaac_deploy_trt`, parallel to
`packages/leapp-visualization`. It keeps its own `pyproject.toml` so it can be
installed on its own; LEAPP still imports it as `isaac_deploy_trt`.

## Install

From the LEAPP repository root:

```bash
pip install -e ./packages/isaac_deploy_trt
# or with tests:
pip install -e "./packages/isaac_deploy_trt[dev]"
```

---

## Commands

### Rewrite one file

```bash
isaac_deploy_trt rewrite IN.onnx -o OUT.onnx
```

Example (the only Apple PnP node that currently needs surgery):

```bash
isaac_deploy_trt rewrite \
  /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx/preprocess_video.onnx \
  -o /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx_surgeon/preprocess_video.onnx
```

### Rewrite a directory into another folder

Copies every `*.onnx` (and matching `.onnx.data`) into `-o`. Only graphs that
hit the three patterns above are edited.

`-o` **must not** be the input directory.

```bash
isaac_deploy_trt rewrite-dir INPUT_DIR -o OUTPUT_DIR
```

GR00T Apple PnP LEAPP pack:

```bash
isaac_deploy_trt rewrite-dir \
  /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx \
  -o /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx_surgeon
```

For `nvidia/GR00T-N1.7-ApplePnP-V1` as exported today:

| File | Surgeon |
|---|---|
| `preprocess_video.onnx` | rewrite (UINT8 Cast, Resize antialias, rank-9 Reshape) |
| `preprocess_state.onnx` | copy |
| `backbone.onnx` | copy |
| `action_head.onnx` | copy |
| `decode_action.onnx` | copy |

YAML (`exported_leapp.yaml`) is **not** copied by `rewrite-dir`. Copy it
yourself if you need a full deploy tree.

### Catalog-check ONNX (validate)

Static check: ops in the graph vs the baked-in TensorRT `operators.md` table.
Does **not** rewrite files. Does **not** load `tensorrt_catalogs/*.json`.

Does **not** catch UINT8 Cast / Resize antialias / rank-9 Reshape. Those only
show up with TensorRT parse (`--build`) or `trtexec`.

```bash
# one file
isaac_deploy_trt check --onnx PATH.onnx

# every .onnx in a folder
isaac_deploy_trt check --onnx-dir DIR

# exit 1 if any file fails the catalog check
isaac_deploy_trt check --onnx-dir DIR --strict

# also TensorRT parse/build (needs TensorRT installed)
isaac_deploy_trt check --onnx-dir DIR --build
```

Examples:

```bash
isaac_deploy_trt check --onnx-dir \
  /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx

isaac_deploy_trt check --onnx-dir \
  /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx_surgeon \
  --strict
```

### Merge a LEAPP DAG into one ONNX file (TensorRTNode, no Triton)

Isaac ROS ``TensorRTNode`` loads a **single** engine. LEAPP export is still
one ONNX per node. This inlines ``pipeline.data_flow`` (fan-out included) into
one graph. ``feedback_flow`` tensors stay as extra I/O (previous tick).
Colliding input/output names rename the input to ``_in_<name>``.

Weights stay external; ``.onnx.data`` files are **symlinked** next to the
merged protobuf (does not load the 6 GB GR00T backbone into RAM).

```bash
isaac_deploy_trt merge-pipeline EXPORTED.yaml -o pipeline.onnx
```

``--rewrite`` applies the GraphSurgeon TensorRT edits to each subgraph before
merge (needed if you have not already run ``rewrite-dir``).

``--model-dir`` defaults to the YAML directory. ``--yaml-out`` overrides the
sibling ``pipeline.yaml`` (TensorRTNode ``input_tensor_names`` /
``output_tensor_names``).

Standalone script (same flags after the YAML path):

```bash
python scripts/merge_pipeline.py EXPORTED.yaml -o pipeline.onnx --rewrite
```

GR00T Apple PnP (rewritten video preprocess + merge):

```bash
isaac_deploy_trt merge-pipeline \
  /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx/exported_leapp.yaml \
  --model-dir /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx \
  -o /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_pipeline/pipeline.onnx \
  --rewrite
```

Then build the engine on the target GPU:

```bash
trtexec \
  --onnx=/mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_pipeline/pipeline.onnx \
  --saveEngine=/mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_pipeline/pipeline.plan \
  --fp16 --memPoolSize=workspace:8192 --skipInference
```

Pass ``engine_file_path`` plus the name lists from ``pipeline.yaml`` into
``TensorRTNode``. This path is Triton-free.

Prefer a **frozen-grid** backbone export if TensorRT classifies
``image_grid_thw`` as a shape tensor (Triton TRT backend also rejects that;
``TensorRTNode`` is different, but freezing still avoids a bad binding).

Do not run this merge on a small machine if you intend to *inline* every
weight into one file; this implementation does not do that. ``trtexec`` on
the full GR00T DAG still needs a large workspace.

### Join operators.md + headers (one JSON)

```bash
isaac_deploy_trt catalog --operators-md operators.md \
  --releases v11.2,v11.1 -o trt_catalog.json
```

### Generate per-release / per-platform catalogs

Writes JSON and Markdown under `-o` (default formats: `json,md`). Use
`--format json` for JSON only. Platform overlays are **not** a live GPU query.

Default platforms are Isaac Debian pockets:

| Key | Isaac dist | Hardware |
|---|---|---|
| `noble` | `noble` | Ubuntu 24.04 amd64 dGPU (same overlay as `x86`) |
| `noble-fastos` | `noble-fastos` | DGX Spark / FastOS ARM (not JetPack) |
| `noble-jetpack` | `noble-jetpack` | Jetson Orin + Thor / JetPack 7.2, TensorRT 10.16 |

SKU overlays `x86`, `orin`, `thor`, `dgx` still work (`orin` is Orin-accurate;
`noble-jetpack` is the union pocket and keeps Thor dtypes). Aliases: `amd64` →
`noble`, `spark` → `noble-fastos`, `jetson`/`jp7` → `noble-jetpack`.

```bash
isaac_deploy_trt generate -o tensorrt_catalogs --print-summary \
  --platforms noble,noble-fastos,noble-jetpack \
  --releases v10.7.0,v10.8.0,v10.9.0,v10.10.0,v10.11,v10.12.0,v10.13.0,v10.13.2,v10.13.3,v10.14,v10.15,v10.16,v11.0,v11.1,v11.2
```

Checked-in databases live in [`tensorrt_catalogs/`](tensorrt_catalogs/index.json).
Look up a GitHub tag in `release_to_database`, then open that directory.

There are no public `v11.3` / `v11.4` tags. Latest OSS 11.x tags are `v11.0`,
`v11.1`, `v11.2`.

---

## Python

```python
from isaac_deploy_trt import rewrite_onnx_file, rewrite_directory

rewrite_onnx_file("preprocess_video.onnx", "preprocess_video_trt.onnx")
rewrite_directory("in_dir", "out_dir")
```

## What this is not

- Not a Hugging Face downloader (source ONNX stays in the export dir)
- Not a substitute for `trtexec` (always build-check on the target GPU)
- Not Isaac ROS / Triton launch (`merge-pipeline` only writes ONNX + YAML)

## Origin

Logic ported from NVIDIA LEAPP (`leapp.backends.onnx_graph_surgery` /
`align_leapp_export_for_tensorrt`), Apache-2.0.
