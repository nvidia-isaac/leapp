# isaac_deploy_trt

Standalone GraphSurgeon toolkit that applies the same TensorRT ONNX optimizations
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
| `BitwiseNot`, BOOL `BitwiseAnd` / `Or` / `Xor` | `Sub(-1, x)`, or `Not` / `And` / `Or` / `Xor` |
| `QLinearConv`, `QLinearMatMul` | Float `Conv` / `MatMul`, zero-point folded into `Sub` |
| `ConvInteger`, `MatMulInteger` | Float product, then `Cast` to INT32 |
| `DynamicQuantizeLinear` | `ReduceMin` / `ReduceMax`, `Round`, `Clip` |
| Static `CenterCropPad` | `Slice` and `Pad` |

`unsupported-ops` marks these with `surgeon`. String, sequence, and random
ops stay unsupported: TensorRT has no equivalent operator.

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

### Optimize one file

```bash
isaac_deploy_trt optimize IN.onnx -o OUT.onnx

# TensorRT 10.16, Jetson Orin catalog
isaac_deploy_trt optimize IN.onnx -o OUT.onnx --tensorrt 10.16 --platform orin
```

Example (the only Apple PnP node that currently needs surgery):

```bash
isaac_deploy_trt optimize \
  /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx/preprocess_video.onnx \
  -o /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx_surgeon/preprocess_video.onnx
```

### Optimize a directory into another folder

Copies every `*.onnx` (and matching `.onnx.data`) into `-o`. Only graphs that
hit the three patterns above are edited.

`-o` **must not** be the input directory.

```bash
isaac_deploy_trt optimize-dir INPUT_DIR -o OUTPUT_DIR --tensorrt 10.16 --platform orin
```

``--tensorrt`` and ``--platform`` select the operator catalog, the same way
they do for ``check``. Unsupported-op decomposition runs only for operators
that catalog marks unsupported. UINT8 Cast, Resize antialias, and Reshape rank
edits still run. ``--platform`` is ``x86``, ``orin``, ``thor``, or ``dgx``, and applies with ``--tensorrt``.
Omitting ``--tensorrt`` uses the minimal catalog, the operators unsupported
on every checked-in TensorRT release. ``merge-pipeline --optimize`` uses that
same default.

GR00T Apple PnP LEAPP pack:

```bash
isaac_deploy_trt optimize-dir \
  /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx \
  -o /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx_surgeon
```

For `nvidia/GR00T-N1.7-ApplePnP-V1` as exported today:

| File | Surgeon |
|---|---|
| `preprocess_video.onnx` | optimize (UINT8 Cast, Resize antialias, rank-9 Reshape) |
| `preprocess_state.onnx` | copy |
| `backbone.onnx` | copy |
| `action_head.onnx` | copy |
| `decode_action.onnx` | copy |

YAML (`exported_leapp.yaml`) is **not** copied by `optimize-dir`. Copy it
yourself if you need a full deploy tree.

### List ONNX operators TensorRT does not support

Prints the static parser table (onnx-tensorrt `operators.md`). No ONNX file
and no Triton / ONNX Runtime install. With no flags this is the TensorRT 11.2
table. `check`, `optimize`, and `merge-pipeline --optimize` use the minimal
catalog unless `--tensorrt` or `--catalog` is set.

```bash
isaac_deploy_trt unsupported-ops
isaac_deploy_trt unsupported-ops --json
isaac_deploy_trt unsupported-ops \
  --catalog tensorrt_catalogs/v11.2/tensorrt_layers_cuda132_x86.json
```

Standalone script (same flags):

```bash
python scripts/list_unsupported_onnx_ops.py
```

### Catalog-check ONNX (validate)

Static check: ops in the graph vs a TensorRT operator catalog.
Does **not** optimize files. With no version flag, the minimal catalog is used:
an operator counts as supported only when every checked-in TensorRT release
marks it supported. `--tensorrt` loads that release from
`tensorrt_catalogs/` (`10.16` and `v10.16` both work). `--platform` selects
`x86`, `orin`, `thor`, or `dgx` for that release. `--catalog` points at one JSON
file and replaces the version flags.

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

# TensorRT 10.16 catalog, Jetson Orin overlay
isaac_deploy_trt check --onnx PATH.onnx --tensorrt 10.16 --platform orin --strict
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

``--optimize`` applies the GraphSurgeon TensorRT edits to each subgraph before
merge (needed if you have not already run ``optimize-dir``). With no
``--tensorrt`` flag it uses the minimal catalog, the same default as
``check`` and ``optimize``.

``--model-dir`` defaults to the YAML directory. ``--yaml-out`` overrides the
sibling ``pipeline.yaml`` (TensorRTNode ``input_tensor_names`` /
``output_tensor_names``).

Standalone script (same flags after the YAML path):

```bash
python scripts/merge_pipeline.py EXPORTED.yaml -o pipeline.onnx --optimize
```

GR00T Apple PnP (rewritten video preprocess + merge):

```bash
isaac_deploy_trt merge-pipeline \
  /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx/exported_leapp.yaml \
  --model-dir /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_onnx \
  -o /mnt/sdb2/workspaces/leapp/gr00t-leapp-export/apple_pnp_pipeline/pipeline.onnx \
  --optimize
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

Each database is one JSON object per platform. Native layers are
`tensorrt_layers` (`gpu`, `dla`, `notes`). ONNX parser ops are
`onnx_operators` (`supported`, `types`, `restrictions`).

The minimal catalog is `tensorrt_catalogs/minimal/tensorrt_layers.json`.
It is the default for `check`, `optimize`, `optimize-dir`, and
`merge-pipeline --optimize` when `--tensorrt` is omitted, and for
`leapp.optimize_graph(trt_compatible=True)`. A native layer is included
only when every checked-in catalog sets `gpu: true`. `onnx_operators` is
the same intersection: an operator is supported only when every catalog
marks it supported.

```bash
isaac_deploy_trt minimal-layers
isaac_deploy_trt minimal-layers -o tensorrt_catalogs/minimal/tensorrt_layers.json
```

There are no public `v11.3` / `v11.4` tags. Latest OSS 11.x tags are `v11.0`,
`v11.1`, `v11.2`.

---

## Python

These helpers are what `optimize` and `optimize-dir` call.

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
