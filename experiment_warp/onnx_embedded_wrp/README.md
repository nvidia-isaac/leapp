# Embedded `.wrp` ONNX Op Prototype

This is a parallel variant of `../generic_onnx_op`. Instead of referencing each
Warp APIC `.wrp` file by path, this prototype **carries the entire APIC bundle
inside the ONNX model as a tensor initializer**. The model is self-contained and
relocatable: no sibling `.wrp` / `_modules/` files, no environment variables, no
path resolution at load time.

## Why an initializer (not an attribute)

The bundle is stored as a `uint8` **tensor initializer** wired in as each node's
last input — not as a node attribute. That choice is what unlocks ONNX's
external-data mechanism:

- **Small models:** the bundle stays inline in the `.onnx` (same as a plain
  embedded blob).
- **Large models:** `save_as_external_data` spills the bundle into a sibling
  `<model>.onnx.data` file — exactly the pattern LEAPP already uses for large
  weights. ONNX Runtime resolves that file **relative to the model path**, so
  you ship `.onnx` + `.onnx.data` together and portability is preserved.

ONNX external data (and the 2 GB protobuf limit) only apply to `TensorProto`
initializers, **not** to attributes — so an attribute-carried bundle could never
be externalized and would hard-cap at 2 GB. The initializer carrier removes both
limits with one mechanism, controlled by a size threshold.

## How it works

- `wrp_bundle.py` packs `<stem>.wrp` plus its `<stem>_modules/` directory into a
  little-endian `WRPB` archive (raw bytes). Relative paths are preserved so the
  `<stem>_modules/` layout survives extraction.
- `make_onnx.py` stores each archive as a `uint8` initializer, wires it as the
  node's last input, and keeps `wrp_name`, `input_names`, `output_names`,
  `output_shape` as attributes. `--external-data` spills bundles to
  `<model>.onnx.data`; `--size-threshold` controls inline-vs-external.
- `wrp_runner_op.cc` (custom op `com.nvidia.warp::WrpRunner`) registers on
  `CUDAExecutionProvider`. It keeps data inputs/outputs device-resident, marks
  only the bundle initializer as CPU input, reads that bundle on the first
  `Compute`, extracts the `WRPB` archive into a private `mkdtemp` directory,
  loads it with `wp_apic_load_graph`, caches the graph, and removes the temp
  directory in its destructor. Per-run data movement is device-to-device on
  ORT's CUDA stream, followed by `wp_apic_launch(graph, stream)`.
- The single op type still serves multiple graphs: each ONNX node carries its
  own embedded bundle.

## Run

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate exp_env

python experiment_warp/onnx_embedded_wrp/generate_wrp.py
cmake -S experiment_warp/onnx_embedded_wrp -B experiment_warp/onnx_embedded_wrp/build
cmake --build experiment_warp/onnx_embedded_wrp/build -j2

# inline bundles (small-model regime)
python experiment_warp/onnx_embedded_wrp/make_onnx.py
python experiment_warp/onnx_embedded_wrp/run_onnx.py

# external-data bundles (large-model regime: .onnx + .onnx.data)
python experiment_warp/onnx_embedded_wrp/make_onnx.py --external-data --size-threshold 0
python experiment_warp/onnx_embedded_wrp/run_onnx.py
```

`run_onnx.py` copies only the `.onnx` (plus its `.onnx.data` sidecar when
present) into a temporary directory with no `.wrp` files and runs from there to
demonstrate self-containment. Use `--no-portable-check` to run in place.

## Limitations / trade-offs

- **CUDA EP required.** The op registers on `CUDAExecutionProvider`; use
  `onnxruntime-gpu` with its CUDA/cuDNN provider dependencies available. In this
  conda environment `run_onnx.py` imports PyTorch before ONNX Runtime so those
  CUDA libraries are preloaded. It runs on GPU machines, not on CPU-only ones.
  A bundle is CUDA-or-CPU per `.wrp` (decided at capture). True no-GPU inference
  would need a CPU capture (`.o` modules in `_modules/`), a CPU replay branch in
  `wrp_runner_op.cc` (`wp_load_obj` + `wp_apic_cpu_replay_graph`), and
  `warp-clang` on the target. Embedding solves file-shipping, not cross-target
  portability.
- Inline mode enlarges the `.onnx`; external-data mode adds a `.onnx.data`
  sidecar you must ship alongside the model (same deal as LEAPP weights).
- Embedded cubins are CUDA-arch / Warp-version specific. Regenerate on Warp
  upgrade or for a different target arch.
- The bundle is loaded once on first `Compute` (temp-dir extraction); concurrent
  `Run` calls on a freshly created session are not supported by this prototype.
- `.wrp` files are **build-time inputs only** (consumed by `make_onnx.py`). Once
  the model exists they are not needed for inference.
