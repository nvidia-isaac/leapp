# Generic `.wrp` ONNX Op Prototype

This directory prototypes one ONNX Runtime custom op type that can launch
different Warp APIC `.wrp` graphs from C++ based on per-node attributes.

The current prototype is intentionally narrow:

- custom op domain: `com.nvidia.warp`
- op name: `WrpRunner`
- inputs: two `tensor(float)` CPU tensors
- outputs: one `tensor(float)` CPU tensor
- output shape: supplied by the ONNX node's `output_shape` attribute
- execution: CPU ONNX Runtime op copies host input buffers into the loaded APIC
  graph, launches the CUDA graph through Warp, synchronizes, then copies the
  named APIC output back to the ONNX output tensor.
- demo model: two `WrpRunner` nodes, each loading a different `.wrp` file with
  the same custom op schema.

## Run

Use the Warp/APIC environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate exp_env
```

Generate the APIC graphs:

```bash
python experiment_warp/generic_onnx_op/generate_wrp.py
```

Build the custom op:

```bash
cmake -S experiment_warp/generic_onnx_op -B experiment_warp/generic_onnx_op/build
cmake --build experiment_warp/generic_onnx_op/build -j2
```

Create and run the ONNX model:

```bash
python experiment_warp/generic_onnx_op/make_onnx.py
python experiment_warp/generic_onnx_op/run_onnx.py
```

## Requirements Exposed

- ONNX Runtime custom op arity is part of the registered schema. A truly generic
  runner needs generated op schemas per input/output count, variadic support, or
  a fixed max-arity convention.
- APIC params are byte-size fixed. ONNX tensor shape and dtype must match the
  `.wrp` snapshot exactly unless the graph is recaptured.
- APIC does not provide ONNX output shape inference. Each node currently passes
  `output_shape`, and the C++ op validates that shape against
  `wp_apic_get_param_size`.
- This prototype uses CPU ORT tensors and Warp's host-pointer APIC API. A
  production version should decide whether GPU I/O binding is required.
- The `.wrp` file and its sibling `_modules/` directory must be shipped together.
- The ONNX node needs stable metadata for `wrp_path`, `input_names`, and
  `output_names`. Production packaging likely needs relative paths or an artifact
  resolver instead of absolute paths.
