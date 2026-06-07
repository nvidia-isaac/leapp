# Warp regions deploy as native APIC `.wrp`, never as ONNX custom ops

LEAPP captures Warp kernels opaquely (one CUDA-graph segment per contiguous warp run) and
deploys each as a native APIC `.wrp` artifact — a peer node-kind alongside torch
(ONNX / TorchScript). We deliberately do NOT embed warp kernels as ONNX custom ops.

## Why

A warp kernel is hand-written CUDA. Wrapping it as an ONNX custom op gains neither of ONNX's
benefits — it is not portable (it still needs warp + CUDA at runtime) and ONNX Runtime cannot
optimize across the op boundary — while adding a 3-way version pin (ONNX opset × ORT
custom-op ABI × `.wrp` format) and blocking the pure-C++ hard-RT `WarpRunner` path (which
would otherwise need ORT in the loop). Keeping `.wrp` native lets the same artifact run under
the Triton python backend and the C++ `WarpRunner` unchanged.

## Consequences

A single leapp node is single node-kind. A graph that interleaves torch and warp is
auto-segmented into multiple native nodes connected by data-flow edges; GPU-residency across
those edges is preserved (no host round-trip). A genuinely mixed single deployable unit, if
ever needed, would be a python-backend *fused* node — still not ONNX.
