# LEAPP

LEAPP captures a mixed PyTorch / NumPy / Warp compute graph by tracing the user's
ordinary code, and emits a portable bundle (per-node model artifacts + a YAML pipeline
spec) that deployment runtimes — Triton today — execute on robots.

## Language

**Leapp node**:
A unit of the captured graph that compiles to exactly one deployable model artifact, and
is single node-kind (all-torch or all-warp). Boundaries are determined automatically at
bridges; a user may also define them manually.
_Avoid_: layer, stage, op

**Region**:
The span of traced code a user marks as belonging together — everything between an
`annotate.input_tensors` and the matching `annotate.output_tensors` under one name. A region
may contain several segments and therefore compile to several nodes.

**Segment**:
A maximal contiguous run of single node-kind operations within a region, bounded by bridges
or the region's edges. Each non-empty segment becomes one auto-generated leapp node.

**Node-kind**:
The kind of artifact a node compiles to — `torch` (ONNX / TorchScript) or `warp`
(an APIC `.wrp`). At the code level this is the node's `backend`.

**Visualization port**:
A rendered input or output row inside a leapp node visualization. It represents an existing
named tensor input or output, including metadata such as shape, dtype, and kind when
available. A visualization port is not a deployable model artifact and is not a separate
graph node.
_Avoid_: treating ports as pipeline nodes

**Warp node**:
A leapp node whose artifact is a native Warp APIC `.wrp` — one captured CUDA graph holding
a contiguous run of warp kernels. A warp node is never wrapped as an ONNX custom op.

**Bridge**:
A `wp.from_torch` / `wp.to_torch` crossing where a tensor passes between a torch node and a
warp node. Bridges are where a graph edge between node-kinds is detected, and where a node
boundary is placed during automatic segmentation.

**Mark** (a tensor):
The user-facing act of designating a tensor as a graph input or output. Everything between
marked tensors is traced automatically; the node structure in between is derived, not
hand-declared.
_Avoid_: annotate (reserved for the lower-level `annotate.*` node API), register
