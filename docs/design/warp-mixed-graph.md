# Warp × mixed-graph — design notes (v1)

Living notes for the unified, non-invasive Warp capture work. Terminology lives in
[`/CONTEXT.md`](../../CONTEXT.md); hard decisions live in [`docs/adr/`](../adr/). This file
holds scope and known limitations that don't rise to an ADR.

## Status

v1 implemented. Delivered:

- **Structured dtypes** (`vec3f`, `quatf`, `transformf`, …) round-trip correctly through the
  warp backend; the dtype registry is the single source of truth.
- **Auto-split of linear torch↔warp regions** at `wp.from_torch` / `wp.to_torch` bridges:
  a single `leapp.start()/stop()` region is automatically segmented into `<prefix>.NN_torch`
  and `<prefix>.NN_warp` LEAPP nodes with no user annotation beyond the existing
  `annotate.input_tensors` / `annotate.output_tensors`.
- **End-to-end projected-gravity-style normalize validated bit-exactly** via InferenceManager:
  `torch-scale → warp-vec3-normalize → torch-reshape` compiles, exports, and replays with
  `max_abs_err < 1e-7` vs. `F.normalize(g * 2, dim=1)`.

Deferred (fast-follow / future):

- Empty-segment pruning (trivial identity torch nodes at pure-warp region boundaries).
- Native warp state (persistent arrays that never cross to torch).
- Triton-generator support for auto-split graphs.
- DLPack and `__cuda_array_interface__` bridge detection.
- Triton deploy runtime (`warp_apic_runtime.py`) structured-dtype OUTPUT handling is untested; it allocates a flat scalar buffer rather than struct-typed elements. Needs a pytriton-backed test (or a struct-typed allocation) before the Triton auto-split deploy path is relied on.

## Model (summary)

The user marks tensors with the existing `annotate.input_tensors` / `annotate.output_tensors`.
Everything between is traced. A region that interleaves torch and warp is auto-segmented at
its bridges (`wp.from_torch` / `wp.to_torch`) into single-kind nodes, named
`<prefix>.<NN>_<kind>` (number first, zero-padded; bare `<prefix>` when a region is
single-kind). Torch segments export to ONNX/TorchScript; warp segments export to a native
APIC `.wrp` ([ADR-0001](../adr/0001-warp-native-wrp-not-onnx.md)).

## Known limitations (v1)

- **Stateless warp nodes only.** A warp node is a pure function of its bridged inputs.
  Persistent / simulation state is expressed as a **fed-back torch tensor** (`state_tensors`
  / `update_state`) that is bridged into the warp segment each step — `wp.from_torch` is
  zero-copy, so this is cheap, and LEAPP's existing feedback machinery handles persistence.
  Native warp-resident state (state arrays that never cross to torch and persist across
  steps) is **deferred**. v1 should fail loudly if it can detect a warp segment mutating a
  persistent, un-bridged array, rather than silently dropping the state.

- **Linear segment chains only** ([ADR-0002](../adr/0002-linear-segment-chain-v1.md)).
  Cross-bridge forks must be expressed as explicit manual nodes.

- **Bridge set is `wp.from_torch` / `wp.to_torch` only.** Other crossings (DLPack,
  `wp.array(ptr=...)`, `__cuda_array_interface__`) are not detected; v1 fails loudly rather
  than silently dropping a warp segment.

- **Empty-segment pruning is deferred (fast-follow).** The warp→torch bridge returns a
  TracedTensor bound to the continuation node so post-warp torch ops trace correctly. A region
  with torch work on both sides of the warp segment therefore yields real (non-empty) nodes. A
  *pure* warp region (no torch ops before `from_torch` / after `to_torch`) gets trivial
  identity torch nodes wrapping the warp node — numerically correct and GPU-resident, but
  not yet pruned. Pruning requires renaming the warp node's output port to the user's marked
  output name; it is a clean follow-up, not in v1.

## Motivating use cases (keep in mind; not yet designed)

- **Isaac Lab — root quaternion → projected gravity in Warp.** A small, common observation
  term implemented as a warp kernel; the canonical "trace a stateless warp op embedded in an
  otherwise-torch policy graph" target.
  - Code: `isaaclab_experimental/envs/mdp/observations.py` (`_projected_gravity_kernel`,
    `projected_gravity`), with a torch-parity test at
    `isaaclab_experimental/test/envs/mdp/test_observations_warp_parity.py`.
  - Confirms the bridge model: the data crosses via `wp.from_torch(..., dtype=...)`
    (e.g. `ProxyArray(wp.from_torch(gravity_dir, dtype=wp.vec3f))` in
    `isaaclab_ovphysx/.../rigid_object_data.py`).
  - **But it needs structured warp dtypes**: kernel inputs are `wp.array(dtype=wp.transformf)`
    (root pose) and `wp.array(dtype=wp.vec3f)` (gravity); only `out` is plain float32 `(N, 3)`.
    The bridge must map torch `[N,7] -> transformf`, `[N,4] -> quatf`, `[N,3] -> vec3f`, etc.
  - Output is written into a **pre-allocated `out` buffer** (`func(env, out, **params) -> None`),
    not returned — the output side is whatever torch tensor aliases `out`.
  - Stateless and reads `gravity` as a constant → fits stateless-v1 + baked-constant rules.
  - Real code wraps the bridge in helpers / `ProxyArray`, so interception must patch the
    `wp.from_torch` / `wp.to_torch` *symbols* (catches indirect callers), not look for literal
    call sites.
- **Dextrah — geometric fabric in Warp** (`https://gitlab-master.nvidia.com/dex/dextrah-unified`).
  A large, real warp body; the stress test for coarse capture and (eventually) the
  stateful / many-launch cases.

## Structured warp dtypes — in v1 scope

Because use-case 1 (projected-gravity) is a first target, **structured warp dtypes are in v1**
(not deferred). The declared dtype is handed to us for free at the bridge
(`wp.from_torch(t, dtype=wp.transformf)`), so we record and round-trip it rather than infer it
(inference would be ambiguous — `[N,4]` is `quatf` or `vec4f`). Each warp port carries **both**
the torch view (`base dtype + shape`, e.g. `float32 [N,3]`) **and** the declared warp struct
dtype (`vec3f`); validation stays strict on the torch side, and the runtime reconstructs the
struct view via `wp.from_torch(t, dtype=...)`. Requires extending
`warp_export_backend._WARP_DTYPE_TO_STR` (currently scalar-only) and allowing the declared
zero-copy reinterpret that the backend presently rejects.
