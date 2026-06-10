# Warp LEAPP Detection POCs

This directory contains trace-time detection experiments for Warp + LEAPP.

## Current Approach: Global Warp Function Profiling

The active direction is `warp_global_leapp_detector.py` plus
`poc_profiled_global_warp_detector.py`. This approach does not use
`TracedWarpArray.__array_interface__` or `.__cuda_array_interface__`.

Instead, it patches Python-visible Warp functions and selected Warp class
methods. A call is recorded as a valid candidate when both conditions hold:

- the call receives at least one LEAPP-traced or detector-tracked Warp array;
- the call returns a Warp array, mutates a Warp array argument/receiver, or has
  explicit output Warp arrays in the function signature.

When a valid call returns, the detector marks returned, output, and mutated Warp
arrays as tracked. That propagation lets later launches using raw intermediates
remain visible without wrapping the user computation in a function.

Run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate exp_env
python -u experiment_warp/global_wrap_detection/poc_profiled_global_warp_detector.py --device cpu
```

Optional CUDA run:

```bash
python -u experiment_warp/global_wrap_detection/poc_profiled_global_warp_detector.py --device cuda:0
```

The profiled smoke test covers:

- `wp.launch`
- consecutive `wp.launch` calls through raw intermediate arrays
- `wp.launch_tiled`
- a module-global pre-patch `wp.launch` alias
- a closure-captured pre-patch `wp.launch` alias, which is an expected blind spot
- `wp.copy`
- `wp.array.zero_`, `wp.array.fill_`, and `wp.array.assign`
- `wp.clone`, `wp.empty_like`, `wp.zeros_like`, and `wp.full_like`
- `wp.array.flatten`
- ignored readback/conversion calls such as `wp.to_torch` and `array.numpy()`
- `wp.launch(..., record_cmd=True)` followed by `Launch.launch()`, currently a
  blind spot because `Launch.launch()` no longer receives array arguments
- `wp.Tape.backward`, where the global `wp.launch` wrapper sees the internal
  adjoint launches

Validated results on both CPU and CUDA in `exp_env`:

- 529 patched call sites
- 26 valid events
- valid raw-intermediate propagation across a 3-launch chain
- valid Tape backward detection via adjoint `wp.launch(..., adjoint=True)` calls

## APIC Eligibility Filter

Python detection is broader than `.wrp` exportability. A detected call should
only become a `WarpSegmentRecipe` event if APIC capture/save/load/replay can
reproduce it.

The following Warp 1.13.0 utility helpers were checked on CUDA:

- `warp.utils.array_sum(values, out=out)`
- `warp.utils.array_inner(a, b, out=out)`
- `warp.utils.array_scan(in_array, out_array)`
- `warp.utils.radix_sort_pairs(keys, values, count)`
- `warp.utils.array_sum(values)` without `out`

The explicit-output utility calls entered `ScopedCapture(apic=True)` and could
save/load, but loaded `.wrp` replay produced no output effect. The no-output
sum path performs host readback and errored during capture. Treat these helpers
as unsupported export boundaries for v1, not as valid recipe events.

## Legacy Array-Interface POC

`poc_detect_leapp_warp_calls.py` is retained as historical evidence for the
older array-interface sentinel idea. That path can detect some launch argument
packing, but it has false positives, misses non-launch array operations, and is
not the current design direction.

## Export Implication

The global profiler should feed a future `WarpSegmentRecipe` owned by the active
LEAPP trace node. The FX graph should store a compact marker with a `segment_id`,
while the in-memory recipe stores live replay handles until `leapp.compile_graph()`
emits one or more `.wrp` files. Consecutive valid events can be coalesced into the
same segment for v1 as long as no unsupported operation or concurrency boundary is
observed. If an unsupported utility helper consumes or mutates tracked arrays
between valid launches, fail closed by ending or invalidating the current fused
segment instead of silently dropping that operation.
