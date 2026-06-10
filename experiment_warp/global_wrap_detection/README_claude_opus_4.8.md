# Trace-time Warp kernel capture — refined design (Claude Opus 4.8)

This document refines the design for capturing Warp kernel launches **at LEAPP
trace time** and exporting them as `.wrp` bundles that run inside ONNX.

> Scope note: this README only covers **trace-time capture**. The downstream
> `.wrp` → ONNX execution path is already de-risked (see
> `experiment_warp/onnx_embedded_wrp/` and
> `experiment_warp/benchmark_warp_kernel_pytorch_interop/`).

**Chosen design: two-pass bookmark / inject.** Warp export requires the user to
run the whole pipeline **twice** (this is an accepted constraint). Pass 1 records
*where* contiguous Warp segments begin and end; pass 2 re-runs the same code and
injects `capture_begin` / `capture_end` at those bookmarks. This is validated by
`test_two_pass_bookmark.py`. Earlier designs and dead ends are recorded at the
end so we do not revisit them.

---

## Requirements being satisfied

1. Keep LEAPP semantics: `input_tensors()` / `output_tensors()` mark node I/O.
2. No need to wrap Warp computation in user-defined functions to export via APIC.
3. Final output is `.wrp` files executable through the ONNX custom op.
4. (Nice-to-have) consecutive Warp launches are stored in the **same** `.wrp`.

## Hard constraints learned along the way

- **You cannot keep an APIC capture open across user Python.** While a capture is
  active you cannot `print` array contents, call `array.numpy()`,
  `wp.synchronize_*`, or do torch interop. The set of capture-illegal operations
  is effectively **infinite and not enumerable**, so any design that keeps a
  capture open across arbitrary user code is unsafe.
- **You cannot fuse by replaying saved/loaded `.wrp` graphs.** Proven in
  `test_wrp_fusion.py`: `capture_launch` of an instantiated graph inside an outer
  `ScopedCapture` raises CUDA error **900** (`cudaErrorStreamCaptureUnsupported`),
  and a graph replay carries no `apic_info` so the outer `.wrp` would be empty.
- **Fusion only works when the actual launches execute *live* inside one
  capture.** Proven in `test_wrp_fusion.py`, `test_command_log_replay.py`, and
  `test_two_pass_bookmark.py`.

---

## Key findings from Warp 1.13 internals

(`site-packages/warp/_src/context.py`, `_src/types.py`)

1. **APIC records at the stream level, not the Python-API level.**
   `capture_begin(apic=True)` ... `capture_end()` records whatever launches
   dispatch on the stream during the region. The recipe (kernel, dim, args,
   ordering) comes from APIC; we only have to control *when* the region opens.

2. **All launches funnel through a small, finite set of low-level dispatch
   primitives** — `runtime.core.wp_cuda_launch_kernel` / `wp_cpu_launch_kernel`
   (call sites at `context.py` 8143/8155/8344/8395/8438, plus the JAX FFI path).
   The set of *high-level* entrypoints (`wp.launch`, `Launch.launch()`,
   `warp.fem`, tile API, …) is open-ended, but they all bottleneck here. This is
   the universal interception point for both detection (pass 1) and injection
   (pass 2), so **unknown launch entrypoints are handled for free.**

3. **Manual `capture_begin`/`capture_end` can be driven programmatically** from
   inside the dispatch wrapper (no `ScopedCapture` context manager needed), and
   `capture_save` of such a graph is valid — proven in `test_two_pass_bookmark.py`.

4. **Tracked-vs-untracked egress is a bounded surface we own.** Host reads of
   *tracked* arrays go through `TracedWarpArray` methods (`numpy`, `__array__`,
   `list`, `__dlpack__`, `tensor`, `__repr__`/`__str__`) plus `wp.synchronize_*`
   and `wp.from_torch`/`wp.to_torch`. Per the user qualifier, untracked arrays are
   constants, so the egress surface that matters is enumerable — unlike the
   infinite set of all capture-illegal ops.

5. **What actually breaks a capture vs. what is harmless** (proven in
   `test_region_detection.py`, 7 launches with one op interleaved):

   | Interleaved op | Capture stays valid? | Why |
   | --- | --- | --- |
   | another warp launch | ✅ yes | pure on-stream work |
   | read tracked array **contents** (`print(x)`, `x.numpy()`, `x.list()`) | ❌ **breaks** | D2H sync → CUDA 906 in `wp_memcpy_d2h` |
   | read tracked array **metadata** (`x.shape`, `x.dtype`, `x.ndim`, `x.ptr`) | ✅ yes | no device read at all |
   | unrelated torch CUDA op on the **default** stream | ❌ **breaks** | legacy default stream depends on the capturing blocking stream → CUDA 906 |
   | unrelated torch CUDA op on a **side** stream (`torch.cuda.Stream`) | ✅ yes | independent stream, no dependency on the capture stream |

   Consequences for region detection:
   - `print(x)` calls `wp.array.__str__` → `str(self.numpy())` (`types.py:3729`),
     a synchronous D2H. So **printing a tracked array's contents is egress** and
     forces a segment boundary; only metadata access (shape/dtype) is free.
   - An unrelated torch op is **not inherently safe** — it breaks the capture on
     the default stream and is only safe if isolated on its own CUDA stream. It
     touches no `TracedWarpArray`, so the pass-1 array hooks **cannot see it**; it
     surfaces only as a pass-2 CUDA 906, caught by the per-segment `try/except`
     backstop (which then splits the segment).

---

## The design: two-pass bookmark / inject

### Pass 1 — record bookmarks (everything eager, no capture)

The user runs the pipeline normally. Nothing is captured, so `print`, `.numpy()`,
and torch interop all work. At the dispatch funnel we tag each launch with an
`(ordinal, call-site)`. A **break** is recorded whenever a tracked-array host
egress happens between two launches (the bounded surface above), and at torch
interop boundaries and `output_tensors()`.

A **segment** is a maximal run of launch ordinals with no break between them. The
output of pass 1 is a list of bookmarks `[(start_ordinal, end_ordinal), …]` plus,
per segment, the external tracked I/O (arrays read but not produced inside the
segment = inputs; arrays written inside = outputs).

### Pass 2 — inject capture by ordinal (the same code runs again)

The user runs the identical pipeline a second time. We gate purely at the
dispatch funnel by ordinal — **no source-line patching, no runtime egress
detection**:

- before the dispatch of a launch whose ordinal is a `segment_start` →
  `capture_begin(device, apic=True)`;
- launches inside the segment just dispatch (recorded into the graph);
- after a launch whose ordinal is a `segment_end` → `capture_end()` →
  `capture_launch(graph)` + `synchronize` (so buffers hold real data for the
  egress that follows) → `capture_save(graph, <save_path>/<node>,
  inputs=<tracked inputs>, outputs=<tracked outputs>)`.

Because the capture window is exactly `[segment_start … segment_end]` and pass 1
proved no egress occurs inside it, **no capture-illegal op can land inside the
window**. Consecutive launches in a segment fuse into one `.wrp` (requirement 4);
an egress between them is a segment boundary, yielding chained `.wrp`s /
`WrpRunner` ops (chaining is proven in the benchmark).

### Tracked / untracked contract (qualifier)

- **Tracked array ⇒ runtime `.wrp` param.** Named in `capture_save`; bound via
  `set_param` / read via `get_param`; mapped to the `WrpRunner` op I/O.
- **Untracked array, `dim`, scalar ⇒ baked constant.** APIC snapshots the bytes
  of every buffer it touches; anything *not named* in `capture_save` stays a
  frozen snapshot — so "inline as a constant" is the **default APIC behavior, no
  extra code** (proven in `test_command_log_replay.py`, where a non-tracked
  `bias` array is baked and a post-capture mutation of it is ignored). `dim` is
  baked per launch and kernel scalars are baked into the launch record. This
  mirrors existing LEAPP semantics (plain `np.ndarray` inlined as constant;
  `register_buffer`).
- **Edge case (documented):** untracked ⇒ constant is only safe if the untracked
  array is genuinely constant in the region. If it is mutated mid-region (or is
  really data-dependent but left untracked), its frozen snapshot may be wrong.
  Anything data-dependent must be tracked; a content-signature check catches the
  mismatch.

### Robustness / fallbacks

- **Determinism between the two runs** is required. Use the same inputs; key
  bookmarks by call-site + per-site count and **verify the call-site at each
  ordinal in pass 2**; on mismatch, fall back to **per-launch capture** (always
  valid, no fusion).
- **Mis-bookmark backstop:** wrap each segment's capture in `try/except`; a CUDA
  capture error (e.g. an un-detected untracked egress inside the window) →
  fall back to splitting that segment into per-launch captures.

### LEAPP graph + export

- Each fused segment emits **one FX `call_function` node** (`leapp::warp_apic`
  marker); tracked output arrays are class-swapped in place
  (`arr.__class__ = TracedWarpArray`; rebind `_name`/`_context`/`_proxy`).
- `compile_graph()` lowers these via the existing ONNX / TorchScript exporters
  into the embedded-bundle `WrpRunner` op. **No new user-facing backend.**

---

## Empirical validation (in this folder)

- **`test_wrp_fusion.py`** — fusing by replaying saved graphs **FAILS** (CUDA 900);
  fusing by live re-issue **PASSES**. → never fuse by graph replay.
- **`test_command_log_replay.py`** — a 3-kernel chain over distinct buffers fuses
  into one `.wrp`; tracked arrays are the only params; non-tracked `bias` is baked
  (not a param, post-capture mutation ignored). **All PASS.**
- **`test_two_pass_bookmark.py`** — the two-pass model end-to-end: pass 1 detects
  segments `[(1,2),(3,3)]` split by one egress; both passes' egress see correct
  eager values (replay-on-close works); the 2-launch segment fuses into one valid
  `.wrp` and both segments reload correctly. **All PASS.**
- **`test_region_detection.py`** — what is / isn't a region boundary, 7 launches
  with one op interleaved inside a single capture: baseline 7-launch fuse **OK**;
  tracked-array **contents** read (`print(x)`, `x.numpy()`) **BREAKS** (CUDA 906);
  tracked-array **metadata** (`x.shape`) **OK**; unrelated torch op on the
  **default** stream **BREAKS** (CUDA 906), on a **side** stream **OK**. Confirms
  contents-egress and default-stream torch are boundaries; metadata and
  stream-isolated torch are not.

---

## Prerequisite refactor (first implementation step)

`TracedWarpArray` must be single-inheritance and virtually registered:

```python
class TracedWarpArray(wp.array):   # NOT (TracedData, wp.array)
    ...

TracedData.register(TracedWarpArray)
```

Multiple inheritance makes CPython pick `TracedData` as the solid base, so
`raw_wp_array.__class__ = TracedWarpArray` fails with an object-layout error.
Single inheritance + `ABC.register()` keeps `isinstance(x, TracedData)` true
without changing the object layout. Copy/factor the needed `TracedData` helpers
(`proxy`/`name`/`context` props, `validate_status`, unwrap/find helpers, `_new`)
into `TracedWarpArray` or module-level utilities.

## Module layout

- New `warp_patching` module (sibling to `global_patching.py`): wraps the
  low-level dispatch funnel (with `wp.launch` as the common fast path) to (pass 1)
  record `(ordinal, call-site)` + egress breaks, and (pass 2) inject
  `capture_begin`/`capture_end` by ordinal and drive `capture_save`. Tracks the
  bounded tracked-array egress surface for break detection. Applied/removed from
  `leapp.start()` / `leapp.stop()` like `apply_traced_data_patches()`. A
  pass counter (run 1 vs run 2) lives in the manager.

## Validation target

`experiment_warp/tracing_situations/situations.py` (`warp_kernel_chain`), run
twice: pass 1 bookmarks, pass 2 captures. `averaged` ends as a `TracedWarpArray`,
`warp_kernel_chain.yaml` shows node I/O + `.wrp` reference, and the example no
longer needs a manual `wp.ScopedCapture` / `wp.capture_save`.

---

## Alternative (kept as fallback): command-log / replay

Instead of a second user run, record (pass 1) the sequence of Warp primitives
that touch tracked arrays, then re-issue that log inside one LEAPP-controlled
`ScopedCapture` at the node boundary (reset tracked inputs to stored values
first). Validated by `test_command_log_replay.py`.

- Pro: capture has **zero user code inside** (max safety); single user run.
- Con: must wrap each replayable primitive and reconstruct the call; an unwrapped
  exotic entrypoint is *missed* (only detected after the fact via content
  signature), whereas two-pass captures exotic entrypoints live for free.

Use command-log/replay where a second full run is unacceptable; otherwise prefer
two-pass.

---

## Discarded ideas (do not revisit)

1. **APIC begin/end tied to `input_tensors`/`output_tensors`.** Holds a capture
   open across user code → blocks `print`/`.numpy()`/interop. Rejected.
2. **Single-pass lazy auto-suspend** (open at funnel, close on the next
   capture-illegal op). Workable given the qualifier (bounded tracked egress), but
   correctness still leans on intercepting every egress at runtime. Two-pass is
   preferred because pass 2 closes by **known ordinal**, removing runtime egress
   detection from the critical path.
3. **Per-launch capture + fuse by replaying the saved graphs.** Fusion step is
   impossible — CUDA error 900 (`test_wrp_fusion.py`).
4. **Ledger driven only by the `__cuda_array_interface__` detector.** The detector
   gives timing, not the recipe, so it cannot drive replay on its own.
