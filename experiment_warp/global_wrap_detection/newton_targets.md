# Newton Targets for Warp APIC Export

Date: 2026-06-10

## First Real Target: Newton FK

`newton.eval_fk` is the best first real third-party target for the traced-array
Warp APIC export path.

Why it is a good fit:

- It is a high-level Newton API, so it tests the "user did not call `wp.launch`
  directly" requirement.
- Its device work lowers to normal Python-visible Warp kernel launch machinery.
- The global Warp function profiler should see the underlying `wp.launch` calls
  even when the user enters through a high-level Newton API.
- It is mostly a straight-line FK operation, not an optimizer loop.
- It should exercise the remaining important production work without pulling in
  every hard case at once:
  - binding raw Warp outputs back to the launch marker,
  - preserving side-effect FX marker nodes,
  - serializing model arrays/constants,
  - replaying the launch segment under APIC,
  - saving one `.wrp` bundle for ONNX execution.

Recommendation: use Newton FK as the first end-to-end "real library" proof after
the current FX marker POC.

## IK Status

Newton IK should be treated as a later, harder target rather than the first
proof, but it is not a feasibility unknown.

Important proven fact: IK is exportable when executed inside a known
`wp.ScopedCapture` / APIC-managed capture boundary in one shot. The remaining
problem is automatic trace-time discovery of the right Warp segment without
forcing the user's surrounding Python code to run inside an always-on APIC
capture.

Current Newton IK is structured around `IKSolver`, a front-end that handles
sampling, optimization, and solution selection. Its `step(...)` path samples
seeds, resets optimizer state, runs either LM or L-BFGS for a fixed iteration
count, computes costs, and selects/gathers the best seed.

Why IK is harder:

- It is an optimizer, not a single kinematics operation.
- It can run many Warp launches per solve, scaling with `iterations`,
  objectives, seeds, and optimizer backend.
- It uses non-kernel Warp operations that must be profiled as valid effects:
  `wp.copy`, `array.zero_`, `array.fill_`, allocation helpers, and possibly
  flattened/sliced array views.
- LM autodiff mode uses `wp.Tape`.
- CUDA objective evaluation uses per-objective `wp.Stream`, `wp.Event`,
  `record_event`, `wait_event`, and `wp.ScopedStream`.
- The result is heavily stateful: solver buffers, damping values, costs,
  residuals, Jacobians, proposed states, accept/reject flags, and best-seed
  buffers are mutated across the solve.

Detection/export outlook:

- Kernel launch detection should be driven by global Warp function profiling,
  not array-interface access. The wrapper records calls that receive traced or
  propagated-tracked Warp arrays and either mutate arrays, return arrays, or
  expose output arrays in the function signature.
- Since one-shot `ScopedCapture` export is known to work, the right design is
  not to prove IK can be captured by APIC; it is to reconstruct the same
  capture boundary automatically from trace-time evidence.
- The current profiler sees kernel launches, copy/fill/zero/allocation/view
  events, raw-intermediate propagation, and `wp.Tape.backward` through its
  internal adjoint `wp.launch(..., adjoint=True)` calls.
- A full automatic IK export likely needs the profiler to feed a broader Warp
  segment recorder that detects the start/end of the relevant contiguous Warp
  activity, records kernel launches plus copy/fill/zero/allocation/view/tape
  events, then replays that fixed segment under APIC to produce the `.wrp`.
- For v1, ignore explicit stream/event concurrency and fail closed if a custom
  stream/event boundary appears in the profiled segment.

Suggested first IK proof, after FK:

- CPU or single-stream CUDA.
- Analytic Jacobian mode first, avoiding `wp.Tape`.
- `n_seeds=1` first, avoiding best-seed selection.
- Fixed `iterations`.
- A small objective set.
- Fail closed on any unsupported operation in the recorded segment.

Verdict: IK is a real target because one-shot APIC capture works, but it is not
the right first automatic trace-time target. FK is the cleaner bridge from the
POC to production export; IK should follow once automatic segment discovery and
non-launch Warp side-effect recording are in place.


## Warp Segment Recipe Design

The trace-time Warp export path should split durable graph metadata from live
Python replay state.

The FX graph should contain only a serializable marker, for example:

```python
leapp_warp_segment(segment_id="warp_segment_0")
```

The actual replay information should live on the active LEAPP node/context that
owns the traced arrays:

```python
TracedTensorNode
  .warp_segments = {
      "warp_segment_0": WarpSegmentRecipe(...)
  }
```

This keeps the FX graph portable while still letting the exporter replay a
notebook-defined or otherwise non-importable Warp kernel before the process
exits.

Recommended structure:

```python
@dataclass
class WarpSegmentRecipe:
    segment_id: str
    node_name: str
    events: list[WarpEventRecipe]
    input_refs: dict[str, WarpArrayRef]
    output_refs: dict[str, WarpArrayRef]


@dataclass
class WarpLaunchRecipe:
    op: str
    kernel: object
    dim: object
    inputs: list[WarpArgSpec]
    outputs: list[WarpArgSpec]
    kwargs: dict
```

The `kernel` field should hold the live `wp.Kernel` object during the trace
session. It should not be serialized into FX as the primary recovery mechanism.
The FX marker can still include human-readable metadata such as kernel key,
module, and qualname for debugging, validation, and fallback lookup.

Export lifecycle:

1. During tracing, the global Warp function profiler detects calls that
   receive traced or propagated-tracked Warp arrays.
2. If the call satisfies the valid-effect rule, the active `TracedTensorNode`
   appends a `WarpLaunchRecipe` or broader `WarpEventRecipe` to the current
   `WarpSegmentRecipe`.
3. The profiler marks returned, output, and mutated Warp arrays as tracked so
   later calls using raw intermediates remain visible.
4. The FX graph receives a small marker that references only `segment_id`.
5. During `compile_graph()`, the Warp export backend rebinds traced inputs and
   outputs, replays the segment under `wp.ScopedCapture`, and writes the `.wrp`.
6. After the `.wrp` is written, runtime execution should depend on the `.wrp`
   artifact, not on the live Python kernel object.

Notebook implication: kernels defined in Jupyter cells do not need a stable
import path as long as `compile_graph()` runs in the same Python process before
the live kernel object disappears. The durable artifact is the emitted `.wrp`,
not the in-memory recipe.

Consecutive Warp launches should be coalesced into one `WarpSegmentRecipe` when
they belong to the same contiguous Warp activity region for a LEAPP node. That
segment can then replay under one APIC capture and produce one `.wrp`.


## Consecutive Reconstruction POC

`poc_reconstruct_warp_segment.py` tests whether trace-time launch recipes can be
replayed into one `.wrp` after the eager user pass.

Result on CUDA:

- A three-launch visible chain recorded three `WarpLaunchRecipe` entries.
- The recipes replayed under one `wp.ScopedCapture(apic=True)`.
- `wp.capture_save(...)` wrote one `.wrp`.
- Reloading the `.wrp`, setting only the traced inputs, and launching once
  matched the eager reference output.
- A non-traced `bias` array was baked into the `.wrp`, not exposed as a runtime
  parameter.

Important implementation detail: the trace-time recorder must be suspended
during APIC replay. Otherwise replaying the live recipe can recursively append
new launches to the same segment.

Blind spot:

- A chain where only the first launch consumes LEAPP-traced arrays recorded only
  one of three launches.
- Later launches that consume only raw intermediate `wp.array` buffers do not
  trigger the traced-array interface hook.

Design implication: the traced-array hook is enough to reconstruct consecutive
visible launches, but generic multi-launch Warp export still needs either
propagation of tracked status to raw output/intermediate arrays or a broader Warp
command recorder for the active segment.


## Global Function Profiling POC

`poc_profiled_global_warp_detector.py` replaces the array-interface detector as
the active trace-time discovery design.

Validated on CPU and CUDA (`cuda:0`) in `exp_env`:

- 529 Python-visible Warp call sites patched.
- 26 valid events recorded in the scenario suite.
- Direct `wp.launch` calls with traced inputs are detected.
- A three-launch raw-intermediate chain is detected after output propagation.
- `wp.copy`, `array.zero_`, `array.fill_`, `array.assign`, `wp.clone`,
  `wp.empty_like`, `wp.zeros_like`, `wp.full_like`, and `array.flatten` satisfy
  the valid-effect rule.
- `wp.to_torch` and `array.numpy()` are ignored because they return non-Warp
  objects and do not mutate/output Warp arrays.
- A module-global `wp.launch` alias imported before patching is patched and
  detected.
- A closure-captured pre-patch alias is a blind spot.
- `wp.launch(..., record_cmd=True)` followed by `Launch.launch()` is a blind spot
  because the later `Launch.launch()` call no longer receives array arguments.
- `wp.Tape.backward` is detected by the global `wp.launch` wrapper observing the
  internal adjoint launch traffic.

### APIC Eligibility Filter

Do not equate Python-level detection with `.wrp` exportability. A call should
enter a `WarpSegmentRecipe` only if APIC capture/save/load/replay can reproduce
the effect.

Checked on CUDA with Warp 1.13.0:

- `warp.utils.array_sum(values, out=out)`
- `warp.utils.array_inner(a, b, out=out)`
- `warp.utils.array_scan(in_array, out_array)`
- `warp.utils.radix_sort_pairs(keys, values, count)`
- `warp.utils.array_sum(values)` without `out`

The explicit-output utility helpers entered `ScopedCapture(apic=True)` and
saved/loaded, but replayed `.wrp` graphs produced no output effect. The
no-output sum path performs host readback and errored during capture. Treat
these helpers as unsupported export boundaries for v1. If one consumes or
modifies tracked arrays between otherwise valid launches, end or invalidate the
current fused segment rather than silently omitting it.

## References

- Newton IK API docs: https://newton-physics.github.io/newton/api/newton_ik.html
- Newton `IKSolver` docs/source: https://newton-physics.github.io/newton/api/_generated/newton.ik.IKSolver.html
- Newton `ik_solver.py`: https://github.com/newton-physics/newton/blob/main/newton/_src/sim/ik/ik_solver.py
- Newton `ik_lm_optimizer.py`: https://github.com/newton-physics/newton/blob/main/newton/_src/sim/ik/ik_lm_optimizer.py
- Warp runtime docs: https://nvidia.github.io/warp/user_guide/runtime.html
