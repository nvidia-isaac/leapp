# NumPy → ATen adapter experiment

The local backlog ticket “Assess NumPy Compatibility Gap”
proposes selective reuse of `torch._numpy` implementations. This worktree implements
that option as the default for active NumPy functions and ufuncs. The original
`NUMPY_UFUNC_TO_TORCH` / `NUMPY_FUNC_TO_TORCH` tables, axis/default conversion
helpers, and special dispatch branches have been deleted. There is no legacy
fallback. A five-entry override registry handles demonstrated semantic/export
exceptions: `cbrt`, `pad`, `nanmean`, `sort`, and `argsort`.

Original comparison baseline: `536883bad09b038bf4ff52db5c5c99da0e2cf950`.
The PR branch is rebased onto `develop`; the baseline measurements below refer to
that original commit. Measurements ran on CPU with Python
3.12.11, PyTorch 2.14.0+cu130, NumPy 2.5.2, and LEAPP 0.7.2.

## Result

The same 38-case, deliberately gap-focused probe passes **1 case before and 36
cases after** for each of FX, scripted TorchScript, traced TorchScript, and
ExportedProgram. Each passing case matches real NumPy values, shape, and dtype
on the capture input, all elements increased by 2, and one element increased by 4.
This is 35 additional passing cases. The earlier fallback prototype passed
31/38; replacing the original mappings and adding the bounded overrides also
fixes cbrt, pad, nanmean, and integer mean. Shared slice lowering now also fixes
reverse reads. This is not an overall NumPy coverage
score; `isclose` is tested alone and in `where`.

| Cases | Baseline | Adapter, all four execution paths |
| --- | --- | --- |
| Existing sin + float mean control | Pass | Pass |
| inv, norm, det, slogdet, cholesky, solve, matrix_power, svd(compute_uv=False), eigvalsh, pinv | Untraced | Pass |
| cross, isclose | Untraced | Pass |
| isclose feeding where | Wrong changed-input results | Pass |
| sinc, degrees, radians, logaddexp, heaviside, copysign | Untraced | Pass |
| diff, gradient, repeat, tile, take, take_along_axis, broadcast_to | Untraced | Pass |
| median, quantile, ptp, unique | Untraced | Pass |
| cbrt | Untraced | Pass; override also tested on negative inputs and signed zero |
| pad | Untraced | Pass for static widths, constant mode, scalar fill |
| nanmean | Untraced | Pass; override also tested on NaN slices and integer promotion |
| float32 FFT | Untraced | Rejected: complex128 versus NumPy complex64 |
| Scalar indexing in `x + x[0, 0]` | Wrong changed-input results | Still wrong; separate interception gap |
| Reversed slicing | Replay fails | Pass; full reversal uses flip, bounded negative strides use index_select |
| int32 mean | Replay fails | Pass through torch._numpy; no override needed |

The exact signatures, inputs, and comparison logic are in
[scripts/compare_numpy_aten.py](scripts/compare_numpy_aten.py).
Tuple-returning operations are published as separate named LEAPP output leaves.
The probe does not establish support for publishing a NumPy namedtuple directly
as one LEAPP output; that existing node-output path has separate limitations.

## Where the cases came from

The reverse read `a[::-1]`, scalar indexing, integer mean, inverse, FFT, and
supported sin/mean control came from the saved probe attached to
the local “Assess NumPy Compatibility Gap” ticket.
Norm, cross, and isclose also have separate backlog tickets. The broader 38-case
matrix was assembled for this experiment by adding representative previously
unmapped NumPy operations. It is a targeted local probe, not an upstream benchmark
or comprehensive conformance suite. The new reverse-slice edge cases were designed
for this implementation and compare against real NumPy.

## Implementation

[aten_adapter.py](leapp/leapp_graph/datatypes/numpy/aten_adapter.py) resolves actual
NumPy callable identities in the NumPy, linalg, and fft namespaces to their
`torch._numpy` counterparts. Random operations are excluded. It captures the
implementation with `make_fx`, functionalizes temporary mutations, checks against
real NumPy, and inserts the resulting tensor operations into the current FX node.
No compatibility-layer objects or calls are needed during replay.

The adapter uses ATen overload packets so scripted TorchScript can resolve them.
[overrides.py](leapp/leapp_graph/datatypes/numpy/overrides.py) is consulted first;
otherwise the discovered `torch._numpy` implementation is used. Both paths use
the same capture and parity checks. Sorting overrides retain the prior static-K
TopK export strategy. A generic ATen constant-copy normalization uses `clone`
instead of `lift_fresh_copy` for legacy ONNX compatibility. NumPy dtype objects
are normalized into compatibility-layer dtypes.

Static dtype queries (`result_type`) remain native metadata. Unsupported calls
raise explicit errors during active
capture instead of returning an eager, potentially frozen intermediate. Inactive
carriers retain native NumPy behavior. Unsupported output mutation is rejected
before modifying caller storage; parity probes run on copies. Validated eager
outputs retain NumPy storage/view behavior.

This approach reuses the same tensor-backed NumPy layer described in the
[PyTorch compiler FAQ](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_faq.html),
without switching LEAPP's node capture to Dynamo.

Static reverse reads are lowered in shared `TracedData` indexing. Complete axis
reversal uses `torch.flip`; other negative strides use `slice.indices` against the
fixed capture shape, a constant index vector, and `torch.index_select`. Axis
selection happens before integer indexing and new-axis insertion, preserving
source-axis numbering. Basic tuples and ellipsis are supported. Runtime bounds
and advanced indices combined with negative strides are rejected during capture.
NumPy's eager view storage is preserved; singleton negative-stride arrays receive
an explicit copy only when converting their values to Torch.

## Validation

**450 tests passed, plus 114 subtests**, including 68 adapter tests, 57 reverse-slice
tests, both NumPy and Torch tracing suites (including the original partial-slice
assignment regressions), NumPy compatibility, non-Warp conversion, and 13 functional
tests for NumPy node connections, state/buffers, persistent carriers, and shared
memory. Coverage includes changed
inputs across FX/scripted TorchScript/PT2, negative values, isclose broadcasting
and NaN/infinity cases, tuple outputs, constant snapshots, rejection without
input mutation, inactive carriers, conversion patches, eager view storage, and
public LEAPP start/annotate/export/reload. The override tests exercise negative
cbrt inputs, signed zero, infinities/NaNs, explicit dtype handling, integer mean
and nanmean across FX/JIT/PT2, and unsupported signatures. Three existing eager
method assertions now compare carrier `.data` so NumPy's test-helper control flow
and internal `out=` mutations do not themselves get traced.

ONNX Dynamo export and ONNX Runtime changed-input replay pass for vector norm,
cross product, isclose feeding where, and representative reverse slices. Reverse
slice tests cover clipped/empty bounds, negative starts, integer indices, multiple
axes, ellipsis, new axes, NumPy integer bounds, and empty/singleton inputs. The wider
matrix does not claim ONNX
support for all new functions.

Five Warp conversion tests fail during CUPTI subscription with `ERROR_UNKNOWN
(999)`. The same five failures were reproduced on the unchanged baseline; they
were excluded from the passing CPU test run.

## Limits

This is a prototype depending on private PyTorch APIs, tested on one installed
version. Capture-time parity does not prove correctness for all runtime inputs:
`cbrt` initially passed positive inputs but failed negative inputs, motivating
the sign-preserving override. Additional signatures and upstream releases need their own parity tests.
The matrix uses valid inputs for linear algebra and does not establish identical
error behavior for singular matrices or other invalid runtime values.

Scalar indexing, constructor interception, Python conditions, raw ndarray writes,
general mutation/alias replay, dynamic input shapes, and GPU execution are outside
this change. The broad legacy mapping tables are gone; carrier methods and shared
indexing remain separate from function dispatch. Eager view storage is preserved, but
that alone does not establish graph replay of mutations through aliases.

The padding override accepts static Python widths, constant mode, and scalar fill.
Stable/structured sorting is explicitly rejected by the TopK overrides. FFT still
fails dtype parity, and scalar indexing retains its existing failure. Reverse
reads use static basic indices and fixed shapes; reverse assignment and mutation
through reversed views are not established by this change. Active `out=` mutation is rejected, including for formerly mapped
functions. These are bounded supported signatures, not complete NumPy support.

Each captured call incurs capture, functionalization, and parity-check cost.
Capture performance has not been benchmarked; replay contains ordinary tensor
operations. No public configuration API, release, or ticket closure is implied.

## Reproduction

From the repository root with the test dependencies installed:

```bash
PYTHONPATH="$PWD" python -m pytest \
  tests/unit_tests/test_numpy_aten_adapter.py \
  tests/unit_tests/test_numpy_slicing.py \
  tests/unit_tests/test_traced_tensor.py \
  tests/unit_tests/test_traced_np_array.py \
  tests/unit_tests/test_numpy_compatibility.py \
  tests/unit_tests/test_conversion.py \
  tests/functional_tests/test_connection_case.py::TestNumpyConnectionCase \
  tests/functional_tests/test_shared_memory_arrays.py::TestNumpySharedMemory \
  tests/functional_tests/test_state_tracking.py::TestNumpyStateAndBufferEquivalents \
  tests/functional_tests/test_passthrough.py::TestPersistentCarrier -q -k 'not Warp'

# The output directory must not already exist.
PYTHONPATH="$PWD" python \
  scripts/compare_numpy_aten.py /tmp/numpy-aten-new-run

# Use the original baseline with the same probe and interpreter.
git worktree add --detach /tmp/leapp-numpy-baseline \
  536883bad09b038bf4ff52db5c5c99da0e2cf950
PYTHONPATH=/tmp/leapp-numpy-baseline python \
  scripts/compare_numpy_aten.py /tmp/numpy-aten-baseline-new-run
```

The comparison is diagnostic: inspect `results.json`, not just its exit status.
The probe writes per-case results and captured FX/PT2 graphs to the supplied
output directory. Baseline and adapter must use the same dependency versions.
