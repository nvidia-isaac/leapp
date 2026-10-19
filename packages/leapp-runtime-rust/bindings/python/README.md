# Python bindings

PyO3 bindings to the Rust bundle and synchronous runtime APIs. Linux x86-64
requires the Triton SDK, CUDA runtime, and selected backend libraries at runtime. NumPy is a package dependency; PyTorch and CuPy are
optional DLPack producers/consumers.

Build with the workspace's Rust toolchain and Python 3.10 or newer:

```sh
python -m pip install maturin
maturin build --release --manifest-path bindings/python/Cargo.toml --auditwheel skip
python -m pip install target/wheels/leapp_runtime-*.whl
```

Set `TRITON_ROOT` and `CUDA_ROOT` when their SDKs are not in
`/opt/tritonserver` and `/usr/local/cuda`. Make the SDK and backend dependencies
available through `LD_LIBRARY_PATH`. The wheel deliberately does not redistribute
NVIDIA libraries. CUDA support is always included; CPU execution and CPU buffers
remain available in the same extension.

```python
import numpy as np
import leapp_runtime as lr

bundle = lr.Bundle.open("exports/policy.yaml")
options = lr.RuntimeOptions(device="cuda:0")
with lr.Runtime.open(bundle, options) as runtime:
    session = runtime.create_session()
    result = session.run(
        {"preprocess/obs": np.zeros(18, dtype=np.float32)},
        output_device="cpu",
    )
    action = result.numpy("policy/action")
    mutable_action = action.copy()
    session.reset()
```

Bundle getters return detached Python values. Pipeline input/output getters are
ordered dictionaries of qualified names to port descriptors. Missing model or
port lookups return `None`. Unhashable or colliding YAML mapping keys are retained in
`MetadataMapping.entries`. Custom YAML tags are unsupported and fail during parsing.

`model_io(model_name)` returns detached input and output name lists, including
internal ports. Names are unqualified, and declaration order and duplicates are
preserved. An absent model returns `None`; a model with no ports returns two
empty lists. Use `input_info` and `output_info` to inspect those ports.
Unresolved pipeline input descriptors are represented by `None`.

Inputs may be NumPy arrays, PyTorch tensors, CuPy arrays, LEAPP tensors, or
other DLPack producers.
Input owners are retained until synchronous inference completes. Do not modify
or resize their storage during a call. Every call supplies all public inputs.
Noncontiguous NumPy, PyTorch, and CuPy inputs are copied to contiguous storage
on their existing device only when necessary. Contiguous inputs remain borrowed;
temporary copies are retained through inference. Other DLPack producers must
provide C-contiguous tensors: DLPack has no standard contiguous-copy operation,
so materialize those inputs explicitly before calling `run`.
CUDA producer work is handed to a private stream and completed before Triton
reads the buffer; this includes nondefault producer streams.

CPU and pinned outputs support `result.numpy(name)` or `tensor.numpy()`.
These are zero-copy, read-only NumPy arrays retaining their allocation owner;
`array.copy()` provides mutable storage. CUDA tensors require explicit host
transfer through a consumer such as CuPy.

DLPack exports **copy by default**, retaining the original device:

```python
import cupy as cp

with lr.Runtime.open(bundle, options) as runtime:
    session = runtime.create_session()
    result = session.run(inputs, output_device="cuda:0")
    action = cp.from_dlpack(result["policy/action"])  # independent mutable storage
    shared = cp.from_dlpack(result["policy/action"].dlpack_view())
```

`dlpack_view()` explicitly acknowledges the read-only sharing contract.
Versioned capsules carry DLPack's read-only flag, but consumers may ignore it;
legacy capsules cannot express it. **Never mutate this shared allocation**:
it can also hold feedback state or data retained by other results. There is no
copy-on-write interception. Ordinary `Tensor.__dlpack__(copy=False)` rejects
the request and points to `dlpack_view()`; the explicit view accepts
`copy=True` for an isolated copy. CPU and CUDA exports support legacy and
versioned capsules. Cross-device export requests fail rather than silently move
data. Native outputs and export copies are complete before publication.

`Session.run`, loading, reset and close release the GIL while doing native work.
Calls on one session serialize; separate sessions may run concurrently.
`Runtime.close()` waits for active calls, closes all associated sessions, and is
idempotent. Further session calls fail. Results, arrays, DLPack allocations and
bundle metadata remain valid after close. Context-manager exit calls close.

Bundle and execution failures raise `BundleError` and `RuntimeError` with
`code`, `phase`, and `message`; bundle errors also include `path`.
Python argument/protocol failures use standard Python exception types.

## Tests

Generate ONNX fixtures with `tests/generate_models.py`, then run from the Rust
package directory:

```sh
LEAPP_TRITON_FIXTURES=target/triton-fixtures \
LEAPP_TRITON_BACKENDS=target/public-api-backends \
LEAPP_TEST_CUDA=1 python -m pytest bindings/python/tests
```

Install pytest, PyTorch and CuPy for the complete interoperability suite.
C++ bindings will be a separate crate and are not needed to use this package.
