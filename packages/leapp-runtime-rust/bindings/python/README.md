# Python bindings

PyO3 bindings to the Rust bundle and synchronous runtime APIs. Linux x86-64
requires the Triton and CUDA runtime SDKs. Both Triton and native execution
are included in the same extension.
NumPy is a package dependency; PyTorch and CuPy are optional DLPack
producers/consumers.

Build with the workspace's Rust toolchain and Python 3.10 or newer:

```sh
python -m pip install maturin
maturin build --release --manifest-path bindings/python/Cargo.toml --auditwheel skip
python -m pip install target/wheels/leapp_runtime-*.whl
```

Install the model backends you need as described in the
[native engine guide](../../README.md#native-engine). ONNX supports CPU and CUDA execution.

Set `TRITON_ROOT` and `CUDA_ROOT` when their SDKs are not in
`/opt/tritonserver` and `/usr/local/cuda`. Make the SDK and backend dependencies
available through `LD_LIBRARY_PATH`. The wheel deliberately does not redistribute
NVIDIA libraries. CUDA support is always included; CPU execution and CPU buffers
remain available in the same extension.

```python
import numpy as np
import lrt as lr

bundle = lr.Bundle.open("exports/policy.yaml")
options = lr.SessionOptions(device="cuda:0", output_device="cpu")
with lr.Session.open(bundle, options) as session:
    result = session.run(
        {"preprocess/obs": np.zeros(18, dtype=np.float32)},
    )
    action = np.from_dlpack(result["policy/action"])
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

Inputs must implement `__dlpack__` and `__dlpack_device__`. NumPy, PyTorch,
CuPy, and returned `DLPackOutput` objects use the same protocol path. Legacy
capsules and DLPack 1.0 versioned capsules are supported. Use producer versions
that implement this protocol; older array interfaces and framework-specific
fallbacks are not supported.

All inputs must be C-contiguous with supported native-endian numeric dtypes.
Materialize noncontiguous inputs explicitly, for example `np.ascontiguousarray(x)`,
`x.contiguous()` in PyTorch, or `cp.ascontiguousarray(x)`. Read-only inputs are
accepted when their producer can export them through DLPack; producer export
errors are propagated. Input leases remain alive until the synchronous engine call completes.
Do not modify or resize input storage during a call.

CUDA producers hand work to an engine-owned stream per device. After importing
the batch, each participating stream is synchronized once before inference.
Producers and capsules remain alive through these waits, including on import
failure. Outputs are ready on return.

Results expose minimal `DLPackOutput` objects with only the two DLPack protocol
methods. Use `np.from_dlpack(result[name])`, `torch.from_dlpack(result[name])`, or
`cp.from_dlpack(result[name])`. Consumers control array APIs, copies, and transfers.
Exports share output storage; repeated exports may alias. Outputs also used as
feedback are copied to their configured public location before publication, so consumer mutation
cannot change the next inference or reset state. NumPy may return a read-only view;
use its `copy()` when a writable NumPy array is required.

Exports support `copy=False` and `copy=None`. Explicit `copy=True` and cross-device
export requests raise `BufferError`; perform those operations in the consumer.
There is no `Tensor`, `DLPackView`, `.numpy()`, or `.dlpack_view()` convenience API.
The session may recycle unexposed storage after all owners release it. Descriptor
access or DLPack export permanently excludes that allocation from reuse. Retain the
consumer's DLPack owner until all its uses complete, including work on CUDA streams.

`Session.run`, loading, reset and close release the GIL while doing native work.
Calls on one session serialize. Separate sessions may be called concurrently;
native peers own separate fixed buffers/bindings and share loaded models and weights.
ONNX peers can overlap. Graphs containing optional ONNX custom operators retain
a shared execution guard.
`session.new_session()` shares loaded models and construction options, but
starts from captured initial state rather than the source's current feedback.
`Session.close()` waits for that session's active call and is idempotent. Other
sessions remain usable. Results, arrays, DLPack allocations and bundle metadata
remain valid after close. Context-manager exit closes only its session.

`SessionOptions` selects the engine and output placement at construction;
`run` takes only inputs. The default is `engine="triton"`; set `engine` to
`"triton"` or `"native"` to select either engine in the same extension. Unknown
engine names raise `ValueError` when opening the session.
Set `output_device` to `"cpu"`, `"pinned"`, or `"cuda:N"` and use `output_overrides`
for individual public outputs. Configure the engine installation before starting
Python, as described in the
[runtime guide](../../README.md#synchronous-execution).

Set `output_pool=lr.OutputPoolOptions(max_entries=64)` to configure retained
buffer count; zero disables caching. Each session has its own cache, including
peers from `new_session()`. Its byte budget follows bundle output sizes and grows
for larger dynamic outputs, with no fixed byte cap. Caching does not limit live
results, and retained outputs survive close.

Bundle and execution failures raise `BundleError` and `RuntimeError` with
`code`, `phase`, and `message`; bundle errors also include `path`.
Python argument/protocol failures use standard Python exception types.

## Tests

Generate ONNX fixtures with `tests/generators/generate_models.py`, then run from the Rust
package directory:

```sh
LEAPP_TRITON_FIXTURES=target/triton-fixtures \
LEAPP_TRITON_BACKENDS=target/public-api-backends \
LEAPP_TEST_CUDA=1 python -m pytest bindings/python/tests
```

Install pytest, PyTorch and CuPy for the complete interoperability suite.
C++ bindings are a separate crate and are not needed to use this package.
