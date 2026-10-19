# Python bundle API

Build from the Rust package directory, using Python 3.10 or newer in a virtual
environment and a Rust toolchain:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install maturin pytest
maturin develop --locked --manifest-path bindings/python/Cargo.toml
python -m pytest bindings/python/tests
```

```python
from leapp_runtime import Bundle, BundleError

bundle = Bundle.open("export/policy.yaml")
for name, spec in bundle.pipeline_outputs().items():
    print(name, spec.dtype, spec.shape)
```

Metadata and port descriptors are detached from the immutable Rust snapshot.
Editing returned metadata does not change later inspection results. Mappings
with unhashable or colliding Python keys become `MetadataMapping` so no entries
are lost. Custom YAML tags are unsupported and raise `BundleError` during parsing.

`BundleError` exposes `code` (an `ErrorCode`), `phase`, `path`, and `message`.
Only bundle inspection is exported. This distribution uses the same
`leapp_runtime` package name as the full runtimes, so use a separate environment.
No inference SDK or NumPy is required.
