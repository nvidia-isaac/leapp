# DLPack ABI and ownership

This runtime module contains the DLPack ABI, lifetime guards, and a thin CPU-slice
interface. It contains no allocation backend or framework conversion API.
`BorrowedTensor::cpu` checks shape and dtype while retaining the Rust slice borrows.
`SharedTensor::to_cpu_vec` validates the layout and returns an owned CPU copy of
CPU, pinned CPU, or CUDA outputs. The engine supplies the synchronous transfer
callback; CUDA selection and synchronization stay in the engine. The method
synchronizes with first exposure and rejects previously exposed storage, preventing
races with writable consumers without extending the foreign ownership protocol.

The runtime compiles this module once as `lrt::dlpack`. Every engine
uses these same ownership types; no separate crate or re-export is needed.
Language bindings use the hidden `HandoffStream` support type for producer
synchronization. Its native implementation stays behind the private engine
module. See [runtime installation](../../README.md#installation) for build requirements.

`include/dlpack/dlpack.h` is the unmodified upstream v1.0 header:
https://github.com/dmlc/dlpack/blob/v1.0/include/dlpack/dlpack.h

`ManagedTensor` owns one producer lease and stays on the importing thread. `BorrowedTensor`
retains a descriptor borrow under an explicit unsafe readiness/lifetime contract.
`SharedTensor` is for engine-produced storage with thread-independent deletion;
constructing it from arbitrary producer storage requires an explicit unsafe
contract. Its exports create new leases sharing the same allocation. Unsafe
`SharedTensor::borrow()` provides a bounded internal borrow; pointers must not
escape it, and all work must finish before the borrow ends.

`OutputBuffer` borrows exclusive caller storage for synchronous output writes.
Its safe CPU constructor retains a mutable slice borrow; the unsafe DLPack constructor
requires valid writable storage. It never transfers ownership or retains a caller
allocation after inference returns.

Regenerate checked-in Rust ABI declarations with Python, clang, and rustfmt:

```sh
python3 src/dlpack/generate_bindings.py
cargo test --locked --test dlpack
```

The tests compare Rust layouts with the actual C header, then exercise deletion,
ownership transfer, legacy/versioned exports, partial failure, and retained metadata.
On upgrades, update the pinned header and generator together, review new enum/flag
semantics, adjust the protocol version actually supported, and repeat Python/C++
interoperability and CUDA tests. Normal builds do not run the generator.
