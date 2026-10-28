<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native backend plugins

The native engine uses [`BackendPlugin`](../backends.rs) to execute model nodes
with ONNX, TensorRT, TorchScript, or NativeRT. The engine handles graph routing,
workspace buffers, and public results; each plugin handles model execution.
This is an internal contributor interface. Applications use the runtime's
Session API, and Triton has its own backend interface.

## Usage and lifecycle

The engine selects a backend from each node's bundle configuration, validates
support, and loads the model. Each workspace has its own plugin instance and
buffers. Instances created through `new_session()` share loaded models, weights,
and model-owned state. Execution-context state is private to each instance and
starts fresh in each peer.

Plugins support two execution paths:

- **Prepared:** bind fixed-address input and output buffers once, then call
  `run()` repeatedly. Buffer contents change between runs; bindings and tensor
  metadata are reused to minimize overhead.
- **Dynamic:** pass inputs to `run_dynamic()` on each call and receive owned
  outputs. This path also supports models with fixed shapes when preparation
  is unavailable.

To add a backend, implement `BackendPlugin` and register its validation and
loading functions in [`backends.rs`](../backends.rs). ONNX implements the trait
in Rust; C++ backends can reuse the [shared bridge](bridge.rs). See the
[runtime README](../../../../README.md) for build and configuration instructions.

## Interface contract

| Method | Responsibility | Default |
| --- | --- | --- |
| `new_session()` | Create an unprepared peer sharing the loaded model and weights, without reloading artifacts or copying bindings or execution-context state. | Required |
| `needs_guard()` | Indicate whether execution must be serialized across peers. The decision must remain stable across all related instances. | `true` |
| `can_prepare()` | Report whether fixed bindings are supported. This permits a preparation attempt; it does not guarantee success. | `false` |
| `is_prepared()` | Report whether this instance has valid prepared bindings. New instances start unprepared. | `false` |
| `prepare(inputs, outputs)` | Bind fixed buffers and cache reusable metadata. On success, `is_prepared()` must become `true`. | Unsupported |
| `run()` | Execute with prepared bindings and current input contents, writing into bound outputs. Return an error if unprepared. | Unsupported |
| `run_dynamic(inputs, transfers)` | Execute with borrowed inputs and return independently owned outputs. The workspace transfer helper is available for copies. | Required |

A minimal backend implements `new_session()` and `run_dynamic()`. Supporting
prepared execution requires all four preparation-related methods:
`can_prepare()`, `is_prepared()`, `prepare()`, and `run()`.

All implementations must honor these guarantees:

- **Tensor compatibility:** inputs and outputs follow declared port order,
  dtype, shape, and device. Inputs are read-only.
- **Completion:** finish all work accessing borrowed storage before returning,
  including on errors or unwinding. Results must be ready for the engine to use.
- **Buffer lifetime:** prepared buffers remain at fixed addresses until the
  plugin instance is destroyed. Retained metadata must be owned or copied, and
  partially prepared instances must be safe to destroy. Preparation failure
  aborts workspace creation.
- **Output ownership:** dynamic outputs survive later calls and plugin
  destruction, without aliasing inputs or mutable backend state.
- **Concurrency and state:** instances are `Send` and called exclusively, but
  may move between threads. Disable the guard only when peers can safely execute
  concurrently. Peer creation must also be safe during another peer's execution,
  and shared model state must be protected from concurrent mutation.

`prepare()` is unsafe because it retains borrowed storage. Its full caller
obligations, including aliasing and access restrictions, are documented on the
[trait](../backends.rs).
