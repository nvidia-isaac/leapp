// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{bundle::*, runtime::*, tensor::*};

#[cxx::bridge(namespace = "leapp::detail")]
pub mod ffi {
    struct StatusData {
        code: i32,
        phase: String,
        message: String,
    }

    enum Dtype {
        Bool,
        Uint8,
        Uint16,
        Uint32,
        Uint64,
        Int8,
        Int16,
        Int32,
        Int64,
        Float16,
        Bfloat16,
        Float32,
        Float64,
    }

    enum Memory {
        Cpu,
        Pinned,
        Cuda,
    }

    struct Location {
        memory: Memory,
        device: u32,
    }

    enum MetadataKind {
        Null,
        Bool,
        Int,
        Uint,
        Float,
        String,
        Sequence,
        Mapping,
    }

    // Preorder nodes carry immediate child counts; mappings alternate key/value nodes.
    struct MetadataNode {
        kind: MetadataKind,
        boolean: bool,
        integer: i64,
        unsigned_integer: u64,
        number: f64,
        text: String,
        children: usize,
    }

    struct PortData {
        present: bool,
        name: String,
        dtype: Dtype,
        shape: Vec<i64>,
        tensor_type: String,
        metadata: Vec<MetadataNode>,
    }

    struct PortName {
        node: String,
        port: String,
    }

    struct ModelIoData {
        present: bool,
        inputs: Vec<String>,
        outputs: Vec<String>,
    }

    enum MetadataField {
        Root,
        System,
        Configs,
        Pipeline,
        Parameters,
        Model,
    }

    struct MetadataData {
        present: bool,
        nodes: Vec<MetadataNode>,
    }

    struct Input {
        name: Vec<u8>,
        dtype: Dtype,
        shape: Vec<i64>,
        address: usize,
        byte_size: usize,
        location: Location,
    }

    struct Placement {
        name: Vec<u8>,
        location: Location,
    }

    struct TensorData {
        dtype: Dtype,
        shape: Vec<i64>,
        location: Location,
        byte_size: usize,
    }

    extern "Rust" {
        type BundleHandle;

        fn bundle_new() -> Box<BundleHandle>;

        fn bundle_open(path: &[u8], output: &mut BundleHandle) -> StatusData;

        fn bundle_path(bundle: &BundleHandle) -> Vec<u8>;

        fn bundle_models(bundle: &BundleHandle) -> Vec<String>;

        fn bundle_model_io(bundle: &BundleHandle, node: &[u8]) -> ModelIoData;

        fn bundle_ports(bundle: &BundleHandle, inputs: bool) -> Vec<PortName>;

        fn bundle_port(bundle: &BundleHandle, node: &[u8], port: &[u8], input: bool) -> PortData;

        fn bundle_metadata(
            bundle: &BundleHandle,
            field: MetadataField,
            node: &[u8],
        ) -> MetadataData;

        type RuntimeHandle;

        fn runtime_new() -> Box<RuntimeHandle>;

        fn runtime_open(
            bundle: &BundleHandle,
            location: Location,
            backends: &[u8],
            output: &mut RuntimeHandle,
        ) -> StatusData;

        fn runtime_close(runtime: &RuntimeHandle) -> StatusData;

        type SessionHandle;

        fn session_new() -> Box<SessionHandle>;

        fn session_create(runtime: &RuntimeHandle, output: &mut SessionHandle) -> StatusData;

        fn session_reset(session: &SessionHandle) -> StatusData;

        // C++ guarantees that all addresses reference readable allocations until return.
        unsafe fn session_run(
            session: &SessionHandle,
            inputs: &[Input],
            location: Location,
            overrides: &[Placement],
            output: &mut ResultHandle,
        ) -> StatusData;

        type ResultHandle;

        fn result_new() -> Box<ResultHandle>;

        fn result_names(result: &ResultHandle) -> Vec<String>;

        fn result_tensor(
            result: &ResultHandle,
            name: &[u8],
            output: &mut TensorHandle,
        ) -> StatusData;

        type TensorHandle;

        fn tensor_new() -> Box<TensorHandle>;

        fn tensor_info(tensor: &TensorHandle) -> TensorData;

        fn tensor_address(tensor: &TensorHandle) -> usize;

        fn tensor_cpu_bytes(tensor: &TensorHandle, bytes: &mut Vec<u8>) -> StatusData;
    }
}
