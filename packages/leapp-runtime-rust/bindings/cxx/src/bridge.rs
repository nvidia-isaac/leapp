// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::bundle::*;

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

    extern "Rust" {
        type BundleHandle;
        fn bundle_new() -> Box<BundleHandle>;
        fn bundle_open(path: &[u8], output: &mut BundleHandle) -> StatusData;
        fn bundle_path(bundle: &BundleHandle) -> Vec<u8>;
        fn bundle_models(bundle: &BundleHandle) -> Vec<String>;
        fn bundle_ports(bundle: &BundleHandle, inputs: bool) -> Vec<PortName>;
        fn bundle_port(bundle: &BundleHandle, node: &[u8], port: &[u8], input: bool) -> PortData;
        fn bundle_metadata(
            bundle: &BundleHandle,
            field: MetadataField,
            node: &[u8],
        ) -> MetadataData;
    }
}
