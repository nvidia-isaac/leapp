// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context models that load prebuilt TensorRT engines through ONNX Runtime.
//!
//! ORT's TensorRT provider deserializes the engine referenced by a single
//! `com.microsoft::EPContext` node instead of building one from ONNX. With
//! `embed_mode=0` it reads `ep_cache_context` relative to the model and rejects
//! absolute paths and `..`, so the model is written to a private directory beside
//! a link to the engine; embedding would copy the engine into a protobuf limited
//! to 2 GiB. The provider binds engine tensors by name, so the graph declares
//! the bundle's port names, dtypes and shapes.

use super::ort_type;
use crate::{
    bundle::{schema::NodeSpec, shapes::ExecutionShape},
    PortSpec,
};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::symlink;

#[cfg(not(unix))]
fn symlink(_: impl AsRef<Path>, _: impl AsRef<Path>) -> std::io::Result<()> {
    Err(std::io::ErrorKind::Unsupported.into())
}

/// Write a context model for `engine`; keep the directory until ORT loads it.
pub(super) fn context_model(
    model: &NodeSpec,
    engine: &Path,
) -> std::io::Result<(tempfile::TempDir, PathBuf)> {
    let engine = engine.canonicalize()?;
    let directory = tempfile::Builder::new()
        .prefix("leapp-tensorrt-")
        .tempdir()?;
    symlink(&engine, directory.path().join("engine"))?;
    let path = directory.path().join("model.onnx");
    std::fs::write(&path, encode(model))?;
    Ok((directory, path))
}

/// Serialize an ONNX ModelProto. Field numbers follow onnx/onnx.proto.
fn encode(model: &NodeSpec) -> Vec<u8> {
    // AttributeProto: name 1, i 3, s 4, and type 20 (INT = 2, STRING = 3).
    let attribute = |name: &str| Message::default().bytes(1, name);
    let mut node = Message::default()
        .bytes(4, "EPContext")
        .bytes(7, "com.microsoft")
        .bytes(5, attribute("embed_mode").varint(3, 0).varint(20, 2))
        .bytes(
            5,
            attribute("ep_cache_context")
                .bytes(4, "engine")
                .varint(20, 3),
        );
    let mut graph = Message::default().bytes(2, "leapp_tensorrt_engine");
    for (port, shape) in model.input_ports() {
        node = node.bytes(1, &port.name);
        graph = graph.bytes(11, value_info(port, shape));
    }
    for (port, shape) in model.output_ports() {
        node = node.bytes(2, &port.name);
        graph = graph.bytes(12, value_info(port, shape));
    }
    Message::default()
        .varint(1, 9)
        .bytes(7, graph.bytes(1, node))
        .bytes(8, Message::default().bytes(1, "").varint(2, 17))
        .bytes(8, Message::default().bytes(1, "com.microsoft").varint(2, 1))
        .0
}

/// ValueInfoProto for a tensor; negative dimensions stay unknown.
fn value_info(port: &PortSpec, shape: &ExecutionShape) -> Message {
    let mut dims = Message::default();
    for &size in &shape.dims {
        let dim = Message::default();
        dims = dims.bytes(
            1,
            if size < 0 {
                dim
            } else {
                dim.varint(1, size as u64)
            },
        );
    }
    let elem_type = ort::sys::ONNXTensorElementDataType::from(ort_type(port.dtype)) as u64;
    let tensor = Message::default().varint(1, elem_type).bytes(2, dims);
    Message::default()
        .bytes(1, &port.name)
        .bytes(2, Message::default().bytes(1, tensor))
}

/// Minimal protobuf writer for non-negative varints and length-delimited fields.
#[derive(Default)]
struct Message(Vec<u8>);

impl AsRef<[u8]> for Message {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl Message {
    fn varint(mut self, field: u32, value: u64) -> Self {
        self.raw(u64::from(field) << 3);
        self.raw(value);
        self
    }

    fn bytes(mut self, field: u32, value: impl AsRef<[u8]>) -> Self {
        let value = value.as_ref();
        self.raw((u64::from(field) << 3) | 2);
        self.raw(value.len() as u64);
        self.0.extend_from_slice(value);
        self
    }

    fn raw(&mut self, mut value: u64) {
        while value >= 0x80 {
            self.0.push((value as u8) | 0x80);
            value >>= 7;
        }
        self.0.push(value as u8);
    }
}
