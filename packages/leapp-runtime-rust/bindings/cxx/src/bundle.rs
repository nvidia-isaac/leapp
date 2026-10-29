// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{dtype::wire_dtype, ffi};
use lrt::{Bundle, Metadata, MetadataValue};
use std::{ffi::OsStr, os::unix::ffi::OsStrExt, path::Path};

pub struct BundleHandle(pub Option<Bundle>);

pub fn bundle_new() -> Box<BundleHandle> {
    Box::new(BundleHandle(None))
}

pub fn bundle_open(path: &[u8], output: &mut BundleHandle) -> ffi::StatusData {
    if output.0.is_some() {
        return crate::invalid("bundle destination is not empty");
    }
    match Bundle::open(Path::new(OsStr::from_bytes(path))) {
        Ok(bundle) => {
            output.0 = Some(bundle);
            crate::success()
        }
        Err(e) => crate::error(e.code(), e.phase(), e.message()),
    }
}

pub fn bundle_path(bundle: &BundleHandle) -> Vec<u8> {
    bundle
        .0
        .as_ref()
        .map(|b| b.yaml_path().as_os_str().as_bytes().to_vec())
        .unwrap_or_default()
}

pub fn bundle_models(bundle: &BundleHandle) -> Vec<String> {
    bundle
        .0
        .as_ref()
        .map(|b| b.model_names().into_iter().map(String::from).collect())
        .unwrap_or_default()
}

pub fn bundle_model_io(bundle: &BundleHandle, node: &[u8]) -> ffi::ModelIoData {
    let io = bundle
        .0
        .as_ref()
        .and_then(|bundle| bundle.model_io(std::str::from_utf8(node).ok()?));
    let present = io.is_some();
    let (inputs, outputs) = io.unwrap_or_default();
    ffi::ModelIoData {
        present,
        inputs: inputs.into_iter().map(String::from).collect(),
        outputs: outputs.into_iter().map(String::from).collect(),
    }
}

pub fn bundle_ports(bundle: &BundleHandle, inputs: bool) -> Vec<ffi::PortName> {
    let Some(b) = &bundle.0 else {
        return Vec::new();
    };
    let ports = if inputs {
        b.pipeline_inputs()
    } else {
        b.pipeline_outputs()
    };
    ports
        .iter()
        .map(|p| ffi::PortName {
            node: p.node.clone(),
            port: p.port.clone(),
        })
        .collect()
}

fn node(kind: ffi::MetadataKind) -> ffi::MetadataNode {
    ffi::MetadataNode {
        kind,
        boolean: false,
        integer: 0,
        unsigned_integer: 0,
        number: 0.,
        text: String::new(),
        children: 0,
    }
}

fn append(value: &MetadataValue, nodes: &mut Vec<ffi::MetadataNode>) {
    use ffi::MetadataKind as K;
    let mut n = node(K::Null);
    match value {
        MetadataValue::Null => {}
        MetadataValue::Bool(v) => {
            n.kind = K::Bool;
            n.boolean = *v;
        }
        MetadataValue::Number(v) => {
            if let Some(i) = v.as_i64() {
                n.kind = K::Int;
                n.integer = i;
            } else if let Some(u) = v.as_u64() {
                n.kind = K::Uint;
                n.unsigned_integer = u;
            } else {
                n.kind = K::Float;
                n.number = v.as_f64().expect("YAML number");
            }
        }
        MetadataValue::String(v) => {
            n.kind = K::String;
            n.text = v.clone();
        }
        MetadataValue::Sequence(v) => {
            n.kind = K::Sequence;
            n.children = v.len();
        }
        MetadataValue::Mapping(v) => {
            n.kind = K::Mapping;
            n.children = v.len();
        }
        MetadataValue::Tagged(_) => unreachable!("bundle loading rejects YAML tags"),
    }
    nodes.push(n);
    match value {
        MetadataValue::Sequence(v) => {
            for v in v {
                append(v, nodes);
            }
        }
        MetadataValue::Mapping(v) => {
            for (k, v) in v {
                append(k, nodes);
                append(v, nodes);
            }
        }
        _ => {}
    }
}

fn metadata(data: &Metadata) -> Vec<ffi::MetadataNode> {
    let mut root = node(ffi::MetadataKind::Mapping);
    root.children = data.len();
    let mut nodes = vec![root];
    for (key, value) in data {
        let mut key_node = node(ffi::MetadataKind::String);
        key_node.text = key.clone();
        nodes.push(key_node);
        append(value, &mut nodes);
    }
    nodes
}

pub fn bundle_metadata(
    b: &BundleHandle,
    field: ffi::MetadataField,
    name: &[u8],
) -> ffi::MetadataData {
    let name = std::str::from_utf8(name).ok();
    let data = b.0.as_ref().and_then(|b| match field {
        ffi::MetadataField::Root => Some(b.metadata()),
        ffi::MetadataField::System => Some(b.system_information()),
        ffi::MetadataField::Configs => Some(b.pipeline_configs()),
        ffi::MetadataField::Pipeline => Some(b.pipeline_metadata()),
        ffi::MetadataField::Parameters => name.and_then(|name| b.model_parameters(name)),
        ffi::MetadataField::Model => name.and_then(|name| b.model_metadata(name)),
        _ => None,
    });
    ffi::MetadataData {
        present: data.is_some(),
        nodes: data.map(metadata).unwrap_or_default(),
    }
}

pub fn bundle_port(b: &BundleHandle, node: &[u8], port: &[u8], input: bool) -> ffi::PortData {
    let spec = b.0.as_ref().and_then(|b| {
        let node = std::str::from_utf8(node).ok()?;
        let port = std::str::from_utf8(port).ok()?;
        if input {
            b.input_info(node, port)
        } else {
            b.output_info(node, port)
        }
    });
    match spec {
        Some(s) => ffi::PortData {
            present: true,
            name: s.name.clone(),
            dtype: wire_dtype(s.dtype),
            shape: s.shape.clone(),
            tensor_type: s.tensor_type.clone(),
            metadata: metadata(&s.metadata),
        },
        None => ffi::PortData {
            present: false,
            name: String::new(),
            dtype: ffi::Dtype::Float32,
            shape: Vec::new(),
            tensor_type: String::new(),
            metadata: Vec::new(),
        },
    }
}
