// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    schema::{BundleDescription, NodeParameters},
    validate, BundleData,
};
use crate::{error::in_phase, BundleError, MetadataValue, PortRef};
use anyhow::{ensure, Context, Result};
use indexmap::IndexMap;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};

pub(crate) fn read_bundle(path: &Path) -> Result<BundleData, BundleError> {
    let (yaml_path, yaml_text) = in_phase(path, "open", || {
        ensure!(!path.as_os_str().is_empty(), "YAML path must not be empty");
        let yaml_path = normalize(&std::path::absolute(path)?);
        let yaml_text = fs::read_to_string(&yaml_path)?;
        Ok((yaml_path, yaml_text))
    })?;
    let base = yaml_path.parent().expect("absolute path has a parent");
    let mut description = in_phase(path, "parse", || {
        // PyYAML keeps the last duplicate mapping entry and expands YAML merges.
        let value = serde_yaml_ng::from_str::<YamlValue>(&yaml_text)?.into_value()?;
        let mut description: BundleDescription = serde_path_to_error::deserialize(value.clone())?;
        for (name, node) in &mut description.models {
            node.original_parameters =
                serde_yaml_ng::from_value(value["models"][name]["parameters"].clone())?;
        }
        Ok(description)
    })?;
    in_phase(path, "validate", || {
        validate::validate_structure(&description)
    })?;
    for (name, node) in &mut description.models {
        in_phase(path, "artifact", || {
            let parameters = &mut node.parameters;
            ensure!(
                !parameters.backend.trim().is_empty(),
                "backend name must not be empty"
            );
            parameters.backend = backend_name(&parameters.backend).into();
            ensure!(
                !parameters.model_path.as_os_str().is_empty(),
                "model path must not be empty"
            );
            let beside_yaml = normalize(&base.join(&parameters.model_path));
            // Python falls back to the supplied path when no adjacent artifact exists.
            parameters.model_path = if beside_yaml.exists() {
                beside_yaml
            } else {
                normalize(&std::path::absolute(&parameters.model_path)?)
            };
            verify_artifact(parameters).with_context(|| {
                format!("node {name} artifact {}", parameters.model_path.display())
            })
        })?;
    }
    let initial_values = in_phase(path, "initial_state", || {
        if let Some(initial) = description
            .pipeline
            .initial_values
            .as_mut()
            .filter(|p| !p.as_os_str().is_empty())
        {
            *initial = normalize(&base.join(&initial));
            ensure!(
                fs::metadata(&initial)?.is_file(),
                "initial values must be a regular file"
            );
            validate::read_initial_values(initial, &description.models)
                .with_context(|| format!("pipeline.initial_values ({})", initial.display()))
        } else {
            Ok(Default::default())
        }
    })?;
    let boundary = |map: &super::schema::BoundaryPorts| {
        map.iter()
            .flat_map(|(node, ports)| {
                ports.iter().map(|port| PortRef {
                    node: node.clone(),
                    port: port.clone(),
                })
            })
            .collect()
    };
    Ok(BundleData {
        yaml_path,
        inputs: boundary(&description.pipeline.inputs),
        outputs: boundary(&description.pipeline.outputs),
        description,
        initial_values,
    })
}

// Inspection normalizes known exporter aliases without requiring an adapter.
pub(crate) fn backend_name(name: &str) -> &str {
    match name {
        "onnx" | "onnx-dynamo" | "onnx-torchscript" => "onnx",
        "jit" | "jit-script" | "jit-trace" => "jit",
        "pt2" | "exported-program" => "pt2",
        "trt" | "tensorrt" => "trt",
        other => other,
    }
}

fn verify_artifact(parameters: &NodeParameters) -> Result<()> {
    ensure!(
        fs::metadata(&parameters.model_path)?.is_file(),
        "artifact is not a regular file"
    );
    if let Some(expected) = &parameters.sha256sum {
        let mut artifact = fs::File::open(&parameters.model_path)?;
        let mut hash = Sha256::new();
        io::copy(&mut artifact, &mut hash)?;
        let actual = format!("{:x}", hash.finalize());
        ensure!(
            actual.eq_ignore_ascii_case(expected),
            "SHA256 mismatch: expected {expected}, got {actual}"
        );
    }
    Ok(())
}

fn normalize(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            other => result.push(other.as_os_str()),
        }
    }
    result
}

// Decode mappings before Value so duplicate keys follow Python's last-value rule.
// Serde's untagged deserializer rejects custom YAML tags, including nested keys.
#[derive(Deserialize)]
#[serde(untagged)]
enum YamlValue {
    Mapping(IndexMap<MetadataValue, YamlValue>),
    Sequence(Vec<YamlValue>),
    Scalar(MetadataValue),
}

impl YamlValue {
    fn into_value(self) -> Result<MetadataValue, serde_yaml_ng::Error> {
        match self {
            Self::Mapping(items) => {
                // Expand inherited maps first so chained merges retain their fields.
                let mut value = MetadataValue::Mapping(
                    items
                        .into_iter()
                        .map(|(key, value)| value.into_value().map(|value| (key, value)))
                        .collect::<Result<_, _>>()?,
                );
                value.apply_merge()?;
                Ok(value)
            }
            Self::Sequence(items) => items
                .into_iter()
                .map(Self::into_value)
                .collect::<Result<_, _>>()
                .map(MetadataValue::Sequence),
            Self::Scalar(value) => Ok(value),
        }
    }
}
