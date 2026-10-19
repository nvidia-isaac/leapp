// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    schema::{BundleDescription, NodeParameters},
    validate, BundleData,
};
use crate::{error::in_phase, BundleError, Metadata, MetadataValue, PortRef};
use anyhow::{ensure, Context, Result};
use indexmap::IndexMap;
use serde::{
    de::{EnumAccess, MapAccess, SeqAccess, VariantAccess, Visitor},
    Deserialize, Deserializer,
};
use serde_yaml_ng::value::{Tag, TaggedValue};
use sha2::{Digest, Sha256};
use std::{
    fmt, fs, io,
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
        // Serde's flatten buffer cannot carry YAML tags. Deserialize known
        // schema fields separately, then attach the original metadata values.
        let mut schema = value.clone();
        let metadata = take_metadata(Some(&mut schema), ROOT_FIELDS);
        let pipeline_metadata = take_metadata(schema.get_mut("pipeline"), PIPELINE_FIELDS);
        if let Some(models) = schema
            .get_mut("models")
            .and_then(MetadataValue::as_mapping_mut)
        {
            for node in models.values_mut() {
                take_metadata(Some(node), MODEL_FIELDS);
                for field in ["inputs", "outputs"] {
                    if let Some(ports) =
                        node.get_mut(field).and_then(MetadataValue::as_sequence_mut)
                    {
                        for port in ports {
                            take_metadata(Some(port), PORT_FIELDS);
                        }
                    }
                }
            }
        }
        let mut description: BundleDescription = serde_path_to_error::deserialize(schema)?;
        description.metadata = metadata;
        description.pipeline.metadata = pipeline_metadata;
        for (name, node) in &mut description.models {
            let mut original = value["models"][name].clone();
            node.metadata = take_metadata(Some(&mut original), MODEL_FIELDS);
            for (field, ports) in [("inputs", &mut node.inputs), ("outputs", &mut node.outputs)] {
                for (index, port) in ports.iter_mut().enumerate() {
                    port.metadata = take_metadata(original[field].get_mut(index), PORT_FIELDS);
                }
            }
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

const ROOT_FIELDS: &[&str] = &["models", "pipeline", "system information"];
const PIPELINE_FIELDS: &[&str] = &[
    "data_flow",
    "feedback_flow",
    "inputs",
    "outputs",
    "initial_values",
    "configs",
];
const MODEL_FIELDS: &[&str] = &["inputs", "outputs", "parameters"];
const PORT_FIELDS: &[&str] = &["name", "dtype", "shape", "type"];

fn take_metadata(value: Option<&mut MetadataValue>, fields: &[&str]) -> Metadata {
    let Some(mapping) = value.and_then(MetadataValue::as_mapping_mut) else {
        // Leave malformed structure to the typed deserializer's path diagnostics.
        return Metadata::new();
    };
    let mut metadata = Metadata::new();
    mapping.retain(|key, value| {
        if let Some(key) = key.as_str() {
            if !fields.contains(&key) {
                metadata.insert(key.into(), value.clone());
                return false;
            }
        }
        true
    });
    metadata
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
enum YamlValue {
    Mapping(IndexMap<MetadataValue, YamlValue>),
    Sequence(Vec<YamlValue>),
    Scalar(MetadataValue),
    Tagged(String, Box<YamlValue>),
}

// An untagged serde enum cannot accept YAML tags. Read each value directly so
// metadata retains tags while mappings still use Python's last-value rule.
impl<'de> Deserialize<'de> for YamlValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct YamlVisitor;

        impl<'de> Visitor<'de> for YamlVisitor {
            type Value = YamlValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a YAML value")
            }

            fn visit_bool<E>(self, value: bool) -> Result<YamlValue, E> {
                Ok(YamlValue::Scalar(MetadataValue::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> Result<YamlValue, E> {
                Ok(YamlValue::Scalar(MetadataValue::Number(value.into())))
            }

            fn visit_u64<E>(self, value: u64) -> Result<YamlValue, E> {
                Ok(YamlValue::Scalar(MetadataValue::Number(value.into())))
            }

            fn visit_f64<E>(self, value: f64) -> Result<YamlValue, E> {
                Ok(YamlValue::Scalar(MetadataValue::Number(value.into())))
            }

            fn visit_str<E>(self, value: &str) -> Result<YamlValue, E> {
                Ok(YamlValue::Scalar(MetadataValue::String(value.into())))
            }

            fn visit_unit<E>(self) -> Result<YamlValue, E> {
                Ok(YamlValue::Scalar(MetadataValue::Null))
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<YamlValue, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element()? {
                    values.push(value);
                }
                Ok(YamlValue::Sequence(values))
            }

            fn visit_map<A: MapAccess<'de>>(self, mut mapping: A) -> Result<YamlValue, A::Error> {
                let mut values = IndexMap::new();
                while let Some((key, value)) = mapping.next_entry()? {
                    values.insert(key, value);
                }
                Ok(YamlValue::Mapping(values))
            }

            fn visit_enum<A: EnumAccess<'de>>(self, tagged: A) -> Result<YamlValue, A::Error> {
                let (tag, value) = tagged.variant::<String>()?;
                Ok(YamlValue::Tagged(tag, Box::new(value.newtype_variant()?)))
            }
        }

        deserializer.deserialize_any(YamlVisitor)
    }
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
            Self::Tagged(tag, value) => Ok(MetadataValue::Tagged(Box::new(TaggedValue {
                tag: Tag::new(tag),
                value: value.into_value()?,
            }))),
        }
    }
}
