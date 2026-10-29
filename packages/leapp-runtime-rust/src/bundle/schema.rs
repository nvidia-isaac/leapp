// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{ensure, Context, Result};
use indexmap::IndexMap;
use serde::{de::DeserializeOwned, Deserialize, Deserializer};
use std::{fmt, path::PathBuf};

use safetensors::Dtype as InitialValueDtype;
pub use serde_yaml_ng::Value as MetadataValue;

pub type Metadata = IndexMap<String, MetadataValue>;
pub(crate) type Connections = IndexMap<PortRef, Vec<PortRef>>;
pub(crate) type BoundaryPorts = IndexMap<String, Vec<String>>;

/// Ordered YAML fields, without artifact resolution or structural validation.
/// Kept private; loading publishes only a validated inspection snapshot.
#[derive(Debug, Deserialize)]
pub(crate) struct BundleDescription {
    #[serde(deserialize_with = "deserialize_non_null")]
    pub models: IndexMap<String, NodeSpec>,
    pub pipeline: Pipeline,
    #[serde(
        rename = "system information",
        deserialize_with = "deserialize_non_null"
    )]
    pub system_information: Metadata,
    #[serde(flatten)]
    pub metadata: Metadata,
}

#[derive(Debug, Deserialize)]
pub(crate) struct Pipeline {
    #[serde(default, deserialize_with = "deserialize_list_map")]
    pub data_flow: Connections,
    #[serde(default, deserialize_with = "deserialize_list_map")]
    pub feedback_flow: Connections,
    #[serde(deserialize_with = "deserialize_list_map")]
    pub inputs: BoundaryPorts,
    #[serde(deserialize_with = "deserialize_list_map")]
    pub outputs: BoundaryPorts,
    pub initial_values: Option<PathBuf>,
    #[serde(default, deserialize_with = "deserialize_non_null")]
    pub configs: Metadata,
    #[serde(flatten)]
    pub metadata: Metadata,
}

#[derive(Debug, Deserialize)]
pub(crate) struct NodeSpec {
    #[serde(deserialize_with = "deserialize_non_null")]
    pub inputs: Vec<PortSpec>,
    #[serde(deserialize_with = "deserialize_non_null")]
    pub outputs: Vec<PortSpec>,
    pub parameters: NodeParameters,
    #[serde(skip)]
    pub original_parameters: Metadata,
    #[serde(flatten)]
    pub metadata: Metadata,
}

#[derive(Debug, Deserialize)]
pub(crate) struct NodeParameters {
    pub backend: String,
    pub model_path: PathBuf,
    pub sha256sum: Option<String>,
}

/// Tensor interface and additional descriptive fields for one model port.
#[derive(Debug, Deserialize)]
pub struct PortSpec {
    pub name: String,
    pub dtype: DataType,
    #[serde(deserialize_with = "deserialize_shape")]
    /// Empty means scalar; 0 is an empty dimension and -1 is dynamic.
    pub shape: Vec<i64>,
    #[serde(default, rename = "type")]
    pub tensor_type: String,
    #[serde(flatten)]
    pub metadata: Metadata,
}

/// Scalar types supported in LEAPP port declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DataType {
    Float16,
    Float32,
    Float64,
    Bfloat16,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
}

/// Actual metadata retained for later bundle conversion.
#[allow(dead_code)]
#[derive(Debug)]
pub(crate) struct InitialValueSpec {
    pub dtype: InitialValueDtype,
    pub shape: Vec<i64>,
}

/// A qualified model and port identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize)]
#[serde(try_from = "String")]
pub struct PortRef {
    pub node: String,
    pub port: String,
}

impl PortRef {
    /// Qualified `node/port` name.
    pub fn name(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for PortRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.node, self.port)
    }
}

impl TryFrom<String> for PortRef {
    type Error = anyhow::Error;

    fn try_from(reference: String) -> Result<Self> {
        let (node, port) = reference
            .split_once('/')
            .with_context(|| format!("expected node/port: {reference:?}"))?;
        ensure!(!port.contains('/'), "expected node/port: {reference:?}");

        Ok(Self {
            node: node.into(),
            port: port.into(),
        })
    }
}

// Value deserialization otherwise accepts null as an empty collection.
fn deserialize_non_null<'de, D: Deserializer<'de>, T: DeserializeOwned>(
    deserializer: D,
) -> Result<T, D::Error> {
    let value = MetadataValue::deserialize(deserializer)?;
    if value.is_null() {
        return Err(serde::de::Error::custom("required field cannot be null"));
    }

    serde_path_to_error::deserialize(value).map_err(serde::de::Error::custom)
}

fn deserialize_list_map<'de, D: Deserializer<'de>, T: DeserializeOwned>(
    deserializer: D,
) -> Result<T, D::Error> {
    let value: MetadataValue = deserialize_non_null(deserializer)?;
    if let Some(map) = value.as_mapping() {
        for (key, item) in map {
            if !item.is_sequence() {
                return Err(serde::de::Error::custom(format!(
                    "{key:?}: expected a list"
                )));
            }
        }
    }

    serde_path_to_error::deserialize(value).map_err(serde::de::Error::custom)
}

fn deserialize_shape<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Vec<i64>, D::Error> {
    match deserialize_non_null(deserializer)? {
        MetadataValue::String(shape_text) => {
            serde_json::from_str(&shape_text).map_err(serde::de::Error::custom)
        }
        value => Vec::deserialize(value).map_err(serde::de::Error::custom),
    }
}
