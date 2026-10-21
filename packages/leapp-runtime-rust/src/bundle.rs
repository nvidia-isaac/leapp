// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable bundle inspection, with parsing and validation in child modules.

mod load;
mod schema;
mod validate;

use crate::BundleError;
use indexmap::IndexMap;
use schema::{BundleDescription, InitialValueSpec};
pub use schema::{DataType, Metadata, MetadataValue, PortRef, PortSpec};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug)]
struct BundleData {
    yaml_path: PathBuf,
    description: BundleDescription,
    inputs: Vec<PortRef>,
    outputs: Vec<PortRef>,
    // Validated initial descriptors are retained for later bundle conversion.
    #[allow(dead_code)]
    initial_values: IndexMap<PortRef, InitialValueSpec>,
}

/// Shared ownership of an immutable, validated bundle snapshot.
///
/// Cloning shares the snapshot. Inspection borrows are tied to the bundle;
/// concurrent reads are safe. Artifact files must remain available and unchanged
/// for any subsequent execution that uses the resolved paths.
#[derive(Debug, Clone)]
pub struct Bundle(Arc<BundleData>);

impl Bundle {
    /// Parse and validate a YAML bundle, resolving artifacts beside its path.
    ///
    /// The path is made absolute and lexically normalized before reading YAML
    /// and resolving relative artifact paths. Symlinks are not canonicalized.
    /// Artifacts are sought beside the YAML first, then at the supplied path.
    ///
    /// Returns a new bundle on success and an error with a phase and diagnostic
    /// on failure. Existing bundles are never modified by an open attempt.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, BundleError> {
        load::read_bundle(path.as_ref()).map(|data| Self(Arc::new(data)))
    }

    /// Absolute YAML path, without resolving symlinks.
    pub fn yaml_path(&self) -> &Path {
        &self.0.yaml_path
    }

    /// Model names in declaration order, borrowed from this snapshot.
    pub fn model_names(&self) -> Vec<&str> {
        self.0
            .description
            .models
            .keys()
            .map(String::as_str)
            .collect()
    }

    /// Input and output names for a model, including internal ports.
    ///
    /// Names are unqualified and borrowed from this snapshot. Both lists preserve
    /// declaration order and duplicates. An absent model returns `None`; a model
    /// with no ports returns two empty lists.
    pub fn model_io(&self, node: &str) -> Option<(Vec<&str>, Vec<&str>)> {
        let model = self.0.description.models.get(node)?;
        Some((
            model.inputs.iter().map(|port| port.name.as_str()).collect(),
            model
                .outputs
                .iter()
                .map(|port| port.name.as_str())
                .collect(),
        ))
    }

    /// Declared input references in order; these need not identify existing ports.
    pub fn pipeline_inputs(&self) -> &[PortRef] {
        &self.0.inputs
    }

    /// Public output references in declaration order.
    pub fn pipeline_outputs(&self) -> &[PortRef] {
        &self.0.outputs
    }

    /// Pipeline configs; an empty mapping when absent.
    pub fn pipeline_configs(&self) -> &Metadata {
        &self.0.description.pipeline.configs
    }

    /// Additional descriptive fields at pipeline level.
    pub fn pipeline_metadata(&self) -> &Metadata {
        &self.0.description.pipeline.metadata
    }

    /// Export provenance from the `system information` mapping.
    pub fn system_information(&self) -> &Metadata {
        &self.0.description.system_information
    }

    /// Additional descriptive fields at the bundle root.
    pub fn metadata(&self) -> &Metadata {
        &self.0.description.metadata
    }

    /// Original exporter parameters, including unknown fields and original paths.
    /// Returns `None` when the model is absent.
    pub fn model_parameters(&self, node: &str) -> Option<&Metadata> {
        Some(&self.0.description.models.get(node)?.original_parameters)
    }

    /// Additional model metadata, or `None` when the model is absent.
    pub fn model_metadata(&self, node: &str) -> Option<&Metadata> {
        Some(&self.0.description.models.get(node)?.metadata)
    }

    /// The first input descriptor with this name, including internal ports.
    /// Returns `None` if absent; duplicate declarations are permitted.
    pub fn input_info(&self, node: &str, port: &str) -> Option<&PortSpec> {
        self.0
            .description
            .models
            .get(node)?
            .inputs
            .iter()
            .find(|p| p.name == port)
    }

    /// The first output descriptor with this name, including internal ports.
    /// Returns `None` if absent; duplicate declarations are permitted.
    pub fn output_info(&self, node: &str, port: &str) -> Option<&PortSpec> {
        self.0
            .description
            .models
            .get(node)?
            .outputs
            .iter()
            .find(|p| p.name == port)
    }
}
