// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::schema::{BundleDescription, InitialValueSpec, NodeSpec, PortRef};
use anyhow::{ensure, Context, Result};
use indexmap::IndexMap;
use safetensors::SafeTensors;
use std::{collections::HashSet, fs, path::Path};

pub(crate) fn validate_structure(description: &BundleDescription) -> Result<()> {
    let mut inputs = IndexMap::new();
    let mut outputs = IndexMap::new();
    for (name, node) in &description.models {
        // Python input buffers keep the last duplicate; edge sources use the first.
        for spec in &node.inputs {
            inputs.insert(
                PortRef {
                    node: name.clone(),
                    port: spec.name.clone(),
                },
                spec,
            );
        }
        for spec in &node.outputs {
            outputs
                .entry(PortRef {
                    node: name.clone(),
                    port: spec.name.clone(),
                })
                .or_insert(spec);
        }
    }

    for flow in [
        &description.pipeline.data_flow,
        &description.pipeline.feedback_flow,
    ] {
        for (source, targets) in flow {
            let source_spec = outputs
                .get(source)
                .with_context(|| format!("unknown output port {source}"))?;
            for target in targets {
                let target_spec = inputs
                    .get(target)
                    .with_context(|| format!("unknown input port {target}"))?;
                ensure!(
                    source_spec.shape == target_spec.shape
                        && source_spec.dtype == target_spec.dtype,
                    "incompatible tensors: {source} -> {target}"
                );
            }
        }
    }

    // Python inspects published outputs during loading, but not external inputs.
    let mut routed = HashSet::new();
    for (node, names) in &description.pipeline.outputs {
        ensure!(
            description.models.contains_key(node),
            "boundary: unknown node {node}"
        );
        for name in names {
            let key = PortRef {
                node: node.clone(),
                port: name.clone(),
            };
            ensure!(outputs.contains_key(&key), "unknown output port {key}");
            routed.insert(key);
        }
    }
    routed.extend(description.pipeline.data_flow.keys().cloned());
    routed.extend(description.pipeline.feedback_flow.keys().cloned());
    for output in outputs.keys() {
        ensure!(routed.contains(output), "output is not routed: {output}");
    }
    Ok(())
}

pub(crate) fn read_initial_values(
    path: &Path,
    models: &IndexMap<String, NodeSpec>,
) -> Result<IndexMap<PortRef, InitialValueSpec>> {
    let bytes = fs::read(path)?;
    // Safetensors owns JSON header and payload-layout validation.
    let (_, metadata) = SafeTensors::read_metadata(&bytes)?;

    let mut initial_values = IndexMap::new();
    for name in metadata.offset_keys() {
        let key = PortRef::try_from(name.clone())?;
        models
            .get(&key.node)
            .and_then(|node| node.inputs.iter().find(|port| port.name == key.port))
            .with_context(|| format!("unknown input port {key}"))?;

        let tensor = metadata
            .info(&name)
            .context("missing initial tensor metadata")?;
        let shape = tensor
            .shape
            .iter()
            .copied()
            .map(i64::try_from)
            .collect::<Result<Vec<_>, _>>()?;
        initial_values.insert(
            key,
            InitialValueSpec {
                dtype: tensor.dtype,
                shape,
            },
        );
    }

    // Return file metadata in YAML node/input order.
    let mut ordered = IndexMap::new();
    for (node, model) in models {
        for port in &model.inputs {
            let key = PortRef {
                node: node.clone(),
                port: port.name.clone(),
            };
            if let Some(spec) = initial_values.shift_remove(&key) {
                ordered.insert(key, spec);
            }
        }
    }

    Ok(ordered)
}
