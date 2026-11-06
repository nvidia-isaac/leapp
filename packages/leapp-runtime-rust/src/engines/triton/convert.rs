// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Convert validated bundles into Triton models and ensemble routing.
//! Triton executes the resulting ensemble; this module does not run a node scheduler.

mod config;
pub(crate) mod repository;
#[cfg(test)]
#[path = "../../../tests/unit_tests/triton/convert.rs"]
mod tests;

mod proto {
    include!(concat!(env!("OUT_DIR"), "/model_config/mod.rs"));
}

use crate::{
    session::graph::{port_ref, require, Graph as ModelPorts},
    Bundle, ExecutionDevice, PortRef, PortSpec,
};
use anyhow::Result;
use proto::model_config::{model_ensembling::Step, ModelConfig, ModelEnsembling};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

const MODEL_VERSION: i64 = 1;

#[derive(Debug)]
pub(crate) struct TritonModelSpec {
    pub config: ModelConfig,
    pub version: i64,
    pub model_path: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct PortBinding {
    pub port: PortRef,
    pub tensor_name: String,
}

#[derive(Debug)]
pub(crate) struct TritonPipeline {
    pub models: Vec<TritonModelSpec>,
    pub ensemble: ModelConfig,
    pub inputs: Vec<PortBinding>,
}

// These constraints belong to Triton conversion, not bundle inspection.
#[cfg(test)]
pub(crate) fn convert_bundle(
    bundle: &Bundle,
    device: ExecutionDevice,
) -> Result<TritonPipeline, crate::RuntimeError> {
    (|| convert(bundle, device, &ModelPorts::new(bundle)?))()
        .map_err(|source| crate::RuntimeError::new("convert", source))
}

pub(super) fn convert(
    bundle: &Bundle,
    device: ExecutionDevice,
    ports: &ModelPorts<'_>,
) -> Result<TritonPipeline> {
    require(
        !bundle.0.description.models.is_empty(),
        "Triton ensembles require a model",
    )?;
    let models = bundle
        .0
        .description
        .models
        .iter()
        .enumerate()
        .map(|(index, (node, model))| {
            Ok(TritonModelSpec {
                config: config::model(node, model, index, device)?,
                version: MODEL_VERSION,
                model_path: model.parameters.model_path.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(build_ensemble(bundle, models, ports))
}

fn build_ensemble(
    bundle: &Bundle,
    models: Vec<TritonModelSpec>,
    ports: &ModelPorts<'_>,
) -> TritonPipeline {
    let description = &bundle.0.description;
    let mut pipeline = TritonPipeline {
        models,
        ensemble: ModelConfig {
            name: "leapp_pipeline".into(),
            platform: "ensemble".into(),
            ..Default::default()
        },
        inputs: Vec::new(),
    };
    // Reserve producer names before assigning names to external inputs.
    let mut tensor_names = ports.outputs.keys().map(PortRef::name).collect();
    let mut input_names = HashMap::new();
    let mut exposed = HashSet::new();
    for (source, targets) in &description.pipeline.data_flow {
        for target in targets {
            input_names.insert(target.clone(), source.name());
        }
    }
    for port in bundle.pipeline_inputs() {
        let input = expose_input(
            port,
            ports.inputs[port],
            &mut pipeline.ensemble,
            &mut input_names,
            &mut tensor_names,
        );
        pipeline.inputs.push(input);
    }
    for port in bundle.pipeline_outputs() {
        expose_output(
            port,
            ports.outputs[port],
            &mut pipeline.ensemble,
            &mut exposed,
        );
    }
    for (source, targets) in &description.pipeline.feedback_flow {
        for target in targets {
            // Graph checks exclude competing sources; an existing binding is a repeated edge.
            if input_names.contains_key(target) {
                continue;
            }
            let input = expose_input(
                target,
                ports.inputs[target],
                &mut pipeline.ensemble,
                &mut input_names,
                &mut tensor_names,
            );
            expose_output(
                source,
                ports.outputs[source],
                &mut pipeline.ensemble,
                &mut exposed,
            );
            pipeline.inputs.push(input);
        }
    }
    add_steps(
        bundle,
        &mut pipeline,
        &mut input_names,
        &mut tensor_names,
        &mut exposed,
    );
    pipeline
}

fn add_steps(
    bundle: &Bundle,
    pipeline: &mut TritonPipeline,
    input_names: &mut HashMap<PortRef, String>,
    tensor_names: &mut HashSet<String>,
    exposed: &mut HashSet<PortRef>,
) {
    let description = &bundle.0.description;
    let mut steps = Vec::new();
    for (index, (node, model)) in description.models.iter().enumerate() {
        let mut step = Step {
            model_name: pipeline.models[index].config.name.clone(),
            model_version: MODEL_VERSION,
            ..Default::default()
        };
        for (port, native) in model
            .inputs
            .iter()
            .zip(&pipeline.models[index].config.input)
        {
            let key = port_ref(node, port);
            if !input_names.contains_key(&key) {
                let input = expose_input(
                    &key,
                    port,
                    &mut pipeline.ensemble,
                    input_names,
                    tensor_names,
                );
                pipeline.inputs.push(input);
            }
            step.input_map
                .insert(native.name.clone(), input_names[&key].clone());
        }
        for (port, native) in model
            .outputs
            .iter()
            .zip(&pipeline.models[index].config.output)
        {
            let key = port_ref(node, port);
            step.output_map.insert(native.name.clone(), key.name());
            // Expose terminal outputs so private branches can also be requested.
            if description
                .pipeline
                .data_flow
                .get(&key)
                .is_none_or(Vec::is_empty)
            {
                expose_output(&key, port, &mut pipeline.ensemble, exposed);
            }
        }
        steps.push(step);
    }
    pipeline.ensemble.set_ensemble_scheduling(ModelEnsembling {
        step: steps,
        ..Default::default()
    });
}

fn expose_input(
    port: &PortRef,
    spec: &PortSpec,
    ensemble: &mut ModelConfig,
    bindings: &mut HashMap<PortRef, String>,
    names: &mut HashSet<String>,
) -> PortBinding {
    let tensor_name = bindings.entry(port.clone()).or_insert_with(|| {
        let mut name = port.name();
        let mut suffix = ensemble.input.len();
        // An input and output may share a LEAPP name; their ensemble tensors cannot.
        while !names.insert(name.clone()) {
            name = format!("__leapp_input_{suffix}");
            suffix += 1;
        }
        ensemble.input.push(config::input(spec, &name));
        name
    });
    PortBinding {
        port: port.clone(),
        tensor_name: tensor_name.clone(),
    }
}

fn expose_output(
    port: &PortRef,
    spec: &PortSpec,
    ensemble: &mut ModelConfig,
    exposed: &mut HashSet<PortRef>,
) {
    let tensor_name = port.name();
    if exposed.insert(port.clone()) {
        ensemble.output.push(config::output(spec, &tensor_name));
    }
}
