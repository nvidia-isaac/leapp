// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Convert validated bundles into Triton models, routing, and initial state.
//! Triton executes the resulting ensemble; this module does not run a node scheduler.

mod config;
#[cfg(feature = "triton")]
pub(crate) mod initial_state;
pub(crate) mod repository;
#[cfg(test)]
mod tests;

mod proto {
    include!(concat!(env!("OUT_DIR"), "/model_config/mod.rs"));
}

use crate::{
    bundle::schema::BundleDescription,
    error::{RuntimeError, Unsupported},
    Bundle, PortRef, PortSpec,
};
use anyhow::{Context, Result};
use proto::model_config::{model_ensembling::Step, ModelConfig, ModelEnsembling};
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

const MODEL_VERSION: i64 = 1;

/// Device used by model instances, independently of input and output storage.
#[derive(Debug, Clone, Copy, Default)]
pub enum ExecutionDevice {
    #[default]
    Cpu,
    Cuda {
        device_id: u32,
    },
}

#[derive(Debug, Default)]
pub(crate) struct ConversionOptions {
    pub device: ExecutionDevice,
}

#[derive(Debug)]
pub(crate) struct ArtifactRef {
    pub model_path: PathBuf,
}

#[derive(Debug)]
pub(crate) struct TritonModelSpec {
    pub config: ModelConfig,
    pub version: i64,
    pub artifact: ArtifactRef,
}

#[derive(Debug, Clone)]
pub(crate) struct PortBinding {
    pub port: PortRef,
    pub tensor_name: String,
}

#[derive(Debug, Clone)]
pub(crate) struct InitialValueRef {
    pub file: PathBuf,
    pub tensor_name: String,
}

#[derive(Debug)]
pub(crate) struct DefaultInputSpec {
    pub input: PortBinding,
    pub initial_value: Option<InitialValueRef>,
}

#[derive(Debug)]
pub(crate) struct FeedbackSpec {
    pub target: PortRef,
    pub input_name: String,
    pub output_name: String,
    pub initial_value: Option<InitialValueRef>,
}

#[derive(Debug)]
pub(crate) struct TritonPipeline {
    pub models: Vec<TritonModelSpec>,
    pub ensemble: ModelConfig,
    pub inputs: Vec<PortBinding>,
    pub outputs: Vec<PortBinding>,
    pub default_inputs: Vec<DefaultInputSpec>,
    pub feedback: Vec<FeedbackSpec>,
}

// These constraints belong to Triton conversion, not bundle inspection.
pub(crate) fn convert_bundle(
    bundle: &Bundle,
    options: &ConversionOptions,
) -> Result<TritonPipeline, RuntimeError> {
    convert(bundle, options).map_err(|source| RuntimeError::new("convert", source))
}

#[derive(Debug, PartialEq, Eq)]
enum InputSource {
    Public,
    Forward(PortRef),
    Feedback(PortRef),
}

fn assign(
    sources: &mut HashMap<PortRef, InputSource>,
    target: &PortRef,
    source: InputSource,
) -> Result<()> {
    if let Some(existing) = sources.get(target) {
        require(
            *existing == source,
            format!("multiple sources for input {target}"),
        )?;
        return Ok(());
    }
    sources.insert(target.clone(), source);
    Ok(())
}

fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if !condition {
        return Err(Unsupported(message.into()).into());
    }
    Ok(())
}

fn convert(bundle: &Bundle, options: &ConversionOptions) -> Result<TritonPipeline> {
    let (models, ports) = convert_models(bundle, options)?;
    check_graph(bundle, &ports)?;
    Ok(build_ensemble(bundle, models, &ports))
}

#[derive(Default)]
struct ModelPorts<'a> {
    inputs: HashMap<PortRef, &'a PortSpec>,
    outputs: HashMap<PortRef, &'a PortSpec>,
}

fn convert_models<'a>(
    bundle: &'a Bundle,
    options: &ConversionOptions,
) -> Result<(Vec<TritonModelSpec>, ModelPorts<'a>)> {
    let description = &bundle.0.description;
    require(
        !description.models.is_empty(),
        "Triton ensembles require a model",
    )?;
    let mut models = Vec::new();
    let mut ports = ModelPorts::default();
    let mut input_names = HashSet::new();
    let mut output_names = HashSet::new();
    for (node, model) in &description.models {
        let config = config::model(node, model, models.len(), options)?;
        index_ports(
            node,
            &model.inputs,
            "input",
            &mut ports.inputs,
            &mut input_names,
        )?;
        index_ports(
            node,
            &model.outputs,
            "output",
            &mut ports.outputs,
            &mut output_names,
        )?;
        models.push(TritonModelSpec {
            config,
            version: MODEL_VERSION,
            artifact: ArtifactRef {
                model_path: model.parameters.model_path.clone(),
            },
        });
    }
    Ok((models, ports))
}

fn check_graph(bundle: &Bundle, ports: &ModelPorts<'_>) -> Result<()> {
    let description = &bundle.0.description;
    let mut sources = HashMap::new();
    for port in bundle.pipeline_inputs() {
        ports
            .inputs
            .get(port)
            .with_context(|| format!("unknown public input {port}"))?;
        assign(&mut sources, port, InputSource::Public)?;
    }
    for (source, targets) in &description.pipeline.data_flow {
        for target in targets {
            assign(&mut sources, target, InputSource::Forward(source.clone()))?;
        }
    }
    check_forward_cycles(description)?;
    for (source, targets) in &description.pipeline.feedback_flow {
        for target in targets {
            assign(&mut sources, target, InputSource::Feedback(source.clone()))?;
        }
    }
    Ok(())
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
        outputs: Vec::new(),
        default_inputs: Vec::new(),
        feedback: Vec::new(),
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
        let output = expose_output(
            port,
            ports.outputs[port],
            &mut pipeline.ensemble,
            &mut exposed,
        );
        pipeline.outputs.push(output);
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
            let output = expose_output(
                source,
                ports.outputs[source],
                &mut pipeline.ensemble,
                &mut exposed,
            );
            pipeline.feedback.push(FeedbackSpec {
                target: input.port,
                input_name: input.tensor_name,
                output_name: output.tensor_name,
                initial_value: initial_value(bundle, target),
            });
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
                pipeline.default_inputs.push(DefaultInputSpec {
                    input,
                    initial_value: initial_value(bundle, &key),
                });
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
) -> PortBinding {
    let tensor_name = port.name();
    if exposed.insert(port.clone()) {
        ensemble.output.push(config::output(spec, &tensor_name));
    }
    PortBinding {
        port: port.clone(),
        tensor_name,
    }
}

fn port_ref(node: &str, spec: &PortSpec) -> PortRef {
    PortRef {
        node: node.into(),
        port: spec.name.clone(),
    }
}

fn index_ports<'a>(
    node: &str,
    specs: &'a [PortSpec],
    direction: &str,
    index: &mut HashMap<PortRef, &'a PortSpec>,
    names: &mut HashSet<String>,
) -> Result<()> {
    for spec in specs {
        let key = port_ref(node, spec);
        require(
            index.insert(key.clone(), spec).is_none(),
            format!("duplicate {direction} port {key}"),
        )?;
        require(
            names.insert(key.name()),
            format!("ambiguous qualified {direction} name {key}"),
        )?;
    }
    Ok(())
}

fn check_forward_cycles(description: &BundleDescription) -> Result<()> {
    let mut next = vec![Vec::new(); description.models.len()];
    let mut indegree = vec![0; next.len()];
    for (source, targets) in &description.pipeline.data_flow {
        for target in targets {
            let from = description
                .models
                .get_index_of(&source.node)
                .context("missing forward source")?;
            let to = description
                .models
                .get_index_of(&target.node)
                .context("missing forward target")?;
            next[from].push(to);
            indegree[to] += 1;
        }
    }
    // Forward dependencies are independent of declaration order. Feedback is delayed.
    let mut ready: Vec<_> = (0..next.len()).filter(|&i| indegree[i] == 0).collect();
    let mut visited = 0;
    while let Some(node) = ready.pop() {
        visited += 1;
        for &target in &next[node] {
            indegree[target] -= 1;
            if indegree[target] == 0 {
                ready.push(target);
            }
        }
    }
    require(visited == next.len(), "data_flow contains a forward cycle")
}

fn initial_value(bundle: &Bundle, target: &PortRef) -> Option<InitialValueRef> {
    if !bundle.0.initial_values.contains_key(target) {
        return None;
    }
    bundle
        .0
        .description
        .pipeline
        .initial_values
        .as_ref()
        .map(|file| InitialValueRef {
            file: file.clone(),
            tensor_name: target.name(),
        })
}
