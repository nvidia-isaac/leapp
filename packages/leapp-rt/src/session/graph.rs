// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{bundle::schema::BundleDescription, error::Unsupported, Bundle, PortRef, PortSpec};
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};

pub(crate) struct Graph<'a> {
    pub inputs: HashMap<PortRef, &'a PortSpec>,
    pub outputs: HashMap<PortRef, &'a PortSpec>,
    pub sources: HashMap<PortRef, InputSource>,
}
impl<'a> Graph<'a> {
    pub fn new(bundle: &'a Bundle) -> Result<Self> {
        let mut graph = Self {
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            sources: HashMap::new(),
        };
        let (mut inputs, mut outputs) = (HashSet::new(), HashSet::new());
        for (node, model) in &bundle.0.description.models {
            index_ports(node, &model.inputs, "input", &mut graph.inputs, &mut inputs)?;
            index_ports(
                node,
                &model.outputs,
                "output",
                &mut graph.outputs,
                &mut outputs,
            )?;
        }
        graph.sources = check_graph(bundle, &graph)?;
        Ok(graph)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum InputSource {
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

pub(crate) fn require(condition: bool, message: impl Into<String>) -> Result<()> {
    if !condition {
        return Err(Unsupported(message.into()).into());
    }
    Ok(())
}

pub(crate) fn port_ref(node: &str, spec: &PortSpec) -> PortRef {
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

fn check_graph(bundle: &Bundle, ports: &Graph<'_>) -> Result<HashMap<PortRef, InputSource>> {
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
    Ok(sources)
}
