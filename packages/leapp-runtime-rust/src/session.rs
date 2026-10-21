// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independent session state over a shared, backend-independent engine.
pub(crate) mod graph;
mod initial_state;
#[cfg(test)]
#[path = "../tests/unit_tests/session.rs"]
mod tests;

use crate::{
    dlpack::{BorrowedTensor, SharedTensor},
    engines::{self, Engine},
    Bundle, ExecutionDevice, MemoryLocation, RuntimeError,
};
use anyhow::{ensure, Context};
use indexmap::{IndexMap, IndexSet};
use std::{collections::HashMap, sync::Arc};

/// Exact allocation locations keyed by logical output names.
#[derive(Debug, Clone, Default)]
pub struct OutputPlacement {
    pub default_location: MemoryLocation,
    pub overrides: HashMap<String, MemoryLocation>,
}
impl OutputPlacement {
    pub(crate) fn location(&self, name: &str) -> MemoryLocation {
        self.overrides
            .get(name)
            .copied()
            .unwrap_or(self.default_location)
    }
}

/// Immutable construction options, inherited by every new session.
#[derive(Debug, Clone, Default)]
pub struct SessionOptions {
    pub device: ExecutionDevice,
    pub output_placement: OutputPlacement,
}

/// Managed DLPack public outputs in bundle declaration order, independent of the session.
/// Repeated output declarations share one named entry. Cloning shares storage.
#[derive(Clone)]
pub struct RunResult {
    outputs: IndexMap<String, SharedTensor>,
}

impl RunResult {
    /// Look up an output by its qualified LEAPP `node/port` name.
    pub fn get(&self, name: &str) -> Option<&SharedTensor> {
        self.outputs.get(name)
    }

    /// Iterate through public names and tensors in declaration order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &SharedTensor)> {
        self.outputs
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }
}

/// Independent feedback state sharing loaded models with sessions from `new_session`.
/// Separate sessions can execute concurrently. Model-internal state remains shared.
pub struct Session {
    engine: Option<Arc<dyn Engine>>,
    bundle: Bundle,
    public_inputs: IndexSet<String>,
    public_outputs: IndexSet<String>,
    initial_inputs: IndexMap<String, SharedTensor>,
    inputs: IndexMap<String, SharedTensor>,
    feedback: IndexMap<String, String>,
    placement: OutputPlacement,
    state_location: MemoryLocation,
}

impl Session {
    /// Load models and capture initial values on the execution device.
    /// Dynamic state requires an exported initial value; other absent values use zeros.
    pub fn open(bundle: &Bundle, options: SessionOptions) -> Result<Self, RuntimeError> {
        let graph = graph::Graph::new(bundle).map_err(|e| RuntimeError::new("convert", e))?;
        for name in options.output_placement.overrides.keys() {
            if !bundle.pipeline_outputs().iter().any(|p| p.name() == *name) {
                return Err(RuntimeError::new(
                    "state",
                    anyhow::anyhow!("unknown public output {name}"),
                ));
            }
        }
        let engine = engines::open(bundle, &graph, &options)?;
        Self::initialize(bundle, options, &graph, engine)
    }

    fn initialize(
        bundle: &Bundle,
        options: SessionOptions,
        graph: &graph::Graph<'_>,
        engine: Arc<dyn Engine>,
    ) -> Result<Self, RuntimeError> {
        let state_location = match options.device {
            ExecutionDevice::Cpu => MemoryLocation::Cpu,
            ExecutionDevice::Cuda { device_id } => MemoryLocation::Cuda { device_id },
        };
        let initial_inputs =
            initial_state::initial_inputs(bundle, graph, state_location, engine.as_ref())?;
        let feedback = bundle
            .0
            .description
            .pipeline
            .feedback_flow
            .iter()
            .flat_map(|(source, targets)| {
                targets
                    .iter()
                    .map(move |target| (target.name(), source.name()))
            })
            .collect();
        Ok(Self {
            engine: Some(engine),
            bundle: bundle.clone(),
            public_inputs: bundle.pipeline_inputs().iter().map(|p| p.name()).collect(),
            public_outputs: bundle.pipeline_outputs().iter().map(|p| p.name()).collect(),
            inputs: initial_inputs.clone(),
            initial_inputs,
            feedback,
            placement: options.output_placement,
            state_location,
        })
    }

    /// Immutable bundle snapshot, also available after close.
    pub fn info(&self) -> &Bundle {
        &self.bundle
    }

    pub fn is_open(&self) -> bool {
        self.engine.is_some()
    }

    fn engine(&self) -> Result<&Arc<dyn Engine>, RuntimeError> {
        self.engine
            .as_ref()
            .ok_or_else(|| RuntimeError::new("execute", anyhow::anyhow!("session is closed")))
    }

    /// Create an independently initialized session without reloading models or files.
    /// Configuration and captured initial values are inherited; current feedback is not.
    pub fn new_session(&self) -> Result<Self, RuntimeError> {
        Ok(Self {
            engine: Some(self.engine()?.clone()),
            bundle: self.bundle.clone(),
            public_inputs: self.public_inputs.clone(),
            public_outputs: self.public_outputs.clone(),
            initial_inputs: self.initial_inputs.clone(),
            inputs: self.initial_inputs.clone(),
            feedback: self.feedback.clone(),
            placement: self.placement.clone(),
            state_location: self.state_location,
        })
    }

    /// Release this session's state and engine reference. Other sessions and results survive.
    pub fn close(&mut self) {
        self.inputs.clear();
        self.initial_inputs.clear();
        self.engine.take();
    }

    /// Restore captured state without rereading files or resetting model-internal state.
    pub fn reset(&mut self) -> Result<(), RuntimeError> {
        self.engine()?;
        self.inputs.clone_from(&self.initial_inputs);
        Ok(())
    }

    /// Borrow every public input exactly once until synchronous execution completes.
    /// Feedback advances only after every output and isolation copy succeeds.
    pub fn run(
        &mut self,
        inputs: &[(&str, BorrowedTensor<'_>)],
    ) -> Result<RunResult, RuntimeError> {
        self.execute(inputs)
            .map_err(|error| match error.downcast::<RuntimeError>() {
                Ok(error) => error,
                Err(error) => RuntimeError::new("execute", error),
            })
    }

    fn execute(&mut self, inputs: &[(&str, BorrowedTensor<'_>)]) -> anyhow::Result<RunResult> {
        let engine = self.engine()?;
        let mut supplied = HashMap::new();
        for &(name, tensor) in inputs {
            ensure!(
                self.public_inputs.contains(name),
                "unknown public input {name}"
            );
            ensure!(
                supplied.insert(name, tensor).is_none(),
                "duplicate public input {name}"
            );
        }
        let mut values = self
            .public_inputs
            .iter()
            .map(|name| {
                Ok((
                    name.as_str(),
                    *supplied
                        .get(name.as_str())
                        .with_context(|| format!("missing public input {name}"))?,
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        // SAFETY: private, ready state leases remain alive and are never exposed to writers.
        values.extend(
            self.inputs
                .iter()
                .map(|(name, tensor)| (name.as_str(), unsafe { tensor.borrow() })),
        );
        // Public placement applies only to public outputs. Private terminals use the default.
        let mut placement = OutputPlacement::default();
        for name in &self.public_outputs {
            placement
                .overrides
                .insert(name.clone(), self.placement.location(name));
        }
        for source in self.feedback.values() {
            placement
                .overrides
                .insert(source.clone(), self.state_location);
        }
        let outputs = engine.run(&values, &placement)?;
        let output = |name: &str| -> anyhow::Result<SharedTensor> {
            outputs.get(name).cloned().ok_or_else(|| {
                crate::error::Internal(format!("missing engine output {name}")).into()
            })
        };
        let public_outputs = self
            .public_outputs
            .iter()
            .map(|name| {
                let value = output(name)?;
                let value = if self.feedback.values().any(|source| source == name) {
                    // SAFETY: completed output storage has not been published to a foreign writer.
                    engine.copy(unsafe { value.borrow() }, self.placement.location(name))?
                } else {
                    value
                };
                Ok((name.clone(), value))
            })
            .collect::<anyhow::Result<_>>()?;
        let feedback = self
            .feedback
            .iter()
            .map(|(input, source)| Ok((input.clone(), output(source)?)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        self.inputs.extend(feedback);
        Ok(RunResult {
            outputs: public_outputs,
        })
    }
}

#[cfg(test)]
#[path = "../tests/functional_tests/triton/state.rs"]
mod native_tests;
