// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Independent session state over a shared, backend-independent engine.
pub(crate) mod graph;
mod initial_state;
pub(crate) mod onnx_tensorrt;
#[cfg(test)]
#[path = "../tests/unit_tests/session.rs"]
mod tests;

use crate::{
    dlpack::{BorrowedTensor, OutputBuffer, SharedTensor},
    engines::{self, Engine},
    Bundle, EngineKind, ExecutionDevice, MemoryLocation, RuntimeError,
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
    /// Select the engine once at initialization; new sessions inherit that selection.
    /// Defaults to native.
    pub engine: EngineKind,
    /// ONNX node names mapped to Triton's TensorRT accelerator parameters.
    /// An entry enables the provider with CUDA fallback; an empty map uses SDK defaults.
    /// See documentation for supported options.
    pub onnx_tensorrt: HashMap<String, HashMap<String, String>>,
    /// Caller-owned streams by CUDA device ID, used by native GPU execution and
    /// transfers. Missing devices use internal streams. Triton ignores this map.
    /// Peers inherit the same handles; keep them alive through every peer's close.
    pub cuda_streams: HashMap<u32, crate::CudaStream>,
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
/// Separate sessions may be called concurrently; an engine may serialize execution.
/// Model-internal state sharing depends on the backend.
pub struct Session {
    engine: Option<Arc<dyn Engine>>,
    bundle: Bundle,
    output_specs: IndexMap<String, (crate::dlpack::DLDataType, Vec<i64>)>,
    state_location: MemoryLocation,
    public_inputs: IndexSet<String>,
    initial_inputs: IndexMap<String, SharedTensor>,
    inputs: IndexMap<String, SharedTensor>,
    feedback: IndexMap<String, String>,
    feedback_sources: IndexSet<String>,
    placement: OutputPlacement,
    engine_placement: OutputPlacement,
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
        let feedback: IndexMap<String, String> = bundle
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
        let feedback_sources: IndexSet<String> = feedback.values().cloned().collect();
        let output_specs: IndexMap<_, _> = bundle
            .pipeline_outputs()
            .iter()
            .map(|port| {
                let spec = graph.outputs[port];
                (
                    port.name(),
                    (
                        engines::memory::dlpack::dl_type(spec.dtype),
                        spec.shape.clone(),
                    ),
                )
            })
            .collect();
        // SessionOptions and graph routing are immutable after initialization.
        let mut engine_placement = OutputPlacement::default();
        for name in output_specs.keys() {
            engine_placement
                .overrides
                .insert(name.clone(), options.output_placement.location(name));
        }
        for source in &feedback_sources {
            engine_placement
                .overrides
                .insert(source.clone(), state_location);
        }
        Ok(Self {
            engine: Some(engine),
            bundle: bundle.clone(),
            output_specs,
            state_location,
            public_inputs: bundle.pipeline_inputs().iter().map(|p| p.name()).collect(),
            inputs: initial_inputs.clone(),
            initial_inputs,
            feedback,
            feedback_sources,
            placement: options.output_placement,
            engine_placement,
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
            engine: Some(self.engine()?.clone().new_session()?),
            bundle: self.bundle.clone(),
            output_specs: self.output_specs.clone(),
            state_location: self.state_location,
            public_inputs: self.public_inputs.clone(),
            initial_inputs: self.initial_inputs.clone(),
            inputs: self.initial_inputs.clone(),
            feedback: self.feedback.clone(),
            feedback_sources: self.feedback_sources.clone(),
            placement: self.placement.clone(),
            engine_placement: self.engine_placement.clone(),
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
    /// Names and tensors are borrowed from the map; matching is by name, independent
    /// of iteration order. Missing or unknown inputs are errors. Inserting a duplicate
    /// key replaces its earlier value before the runtime receives the map.
    /// Feedback advances only after every output and isolation copy succeeds.
    ///
    /// ```
    /// use lrt::{dlpack::BorrowedTensor, RunResult, RuntimeError, Session};
    /// use std::collections::HashMap;
    ///
    /// fn step(session: &mut Session) -> Result<RunResult, RuntimeError> {
    ///     let values = [1_f32, 2., 3.];
    ///     let name = String::from("model/input");
    ///     let inputs = HashMap::from([
    ///         (name.as_str(), BorrowedTensor::cpu(&values, &[3]).unwrap()),
    ///     ]);
    ///     session.run(&inputs)
    /// }
    /// ```
    pub fn run(
        &mut self,
        inputs: &HashMap<&str, BorrowedTensor<'_>>,
    ) -> Result<RunResult, RuntimeError> {
        self.run_into(inputs, &mut HashMap::new())
    }

    /// Supplied outputs use caller storage; omitted outputs are allocated.
    /// Native backends bind supplied buffers directly; Triton copies ensemble results into them.
    /// The result contains only allocated outputs. Supplied buffers must have
    /// compatible dtype, shape, device and alignment and cannot overlap inputs
    /// or each other. On failure contents are unspecified; feedback is unchanged.
    pub fn run_into(
        &mut self,
        inputs: &HashMap<&str, BorrowedTensor<'_>>,
        buffers: &mut HashMap<&str, OutputBuffer<'_>>,
    ) -> Result<RunResult, RuntimeError> {
        self.execute(inputs, buffers)
            .map_err(|error| match error.downcast::<RuntimeError>() {
                Ok(error) => error,
                Err(error) => RuntimeError::new("execute", error),
            })
    }

    fn execute(
        &mut self,
        inputs: &HashMap<&str, BorrowedTensor<'_>>,
        buffers: &mut HashMap<&str, OutputBuffer<'_>>,
    ) -> anyhow::Result<RunResult> {
        let engine = self.engine()?;
        let mut remaining = buffers.iter();
        while let Some((&name, buffer)) = remaining.next() {
            let (dtype, expected) = self
                .output_specs
                .get(name)
                .with_context(|| format!("unknown public output {name}"))?;
            let descriptor = buffer.descriptor();
            let shape = buffer.shape();
            ensure!(
                (
                    descriptor.dtype.code,
                    descriptor.dtype.bits,
                    descriptor.dtype.lanes
                ) == (dtype.code, dtype.bits, dtype.lanes)
                    && shape.len() == expected.len()
                    && shape.iter().zip(expected).all(|(&a, &b)| b == -1 || a == b),
                "output buffer {name} does not match declared dtype/shape"
            );
            let address = buffer.address();
            for (&other, next) in remaining.clone() {
                let next_descriptor = next.descriptor();
                ensure!(
                    !engines::memory::dlpack::ranges_overlap(
                        descriptor.device,
                        address,
                        buffer.bytes,
                        next_descriptor.device,
                        next.address(),
                        next.bytes
                    ),
                    "output buffers {name} and {other} overlap"
                );
            }
        }
        for name in inputs.keys() {
            ensure!(
                self.public_inputs.contains(*name),
                "unknown public input {name}"
            );
        }
        let mut values = self
            .public_inputs
            .iter()
            .map(|name| {
                Ok((
                    name.as_str(),
                    *inputs
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
        let mut outputs = engine.run(&values, &self.engine_placement, buffers)?;
        if !buffers.is_empty() {
            for name in &self.feedback_sources {
                if let Some(buffer) = buffers.get(name.as_str()) {
                    // SAFETY: successful inference initialized this exclusive buffer.
                    outputs.insert(
                        name.clone(),
                        engine.copy(unsafe { buffer.borrow() }, self.state_location)?,
                    );
                }
            }
        }
        let output = |name: &str| -> anyhow::Result<SharedTensor> {
            outputs.get(name).cloned().ok_or_else(|| {
                crate::error::Internal(format!("missing engine output {name}")).into()
            })
        };
        let public_outputs = self
            .output_specs
            .keys()
            .filter(|name| !buffers.contains_key(name.as_str()))
            .map(|name| {
                let value = output(name)?;
                let value = if self.feedback_sources.contains(name) {
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
#[path = "../tests/functional_tests/session/state.rs"]
mod native_tests;
