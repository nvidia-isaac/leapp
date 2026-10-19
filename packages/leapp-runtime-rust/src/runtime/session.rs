// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{RunResult, RuntimeInner};
use crate::{MemoryLocation, RuntimeError, Tensor, TensorView};
use anyhow::{ensure, Context};
use indexmap::IndexMap;
use std::{collections::HashMap, sync::Arc};

/// Exact output allocation locations; unsupported placements return an error.
#[derive(Debug, Clone, Default)]
pub struct OutputPlacement {
    /// Location for outputs without an override. Defaults to CPU.
    pub default_location: MemoryLocation,
    /// Per-output overrides keyed by public LEAPP `node/port` names.
    pub overrides: HashMap<String, MemoryLocation>,
}

/// Options for one synchronous invocation.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Placement of public outputs. Hidden feedback outputs use CPU storage.
    pub output_placement: OutputPlacement,
}

/// Independent feedback state over shared models. Separate sessions may run concurrently.
/// Model-internal mutable state, if any, is shared and is not reset by this API.
pub struct Session {
    runtime: Arc<RuntimeInner>,
    inputs: IndexMap<String, Tensor>,
}

impl Session {
    pub(super) fn new(runtime: Arc<RuntimeInner>) -> Self {
        Self {
            inputs: runtime.initial_inputs.clone(),
            runtime,
        }
    }

    /// Restore captured initial feedback and default inputs without reloading files.
    /// Retained results are unaffected. Backend-internal model state is not reset.
    pub fn reset(&mut self) -> Result<(), RuntimeError> {
        self.inputs.clone_from(&self.runtime.initial_inputs);
        Ok(())
    }

    /// Execute with CPU outputs, advancing feedback only after a successful call.
    pub fn run(&mut self, inputs: &[(&str, TensorView<'_>)]) -> Result<RunResult, RuntimeError> {
        self.run_with_options(inputs, &RunOptions::default())
    }

    /// Execute synchronously with explicit public output placement.
    ///
    /// Supply every public input exactly once using its qualified name. Borrowed
    /// inputs are released before returning. Feedback reads the previous successful
    /// call's outputs; validation or inference failures leave session state unchanged.
    /// A public output also used as feedback retains its requested storage location.
    pub fn run_with_options(
        &mut self,
        inputs: &[(&str, TensorView<'_>)],
        options: &RunOptions,
    ) -> Result<RunResult, RuntimeError> {
        self.execute(inputs, options)
            .map_err(|error| match error.downcast::<RuntimeError>() {
                Ok(error) => error,
                Err(error) => RuntimeError::new("execute", error),
            })
    }

    fn execute(
        &mut self,
        inputs: &[(&str, TensorView<'_>)],
        options: &RunOptions,
    ) -> anyhow::Result<RunResult> {
        let bindings = &self.runtime.bindings;
        let mut supplied = HashMap::new();
        for &(name, tensor) in inputs {
            ensure!(
                bindings.inputs.contains_key(name),
                "unknown public input {name}"
            );
            ensure!(
                supplied.insert(name, tensor).is_none(),
                "duplicate public input {name}"
            );
        }
        let mut native_inputs = Vec::new();
        for (public, native) in &bindings.inputs {
            let tensor = supplied
                .get(public.as_str())
                .with_context(|| format!("missing public input {public}"))?;
            native_inputs.push((native.as_str(), tensor.0));
        }
        native_inputs.extend(
            self.inputs
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor.0.view())),
        );
        let mut placement = leapp_triton::OutputPlacement::default();
        for name in options.output_placement.overrides.keys() {
            ensure!(
                bindings.outputs.contains_key(name),
                "unknown public output {name}"
            );
        }
        for (public, native) in &bindings.outputs {
            let location = options
                .output_placement
                .overrides
                .get(public)
                .copied()
                .unwrap_or(options.output_placement.default_location);
            placement.overrides.insert(native.clone(), location);
        }
        let outputs: HashMap<_, _> = self
            .runtime
            .native
            .infer(&self.runtime.model, &native_inputs, &placement)?
            .into_iter()
            .map(|(name, tensor)| (name, Tensor(tensor)))
            .collect();
        let output = |name: &str| -> anyhow::Result<Tensor> {
            outputs.get(name).cloned().ok_or_else(|| {
                crate::error::Internal(format!("missing Triton output {name}")).into()
            })
        };
        let public_outputs = bindings
            .outputs
            .iter()
            .map(|(public, native)| Ok((public.clone(), output(native)?)))
            .collect::<anyhow::Result<_>>()?;
        let feedback = self
            .runtime
            .feedback
            .iter()
            .map(|(input, source)| Ok((input.clone(), output(source)?)))
            .collect::<anyhow::Result<Vec<_>>>()?;
        // Publish all next-state buffers together, after validating every result.
        self.inputs.extend(feedback);
        Ok(RunResult {
            outputs: public_outputs,
        })
    }
}
