// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared runtime ownership and per-session feedback over embedded Triton.

mod result;
mod session;

use crate::{
    adapter::{TritonOptions, TritonRuntime},
    execution::{
        convert_bundle, initial_state, repository::ModelRepository, ConversionOptions, PortBinding,
    },
    Bundle, ExecutionDevice, RuntimeError, Tensor,
};
use indexmap::IndexMap;
pub use result::RunResult;
pub use session::{OutputPlacement, RunOptions, Session};
use std::{path::PathBuf, sync::Arc};

/// Model placement and the directory containing installed Triton backends.
#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    /// Device on which all model instances execute. Defaults to CPU.
    pub device: ExecutionDevice,
    /// Backend installation root; defaults to `/opt/tritonserver/backends`.
    pub backend_directory: PathBuf,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        Self {
            device: ExecutionDevice::Cpu,
            backend_directory: "/opt/tritonserver/backends".into(),
        }
    }
}

/// Loaded models shared by independent synchronous sessions.
///
/// Sessions retain the runtime after this handle is dropped. The last owner
/// stops Triton before removing the generated repository. Original artifacts
/// and installed backend libraries must remain available throughout that lifetime.
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

struct RuntimeInner {
    native: TritonRuntime,
    bundle: Bundle,
    model: String,
    bindings: Bindings,
    initial_inputs: IndexMap<String, Tensor>,
    feedback: Vec<(String, String)>,
}

struct Bindings {
    inputs: IndexMap<String, String>,
    outputs: IndexMap<String, String>,
}

fn bindings(ports: &[PortBinding]) -> IndexMap<String, String> {
    ports
        .iter()
        .map(|p| (p.port.name(), p.tensor_name.clone()))
        .collect()
}

impl Runtime {
    /// Convert and load the bundle, capturing initial state for new sessions.
    ///
    /// Feedback and unconnected inputs use exported initial values when present,
    /// otherwise zero tensors. Dynamic state needs a concrete exported initial value.
    pub fn open(bundle: &Bundle, options: RuntimeOptions) -> Result<Self, RuntimeError> {
        let conversion = ConversionOptions {
            device: options.device,
        };
        let pipeline = convert_bundle(bundle, &conversion)?;
        let initial_inputs = initial_state::initial_inputs(bundle, &pipeline)?;
        let repository = ModelRepository::create(&pipeline)?;
        let native = TritonRuntime::open(
            repository,
            &TritonOptions {
                backend_directory: options.backend_directory,
            },
        )?;
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                native,
                bundle: bundle.clone(),
                model: pipeline.ensemble.name,
                bindings: Bindings {
                    inputs: bindings(&pipeline.inputs),
                    outputs: bindings(&pipeline.outputs),
                },
                initial_inputs,
                feedback: pipeline
                    .feedback
                    .into_iter()
                    .map(|p| (p.input_name, p.output_name))
                    .collect(),
            }),
        })
    }

    /// Immutable bundle snapshot used to load the models.
    pub fn info(&self) -> &Bundle {
        &self.inner.bundle
    }

    /// Create an isolated session initialized from the captured initial values.
    pub fn create_session(&self) -> Result<Session, RuntimeError> {
        Ok(Session::new(self.inner.clone()))
    }
}
