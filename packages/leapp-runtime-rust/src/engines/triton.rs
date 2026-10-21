// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Triton conversion, native execution, storage, and model repository ownership.
mod convert;
mod memory;
mod native;

use crate::dlpack::{BorrowedTensor, SharedTensor};
use crate::MemoryLocation;
use crate::OutputPlacement;
pub(crate) use memory::cuda::HandoffStream;
use native::error::{Error, ErrorKind, Result};
use std::path::PathBuf;

/// Installation settings specific to Triton.
#[derive(Debug, Clone)]
struct Options {
    backend_directory: PathBuf,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            backend_directory: std::env::var_os("LEAPP_TRITON_BACKENDS")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    std::env::var_os("TRITON_ROOT")
                        .map(PathBuf::from)
                        .unwrap_or_else(|| "/opt/tritonserver".into())
                        .join("backends")
                }),
        }
    }
}

struct TritonEngine {
    // Shutdown must precede removal of the generated repository.
    native: native::Server,
    _repository: convert::repository::ModelRepository,
    model: String,
    inputs: std::collections::HashMap<String, String>,
}

pub(super) fn open(
    bundle: &crate::Bundle,
    graph: &crate::session::graph::Graph<'_>,
    options: &crate::SessionOptions,
) -> std::result::Result<std::sync::Arc<dyn crate::engines::Engine>, crate::RuntimeError> {
    options.output_placement.default_location.validate()?;
    for location in options.output_placement.overrides.values() {
        location.validate()?;
    }
    let pipeline = convert::convert(bundle, options.device, graph)
        .map_err(|e| crate::RuntimeError::new("convert", e))?;
    let repository = convert::repository::ModelRepository::create(&pipeline)?;
    Ok(std::sync::Arc::new(TritonEngine::load(
        &pipeline,
        repository,
        &Options::default(),
    )?))
}

impl TritonEngine {
    fn load(
        pipeline: &convert::TritonPipeline,
        repository: convert::repository::ModelRepository,
        options: &Options,
    ) -> std::result::Result<Self, crate::RuntimeError> {
        let inputs = pipeline
            .inputs
            .iter()
            .map(|p| (p.port.name(), p.tensor_name.clone()))
            .collect();
        let native = native::Server::open(repository.path(), &options.backend_directory)?;
        Ok(Self {
            native,
            _repository: repository,
            model: pipeline.ensemble.name.clone(),
            inputs,
        })
    }
}

impl crate::engines::Engine for TritonEngine {
    fn run(
        &self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        placement: &OutputPlacement,
    ) -> std::result::Result<indexmap::IndexMap<String, SharedTensor>, crate::RuntimeError> {
        let inputs = inputs
            .iter()
            .map(|(name, value)| {
                self.inputs
                    .get(*name)
                    .map(|native| (native.as_str(), *value))
                    .ok_or_else(|| {
                        crate::RuntimeError::new(
                            "execute",
                            crate::error::Internal(format!("unbound input {name}")).into(),
                        )
                    })
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        // Ensemble output names already use logical LEAPP names, including private terminals.
        Ok(self
            .native
            .infer(&self.model, &inputs, placement)?
            .into_iter()
            .collect())
    }
    fn copy(
        &self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
    ) -> std::result::Result<SharedTensor, crate::RuntimeError> {
        self.native.copy_output(input, target).map_err(Into::into)
    }
}

#[cfg(test)]
#[path = "../../tests/functional_tests/triton.rs"]
pub(crate) mod tests;
