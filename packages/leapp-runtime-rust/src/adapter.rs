// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keep the embedded Triton server and its temporary model repository alive together.
//! Unsafe Triton and CUDA calls live in the separate `leapp-triton` crate.

use crate::{error::RuntimeError, execution::repository::ModelRepository};
pub(crate) use leapp_triton::{OutputPlacement, Tensor, TensorView};
use std::path::PathBuf;

pub(crate) struct TritonOptions {
    pub backend_directory: PathBuf,
}

pub(crate) struct TritonRuntime {
    // The native instance must stop before its repository is removed.
    native: leapp_triton::Runtime,
    _repository: ModelRepository,
}

impl TritonRuntime {
    pub(crate) fn open(
        repository: ModelRepository,
        options: &TritonOptions,
    ) -> Result<Self, RuntimeError> {
        let native = leapp_triton::Runtime::open(repository.path(), &options.backend_directory)
            .map_err(native_error)?;
        Ok(Self {
            native,
            _repository: repository,
        })
    }

    pub(crate) fn infer(
        &self,
        model_name: &str,
        inputs: &[(&str, TensorView<'_>)],
        placement: &OutputPlacement,
    ) -> Result<Vec<(String, Tensor)>, RuntimeError> {
        self.native
            .infer(model_name, inputs, placement)
            .map_err(native_error)
    }
}

fn native_error(error: leapp_triton::Error) -> RuntimeError {
    RuntimeError::new(error.phase(), error.into())
}

#[cfg(test)]
mod tests;
