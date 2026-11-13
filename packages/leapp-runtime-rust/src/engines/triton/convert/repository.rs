// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{proto::model_config::ModelConfig, TritonPipeline, MODEL_VERSION};
use crate::error::RuntimeError;
use anyhow::{ensure, Context, Result};
use std::{ffi::OsStr, fs, path::Path};
use tempfile::TempDir;

/// Owns the generated repository, but not the linked artifact directories.
/// Keep the repository and original artifacts alive until Triton is destroyed.
#[derive(Debug)]
pub(crate) struct ModelRepository {
    directory: TempDir,
}

impl ModelRepository {
    pub(crate) fn create(pipeline: &TritonPipeline) -> Result<Self, RuntimeError> {
        (|| {
            let directory = tempfile::Builder::new()
                .prefix("leapp-triton-")
                .tempdir()
                .context("create temporary model repository")?;
            populate(pipeline, directory)
        })()
        .map_err(|source| RuntimeError::new("repository", source))
    }

    pub(crate) fn path(&self) -> &Path {
        self.directory.path()
    }
}

fn populate(pipeline: &TritonPipeline, directory: TempDir) -> Result<ModelRepository> {
    // The temporary owner also removes partial contents if any write fails.
    for model in &pipeline.models {
        write_model(
            directory.path(),
            &model.config,
            model.version,
            Some(&model.model_path),
        )?;
    }
    write_model(directory.path(), &pipeline.ensemble, MODEL_VERSION, None)?;
    Ok(ModelRepository { directory })
}

fn write_model(
    root: &Path,
    config: &ModelConfig,
    version: i64,
    artifact: Option<&Path>,
) -> Result<()> {
    ensure!(
        Path::new(&config.name).file_name() == Some(OsStr::new(&config.name)),
        "invalid Triton model directory name: {:?}",
        config.name
    );
    ensure!(version > 0, "invalid Triton model version: {version}");
    let directory = root.join(&config.name);
    fs::create_dir(&directory)
        .with_context(|| format!("create model directory {}", directory.display()))?;
    let version_directory = directory.join(version.to_string());
    if let Some(artifact) = artifact {
        let parent = artifact
            .parent()
            .context("model artifact has no parent directory")?;
        // Link the whole directory so external weights keep their relative paths.
        link_directory(parent, &version_directory)?;
    } else {
        fs::create_dir(&version_directory).with_context(|| {
            format!(
                "create ensemble version directory {}",
                version_directory.display()
            )
        })?;
    }
    let path = directory.join("config.pbtxt");
    fs::write(&path, protobuf::text_format::print_to_string_pretty(config))
        .with_context(|| format!("write model configuration {}", path.display()))
}

#[cfg(unix)]
fn link_directory(source: &Path, link: &Path) -> Result<()> {
    std::os::unix::fs::symlink(source, link)
        .with_context(|| format!("link {} to {}", link.display(), source.display()))
}

#[cfg(not(unix))]
fn link_directory(_source: &Path, _link: &Path) -> Result<()> {
    Err(crate::error::Unsupported("model repository links require Unix".into()).into())
}

#[cfg(all(test, unix))]
#[path = "../../../../tests/unit_tests/triton/repository.rs"]
mod tests;
