// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::engines::triton::convert::TritonModelSpec;

#[test]
fn failed_population_removes_partial_repository_without_following_links() {
    let artifacts = tempfile::tempdir().unwrap();
    let artifact = artifacts.path().join("model.onnx");
    fs::write(&artifact, b"original model").unwrap();
    let config = ModelConfig {
        name: "same_name".into(),
        ..Default::default()
    };
    let pipeline = TritonPipeline {
        models: vec![TritonModelSpec {
            config: config.clone(),
            version: 1,
            model_path: artifact.clone(),
        }],
        ensemble: config,
        inputs: Vec::new(),
    };
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().to_owned();
    let error = populate(&pipeline, directory).unwrap_err();
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::AlreadyExists
    );
    assert!(!path.exists());
    assert_eq!(fs::read(artifact).unwrap(), b"original model");
}
