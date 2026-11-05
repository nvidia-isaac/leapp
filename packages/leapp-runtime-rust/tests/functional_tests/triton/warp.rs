// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Bundle, ExecutionDevice};
use std::path::PathBuf;

#[test]
#[ignore = "requires Warp adapters, CUDA, and generated Warp fixtures"]
fn warp_plugin_load_errors_are_reported() {
    use crate::engines::triton::{convert, Options, TritonEngine};
    let root = PathBuf::from(std::env::var_os("LEAPP_WARP_FIXTURES").unwrap());
    for (fixture, backend, filename) in [
        ("warp_onnx", "onnxruntime", "libleapp_wrp_onnx_custom_op.so"),
        (
            "warp_pt2_cpu",
            "leapp_pt2",
            "libleapp_wrp_torch_custom_op.so",
        ),
    ] {
        let bundle = Bundle::open(root.join(format!("{fixture}.yaml"))).unwrap();
        let mut pipeline = convert::convert_bundle(&bundle, ExecutionDevice::Cpu).unwrap();
        let installation = tempfile::tempdir().unwrap();
        std::fs::create_dir(installation.path().join(backend)).unwrap();
        std::fs::write(
            installation.path().join(backend).join(filename),
            b"invalid library",
        )
        .unwrap();
        convert::configure_plugins(&mut pipeline, installation.path()).unwrap();
        let repository = convert::repository::ModelRepository::create(&pipeline).unwrap();
        let error = TritonEngine::load(&pipeline, repository, &Options::default())
            .err()
            .unwrap();
        assert_eq!(error.phase(), "load");
        assert!(error.to_string().contains("failed to load"), "{error}");
    }
}
