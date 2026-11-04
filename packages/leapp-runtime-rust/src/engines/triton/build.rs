// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{env, path::PathBuf};

pub fn build() {
    let proto = "src/engines/triton/convert";
    println!("cargo:rerun-if-changed={proto}/model_config.proto");
    protobuf_codegen::Codegen::new()
        .pure()
        .include(proto)
        .input(format!("{proto}/model_config.proto"))
        .cargo_out_dir("model_config")
        .run_from_script();
    println!("cargo:rerun-if-env-changed=LEAPP_ANALYSIS_ONLY");
    // Editor analysis needs Rust type checking, but does not link the native SDKs.
    if env::var("LEAPP_ANALYSIS_ONLY").as_deref() == Ok("1") {
        return;
    }
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "linux",
        "CUDA memory support requires Linux"
    );
    let cuda = PathBuf::from(env::var_os("CUDA_ROOT").unwrap_or_else(|| "/usr/local/cuda".into()));
    assert!(
        cuda.join("lib64/libcudart.so").is_file(),
        "supply CUDA_ROOT with lib64/libcudart.so"
    );
    println!(
        "cargo:rustc-link-search=native={}",
        cuda.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=dylib=cudart");
}
