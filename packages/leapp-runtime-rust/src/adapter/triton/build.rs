// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=TRITON_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "linux",
        "Triton integration requires Linux"
    );
    let root =
        PathBuf::from(env::var_os("TRITON_ROOT").unwrap_or_else(|| "/opt/tritonserver".into()));
    assert!(
        root.join("lib/libtritonserver.so").is_file(),
        "supply TRITON_ROOT with lib/libtritonserver.so"
    );
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=tritonserver");
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
