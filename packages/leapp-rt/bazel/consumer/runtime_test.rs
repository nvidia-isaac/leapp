// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[test]
fn inspect_and_reject_empty_execution_without_backend_sdks() {
    let path = std::env::temp_dir().join(format!("leapp-bazel-rust-{}.yaml", std::process::id()));
    std::fs::write(
        &path,
        "models: {}\npipeline: {inputs: {}, outputs: {}}\nsystem information: {}\n",
    )
    .unwrap();
    let bundle = lrt::Bundle::open(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    assert!(bundle.model_names().is_empty());
    let error = lrt::Session::open(&bundle, lrt::SessionOptions::default())
        .err()
        .unwrap();
    assert_eq!(error.code(), lrt::ErrorCode::InvalidArgument);
    assert_eq!(error.phase(), "convert");
}
