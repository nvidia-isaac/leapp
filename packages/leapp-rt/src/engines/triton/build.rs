// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub fn build() {
    let proto = "src/engines/triton/convert";
    println!("cargo:rerun-if-changed={proto}/model_config.proto");
    protobuf_codegen::Codegen::new()
        .pure()
        .include(proto)
        .input(format!("{proto}/model_config.proto"))
        .cargo_out_dir("model_config")
        .run_from_script();
}
