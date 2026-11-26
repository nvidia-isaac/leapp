// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() {
    cxx_build::bridge("src/bridge.rs")
        .std("c++20")
        .compile("leapp-cxx-bridge");
    println!("cargo:rerun-if-changed=src/bridge.rs");
}
