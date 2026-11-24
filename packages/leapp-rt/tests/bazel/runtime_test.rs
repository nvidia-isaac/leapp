// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[test]
fn links_runtime() {
    assert!(lrt::Bundle::open("missing.yaml").is_err());
}
