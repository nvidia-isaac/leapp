// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use leapp_runtime::dlpack::*;
use std::{
    mem::{align_of, offset_of, size_of},
    process::Command,
};

#[test]
fn generated_layout_matches_the_pinned_c_header() {
    let dir = std::env::temp_dir().join(format!("leapp-dlpack-abi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c = dir.join("abi.c");
    let exe = dir.join("abi");
    std::fs::write(&c,r#"
#include <stdio.h>
#include <stddef.h>
#include <dlpack/dlpack.h>
#define PRINT(T) printf("%zu %zu ", sizeof(T), _Alignof(T))
int main(void) {
 PRINT(DLDevice); PRINT(DLDataType); PRINT(DLTensor); PRINT(DLManagedTensor); PRINT(struct DLManagedTensorVersioned);
 printf("%zu %zu %zu %zu %zu %zu", offsetof(DLTensor, shape), offsetof(DLTensor, byte_offset),
 offsetof(DLManagedTensor, deleter), offsetof(struct DLManagedTensorVersioned, deleter),
 offsetof(struct DLManagedTensorVersioned, flags), offsetof(struct DLManagedTensorVersioned, dl_tensor));
}
"#).unwrap();
    assert!(Command::new("cc")
        .args(["-std=c11", "-I"])
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/src/dlpack/include"))
        .arg(&c)
        .arg("-o")
        .arg(&exe)
        .status()
        .unwrap()
        .success());
    let output = Command::new(&exe).output().unwrap();
    assert!(output.status.success());
    let actual: Vec<usize> = String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(
        actual,
        [
            size_of::<DLDevice>(),
            align_of::<DLDevice>(),
            size_of::<DLDataType>(),
            align_of::<DLDataType>(),
            size_of::<DLTensor>(),
            align_of::<DLTensor>(),
            size_of::<DLManagedTensor>(),
            align_of::<DLManagedTensor>(),
            size_of::<DLManagedTensorVersioned>(),
            align_of::<DLManagedTensorVersioned>(),
            offset_of!(DLTensor, shape),
            offset_of!(DLTensor, byte_offset),
            offset_of!(DLManagedTensor, deleter),
            offset_of!(DLManagedTensorVersioned, deleter),
            offset_of!(DLManagedTensorVersioned, flags),
            offset_of!(DLManagedTensorVersioned, dl_tensor)
        ]
    );
    std::fs::remove_dir_all(dir).unwrap();
}
