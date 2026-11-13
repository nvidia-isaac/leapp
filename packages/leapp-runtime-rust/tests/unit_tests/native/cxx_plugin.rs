// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::ErrorCode;
use serde_json::json;

fn plugins() -> Option<PathBuf> {
    let directory = std::env::var_os("LEAPP_NATIVE_TEST_PLUGINS").map(PathBuf::from)?;
    assert!(directory.is_dir(), "build the enabled C++ plugin fixture");
    Some(directory)
}

fn load(directory: &std::path::Path, mode: &str, outputs: usize) -> Result<Model> {
    let spec = serde_json::from_value(json!({
        "inputs": [{"name":"input", "dtype":"float32", "shape":[-1]}],
        "outputs": (0..outputs).map(|i| json!({
            "name":format!("output{i}"), "dtype":"float32", "shape":[-1]
        })).collect::<Vec<_>>(),
        "parameters": {"backend":"test", "model_path":mode}
    }))?;
    Model::load_library(
        &spec,
        MemoryLocation::Cpu,
        None,
        ffi::Library::open(&directory.join("libleapp_plugin_fixture.so"), "fixture")?,
    )
}

#[test]
fn cxx_dynamic_offsets_mixed_outputs_and_retained_owners() {
    let Some(directory) = plugins() else { return };
    let mut model = load(&directory, "dynamic", 2).unwrap();
    assert!(model.needs_guard());
    assert!(!model.can_prepare());
    let mut source = [99_f32, 2., 3.];
    let shape = [2_i64];
    let mut input = *BorrowedTensor::cpu(&source[..2], &shape)
        .unwrap()
        .descriptor();
    input.byte_offset = 4;
    let mut transfers = crate::engines::native::Transfers::default();
    // SAFETY: the offset selects two f32s alive until synchronous return.
    let first =
        unsafe { model.run_dynamic(&[BorrowedTensor::new(&input)], &[], &mut transfers) }.unwrap();
    let mut peer = model.new_session().unwrap();
    drop(model);
    source[1] = 8.;
    std::hint::black_box(&source);
    let mut target = [0_f32; 3];
    let mut output = *BorrowedTensor::cpu(&target[..2], &shape)
        .unwrap()
        .descriptor();
    output.byte_offset = 4;
    // SAFETY: source and writable destination are disjoint and retained.
    let second = unsafe {
        peer.run_dynamic(
            &[BorrowedTensor::new(&input)],
            &[(1, output)],
            &mut transfers,
        )
    }
    .unwrap();
    drop(peer);
    std::hint::black_box(&mut target);
    assert_eq!(target, [0., 8., 3.]);
    assert!(second[1].is_none());
    assert_eq!(
        first[0].as_ref().unwrap().to_cpu_vec::<f32>().unwrap(),
        [2., 3.]
    );
    assert_eq!(
        second[0].as_ref().unwrap().to_cpu_vec::<f32>().unwrap(),
        [8., 3.]
    );
}

#[test]
fn cxx_prepared_routes_and_peer_state_cross_the_bridge() {
    let Some(directory) = plugins() else { return };
    let mut model = load(&directory, "prepared", 1).unwrap();
    let source = [1_f32, 2.];
    let shape = [2_i64];
    let mut target = [0_f32; 2];
    let output = *BorrowedTensor::cpu(&target, &shape).unwrap().descriptor();
    // SAFETY: prepared storage and metadata remain valid until model destruction.
    unsafe {
        model.prepare(
            &[BorrowedTensor::cpu(&source, &shape).unwrap()],
            &[BorrowedTensor::new(&output)],
        )
    }
    .unwrap();
    assert!(model.is_prepared());
    assert!(!model.new_session().unwrap().is_prepared());
    let replacement = [7_f32, 9.];
    let mut overridden = [0_f32; 2];
    let binding = *BorrowedTensor::cpu(&overridden, &shape)
        .unwrap()
        .descriptor();
    // SAFETY: both calls complete before releasing their input/output owners.
    unsafe {
        model
            .run(
                &[("value", BorrowedTensor::cpu(&replacement, &shape).unwrap())],
                &[InputRoute {
                    position: 0,
                    boundary: 0,
                }],
                &[(0, binding)],
            )
            .unwrap();
        model.run(&[], &[], &[]).unwrap();
    }
    drop(model);
    std::hint::black_box(&mut target);
    std::hint::black_box(&mut overridden);
    assert_eq!(target, source);
    assert_eq!(overridden, replacement);
}

#[test]
fn cxx_errors_and_incompatible_libraries_are_rejected() {
    let Some(directory) = plugins() else { return };
    for (mode, expected) in [
        ("1", ErrorCode::InvalidArgument),
        ("2", ErrorCode::NotFound),
        ("3", ErrorCode::Unsupported),
        ("4", ErrorCode::Io),
        ("5", ErrorCode::Internal),
    ] {
        let error = load(&directory, mode, 1).err().unwrap();
        assert_eq!(super::super::error("load", error).code(), expected);
    }
    for (mode, expected) in [
        ("throw", ErrorCode::Internal),
        ("partial_error", ErrorCode::Internal),
        ("copy_error", ErrorCode::InvalidArgument),
    ] {
        let mut model = load(&directory, mode, 1).unwrap();
        let mut transfers = crate::engines::native::Transfers::default();
        // SAFETY: the fixture completes or cleans up before these owners drop.
        let error = unsafe {
            model.run_dynamic(
                &[BorrowedTensor::cpu(&[5_f32], &[1]).unwrap()],
                &[],
                &mut transfers,
            )
        }
        .err()
        .unwrap();
        assert_eq!(super::super::error("execute", error).code(), expected);
    }
    let path = directory.join("libleapp_plugin_fixture.so");
    for name in ["incompatible", "absent"] {
        let error = ffi::Library::open(&path, name).err().unwrap();
        assert_eq!(
            super::super::error("load", error).code(),
            ErrorCode::Unsupported
        );
    }
}
