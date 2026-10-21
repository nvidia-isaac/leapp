// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::engines::triton::memory::Buffer;

#[test]
fn handoff_producer_restores_device_on_return_error_and_panic() {
    let (mut original, mut count) = (0, 0);
    // SAFETY: both arguments are live output parameters.
    unsafe {
        check(cudaGetDevice(&mut original)).unwrap();
        check(cudaGetDeviceCount(&mut count)).unwrap();
    }
    let target = ((original + 1) % count) as u32;
    let stream = HandoffStream::new(target).unwrap();
    let current = || {
        let mut device = -1;
        // SAFETY: device is a live output parameter.
        unsafe { check(cudaGetDevice(&mut device)).unwrap() };
        device
    };
    assert_eq!(stream.with_device(current).unwrap(), target as c_int);
    assert_eq!(current(), original);
    let error = stream
        .with_device(|| {
            assert_eq!(current(), target as c_int);
            Err::<(), _>("producer failed")
        })
        .unwrap();
    assert!(error.is_err());
    assert_eq!(current(), original);
    assert!(std::panic::catch_unwind(|| {
        let _ = stream.with_device(|| panic!("producer panicked"));
    })
    .is_err());
    assert_eq!(current(), original);
    // Also restore after a callback changes an initially matching device.
    let same = HandoffStream::new(original as u32).unwrap();
    same.with_device(|| {
        // SAFETY: target is an available CUDA device on this thread.
        unsafe { check(cudaSetDevice(target as c_int)).unwrap() };
    })
    .unwrap();
    assert_eq!(current(), original);
}

#[test]
fn allocation_transfer_and_release_restore_the_calling_device() {
    let (mut original, mut count) = (0, 0);
    // SAFETY: both parameters are live integer outputs.
    unsafe {
        check(cudaGetDevice(&mut original)).unwrap();
        check(cudaGetDeviceCount(&mut count)).unwrap();
    }
    assert!(count > 0);
    let target = ((original + 1) % count) as u32;
    let tensor = Buffer::allocate(8, MemoryLocation::Cuda { device_id: target }).unwrap();
    let values = [1.0f32, 2.0];
    let mut output = [0.0f32; 2];
    unsafe {
        tensor.copy_from(values.as_ptr().cast()).unwrap();
        tensor.copy_to(output.as_mut_ptr().cast()).unwrap();
    }
    assert_eq!(output, values);
    let tensor = crate::engines::triton::memory::dlpack::from_buffer(
        crate::dlpack::DLDataType {
            code: 2,
            bits: 32,
            lanes: 1,
        },
        vec![2],
        std::sync::Arc::new(tensor),
    )
    .unwrap();
    for location in [
        MemoryLocation::Cpu,
        MemoryLocation::PinnedCpu,
        MemoryLocation::Cuda { device_id: target },
        MemoryLocation::Cuda {
            device_id: original as u32,
        },
    ] {
        // SAFETY: tensor owns ready storage and has no writable exports.
        let copied = crate::engines::triton::memory::dlpack::copy(
            unsafe { tensor.borrow() },
            Some(location),
        )
        .unwrap();
        assert_eq!(copied.to_cpu_vec::<f32>().unwrap(), values);
        assert_eq!(
            crate::engines::triton::memory::dlpack::location(copied.descriptor().device).unwrap(),
            location
        );
    }
    assert_eq!(tensor.to_cpu_vec::<f32>().unwrap(), values);
    assert!(tensor.to_cpu_vec::<i32>().is_err());
    let mut current = -1;
    // SAFETY: current is a live output parameter.
    unsafe {
        check(cudaGetDevice(&mut current)).unwrap();
    }
    assert_eq!(current, original);
    drop(tensor);
    // SAFETY: current is a live output parameter.
    unsafe {
        check(cudaGetDevice(&mut current)).unwrap();
    }
    assert_eq!(current, original);
}
