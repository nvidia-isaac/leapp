// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::engines::memory::Buffer;

#[test]
fn cpu_operations_survive_missing_cuda() {
    const CHILD: &str = "LEAPP_TEST_MISSING_CUDA_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let directory = tempfile::tempdir().unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "engines::memory::cuda::tests::cpu_operations_survive_missing_cuda",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("CUDA_ROOT", directory.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        return;
    }

    assert!(cuda_runtime_api::loaded().is_none());
    MemoryLocation::Cpu.validate().unwrap();
    let values = [1_u32, 2];
    let buffer = Buffer::allocate(8, MemoryLocation::Cpu).unwrap();
    // SAFETY: null selects the legacy default stream without owning resources.
    let stream = unsafe { CudaStream::from_raw(ptr::null_mut()) };
    assert_eq!(stream.as_raw() as usize, 1);
    assert!(cuda_runtime_api::loaded().is_none());

    for result in [
        validate(MemoryLocation::Cuda { device_id: 0 }),
        stream.validate(0),
        Buffer::allocate(8, MemoryLocation::PinnedCpu).map(|_| ()),
    ] {
        let error = result.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Unsupported);
        assert!(error.to_string().contains("dlopen failed: "), "{error}");
        assert!(error.to_string().contains("CUDA_ROOT"));
        assert!(error.to_string().contains("libcudart.so"));
        assert!(error
            .to_string()
            .contains(&std::env::var("CUDA_ROOT").unwrap()));
    }
    assert!(cuda_runtime_api::loaded().is_none());
    let mut output = [0_u32; 2];
    // SAFETY: both arrays have exactly the buffer's length and remain live.
    unsafe {
        buffer.copy_from(values.as_ptr().cast()).unwrap();
        buffer.copy_to(output.as_mut_ptr().cast()).unwrap();
    }
    assert_eq!(output, values);
}

#[test]
fn borrowed_stream_validation_and_lifetime() {
    if std::env::var("LEAPP_TEST_CUDA").as_deref() != Ok("1") {
        return;
    }
    let api = cuda_runtime_api::load().unwrap();
    let owner = HandoffStream::new(0).unwrap();
    // SAFETY: owner retains this primary-context stream through all borrowed uses.
    let stream = unsafe { CudaStream::from_raw(owner.stream) };
    stream.validate(0).unwrap();
    let borrowed = HandoffStream::borrowed(0, stream);
    borrowed.synchronize().unwrap();
    drop(borrowed);
    owner.synchronize().unwrap();
    let mut count = 0;
    unsafe { api.check((api.cudaGetDeviceCount)(&mut count)).unwrap() };
    if count > 1 {
        assert_eq!(
            stream.validate(1).unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
        owner.synchronize().unwrap();
    }
    assert!(stream.validate(count as u32).is_err());
    owner.synchronize().unwrap();
}

#[test]
fn handoff_producer_restores_device_on_return_error_and_panic() {
    if std::env::var("LEAPP_TEST_CUDA").as_deref() != Ok("1") {
        return;
    }
    let api = cuda_runtime_api::load().unwrap();
    let (mut original, mut count) = (0, 0);
    // SAFETY: both arguments are live output parameters.
    unsafe {
        api.check((api.cudaGetDevice)(&mut original)).unwrap();
        api.check((api.cudaGetDeviceCount)(&mut count)).unwrap();
    }
    let target = ((original + 1) % count) as u32;
    let stream = HandoffStream::new(target).unwrap();
    let current = || {
        let mut device = -1;
        // SAFETY: device is a live output parameter.
        unsafe { api.check((api.cudaGetDevice)(&mut device)).unwrap() };
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
        unsafe { api.check((api.cudaSetDevice)(target as c_int)).unwrap() };
    })
    .unwrap();
    assert_eq!(current(), original);
}

#[test]
fn allocation_transfer_and_release_restore_the_calling_device() {
    if std::env::var("LEAPP_TEST_CUDA").as_deref() != Ok("1") {
        return;
    }
    let api = cuda_runtime_api::load().unwrap();
    let (mut original, mut count) = (0, 0);
    // SAFETY: both parameters are live integer outputs.
    unsafe {
        api.check((api.cudaGetDevice)(&mut original)).unwrap();
        api.check((api.cudaGetDeviceCount)(&mut count)).unwrap();
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
    let tensor = crate::engines::memory::dlpack::from_buffer(
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
        let copied =
            crate::engines::memory::dlpack::copy(unsafe { tensor.borrow() }, Some(location))
                .unwrap();
        assert_eq!(copied.to_cpu_vec::<f32>().unwrap(), values);
        assert_eq!(
            crate::engines::memory::dlpack::location(copied.descriptor().device).unwrap(),
            location
        );
    }
    assert_eq!(tensor.to_cpu_vec::<f32>().unwrap(), values);
    assert!(tensor.to_cpu_vec::<i32>().is_err());
    let mut current = -1;
    // SAFETY: current is a live output parameter.
    unsafe {
        api.check((api.cudaGetDevice)(&mut current)).unwrap();
    }
    assert_eq!(current, original);
    drop(tensor);
    // SAFETY: current is a live output parameter.
    unsafe {
        api.check((api.cudaGetDevice)(&mut current)).unwrap();
    }
    assert_eq!(current, original);
}
