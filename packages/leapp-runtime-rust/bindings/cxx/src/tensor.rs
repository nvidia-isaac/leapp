// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{dtype::wire_dtype, ffi};
use leapp_runtime::{DataType, MemoryLocation};

pub struct TensorHandle(pub Option<leapp_triton::Tensor>);

pub fn tensor_new() -> Box<TensorHandle> {
    Box::new(TensorHandle(None))
}

pub fn dtype(value: ffi::Dtype) -> Result<DataType, ffi::StatusData> {
    use ffi::Dtype as W;

    Ok(match value {
        W::Bool => DataType::Bool,
        W::Uint8 => DataType::Uint8,
        W::Uint16 => DataType::Uint16,
        W::Uint32 => DataType::Uint32,
        W::Uint64 => DataType::Uint64,
        W::Int8 => DataType::Int8,
        W::Int16 => DataType::Int16,
        W::Int32 => DataType::Int32,
        W::Int64 => DataType::Int64,
        W::Float16 => DataType::Float16,
        W::Bfloat16 => DataType::Bfloat16,
        W::Float32 => DataType::Float32,
        W::Float64 => DataType::Float64,
        _ => return Err(crate::invalid("unknown tensor dtype")),
    })
}

pub fn location(v: ffi::Location) -> Result<MemoryLocation, ffi::StatusData> {
    match v.memory {
        ffi::Memory::Cpu if v.device == 0 => Ok(MemoryLocation::Cpu),
        ffi::Memory::Pinned if v.device == 0 => Ok(MemoryLocation::PinnedCpu),
        ffi::Memory::Cuda => Ok(MemoryLocation::Cuda {
            device_id: v.device,
        }),
        _ => Err(crate::invalid("invalid memory location")),
    }
}

pub fn tensor_info(t: &TensorHandle) -> ffi::TensorData {
    let Some(t) = &t.0 else {
        return ffi::TensorData {
            dtype: ffi::Dtype::Float32,
            shape: Vec::new(),
            location: ffi::Location {
                memory: ffi::Memory::Cpu,
                device: 0,
            },
            byte_size: 0,
        };
    };
    let (memory, device) = match t.location() {
        MemoryLocation::Cpu => (ffi::Memory::Cpu, 0),
        MemoryLocation::PinnedCpu => (ffi::Memory::Pinned, 0),
        MemoryLocation::Cuda { device_id } => (ffi::Memory::Cuda, device_id),
    };
    ffi::TensorData {
        dtype: wire_dtype(t.dtype().into()),
        shape: t.shape().into(),
        location: ffi::Location { memory, device },
        byte_size: t.byte_size(),
    }
}

// Only pointer-returning facade methods call this; metadata and copies stay reusable.
pub fn tensor_address(t: &TensorHandle) -> usize {
    t.0.as_ref().map_or(0, |t| t.data_ptr() as usize)
}

pub fn tensor_cpu_bytes(t: &TensorHandle, bytes: &mut Vec<u8>) -> ffi::StatusData {
    let Some(t) = &t.0 else {
        return crate::invalid("tensor is empty");
    };
    match t.copy_to_cpu_bytes() {
        Ok(value) => {
            *bytes = value;
            crate::success()
        }
        Err(e) => crate::native_error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use leapp_runtime::{Bundle, RunOptions, Runtime, RuntimeOptions, Session, TensorView};
    use std::{env, path::PathBuf};

    fn runtime() -> Runtime {
        let fixtures =
            PathBuf::from(env::var_os("LEAPP_TRITON_FIXTURES").expect("set LEAPP_TRITON_FIXTURES"));
        Runtime::open(
            &Bundle::open(fixtures.join("dynamic.yaml")).unwrap(),
            RuntimeOptions {
                backend_directory: env::var_os("LEAPP_TRITON_BACKENDS")
                    .expect("set LEAPP_TRITON_BACKENDS")
                    .into(),
                ..RuntimeOptions::default()
            },
        )
        .unwrap()
    }

    fn output(session: &mut Session, location: MemoryLocation) -> TensorHandle {
        let mut options = RunOptions::default();
        options.output_placement.default_location = location;
        let result = session
            .run_with_options(
                &[(
                    "dynamic/input",
                    TensorView::cpu(&[1.0_f32, 2.0], &[2]).unwrap(),
                )],
                &options,
            )
            .unwrap();
        TensorHandle(Some(result.get("dynamic/output").unwrap().native().clone()))
    }

    // Observe identity through a host borrow; data_ptr() would itself exclude reuse.
    fn host_address(tensor: &TensorHandle) -> usize {
        tensor
            .0
            .as_ref()
            .unwrap()
            .as_cpu_slice::<f32>()
            .unwrap()
            .as_ptr() as usize
    }

    #[test]
    fn metadata_and_host_copies_preserve_output_pool_reuse() {
        let runtime = runtime();
        let mut session = runtime.create_session().unwrap();
        for location in [MemoryLocation::Cpu, MemoryLocation::PinnedCpu] {
            let first = output(&mut session, location);
            let address = host_address(&first);
            let retained = first.0.as_ref().unwrap().clone();
            let info = tensor_info(&first);
            assert!(info.dtype == ffi::Dtype::Float32);
            assert_eq!(info.shape, [2]);
            assert_eq!(info.byte_size, 8);
            let mut bytes = Vec::new();
            assert_eq!(tensor_cpu_bytes(&first, &mut bytes).code, 0);
            assert_eq!(
                bytes,
                [1.0_f32, 2.0]
                    .into_iter()
                    .flat_map(f32::to_ne_bytes)
                    .collect::<Vec<_>>()
            );
            drop(first);
            let second = output(&mut session, location);
            assert_ne!(address, host_address(&second));
            assert_eq!(retained.as_cpu_slice::<f32>().unwrap(), [1.0, 2.0]);
            drop((second, retained));
            let reused = output(&mut session, location);
            assert_eq!(address, host_address(&reused));
        }
    }

    #[test]
    fn pointer_exports_exclude_reuse_and_survive_runtime_drop() {
        let runtime = runtime();
        let mut session = runtime.create_session().unwrap();
        for location in [
            MemoryLocation::Cpu,
            MemoryLocation::PinnedCpu,
            MemoryLocation::Cuda { device_id: 0 },
        ] {
            let first = output(&mut session, location);
            let alias = TensorHandle(first.0.clone());
            let address = tensor_address(&alias);
            assert_ne!(address, 0);
            drop((first, alias));
            let next = output(&mut session, location);
            assert_ne!(address, tensor_address(&next));
        }
        let retained = output(&mut session, MemoryLocation::Cpu);
        let address = tensor_address(&retained);
        drop((session, runtime));
        assert_eq!(address, tensor_address(&retained));
        let mut bytes = Vec::new();
        assert_eq!(tensor_cpu_bytes(&retained, &mut bytes).code, 0);
        assert_eq!(
            bytes,
            [1.0_f32, 2.0]
                .into_iter()
                .flat_map(f32::to_ne_bytes)
                .collect::<Vec<_>>()
        );
        assert_eq!(tensor_address(&TensorHandle(None)), 0);
    }
}
