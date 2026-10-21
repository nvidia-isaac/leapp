// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::dlpack::{BorrowedTensor, DLDataType, DLDevice, DLTensor, SharedTensor};
use crate::{
    engines::Engine,
    session::graph::{Graph, InputSource},
    Bundle, DataType, MemoryLocation, PortRef, RuntimeError,
};
use anyhow::{ensure, Context};
use indexmap::IndexMap;
use safetensors::SafeTensors;
use std::fs;

pub(crate) fn initial_inputs(
    bundle: &Bundle,
    graph: &Graph<'_>,
    location: MemoryLocation,
    engine: &dyn Engine,
) -> Result<IndexMap<String, SharedTensor>, RuntimeError> {
    (|| {
        let ports: Vec<_> = bundle
            .0
            .description
            .models
            .iter()
            .flat_map(|(node, model)| {
                model.inputs.iter().map(move |spec| PortRef {
                    node: node.clone(),
                    port: spec.name.clone(),
                })
            })
            .filter(|port| {
                matches!(
                    graph.sources.get(port),
                    None | Some(InputSource::Feedback(_))
                )
            })
            .collect();
        // Read consumed initial values once. Inspection owns metadata validation.
        let bytes = if ports
            .iter()
            .any(|p| bundle.0.initial_values.contains_key(p))
        {
            bundle
                .0
                .description
                .pipeline
                .initial_values
                .as_ref()
                .map(fs::read)
                .transpose()?
        } else {
            None
        };
        let tensors = bytes.as_deref().map(SafeTensors::deserialize).transpose()?;
        ports
            .into_iter()
            .map(|port| {
                let name = port.name();
                let tensor = if bundle.0.initial_values.contains_key(&port) {
                    let value = tensors
                        .as_ref()
                        .context("missing initial values")?
                        .tensor(&name)?;
                    let shape = value
                        .shape()
                        .iter()
                        .copied()
                        .map(i64::try_from)
                        .collect::<Result<Vec<_>, _>>()?;
                    let dtype = initial_dtype(value.dtype())?;
                    copy_initial(value.data(), dtype, &shape, location, engine)?
                } else {
                    zero_input(bundle, &port, location, engine)?
                };
                Ok((name.clone(), tensor))
            })
            .collect::<anyhow::Result<_>>()
    })()
    .map_err(|source| RuntimeError::new("state", source))
}

fn zero_input(
    bundle: &Bundle,
    port: &PortRef,
    location: MemoryLocation,
    engine: &dyn Engine,
) -> anyhow::Result<SharedTensor> {
    let spec = bundle
        .input_info(&port.node, &port.port)
        .context("missing state input")?;
    let mut size = usize::from(dl_dtype(spec.dtype).bits / 8);
    for &dim in &spec.shape {
        ensure!(
            dim >= 0,
            "dynamic state input {port} requires an exported initial value"
        );
        size = size
            .checked_mul(usize::try_from(dim)?)
            .context("state tensor size overflow")?;
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .context("allocate initial state")?;
    bytes.resize(size, 0);
    copy_initial(&bytes, spec.dtype, &spec.shape, location, engine)
}

fn initial_dtype(dtype: safetensors::Dtype) -> anyhow::Result<DataType> {
    use safetensors::Dtype;

    Ok(match dtype {
        Dtype::BOOL => DataType::Bool,
        Dtype::U8 => DataType::Uint8,
        Dtype::U16 => DataType::Uint16,
        Dtype::U32 => DataType::Uint32,
        Dtype::U64 => DataType::Uint64,
        Dtype::I8 => DataType::Int8,
        Dtype::I16 => DataType::Int16,
        Dtype::I32 => DataType::Int32,
        Dtype::I64 => DataType::Int64,
        Dtype::F16 => DataType::Float16,
        Dtype::BF16 => DataType::Bfloat16,
        Dtype::F32 => DataType::Float32,
        Dtype::F64 => DataType::Float64,
        _ => return Err(crate::error::Unsupported(format!("initial dtype {dtype:?}")).into()),
    })
}

fn copy_initial(
    bytes: &[u8],
    dtype: DataType,
    shape: &[i64],
    location: MemoryLocation,
    engine: &dyn Engine,
) -> anyhow::Result<SharedTensor> {
    let dtype = dl_dtype(dtype);
    let expected = shape
        .iter()
        .try_fold(usize::from(dtype.bits / 8), |n, &d| {
            n.checked_mul(usize::try_from(d).ok()?)
        })
        .context("initial tensor size overflow")?;
    ensure!(expected == bytes.len(), "initial tensor byte size mismatch");
    let descriptor = DLTensor {
        data: bytes.as_ptr().cast_mut().cast(),
        device: DLDevice {
            device_type: 1,
            device_id: 0,
        },
        ndim: i32::try_from(shape.len())?,
        dtype,
        shape: shape.as_ptr().cast_mut(),
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    };
    // SAFETY: bytes and shape remain borrowed until the synchronous engine copy completes.
    Ok(engine.copy(unsafe { BorrowedTensor::new(&descriptor) }, location)?)
}

fn dl_dtype(dtype: DataType) -> DLDataType {
    let (code, bits) = match dtype {
        DataType::Bool => (6, 8),
        DataType::Uint8 => (1, 8),
        DataType::Uint16 => (1, 16),
        DataType::Uint32 => (1, 32),
        DataType::Uint64 => (1, 64),
        DataType::Int8 => (0, 8),
        DataType::Int16 => (0, 16),
        DataType::Int32 => (0, 32),
        DataType::Int64 => (0, 64),
        DataType::Float16 => (2, 16),
        DataType::Float32 => (2, 32),
        DataType::Float64 => (2, 64),
        DataType::Bfloat16 => (4, 16),
    };
    DLDataType {
        code,
        bits,
        lanes: 1,
    }
}
