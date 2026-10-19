// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    execution::TritonPipeline, Bundle, DataType, MemoryLocation, PortRef, RuntimeError, Tensor,
};
use anyhow::{ensure, Context};
use indexmap::IndexMap;
use safetensors::SafeTensors;
use std::fs;

pub(crate) fn initial_inputs(
    bundle: &Bundle,
    pipeline: &TritonPipeline,
) -> Result<IndexMap<String, Tensor>, RuntimeError> {
    (|| {
        let ports: Vec<_> = pipeline
            .default_inputs
            .iter()
            .map(|p| {
                (
                    &p.input.port,
                    &p.input.tensor_name,
                    p.initial_value.as_ref(),
                )
            })
            .chain(
                pipeline
                    .feedback
                    .iter()
                    .map(|p| (&p.target, &p.input_name, p.initial_value.as_ref())),
            )
            .collect();
        // Read the payload once. Public and forward-connected inputs are supplied by
        // the caller or ensemble, so their exported initial values are not consumed.
        let bytes = ports
            .iter()
            .find_map(|(_, _, initial)| *initial)
            .map(|initial| fs::read(&initial.file))
            .transpose()?;
        let tensors = bytes.as_deref().map(SafeTensors::deserialize).transpose()?;
        ports
            .into_iter()
            .map(|(port, name, initial)| {
                let tensor = if let Some(initial) = initial {
                    let value = tensors
                        .as_ref()
                        .context("missing initial values")?
                        .tensor(&initial.tensor_name)?;
                    let shape = value
                        .shape()
                        .iter()
                        .copied()
                        .map(i64::try_from)
                        .collect::<Result<Vec<_>, _>>()?;
                    let dtype = initial_dtype(value.dtype())?;
                    Tensor::copy_from_cpu_bytes(value.data(), dtype, &shape, MemoryLocation::Cpu)?
                } else {
                    zero_input(bundle, port)?
                };
                Ok((name.clone(), tensor))
            })
            .collect::<anyhow::Result<_>>()
    })()
    .map_err(|source| RuntimeError::new("state", source))
}

fn zero_input(bundle: &Bundle, port: &PortRef) -> anyhow::Result<Tensor> {
    let spec = bundle
        .input_info(&port.node, &port.port)
        .context("missing state input")?;
    let width: usize = match spec.dtype {
        DataType::Bool | DataType::Uint8 | DataType::Int8 => 1,
        DataType::Float16 | DataType::Bfloat16 | DataType::Uint16 | DataType::Int16 => 2,
        DataType::Float32 | DataType::Uint32 | DataType::Int32 => 4,
        DataType::Float64 | DataType::Uint64 | DataType::Int64 => 8,
    };
    let mut size = width;
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
    Ok(Tensor::copy_from_cpu_bytes(
        &bytes,
        spec.dtype,
        &spec.shape,
        MemoryLocation::Cpu,
    )?)
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
