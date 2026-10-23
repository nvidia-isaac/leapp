// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{proto::model_config::*, require, ExecutionDevice, MODEL_VERSION};
use crate::{bundle::schema::NodeSpec, DataType as BundleDataType, PortSpec};
use anyhow::{Context, Result};

pub(super) fn model(
    node: &str,
    model: &NodeSpec,
    index: usize,
    device: ExecutionDevice,
) -> Result<ModelConfig> {
    require(
        matches!(
            model.parameters.backend.as_str(),
            "onnx" | "pt2" | "jit" | "trt"
        ),
        format!("{node}: unsupported execution backend"),
    )?;
    require(
        model.parameters.backend != "trt" || matches!(device, ExecutionDevice::Cuda { .. }),
        format!("{node}: TensorRT requires CUDA execution"),
    )?;
    require(
        !model.outputs.is_empty(),
        format!("{node}: an ensemble step requires an output"),
    )?;
    for port in model.inputs.iter().chain(&model.outputs) {
        require(
            !port.name.is_empty() && !port.name.contains('\0') && !node.contains('\0'),
            format!(
                "{node}/{name}: invalid Triton tensor name",
                name = port.name
            ),
        )?;
        require(!port.shape.is_empty() && port.shape.iter().all(|&d| d > 0 || d == -1), format!("{node}/{name}: Triton configuration requires a nonempty shape with positive dimensions or -1", name=port.name))?;
    }
    let instance = match device {
        ExecutionDevice::Cpu => ModelInstanceGroup {
            kind: model_instance_group::Kind::KIND_CPU.into(),
            count: 1,
            ..Default::default()
        },
        ExecutionDevice::Cuda { device_id } => ModelInstanceGroup {
            kind: model_instance_group::Kind::KIND_GPU.into(),
            count: 1,
            gpus: vec![i32::try_from(device_id)
                .context("CUDA device ID exceeds Triton's integer range")?],
            ..Default::default()
        },
    };
    let mut version_policy = ModelVersionPolicy::new();
    version_policy.set_specific(model_version_policy::Specific {
        versions: vec![MODEL_VERSION],
        ..Default::default()
    });
    let mut config = ModelConfig {
        name: format!("leapp_node_{index}"),
        backend: match model.parameters.backend.as_str() {
            "jit" => "leapp_torchscript",
            "pt2" => "leapp_pt2",
            "trt" => "tensorrt",
            _ => "onnxruntime",
        }
        .into(),
        platform: match model.parameters.backend.as_str() {
            "trt" => "tensorrt_plan",
            _ => "",
        }
        .into(),
        default_model_filename: model
            .parameters
            .model_path
            .file_name()
            .and_then(|s| s.to_str())
            .context("model filename must be UTF-8")?
            .into(),
        version_policy: Some(version_policy).into(),
        instance_group: vec![instance],
        input: model.inputs.iter().map(|p| input(p, &p.name)).collect(),
        output: model.outputs.iter().map(|p| output(p, &p.name)).collect(),
        ..Default::default()
    };
    // TorchScript arguments and results follow the exported port order.
    if model.parameters.backend == "jit" {
        for (index, port) in config.input.iter_mut().enumerate() {
            port.name = format!("INPUT__{index}");
        }
        for (index, port) in config.output.iter_mut().enumerate() {
            port.name = format!("OUTPUT__{index}");
        }
    }
    Ok(config)
}

pub(super) fn input(port: &PortSpec, name: &str) -> ModelInput {
    ModelInput {
        name: name.into(),
        data_type: dtype(port.dtype).into(),
        dims: port.shape.clone(),
        ..Default::default()
    }
}

pub(super) fn output(port: &PortSpec, name: &str) -> ModelOutput {
    ModelOutput {
        name: name.into(),
        data_type: dtype(port.dtype).into(),
        dims: port.shape.clone(),
        ..Default::default()
    }
}

fn dtype(dtype: BundleDataType) -> DataType {
    match dtype {
        BundleDataType::Bool => DataType::TYPE_BOOL,
        BundleDataType::Uint8 => DataType::TYPE_UINT8,
        BundleDataType::Uint16 => DataType::TYPE_UINT16,
        BundleDataType::Uint32 => DataType::TYPE_UINT32,
        BundleDataType::Uint64 => DataType::TYPE_UINT64,
        BundleDataType::Int8 => DataType::TYPE_INT8,
        BundleDataType::Int16 => DataType::TYPE_INT16,
        BundleDataType::Int32 => DataType::TYPE_INT32,
        BundleDataType::Int64 => DataType::TYPE_INT64,
        BundleDataType::Float16 => DataType::TYPE_FP16,
        BundleDataType::Float32 => DataType::TYPE_FP32,
        BundleDataType::Float64 => DataType::TYPE_FP64,
        BundleDataType::Bfloat16 => DataType::TYPE_BF16,
    }
}
