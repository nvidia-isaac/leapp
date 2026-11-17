// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Bundle, EngineKind, ExecutionDevice, SessionOptions};
use anyhow::{ensure, Context, Result};
use std::collections::HashMap;

pub(crate) fn validate(bundle: &Bundle, options: &SessionOptions) -> Result<()> {
    for (node, parameters) in &options.onnx_tensorrt {
        let model = bundle
            .0
            .description
            .models
            .get(node)
            .with_context(|| format!("unknown ONNX TensorRT node {node}"))?;
        ensure!(
            model.parameters.backend == "onnx",
            "node {node}: TensorRT acceleration requires ONNX"
        );
        ensure!(
            matches!(options.device, ExecutionDevice::Cuda { .. }),
            "node {node}: TensorRT acceleration requires CUDA execution"
        );
        for (key, value) in parameters {
            // ORT graph replay retains addresses across borrowed runs and peers.
            ensure!(
                options.engine != EngineKind::Native
                    || key != "trt_cuda_graph_enable"
                    || matches!(value.as_str(), "0" | "false"),
                "node {node}: trt_cuda_graph_enable is unsupported by the native engine; \
                 CUDA graph capture requires stable input and output storage"
            );
            ensure!(
                !key.contains('\0') && !value.contains('\0'),
                "node {node}: TensorRT parameters must not contain NUL"
            );
            ensure!(
                !matches!(
                    key.as_str(),
                    "device_id" | "user_compute_stream" | "has_user_compute_stream"
                ),
                "node {node}: {key} is controlled by SessionOptions"
            );
        }
    }
    Ok(())
}

/// Match the aliases used by Triton's ONNX Runtime backend; ORT owns other options.
pub(crate) fn provider_options(
    parameters: &HashMap<String, String>,
) -> Result<HashMap<String, String>> {
    let mut result = HashMap::new();
    for (key, value) in parameters {
        let (key, value) = match key.as_str() {
            "precision_mode" => match value.as_str() {
                "FP32" => continue,
                "FP16" => ("trt_fp16_enable", "1"),
                "BF16" => ("trt_bf16_enable", "1"),
                "INT8" => ("trt_int8_enable", "1"),
                _ => anyhow::bail!("unsupported TensorRT precision_mode {value}"),
            },
            "max_workspace_size_bytes" => ("trt_max_workspace_size", value.as_str()),
            "int8_calibration_table_name" => ("trt_int8_calibration_table_name", value.as_str()),
            "int8_use_native_calibration_table" => {
                ("trt_int8_use_native_calibration_table", value.as_str())
            }
            _ => (key.as_str(), value.as_str()),
        };
        ensure!(
            result.insert(key.to_owned(), value.to_owned()).is_none(),
            "duplicate TensorRT option {key}"
        );
    }
    Ok(result)
}
