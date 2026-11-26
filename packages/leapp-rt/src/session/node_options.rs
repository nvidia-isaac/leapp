// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-node options routed to each node's backend.
//!
//! `SessionOptions::node_options` maps bundle node names to string options that
//! only that node's backend interprets, so session options never name a backend.
//! Keys are `name` or `namespace.name`; each backend defines its bare keys and
//! namespaces, which may be interpreted or forwarded to its SDK. Validation runs
//! before any model loads. Only ONNX accepts options so far.

use crate::{Bundle, EngineKind, ExecutionDevice, SessionOptions};
use anyhow::{bail, ensure, Context, Result};
use std::{collections::HashMap, path::Path};

pub(crate) fn validate(bundle: &Bundle, options: &SessionOptions) -> Result<()> {
    for (node, parameters) in &options.node_options {
        let model = bundle
            .0
            .description
            .models
            .get(node)
            .with_context(|| format!("node options for unknown node {node}"))?;
        let backend = model.parameters.backend.as_str();
        if backend != "onnx" {
            ensure!(
                parameters.is_empty(),
                "node {node}: {backend} nodes accept no node options"
            );
            continue;
        }
        let onnx = Onnx::parse(parameters).with_context(|| format!("node {node}"))?;
        if parameters.is_empty() {
            continue;
        }
        ensure!(
            matches!(options.device, ExecutionDevice::Cuda { .. }),
            "node {node}: ONNX node options require CUDA execution"
        );
        if options.engine == EngineKind::Triton {
            // Triton configures ONNX Runtime only through TensorRT accelerator parameters.
            ensure!(
                onnx.cuda_options.is_empty() && onnx.engine.is_none() && !onnx.replay,
                "node {node}: cuda.*, engine_path and cuda_graph_replay require native execution"
            );
        }
        if let Some(engine) = onnx.engine {
            ensure!(
                engine.is_file(),
                "node {node}: TensorRT engine {} is missing or not a regular file",
                engine.display()
            );
        }
    }
    Ok(())
}

/// ONNX Runtime selections parsed from one ONNX node's options.
///
/// Bare keys are LEAPP settings. `tensorrt.*` and `cuda.*` configure ONNX
/// Runtime's TensorRT and CUDA providers: keys are forwarded without the prefix,
/// except Triton-style TensorRT aliases, which are translated to ORT's names.
#[derive(Default)]
pub(crate) struct Onnx<'a> {
    /// `execution_provider: tensorrt`; otherwise the CUDA provider alone.
    pub tensorrt: bool,
    /// Prebuilt TensorRT engine replacing conversion from the ONNX model.
    pub engine: Option<&'a Path>,
    /// CUDA graph replay through the active provider's flag.
    pub replay: bool,
    /// TensorRT provider options using ORT's names.
    pub tensorrt_options: HashMap<String, String>,
    /// CUDA provider options, also applied when it is TensorRT's fallback.
    pub cuda_options: HashMap<String, String>,
}

impl<'a> Onnx<'a> {
    pub fn parse(parameters: &'a HashMap<String, String>) -> Result<Self> {
        let tensorrt = match parameters.get("execution_provider").map(String::as_str) {
            None | Some("cuda") => false,
            Some("tensorrt") => true,
            Some(other) => bail!("execution_provider must be cuda or tensorrt; got {other}"),
        };
        let mut result = Self {
            tensorrt,
            ..Default::default()
        };
        for (key, value) in parameters {
            ensure!(
                !key.contains('\0') && !value.contains('\0'),
                "node options must not contain NUL"
            );
            let (options, name, value) = match key.split_once('.') {
                None => {
                    match key.as_str() {
                        "execution_provider" => {}
                        "engine_path" => {
                            ensure!(tensorrt, "engine_path requires execution_provider tensorrt");
                            result.engine = Some(Path::new(value));
                        }
                        "cuda_graph_replay" => {
                            result.replay = match value.as_str() {
                                "0" | "false" => false,
                                "1" | "true" => true,
                                _ => bail!("{key} must be 0, 1, false or true; got {value}"),
                            };
                        }
                        _ => bail!("unknown ONNX node option {key}"),
                    }
                    continue;
                }
                Some(("tensorrt", name)) => {
                    ensure!(tensorrt, "{key} requires execution_provider tensorrt");
                    // Match the aliases used by Triton's ONNX Runtime backend.
                    let (name, value) = match name {
                        "precision_mode" => match value.as_str() {
                            "FP32" => continue,
                            "FP16" => ("trt_fp16_enable", "1"),
                            "BF16" => ("trt_bf16_enable", "1"),
                            "INT8" => ("trt_int8_enable", "1"),
                            _ => bail!("unsupported TensorRT precision_mode {value}"),
                        },
                        "max_workspace_size_bytes" => ("trt_max_workspace_size", value.as_str()),
                        "int8_calibration_table_name" => {
                            ("trt_int8_calibration_table_name", value.as_str())
                        }
                        "int8_use_native_calibration_table" => {
                            ("trt_int8_use_native_calibration_table", value.as_str())
                        }
                        name => (name, value.as_str()),
                    };
                    (&mut result.tensorrt_options, name, value)
                }
                Some(("cuda", name)) => (&mut result.cuda_options, name, value.as_str()),
                Some((namespace, _)) => bail!("unknown ONNX node option namespace {namespace}"),
            };
            ensure!(
                !matches!(
                    name,
                    "device_id" | "user_compute_stream" | "has_user_compute_stream"
                ),
                "{key} is controlled by SessionOptions"
            );
            ensure!(
                !matches!(name, "trt_cuda_graph_enable" | "enable_cuda_graph"),
                "{key} is controlled by cuda_graph_replay"
            );
            ensure!(
                options.insert(name.to_owned(), value.to_owned()).is_none(),
                "duplicate provider option {key}"
            );
        }
        Ok(result)
    }
}
