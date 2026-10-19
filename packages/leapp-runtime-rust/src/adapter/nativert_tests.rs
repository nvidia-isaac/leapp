// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{error::RuntimeError, execution::repository::ModelRepository};
use crate::{
    execution::{convert_bundle, ConversionOptions, ExecutionDevice, TritonPipeline},
    Bundle,
};
use leapp_triton::MemoryLocation;
use std::path::PathBuf;
use std::{env, thread};

fn devices() -> Vec<ExecutionDevice> {
    vec![ExecutionDevice::Cpu, ExecutionDevice::Cuda { device_id: 0 }]
}

fn locations() -> Vec<MemoryLocation> {
    vec![
        MemoryLocation::Cpu,
        MemoryLocation::PinnedCpu,
        MemoryLocation::Cuda { device_id: 0 },
    ]
}

fn options() -> TritonOptions {
    TritonOptions {
        backend_directory: env::var_os("LEAPP_TRITON_BACKENDS")
            .map(PathBuf::from)
            .expect("set LEAPP_TRITON_BACKENDS to a directory containing leapp_pt2"),
    }
}

fn pipeline(name: &str, device: ExecutionDevice) -> TritonPipeline {
    let directory = env::var_os("LEAPP_PT2_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/pt2-fixtures"));
    let suffix = match device {
        ExecutionDevice::Cpu => "cpu",
        ExecutionDevice::Cuda { .. } => "cuda",
    };
    let bundle = Bundle::open(directory.join(format!("{name}_{suffix}.yaml")))
        .expect("generate fixtures with tests/generate_pt2_models.py first");
    convert_bundle(&bundle, &ConversionOptions { device }).unwrap()
}

fn open(pipeline: &TritonPipeline) -> Result<TritonRuntime, RuntimeError> {
    TritonRuntime::open(ModelRepository::create(pipeline).unwrap(), &options())
}

fn output<'a>(outputs: &'a [(String, Tensor)], name: &str) -> &'a Tensor {
    &outputs.iter().find(|(key, _)| key == name).unwrap().1
}

#[test]
#[ignore = "requires the NativeRT backend and generated PT2 fixtures"]
fn weights_noncontiguous_outputs_and_memory_placements() {
    for device in devices() {
        let pipeline = pipeline("weighted", device);
        let runtime = open(&pipeline).unwrap();
        let mut retained = Vec::new();
        for input_location in locations() {
            let input =
                Tensor::copy_from_cpu(&[1.0f32, 2.0, 3.0, 4.0], &[4], input_location).unwrap();
            for output_location in locations() {
                let placement = OutputPlacement {
                    default_location: output_location,
                    ..Default::default()
                };
                retained = runtime
                    .infer(
                        &pipeline.ensemble.name,
                        &[(pipeline.inputs[0].tensor_name.as_str(), input.view())],
                        &placement,
                    )
                    .unwrap();
                let weighted = output(&retained, "model/output_0");
                assert_eq!(weighted.location(), output_location);
                assert_eq!(
                    weighted.copy_to_cpu::<f32>().unwrap(),
                    [12.0, 26.0, 42.0, 60.0]
                );
                let transposed = output(&retained, "model/output_1");
                assert_eq!(transposed.shape(), [2, 2]);
                assert_eq!(transposed.location(), output_location);
                assert_eq!(
                    transposed.copy_to_cpu::<f32>().unwrap(),
                    [1.0, 3.0, 2.0, 4.0]
                );
            }
        }
        let repository = runtime._repository.path().to_owned();
        drop(runtime);
        assert!(!repository.exists());
        assert_eq!(
            output(&retained, "model/output_0")
                .copy_to_cpu::<f32>()
                .unwrap(),
            [12.0, 26.0, 42.0, 60.0]
        );
    }
}

#[test]
#[ignore = "requires the NativeRT backend and generated PT2 fixtures"]
fn mutation_is_contained_and_concurrent_requests_are_independent() {
    for device in devices() {
        let pipeline = pipeline("mutating", device);
        let runtime = open(&pipeline).unwrap();
        thread::scope(|scope| {
            for worker in 0..4 {
                let (runtime, pipeline) = (&runtime, &pipeline);
                scope.spawn(move || {
                    for location in locations() {
                        let data = [worker as f32; 4];
                        let input = Tensor::copy_from_cpu(&data, &[4], location).unwrap();
                        for _ in 0..3 {
                            let outputs = runtime
                                .infer(
                                    &pipeline.ensemble.name,
                                    &[(pipeline.inputs[0].tensor_name.as_str(), input.view())],
                                    &OutputPlacement::default(),
                                )
                                .unwrap();
                            assert_eq!(
                                output(&outputs, "model/output_0")
                                    .copy_to_cpu::<f32>()
                                    .unwrap(),
                                [(worker as f32 + 2.0) * 3.0; 4]
                            );
                            assert_eq!(input.copy_to_cpu::<f32>().unwrap(), data);
                        }
                    }
                });
            }
        });
    }
}

#[test]
#[ignore = "requires the NativeRT backend and generated PT2 fixtures"]
fn dynamic_shapes_backend_error_and_recovery() {
    for device in devices() {
        let pipeline = pipeline("reshape", device);
        let runtime = open(&pipeline).unwrap();
        let name = pipeline.inputs[0].tensor_name.as_str();
        let bad = TensorView::cpu(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let error = runtime
            .infer(
                &pipeline.ensemble.name,
                &[(name, bad)],
                &OutputPlacement::default(),
            )
            .err()
            .unwrap();
        assert_eq!(error.phase(), "execute");
        for size in [0, 2, 8] {
            let data = vec![7.0f32; size];
            let shape = [size as i64];
            let outputs = runtime
                .infer(
                    &pipeline.ensemble.name,
                    &[(name, TensorView::cpu(&data, &shape).unwrap())],
                    &OutputPlacement::default(),
                )
                .unwrap();
            let value = output(&outputs, "model/output_0");
            assert_eq!(value.shape(), [size as i64 / 2, 2]);
            assert_eq!(value.copy_to_cpu::<f32>().unwrap(), data);
        }
    }
}

#[test]
#[ignore = "requires the NativeRT backend and generated PT2 fixtures"]
fn integer_tensors() {
    for device in devices() {
        let pipeline = pipeline("integer", device);
        let runtime = open(&pipeline).unwrap();
        let outputs = runtime
            .infer(
                &pipeline.ensemble.name,
                &[(
                    pipeline.inputs[0].tensor_name.as_str(),
                    TensorView::cpu(&[1i64, 2, 3, 4], &[4]).unwrap(),
                )],
                &OutputPlacement::default(),
            )
            .unwrap();
        assert_eq!(
            output(&outputs, "model/output_0")
                .copy_to_cpu::<i64>()
                .unwrap(),
            [2, 3, 4, 5]
        );
    }
}

#[test]
#[ignore = "requires the NativeRT backend and generated PT2 fixtures"]
fn incompatible_archive_devices_are_rejected() {
    let mut cpu = pipeline("weighted", ExecutionDevice::Cpu);
    let cuda = pipeline("weighted", ExecutionDevice::Cuda { device_id: 0 });
    cpu.models[0].artifact.model_path = cuda.models[0].artifact.model_path.clone();
    let error = open(&cpu).err().unwrap();
    assert_eq!(error.phase(), "load");
    // Triton logs the backend diagnostic and returns an aggregate load error.
    assert!(error.to_string().contains("failed to load"), "{error}");
}

#[test]
#[ignore = "requires NativeRT, ONNX Runtime, and both sets of generated fixtures"]
fn mixed_onnx_pt2_ensemble() {
    let onnx_fixtures = env::var_os("LEAPP_TRITON_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/triton-fixtures")
        });
    for device in devices() {
        let pt2 = pipeline("weighted", device);
        let port = |name: &str, shape: &[i64]| {
            serde_json::json!({
                "name": name, "dtype": "float32", "shape": shape, "type": "tensor"
            })
        };
        let value = serde_json::json!({
            "models": {
                "preprocess": {
                    "inputs": [port("observation", &[4])],
                    "outputs": [port("features", &[4])],
                    "parameters": {"backend": "onnx", "model_path": onnx_fixtures.join("preprocess/model.onnx")}
                },
                "weighted": {
                    "inputs": [port("values", &[4])],
                    "outputs": [port("output_0", &[4]), port("output_1", &[2, 2])],
                    "parameters": {"backend": "pt2", "model_path": pt2.models[0].artifact.model_path}
                }
            },
            "pipeline": {
                "data_flow": {"preprocess/features": ["weighted/values"]},
                "feedback_flow": {},
                "inputs": {"preprocess": ["observation"]},
                "outputs": {"weighted": ["output_0", "output_1"]},
                "configs": {}
            },
            "system information": {"leapp config version": "1.3"}
        });
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("mixed.yaml");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let bundle = Bundle::open(path).unwrap();
        let pipeline = convert_bundle(&bundle, &ConversionOptions { device }).unwrap();
        let runtime = open(&pipeline).unwrap();
        let outputs = runtime
            .infer(
                &pipeline.ensemble.name,
                &[(
                    pipeline.inputs[0].tensor_name.as_str(),
                    TensorView::cpu(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap(),
                )],
                &OutputPlacement::default(),
            )
            .unwrap();
        assert_eq!(
            output(&outputs, "weighted/output_0")
                .copy_to_cpu::<f32>()
                .unwrap(),
            [14.0, 29.0, 46.0, 65.0]
        );
        assert_eq!(
            output(&outputs, "weighted/output_1")
                .copy_to_cpu::<f32>()
                .unwrap(),
            [2.0, 4.0, 3.0, 5.0]
        );
    }
}
