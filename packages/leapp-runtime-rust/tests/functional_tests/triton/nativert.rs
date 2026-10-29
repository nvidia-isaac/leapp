// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::dlpack::{BorrowedTensor, SharedTensor};
use crate::engines::memory::OutputPool;
use crate::engines::triton::tests::support::{placed, TestTensor};
use crate::engines::triton::*;
use crate::MemoryLocation;
use crate::{engines::triton::convert::repository::ModelRepository, error::RuntimeError};
use crate::{
    engines::triton::convert::{convert_bundle, TritonPipeline},
    Bundle,
};
use crate::{ExecutionDevice, OutputPoolOptions, Session, SessionOptions};
use std::path::PathBuf;
use std::{collections::HashMap, env, sync::Arc, thread};

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

fn options() -> Options {
    Options {
        backend_directory: env::var_os("LEAPP_TRITON_BACKENDS")
            .map(PathBuf::from)
            .expect("set LEAPP_TRITON_BACKENDS to a directory containing leapp_pt2"),
    }
}

fn bundle(name: &str, device: ExecutionDevice) -> Bundle {
    let directory = env::var_os("LEAPP_PT2_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/pt2-fixtures"));
    let suffix = match device {
        ExecutionDevice::Cpu => "cpu",
        ExecutionDevice::Cuda { .. } => "cuda",
    };
    Bundle::open(directory.join(format!("{name}_{suffix}.yaml")))
        .expect("generate fixtures with tests/generators/generate_pt2_models.py first")
}

fn pipeline(name: &str, device: ExecutionDevice) -> TritonPipeline {
    convert_bundle(&bundle(name, device), device).unwrap()
}

fn open(pipeline: &TritonPipeline) -> std::result::Result<TritonEngine, RuntimeError> {
    TritonEngine::load(
        pipeline,
        ModelRepository::create(pipeline).unwrap(),
        &options(),
    )
}

fn output<'a>(outputs: &'a [(String, SharedTensor)], name: &str) -> &'a SharedTensor {
    &outputs.iter().find(|(key, _)| key == name).unwrap().1
}

#[test]
#[ignore = "requires the NativeRT backend and generated PT2 fixtures"]
fn session_pool_limits_preserve_peer_and_retained_outputs() {
    for device in devices() {
        let bundle = bundle("weighted", device);
        for max_entries in [0, 1, 64] {
            let mut session = Session::open(
                &bundle,
                SessionOptions {
                    engine: crate::EngineKind::Triton,
                    cuda_streams: Default::default(),
                    device,
                    output_placement: OutputPlacement {
                        default_location: match device {
                            ExecutionDevice::Cpu => MemoryLocation::Cpu,
                            ExecutionDevice::Cuda { device_id } => {
                                MemoryLocation::Cuda { device_id }
                            }
                        },
                        ..Default::default()
                    },
                    output_pool: OutputPoolOptions { max_entries },
                },
            )
            .unwrap();
            let mut peer = session.new_session().unwrap();
            let input = |values| {
                HashMap::from([("model/values", BorrowedTensor::cpu(values, &[4]).unwrap())])
            };
            let retained = session.run(&input(&[1_f32, 2., 3., 4.])).unwrap();
            let read = |result: &crate::RunResult| {
                result
                    .get("model/output_0")
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap()
            };
            for _ in 0..3 {
                assert_eq!(
                    read(&session.run(&input(&[2_f32; 4])).unwrap()),
                    [14., 26., 38., 50.]
                );
                assert_eq!(
                    read(&peer.run(&input(&[3_f32; 4])).unwrap()),
                    [16., 29., 42., 55.]
                );
                assert_eq!(read(&retained), [12., 26., 42., 60.]);
            }
            session.close();
            assert_eq!(
                read(&peer.run(&input(&[3_f32; 4])).unwrap()),
                [16., 29., 42., 55.]
            );
            peer.close();
            assert_eq!(read(&retained), [12., 26., 42., 60.]);
        }
    }
}

#[test]
#[ignore = "requires the NativeRT backend and generated PT2 fixtures"]
fn weights_noncontiguous_outputs_and_memory_placements() {
    for device in devices() {
        let pipeline = pipeline("weighted", device);
        let runtime = open(&pipeline).unwrap();
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        let mut retained = Vec::new();
        for input_location in locations() {
            let input = placed(&[1.0f32, 2.0, 3.0, 4.0], &[4], input_location).unwrap();
            for output_location in locations() {
                let placement = OutputPlacement {
                    default_location: output_location,
                    ..Default::default()
                };
                retained = runtime
                    .native
                    .infer(
                        &pipeline.ensemble.name,
                        &[(pipeline.inputs[0].tensor_name.as_str(), input.view())],
                        &placement,
                        &pool,
                    )
                    .unwrap();
                let weighted = output(&retained, "model/output_0");
                assert_eq!(weighted.location(), output_location);
                assert_eq!(
                    weighted.to_cpu_vec::<f32>().unwrap(),
                    [12.0, 26.0, 42.0, 60.0]
                );
                let transposed = output(&retained, "model/output_1");
                assert_eq!(transposed.shape(), [2, 2]);
                assert_eq!(transposed.location(), output_location);
                assert_eq!(
                    transposed.to_cpu_vec::<f32>().unwrap(),
                    [1.0, 3.0, 2.0, 4.0]
                );
            }
        }
        let repository = runtime._repository.path().to_owned();
        drop(runtime);
        drop(pool);
        assert!(!repository.exists());
        assert_eq!(
            output(&retained, "model/output_0")
                .to_cpu_vec::<f32>()
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
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        thread::scope(|scope| {
            for worker in 0..4 {
                let (runtime, pipeline, pool) = (&runtime, &pipeline, &pool);
                scope.spawn(move || {
                    for location in locations() {
                        let data = [worker as f32; 4];
                        let input = placed(&data, &[4], location).unwrap();
                        for _ in 0..3 {
                            let outputs = runtime
                                .native
                                .infer(
                                    &pipeline.ensemble.name,
                                    &[(pipeline.inputs[0].tensor_name.as_str(), input.view())],
                                    &OutputPlacement::default(),
                                    pool,
                                )
                                .unwrap();
                            assert_eq!(
                                output(&outputs, "model/output_0")
                                    .to_cpu_vec::<f32>()
                                    .unwrap(),
                                [(worker as f32 + 2.0) * 3.0; 4]
                            );
                            assert_eq!(input.to_cpu_vec::<f32>().unwrap(), data);
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
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        let name = pipeline.inputs[0].tensor_name.as_str();
        let bad = BorrowedTensor::cpu(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let error = runtime
            .native
            .infer(
                &pipeline.ensemble.name,
                &[(name, bad)],
                &OutputPlacement::default(),
                &pool,
            )
            .err()
            .unwrap();
        assert_eq!(error.phase(), "execute");
        for size in [0, 2, 8] {
            let data = vec![7.0f32; size];
            let shape = [size as i64];
            let outputs = runtime
                .native
                .infer(
                    &pipeline.ensemble.name,
                    &[(name, BorrowedTensor::cpu(&data, &shape).unwrap())],
                    &OutputPlacement::default(),
                    &pool,
                )
                .unwrap();
            let value = output(&outputs, "model/output_0");
            assert_eq!(value.shape(), [size as i64 / 2, 2]);
            assert_eq!(value.to_cpu_vec::<f32>().unwrap(), data);
        }
    }
}

#[test]
#[ignore = "requires the NativeRT backend and generated PT2 fixtures"]
fn integer_tensors() {
    for device in devices() {
        let pipeline = pipeline("integer", device);
        let runtime = open(&pipeline).unwrap();
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        let outputs = runtime
            .native
            .infer(
                &pipeline.ensemble.name,
                &[(
                    pipeline.inputs[0].tensor_name.as_str(),
                    BorrowedTensor::cpu(&[1i64, 2, 3, 4], &[4]).unwrap(),
                )],
                &OutputPlacement::default(),
                &pool,
            )
            .unwrap();
        assert_eq!(
            output(&outputs, "model/output_0")
                .to_cpu_vec::<i64>()
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
    cpu.models[0].model_path = cuda.models[0].model_path.clone();
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
                    "parameters": {"backend": "pt2", "model_path": pt2.models[0].model_path}
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
        let pipeline = convert_bundle(&bundle, device).unwrap();
        let runtime = open(&pipeline).unwrap();
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        let outputs = runtime
            .native
            .infer(
                &pipeline.ensemble.name,
                &[(
                    pipeline.inputs[0].tensor_name.as_str(),
                    BorrowedTensor::cpu(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap(),
                )],
                &OutputPlacement::default(),
                &pool,
            )
            .unwrap();
        assert_eq!(
            output(&outputs, "weighted/output_0")
                .to_cpu_vec::<f32>()
                .unwrap(),
            [14.0, 29.0, 46.0, 65.0]
        );
        assert_eq!(
            output(&outputs, "weighted/output_1")
                .to_cpu_vec::<f32>()
                .unwrap(),
            [2.0, 4.0, 3.0, 5.0]
        );
    }
}
