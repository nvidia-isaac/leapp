// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::execution::repository::ModelRepository;
use crate::{
    execution::{convert_bundle, ConversionOptions, ExecutionDevice, TritonPipeline},
    Bundle, ErrorCode,
};
use leapp_triton::{DataType, MemoryLocation};
use std::path::PathBuf;
use std::{env, path::Path, thread};

fn devices() -> Vec<ExecutionDevice> {
    vec![ExecutionDevice::Cpu, ExecutionDevice::Cuda { device_id: 0 }]
}

fn options() -> TritonOptions {
    let root = env::var_os("TRITON_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/opt/tritonserver".into());
    TritonOptions {
        backend_directory: env::var_os("LEAPP_TRITON_BACKENDS")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("backends")),
    }
}

fn pipeline(name: &str, device: ExecutionDevice) -> TritonPipeline {
    let fixtures = env::var_os("LEAPP_TRITON_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/triton-fixtures"));
    let bundle = Bundle::open(fixtures.join(format!("{name}.yaml")))
        .expect("generate ONNX fixtures with tests/generate_models.py first");
    convert_bundle(&bundle, &ConversionOptions { device }).unwrap()
}

fn open(name: &str, device: ExecutionDevice) -> (TritonRuntime, TritonPipeline) {
    let pipeline = pipeline(name, device);
    let repository = ModelRepository::create(&pipeline).unwrap();
    (
        TritonRuntime::open(repository, &options()).unwrap(),
        pipeline,
    )
}

fn tensor<'a>(outputs: &'a [(String, Tensor)], name: &str) -> &'a Tensor {
    &outputs.iter().find(|(key, _)| key == name).unwrap().1
}

fn values(outputs: &[(String, Tensor)], name: &str) -> Vec<f32> {
    tensor(outputs, name).copy_to_cpu().unwrap()
}

#[test]
fn executes_ensemble_with_external_weights_and_retains_outputs_after_drop() {
    for device in devices() {
        let (runtime, pipeline) = open("policy", device);
        let repository = runtime._repository.path().to_owned();
        let obs = [1.0f32, 2.0, 3.0, 4.0];
        let bias = [10.0f32, 20.0, 30.0, 40.0];
        let inputs = [
            (
                pipeline.inputs[0].tensor_name.as_str(),
                TensorView::cpu(&obs, &[4]).unwrap(),
            ),
            (
                pipeline.inputs[1].tensor_name.as_str(),
                TensorView::cpu(&bias, &[4]).unwrap(),
            ),
        ];
        let mut outputs = Vec::new();
        for _ in 0..3 {
            outputs = runtime
                .infer(
                    &pipeline.ensemble.name,
                    &inputs,
                    &OutputPlacement::default(),
                )
                .unwrap();
            assert_eq!(
                values(&outputs, &pipeline.outputs[0].tensor_name),
                [14.0, 29.0, 46.0, 65.0]
            );
            assert_eq!(values(&outputs, "policy/echo"), [-8.0, -17.0, -26.0, -35.0]);
        }
        drop(runtime);
        assert!(!repository.exists());
        assert_eq!(
            tensor(&outputs, &pipeline.outputs[0].tensor_name)
                .as_cpu_slice::<f32>()
                .unwrap(),
            [14.0, 29.0, 46.0, 65.0]
        );
        assert!(tensor(&outputs, &pipeline.outputs[0].tensor_name)
            .as_cpu_slice::<i32>()
            .is_err());
    }
}

#[test]
fn concurrent_requests_have_independent_inputs_and_results() {
    for device in devices() {
        let (runtime, pipeline) = open("dynamic", device);
        thread::scope(|scope| {
            for worker in 0..4 {
                let (runtime, pipeline) = (&runtime, &pipeline);
                scope.spawn(move || {
                    for call in 0..8 {
                        let mut data = vec![(worker * 8 + call) as f32; 7];
                        let outputs = runtime
                            .infer(
                                &pipeline.ensemble.name,
                                &[(
                                    pipeline.inputs[0].tensor_name.as_str(),
                                    TensorView::cpu(&data, &[7]).unwrap(),
                                )],
                                &OutputPlacement::default(),
                            )
                            .unwrap();
                        let expected = data.clone();
                        data.fill(-1.0);
                        assert_eq!(values(&outputs, &pipeline.outputs[0].tensor_name), expected);
                    }
                });
            }
        });
    }
}

#[test]
fn caller_supplies_defaults_and_carries_feedback_without_copying_results() {
    for device in devices() {
        let (runtime, pipeline) = open("defaults", device);
        let obs = [1.0f32, 2.0, 3.0, 4.0];
        let zeros = [0.0f32; 4];
        let inputs = [
            (
                pipeline.inputs[0].tensor_name.as_str(),
                TensorView::cpu(&obs, &[4]).unwrap(),
            ),
            (
                pipeline.default_inputs[0].input.tensor_name.as_str(),
                TensorView::cpu(&zeros, &[4]).unwrap(),
            ),
        ];
        let outputs = runtime
            .infer(
                &pipeline.ensemble.name,
                &inputs,
                &OutputPlacement::default(),
            )
            .unwrap();
        assert_eq!(
            values(&outputs, &pipeline.outputs[0].tensor_name),
            [4.0, 9.0, 16.0, 25.0]
        );
        drop(runtime);

        let (runtime, pipeline) = open("feedback", device);
        let location = match device {
            ExecutionDevice::Cpu => MemoryLocation::Cpu,
            ExecutionDevice::Cuda { device_id } => MemoryLocation::Cuda { device_id },
        };
        let obs = Tensor::copy_from_cpu(&obs, &[4], location).unwrap();
        let mut state = Tensor::copy_from_cpu(&zeros, &[4], location).unwrap();
        let placement = OutputPlacement {
            default_location: location,
            ..Default::default()
        };
        for step in 1..=3 {
            let inputs = [
                (pipeline.inputs[0].tensor_name.as_str(), obs.view()),
                (pipeline.feedback[0].input_name.as_str(), state.view()),
            ];
            let outputs = runtime
                .infer(&pipeline.ensemble.name, &inputs, &placement)
                .unwrap();
            state = tensor(&outputs, &pipeline.feedback[0].output_name).clone();
            assert_eq!(state.location(), location);
            assert_eq!(
                state.copy_to_cpu::<f32>().unwrap(),
                (1..=4).map(|i| (i * step) as f32).collect::<Vec<_>>()
            );
        }
        drop(runtime);
        assert_eq!(state.copy_to_cpu::<f32>().unwrap(), [3.0, 6.0, 9.0, 12.0]);
    }
}

#[test]
fn dynamic_and_empty_outputs_have_concrete_shapes() {
    for device in devices() {
        let (runtime, pipeline) = open("dynamic", device);
        for size in [0, 1, 7] {
            let data = vec![2.0f32; size];
            let shape = [size as i64];
            let outputs = runtime
                .infer(
                    &pipeline.ensemble.name,
                    &[(
                        pipeline.inputs[0].tensor_name.as_str(),
                        TensorView::cpu(&data, &shape).unwrap(),
                    )],
                    &OutputPlacement::default(),
                )
                .unwrap();
            let output = tensor(&outputs, &pipeline.outputs[0].tensor_name);
            assert_eq!(output.dtype(), DataType::Float32);
            assert_eq!(output.shape(), shape);
            assert_eq!(output.as_cpu_slice::<f32>().unwrap(), data);
        }
    }
}

#[test]
fn invalid_requests_and_backend_errors_allow_subsequent_inference() {
    for device in devices() {
        let (runtime, pipeline) = open("dynamic", device);
        let model = &pipeline.ensemble.name;
        let name = pipeline.inputs[0].tensor_name.as_str();
        let good = TensorView::cpu(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let wrong_type = TensorView::cpu(&[1i32, 2, 3, 4], &[4]).unwrap();
        let placement = OutputPlacement::default();
        assert!(runtime.infer(model, &[], &placement).is_err());
        assert_eq!(
            runtime
                .infer("missing", &[], &placement)
                .err()
                .unwrap()
                .code(),
            ErrorCode::NotFound
        );
        for inputs in [
            vec![(name, wrong_type)],
            vec![("missing", good)],
            vec![("bad\0name", good)],
            vec![(name, good), (name, good)],
        ] {
            assert!(runtime.infer(model, &inputs, &placement).is_err());
        }
        let unknown_output = OutputPlacement {
            overrides: [("missing".into(), MemoryLocation::Cpu)].into(),
            ..Default::default()
        };
        assert_eq!(
            runtime
                .infer(model, &[(name, good)], &unknown_output)
                .err()
                .unwrap()
                .code(),
            ErrorCode::InvalidArgument
        );
        let outputs = runtime.infer(model, &[(name, good)], &placement).unwrap();
        assert_eq!(
            values(&outputs, &pipeline.outputs[0].tensor_name),
            [1.0, 2.0, 3.0, 4.0]
        );
        drop(runtime);

        let (runtime, pipeline) = open("reshape", device);
        let name = pipeline.inputs[0].tensor_name.as_str();
        let bad = TensorView::cpu(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let error = runtime
            .infer(&pipeline.ensemble.name, &[(name, bad)], &placement)
            .err()
            .unwrap();
        assert_eq!(error.phase(), "execute");
        assert!(error.to_string().contains("Reshape"), "{error}");
        let outputs = runtime
            .infer(&pipeline.ensemble.name, &[(name, good)], &placement)
            .unwrap();
        assert_eq!(
            values(&outputs, &pipeline.outputs[0].tensor_name),
            [1.0, 2.0, 3.0, 4.0]
        );
    }
}

#[test]
fn load_failure_cleans_repository_and_does_not_prevent_reopening() {
    let mut invalid = pipeline("dynamic", ExecutionDevice::Cpu);
    invalid.models[0].config.default_model_filename = "missing.onnx".into();
    let repository = ModelRepository::create(&invalid).unwrap();
    let path = repository.path().to_owned();
    let error = TritonRuntime::open(repository, &options()).err().unwrap();
    assert_eq!(error.phase(), "load");
    assert!(!path.exists());
    let (runtime, _) = open("dynamic", ExecutionDevice::Cpu);
    assert!(runtime._repository.path().exists());
}

#[test]
fn all_input_output_memory_combinations_work_on_cpu_and_cuda_models() {
    let locations = [
        MemoryLocation::Cpu,
        MemoryLocation::PinnedCpu,
        MemoryLocation::Cuda { device_id: 0 },
    ];
    for device in devices() {
        let (runtime, pipeline) = open("dynamic", device);
        for input_location in locations {
            for output_location in locations {
                for size in [0, 4] {
                    let data = vec![3.0f32; size];
                    let input =
                        Tensor::copy_from_cpu(&data, &[size as i64], input_location).unwrap();
                    let placement = OutputPlacement {
                        default_location: output_location,
                        ..Default::default()
                    };
                    let outputs = runtime
                        .infer(
                            &pipeline.ensemble.name,
                            &[(pipeline.inputs[0].tensor_name.as_str(), input.view())],
                            &placement,
                        )
                        .unwrap();
                    let output = tensor(&outputs, &pipeline.outputs[0].tensor_name);
                    assert_eq!(output.location(), output_location);
                    assert_eq!(output.copy_to_cpu::<f32>().unwrap(), data);
                    if matches!(output_location, MemoryLocation::Cuda { .. }) {
                        assert!(output.as_cpu_slice::<f32>().is_err());
                    } else {
                        assert_eq!(output.as_cpu_slice::<f32>().unwrap(), data);
                    }
                }
            }
        }
    }
}

#[test]
fn outputs_can_use_distinct_memory_locations() {
    for device in devices() {
        let (runtime, pipeline) = open("policy", device);
        let bias = Tensor::copy_from_cpu(
            &[10.0f32, 20.0, 30.0, 40.0],
            &[4],
            MemoryLocation::Cuda { device_id: 0 },
        )
        .unwrap();
        let obs = TensorView::cpu(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let inputs = [
            (pipeline.inputs[0].tensor_name.as_str(), obs),
            (pipeline.inputs[1].tensor_name.as_str(), bias.view()),
        ];
        let placement = OutputPlacement {
            default_location: MemoryLocation::Cuda { device_id: 0 },
            overrides: [("policy/echo".into(), MemoryLocation::PinnedCpu)].into(),
        };
        let outputs = runtime
            .infer(&pipeline.ensemble.name, &inputs, &placement)
            .unwrap();
        assert_eq!(
            tensor(&outputs, "policy/echo").location(),
            MemoryLocation::PinnedCpu
        );
        assert_eq!(
            tensor(&outputs, "policy/action").location(),
            MemoryLocation::Cuda { device_id: 0 }
        );
        assert_eq!(values(&outputs, "policy/action"), [14.0, 29.0, 46.0, 65.0]);
        let invalid = OutputPlacement {
            default_location: MemoryLocation::Cuda {
                device_id: u32::MAX,
            },
            ..Default::default()
        };
        let error = runtime
            .infer(&pipeline.ensemble.name, &inputs, &invalid)
            .err()
            .unwrap();
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
        assert_eq!(values(&outputs, "policy/echo"), [-8.0, -17.0, -26.0, -35.0]);
    }
}
