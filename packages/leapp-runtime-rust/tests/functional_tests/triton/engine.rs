// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::dlpack::{BorrowedTensor, SharedTensor};
use crate::engines::memory::OutputPool;
use crate::engines::triton::convert::repository::ModelRepository;
use crate::engines::triton::tests::support::{placed, TestTensor};
use crate::engines::triton::*;
use crate::MemoryLocation;
use crate::{
    engines::triton::convert::{convert_bundle, TritonPipeline},
    Bundle,
};
use crate::{ExecutionDevice, OutputPoolOptions};
use std::path::PathBuf;
use std::sync::Arc;
use std::{env, path::Path, thread};

fn devices() -> Vec<ExecutionDevice> {
    vec![ExecutionDevice::Cpu, ExecutionDevice::Cuda { device_id: 0 }]
}

fn pipeline(name: &str, device: ExecutionDevice) -> TritonPipeline {
    let fixtures = env::var_os("LEAPP_TRITON_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("target/triton-fixtures")
        });
    let bundle = Bundle::open(fixtures.join(format!("{name}.yaml")))
        .expect("generate ONNX fixtures with tests/generators/generate_models.py first");
    convert_bundle(&bundle, device).unwrap()
}

fn open(name: &str, device: ExecutionDevice) -> (TritonEngine, TritonPipeline) {
    let pipeline = pipeline(name, device);
    let repository = ModelRepository::create(&pipeline).unwrap();
    (
        TritonEngine::load(&pipeline, repository, &Options::default()).unwrap(),
        pipeline,
    )
}

fn tensor<'a>(outputs: &'a [(String, SharedTensor)], name: &str) -> &'a SharedTensor {
    &outputs.iter().find(|(key, _)| key == name).unwrap().1
}

fn values(outputs: &[(String, SharedTensor)], name: &str) -> Vec<f32> {
    tensor(outputs, name).to_cpu_vec().unwrap()
}

#[test]
fn executes_ensemble_with_external_weights_and_retains_outputs_after_drop() {
    for device in devices() {
        let (runtime, pipeline) = open("policy", device);
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        let repository = runtime._repository.path().to_owned();
        let obs = [1.0f32, 2.0, 3.0, 4.0];
        let bias = [10.0f32, 20.0, 30.0, 40.0];
        let bias_dl = BorrowedTensor::cpu(&bias, &[4]).unwrap();
        let obs_dl = BorrowedTensor::cpu(&obs, &[4]).unwrap();
        let inputs = [
            (pipeline.inputs[0].tensor_name.as_str(), obs_dl),
            (pipeline.inputs[1].tensor_name.as_str(), bias_dl),
        ];
        let mut outputs = Vec::new();
        for _ in 0..3 {
            outputs = runtime
                .native
                .infer(
                    &pipeline.ensemble.name,
                    &inputs,
                    &OutputPlacement::default(),
                    &pool,
                )
                .unwrap();
            assert_eq!(
                values(&outputs, &pipeline.ensemble.output[0].name),
                [14.0, 29.0, 46.0, 65.0]
            );
            assert_eq!(values(&outputs, "policy/echo"), [-8.0, -17.0, -26.0, -35.0]);
        }
        drop(runtime);
        drop(pool);
        assert!(!repository.exists());
        assert_eq!(
            tensor(&outputs, &pipeline.ensemble.output[0].name)
                .to_cpu_vec::<f32>()
                .unwrap(),
            [14.0, 29.0, 46.0, 65.0]
        );
        assert!(tensor(&outputs, &pipeline.ensemble.output[0].name)
            .to_cpu_vec::<i32>()
            .is_err());
    }
}

#[test]
fn concurrent_requests_have_independent_inputs_and_results() {
    for device in devices() {
        let (runtime, pipeline) = open("dynamic", device);
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        thread::scope(|scope| {
            for worker in 0..4 {
                let (runtime, pipeline, pool) = (&runtime, &pipeline, &pool);
                scope.spawn(move || {
                    for call in 0..8 {
                        let mut data = vec![(worker * 8 + call) as f32; 7];
                        let outputs = runtime
                            .native
                            .infer(
                                &pipeline.ensemble.name,
                                &[(
                                    pipeline.inputs[0].tensor_name.as_str(),
                                    BorrowedTensor::cpu(&data, &[7]).unwrap(),
                                )],
                                &OutputPlacement::default(),
                                pool,
                            )
                            .unwrap();
                        let expected = data.clone();
                        data.fill(-1.0);
                        assert_eq!(
                            values(&outputs, &pipeline.ensemble.output[0].name),
                            expected
                        );
                    }
                });
            }
        });
    }
}

#[test]
fn invalid_requests_and_backend_errors_allow_subsequent_inference() {
    for device in devices() {
        let (runtime, pipeline) = open("dynamic", device);
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        let model = &pipeline.ensemble.name;
        let name = pipeline.inputs[0].tensor_name.as_str();
        let good = BorrowedTensor::cpu(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let wrong_type = BorrowedTensor::cpu(&[1i32, 2, 3, 4], &[4]).unwrap();
        let placement = OutputPlacement::default();
        assert!(runtime.native.infer(model, &[], &placement, &pool).is_err());
        assert_eq!(
            runtime
                .native
                .infer("missing", &[], &placement, &pool)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::NotFound
        );
        for inputs in [
            vec![(name, wrong_type)],
            vec![("missing", good)],
            vec![("bad\0name", good)],
            vec![(name, good), (name, good)],
        ] {
            assert!(runtime
                .native
                .infer(model, &inputs, &placement, &pool)
                .is_err());
        }
        let unknown_output = OutputPlacement {
            overrides: [("missing".into(), MemoryLocation::Cpu)].into(),
            ..Default::default()
        };
        assert_eq!(
            runtime
                .native
                .infer(model, &[(name, good)], &unknown_output, &pool)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidArgument
        );
        let outputs = runtime
            .native
            .infer(model, &[(name, good)], &placement, &pool)
            .unwrap();
        assert_eq!(
            values(&outputs, &pipeline.ensemble.output[0].name),
            [1.0, 2.0, 3.0, 4.0]
        );
        drop(runtime);

        let (runtime, pipeline) = open("reshape", device);
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        let name = pipeline.inputs[0].tensor_name.as_str();
        let bad = BorrowedTensor::cpu(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let error = runtime
            .native
            .infer(&pipeline.ensemble.name, &[(name, bad)], &placement, &pool)
            .err()
            .unwrap();
        assert_eq!(error.phase(), "execute");
        assert!(error.to_string().contains("Reshape"), "{error}");
        let outputs = runtime
            .native
            .infer(&pipeline.ensemble.name, &[(name, good)], &placement, &pool)
            .unwrap();
        assert_eq!(
            values(&outputs, &pipeline.ensemble.output[0].name),
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
    let error = TritonEngine::load(&invalid, repository, &Options::default())
        .err()
        .unwrap();
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
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        for input_location in locations {
            for output_location in locations {
                for size in [0, 4] {
                    let data = vec![3.0f32; size];
                    let input = placed(&data, &[size as i64], input_location).unwrap();
                    let placement = OutputPlacement {
                        default_location: output_location,
                        ..Default::default()
                    };
                    let outputs = runtime
                        .native
                        .infer(
                            &pipeline.ensemble.name,
                            &[(pipeline.inputs[0].tensor_name.as_str(), input.view())],
                            &placement,
                            &pool,
                        )
                        .unwrap();
                    let output = tensor(&outputs, &pipeline.ensemble.output[0].name);
                    assert_eq!(output.to_cpu_vec::<f32>().unwrap(), data);
                    assert_eq!(output.location(), output_location);
                    assert_eq!(output.shape(), [size as i64]);
                }
            }
        }
    }
}

#[test]
fn outputs_can_use_distinct_memory_locations() {
    for device in devices() {
        let (runtime, pipeline) = open("policy", device);
        let pool = Arc::new(OutputPool::new(OutputPoolOptions::default(), 0));
        let bias = placed(
            &[10.0f32, 20.0, 30.0, 40.0],
            &[4],
            MemoryLocation::Cuda { device_id: 0 },
        )
        .unwrap();
        let obs = BorrowedTensor::cpu(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let inputs = [
            (pipeline.inputs[0].tensor_name.as_str(), obs),
            (pipeline.inputs[1].tensor_name.as_str(), bias.view()),
        ];
        let placement = OutputPlacement {
            default_location: MemoryLocation::Cuda { device_id: 0 },
            overrides: [("policy/echo".into(), MemoryLocation::PinnedCpu)].into(),
        };
        let outputs = runtime
            .native
            .infer(&pipeline.ensemble.name, &inputs, &placement, &pool)
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
            .native
            .infer(&pipeline.ensemble.name, &inputs, &invalid, &pool)
            .err()
            .unwrap();
        assert_eq!(error.kind(), ErrorKind::InvalidArgument);
        assert_eq!(values(&outputs, "policy/echo"), [-8.0, -17.0, -26.0, -35.0]);
    }
}
