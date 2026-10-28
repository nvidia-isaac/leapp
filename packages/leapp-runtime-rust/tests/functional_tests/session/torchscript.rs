// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::support;
use lrt::dlpack::BorrowedTensor;
use lrt::{Bundle, ExecutionDevice, MemoryLocation, OutputPoolOptions, Session, SessionOptions};
use std::{collections::HashMap, env, fs, path::PathBuf};
use support::{placed, TestTensor};

fn fixtures() -> PathBuf {
    env::var_os("LEAPP_TORCHSCRIPT_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/torchscript-fixtures")
        })
}

fn open(name: &str, device: ExecutionDevice) -> Session {
    let bundle = Bundle::open(fixtures().join(format!("{name}.yaml"))).unwrap();
    Session::open(
        &bundle,
        SessionOptions {
            engine: support::engine(),
            device,
            ..Default::default()
        },
    )
    .unwrap()
}

fn devices() -> Vec<ExecutionDevice> {
    let mut devices = vec![ExecutionDevice::Cpu];
    devices.push(ExecutionDevice::Cuda { device_id: 0 });
    devices
}

fn locations() -> [MemoryLocation; 2] {
    [MemoryLocation::Cpu, MemoryLocation::Cuda { device_id: 0 }]
}

#[test]
#[ignore = "requires LEAPP TorchScript/ONNX backends and TorchScript fixtures"]
fn scripted_traced_and_mixed_match_pytorch_with_positional_ports() {
    let expected: Vec<Vec<f32>> =
        serde_json::from_slice(&fs::read(fixtures().join("expected.json")).unwrap()).unwrap();
    for device in devices() {
        for name in ["scripted", "traced", "mixed"] {
            let runtime = open(name, device);

            let mut retained = None;
            for location in locations() {
                let values = placed(&[1f32, 2., 3., 4.], &[4], location).unwrap();
                let bias = placed(&[10f32, 20., 30., 40.], &[4], location).unwrap();
                let mut options = SessionOptions {
                    engine: support::engine(),
                    device,
                    ..Default::default()
                };
                options.output_placement.default_location = location;
                let input = if name == "mixed" {
                    "preprocess/values"
                } else {
                    "model/z_values"
                };
                let mut session = Session::open(runtime.info(), options).unwrap();
                // Request order differs from the exported argument order.
                let result = session
                    .run(&HashMap::from([
                        ("model/INPUT__9", bias.view()),
                        (input, values.view()),
                    ]))
                    .unwrap();
                assert_eq!(
                    result.iter().map(|(name, _)| name).collect::<Vec<_>>(),
                    ["model/z_result", "model/OUTPUT__7"]
                );
                for (index, name) in ["model/z_result", "model/OUTPUT__7"].iter().enumerate() {
                    let tensor = result.get(name).unwrap();
                    assert_eq!(tensor.location(), location);
                    assert_eq!(tensor.to_cpu_vec::<f32>().unwrap(), expected[index]);
                }
                assert_eq!(result.get("model/OUTPUT__7").unwrap().shape(), [2, 2]);
                retained = Some(result);
            }
            assert_eq!(
                retained
                    .unwrap()
                    .get("model/z_result")
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap(),
                expected[0]
            );
        }
    }
}

#[test]
#[ignore = "requires LEAPP TorchScript backend and TorchScript fixtures"]
fn inplace_models_do_not_modify_borrowed_inputs() {
    for device in devices() {
        let runtime = open("mutating", device);
        let mut session = runtime.new_session().unwrap();
        for location in locations() {
            let values = placed(&[1f32, 2., 3., 4.], &[4], location).unwrap();
            for _ in 0..2 {
                let result = session
                    .run(&HashMap::from([("model/values", values.view())]))
                    .unwrap();
                assert_eq!(
                    result
                        .get("model/result")
                        .unwrap()
                        .to_cpu_vec::<f32>()
                        .unwrap(),
                    [9., 12., 15., 18.]
                );
                assert_eq!(values.to_cpu_vec::<f32>().unwrap(), [1., 2., 3., 4.]);
            }
        }
    }
}

#[test]
#[ignore = "requires LEAPP TorchScript backend and TorchScript fixtures"]
fn dynamic_shapes_integer_tensors_and_recoverable_execution_errors() {
    for device in devices() {
        let runtime = open("dynamic", device);
        let mut session = runtime.new_session().unwrap();
        for size in [4, 6, 3, 4] {
            let values: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let result = session.run(&HashMap::from([(
                "model/values",
                BorrowedTensor::cpu(&values, &[size]).unwrap(),
            )]));
            if size == 3 {
                assert_eq!(result.err().unwrap().phase(), "execute");
            } else {
                let result = result.unwrap();
                let output = result.get("model/result").unwrap();
                assert_eq!(output.shape(), [size / 2, 2]);
                assert_eq!(output.to_cpu_vec::<f32>().unwrap(), values);
            }
        }
        let runtime = open("integer", device);
        let mut session = runtime.new_session().unwrap();
        let result = session
            .run(&HashMap::from([(
                "model/values",
                BorrowedTensor::cpu(&[1i64, 2, 3, 4], &[4]).unwrap(),
            )]))
            .unwrap();
        assert_eq!(
            result
                .get("model/result")
                .unwrap()
                .to_cpu_vec::<i64>()
                .unwrap(),
            [2, 3, 4, 5]
        );
    }
}

#[test]
#[ignore = "requires LEAPP TorchScript and NativeRT backends and TorchScript fixtures"]
fn torchscript_and_nativert_coexist() {
    let runtime = open("mixed_pt2", ExecutionDevice::Cpu);
    let mut session = runtime.new_session().unwrap();
    let result = session
        .run(&HashMap::from([
            (
                "preprocess/values",
                BorrowedTensor::cpu(&[1f32, 2., 3., 4.], &[4]).unwrap(),
            ),
            (
                "model/INPUT__9",
                BorrowedTensor::cpu(&[10f32, 20., 30., 40.], &[4]).unwrap(),
            ),
        ]))
        .unwrap();
    assert_eq!(
        result
            .get("model/z_result")
            .unwrap()
            .to_cpu_vec::<f32>()
            .unwrap(),
        [14., 29., 46., 65.]
    );
}

#[test]
#[ignore = "requires LEAPP TorchScript backend and fixtures"]
fn mutation_does_not_cross_ensemble_consumers() {
    for device in devices() {
        let runtime = open("fanout", device);
        let mut session = runtime.new_session().unwrap();
        for _ in 0..3 {
            let values = [1f32, 2., 3., 4.];
            let result = session
                .run(&HashMap::from([(
                    "producer/values",
                    BorrowedTensor::cpu(&values, &[4]).unwrap(),
                )]))
                .unwrap();
            assert_eq!(
                result
                    .get("observer/result")
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap(),
                values
            );
            assert_eq!(
                result
                    .get("consumer/result")
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap(),
                [9., 12., 15., 18.]
            );
            assert_eq!(values, [1., 2., 3., 4.]);
        }
    }
}

#[test]
#[ignore = "requires LEAPP TorchScript backend and fixtures"]
fn failed_mutation_preserves_feedback_and_retained_outputs() {
    for device in devices() {
        let bundle = Bundle::open(fixtures().join("mutating_feedback.yaml")).unwrap();
        let runtime = Session::open(
            &bundle,
            SessionOptions {
                device,
                output_pool: OutputPoolOptions { max_entries: 1 },
                ..Default::default()
            },
        )
        .unwrap();
        let mut first = runtime.new_session().unwrap();
        let mut second = runtime.new_session().unwrap();
        let observation = BorrowedTensor::cpu(&[1f32; 4], &[4]).unwrap();
        let input = HashMap::from([("accumulator/observation", observation)]);
        let read = |r: &lrt::RunResult| {
            r.get("accumulator/next")
                .unwrap()
                .to_cpu_vec::<f32>()
                .unwrap()
        };
        let retained = first.run(&input).unwrap();
        assert_eq!(read(&first.run(&input).unwrap()), [2.; 4]);
        assert!(first
            .run(&HashMap::from([(
                "accumulator/observation",
                BorrowedTensor::cpu(&[-10f32; 4], &[4]).unwrap()
            )]))
            .is_err());
        assert_eq!(read(&first.run(&input).unwrap()), [3.; 4]);
        assert_eq!(read(&second.run(&input).unwrap()), [1.; 4]);
        first.reset().unwrap();
        assert_eq!(read(&first.run(&input).unwrap()), [1.; 4]);
        drop(first);
        drop(second);
        drop(runtime);
        assert_eq!(read(&retained), [1.; 4]);
    }
}

#[test]
#[ignore = "requires LEAPP TorchScript backend and fixtures"]
fn nested_subgraphs_execute_with_owned_inputs() {
    for device in devices() {
        let runtime = open("forked", device);
        let mut session = runtime.new_session().unwrap();
        let result = session
            .run(&HashMap::from([(
                "model/values",
                BorrowedTensor::cpu(&[1f32, 2., 3., 4.], &[4]).unwrap(),
            )]))
            .unwrap();
        assert_eq!(
            result
                .get("model/result")
                .unwrap()
                .to_cpu_vec::<f32>()
                .unwrap(),
            [2., 3., 4., 5.]
        );
    }
}
