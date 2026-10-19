// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "triton")]

use leapp_runtime::{
    Bundle, ExecutionDevice, MemoryLocation, RunOptions, Runtime, RuntimeOptions, Tensor,
    TensorView,
};
use std::{env, fs, path::PathBuf};

fn fixtures() -> PathBuf {
    env::var_os("LEAPP_TORCHSCRIPT_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/torchscript-fixtures")
        })
}

fn open(name: &str, device: ExecutionDevice) -> Runtime {
    let bundle = Bundle::open(fixtures().join(format!("{name}.yaml"))).unwrap();
    Runtime::open(
        &bundle,
        RuntimeOptions {
            device,
            backend_directory: env::var_os("LEAPP_TRITON_BACKENDS")
                .map(PathBuf::from)
                .unwrap_or_else(|| RuntimeOptions::default().backend_directory),
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
            let mut session = runtime.create_session().unwrap();
            drop(runtime);
            let mut retained = None;
            for location in locations() {
                let values = Tensor::copy_from_cpu(&[1f32, 2., 3., 4.], &[4], location).unwrap();
                let bias = Tensor::copy_from_cpu(&[10f32, 20., 30., 40.], &[4], location).unwrap();
                let mut options = RunOptions::default();
                options.output_placement.default_location = location;
                let input = if name == "mixed" {
                    "preprocess/values"
                } else {
                    "model/z_values"
                };
                // Request order differs from the exported argument order.
                let result = session
                    .run_with_options(
                        &[("model/INPUT__9", bias.view()), (input, values.view())],
                        &options,
                    )
                    .unwrap();
                assert_eq!(
                    result.iter().map(|(name, _)| name).collect::<Vec<_>>(),
                    ["model/z_result", "model/OUTPUT__7"]
                );
                for (index, name) in ["model/z_result", "model/OUTPUT__7"].iter().enumerate() {
                    let tensor = result.get(name).unwrap();
                    assert_eq!(tensor.location(), location);
                    assert_eq!(tensor.copy_to_cpu::<f32>().unwrap(), expected[index]);
                }
                assert_eq!(result.get("model/OUTPUT__7").unwrap().shape(), [2, 2]);
                retained = Some(result);
            }
            drop(session);
            assert_eq!(
                retained
                    .unwrap()
                    .get("model/z_result")
                    .unwrap()
                    .copy_to_cpu::<f32>()
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
        let mut session = runtime.create_session().unwrap();
        for location in locations() {
            let values = Tensor::copy_from_cpu(&[1f32, 2., 3., 4.], &[4], location).unwrap();
            for _ in 0..2 {
                let result = session.run(&[("model/values", values.view())]).unwrap();
                assert_eq!(
                    result
                        .get("model/result")
                        .unwrap()
                        .copy_to_cpu::<f32>()
                        .unwrap(),
                    [9., 12., 15., 18.]
                );
                assert_eq!(values.copy_to_cpu::<f32>().unwrap(), [1., 2., 3., 4.]);
            }
        }
    }
}

#[test]
#[ignore = "requires LEAPP TorchScript backend and TorchScript fixtures"]
fn dynamic_shapes_integer_tensors_and_recoverable_execution_errors() {
    for device in devices() {
        let runtime = open("dynamic", device);
        let mut session = runtime.create_session().unwrap();
        for size in [4, 6, 3, 4] {
            let values: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let result =
                session.run(&[("model/values", TensorView::cpu(&values, &[size]).unwrap())]);
            if size == 3 {
                assert_eq!(result.err().unwrap().phase(), "execute");
            } else {
                let result = result.unwrap();
                let output = result.get("model/result").unwrap();
                assert_eq!(output.shape(), [size / 2, 2]);
                assert_eq!(output.copy_to_cpu::<f32>().unwrap(), values);
            }
        }
        let runtime = open("integer", device);
        let mut session = runtime.create_session().unwrap();
        let result = session
            .run(&[(
                "model/values",
                TensorView::cpu(&[1i64, 2, 3, 4], &[4]).unwrap(),
            )])
            .unwrap();
        assert_eq!(
            result
                .get("model/result")
                .unwrap()
                .copy_to_cpu::<i64>()
                .unwrap(),
            [2, 3, 4, 5]
        );
    }
}

#[test]
#[ignore = "requires LEAPP TorchScript and NativeRT backends and TorchScript fixtures"]
fn torchscript_and_nativert_coexist() {
    let runtime = open("mixed_pt2", ExecutionDevice::Cpu);
    let mut session = runtime.create_session().unwrap();
    let result = session
        .run(&[
            (
                "preprocess/values",
                TensorView::cpu(&[1f32, 2., 3., 4.], &[4]).unwrap(),
            ),
            (
                "model/INPUT__9",
                TensorView::cpu(&[10f32, 20., 30., 40.], &[4]).unwrap(),
            ),
        ])
        .unwrap();
    assert_eq!(
        result
            .get("model/z_result")
            .unwrap()
            .copy_to_cpu::<f32>()
            .unwrap(),
        [14., 29., 46., 65.]
    );
}

#[test]
#[ignore = "requires LEAPP TorchScript backend and fixtures"]
fn mutation_does_not_cross_ensemble_consumers() {
    for device in devices() {
        let runtime = open("fanout", device);
        let mut session = runtime.create_session().unwrap();
        for _ in 0..3 {
            let values = [1f32, 2., 3., 4.];
            let result = session
                .run(&[("producer/values", TensorView::cpu(&values, &[4]).unwrap())])
                .unwrap();
            assert_eq!(
                result
                    .get("observer/result")
                    .unwrap()
                    .copy_to_cpu::<f32>()
                    .unwrap(),
                values
            );
            assert_eq!(
                result
                    .get("consumer/result")
                    .unwrap()
                    .copy_to_cpu::<f32>()
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
        let runtime = open("mutating_feedback", device);
        let mut first = runtime.create_session().unwrap();
        let mut second = runtime.create_session().unwrap();
        let input = [(
            "accumulator/observation",
            TensorView::cpu(&[1f32; 4], &[4]).unwrap(),
        )];
        let read = |r: &leapp_runtime::RunResult| {
            r.get("accumulator/next")
                .unwrap()
                .copy_to_cpu::<f32>()
                .unwrap()
        };
        let retained = first.run(&input).unwrap();
        assert_eq!(read(&first.run(&input).unwrap()), [2.; 4]);
        assert!(first
            .run(&[(
                "accumulator/observation",
                TensorView::cpu(&[-10f32; 4], &[4]).unwrap()
            )])
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
        let mut session = runtime.create_session().unwrap();
        let result = session
            .run(&[(
                "model/values",
                TensorView::cpu(&[1f32, 2., 3., 4.], &[4]).unwrap(),
            )])
            .unwrap();
        assert_eq!(
            result
                .get("model/result")
                .unwrap()
                .copy_to_cpu::<f32>()
                .unwrap(),
            [2., 3., 4., 5.]
        );
    }
}
