// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "triton")]

use leapp_runtime::{
    Bundle, ExecutionDevice, MemoryLocation, RunOptions, Runtime, RuntimeOptions, Tensor,
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
