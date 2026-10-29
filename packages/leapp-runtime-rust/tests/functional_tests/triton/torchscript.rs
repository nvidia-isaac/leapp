// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::support;
use lrt::{Bundle, ExecutionDevice, MemoryLocation, Session, SessionOptions};
use std::{env, fs, path::PathBuf};
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
                    .run(&std::collections::HashMap::from([
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
