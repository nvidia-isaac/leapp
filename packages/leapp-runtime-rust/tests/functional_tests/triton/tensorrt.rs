// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::support;
use lrt::dlpack::BorrowedTensor;
use lrt::{Bundle, ExecutionDevice, MemoryLocation, Session, SessionOptions};
use std::{env, path::PathBuf};
use support::{placed, TestTensor};

fn bundle(name: &str) -> Bundle {
    let root = env::var_os("LEAPP_TENSORRT_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tensorrt-fixtures")
        });
    Bundle::open(root.join(format!("{name}.yaml"))).expect(
        "generate TensorRT fixtures with tests/generators/generate_tensorrt_models.py first",
    )
}

fn options() -> SessionOptions {
    SessionOptions {
        device: ExecutionDevice::Cuda { device_id: 0 },
        ..Default::default()
    }
}

fn locations() -> [MemoryLocation; 2] {
    [MemoryLocation::Cpu, MemoryLocation::Cuda { device_id: 0 }]
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn ensembles_support_host_and_device_buffers_and_retained_results() {
    for name in ["policy", "mixed"] {
        let runtime = Session::open(&bundle(name), options()).unwrap();

        let mut retained = None;
        for location in locations() {
            let observation = placed(&[1f32, 2., 3., 4.], &[4], location).unwrap();
            let bias = placed(&[10f32, 20., 30., 40.], &[4], location).unwrap();
            let mut options = options();
            options.output_placement.default_location = location;
            let mut session = Session::open(runtime.info(), options).unwrap();
            let result = session
                .run(&std::collections::HashMap::from([
                    ("preprocess/observation", observation.view()),
                    ("policy/bias", bias.view()),
                ]))
                .unwrap();
            assert_eq!(result.iter().count(), 1);
            let action = result.get("policy/action").unwrap();
            assert_eq!(action.location(), location);
            assert_eq!(action.to_cpu_vec::<f32>().unwrap(), [14., 29., 46., 65.]);
            retained = Some(action.clone());
        }
        drop(runtime);
        assert_eq!(
            retained.unwrap().to_cpu_vec::<f32>().unwrap(),
            [14., 29., 46., 65.]
        );
    }
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn dynamic_profile_bounds_report_errors_without_poisoning_session() {
    let runtime = Session::open(&bundle("dynamic"), options()).unwrap();
    let mut session = runtime.new_session().unwrap();
    for size in [1, 4, 8, 9, 4] {
        let data: Vec<f32> = (0..size).map(|v| v as f32).collect();
        let result = session.run(&std::collections::HashMap::from([(
            "dynamic/input",
            BorrowedTensor::cpu(&data, &[size]).unwrap(),
        )]));
        if size == 9 {
            assert_eq!(result.err().unwrap().phase(), "execute");
        } else {
            let result = result.unwrap();
            let output = result.get("dynamic/output").unwrap();
            assert_eq!(output.shape(), [size]);
            assert_eq!(output.to_cpu_vec::<f32>().unwrap(), data);
        }
    }
}
