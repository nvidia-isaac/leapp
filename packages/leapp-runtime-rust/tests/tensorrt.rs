// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "triton")]

use leapp_runtime::{
    Bundle, ExecutionDevice, MemoryLocation, RunOptions, Runtime, RuntimeOptions, Tensor,
    TensorView,
};
use std::{env, path::PathBuf};

fn bundle(name: &str) -> Bundle {
    let root = env::var_os("LEAPP_TENSORRT_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tensorrt-fixtures")
        });
    Bundle::open(root.join(format!("{name}.yaml")))
        .expect("generate TensorRT fixtures with tests/generate_tensorrt_models.py first")
}

fn options() -> RuntimeOptions {
    RuntimeOptions {
        device: ExecutionDevice::Cuda { device_id: 0 },
        backend_directory: env::var_os("LEAPP_TRITON_BACKENDS")
            .map(PathBuf::from)
            .unwrap_or_else(|| RuntimeOptions::default().backend_directory),
    }
}

fn locations() -> [MemoryLocation; 2] {
    [MemoryLocation::Cpu, MemoryLocation::Cuda { device_id: 0 }]
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn ensembles_support_host_and_device_buffers_and_retained_results() {
    for name in ["policy", "mixed"] {
        let runtime = Runtime::open(&bundle(name), options()).unwrap();
        let mut session = runtime.create_session().unwrap();
        drop(runtime);
        let mut retained = None;
        for location in locations() {
            let observation = Tensor::copy_from_cpu(&[1f32, 2., 3., 4.], &[4], location).unwrap();
            let bias = Tensor::copy_from_cpu(&[10f32, 20., 30., 40.], &[4], location).unwrap();
            let mut options = RunOptions::default();
            options.output_placement.default_location = location;
            let result = session
                .run_with_options(
                    &[
                        ("preprocess/observation", observation.view()),
                        ("policy/bias", bias.view()),
                    ],
                    &options,
                )
                .unwrap();
            assert_eq!(result.iter().count(), 1);
            let action = result.get("policy/action").unwrap();
            assert_eq!(action.location(), location);
            assert_eq!(action.copy_to_cpu::<f32>().unwrap(), [14., 29., 46., 65.]);
            retained = Some(action.clone());
        }
        drop(session);
        assert_eq!(
            retained.unwrap().copy_to_cpu::<f32>().unwrap(),
            [14., 29., 46., 65.]
        );
    }
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn dynamic_profile_bounds_report_errors_without_poisoning_session() {
    let runtime = Runtime::open(&bundle("dynamic"), options()).unwrap();
    let mut session = runtime.create_session().unwrap();
    for size in [1, 4, 8, 9, 4] {
        let data: Vec<f32> = (0..size).map(|v| v as f32).collect();
        let result = session.run(&[("dynamic/input", TensorView::cpu(&data, &[size]).unwrap())]);
        if size == 9 {
            assert_eq!(result.err().unwrap().phase(), "execute");
        } else {
            let result = result.unwrap();
            let output = result.get("dynamic/output").unwrap();
            assert_eq!(output.shape(), [size]);
            assert_eq!(output.as_cpu_slice::<f32>().unwrap(), data);
        }
    }
}
