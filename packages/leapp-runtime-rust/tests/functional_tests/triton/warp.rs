// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::support::{placed, TestTensor};
use crate::{
    Bundle, ExecutionDevice, MemoryLocation, OutputPlacement, OutputPoolOptions, Session,
    SessionOptions,
};
use std::{collections::HashMap, path::PathBuf};

#[test]
#[ignore = "requires Warp adapters, CUDA, and generated Warp fixtures"]
fn warp_plugins_execute_through_sessions() {
    let root = std::env::var_os("LEAPP_WARP_FIXTURES")
        .map(PathBuf::from)
        .expect("set LEAPP_WARP_FIXTURES after running generate_warp_models.py");
    let extract_root = std::env::var_os("WARP_APIC_TMPDIR")
        .expect("set WARP_APIC_TMPDIR to an empty directory for lifecycle checks");
    let graph_count = || std::fs::read_dir(&extract_root).unwrap().count();
    for (fixture, device) in [
        ("warp_onnx", ExecutionDevice::Cpu),
        ("warp_onnx", ExecutionDevice::Cuda { device_id: 0 }),
        ("warp_pt2_cpu", ExecutionDevice::Cpu),
        ("warp_pt2_cuda", ExecutionDevice::Cuda { device_id: 0 }),
    ] {
        let bundle = Bundle::open(root.join(format!("{fixture}.yaml"))).unwrap();
        for location in [
            MemoryLocation::Cpu,
            MemoryLocation::PinnedCpu,
            MemoryLocation::Cuda { device_id: 0 },
        ] {
            for max_entries in [0, 1, 64] {
                let graphs_before = graph_count();
                let mut session = Session::open(
                    &bundle,
                    SessionOptions {
                        device,
                        output_placement: OutputPlacement {
                            default_location: location,
                            ..Default::default()
                        },
                        output_pool: OutputPoolOptions { max_entries },
                    },
                )
                .unwrap();
                let gain = placed(&[1_i32, 2, 3, 4], &[4], location).unwrap();
                let run = |session: &mut Session, offset: f32| {
                    let values: Vec<_> = (0..12).map(|i| i as f32 + offset).collect();
                    let x = placed(&values, &[4, 3], location).unwrap();
                    session
                        .run(&HashMap::from([
                            ("model/x", x.view()),
                            ("model/gain", gain.view()),
                        ]))
                        .unwrap()
                };
                let expected = |offset: f32| -> Vec<f32> {
                    (0..12)
                        .map(|i| (i as f32 + offset) * (i / 3 + 1) as f32 + 4.0)
                        .collect()
                };
                let retained = run(&mut session, 0.0);
                assert_eq!(
                    retained
                        .get("model/result")
                        .unwrap()
                        .to_cpu_vec::<f32>()
                        .unwrap(),
                    expected(0.0),
                    "{fixture} {device:?} {location:?} pool={max_entries}"
                );
                for offset in [1.0, 2.0, 3.0] {
                    assert_eq!(
                        run(&mut session, offset)
                            .get("model/result")
                            .unwrap()
                            .to_cpu_vec::<f32>()
                            .unwrap(),
                        expected(offset)
                    );
                }
                assert_eq!(graph_count(), graphs_before + 1);
                let mut peer = session.new_session().unwrap();
                drop(session);
                assert_eq!(graph_count(), graphs_before + 1);
                let result = run(&mut peer, 4.0);
                drop(peer);
                let output = result.get("model/result").unwrap();
                assert_eq!(output.location(), location);
                assert_eq!(output.to_cpu_vec::<f32>().unwrap(), expected(4.0));
                assert_eq!(
                    retained
                        .get("model/result")
                        .unwrap()
                        .to_cpu_vec::<f32>()
                        .unwrap(),
                    expected(0.0)
                );
                drop(result);
                drop(retained);
                assert_eq!(graph_count(), graphs_before);
            }
        }
    }
}

#[test]
#[ignore = "requires Warp adapters, CUDA, and generated Warp fixtures"]
fn warp_plugin_load_errors_are_reported() {
    use crate::engines::triton::{convert, Options, TritonEngine};
    let root = PathBuf::from(std::env::var_os("LEAPP_WARP_FIXTURES").unwrap());
    for (fixture, backend, filename) in [
        ("warp_onnx", "onnxruntime", "libleapp_wrp_onnx_custom_op.so"),
        (
            "warp_pt2_cpu",
            "leapp_pt2",
            "libleapp_wrp_torch_custom_op.so",
        ),
    ] {
        let bundle = Bundle::open(root.join(format!("{fixture}.yaml"))).unwrap();
        let mut pipeline = convert::convert_bundle(&bundle, ExecutionDevice::Cpu).unwrap();
        let installation = tempfile::tempdir().unwrap();
        std::fs::create_dir(installation.path().join(backend)).unwrap();
        std::fs::write(
            installation.path().join(backend).join(filename),
            b"invalid library",
        )
        .unwrap();
        convert::configure_plugins(&mut pipeline, installation.path()).unwrap();
        let repository = convert::repository::ModelRepository::create(&pipeline).unwrap();
        let error = TritonEngine::load(&pipeline, repository, &Options::default())
            .err()
            .unwrap();
        assert_eq!(error.phase(), "load");
        assert!(error.to_string().contains("failed to load"), "{error}");
    }
}
