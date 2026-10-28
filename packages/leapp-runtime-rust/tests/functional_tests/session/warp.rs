// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::support::{engine, placed, TestTensor};
use crate::dlpack::SharedTensor;
use crate::{
    Bundle, ExecutionDevice, MemoryLocation, OutputPlacement, OutputPoolOptions, Session,
    SessionOptions,
};
use std::path::PathBuf;

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
                let open = || {
                    Session::open(
                        &bundle,
                        SessionOptions {
                            engine: engine(),
                            output_pool: OutputPoolOptions { max_entries },
                            device,
                            output_placement: OutputPlacement {
                                default_location: location,
                                ..Default::default()
                            },
                        },
                    )
                    .unwrap()
                };
                let mut session = open();
                let x = placed(
                    &(0..12).map(|i| i as f32).collect::<Vec<_>>(),
                    &[4, 3],
                    location,
                )
                .unwrap();
                let gain = placed(&[1_i32, 2, 3, 4], &[4], location).unwrap();
                let run = |session: &mut Session, x: &SharedTensor, gain: &SharedTensor| {
                    session
                        .run(&std::collections::HashMap::from([
                            ("model/x", x.view()),
                            ("model/gain", gain.view()),
                        ]))
                        .unwrap()
                };
                let expected: Vec<_> = (0..12)
                    .map(|i| i as f32 * (i / 3 + 1) as f32 + 4.0)
                    .collect();
                let retained = run(&mut session, &x, &gain);
                let changed_x = placed(&[2_f32; 12], &[4, 3], location).unwrap();
                let changed_gain = placed(&[4_i32, 3, 2, 1], &[4], location).unwrap();
                for _ in 0..3 {
                    assert_eq!(
                        run(&mut session, &changed_x, &changed_gain)
                            .get("model/result")
                            .unwrap()
                            .to_cpu_vec::<f32>()
                            .unwrap(),
                        [12., 12., 12., 10., 10., 10., 8., 8., 8., 6., 6., 6.],
                        "{fixture} {device:?} {location:?} pool={max_entries}"
                    );
                }
                assert_eq!(
                    retained
                        .get("model/result")
                        .unwrap()
                        .to_cpu_vec::<f32>()
                        .unwrap(),
                    expected,
                    "later execution must not overwrite retained output"
                );
                drop(retained);
                assert_eq!(graph_count(), graphs_before + 1);
                let mut peer = session.new_session().unwrap();
                drop(session);
                assert_eq!(graph_count(), graphs_before + 1);
                // Native models share the adapter cache. Unloading one must not
                // evict another's runner. Triton uses process-wide server resources
                // that cannot safely overlap across independent Session::open calls.
                let independent = if engine() == crate::EngineKind::Native {
                    let mut other = open();
                    drop(run(&mut other, &x, &gain));
                    assert_eq!(graph_count(), graphs_before + 2);
                    Some(other)
                } else {
                    None
                };
                let result = run(&mut peer, &x, &gain);
                drop(peer);
                let output = result.get("model/result").unwrap();
                assert_eq!(output.location(), location);
                assert_eq!(output.to_cpu_vec::<f32>().unwrap(), expected);
                drop(result);
                if let Some(mut independent) = independent {
                    assert_eq!(graph_count(), graphs_before + 1);
                    assert_eq!(
                        run(&mut independent, &x, &gain)
                            .get("model/result")
                            .unwrap()
                            .to_cpu_vec::<f32>()
                            .unwrap(),
                        expected
                    );
                    drop(independent);
                }
                assert_eq!(graph_count(), graphs_before);
            }
        }
    }
}

#[test]
#[ignore = "requires native backend libraries and generated Warp fixtures"]
fn native_warp_plugin_load_errors_are_reported() {
    const CHILD: &str = "LEAPP_TEST_BAD_WARP_PLUGIN";
    let fixtures = PathBuf::from(std::env::var_os("LEAPP_WARP_FIXTURES").unwrap());
    if let Ok(fixture) = std::env::var(CHILD) {
        let bundle = Bundle::open(fixtures.join(format!("{fixture}.yaml"))).unwrap();
        let error = Session::open(
            &bundle,
            SessionOptions {
                engine: crate::EngineKind::Native,
                ..Default::default()
            },
        )
        .err()
        .expect("an invalid installed adapter must fail model loading");
        assert_eq!(error.phase(), "load");
        assert!(error.to_string().contains("libleapp_wrp_"), "{error}");
        return;
    }

    let backends = PathBuf::from(std::env::var_os("LEAPP_NATIVE_BACKENDS").unwrap());
    for (fixture, backend, filename) in [
        ("warp_onnx", "onnxruntime", "libleapp_wrp_onnx_custom_op.so"),
        (
            "warp_pt2_cpu",
            "nativert",
            "libleapp_wrp_torch_custom_op.so",
        ),
    ] {
        for dangling in [false, true] {
            let installation = tempfile::tempdir().unwrap();
            let directory = installation.path().join(backend);
            std::fs::create_dir(&directory).unwrap();
            for entry in std::fs::read_dir(backends.join(backend)).unwrap() {
                let entry = entry.unwrap();
                if entry.file_name() != filename {
                    std::os::unix::fs::symlink(entry.path(), directory.join(entry.file_name()))
                        .unwrap();
                }
            }
            let library = directory.join(filename);
            if dangling {
                std::os::unix::fs::symlink(directory.join("missing.so"), library).unwrap();
            } else {
                std::fs::write(library, b"invalid library").unwrap();
            }
            // Library handles are cached process-wide. Isolate both that cache
            // and environment overrides from concurrent tests.
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "engines::tests::warp::native_warp_plugin_load_errors_are_reported",
                    "--ignored",
                    "--test-threads=1",
                ])
                .env(CHILD, fixture)
                .env("LEAPP_NATIVE_BACKENDS", installation.path())
                .status()
                .unwrap();
            assert!(status.success(), "{fixture}, dangling link: {dangling}");
        }
    }
}
