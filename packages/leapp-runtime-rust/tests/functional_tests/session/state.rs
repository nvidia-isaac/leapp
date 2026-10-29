// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::engines::tests::support::{placed, TestTensor};
use std::{env, path::PathBuf};

#[test]
fn native_output_snapshots_use_the_session_pool() {
    let fixtures = env::var_os("LEAPP_TRITON_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/triton-fixtures"));
    for (fixture, shape, values) in [
        ("scalar", vec![], vec![1_f32]),
        ("dynamic", vec![4], vec![1_f32, 2., 3., 4.]),
    ] {
        let bundle = Bundle::open(fixtures.join(format!("{fixture}.yaml"))).unwrap();
        let input_name = format!("{fixture}/input");
        let output_name = format!("{fixture}/output");
        for device in [ExecutionDevice::Cpu, ExecutionDevice::Cuda { device_id: 0 }] {
            for location in [
                MemoryLocation::Cpu,
                MemoryLocation::PinnedCpu,
                MemoryLocation::Cuda { device_id: 0 },
            ] {
                for max_entries in [0, 1, 64] {
                    let mut session = Session::open(
                        &bundle,
                        SessionOptions {
                            engine: crate::EngineKind::Native,
                            cuda_streams: Default::default(),
                            device,
                            output_pool: OutputPoolOptions { max_entries },
                            output_placement: OutputPlacement {
                                default_location: location,
                                ..Default::default()
                            },
                        },
                    )
                    .unwrap();
                    let seed = |session: &Session| {
                        let buffer = session
                            .output_pool
                            .allocate(values.len() * 4, location)
                            .unwrap();
                        assert_eq!(
                            Arc::strong_count(&buffer),
                            if max_entries == 0 { 1 } else { 2 }
                        );
                        buffer.ptr.as_ptr().cast::<std::ffi::c_void>()
                    };
                    // The pool keeps this allocation alive. A new, unpooled allocation
                    // cannot reuse its address, so this checks actual pool integration.
                    let address = seed(&session);
                    let run = |session: &mut Session, values: &[f32]| {
                        let input = placed(values, &shape, location).unwrap();
                        session
                            .run(&HashMap::from([(input_name.as_str(), input.view())]))
                            .unwrap()
                            .get(&output_name)
                            .unwrap()
                            .clone()
                    };
                    let first = run(&mut session, &values);
                    if max_entries != 0 {
                        assert_eq!(first.view().descriptor().data, address);
                    }
                    let changed: Vec<_> = values.iter().map(|value| value + 10.).collect();
                    let second = run(&mut session, &changed);
                    assert_ne!(
                        first.view().descriptor().data,
                        second.view().descriptor().data
                    );
                    assert_eq!(first.to_cpu_vec::<f32>().unwrap(), values);
                    assert_eq!(second.to_cpu_vec::<f32>().unwrap(), changed);
                    drop((first, second));
                    let reused = run(&mut session, &values);
                    if max_entries != 0 {
                        assert_eq!(reused.view().descriptor().data, address);
                    }

                    let mut peer = session.new_session().unwrap();
                    let peer_address = seed(&peer);
                    let other = run(&mut peer, &changed);
                    if max_entries != 0 {
                        assert_eq!(other.view().descriptor().data, peer_address);
                    }
                    assert_ne!(
                        reused.view().descriptor().data,
                        other.view().descriptor().data
                    );
                    // Exporting the result must keep it out of subsequent reuse.
                    let lease = reused.export(true);
                    let next = run(&mut session, &changed);
                    assert_ne!(
                        reused.view().descriptor().data,
                        next.view().descriptor().data
                    );
                    session.close();
                    peer.close();
                    assert_eq!(next.to_cpu_vec::<f32>().unwrap(), changed);
                    assert_eq!(other.to_cpu_vec::<f32>().unwrap(), changed);
                    // SAFETY: this test never gives the exported lease to a writer.
                    let copied = engines::memory::dlpack::copy(
                        unsafe { reused.borrow() },
                        Some(MemoryLocation::Cpu),
                    )
                    .unwrap();
                    assert_eq!(copied.to_cpu_vec::<f32>().unwrap(), values);
                    drop(lease);
                }
            }
        }
    }
}

#[test]
fn feedback_with_host_and_device_memory_placement() {
    let fixtures = env::var_os("LEAPP_TRITON_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/triton-fixtures")
        });
    let bundle = Bundle::open(fixtures.join("feedback.yaml")).unwrap();
    let locations = [
        MemoryLocation::Cpu,
        MemoryLocation::PinnedCpu,
        MemoryLocation::Cuda { device_id: 0 },
    ];
    for (device, state_location) in [
        (ExecutionDevice::Cpu, MemoryLocation::Cpu),
        (
            ExecutionDevice::Cuda { device_id: 0 },
            MemoryLocation::Cuda { device_id: 0 },
        ),
    ] {
        let check_state = |session: &Session, expected| {
            for tensor in session.inputs.values() {
                assert_eq!(tensor.location(), state_location);
                assert_eq!(tensor.to_cpu_vec::<f32>().unwrap(), [expected; 4]);
            }
        };
        for input_location in locations {
            for output_location in locations {
                let options = SessionOptions {
                    device,
                    engine: crate::engines::tests::support::engine(),
                    cuda_streams: Default::default(),
                    output_pool: Default::default(),
                    output_placement: OutputPlacement {
                        overrides: [("accumulator/next".into(), output_location)].into(),
                        ..Default::default()
                    },
                };
                let mut session = Session::open(&bundle, options.clone()).unwrap();
                check_state(&session, 0.);
                let input = placed(&[2f32; 4], &[4], input_location).unwrap();
                let inputs = HashMap::from([("accumulator/observation", input.view())]);
                for expected in [2., 4.] {
                    let result = session.run(&inputs).unwrap();
                    let value = result.get("accumulator/next").unwrap();
                    assert_eq!(value.location(), output_location);
                    assert_eq!(value.to_cpu_vec::<f32>().unwrap(), [expected; 4]);
                    check_state(&session, expected);
                }
                let mut peer = session.new_session().unwrap();
                assert!(!Arc::ptr_eq(&session.output_pool, &peer.output_pool));
                check_state(&peer, 0.);
                assert_eq!(
                    peer.run(&inputs)
                        .unwrap()
                        .get("accumulator/next")
                        .unwrap()
                        .location(),
                    output_location
                );
                let mut invalid = options;
                invalid.output_placement.overrides.insert(
                    "accumulator/next".into(),
                    MemoryLocation::Cuda {
                        device_id: u32::MAX,
                    },
                );
                assert!(Session::open(&bundle, invalid).is_err());
                check_state(&session, 4.);
                session.reset().unwrap();
                check_state(&session, 0.);
                session.close();
                check_state(&peer, 2.);
                peer.run(&inputs).unwrap();
            }
        }
    }
}
