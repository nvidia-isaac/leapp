// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::engines::tests::support::{placed, TestTensor};
use std::{env, path::PathBuf};

#[test]
fn native_outputs_allocate_fresh_storage_or_use_caller_buffers() {
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
            let location = match device {
                ExecutionDevice::Cpu => MemoryLocation::Cpu,
                ExecutionDevice::Cuda { device_id } => MemoryLocation::Cuda { device_id },
            };
            let mut session = Session::open(
                &bundle,
                SessionOptions {
                    engine: crate::EngineKind::Native,
                    device,
                    output_placement: OutputPlacement {
                        default_location: location,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            let input = placed(&values, &shape, location).unwrap();
            let inputs = HashMap::from([(input_name.as_str(), input.view())]);
            let first = session.run(&inputs).unwrap();
            let second = session.run(&inputs).unwrap();
            assert_ne!(
                first.get(&output_name).unwrap().view().descriptor().data,
                second.get(&output_name).unwrap().view().descriptor().data
            );
            let output = placed(&vec![0_f32; values.len()], &shape, location).unwrap();
            let descriptor = *output.descriptor();
            let mut buffers = HashMap::from([(
                output_name.as_str(),
                unsafe { OutputBuffer::new(&descriptor) }.unwrap(),
            )]);
            let supplied = session.run_into(&inputs, &mut buffers).unwrap();
            assert!(supplied.get(&output_name).is_none());
            drop(buffers);
            let copied =
                crate::engines::memory::dlpack::copy(output.view(), Some(MemoryLocation::Cpu))
                    .unwrap();
            assert_eq!(copied.to_cpu_vec::<f32>().unwrap(), values);
            session.close();
            assert_eq!(
                first
                    .get(&output_name)
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap(),
                values
            );
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
    let engine = crate::engines::tests::support::engine();
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
            if engine == crate::EngineKind::Native && input_location != state_location {
                continue;
            }
            for output_location in locations {
                let options = SessionOptions {
                    device,
                    engine,
                    cuda_streams: Default::default(),
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
