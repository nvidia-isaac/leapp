// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(super) fn bundle(backend: &str, name: &str, device: ExecutionDevice) -> Bundle {
    let (variable, directory, name) = if backend == "jit" {
        (
            "LEAPP_TORCHSCRIPT_FIXTURES",
            "torchscript-fixtures",
            name.to_string(),
        )
    } else {
        let suffix = match device {
            ExecutionDevice::Cpu => "cpu",
            ExecutionDevice::Cuda { .. } => "cuda",
        };
        (
            "LEAPP_PT2_FIXTURES",
            "pt2-fixtures",
            format!("{name}_{suffix}"),
        )
    };
    let root = std::env::var_os(variable)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join(directory)
        });
    Bundle::open(root.join(format!("{name}.yaml"))).unwrap()
}

fn engine(bundle: &Bundle, device: ExecutionDevice) -> NativeEngine {
    NativeEngine::load(
        bundle,
        &Graph::new(bundle).unwrap(),
        &SessionOptions {
            engine: crate::EngineKind::Native,
            device,
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
#[ignore = "requires native TorchScript/NativeRT backends and CPU/CUDA fixtures"]
fn fixed_torch_buffers_and_bindings_survive_changed_inputs_and_placements() {
    for device in devices() {
        for (backend, fixture, input, output) in [
            ("jit", "scripted", "model/z_values", "model/z_result"),
            ("jit", "alias_identity", "model/z_values", "model/z_result"),
            ("pt2", "weighted", "model/values", "model/output_0"),
        ] {
            let bundle = bundle(backend, fixture, device);
            let engine = engine(&bundle, device);
            let _peer = assert_private_peer(&engine);
            let before = identities(&engine.workspace.lock().unwrap());
            assert!(before.1.iter().all(Option::is_some));
            let mut retained = Vec::new();
            let execution = execution_location(&engine);
            // Reject incompatible devices once each, then exercise recovery and placements.
            for input_location in locations()
                .into_iter()
                .filter(|&location| location != execution)
                .chain(std::iter::once(execution))
            {
                let offsets: &[f32] = if input_location == execution {
                    &[0., 7., -2.]
                } else {
                    &[0.]
                };
                for &offset in offsets {
                    let data = [1. + offset, 2. + offset, 3. + offset, 4. + offset];
                    let values = placed(&data, &[4], input_location).unwrap();
                    let bias = placed(&[10f32, 20., 30., 40.], &[4], input_location).unwrap();
                    let mut inputs = vec![(input, values.view())];
                    if backend == "jit" {
                        let other = if fixture == "alias_identity" && offset == 0. {
                            values.view()
                        } else {
                            bias.view()
                        };
                        inputs.push(("model/INPUT__9", other));
                    }
                    if input_location != execution {
                        let error = engine
                            .run(&inputs, &OutputPlacement::default(), &mut HashMap::new())
                            .err()
                            .unwrap();
                        assert_eq!(error.code(), ErrorCode::InvalidArgument);
                        assert!(error.to_string().contains("execution device"));
                        continue;
                    }
                    for output_location in locations() {
                        let result = engine
                            .run(
                                &inputs,
                                &OutputPlacement {
                                    default_location: output_location,
                                    ..Default::default()
                                },
                                &mut HashMap::new(),
                            )
                            .unwrap();
                        let expected = if fixture == "alias_identity" {
                            data.map(|value| value + if offset == 0. { 1. } else { 2. })
                        } else {
                            [
                                data[0] * 2. + 10.,
                                data[1] * 3. + 20.,
                                data[2] * 4. + 30.,
                                data[3] * 5. + 40.,
                            ]
                        };
                        assert_eq!(values.to_cpu_vec::<f32>().unwrap(), data);
                        assert_eq!(result[output].location(), output_location);
                        assert_eq!(identities(&engine.workspace.lock().unwrap()), before);
                        retained.push((expected, result));
                    }
                }
            }
            drop(engine);
            for (expected, result) in retained {
                assert_eq!(result[output].to_cpu_vec::<f32>().unwrap(), expected);
            }
        }
    }
}

#[test]
#[ignore = "requires native TorchScript backend and CPU/CUDA fixtures"]
fn torch_execution_preserves_inputs_for_mutation_retention_and_background_work() {
    for device in devices() {
        for name in ["mutating", "retaining", "forked"] {
            let bundle = bundle("jit", name, device);
            let engine = engine(&bundle, device);
            let input = placed(&[1f32; 4], &[4], execution_location(&engine)).unwrap();
            let name = bundle.pipeline_inputs()[0].name();
            engine
                .run(
                    &[(&name, input.view())],
                    &OutputPlacement::default(),
                    &mut HashMap::new(),
                )
                .unwrap();
            assert_eq!(input.to_cpu_vec::<f32>().unwrap(), [1.; 4]);
        }
    }
}

#[test]
#[ignore = "requires native NativeRT backend and CPU/CUDA fixtures"]
fn nativert_peers_preserve_shared_persistent_state() {
    for device in devices() {
        let location = match device {
            ExecutionDevice::Cpu => MemoryLocation::Cpu,
            ExecutionDevice::Cuda { device_id } => MemoryLocation::Cuda { device_id },
        };
        let bundle = bundle("pt2", "stateful", device);
        let mut parent = Session::open(
            &bundle,
            SessionOptions {
                engine: crate::EngineKind::Native,
                device,
                ..Default::default()
            },
        )
        .unwrap();
        let mut peers = (0..3)
            .map(|_| parent.new_session().unwrap())
            .collect::<Vec<_>>();
        parent.close();
        let start = std::sync::Barrier::new(peers.len());
        let mut values = std::thread::scope(|scope| {
            peers
                .iter_mut()
                .map(|peer| {
                    let start = &start;
                    scope.spawn(move || {
                        let input = placed(&[1f32; 4], &[4], location).unwrap();
                        start.wait();
                        (0..10)
                            .map(|_| {
                                peer.reset().unwrap();
                                let result = peer
                                    .run(&std::collections::HashMap::from([(
                                        "model/values",
                                        input.view(),
                                    )]))
                                    .unwrap();
                                let value = result
                                    .get("model/output_0")
                                    .unwrap()
                                    .to_cpu_vec::<f32>()
                                    .unwrap();
                                assert!(value.iter().all(|&x| x == value[0]));
                                value[0] as usize
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|thread| thread.join().unwrap())
                .collect::<Vec<_>>()
        });
        values.sort_unstable();
        assert_eq!(values, (1..=30).collect::<Vec<_>>());
    }
}

#[test]
#[ignore = "requires native NativeRT backend and CPU/CUDA fixtures"]
fn nativert_dynamic_shapes_recover_after_execution_errors() {
    for device in devices() {
        let bundle = bundle("pt2", "reshape", device);
        let engine = engine(&bundle, device);
        assert!(!engine.workspace.lock().unwrap().models[0].is_prepared());
        let mut retained = Vec::new();
        for size in [4, 3, 0, 8, 2] {
            let values = vec![size as f32; size];
            let shape = [size as i64];
            let result = engine.run(
                &[(
                    "model/values",
                    BorrowedTensor::cpu(&values, &shape).unwrap(),
                )],
                &OutputPlacement::default(),
                &mut HashMap::new(),
            );
            if size == 3 {
                assert_eq!(result.err().unwrap().phase(), "execute");
            } else {
                let result = result.unwrap();
                assert_eq!(result["model/output_0"].shape(), [size as i64 / 2, 2]);
                retained.push((values, result));
            }
        }
        drop(engine);
        for (expected, result) in retained {
            assert_eq!(
                result["model/output_0"].to_cpu_vec::<f32>().unwrap(),
                expected
            );
        }
    }
}

#[test]
#[ignore = "requires native NativeRT backend and CPU/CUDA fixtures"]
fn nativert_input_mutation_is_isolated_across_concurrent_sessions() {
    for device in devices() {
        let bundle = bundle("pt2", "mutating", device);
        let session = Session::open(
            &bundle,
            SessionOptions {
                engine: crate::EngineKind::Native,
                device,
                ..Default::default()
            },
        )
        .unwrap();
        std::thread::scope(|scope| {
            for worker in 0..4 {
                let mut peer = session.new_session().unwrap();
                scope.spawn(move || {
                    for location in locations() {
                        let values = placed(&[worker as f32; 4], &[4], location).unwrap();
                        for _ in 0..3 {
                            let result = peer
                                .run(&std::collections::HashMap::from([(
                                    "model/values",
                                    values.view(),
                                )]))
                                .unwrap();
                            assert_eq!(
                                result
                                    .get("model/output_0")
                                    .unwrap()
                                    .to_cpu_vec::<f32>()
                                    .unwrap(),
                                [(worker as f32 + 2.) * 3.; 4]
                            );
                            assert_eq!(values.to_cpu_vec::<f32>().unwrap(), [worker as f32; 4]);
                        }
                    }
                });
            }
        });
    }
}

#[test]
#[ignore = "requires native TorchScript/NativeRT backends and CPU/CUDA fixtures"]
fn torch_scalars_empty_tensors_and_dynamic_dtype_transport() {
    for device in devices() {
        for backend in ["jit", "pt2"] {
            let output = if backend == "jit" {
                "model/result"
            } else {
                "model/output_0"
            };
            for (name, data, shape) in [
                ("scalar", &[3f32][..], &[][..]),
                ("empty", &[][..], &[0][..]),
            ] {
                let bundle = bundle(backend, name, device);
                let engine = engine(&bundle, device);
                assert!(engine.workspace.lock().unwrap().models[0].is_prepared());
                let input = placed(data, shape, execution_location(&engine)).unwrap();
                let result = engine
                    .run(
                        &[("model/values", input.view())],
                        &OutputPlacement::default(),
                        &mut HashMap::new(),
                    )
                    .unwrap();
                assert_eq!(result[output].shape(), shape);
                assert_eq!(result[output].to_cpu_vec::<f32>().unwrap(), data);
            }
            // Force dynamic dispatch so the C++ output dtype travels back to Rust.
            for name in ["double", "short"] {
                let mut bundle = bundle(backend, name, device);
                let description = &mut std::sync::Arc::get_mut(&mut bundle.0).unwrap().description;
                description.models["model"].inputs[0].shape = vec![-1];
                description.models["model"].outputs[0].shape = vec![-1];
                let engine = engine(&bundle, device);
                assert!(!engine.workspace.lock().unwrap().models[0].is_prepared());
                if name == "double" {
                    let values = [1.5f64, 2.25, 3.125, 4.0625];
                    let result = engine
                        .run(
                            &[("model/values", BorrowedTensor::cpu(&values, &[4]).unwrap())],
                            &OutputPlacement::default(),
                            &mut HashMap::new(),
                        )
                        .unwrap();
                    assert_eq!(result[output].to_cpu_vec::<f64>().unwrap(), values);
                } else {
                    let values = [1i16, -2, 3, -4];
                    let result = engine
                        .run(
                            &[("model/values", BorrowedTensor::cpu(&values, &[4]).unwrap())],
                            &OutputPlacement::default(),
                            &mut HashMap::new(),
                        )
                        .unwrap();
                    assert_eq!(result[output].to_cpu_vec::<i16>().unwrap(), values);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires native TorchScript/NativeRT backends and CPU/CUDA fixtures"]
fn retained_inputs_and_pt2_buffer_updates_preserve_model_state() {
    for device in devices() {
        let saved = bundle("jit", "retaining", device);
        let mut session = Session::open(
            &saved,
            SessionOptions {
                engine: crate::EngineKind::Native,
                device,
                ..Default::default()
            },
        )
        .unwrap();
        let mut previous = [0f32; 4];
        for data in [[1f32; 4], [7.; 4], [-3.; 4]] {
            let result = session
                .run(&std::collections::HashMap::from([(
                    "model/values",
                    BorrowedTensor::cpu(&data, &[4]).unwrap(),
                )]))
                .unwrap();
            assert_eq!(
                result
                    .get("model/result")
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap(),
                previous
            );
            previous = data;
        }
        let stateful = bundle("pt2", "stateful", device);
        let mut session = Session::open(
            &stateful,
            SessionOptions {
                engine: crate::EngineKind::Native,
                device,
                ..Default::default()
            },
        )
        .unwrap();
        let location = match device {
            ExecutionDevice::Cpu => MemoryLocation::Cpu,
            ExecutionDevice::Cuda { device_id } => MemoryLocation::Cuda { device_id },
        };
        let input = placed(&[1f32; 4], &[4], location).unwrap();
        for step in 1..=3 {
            let result = session
                .run(&std::collections::HashMap::from([(
                    "model/values",
                    input.view(),
                )]))
                .unwrap();
            assert_eq!(
                result
                    .get("model/output_0")
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap(),
                [step as f32; 4]
            );
            // Reset only affects explicit feedback, not model-internal state.
            session.reset().unwrap();
        }
    }
}

#[test]
#[ignore = "requires native NativeRT backend and CPU/CUDA fixtures"]
fn nativert_places_archive_weights_on_the_selected_device() {
    for (exported, execution) in [
        (ExecutionDevice::Cpu, ExecutionDevice::Cuda { device_id: 0 }),
        (ExecutionDevice::Cuda { device_id: 0 }, ExecutionDevice::Cpu),
    ] {
        let bundle = bundle("pt2", "weighted", exported);
        let mut session = Session::open(
            &bundle,
            SessionOptions {
                engine: crate::EngineKind::Native,
                device: execution,
                ..Default::default()
            },
        )
        .unwrap();
        let location = match execution {
            ExecutionDevice::Cpu => MemoryLocation::Cpu,
            ExecutionDevice::Cuda { device_id } => MemoryLocation::Cuda { device_id },
        };
        let input = placed(&[1f32, 2., 3., 4.], &[4], location).unwrap();
        let result = session
            .run(&std::collections::HashMap::from([(
                "model/values",
                input.view(),
            )]))
            .unwrap();
        assert_eq!(
            result
                .get("model/output_0")
                .unwrap()
                .to_cpu_vec::<f32>()
                .unwrap(),
            [12., 26., 42., 60.]
        );
    }
}
