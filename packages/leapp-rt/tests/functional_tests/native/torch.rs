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
#[ignore = "requires native TorchScript/ONNX backends, fixtures and CUDA"]
fn supplied_cuda_outputs_do_not_round_trip_through_cpu() {
    let device = ExecutionDevice::Cuda { device_id: 0 };
    let location = MemoryLocation::Cuda { device_id: 0 };
    for name in [
        "constant_forward",
        "constant_forward_dynamic",
        "constant_forward_dynamic_consumer",
    ] {
        let engine = engine(&bundle("jit", name, device), device);
        let input = placed(&[0_f32; 4], &[4], location).unwrap();
        for size in [4, 3, 4] {
            let bias = placed(&vec![0_f32; size], &[size as i64], location).unwrap();
            let target = placed(&[0_f32; 4], &[4], location).unwrap();
            let descriptor = *target.view().descriptor();
            let mut outputs = HashMap::from([(
                "preprocess/constant",
                // SAFETY: this exclusive target is disjoint and alive through execution.
                unsafe { OutputBuffer::new(&descriptor).unwrap() },
            )]);
            engine.workspace.lock().unwrap().transfers.copies.clear();
            let result = engine.run(
                &[
                    ("model/INPUT__9", bias.view()),
                    ("preprocess/values", input.view()),
                ],
                &OutputPlacement::default(),
                &mut outputs,
            );
            let workspace = engine.workspace.lock().unwrap();
            assert!(workspace.forwarded_bindings.is_empty());
            if size == 3 {
                assert!(result.is_err());
                continue;
            }
            result.unwrap();
            let copies = &workspace.transfers.copies;
            assert!(
                copies.iter().all(|&pair| pair == (location, location)),
                "forwarding a CUDA destination must not introduce host copies: {copies:?}"
            );
            assert_eq!(target.to_cpu_vec::<f32>().unwrap(), [1., 2., 3., 4.]);
        }
    }
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
                                    default_location: Some(output_location),
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
#[ignore = "requires native TorchScript/NativeRT backends and CPU/CUDA fixtures"]
fn torch_bounded_shapes_and_supplied_outputs_recover_after_errors() {
    for device in devices() {
        for (backend, name, output) in [
            ("jit", "dynamic", "model/result"),
            ("pt2", "reshape", "model/output_0"),
        ] {
            let mut bundle = bundle(backend, name, device);
            let description = &mut Arc::get_mut(&mut bundle.0).unwrap().description;
            description
                .system_information
                .insert("shape contract version".into(), 1.into());
            let model = &mut description.models["model"];
            for (port, shape, max) in [
                (&mut model.inputs[0], vec![4], 8),
                (&mut model.outputs[0], vec![2, 2], 3),
            ] {
                let mut axes = vec![json!({}); shape.len()];
                axes[0] = json!({"variable":{"symbol":port.name,"min":0,"max":max}});
                port.shape = shape;
                port.metadata
                    .insert("axes".into(), yaml_serde::to_value(axes).unwrap());
            }
            // Recompile the edited contracts, as loading does.
            crate::bundle::shapes::compile(description).unwrap();
            let original = Session::open(
                &bundle,
                SessionOptions {
                    device,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut session = original.new_session().unwrap();
            drop(original);
            let location = crate::engines::tests::support::execution_location(device);
            let mut retained = Vec::new();
            for (frame, size) in [4, 3, 0, 6, 8, 10, 2, 4].into_iter().enumerate() {
                let values = vec![frame as f32; size];
                let input = placed(&values, &[size as i64], location).unwrap();
                let inputs = HashMap::from([("model/values", input.view())]);
                let result = session.run(&inputs);
                if matches!(size, 3 | 8 | 10) {
                    // Invalid reshape, producer output bound, and public input bound.
                    let error = result.err().expect("invalid shape must fail");
                    if size == 3 {
                        assert_eq!(error.phase(), "execute");
                    } else {
                        assert!(error.to_string().contains("declared dtype/shape"));
                    }
                    continue;
                }
                let result = result.unwrap().get(output).unwrap().clone();
                assert_eq!(result.shape(), [size as i64 / 2, 2]);
                retained.push((values.clone(), result));
                // The wrong shape satisfies the bounds but differs from the model result.
                let wrong_rows = if size == 2 { 2 } else { 1 };
                let wrong = placed(
                    &vec![0_f32; wrong_rows * 2],
                    &[wrong_rows as i64, 2],
                    location,
                )
                .unwrap();
                let descriptor = *wrong.view().descriptor();
                assert!(session
                    .run_into(
                        &inputs,
                        &mut HashMap::from([(output, unsafe {
                            // SAFETY: disjoint destination stays alive through synchronous execution.
                            OutputBuffer::new(&descriptor).unwrap()
                        })])
                    )
                    .is_err());
                let target =
                    placed(&vec![-999_f32; size], &[size as i64 / 2, 2], location).unwrap();
                let descriptor = *target.view().descriptor();
                let mut buffers = HashMap::from([(output, unsafe {
                    // SAFETY: target is exclusively writable and alive through completion.
                    OutputBuffer::new(&descriptor).unwrap()
                })]);
                let result = session.run_into(&inputs, &mut buffers).unwrap();
                assert!(result.get(output).is_none());
                assert_eq!(buffers[output].descriptor().data, descriptor.data);
                drop(buffers);
                assert_eq!(target.to_cpu_vec::<f32>().unwrap(), values);
                assert_eq!(input.to_cpu_vec::<f32>().unwrap(), values);
            }
            drop(session);
            for (expected, result) in retained {
                assert_eq!(result.to_cpu_vec::<f32>().unwrap(), expected);
            }
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
                crate::bundle::shapes::compile(description).unwrap();
                let engine = engine(&bundle, device);
                assert!(!engine.workspace.lock().unwrap().models[0].is_prepared());
                if name == "double" {
                    let values = [1.5f64, 2.25, 3.125, 4.0625];
                    let input = placed(&values, &[4], execution_location(&engine)).unwrap();
                    let result = engine
                        .run(
                            &[("model/values", input.view())],
                            &OutputPlacement::default(),
                            &mut HashMap::new(),
                        )
                        .unwrap();
                    assert_eq!(result[output].to_cpu_vec::<f64>().unwrap(), values);
                } else {
                    let values = [1i16, -2, 3, -4];
                    let input = placed(&values, &[4], execution_location(&engine)).unwrap();
                    let result = engine
                        .run(
                            &[("model/values", input.view())],
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

#[test]
#[ignore = "requires native TorchScript/NativeRT backends and CPU/CUDA fixtures"]
fn torch_results_are_lent_without_copies() {
    for device in devices() {
        for (dynamic, fail) in [(false, false), (false, true), (true, false), (true, true)] {
            // A NativeRT node feeds a TorchScript node.
            let mut pipeline = bundle("jit", "mixed_pt2", device);
            let description = &mut Arc::get_mut(&mut pipeline.0).unwrap().description;
            if dynamic {
                let preprocess = &mut description.models["preprocess"];
                preprocess.inputs[0].shape = vec![-1];
                preprocess.outputs[0].shape = vec![-1];
                description.models["model"].inputs[0].shape = vec![-1];
            }
            if fail {
                // The model's result no longer matches its declared shape.
                description.models["model"].outputs[0].shape = vec![3];
            }
            crate::bundle::shapes::compile(description).unwrap();
            let engine = engine(&pipeline, device);
            let location = execution_location(&engine);
            let input = placed(&[2_f32; 4], &[4], location).unwrap();
            let canary = placed(&[-999_f32; 4], &[4], location).unwrap();
            let id = {
                let mut guard = engine.workspace.lock().unwrap();
                let workspace = &mut *guard;
                let id = workspace.plan.calls[0].outputs[0];
                if !dynamic {
                    // SAFETY: the prepared destination is idle and matches the canary.
                    unsafe {
                        workspace.slots[id]
                            .assign_validated(canary.view(), 16, &mut workspace.transfers)
                            .unwrap();
                    }
                }
                workspace.transfers.copies.clear();
                id
            };
            let result = engine.run(
                &[
                    ("model/INPUT__9", input.view()),
                    ("preprocess/values", input.view()),
                ],
                &OutputPlacement::default(),
                &mut HashMap::new(),
            );
            let mut workspace = engine.workspace.lock().unwrap();
            // Lent results are released after the graph, also on failure.
            assert!(workspace
                .models
                .iter()
                .all(|model| model.forwarded_output(0).is_none()));
            if fail {
                let error = result.err().unwrap().to_string();
                assert!(error.contains("execute node model"), "{error}");
                continue;
            }
            assert!(workspace.transfers.copies.is_empty());
            assert_eq!(
                result.unwrap()["model/z_result"]
                    .to_cpu_vec::<f32>()
                    .unwrap(),
                [2., 3., 4., 5.].map(|scale| 3. * scale + 2.)
            );
            if !dynamic {
                // The producer's result was lent, not written into its fixed slot.
                let workspace = &mut *workspace;
                let slot = workspace.slots[id]
                    .snapshot(MemoryLocation::Cpu, &mut workspace.transfers)
                    .unwrap();
                assert_eq!(slot.to_cpu_vec::<f32>().unwrap(), [-999.; 4]);
            }
        }
    }
}
