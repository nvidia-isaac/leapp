// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::sync::{Arc, Barrier};

fn fixture(name: &str) -> PathBuf {
    std::env::var_os("LEAPP_TENSORRT_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tensorrt-fixtures")
        })
        .join(format!("{name}.yaml"))
}

fn bundle(name: &str) -> Bundle {
    Bundle::open(fixture(name)).expect("generate TensorRT fixtures first")
}

fn options() -> SessionOptions {
    SessionOptions {
        engine: crate::EngineKind::Native,
        device: ExecutionDevice::Cuda { device_id: 0 },
        ..Default::default()
    }
}

fn load(name: &str) -> NativeEngine {
    let bundle = bundle(name);
    NativeEngine::load(&bundle, &Graph::new(&bundle).unwrap(), &options()).unwrap()
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn fixed_bindings_reuse_addresses_metadata_and_contexts_in_mixed_graphs() {
    for name in ["policy", "mixed", "dynamic_fixed", "shape_fixed"] {
        let engine = load(name);
        let _peer = assert_private_peer(&engine);
        let before = identities(&engine.workspace.lock().unwrap());
        assert!(before.1.iter().all(Option::is_some), "{name}");
        let mut retained = Vec::new();
        for input_location in locations() {
            for output_location in locations() {
                for offset in [0f32, 1., 2., 0.] {
                    let inputs = match name {
                        "dynamic_fixed" => vec![(
                            "dynamic/input",
                            placed(&[offset; 8], &[8], input_location).unwrap(),
                        )],
                        "shape_fixed" => vec![(
                            "shape/data",
                            placed(&[offset; 4], &[2, 2], input_location).unwrap(),
                        )],
                        _ => vec![
                            (
                                "preprocess/observation",
                                placed(&[1f32, 2., 3., 4.], &[4], input_location).unwrap(),
                            ),
                            (
                                "policy/bias",
                                placed(
                                    &[10f32, 20., 30., 40.].map(|v| v + offset),
                                    &[4],
                                    input_location,
                                )
                                .unwrap(),
                            ),
                        ],
                    };
                    let inputs = inputs
                        .iter()
                        .map(|(name, value)| (*name, value.view()))
                        .collect::<Vec<_>>();
                    let result = engine
                        .run(
                            &inputs,
                            &OutputPlacement {
                                default_location: output_location,
                                ..Default::default()
                            },
                            &Arc::new(OutputPool::new(Default::default(), 0)),
                        )
                        .unwrap();
                    let output_name = match name {
                        "dynamic_fixed" => "dynamic/output",
                        "shape_fixed" => "shape/output",
                        _ => "policy/action",
                    };
                    let value = result[output_name].clone();
                    assert_eq!(value.location(), output_location);
                    if name == "shape_fixed" {
                        assert_eq!(value.to_cpu_vec::<i32>().unwrap(), [2, 2]);
                    } else {
                        let expected = if name == "dynamic_fixed" {
                            vec![offset; 8]
                        } else {
                            [14., 29., 46., 65.].map(|v| v + offset).to_vec()
                        };
                        assert_eq!(value.to_cpu_vec::<f32>().unwrap(), expected);
                        retained.push((expected, value));
                    }
                    assert_eq!(identities(&engine.workspace.lock().unwrap()), before);
                }
            }
        }
        drop(engine);
        for (expected, value) in retained {
            assert_eq!(value.to_cpu_vec::<f32>().unwrap(), expected);
        }
    }
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn peer_profiles_match_all_inputs_and_recover_after_rejection() {
    let session = Session::open(&bundle("concat"), options()).unwrap();
    let peer = session.new_session().unwrap();
    let start = Barrier::new(2);
    std::thread::scope(|scope| {
        for (index, mut session) in [session, peer].into_iter().enumerate() {
            let start = &start;
            scope.spawn(move || {
                let cases = [(2usize, 2usize), (2, 5), (3, 2), (3, 6), (1, 1)];
                start.wait();
                for step in 0..cases.len() {
                    let (left, right) = cases[(step + index) % cases.len()];
                    let a = vec![index as f32 + 1.; left];
                    let b = vec![index as f32 + 2.; right];
                    let result = session.run(&std::collections::HashMap::from([
                        (
                            "concat/left",
                            BorrowedTensor::cpu(&a, &[left as i64]).unwrap(),
                        ),
                        (
                            "concat/right",
                            BorrowedTensor::cpu(&b, &[right as i64]).unwrap(),
                        ),
                    ]));
                    if (left, right) == (3, 2) {
                        let error = result.err().unwrap();
                        assert_eq!(error.code(), ErrorCode::InvalidArgument);
                        assert!(error.to_string().contains("no optimization profile"));
                    } else {
                        assert_eq!(
                            result
                                .unwrap()
                                .get("concat/output")
                                .unwrap()
                                .to_cpu_vec::<f32>()
                                .unwrap(),
                            a.into_iter().chain(b).collect::<Vec<_>>()
                        );
                    }
                }
            });
        }
    });
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn shape_control_values_refresh_with_concrete_bundle_dimensions() {
    let engine = load("slice");
    // Fixed dimensions do not make control values static.
    assert!(!engine.workspace.lock().unwrap().models[0].is_prepared());
    let data = [10f32, 20., 30., 40.];
    let mut retained = Vec::new();
    for location in locations() {
        for (start, end) in [(0i64, 2i64), (1, 3), (2, 4), (0, 5), (1, 4), (1, 3), (0, 2)] {
            let input = placed(&data, &[4], location).unwrap();
            let lower = placed(&[start], &[1], location).unwrap();
            let upper = placed(&[end], &[1], location).unwrap();
            let result = engine.run(
                &[
                    ("slice/data", input.view()),
                    ("slice/start", lower.view()),
                    ("slice/end", upper.view()),
                ],
                &OutputPlacement::default(),
                &Arc::new(OutputPool::new(Default::default(), 0)),
            );
            if end - start != 2 {
                assert_eq!(result.err().unwrap().code(), ErrorCode::InvalidArgument);
            } else {
                let result = result.unwrap();
                let expected = data[start as usize..end as usize].to_vec();
                assert_eq!(
                    result["slice/output"].to_cpu_vec::<f32>().unwrap(),
                    expected
                );
                retained.push((expected, result));
            }
        }
    }
    drop(engine);
    for (expected, result) in retained {
        assert_eq!(
            result["slice/output"].to_cpu_vec::<f32>().unwrap(),
            expected
        );
    }
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn data_dependent_outputs_empty_tensors_and_host_shape_outputs() {
    for name in ["nonzero", "nonzero_fixed"] {
        let mut session = Session::open(&bundle(name), options()).unwrap();
        let mut retained = Vec::new();
        for data in [[0f32, 2., 0., -1.], [0.; 4], [1.; 4], [0., 2., 0., -1.]] {
            let expected: Vec<_> = data
                .iter()
                .enumerate()
                .filter(|(_, v)| **v != 0.)
                .map(|(i, _)| i as i64)
                .collect();
            let result = session.run(&std::collections::HashMap::from([(
                "nonzero/data",
                BorrowedTensor::cpu(&data, &[4]).unwrap(),
            )]));
            if name == "nonzero_fixed" && expected.len() != 2 {
                assert_eq!(result.err().unwrap().code(), ErrorCode::InvalidArgument);
            } else {
                let result = result.unwrap();
                let output = result.get("nonzero/indices").unwrap();
                assert_eq!(output.shape(), [1, expected.len() as i64]);
                assert_eq!(output.to_cpu_vec::<i64>().unwrap(), expected);
                retained.push((expected, output.clone()));
            }
        }
        drop(session);
        for (expected, result) in retained {
            assert_eq!(result.to_cpu_vec::<i64>().unwrap(), expected);
        }
    }
    for (name, data, shape) in [("empty", vec![], vec![0]), ("scalar", vec![3f32], vec![])] {
        let mut session = Session::open(&bundle(name), options()).unwrap();
        let input_name = format!("{name}/input");
        let result = session
            .run(&std::collections::HashMap::from([(
                input_name.as_str(),
                BorrowedTensor::cpu(&data, &shape).unwrap(),
            )]))
            .unwrap();
        let output = result.get(&format!("{name}/output")).unwrap();
        assert_eq!(output.shape(), shape);
        assert_eq!(output.to_cpu_vec::<f32>().unwrap(), data);
    }
    let mut session = Session::open(&bundle("shape"), options()).unwrap();
    for rows in [1, 4, 2] {
        let data = vec![0f32; rows as usize * 2];
        let result = session
            .run(&std::collections::HashMap::from([(
                "shape/data",
                BorrowedTensor::cpu(&data, &[rows, 2]).unwrap(),
            )]))
            .unwrap();
        assert_eq!(
            result
                .get("shape/output")
                .unwrap()
                .to_cpu_vec::<i32>()
                .unwrap(),
            [rows as i32, 2]
        );
    }
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn feedback_is_independent_and_survives_reset_close_and_deleted_engines() {
    let directory = tempfile::tempdir().unwrap();
    let mut document: Value =
        serde_json::from_slice(&fs::read(fixture("feedback")).unwrap()).unwrap();
    let destination = directory.path().join("model.plan");
    fs::copy(
        fixture("feedback")
            .parent()
            .unwrap()
            .join("accumulator/model.plan"),
        &destination,
    )
    .unwrap();
    document["models"]["accumulator"]["parameters"]["model_path"] = json!(destination);
    let bundle = write_bundle(&document, directory.path());
    let mut original = Session::open(&bundle, options()).unwrap();
    directory.close().unwrap();
    let peers: Vec<_> = (0..3).map(|_| original.new_session().unwrap()).collect();
    original.close();
    let barrier = Arc::new(Barrier::new(3));
    let workers: Vec<_> = peers
        .into_iter()
        .enumerate()
        .map(|(index, mut session)| {
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let value = (index + 1) as f32;
                let input = placed(&[value; 4], &[4], locations()[index]).unwrap();
                barrier.wait();
                let retained = session
                    .run(&std::collections::HashMap::from([(
                        "accumulator/observation",
                        input.view(),
                    )]))
                    .unwrap();
                for step in 2..=4 {
                    assert_eq!(
                        session
                            .run(&std::collections::HashMap::from([(
                                "accumulator/observation",
                                input.view()
                            )]))
                            .unwrap()
                            .get("accumulator/next")
                            .unwrap()
                            .to_cpu_vec::<f32>()
                            .unwrap(),
                        [value * step as f32; 4]
                    );
                }
                session.reset().unwrap();
                assert_eq!(
                    session
                        .run(&std::collections::HashMap::from([(
                            "accumulator/observation",
                            input.view()
                        )]))
                        .unwrap()
                        .get("accumulator/next")
                        .unwrap()
                        .to_cpu_vec::<f32>()
                        .unwrap(),
                    [value; 4]
                );
                session.close();
                assert_eq!(
                    retained
                        .get("accumulator/next")
                        .unwrap()
                        .to_cpu_vec::<f32>()
                        .unwrap(),
                    [value; 4]
                );
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn cpu_rejection_and_engine_metadata_errors_are_classified() {
    let error = Session::open(
        &bundle("policy"),
        SessionOptions {
            engine: crate::EngineKind::Native,
            ..Default::default()
        },
    )
    .err()
    .unwrap();
    assert_eq!(error.phase(), "load");
    assert_eq!(error.code(), ErrorCode::Unsupported);
    assert!(error
        .to_string()
        .contains("TensorRT requires CUDA execution"));
    let source = fixture("dynamic");
    let original: Value = serde_json::from_slice(&fs::read(&source).unwrap()).unwrap();
    for mismatch in ["name", "dtype", "rank"] {
        let directory = tempfile::tempdir().unwrap();
        let mut document = original.clone();
        let model = &mut document["models"]["dynamic"];
        model["parameters"]["model_path"] =
            json!(source.parent().unwrap().join("dynamic/model.plan"));
        match mismatch {
            "name" => model["inputs"][0]["name"] = json!("renamed"),
            "dtype" => model["inputs"][0]["dtype"] = json!("int64"),
            _ => model["inputs"][0]["shape"] = json!([1, 1]),
        }
        if mismatch == "name" {
            document["pipeline"]["inputs"]["dynamic"] = json!(["renamed"]);
        }
        let error = Session::open(&write_bundle(&document, directory.path()), options())
            .err()
            .unwrap();
        assert_eq!(error.phase(), "load");
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
        assert!(error.to_string().contains(mismatch), "{error}");
    }
}

#[test]
#[ignore = "requires TensorRT backend, GPU, and generated engines"]
fn rng_contexts_are_private_and_prepared_across_peers() {
    let engine = load("random");
    let run = |engine: &NativeEngine, low: f32| {
        engine
            .run(
                &[("random/low", BorrowedTensor::cpu(&[low], &[1]).unwrap())],
                &OutputPlacement::default(),
                &Arc::new(OutputPool::new(Default::default(), 0)),
            )
            .unwrap()["random/output"]
            .clone()
    };
    let retained = run(&engine, 0.);
    let expected = retained.to_cpu_vec::<f32>().unwrap();
    let peer = assert_private_peer(&engine);
    assert!(engine.execution_guard.is_none());
    {
        let original = engine.workspace.lock().unwrap();
        let other = peer.workspace.lock().unwrap();
        assert_ne!(
            original.models[0].execution_identity(),
            other.models[0].execution_identity()
        );
        assert!(original.models[0].is_prepared());
        assert!(other.models[0].is_prepared());
    }
    let start = Barrier::new(2);
    std::thread::scope(|scope| {
        for engine in [&engine, &peer] {
            let start = &start;
            let run = &run;
            scope.spawn(move || {
                let before = identities(&engine.workspace.lock().unwrap());
                start.wait();
                for low in [0., 0.5, 0.25] {
                    let values = run(engine, low).to_cpu_vec::<f32>().unwrap();
                    assert_eq!(values.len(), 16);
                    assert!(values.iter().all(|value| (low..1.).contains(value)));
                    assert_eq!(identities(&engine.workspace.lock().unwrap()), before);
                }
            });
        }
    });
    // Context ownership is the contract; independent contexts need not use distinct seeds.
    drop(engine);
    run(&peer, 0.);
    assert_eq!(retained.to_cpu_vec::<f32>().unwrap(), expected);
}
