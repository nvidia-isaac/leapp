// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::engines::tests::support::{placed, TestTensor};
use crate::{ErrorCode, Session};
use serde_json::{json, Value};
use std::{collections::HashMap, fs, path::PathBuf};

fn execution_location(engine: &NativeEngine) -> MemoryLocation {
    match engine.options.device {
        ExecutionDevice::Cpu => MemoryLocation::Cpu,
        ExecutionDevice::Cuda { device_id } => MemoryLocation::Cuda { device_id },
    }
}

fn fixtures() -> PathBuf {
    std::env::var_os("LEAPP_TRITON_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/triton-fixtures"))
}

fn fixture(name: &str) -> Value {
    let root = fixtures();
    let mut data: Value =
        serde_json::from_slice(&fs::read(root.join(format!("{name}.yaml"))).unwrap()).unwrap();
    for model in data["models"].as_object_mut().unwrap().values_mut() {
        model["parameters"]["model_path"] =
            json!(root.join(model["parameters"]["model_path"].as_str().unwrap()));
    }
    data
}

fn devices() -> [ExecutionDevice; 2] {
    [ExecutionDevice::Cpu, ExecutionDevice::Cuda { device_id: 0 }]
}

fn locations() -> [MemoryLocation; 3] {
    [
        MemoryLocation::Cpu,
        MemoryLocation::PinnedCpu,
        MemoryLocation::Cuda { device_id: 0 },
    ]
}

type Identities = (Vec<(usize, usize)>, Vec<Option<Vec<usize>>>);

fn identities(workspace: &Workspace) -> Identities {
    (
        workspace
            .slots
            .iter()
            .map(|slot| {
                let value = slot.view().unwrap();
                (
                    value.descriptor().data as usize,
                    value.descriptor().shape as usize,
                )
            })
            .collect(),
        workspace
            .models
            .iter()
            .map(|model| model.prepared_identity())
            .collect(),
    )
}

// Each peer retains the loaded object, but none of its prepared bindings/storage.
fn assert_private_peer(engine: &NativeEngine) -> NativeEngine {
    let peer = engine.peer().unwrap();
    let original = engine.workspace.lock().unwrap();
    let other = peer
        .workspace
        .try_lock()
        .expect("peers must have independent locks");
    assert!(Arc::ptr_eq(&original.plan, &other.plan));
    assert_eq!(
        original
            .models
            .iter()
            .map(|model| model.loaded_identity())
            .collect::<Vec<_>>(),
        other
            .models
            .iter()
            .map(|model| model.loaded_identity())
            .collect::<Vec<_>>()
    );
    let a = identities(&original);
    let b = identities(&other);
    for (left, right) in a.0.iter().zip(&b.0) {
        assert_ne!(left.0, right.0, "workspace allocations must be disjoint");
        assert_ne!(left.1, right.1, "shape storage must be private");
    }
    for (left, right) in a.1.iter().zip(&b.1) {
        if left.is_some() || right.is_some() {
            assert_ne!(left, right, "prepared metadata must be private");
        }
    }
    drop(other);
    peer
}

#[test]
fn onnx_peers_share_models_and_run_with_independent_workspaces() {
    for device in devices() {
        let bundle = Bundle::open(fixtures().join("slice.yaml")).unwrap();
        let engine = NativeEngine::load(
            &bundle,
            &Graph::new(&bundle).unwrap(),
            &SessionOptions {
                engine: crate::EngineKind::Native,
                device,
                ..Default::default()
            },
        )
        .unwrap();
        let peer = assert_private_peer(&engine);
        // Lock the original workspace throughout the peer's call. This proves
        // the peer can execute without acquiring the original workspace lock.
        let original = engine.workspace.lock().unwrap();
        let data = placed(&[10f32, 20., 30., 40.], &[4], execution_location(&engine)).unwrap();
        let lower = placed(&[1i64], &[1], execution_location(&engine)).unwrap();
        let upper = placed(&[3i64], &[1], execution_location(&engine)).unwrap();
        let output = peer
            .run(
                &[
                    ("slice/data", data.view()),
                    ("slice/start", lower.view()),
                    ("slice/end", upper.view()),
                ],
                &OutputPlacement::default(),
                &mut HashMap::new(),
            )
            .unwrap();
        assert_eq!(
            output["slice/output"].to_cpu_vec::<f32>().unwrap(),
            [20., 30.]
        );
        drop(original);

        // Failed peer allocation leaves the loaded parent usable. A checked
        // size overflow exercises the allocation failure path without an OOM.
        let workspace = engine.workspace.lock().unwrap();
        let invalid = Plan {
            calls: vec![],
            inputs: IndexMap::new(),
            public_inputs: 0,
            outputs: IndexMap::new(),
            terminal_outputs: Default::default(),
            slots: vec![Arc::new(plan::TensorSpec {
                name: "invalid".into(),
                dtype: crate::DataType::Float32,
                shape: vec![i64::MAX, i64::MAX],
            })],
        };
        let models = workspace
            .models
            .iter()
            .map(|model| model.new_session())
            .collect::<anyhow::Result<Vec<_>>>()
            .unwrap();
        drop(workspace);
        assert!(
            NativeEngine::with_models(Arc::new(invalid), models, None, &engine.options).is_err()
        );

        let start = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            for (index, engine) in [&engine, &peer].into_iter().enumerate() {
                let start = &start;
                scope.spawn(move || {
                    let before = identities(&engine.workspace.lock().unwrap());
                    start.wait();
                    for iteration in 0..40 {
                        let first = ((iteration + index) % 3) as i64;
                        let data =
                            placed(&[10f32, 20., 30., 40.], &[4], execution_location(engine))
                                .unwrap();
                        let lower = placed(&[first], &[1], execution_location(engine)).unwrap();
                        let upper = placed(&[first + 2], &[1], execution_location(engine)).unwrap();
                        let result = engine
                            .run(
                                &[
                                    ("slice/data", data.view()),
                                    ("slice/start", lower.view()),
                                    ("slice/end", upper.view()),
                                ],
                                &OutputPlacement::default(),
                                &mut HashMap::new(),
                            )
                            .unwrap();
                        assert_eq!(
                            result["slice/output"].to_cpu_vec::<f32>().unwrap(),
                            [10. * (first + 1) as f32, 10. * (first + 2) as f32]
                        );
                    }
                    assert_eq!(before, identities(&engine.workspace.lock().unwrap()));
                });
            }
        });
    }
}

#[test]
fn fixed_buffers_and_ort_metadata_survive_changing_cpu_and_cuda_controls() {
    let directory = tempfile::tempdir().unwrap();
    let mut data = fixture("slice");
    // Session argument order stays independent of the model port order.
    data["models"]["slice"]["inputs"]
        .as_array_mut()
        .unwrap()
        .reverse();
    let bundle = write_bundle(&data, directory.path());
    for device in devices() {
        let engine = NativeEngine::load(
            &bundle,
            &Graph::new(&bundle).unwrap(),
            &SessionOptions {
                engine: crate::EngineKind::Native,
                device,
                ..Default::default()
            },
        )
        .unwrap();
        let before = identities(&engine.workspace.lock().unwrap());
        assert!(before.1.iter().all(Option::is_some));
        let direct_inputs = engine.inputs.iter().any(|(_, alignment)| *alignment != 0);
        // An invalid fixed-output shape must not invalidate cached ORT handles.
        let data = placed(&[10f32, 20., 30., 40.], &[4], execution_location(&engine)).unwrap();
        let lower = placed(&[0i64], &[1], execution_location(&engine)).unwrap();
        let upper = placed(&[1i64], &[1], execution_location(&engine)).unwrap();
        assert!(engine
            .run(
                &[
                    ("slice/data", data.view()),
                    ("slice/start", lower.view()),
                    ("slice/end", upper.view()),
                ],
                &OutputPlacement::default(),
                &mut HashMap::new(),
            )
            .is_err());
        assert_eq!(identities(&engine.workspace.lock().unwrap()), before);
        let mut retained = Vec::new();
        for input_location in locations() {
            let data = placed(&[10f32, 20., 30., 40.], &[4], input_location).unwrap();
            for output_location in locations() {
                for start in [0i64, 2, 1, 0] {
                    let lower = placed(&[start], &[1], input_location).unwrap();
                    let upper = placed(&[start + 2], &[1], input_location).unwrap();
                    let result = engine.run(
                        &[
                            ("slice/data", data.view()),
                            ("slice/start", lower.view()),
                            ("slice/end", upper.view()),
                        ],
                        &OutputPlacement {
                            default_location: output_location,
                            ..Default::default()
                        },
                        &mut HashMap::new(),
                    );
                    if direct_inputs && input_location != execution_location(&engine) {
                        let error = result.err().unwrap();
                        assert_eq!(error.code(), ErrorCode::InvalidArgument);
                        assert!(error.to_string().contains("execution device"));
                        continue;
                    }
                    let result = result.unwrap();
                    assert_eq!(result["slice/output"].location(), output_location);
                    retained.push((start as usize, result));
                    assert_eq!(identities(&engine.workspace.lock().unwrap()), before);
                }
            }
        }
        drop(engine);
        for (start, output) in retained {
            assert_eq!(
                output["slice/output"].to_cpu_vec::<f32>().unwrap(),
                [10., 20., 30., 40.][start..start + 2]
            );
        }
    }
}

#[test]
fn onnx_port_names_and_metadata_are_validated_at_load() {
    for device in devices() {
        for (ports, index, field, value, message) in [
            ("inputs", 1, "name", json!("missing_input"), "missing_input"),
            (
                "outputs",
                0,
                "name",
                json!("missing_output"),
                "missing_output",
            ),
            ("inputs", 1, "dtype", json!("float32"), "start"),
            ("inputs", 1, "shape", json!([2]), "start"),
            ("outputs", 0, "dtype", json!("int64"), "output"),
            ("outputs", 0, "shape", json!([3]), "output"),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let mut data = fixture("slice");
            data["models"]["slice"][ports][index][field] = value.clone();
            if field == "name" {
                data["pipeline"][ports]["slice"][index] = value;
            }
            let bundle = write_bundle(&data, directory.path());
            let error = Session::open(
                &bundle,
                SessionOptions {
                    engine: crate::EngineKind::Native,
                    device,
                    ..Default::default()
                },
            )
            .err()
            .unwrap();
            assert_eq!(error.code(), ErrorCode::InvalidArgument);
            assert_eq!(error.phase(), "load");
            assert!(error.to_string().contains(message), "{error}");
        }
    }
}

#[test]
fn forward_edges_share_fixed_storage_and_follow_dependency_order() {
    let bundle = Bundle::open(fixtures().join("policy.yaml")).unwrap();
    for device in devices() {
        let engine = NativeEngine::load(
            &bundle,
            &Graph::new(&bundle).unwrap(),
            &SessionOptions {
                engine: crate::EngineKind::Native,
                device,
                ..Default::default()
            },
        )
        .unwrap();
        let workspace = engine.workspace.lock().unwrap();
        let calls = &workspace.plan.calls;
        assert_eq!(
            calls
                .iter()
                .map(|call| call.name.as_str())
                .collect::<Vec<_>>(),
            ["preprocess", "policy"]
        );
        assert_eq!(calls[0].outputs[0], calls[1].inputs[0]);
        assert!(workspace.models.iter().all(|model| model.is_prepared()));
    }
}

#[test]
fn private_terminal_outputs_are_not_published() {
    for device in devices() {
        let directory = tempfile::tempdir().unwrap();
        let mut data = fixture("named_ports");
        data["pipeline"]["data_flow"] = json!({"arithmetic/difference": []});
        data["pipeline"]["outputs"] = json!({"arithmetic": ["sum"]});
        let bundle = write_bundle(&data, directory.path());
        let engine = NativeEngine::load(
            &bundle,
            &Graph::new(&bundle).unwrap(),
            &SessionOptions {
                engine: crate::EngineKind::Native,
                device,
                ..Default::default()
            },
        )
        .unwrap();
        let location = execution_location(&engine);
        let left = placed(&[7f32; 4], &[4], location).unwrap();
        let right = placed(&[2f32; 4], &[4], location).unwrap();
        let result = engine
            .run(
                &[
                    ("arithmetic/left", left.view()),
                    ("arithmetic/right", right.view()),
                ],
                &OutputPlacement::default(),
                &mut HashMap::new(),
            )
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result["arithmetic/sum"].to_cpu_vec::<f32>().unwrap(),
            [9.; 4]
        );
    }
}

#[test]
fn scalar_and_empty_outputs_use_the_prepared_path() {
    for device in devices() {
        for (name, values, shape) in [
            ("scalar", &[3f32][..], &[][..]),
            ("empty", &[][..], &[0][..]),
        ] {
            let bundle = Bundle::open(fixtures().join(format!("{name}.yaml"))).unwrap();
            let engine = NativeEngine::load(
                &bundle,
                &Graph::new(&bundle).unwrap(),
                &SessionOptions {
                    engine: crate::EngineKind::Native,
                    device,
                    ..Default::default()
                },
            )
            .unwrap();
            assert!(engine.workspace.lock().unwrap().models[0].is_prepared());
            let input = format!("{name}/input");
            let output = format!("{name}/output");
            for _ in 0..2 {
                let values_input = placed(values, shape, execution_location(&engine)).unwrap();
                let result = engine
                    .run(
                        &[(&input, values_input.view())],
                        &OutputPlacement::default(),
                        &mut HashMap::new(),
                    )
                    .unwrap();
                assert_eq!(result[&output].to_cpu_vec::<f32>().unwrap(), values);
                assert_eq!(result[&output].shape(), shape);
            }
        }
    }
}

fn policy_copy(directory: &std::path::Path) -> Value {
    let mut data: Value =
        serde_json::from_slice(&fs::read(fixtures().join("policy.yaml")).unwrap()).unwrap();
    for (name, model) in data["models"].as_object_mut().unwrap() {
        let destination = directory.join(name);
        fs::create_dir(&destination).unwrap();
        fs::copy(
            fixtures().join(name).join("model.onnx"),
            destination.join("model.onnx"),
        )
        .unwrap();
        let weights = fixtures().join(name).join("weights.bin");
        if weights.exists() {
            fs::copy(weights, destination.join("weights.bin")).unwrap();
        }
        model["parameters"]["model_path"] = json!(destination.join("model.onnx"));
    }
    data
}

fn write_bundle(data: &Value, directory: &std::path::Path) -> Bundle {
    let path = directory.join("bundle.yaml");
    fs::write(&path, serde_json::to_vec(data).unwrap()).unwrap();
    Bundle::open(path).unwrap()
}

#[test]
fn new_sessions_and_reset_do_not_reopen_models_or_external_weights() {
    for device in devices() {
        let directory = tempfile::tempdir().unwrap();
        let data = policy_copy(directory.path());
        let bundle = write_bundle(&data, directory.path());
        let mut original = Session::open(
            &bundle,
            SessionOptions {
                engine: crate::EngineKind::Native,
                device,
                ..Default::default()
            },
        )
        .unwrap();
        directory.close().unwrap();
        let mut peer = original.new_session().unwrap();
        original.close();
        let location = match device {
            ExecutionDevice::Cpu => MemoryLocation::Cpu,
            ExecutionDevice::Cuda { device_id } => MemoryLocation::Cuda { device_id },
        };
        let observation = placed(&[1f32, 2., 3., 4.], &[4], location).unwrap();
        let bias = placed(&[10f32, 20., 30., 40.], &[4], location).unwrap();
        for _ in 0..2 {
            let result = peer
                .run(&std::collections::HashMap::from([
                    ("preprocess/observation", observation.view()),
                    ("policy/bias", bias.view()),
                ]))
                .unwrap();
            assert_eq!(
                result
                    .get("policy/action")
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap(),
                [14., 29., 46., 65.]
            );
            peer.reset().unwrap();
        }
    }
}

#[test]
fn unsupported_backends_are_rejected_before_any_model_is_loaded() {
    let directory = tempfile::tempdir().unwrap();
    let mut data = policy_copy(directory.path());
    // The earlier ONNX model would fail to load if backend validation were late.
    fs::write(directory.path().join("policy/model.onnx"), b"invalid ONNX").unwrap();
    data["models"]["preprocess"]["parameters"]["backend"] = json!("unsupported-test-backend");
    let bundle = write_bundle(&data, directory.path());
    let error = Session::open(
        &bundle,
        SessionOptions {
            engine: crate::EngineKind::Native,
            ..Default::default()
        },
    )
    .err()
    .unwrap();
    assert_eq!(error.code(), ErrorCode::Unsupported);
    assert_eq!(error.phase(), "load");
    assert!(error.to_string().contains("preprocess"));
    assert!(error.to_string().contains("unsupported-test-backend"));
}

#[test]
fn dynamic_outputs_feed_fixed_and_dynamic_nodes_and_recover_after_shape_errors() {
    for (device, output_shape) in devices()
        .into_iter()
        .flat_map(|device| [vec![-1], vec![4]].map(|shape| (device, shape)))
    {
        let directory = tempfile::tempdir().unwrap();
        let policy = policy_copy(directory.path());
        let mut dynamic: Value =
            serde_json::from_slice(&fs::read(fixtures().join("dynamic.yaml")).unwrap()).unwrap();
        dynamic["models"]["dynamic"]["parameters"]["model_path"] =
            json!(fixtures().join("dynamic/model.onnx"));
        let mut data = json!({
            "system information": {"leapp config version": "1.3"},
            "models": {
                "first": dynamic["models"]["dynamic"].clone(),
                "fixed": policy["models"]["preprocess"].clone(),
                "last": dynamic["models"]["dynamic"].clone()
            },
            "pipeline": {
                "inputs": {"first": ["input"]},
                "outputs": {"last": ["output"]},
                "data_flow": {"first/output": ["fixed/observation"], "fixed/features": ["last/input"]},
                "feedback_flow": {}, "configs": {}
            }
        });
        // Exercise declared output dimensions that the dynamic ONNX artifact
        // can only establish during execution.
        data["models"]["first"]["outputs"][0]["shape"] = json!(output_shape);
        data["models"]["fixed"]["inputs"][0]["shape"] = json!(output_shape);
        data["models"]["last"]["inputs"][0]["shape"] = json!([4]);
        let bundle = write_bundle(&data, directory.path());
        let mut session = Session::open(
            &bundle,
            SessionOptions {
                engine: crate::EngineKind::Native,
                device,
                ..Default::default()
            },
        )
        .unwrap();
        let location = crate::engines::tests::support::execution_location(device);
        let input = placed(&[1f32; 4], &[4], location).unwrap();
        let first = session
            .run(&HashMap::from([("first/input", input.view())]))
            .unwrap();
        let invalid_shape = placed(&[9f32; 3], &[3], location).unwrap();
        assert!(session
            .run(&HashMap::from([("first/input", invalid_shape.view())]))
            .is_err());
        let input = placed(&[3f32; 4], &[4], location).unwrap();
        let second = session
            .run(&HashMap::from([("first/input", input.view())]))
            .unwrap();
        assert_eq!(
            first
                .get("last/output")
                .unwrap()
                .to_cpu_vec::<f32>()
                .unwrap(),
            [2.; 4]
        );
        assert_eq!(
            second
                .get("last/output")
                .unwrap()
                .to_cpu_vec::<f32>()
                .unwrap(),
            [4.; 4]
        );
    }
}

#[path = "native/tensorrt.rs"]
mod tensorrt;

#[path = "native/torch.rs"]
mod torch;

#[path = "native/streams.rs"]
mod streams;
