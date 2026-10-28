// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::support;
use lrt::dlpack::BorrowedTensor;
use lrt::{Bundle, ErrorCode, ExecutionDevice, MemoryLocation, RunResult, Session, SessionOptions};
use serde_json::{json, Value};
use std::{env, fs, path::PathBuf, thread};
use support::TestTensor;

fn fixtures(kind: &str) -> PathBuf {
    env::var_os(format!("LEAPP_{}_FIXTURES", kind.to_uppercase()))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("target/{kind}-fixtures"))
        })
}

fn options(device: ExecutionDevice) -> SessionOptions {
    SessionOptions {
        engine: support::engine(),
        device,
        ..Default::default()
    }
}

fn devices() -> Vec<ExecutionDevice> {
    let mut result = vec![ExecutionDevice::Cpu];
    result.push(ExecutionDevice::Cuda { device_id: 0 });
    result
}

fn open(name: &str, device: ExecutionDevice) -> Session {
    Session::open(
        &Bundle::open(fixtures("triton").join(format!("{name}.yaml"))).unwrap(),
        options(device),
    )
    .unwrap()
}

fn values(result: &RunResult, name: &str) -> Vec<f32> {
    result.get(name).unwrap().to_cpu_vec().unwrap()
}

fn step(session: &mut Session, value: f32) -> RunResult {
    session
        .run(&std::collections::HashMap::from([(
            "accumulator/observation",
            BorrowedTensor::cpu(&[value; 4], &[4]).unwrap(),
        )]))
        .unwrap()
}

#[test]
fn runtime_engine_selection_and_independent_feedback() {
    use lrt::EngineKind;

    assert_eq!(SessionOptions::default().engine, EngineKind::Triton);
    let bundle = Bundle::open(fixtures("triton").join("feedback.yaml")).unwrap();
    for device in devices() {
        let mut sessions = Vec::new();
        for engine in [EngineKind::Triton, EngineKind::Native] {
            let result = Session::open(
                &bundle,
                SessionOptions {
                    engine,
                    device,
                    ..Default::default()
                },
            );
            sessions.push(result.unwrap());
        }
        // Keep both engines alive and interleave their independent feedback updates.
        for expected in [1., 2.] {
            for session in &mut sessions {
                assert_eq!(
                    values(&step(session, 1.), "accumulator/next"),
                    [expected; 4]
                );
            }
        }
        let peers: Vec<_> = sessions.iter().map(|s| s.new_session().unwrap()).collect();
        drop(sessions);
        for mut peer in peers {
            assert_eq!(values(&step(&mut peer, 3.), "accumulator/next"), [3.; 4]);
            peer.reset().unwrap();
            assert_eq!(values(&step(&mut peer, 1.), "accumulator/next"), [1.; 4]);
        }
    }
}

// Modified fixtures keep artifact references absolute while their YAML remains temporary.
fn fixture(name: &str) -> Value {
    let root = fixtures("triton");
    let mut value: Value =
        serde_json::from_slice(&fs::read(root.join(format!("{name}.yaml"))).unwrap()).unwrap();
    for model in value["models"].as_object_mut().unwrap().values_mut() {
        model["parameters"]["model_path"] =
            json!(root.join(model["parameters"]["model_path"].as_str().unwrap()));
    }
    value
}

fn bundle(value: &Value, directory: &tempfile::TempDir) -> Bundle {
    let path = directory.path().join("bundle.yaml");
    fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    Bundle::open(path).unwrap()
}

fn initial(value: &mut Value, directory: &tempfile::TempDir, port: &str, data: &[f32]) {
    let bytes: Vec<_> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tensor =
        safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![data.len()], &bytes)
            .unwrap();
    let path = directory.path().join("initial.safetensors");
    safetensors::serialize_to_file([(port, tensor)], None, &path).unwrap();
    value["pipeline"]["initial_values"] = json!(path);
}

#[test]
fn onnx_ports_bind_by_name_for_all_declaration_orders() {
    for device in devices() {
        for name in ["named_ports", "named_ports_dynamic"] {
            for (reverse_inputs, reverse_outputs) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                let directory = tempfile::tempdir().unwrap();
                let mut data = fixture(name);
                for (ports, reverse) in [("inputs", reverse_inputs), ("outputs", reverse_outputs)] {
                    if reverse {
                        data["models"]["arithmetic"][ports]
                            .as_array_mut()
                            .unwrap()
                            .reverse();
                    }
                }
                let mut session =
                    Session::open(&bundle(&data, &directory), options(device)).unwrap();
                let mut retained = Vec::new();
                for (step, dynamic_length) in [4, 2, 7].into_iter().enumerate() {
                    let length = if name == "named_ports" {
                        4
                    } else {
                        dynamic_length
                    };
                    let left: Vec<f32> =
                        (0..length).map(|i| i as f32 + 7.0 + step as f32).collect();
                    let right = vec![2.0 + step as f32 * 3.0; length];
                    let shape = [length as i64];
                    let result = session
                        .run(&std::collections::HashMap::from([
                            (
                                "arithmetic/right",
                                BorrowedTensor::cpu(&right, &shape).unwrap(),
                            ),
                            (
                                "arithmetic/left",
                                BorrowedTensor::cpu(&left, &shape).unwrap(),
                            ),
                        ]))
                        .unwrap();
                    let expected = [
                        left.iter()
                            .zip(&right)
                            .map(|(a, b)| a - b)
                            .collect::<Vec<_>>(),
                        left.iter()
                            .zip(&right)
                            .map(|(a, b)| a + b)
                            .collect::<Vec<_>>(),
                    ];
                    for (port, expected) in ["difference", "sum"].into_iter().zip(&expected) {
                        assert_eq!(values(&result, &format!("arithmetic/{port}")), *expected,
                            "{device:?} {name}: inputs reversed={reverse_inputs}, outputs reversed={reverse_outputs}");
                    }
                    retained.push((result, expected));
                }
                session.close();
                for (result, expected) in retained {
                    assert_eq!(values(&result, "arithmetic/difference"), expected[0]);
                    assert_eq!(values(&result, "arithmetic/sum"), expected[1]);
                }
            }
        }
    }
}

#[test]
fn public_names_results_and_lifetimes() {
    for device in devices() {
        let runtime = open("policy", device);
        assert_eq!(runtime.info().pipeline_outputs()[0].name(), "policy/action");
        let mut session = runtime.new_session().unwrap();
        drop(runtime);
        let result = session
            .run(&std::collections::HashMap::from([
                (
                    "policy/bias",
                    BorrowedTensor::cpu(&[10f32, 20., 30., 40.], &[4]).unwrap(),
                ),
                (
                    "preprocess/observation",
                    BorrowedTensor::cpu(&[1f32, 2., 3., 4.], &[4]).unwrap(),
                ),
            ]))
            .unwrap();
        assert_eq!(
            result.iter().map(|(name, _)| name).collect::<Vec<_>>(),
            ["policy/action"]
        );
        assert!(result.get("policy/echo").is_none());
        drop(session);
        let tensor = result.get("policy/action").unwrap().clone();
        drop(result);
        assert_eq!(tensor.to_cpu_vec::<f32>().unwrap(), [14., 29., 46., 65.]);
    }
}

#[test]
fn validation_failures_leave_session_usable() {
    let mut session = open("dynamic", ExecutionDevice::Cpu).new_session().unwrap();
    let view = BorrowedTensor::cpu(&[1f32], &[1]).unwrap();
    for inputs in [vec![], vec![("bad", view)]] {
        let inputs = inputs.into_iter().collect();
        let error = session.run(&inputs).err().unwrap();
        assert_eq!(error.code(), ErrorCode::InvalidArgument);
        assert_eq!(error.phase(), "execute");
    }
    let mut options = options(ExecutionDevice::Cpu);
    options
        .output_placement
        .overrides
        .insert("bad".into(), MemoryLocation::Cpu);
    assert!(Session::open(session.info(), options).is_err());
    assert!(session
        .run(&std::collections::HashMap::from([(
            "dynamic/input",
            BorrowedTensor::cpu(&[1i32], &[1]).unwrap()
        )]))
        .is_err());
    let result = session
        .run(&std::collections::HashMap::from([("dynamic/input", view)]))
        .unwrap();
    let reused = session
        .run(&std::collections::HashMap::from([(
            "dynamic/input",
            result.get("dynamic/output").unwrap().view(),
        )]))
        .unwrap();
    assert_eq!(values(&reused, "dynamic/output"), [1.]);
    let empty = session
        .run(&std::collections::HashMap::from([(
            "dynamic/input",
            BorrowedTensor::cpu::<f32>(&[], &[0]).unwrap(),
        )]))
        .unwrap();
    assert_eq!(empty.get("dynamic/output").unwrap().shape(), [0]);
}

#[test]
fn output_order_duplicates_and_input_output_name_collision() {
    let directory = tempfile::tempdir().unwrap();
    let mut data = fixture("policy");
    data["pipeline"]["outputs"]["policy"] = json!(["echo", "action", "echo"]);
    data["pipeline"]["inputs"]["policy"] = json!(["bias", "bias"]);
    let mut session = Session::open(&bundle(&data, &directory), options(ExecutionDevice::Cpu))
        .unwrap()
        .new_session()
        .unwrap();
    let result = session
        .run(&std::collections::HashMap::from([
            (
                "policy/bias",
                BorrowedTensor::cpu(&[0f32; 4], &[4]).unwrap(),
            ),
            (
                "preprocess/observation",
                BorrowedTensor::cpu(&[1f32; 4], &[4]).unwrap(),
            ),
        ]))
        .unwrap();
    assert_eq!(
        result.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        ["policy/echo", "policy/action"]
    );
    // The ONNX artifact permits one tensor to be both graph input and output.
    let mut data = fixture("dynamic");
    data["models"]["dynamic"]["outputs"][0]["name"] = json!("input");
    data["pipeline"]["outputs"]["dynamic"] = json!(["input"]);
    // Use a dedicated identity artifact with that interface generated by the fixture script.
    data["models"]["dynamic"]["parameters"]["model_path"] =
        json!(fixtures("triton").join("passthrough/model.onnx"));
    let mut session = Session::open(&bundle(&data, &directory), options(ExecutionDevice::Cpu))
        .unwrap()
        .new_session()
        .unwrap();
    assert_eq!(
        values(
            &session
                .run(&std::collections::HashMap::from([(
                    "dynamic/input",
                    BorrowedTensor::cpu(&[5f32], &[1]).unwrap()
                )]))
                .unwrap(),
            "dynamic/input"
        ),
        [5.]
    );
}

#[test]
fn feedback_advances_per_session_reset_and_failures_are_atomic() {
    for device in devices() {
        let directory = tempfile::tempdir().unwrap();
        let mut data = fixture("feedback");
        initial(&mut data, &directory, "accumulator/state", &[10.; 4]);
        // A second branch can fail during inference, after feedback computation starts.
        let reshape = fixture("reshape");
        data["models"]["reshape"] = reshape["models"]["reshape"].clone();
        data["pipeline"]["inputs"]["reshape"] = json!(["input"]);
        data["pipeline"]["outputs"]["reshape"] = json!(["output"]);
        let runtime = Session::open(&bundle(&data, &directory), options(device)).unwrap();
        fs::remove_file(directory.path().join("initial.safetensors")).unwrap();
        let mut session = runtime.new_session().unwrap();
        let run = |session: &mut Session, reshape: &[f32]| {
            session.run(&std::collections::HashMap::from([
                (
                    "accumulator/observation",
                    BorrowedTensor::cpu(&[1f32; 4], &[4]).unwrap(),
                ),
                (
                    "reshape/input",
                    BorrowedTensor::cpu(reshape, &[reshape.len() as i64]).unwrap(),
                ),
            ]))
        };
        let first = run(&mut session, &[1., 2.]).unwrap();
        assert_eq!(values(&first, "accumulator/next"), [11.; 4]);
        let lease = first.get("accumulator/next").unwrap().export(true);
        let descriptor = lease.descriptor().unwrap();
        // SAFETY: default outputs are CPU float32, with no concurrent readers.
        unsafe {
            std::slice::from_raw_parts_mut(descriptor.data.cast::<f32>(), 4).fill(99.);
        }
        assert!(run(&mut session, &[1., 2., 3.]).is_err());
        assert_eq!(
            values(&run(&mut session, &[1., 2.]).unwrap(), "accumulator/next"),
            [12.; 4]
        );
        session.reset().unwrap();
        assert_eq!(
            values(&run(&mut session, &[1., 2.]).unwrap(), "accumulator/next"),
            [11.; 4]
        );
        drop(first);
        assert_eq!(unsafe { *descriptor.data.cast::<f32>() }, 99.);
        let mut other = runtime.new_session().unwrap();
        assert_eq!(
            values(&run(&mut other, &[1., 2.]).unwrap(), "accumulator/next"),
            [11.; 4]
        );
    }
}

#[test]
fn feedback_zero_initialization_and_concurrent_sessions() {
    let runtime = open("feedback", ExecutionDevice::Cpu);
    let threads: Vec<_> = (1..=3)
        .map(|i| {
            let mut session = runtime.new_session().unwrap();
            thread::spawn(move || {
                for iteration in 1..=4 {
                    assert_eq!(
                        values(&step(&mut session, i as f32), "accumulator/next"),
                        [(i * iteration) as f32; 4]
                    );
                }
            })
        })
        .collect();
    drop(runtime);
    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn hidden_feedback_fanout_and_default_inputs() {
    for device in devices() {
        let directory = tempfile::tempdir().unwrap();
        let mut data = fixture("feedback");
        data["models"]["follower"] = data["models"]["accumulator"].clone();
        data["pipeline"]["feedback_flow"]["accumulator/next"] =
            json!(["accumulator/state", "follower/state"]);
        data["pipeline"]["inputs"]["follower"] = json!(["observation"]);
        data["pipeline"]["outputs"] = json!({"follower":["next"]});
        let mut session = Session::open(&bundle(&data, &directory), options(device))
            .unwrap()
            .new_session()
            .unwrap();
        for expected in [10., 11., 12.] {
            let result = session
                .run(&std::collections::HashMap::from([
                    (
                        "accumulator/observation",
                        BorrowedTensor::cpu(&[1f32; 4], &[4]).unwrap(),
                    ),
                    (
                        "follower/observation",
                        BorrowedTensor::cpu(&[10f32; 4], &[4]).unwrap(),
                    ),
                ]))
                .unwrap();
            assert_eq!(result.iter().count(), 1);
            assert_eq!(values(&result, "follower/next"), [expected; 4]);
        }
        for seeded in [false, true] {
            let mut data = fixture("defaults");
            if seeded {
                initial(&mut data, &directory, "policy/bias", &[10.; 4]);
            }
            let mut session = Session::open(&bundle(&data, &directory), options(device))
                .unwrap()
                .new_session()
                .unwrap();
            let result = session
                .run(&std::collections::HashMap::from([(
                    "preprocess/observation",
                    BorrowedTensor::cpu(&[1f32; 4], &[4]).unwrap(),
                )]))
                .unwrap();
            let bias = if seeded { 10. } else { 0. };
            assert_eq!(
                values(&result, "policy/action"),
                [4. + bias, 6. + bias, 8. + bias, 10. + bias]
            );
        }
    }
}

#[test]
fn dynamic_state_requires_initial_shape() {
    let directory = tempfile::tempdir().unwrap();
    let mut data = fixture("feedback");
    for direction in ["inputs", "outputs"] {
        for port in data["models"]["accumulator"][direction]
            .as_array_mut()
            .unwrap()
        {
            port["shape"] = json!([-1]);
        }
    }
    data["models"]["accumulator"]["parameters"]["model_path"] =
        json!(fixtures("triton").join("dynamic_accumulator/model.onnx"));
    let error = Session::open(&bundle(&data, &directory), options(ExecutionDevice::Cpu))
        .err()
        .unwrap();
    assert_eq!(error.phase(), "state");
    assert!(error
        .to_string()
        .contains("requires an exported initial value"));
    initial(&mut data, &directory, "accumulator/state", &[10.; 4]);
    let mut session = Session::open(&bundle(&data, &directory), options(ExecutionDevice::Cpu))
        .unwrap()
        .new_session()
        .unwrap();
    assert_eq!(
        values(&step(&mut session, 1.), "accumulator/next"),
        [11.; 4]
    );
}

#[test]
fn failed_backend_load_reports_an_error() {
    const CHILD: &str = "LEAPP_TEST_FAILED_ONNX_LOAD";

    // The supplied ONNX backend can leave its singleton unusable after rejecting
    // a model's shape. Isolate the failure so other tests can still load models.
    if env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(env::current_exe().unwrap())
            .args([
                "--exact",
                "engines::tests::execution::failed_backend_load_reports_an_error",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let mut data = fixture("policy");
    data["models"]["preprocess"]["inputs"][0]["shape"] = json!([3]);
    let error = Session::open(&bundle(&data, &directory), options(ExecutionDevice::Cpu))
        .err()
        .unwrap();
    assert_eq!(error.phase(), "load");
    let expected = match support::engine() {
        lrt::EngineKind::Triton => "failed to load",
        lrt::EngineKind::Native => "incompatible dtype or shape",
    };
    assert!(error.to_string().contains(expected));
}

#[test]
#[ignore = "requires the installed NativeRT backend and PT2 fixtures"]
fn public_pt2_execution() {
    for device in devices() {
        let suffix = match device {
            ExecutionDevice::Cpu => "cpu",
            _ => "cuda",
        };
        let bundle = Bundle::open(fixtures("pt2").join(format!("weighted_{suffix}.yaml"))).unwrap();
        let runtime = Session::open(
            &bundle,
            SessionOptions {
                engine: lrt::EngineKind::Triton,
                ..options(device)
            },
        )
        .unwrap();
        let mut session = runtime.new_session().unwrap();
        let name = bundle.pipeline_inputs()[0].name();
        let result = session
            .run(&std::collections::HashMap::from([(
                name.as_str(),
                BorrowedTensor::cpu(&[1f32, 2., 3., 4.], &[4]).unwrap(),
            )]))
            .unwrap();
        assert_eq!(values(&result, "model/output_0"), [12., 26., 42., 60.]);
    }
}
