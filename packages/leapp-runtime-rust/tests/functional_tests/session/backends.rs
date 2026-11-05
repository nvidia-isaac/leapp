// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    dlpack::BorrowedTensor, Bundle, EngineKind, ErrorCode, ExecutionDevice, Session, SessionOptions,
};
use serde_json::{json, Value};
use std::{collections::HashMap, env, fs, os::unix::fs::symlink, path::Path, process::Command};

const CASE: &str = "LEAPP_TEST_BACKEND_LOAD_CASE";
const BACKENDS: &[(&str, &str, &str)] = &[
    (
        "onnx",
        "onnxruntime/libonnxruntime.so",
        "onnxruntime/libtriton_onnxruntime.so",
    ),
    (
        "jit",
        "torchscript/libleapp_torchscript.so",
        "leapp_torchscript/libtriton_leapp_torchscript.so",
    ),
    (
        "pt2",
        "nativert/libleapp_nativert_backend.so",
        "leapp_pt2/libtriton_leapp_pt2.so",
    ),
    (
        "trt",
        "tensorrt/libleapp_tensorrt.so",
        "tensorrt/libtriton_tensorrt.so",
    ),
];

fn fixture(root: &Path, backend: &str) -> std::path::PathBuf {
    let fixtures = env::var_os("LEAPP_TRITON_FIXTURES")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("target/triton-fixtures"));
    let mut data: Value =
        serde_json::from_slice(&fs::read(fixtures.join("feedback.yaml")).unwrap()).unwrap();
    let parameters = &mut data["models"]["accumulator"]["parameters"];
    parameters["model_path"] = json!(fixtures.join(parameters["model_path"].as_str().unwrap()));
    parameters["backend"] = json!(backend);
    let path = root.join("bundle.yaml");
    fs::write(&path, serde_json::to_vec(&data).unwrap()).unwrap();
    path
}

fn child(test: &str, case: &str, root: &Path, bundle: &Path) -> bool {
    // ORT and Triton retain process-global state. Isolate failures and loader
    // overrides without mutating the parent test process's environment.
    let output = Command::new(env::current_exe().unwrap())
        .args(["--exact", test, "--test-threads=1", "--nocapture"])
        .env(CASE, case)
        .env("LEAPP_TEST_BACKEND_BUNDLE", bundle)
        .env("LEAPP_NATIVE_BACKENDS", root.join("native"))
        .env("LEAPP_TRITON_BACKENDS", root.join("triton"))
        .env(
            "ORT_DYLIB_PATH",
            root.join("native/onnxruntime/libonnxruntime.so"),
        )
        .output()
        .unwrap();
    if !output.status.success() {
        eprintln!(
            "{case}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output.status.success()
}

fn options(engine: &str, backend: &str) -> SessionOptions {
    SessionOptions {
        engine: if engine == "native" {
            EngineKind::Native
        } else {
            EngineKind::Triton
        },
        device: if backend == "trt" {
            ExecutionDevice::Cuda { device_id: 0 }
        } else {
            ExecutionDevice::Cpu
        },
        ..Default::default()
    }
}

fn shared_library(source: &Path, destination: &Path, extra: &[&str]) {
    let output = Command::new("cc")
        .args(["-shared", "-fPIC", "-Wl,--no-undefined"])
        .arg(source)
        .args(extra)
        .arg("-o")
        .arg(destination)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn broken_library(path: &Path, fault: &str) {
    let directory = path.parent().unwrap();
    fs::create_dir_all(directory).unwrap();
    let source = directory.join("stub.c");
    match fault {
        "missing" => {}
        "invalid" => fs::write(path, b"not a shared library").unwrap(),
        "dependency" => {
            let dependency = directory.join("libleapp_test_dependency.so");
            fs::write(&source, "void leapp_test_dependency(void) {}\n").unwrap();
            shared_library(
                &source,
                &dependency,
                &["-Wl,-soname,libleapp_test_dependency.so"],
            );
            fs::write(&source, "extern void leapp_test_dependency(void); void stub(void) { leapp_test_dependency(); }\n").unwrap();
            shared_library(
                &source,
                path,
                &[
                    "-L",
                    directory.to_str().unwrap(),
                    "-Wl,--no-as-needed",
                    "-lleapp_test_dependency",
                    "-Wl,-rpath,$ORIGIN",
                ],
            );
            fs::remove_file(dependency).unwrap();
        }
        _ => unreachable!(),
    }
}

#[test]
fn backend_load_errors_identify_the_node_and_library() {
    if let Ok(case) = env::var(CASE) {
        let parts: Vec<_> = case.split('/').collect();
        let (engine, backend, fault) = (parts[0], parts[1], parts[2]);
        // Inspection accepts the model's opaque artifact without loading any
        // backend, including the broken libraries installed by the parent.
        let bundle = Bundle::open(env::var_os("LEAPP_TEST_BACKEND_BUNDLE").unwrap()).unwrap();
        assert_eq!(bundle.model_names(), ["accumulator"]);
        let error = Session::open(&bundle, options(engine, backend))
            .err()
            .expect("backend must fail loading");
        assert_eq!(error.phase(), "load");
        let (_, native, triton) = BACKENDS
            .iter()
            .find(|(name, _, _)| *name == backend)
            .unwrap();
        let (variable, library) = if engine == "native" {
            ("LEAPP_NATIVE_BACKENDS", native)
        } else {
            ("LEAPP_TRITON_BACKENDS", triton)
        };
        let path = Path::new(&env::var_os(variable).unwrap()).join(library);
        let message = error.to_string();
        assert!(message.contains("accumulator"), "{error}");
        assert!(message.contains(path.to_str().unwrap()), "{error}");
        if fault == "dependency" {
            assert!(message.contains("libleapp_test_dependency.so"), "{error}");
        } else if fault == "invalid" {
            assert!(message.contains("file too short"), "{error}");
        }
        if engine == "native" {
            let expected = if backend == "onnx" && fault == "missing" {
                ErrorCode::NotFound
            } else {
                ErrorCode::Unsupported
            };
            assert_eq!(error.code(), expected, "{error}");
        }
        return;
    }
    let mut failed = Vec::new();
    for engine in ["native", "triton"] {
        for &(backend, native, triton) in BACKENDS {
            if backend == "trt" && env::var("LEAPP_TEST_CUDA").as_deref() != Ok("1") {
                continue;
            }
            for fault in ["missing", "invalid", "dependency"] {
                let root = tempfile::tempdir().unwrap();
                let (directory, library) = if engine == "native" {
                    ("native", native)
                } else {
                    ("triton", triton)
                };
                broken_library(&root.path().join(directory).join(library), fault);
                let case = format!("{engine}/{backend}/{fault}");
                if !child(
                    "engines::tests::backends::backend_load_errors_identify_the_node_and_library",
                    &case,
                    root.path(),
                    &fixture(root.path(), backend),
                ) {
                    failed.push(case);
                }
            }
        }
    }
    assert!(
        failed.is_empty(),
        "failed backend loading cases: {failed:?}"
    );
}

#[test]
fn unrelated_backends_do_not_affect_onnx_execution() {
    if let Ok(case) = env::var(CASE) {
        let (engine, _) = case.split_once('/').unwrap();
        let bundle = Bundle::open(env::var_os("LEAPP_TEST_BACKEND_BUNDLE").unwrap()).unwrap();
        let mut devices = vec![ExecutionDevice::Cpu];
        if env::var("LEAPP_TEST_CUDA").as_deref() == Ok("1") {
            devices.push(ExecutionDevice::Cuda { device_id: 0 });
        }
        for device in devices {
            let mut session = Session::open(
                &bundle,
                SessionOptions {
                    device,
                    ..options(engine, "onnx")
                },
            )
            .unwrap();
            let mut peer = session.new_session().unwrap();
            let values = [1_f32; 4];
            let inputs = HashMap::from([(
                "accumulator/observation",
                BorrowedTensor::cpu(&values, &[4]).unwrap(),
            )]);
            let retained = session.run(&inputs).unwrap();
            drop(session);
            assert_eq!(
                peer.run(&inputs)
                    .unwrap()
                    .get("accumulator/next")
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap(),
                [1.; 4]
            );
            drop(peer);
            assert_eq!(
                retained
                    .get("accumulator/next")
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap(),
                [1.; 4]
            );
        }
        return;
    }
    for (engine, fault) in [
        ("native", "missing"),
        ("native", "invalid"),
        ("triton", "missing"),
        ("triton", "invalid"),
    ] {
        let root = tempfile::tempdir().unwrap();
        for &(_, native, triton) in BACKENDS {
            for path in [
                root.path().join("native").join(native),
                root.path().join("triton").join(triton),
            ] {
                broken_library(&path, fault);
            }
        }
        if engine == "native" {
            let library = env::var_os("ORT_DYLIB_PATH")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| {
                    Path::new(&env::var_os("LEAPP_NATIVE_BACKENDS").unwrap())
                        .join("onnxruntime/libonnxruntime.so")
                });
            let target = root.path().join("native/onnxruntime/libonnxruntime.so");
            if fault == "invalid" {
                fs::remove_file(&target).unwrap();
            }
            symlink(library, target).unwrap();
        } else {
            let library =
                Path::new(&env::var_os("LEAPP_TRITON_BACKENDS").unwrap()).join("onnxruntime");
            let target = root.path().join("triton/onnxruntime");
            if fault == "invalid" {
                fs::remove_file(target.join("libtriton_onnxruntime.so")).unwrap();
            }
            fs::remove_dir(&target).unwrap();
            symlink(library, target).unwrap();
        }
        assert!(child(
            "engines::tests::backends::unrelated_backends_do_not_affect_onnx_execution",
            &format!("{engine}/{fault}"),
            root.path(),
            &fixture(root.path(), "onnx"),
        ));
    }
}
