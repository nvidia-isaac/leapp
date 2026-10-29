// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::{engines::memory::cuda::HandoffStream, CudaStream, RunResult};

fn values(result: &RunResult) -> Vec<(String, Vec<f32>)> {
    result
        .iter()
        .map(|(name, tensor)| (name.to_owned(), tensor.to_cpu_vec().unwrap()))
        .collect()
}

fn exercise(bundle: &Bundle, device: ExecutionDevice, inputs: &HashMap<&str, BorrowedTensor<'_>>) {
    // The external owner outlives both the original session and all peers.
    let owner = HandoffStream::new(0).unwrap();
    let stream = unsafe { CudaStream::from_raw(owner.address() as *mut _) };
    eprintln!("external CUDA stream: {:#x}", owner.address());
    let options = SessionOptions {
        engine: crate::EngineKind::Native,
        device,
        ..Default::default()
    };
    let expected = {
        let mut session = Session::open(bundle, options.clone()).unwrap();
        values(&session.run(inputs).unwrap())
    };
    for location in locations() {
        let mut session = Session::open(
            bundle,
            SessionOptions {
                cuda_streams: HashMap::from([(0, stream)]),
                output_placement: OutputPlacement {
                    default_location: location,
                    ..Default::default()
                },
                ..options.clone()
            },
        )
        .unwrap();
        let retained = session.run(inputs).unwrap();
        assert_eq!(values(&retained), expected);
        session.reset().unwrap();
        assert_eq!(values(&session.run(inputs).unwrap()), expected);
        let peer = session.new_session().unwrap();
        session.close();
        drop(session);
        // Streams are stable handles, including after moving a peer to another
        // host thread. Completion protects the retained outputs on both paths.
        std::thread::scope(|scope| {
            scope
                .spawn(|| {
                    let mut peer = peer;
                    assert_eq!(values(&peer.run(inputs).unwrap()), expected);
                    peer.close();
                })
                .join()
                .unwrap();
        });
        assert_eq!(values(&retained), expected);
        // Closing every user must not destroy the caller's stream.
        owner.synchronize().unwrap();
    }
}

#[test]
fn external_streams_support_onnx_fixed_dynamic_feedback_and_cpu_transfers() {
    for device in devices() {
        for (name, input) in [
            ("feedback", "accumulator/observation"),
            ("dynamic", "dynamic/input"),
        ] {
            let bundle = Bundle::open(fixtures().join(format!("{name}.yaml"))).unwrap();
            let input = HashMap::from([(
                input,
                BorrowedTensor::cpu(&[1_f32, 2., 3., 4.], &[4]).unwrap(),
            )]);
            exercise(&bundle, device, &input);
        }
    }
}

#[test]
fn legacy_stream_and_rejected_thread_local_stream() {
    let bundle = Bundle::open(fixtures().join("feedback.yaml")).unwrap();
    // SAFETY: CUDA's default-stream sentinels are permanent handles.
    let legacy = unsafe { CudaStream::from_raw(std::ptr::null_mut()) };
    let per_thread = unsafe { CudaStream::from_raw(2_usize as *mut _) };
    let options = SessionOptions {
        engine: crate::EngineKind::Native,
        device: ExecutionDevice::Cuda { device_id: 0 },
        cuda_streams: HashMap::from([(0, legacy)]),
        ..Default::default()
    };
    let input = HashMap::from([(
        "accumulator/observation",
        BorrowedTensor::cpu(&[1_f32; 4], &[4]).unwrap(),
    )]);
    let mut session = Session::open(&bundle, options.clone()).unwrap();
    assert_eq!(values(&session.run(&input).unwrap())[0].1, [1.; 4]);
    session.close();
    let options = SessionOptions {
        cuda_streams: HashMap::from([(0, per_thread)]),
        ..options
    };
    let error = Session::open(&bundle, options.clone()).err().unwrap();
    assert_eq!(error.code(), ErrorCode::InvalidArgument);
    assert!(error.to_string().contains("cudaStreamPerThread"));
    {
        let mut session = Session::open(
            &bundle,
            SessionOptions {
                engine: crate::EngineKind::Triton,
                ..options
            },
        )
        .unwrap();
        assert_eq!(values(&session.run(&input).unwrap())[0].1, [1.; 4]);
    }
}

#[test]
#[ignore = "requires native TorchScript/NativeRT backends and CPU/CUDA fixtures"]
fn external_streams_support_torch_fixed_dynamic_and_worker_paths() {
    for device in devices() {
        for name in ["scripted", "dynamic", "forked"] {
            let bundle = super::torch::bundle("jit", name, device);
            let mut inputs = HashMap::from([(
                "model/values",
                BorrowedTensor::cpu(&[1_f32, 2., 3., 4.], &[4]).unwrap(),
            )]);
            if name == "scripted" {
                let value = inputs.remove("model/values").unwrap();
                inputs.insert("model/z_values", value);
                inputs.insert("model/INPUT__9", value);
            }
            exercise(&bundle, device, &inputs);
        }
        for name in ["weighted", "reshape"] {
            let bundle = super::torch::bundle("pt2", name, device);
            exercise(
                &bundle,
                device,
                &HashMap::from([(
                    "model/values",
                    BorrowedTensor::cpu(&[1_f32, 2., 3., 4.], &[4]).unwrap(),
                )]),
            );
        }
    }
}

#[test]
#[ignore = "requires native TensorRT backend and generated engines"]
fn external_streams_support_tensorrt_fixed_and_dynamic() {
    for name in ["dynamic_fixed", "dynamic"] {
        let bundle = super::tensorrt::bundle(name);
        let data = [1_f32; 8];
        let count = if name == "dynamic_fixed" { 8 } else { 4 };
        exercise(
            &bundle,
            ExecutionDevice::Cuda { device_id: 0 },
            &HashMap::from([(
                "dynamic/input",
                BorrowedTensor::cpu(&data[..count], &[count as i64]).unwrap(),
            )]),
        );
    }
}

#[test]
#[ignore = "requires Warp adapters, CUDA, and generated Warp fixtures"]
fn external_streams_support_warp_gpu_replay() {
    let root = PathBuf::from(std::env::var_os("LEAPP_WARP_FIXTURES").unwrap());
    for name in ["warp_onnx", "warp_pt2_cuda"] {
        let bundle = Bundle::open(root.join(format!("{name}.yaml"))).unwrap();
        exercise(
            &bundle,
            ExecutionDevice::Cuda { device_id: 0 },
            &HashMap::from([
                (
                    "model/x",
                    BorrowedTensor::cpu(&[2_f32; 12], &[4, 3]).unwrap(),
                ),
                (
                    "model/gain",
                    BorrowedTensor::cpu(&[1_i32, 2, 3, 4], &[4]).unwrap(),
                ),
            ]),
        );
    }
}
