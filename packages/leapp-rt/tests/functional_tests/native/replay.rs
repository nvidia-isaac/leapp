// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

const CUDA: MemoryLocation = MemoryLocation::Cuda { device_id: 0 };

fn slot(transfers: &mut Transfers, value: f32) -> plan::Slot {
    let mut slot = plan::Slot::new(
        Arc::new(plan::TensorSpec {
            name: "test".into(),
            dtype: crate::DataType::Float32,
            shape: vec![4].into(),
        }),
        false,
        CUDA,
    )
    .unwrap();
    unsafe {
        slot.assign_validated(
            BorrowedTensor::cpu(&[value; 4], &[4]).unwrap(),
            16,
            transfers,
        )
        .unwrap()
    };
    slot
}

fn values(slot: &plan::Slot, transfers: &mut Transfers) -> Vec<f32> {
    slot.snapshot(MemoryLocation::Cpu, transfers)
        .unwrap()
        .to_cpu_vec()
        .unwrap()
}

#[test]
#[ignore = "requires ONNX Runtime TensorRT provider and GPU"]
fn onnx_tensorrt_replay_uses_live_values_and_fixed_feedback() {
    replay_test(true);
}

#[test]
#[ignore = "requires ONNX Runtime CUDA provider and GPU"]
fn onnx_cuda_replay_uses_live_values_and_fixed_feedback() {
    replay_test(false);
}

fn replay_test(tensorrt: bool) {
    let owner = crate::engines::memory::cuda::HandoffStream::new(0).unwrap();
    let stream = unsafe { crate::CudaStream::from_raw(owner.address() as *mut _) };
    for stream in [None, Some(stream)] {
        exercise_replay(stream, tensorrt);
    }
}

fn exercise_replay(stream: Option<crate::CudaStream>, tensorrt: bool) {
    // Cover the shared selector and the existing TensorRT-specific option.
    let shared_selector = !tensorrt || stream.is_some();
    let options = |names: &[&str], replay: bool| SessionOptions {
        device: ExecutionDevice::Cuda { device_id: 0 },
        onnx_tensorrt: names
            .iter()
            .filter(|_| tensorrt)
            .map(|name| {
                (
                    (*name).into(),
                    HashMap::from([
                        (
                            "trt_cuda_graph_enable".into(),
                            if replay && !shared_selector { "1" } else { "0" }.into(),
                        ),
                        ("trt_builder_optimization_level".into(), "0".into()),
                    ]),
                )
            })
            .collect(),
        cuda_graph_replay: names
            .iter()
            .filter(|_| replay && shared_selector)
            .map(|name| (*name).to_owned())
            .collect(),
        cuda_streams: stream.map(|stream| (0, stream)).into_iter().collect(),
        ..Default::default()
    };
    let mut transfers = Transfers::new(CUDA, &SessionOptions::default()).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let bundle = write_bundle(&policy_copy(directory.path()), directory.path());
    // ORT owns capture eligibility. Excluding this node's only operation forces
    // fallback, which must reject replay without a LEAPP operator allow-list.
    if tensorrt {
        let mut unsupported = options(&["preprocess"], true);
        unsupported
            .onnx_tensorrt
            .get_mut("preprocess")
            .unwrap()
            .insert("trt_op_types_to_exclude".into(), "Add".into());
        let error = Session::open(&bundle, unsupported).err().unwrap();
        assert!(
            error
                .to_string()
                .contains("all the graph nodes have not been assigned"),
            "{error}"
        );
    }
    let mut normal = Session::open(&bundle, options(&["preprocess", "policy"], false)).unwrap();
    let mut replay = Session::open(&bundle, options(&["preprocess", "policy"], true)).unwrap();
    assert!(normal.new_session().is_ok());
    assert!(replay
        .new_session()
        .err()
        .unwrap()
        .to_string()
        .contains("new_session() is unavailable when ONNX CUDA replay is enabled"));
    let bias = slot(&mut transfers, 1.);
    let mut storage = [1., 3.].map(|base| (slot(&mut transfers, base), slot(&mut transfers, 99.)));
    for ((session, base), (input, output)) in [(&mut normal, 1.), (&mut replay, 3.)]
        .into_iter()
        .zip(&mut storage)
    {
        for step in 0..20 {
            let value = base + step as f32;
            unsafe {
                input
                    .assign_validated(
                        BorrowedTensor::cpu(&[value; 4], &[4]).unwrap(),
                        16,
                        &mut transfers,
                    )
                    .unwrap()
            };
            let destination = *output.view().unwrap().descriptor();
            let result = session
                .run_into(
                    &HashMap::from([
                        ("preprocess/observation", input.view().unwrap()),
                        ("policy/bias", bias.view().unwrap()),
                    ]),
                    &mut HashMap::from([("policy/action", unsafe {
                        OutputBuffer::new(&destination).unwrap()
                    })]),
                )
                .unwrap();
            assert!(result.iter().next().is_none());
            assert_eq!(
                values(output, &mut transfers),
                [2., 3., 4., 5.].map(|scale| (value + 1.) * scale + 1.)
            );
            // A caller may overwrite destinations between replays.
            unsafe {
                output
                    .assign_validated(
                        BorrowedTensor::cpu(&[99_f32; 4], &[4]).unwrap(),
                        16,
                        &mut transfers,
                    )
                    .unwrap()
            };
        }
    }
    normal.close();
    replay.close();

    // Keep next-state private and route it through a second node to a public
    // output. Capture must use a fixed internal output, not a fresh snapshot.
    let directory = tempfile::tempdir().unwrap();
    let mut data = fixture("feedback");
    data["models"]["readout"] = data["models"]["accumulator"].clone();
    data["pipeline"]["data_flow"] = json!({"accumulator/next": ["readout/observation"]});
    data["pipeline"]["outputs"] = json!({"readout": ["next"]});
    let bundle = write_bundle(&data, directory.path());
    let mut session = Session::open(&bundle, options(&["accumulator", "readout"], true)).unwrap();
    let input = slot(&mut transfers, 1.);
    let output = slot(&mut transfers, 0.);
    let other = slot(&mut transfers, 99.);
    let inputs = HashMap::from([("accumulator/observation", input.view().unwrap())]);
    assert!(session
        .run(&inputs)
        .err()
        .unwrap()
        .to_string()
        .contains("run_into"));
    let destination = *output.view().unwrap().descriptor();
    for expected in [1., 2., 3., 4.] {
        session
            .run_into(
                &inputs,
                &mut HashMap::from([("readout/next", unsafe {
                    OutputBuffer::new(&destination).unwrap()
                })]),
            )
            .unwrap();
        assert_eq!(values(&output, &mut transfers), [expected; 4]);
        let changed = *other.view().unwrap().descriptor();
        assert!(session
            .run_into(
                &inputs,
                &mut HashMap::from([("readout/next", unsafe {
                    OutputBuffer::new(&changed).unwrap()
                })])
            )
            .err()
            .unwrap()
            .to_string()
            .contains("addresses changed"));
    }
    session.reset().unwrap();
    session
        .run_into(
            &inputs,
            &mut HashMap::from([("readout/next", unsafe {
                OutputBuffer::new(&destination).unwrap()
            })]),
        )
        .unwrap();
    assert_eq!(values(&output, &mut transfers), [1.; 4]);
    session.close();
}
