// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dlpack::{DLManagedTensorVersioned, DLPackVersion, ManagedTensor};
use std::sync::atomic::{AtomicBool, Ordering};

// A CPU engine lets the test inject an output copy failure.
#[derive(Default)]
struct CpuEngine {
    fail_copy: AtomicBool,
}

fn tensor(values: Vec<f32>) -> SharedTensor {
    let owner = Box::new((values, [3_i64]));
    let dl_tensor = *BorrowedTensor::cpu(&owner.0, &owner.1)
        .unwrap()
        .descriptor();
    unsafe extern "C" fn delete(pointer: *mut DLManagedTensorVersioned) {
        // SAFETY: this deleter receives its unique managed tensor and boxed owner.
        unsafe {
            let tensor = Box::from_raw(pointer);
            drop(Box::from_raw(
                tensor.manager_ctx.cast::<(Vec<f32>, [i64; 1])>(),
            ));
        }
    }
    let pointer = Box::into_raw(Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: Box::into_raw(owner).cast(),
        deleter: Some(delete),
        flags: 0,
        dl_tensor,
    }));
    // SAFETY: the owner retains ready CPU storage and metadata, deletable on any thread.
    unsafe { SharedTensor::new(ManagedTensor::from_versioned(pointer).unwrap(), None).unwrap() }
}

impl Engine for CpuEngine {
    fn new_session(self: Arc<Self>) -> Result<Arc<dyn Engine>, RuntimeError> {
        Ok(self)
    }

    fn run(
        &self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        _: &OutputPlacement,
        buffers: &mut HashMap<&str, OutputBuffer<'_>>,
    ) -> Result<IndexMap<String, SharedTensor>, RuntimeError> {
        let get = |name| {
            inputs
                .iter()
                .find(|(key, _)| *key == name)
                .unwrap()
                .1
                .as_cpu_slice::<f32>()
                .unwrap()
        };
        let next = tensor(
            get("z_observation/joint_pos")
                .iter()
                .zip(get("a_policy/state"))
                .map(|(input, state)| input + state)
                .collect(),
        );
        let mut outputs: IndexMap<String, SharedTensor> = [
            "a_policy/next_state",
            "a_policy/actions",
            "z_observation/features",
        ]
        .map(|name| (name.into(), next.clone()))
        .into();
        for (&name, buffer) in buffers.iter() {
            if let Some(value) = outputs.shift_remove(name) {
                // SAFETY: this fixture's validated f32 buffers are disjoint and live.
                let source = unsafe { value.borrow() }.as_cpu_slice::<f32>().unwrap();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source.as_ptr(),
                        buffer.descriptor().data.cast::<f32>(),
                        source.len(),
                    )
                };
            }
        }
        Ok(outputs)
    }
    fn copy(
        &self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
    ) -> Result<SharedTensor, RuntimeError> {
        assert_eq!(target, MemoryLocation::Cpu);
        if self.fail_copy.load(Ordering::SeqCst) {
            return Err(RuntimeError::new(
                "execute",
                anyhow::anyhow!("injected copy failure"),
            ));
        }
        Ok(tensor(input.as_cpu_slice::<f32>().unwrap().to_vec()))
    }
}

#[test]
fn copy_failure_preserves_feedback() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("model.bin"), b"abc").unwrap();
    let path = directory.path().join("pipeline.yaml");
    std::fs::write(&path, include_str!("../fixtures/pipeline.yaml")).unwrap();
    let bundle = Bundle::open(path).unwrap();
    let graph = graph::Graph::new(&bundle).unwrap();
    let engine = Arc::new(CpuEngine::default());
    let mut session =
        Session::initialize(&bundle, SessionOptions::default(), &graph, engine.clone()).unwrap();
    let step = |session: &mut Session| {
        session
            .run(&std::collections::HashMap::from([(
                "z_observation/joint_pos",
                BorrowedTensor::cpu(&[1_f32; 3], &[3]).unwrap(),
            )]))
            .map(|result| {
                result
                    .get("a_policy/next_state")
                    .unwrap()
                    .to_cpu_vec::<f32>()
                    .unwrap()
            })
    };
    assert_eq!(step(&mut session).unwrap(), [1.; 3]);
    engine.fail_copy.store(true, Ordering::SeqCst);
    assert!(step(&mut session)
        .unwrap_err()
        .to_string()
        .contains("injected copy failure"));
    engine.fail_copy.store(false, Ordering::SeqCst);
    // The failed run must not advance feedback: the next result is 2, not 3.
    assert_eq!(step(&mut session).unwrap(), [2.; 3]);
    let inputs = HashMap::from([(
        "z_observation/joint_pos",
        BorrowedTensor::cpu(&[1_f32; 3], &[3]).unwrap(),
    )]);
    let mut destination = [0_f32; 3];
    engine.fail_copy.store(true, Ordering::SeqCst);
    {
        let mut outputs = HashMap::from([(
            "a_policy/next_state",
            OutputBuffer::cpu(&mut destination, &[3]).unwrap(),
        )]);
        assert!(session.run_into(&inputs, &mut outputs).is_err());
    }
    assert_eq!(destination, [3.; 3]);
    engine.fail_copy.store(false, Ordering::SeqCst);
    assert_eq!(step(&mut session).unwrap(), [3.; 3]);
    {
        let mut outputs = HashMap::from([(
            "a_policy/next_state",
            OutputBuffer::cpu(&mut destination, &[3]).unwrap(),
        )]);
        let result = session.run_into(&inputs, &mut outputs).unwrap();
        assert!(result.get("a_policy/next_state").is_none());
    }
    assert_eq!(destination, [4.; 3]);
    destination.fill(99.);
    assert_eq!(step(&mut session).unwrap(), [5.; 3]);
}

#[test]
fn peers_inherit_placement_and_close_independently() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("model.bin"), b"abc").unwrap();
    let path = directory.path().join("pipeline.yaml");
    std::fs::write(&path, include_str!("../fixtures/pipeline.yaml")).unwrap();
    let bundle = Bundle::open(path).unwrap();
    let graph = graph::Graph::new(&bundle).unwrap();
    let options = SessionOptions {
        output_placement: OutputPlacement {
            default_location: MemoryLocation::PinnedCpu,
            overrides: HashMap::from([(
                "a_policy/next_state".into(),
                MemoryLocation::Cuda { device_id: 0 },
            )]),
        },
        ..Default::default()
    };
    let mut session =
        Session::initialize(&bundle, options, &graph, Arc::new(CpuEngine::default())).unwrap();
    let mut peer = session.new_session().unwrap();
    // Effective engine placement preserves private state independently of
    // public overrides, and peers inherit the immutable routing metadata.
    for current in [&session, &peer] {
        assert_eq!(
            current.engine_placement.location("a_policy/next_state"),
            MemoryLocation::Cpu
        );
        assert_eq!(
            current.engine_placement.location("a_policy/actions"),
            MemoryLocation::PinnedCpu
        );
        assert_eq!(
            current.placement.location("a_policy/next_state"),
            MemoryLocation::Cuda { device_id: 0 }
        );
    }
    session.close();
    assert!(peer.is_open());
    peer.close();
}
