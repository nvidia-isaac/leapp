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
    unsafe {
        SharedTensor::new(ManagedTensor::from_versioned(pointer).unwrap(), None, None).unwrap()
    }
}

impl Engine for CpuEngine {
    fn run(
        &self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        _: &OutputPlacement,
        _: &Arc<OutputPool>,
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
        Ok([
            "a_policy/next_state",
            "a_policy/actions",
            "z_observation/features",
        ]
        .map(|name| (name.into(), next.clone()))
        .into())
    }
    fn copy(
        &self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
        _: Option<&OutputPool>,
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
        Session::initialize(&bundle, SessionOptions::default(), &graph, engine.clone())
            .unwrap();
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
}

#[test]
fn peer_caches_inherit_policy_and_close_independently() {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("model.bin"), b"abc").unwrap();
    let path = directory.path().join("pipeline.yaml");
    std::fs::write(&path, include_str!("../fixtures/pipeline.yaml")).unwrap();
    let bundle = Bundle::open(path).unwrap();
    let graph = graph::Graph::new(&bundle).unwrap();
    for max_entries in [0, 1] {
        let options = SessionOptions {
            output_pool: OutputPoolOptions { max_entries },
            ..Default::default()
        };
        let mut session =
            Session::initialize(&bundle, options, &graph, Arc::new(CpuEngine::default()))
                .unwrap();
        let source = session
            .output_pool
            .allocate(12, MemoryLocation::Cpu)
            .unwrap();
        assert_eq!(Arc::strong_count(&source), 1 + max_entries);
        let mut peer = session.new_session().unwrap();
        let retained = peer.output_pool.allocate(12, MemoryLocation::Cpu).unwrap();
        assert_eq!(Arc::strong_count(&retained), 1 + max_entries);
        assert_ne!(source.ptr, retained.ptr);
        session.close();
        assert_eq!(Arc::strong_count(&source), 1);
        assert_eq!(Arc::strong_count(&retained), 1 + max_entries);
        assert!(peer.is_open());
        peer.close();
        assert_eq!(Arc::strong_count(&retained), 1);
    }
}
