// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::DataType;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Weak,
};

// A minimal dynamic-only backend: the defaults provide guarded execution and
// unsupported prepared calls. No SDK is needed to exercise trait dispatch.
struct CopyPlugin(Arc<()>);

impl BackendPlugin for CopyPlugin {
    fn new_session(&self) -> anyhow::Result<Box<dyn BackendPlugin>> {
        Ok(Box::new(Self(self.0.clone())))
    }

    fn run_dynamic(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        transfers: &mut Transfers,
    ) -> anyhow::Result<Vec<SharedTensor>> {
        Ok(vec![transfers.copy(inputs[0], MemoryLocation::Cpu)?])
    }

    fn loaded_identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

fn plan(nodes: usize) -> Arc<Plan> {
    let spec = || plan::TensorSpec {
        name: "value".into(),
        dtype: DataType::Float32,
        shape: vec![1],
    };
    Arc::new(Plan {
        calls: (0..nodes)
            .map(|i| plan::Call {
                name: format!("node{i}"),
                model: i,
                inputs: vec![i],
                input_specs: vec![spec()],
                outputs: vec![i + 1],
            })
            .collect(),
        inputs: IndexMap::from([("input".into(), 0)]),
        outputs: IndexMap::from([("output".into(), nodes)]),
        slots: (0..=nodes).map(|_| Arc::new(spec())).collect(),
    })
}

#[test]
fn dynamic_only_trait_defaults_and_peers_preserve_output_ownership() {
    let mut model: Box<dyn BackendPlugin> = Box::new(CopyPlugin(Arc::new(())));
    assert!(model.needs_guard());
    assert!(!model.can_prepare());
    assert!(!model.is_prepared());
    assert!(!model.run_borrowed(&[]).unwrap());
    assert_eq!(
        backends::error("execute", model.run().unwrap_err()).code(),
        crate::ErrorCode::Unsupported
    );
    // SAFETY: empty slices have no storage to retain; this optional path rejects them.
    assert_eq!(
        backends::error("prepare", unsafe { model.prepare(&[], &[]) }.unwrap_err()).code(),
        crate::ErrorCode::Unsupported
    );

    let parent = NativeEngine::with_models(
        plan(1),
        vec![model],
        MemoryLocation::Cpu,
        Some(Arc::new(Mutex::new(()))),
        Default::default(),
    )
    .unwrap();
    let peer = parent.peer().unwrap();
    assert!(Arc::ptr_eq(
        parent.execution_guard.as_ref().unwrap(),
        peer.execution_guard.as_ref().unwrap()
    ));
    {
        let a = parent.workspace.lock().unwrap();
        let b = peer.workspace.lock().unwrap();
        assert_eq!(a.models[0].loaded_identity(), b.models[0].loaded_identity());
        assert!(!b.models[0].is_prepared());
        assert_ne!(
            a.slots[0].view().unwrap().descriptor().data,
            b.slots[0].view().unwrap().descriptor().data
        );
    }
    let mut values = [2f32];
    let first = parent
        .run(
            &[("input", BorrowedTensor::cpu(&values, &[1]).unwrap())],
            &OutputPlacement::default(),
            &Arc::new(OutputPool::new(Default::default(), 0)),
        )
        .unwrap();
    drop(parent);
    values[0] = 7.;
    let second = peer
        .run(
            &[("input", BorrowedTensor::cpu(&values, &[1]).unwrap())],
            &OutputPlacement::default(),
            &Arc::new(OutputPool::new(Default::default(), 0)),
        )
        .unwrap();
    drop(peer);
    assert_eq!(first["output"].to_cpu_vec::<f32>().unwrap(), [2.]);
    assert_eq!(second["output"].to_cpu_vec::<f32>().unwrap(), [7.]);
}

// A prepared backend verifies direct routing, decline, and no retry after an
// execution error without needing an SDK or adding production test hooks.
struct PreparedProbe {
    outcome: Result<bool, &'static str>,
    input: usize,
    output: usize,
    runs: Arc<AtomicUsize>,
}

impl BackendPlugin for PreparedProbe {
    fn new_session(&self) -> anyhow::Result<Box<dyn BackendPlugin>> {
        Ok(Box::new(Self {
            input: 0,
            output: 0,
            runs: self.runs.clone(),
            outcome: self.outcome,
        }))
    }

    fn can_prepare(&self) -> bool {
        true
    }

    fn is_prepared(&self) -> bool {
        self.output != 0
    }

    unsafe fn prepare(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        outputs: &[BorrowedTensor<'_>],
    ) -> anyhow::Result<()> {
        self.input = inputs[0].descriptor().data as usize;
        self.output = outputs[0].descriptor().data as usize;
        Ok(())
    }

    fn run(&mut self) -> anyhow::Result<()> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        // SAFETY: this fixture prepares one f32 input/output, retained by the workspace.
        unsafe {
            *(self.output as *mut f32) = *(self.input as *const f32);
        }
        Ok(())
    }

    fn run_borrowed(&mut self, inputs: &[BorrowedTensor<'_>]) -> anyhow::Result<bool> {
        assert_ne!(inputs[0].descriptor().data as usize, self.input);
        let handled = self.outcome.map_err(anyhow::Error::msg)?;
        if handled {
            let value = inputs[0]
                .as_cpu_slice::<f32>()
                .map_err(anyhow::Error::msg)?[0];
            // SAFETY: the prepared output is exclusively owned under the workspace lock.
            unsafe {
                *(self.output as *mut f32) = value;
            }
        }
        Ok(handled)
    }

    fn run_dynamic(
        &mut self,
        _: &[BorrowedTensor<'_>],
        _: &mut Transfers,
    ) -> anyhow::Result<Vec<SharedTensor>> {
        unreachable!("prepared fixture")
    }

    fn loaded_identity(&self) -> usize {
        Arc::as_ptr(&self.runs) as usize
    }
}

#[test]
fn borrowed_execution_only_falls_back_after_an_explicit_decline() {
    for outcome in [Ok(true), Ok(false), Err("injected execution failure")] {
        let runs = Arc::new(AtomicUsize::new(0));
        let model = PreparedProbe {
            outcome,
            input: 0,
            output: 0,
            runs: runs.clone(),
        };
        let engine = NativeEngine::with_models(
            plan(1),
            vec![Box::new(model)],
            MemoryLocation::Cpu,
            None,
            Default::default(),
        )
        .unwrap();
        let result = engine.run(
            &[("input", BorrowedTensor::cpu(&[7f32], &[1]).unwrap())],
            &OutputPlacement::default(),
            &Arc::new(OutputPool::new(Default::default(), 0)),
        );
        if outcome.is_err() {
            assert!(result
                .err()
                .unwrap()
                .to_string()
                .contains("injected execution failure"));
        } else {
            assert_eq!(result.unwrap()["output"].to_cpu_vec::<f32>().unwrap(), [7.]);
        }
        assert_eq!(
            runs.load(Ordering::SeqCst),
            usize::from(outcome == Ok(false))
        );
    }
}

// Simulate an SDK retaining partial binding resources before prepare fails.
struct PartialBinding {
    node: usize,
    slot: Weak<plan::TensorSpec>,
    events: Arc<Mutex<Vec<String>>>,
    prepared: bool,
}

impl BackendPlugin for PartialBinding {
    fn new_session(&self) -> anyhow::Result<Box<dyn BackendPlugin>> {
        Ok(Box::new(Self {
            node: self.node,
            slot: self.slot.clone(),
            events: self.events.clone(),
            prepared: false,
        }))
    }

    fn can_prepare(&self) -> bool {
        true
    }

    fn is_prepared(&self) -> bool {
        self.prepared
    }

    unsafe fn prepare(
        &mut self,
        _inputs: &[BorrowedTensor<'_>],
        _outputs: &[BorrowedTensor<'_>],
    ) -> anyhow::Result<()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("prepare{}", self.node));
        anyhow::ensure!(self.node == 0, "injected preparation failure");
        self.prepared = true;
        Ok(())
    }

    fn run_dynamic(
        &mut self,
        _inputs: &[BorrowedTensor<'_>],
        _transfers: &mut Transfers,
    ) -> anyhow::Result<Vec<SharedTensor>> {
        anyhow::bail!("failed construction must never reach execution")
    }

    fn loaded_identity(&self) -> usize {
        Arc::as_ptr(&self.events) as usize
    }
}

impl Drop for PartialBinding {
    fn drop(&mut self) {
        // Exactly two owners: the plan specification and the workspace Slot.
        // This observes drop order without dereferencing potentially freed data.
        assert_eq!(self.slot.strong_count(), 2);
        self.events
            .lock()
            .unwrap()
            .push(format!("drop{}", self.node));
    }
}

#[test]
fn failed_preparation_drops_all_trait_objects_before_workspace_slots() {
    let plan = plan(2);
    let events = Arc::new(Mutex::new(Vec::new()));
    let models = (0..2)
        .map(|node| {
            Box::new(PartialBinding {
                node,
                slot: Arc::downgrade(&plan.slots[node + 1]),
                events: events.clone(),
                prepared: false,
            }) as Box<dyn BackendPlugin>
        })
        .collect();
    let result = NativeEngine::with_models(
        plan.clone(),
        models,
        MemoryLocation::Cpu,
        None,
        Default::default(),
    );
    let error = result.err().expect("second preparation must fail");
    assert_eq!(error.phase(), "prepare");
    assert!(error.to_string().contains("injected preparation failure"));
    let events = events.lock().unwrap();
    assert_eq!(events[..2], ["prepare0", "prepare1"]);
    // Only model-before-storage order is contractual; Vec element drop order isn't.
    let mut dropped = events[2..].to_vec();
    dropped.sort();
    assert_eq!(dropped, ["drop0", "drop1"]);
    assert_eq!(Arc::strong_count(&plan.slots[1]), 1);
    assert_eq!(Arc::strong_count(&plan.slots[2]), 1);
}
