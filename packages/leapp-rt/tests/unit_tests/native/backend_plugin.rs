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
struct CopyPlugin(Arc<usize>);

impl BackendPlugin for CopyPlugin {
    fn new_session(&self) -> anyhow::Result<Box<dyn BackendPlugin>> {
        Ok(Box::new(Self(self.0.clone())))
    }

    unsafe fn run_dynamic(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        outputs: &[(usize, DLTensor)],
        transfers: &mut Transfers,
    ) -> anyhow::Result<Vec<Option<SharedTensor>>> {
        // The backend must receive the original input allocation.
        assert_eq!(inputs[0].descriptor().data as usize, *self.0);
        if let Some(output) = outputs.first() {
            let input = inputs[0].as_cpu_slice::<f32>().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    output.1.data.cast::<f32>(),
                    input.len(),
                )
            };
            Ok(vec![None])
        } else {
            Ok(vec![Some(transfers.copy(inputs[0], MemoryLocation::Cpu)?)])
        }
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
                outputs: vec![i + 1],
            })
            .collect(),
        inputs: IndexMap::from([("input".into(), 0)]),
        public_inputs: 1,
        outputs: IndexMap::from([("output".into(), nodes)]),
        slots: (0..=nodes).map(|_| Arc::new(spec())).collect(),
        terminal_outputs: [nodes].into_iter().collect(),
    })
}

#[test]
fn dynamic_only_trait_defaults_and_peers_preserve_output_ownership() {
    let mut values = [2f32];
    let address = Arc::new(values.as_ptr() as usize);
    let mut model: Box<dyn BackendPlugin> = Box::new(CopyPlugin(address));
    assert!(model.needs_guard());
    assert!(!model.can_prepare());
    assert!(!model.is_prepared());
    assert_eq!(model.input_alignment(), 0);
    assert_eq!(
        backends::error("execute", unsafe { model.run(&[], &[], &[]) }.unwrap_err()).code(),
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
        Some(Arc::new(Mutex::new(()))),
        &SessionOptions::default(),
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
    let first = parent
        .run(
            &[("input", BorrowedTensor::cpu(&values, &[1]).unwrap())],
            &OutputPlacement::default(),
            &mut HashMap::new(),
        )
        .unwrap();
    drop(parent);
    values[0] = 7.;
    let second = peer
        .run(
            &[("input", BorrowedTensor::cpu(&values, &[1]).unwrap())],
            &OutputPlacement::default(),
            &mut HashMap::new(),
        )
        .unwrap();
    drop(peer);
    assert_eq!(first["output"].to_cpu_vec::<f32>().unwrap(), [2.]);
    assert_eq!(second["output"].to_cpu_vec::<f32>().unwrap(), [7.]);
}

// A prepared backend verifies preselected input routing and no retry after an
// execution error without needing an SDK or adding production test hooks.
struct PreparedProbe {
    alignment: usize,
    outcome: Result<(), &'static str>,
    input: usize,
    outputs: Vec<usize>,
    runs: Arc<AtomicUsize>,
}

impl BackendPlugin for PreparedProbe {
    fn new_session(&self) -> anyhow::Result<Box<dyn BackendPlugin>> {
        Ok(Box::new(Self {
            input: 0,
            outputs: Vec::new(),
            runs: self.runs.clone(),
            alignment: self.alignment,
            outcome: self.outcome,
        }))
    }

    fn can_prepare(&self) -> bool {
        true
    }

    fn is_prepared(&self) -> bool {
        !self.outputs.is_empty()
    }

    fn input_alignment(&self) -> usize {
        self.alignment
    }

    unsafe fn prepare(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        outputs: &[BorrowedTensor<'_>],
    ) -> anyhow::Result<()> {
        self.input = inputs[0].descriptor().data as usize;
        self.outputs = outputs
            .iter()
            .map(|view| view.descriptor().data as usize)
            .collect();
        Ok(())
    }

    unsafe fn run(
        &mut self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        routes: &[InputRoute],
        bindings: &[(usize, DLTensor)],
    ) -> anyhow::Result<()> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        self.outcome.map_err(anyhow::Error::msg)?;
        let address = routes.first().map_or(self.input, |route| {
            let tensor = inputs[route.boundary].1.descriptor();
            let address = tensor.data as usize + tensor.byte_offset as usize;
            assert_ne!(address, self.input);
            address
        });
        // SAFETY: this fixture's validated f32 input and prepared output stay
        // alive under the workspace lock, whether routed or already bound.
        for (position, &original) in self.outputs.iter().enumerate() {
            let output = bindings
                .iter()
                .find(|output| output.0 == position)
                .map_or(original, |output| {
                    output.1.data as usize + output.1.byte_offset as usize
                });
            unsafe { *(output as *mut f32) = *(address as *const f32) };
        }
        Ok(())
    }

    unsafe fn run_dynamic(
        &mut self,
        _: &[BorrowedTensor<'_>],
        _: &[(usize, DLTensor)],
        _: &mut Transfers,
    ) -> anyhow::Result<Vec<Option<SharedTensor>>> {
        unreachable!("prepared fixture")
    }

    fn loaded_identity(&self) -> usize {
        Arc::as_ptr(&self.runs) as usize
    }
}

#[test]
fn prepared_execution_uses_one_entry_point_and_never_retries_errors() {
    for (alignment, outcome) in [
        (1, Ok(())),
        (0, Ok(())),
        (1, Err("injected execution failure")),
    ] {
        let runs = Arc::new(AtomicUsize::new(0));
        let model = PreparedProbe {
            alignment,
            outcome,
            input: 0,
            outputs: Vec::new(),
            runs: runs.clone(),
        };
        let engine = NativeEngine::with_models(
            plan(1),
            vec![Box::new(model)],
            None,
            &SessionOptions::default(),
        )
        .unwrap();
        let result = engine.run(
            &[("input", BorrowedTensor::cpu(&[7f32], &[1]).unwrap())],
            &OutputPlacement::default(),
            &mut HashMap::new(),
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
        assert_eq!(runs.load(Ordering::SeqCst), 1);
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

    unsafe fn run_dynamic(
        &mut self,
        _inputs: &[BorrowedTensor<'_>],
        _outputs: &[(usize, DLTensor)],
        _transfers: &mut Transfers,
    ) -> anyhow::Result<Vec<Option<SharedTensor>>> {
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
    let result = NativeEngine::with_models(plan.clone(), models, None, &SessionOptions::default());
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
