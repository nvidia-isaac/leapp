// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared loaded models with one reusable, exclusively owned workspace per session.

mod backends;
mod plan;
mod transfers;

use crate::{
    dlpack::{BorrowedTensor, SharedTensor},
    engines::{memory::OutputPool, Engine},
    session::graph::Graph,
    Bundle, ExecutionDevice, MemoryLocation, OutputPlacement, RuntimeError, SessionOptions,
};
use anyhow::{ensure, Context};
use backends::BackendPlugin;
use indexmap::IndexMap;
use plan::Plan;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use transfers::Transfers;

pub(super) fn open(
    bundle: &Bundle,
    graph: &Graph<'_>,
    options: &SessionOptions,
) -> Result<Arc<dyn Engine>, RuntimeError> {
    NativeEngine::load(bundle, graph, options).map(|engine| Arc::new(engine) as Arc<dyn Engine>)
}

impl NativeEngine {
    fn load(
        bundle: &Bundle,
        graph: &Graph<'_>,
        options: &SessionOptions,
    ) -> Result<Self, RuntimeError> {
        let location = match options.device {
            ExecutionDevice::Cpu => MemoryLocation::Cpu,
            ExecutionDevice::Cuda { device_id } => MemoryLocation::Cuda { device_id },
        };
        // Reject unsupported backends/devices before loading any model or workspace.
        for (name, spec) in &bundle.0.description.models {
            backends::validate(spec, location)
                .map_err(|e| backends::error("load", e.context(format!("node {name}"))))?;
        }

        location.validate()?;
        for (&device, &stream) in &options.cuda_streams {
            stream.validate(device)?;
        }
        options.output_placement.default_location.validate()?;
        for target in options.output_placement.overrides.values() {
            target.validate()?;
        }

        let plan = Plan::new(bundle, graph).map_err(|e| RuntimeError::new("convert", e))?;
        let compute_stream = match location {
            MemoryLocation::Cuda { device_id } => options.cuda_streams.get(&device_id).copied(),
            _ => None,
        };
        let models = bundle
            .0
            .description
            .models
            .iter()
            .map(|(name, spec)| {
                backends::load(spec, location, compute_stream)
                    .map_err(|e| backends::error("load", e.context(format!("load node {name}"))))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let guarded = models.iter().any(|model| model.needs_guard());
        Self::with_models(
            Arc::new(plan),
            models,
            location,
            guarded.then(|| Arc::new(Mutex::new(()))),
            options.cuda_streams.clone(),
        )
    }

    fn with_models(
        plan: Arc<Plan>,
        models: Vec<Box<dyn BackendPlugin>>,
        location: MemoryLocation,
        execution_guard: Option<Arc<Mutex<()>>>,
        cuda_streams: HashMap<u32, crate::CudaStream>,
    ) -> Result<Self, RuntimeError> {
        let slots = plan
            .slots
            .iter()
            .map(|spec| plan::Slot::new(spec.clone(), location))
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|e| RuntimeError::new("prepare", e))?;
        let input_order = vec![usize::MAX; plan.inputs.len()];
        let mut workspace = Workspace {
            models,
            slots,
            plan,
            input_order,
            transfers: Transfers::new(location, &cuda_streams)
                .map_err(|e| backends::error("prepare", e))?,
        };
        // Workspace's field order also protects buffers on failed preparation.
        for call in &workspace.plan.calls {
            if workspace.models[call.model].can_prepare()
                && call
                    .inputs
                    .iter()
                    .chain(&call.outputs)
                    .all(|&id| workspace.slots[id].fixed.is_some())
            {
                let inputs = call
                    .inputs
                    .iter()
                    .map(|&id| workspace.slots[id].view())
                    .collect::<anyhow::Result<Vec<_>>>()
                    .map_err(|e| RuntimeError::new("prepare", e))?;
                for (spec, &value) in call.input_specs.iter().zip(&inputs) {
                    plan::matches(spec, value).map_err(|e| RuntimeError::new("prepare", e))?;
                }

                let outputs = call
                    .outputs
                    .iter()
                    .map(|&id| workspace.slots[id].view())
                    .collect::<anyhow::Result<Vec<_>>>()
                    .map_err(|e| RuntimeError::new("prepare", e))?;
                // SAFETY: workspace allocations outlive models/bindings, including failed preparation.
                unsafe { workspace.models[call.model].prepare(&inputs, &outputs) }
                    .map_err(|e| backends::error("prepare", e))?;
            }
        }

        Ok(Self {
            workspace: Mutex::new(workspace),
            location,
            execution_guard,
            cuda_streams,
        })
    }

    fn peer(&self) -> Result<Self, RuntimeError> {
        let workspace = self.workspace.lock().map_err(|_| {
            RuntimeError::new("prepare", anyhow::anyhow!("native workspace lock poisoned"))
        })?;
        let models = workspace
            .models
            .iter()
            .map(|model| model.new_session())
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|e| backends::error("prepare", e))?;
        Self::with_models(
            workspace.plan.clone(),
            models,
            self.location,
            self.execution_guard.clone(),
            self.cuda_streams.clone(),
        )
    }
}

struct NativeEngine {
    workspace: Mutex<Workspace>,
    location: MemoryLocation,
    execution_guard: Option<Arc<Mutex<()>>>,
    cuda_streams: HashMap<u32, crate::CudaStream>,
}

struct Workspace {
    // Prepared wrappers borrow slots. Drop all models before workspace storage.
    models: Vec<Box<dyn BackendPlugin>>,
    slots: Vec<plan::Slot>,
    plan: Arc<Plan>,
    input_order: Vec<usize>,
    transfers: Transfers,
}

impl Engine for NativeEngine {
    fn new_session(self: Arc<Self>) -> Result<Arc<dyn Engine>, RuntimeError> {
        Ok(Arc::new(self.peer()?))
    }

    fn run(
        &self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        placement: &OutputPlacement,
        pool: &Arc<OutputPool>,
    ) -> Result<IndexMap<String, SharedTensor>, RuntimeError> {
        let mut workspace = self.workspace.lock().map_err(|_| {
            RuntimeError::new(
                "execute",
                crate::error::Internal("native workspace lock poisoned".into()).into(),
            )
        })?;

        // Stateful/unclassified models keep the existing graph-wide order.
        // Hold the guard through output snapshots, including dynamic result copies.
        let _execution = self
            .execution_guard
            .as_ref()
            .map(|guard| guard.lock())
            .transpose()
            .map_err(|_| {
                RuntimeError::new("execute", anyhow::anyhow!("native model lock poisoned"))
            })?;
        let result = workspace.run(inputs, placement, pool);

        // No dynamic value from a failed/previous call can feed a later call.
        for slot in &mut workspace.slots {
            slot.clear_dynamic();
        }

        result.map_err(|e| backends::error("execute", e))
    }

    fn copy(
        &self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
        pool: Option<&OutputPool>,
    ) -> Result<SharedTensor, RuntimeError> {
        let mut workspace = self.workspace.lock().map_err(|_| {
            RuntimeError::new("copy", anyhow::anyhow!("native workspace lock poisoned"))
        })?;
        match pool {
            Some(pool) => workspace.transfers.copy_pooled(input, target, pool),
            None => workspace.transfers.copy(input, target),
        }
        .map_err(|e| backends::error("copy", e))
    }
}

impl Workspace {
    fn run(
        &mut self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        placement: &OutputPlacement,
        pool: &OutputPool,
    ) -> anyhow::Result<IndexMap<String, SharedTensor>> {
        ensure!(
            inputs.len() == self.plan.inputs.len(),
            "native input count mismatch"
        );

        // Validate the complete boundary before changing any workspace storage.
        self.input_order.fill(usize::MAX);
        for (position, &(name, value)) in inputs.iter().enumerate() {
            let (index, _, &id) = self
                .plan
                .inputs
                .get_full(name)
                .with_context(|| format!("unknown input {name}"))?;
            ensure!(
                self.input_order[index] == usize::MAX,
                "duplicate input {name}"
            );
            plan::matches(&self.slots[id].spec, value)?;
            self.input_order[index] = position;
        }

        ensure!(
            self.input_order.iter().all(|&index| index != usize::MAX),
            "missing native input"
        );

        for name in self.plan.outputs.keys() {
            placement.location(name).validate()?;
        }
        for ((_, &id), &position) in self.plan.inputs.iter().zip(&self.input_order) {
            self.slots[id].assign(inputs[position].1, &mut self.transfers)?;
        }

        for call in &self.plan.calls {
            let model = &mut self.models[call.model];
            if model.is_prepared() {
                model
                    .run()
                    .with_context(|| format!("execute node {}", call.name))?;
            } else {
                for (&id, spec) in call.inputs.iter().zip(&call.input_specs) {
                    plan::matches(spec, self.slots[id].view()?)?;
                }

                let inputs = call
                    .inputs
                    .iter()
                    .map(|&id| self.slots[id].view())
                    .collect::<anyhow::Result<Vec<_>>>()?;
                let outputs = model
                    .run_dynamic(&inputs, &mut self.transfers)
                    .with_context(|| format!("execute node {}", call.name))?;
                ensure!(
                    outputs.len() == call.outputs.len(),
                    "native backend output count mismatch"
                );

                for (&id, value) in call.outputs.iter().zip(&outputs) {
                    // SAFETY: newly returned independent outputs are ready and unpublished.
                    self.slots[id].assign(unsafe { value.borrow() }, &mut self.transfers)?;
                }
            }
        }

        self.plan
            .outputs
            .iter()
            .map(|(name, &id)| {
                // Snapshot before releasing the workspace lock, including private feedback.
                Ok((
                    name.clone(),
                    self.transfers.copy_pooled(
                        self.slots[id].view()?,
                        placement.location(name),
                        pool,
                    )?,
                ))
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "../../tests/functional_tests/native.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/unit_tests/native/backend_plugin.rs"]
mod backend_plugin_tests;
