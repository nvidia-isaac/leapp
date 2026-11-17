// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared loaded models with one reusable, exclusively owned workspace per session.

mod backends;
mod plan;
mod transfers;

use crate::{
    dlpack::{BorrowedTensor, DLDevice, DLTensor, OutputBuffer, SharedTensor},
    engines::{memory::Buffer, Engine},
    session::graph::Graph,
    Bundle, ExecutionDevice, MemoryLocation, OutputPlacement, RuntimeError, SessionOptions,
};
use anyhow::{ensure, Context};
use backends::{BackendPlugin, InputRoute};
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
                backends::load(
                    spec,
                    location,
                    compute_stream,
                    options.onnx_tensorrt.get(name),
                )
                .map_err(|e| backends::error("load", e.context(format!("load node {name}"))))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let guarded = models.iter().any(|model| model.needs_guard());
        Self::with_models(
            Arc::new(plan),
            models,
            guarded.then(|| Arc::new(Mutex::new(()))),
            options,
        )
    }

    fn with_models(
        plan: Arc<Plan>,
        models: Vec<Box<dyn BackendPlugin>>,
        execution_guard: Option<Arc<Mutex<()>>>,
        options: &SessionOptions,
    ) -> Result<Self, RuntimeError> {
        let location = match options.device {
            ExecutionDevice::Cpu => MemoryLocation::Cpu,
            ExecutionDevice::Cuda { device_id } => MemoryLocation::Cuda { device_id },
        };
        let slots: Vec<_> = plan
            .slots
            .iter()
            .enumerate()
            .map(|(id, spec)| {
                plan::Slot::new(spec.clone(), plan.terminal_outputs.contains(&id), location)
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|e| RuntimeError::new("prepare", e))?;
        let input_bytes = plan
            .inputs
            .values()
            .map(|&id| slots[id].fixed.as_ref().map_or(0, |fixed| fixed.buffer.len))
            .collect();
        let mut input_positions = vec![None; plan.slots.len()];
        for (position, &slot) in plan.inputs.values().enumerate() {
            input_positions[slot] = Some(position);
        }
        let dynamic_slots = slots
            .iter()
            .enumerate()
            .filter_map(|(id, slot)| slot.fixed.is_none().then_some(id))
            .collect();
        let calls = Vec::with_capacity(plan.calls.len());
        let mut input_alignments = vec![0; plan.public_inputs];
        let mut output_alignments = HashMap::new();
        let binding_capacity = plan
            .calls
            .iter()
            .map(|call| call.outputs.len())
            .max()
            .unwrap_or(0);
        let device = crate::engines::memory::dlpack::device(location)
            .map_err(|e| RuntimeError::new("prepare", e.into()))?;
        let mut workspace = Workspace {
            models,
            slots,
            plan,
            input_positions,
            input_bytes,
            staged_inputs: Vec::new(),
            calls,
            dynamic_slots,
            output_bindings: Vec::with_capacity(binding_capacity),
            transfers: Transfers::new(location, options)
                .map_err(|e| backends::error("prepare", e))?,
        };
        // Workspace's field order also protects buffers on failed preparation.
        for call in &workspace.plan.calls {
            let mut external_inputs = Vec::new();
            for (position, &id) in call.inputs.iter().enumerate() {
                if let Some(order) = workspace.input_positions[id] {
                    external_inputs.push(InputRoute {
                        position,
                        boundary: order,
                    });
                }
            }
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
            let prepared = workspace.models[call.model].is_prepared();
            if !prepared {
                workspace
                    .dynamic_slots
                    .extend(call.outputs.iter().copied().filter(|&id| {
                        workspace.slots[id].terminal && workspace.slots[id].fixed.is_some()
                    }));
            }
            let input_alignment = if prepared {
                workspace.models[call.model].input_alignment()
            } else {
                // Dynamic calls borrow Session inputs on the execution device.
                1
            };
            if input_alignment != 0 {
                for route in &external_inputs {
                    if route.boundary < workspace.plan.public_inputs {
                        input_alignments[route.boundary] =
                            input_alignments[route.boundary].max(input_alignment);
                    }
                }
            } else {
                workspace.staged_inputs.extend(
                    external_inputs
                        .iter()
                        .map(|route| (call.inputs[route.position], route.boundary)),
                );
                external_inputs.clear();
            }
            for &id in &call.outputs {
                if let Some((name, _)) =
                    workspace.plan.outputs.iter().find(|(_, slot)| **slot == id)
                {
                    output_alignments.insert(
                        name.clone(),
                        workspace.models[call.model].output_alignment(),
                    );
                }
            }
            workspace.calls.push(CallState {
                outputs: call
                    .outputs
                    .iter()
                    .enumerate()
                    .filter_map(|(position, &id)| {
                        workspace
                            .plan
                            .outputs
                            .iter()
                            .find(|(_, slot)| **slot == id)
                            .map(|(name, _)| (position, id, name.clone()))
                    })
                    .collect(),
                execute: if prepared {
                    Workspace::run_prepared
                } else {
                    Workspace::run_dynamic
                },
                external_inputs: if prepared {
                    external_inputs
                } else {
                    Vec::new()
                },
            });
        }

        let inputs = workspace
            .plan
            .inputs
            .values()
            .take(workspace.plan.public_inputs)
            .zip(input_alignments)
            .map(|(&id, alignment)| (workspace.slots[id].spec.clone(), alignment))
            .collect();
        Ok(Self {
            inputs,
            output_alignments,
            device,
            workspace: Mutex::new(workspace),
            execution_guard,
            options: options.clone(),
        })
    }

    fn validate(
        &self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        outputs: &HashMap<&str, OutputBuffer<'_>>,
        lengths: &mut [usize],
    ) -> anyhow::Result<()> {
        // Names/order are established by Session. Private state cannot alias caller
        // storage and was validated when loaded. Only public metadata can change.
        for (((spec, alignment), (_, value)), bytes) in
            self.inputs.iter().zip(inputs).zip(lengths.iter_mut())
        {
            let tensor = value.descriptor();
            if *alignment != 0 {
                ensure!(
                    tensor.device.device_type == self.device.device_type
                        && tensor.device.device_id == self.device.device_id,
                    "port {} must be on the execution device for direct input binding",
                    spec.name
                );
            }
            *bytes = plan::matches(spec, *value, self.device)?;
            let address = tensor.data as usize + tensor.byte_offset as usize;
            if *alignment > 1 {
                ensure!(
                    address.is_multiple_of(*alignment),
                    "port {} requires a {alignment}-byte aligned address for direct input binding",
                    spec.name
                );
            }
        }
        // Validate only supplied outputs, using requirements resolved at initialization.
        for (&name, output) in outputs {
            let tensor = output.descriptor();
            ensure!(
                tensor.device.device_type == self.device.device_type
                    && tensor.device.device_id == self.device.device_id,
                "output buffer {name} must be on the execution device"
            );
            let alignment = self.output_alignments[name];
            ensure!(
                output.bytes == 0
                    || (tensor.data as usize + tensor.byte_offset as usize)
                        .is_multiple_of(alignment),
                "output buffer {name} requires a {alignment}-byte aligned address"
            );
            for ((_, input), &bytes) in inputs.iter().zip(lengths.iter()).take(self.inputs.len()) {
                let source = input.descriptor();
                ensure!(
                    !crate::engines::memory::dlpack::ranges_overlap(
                        source.device,
                        source.data as usize + source.byte_offset as usize,
                        bytes,
                        tensor.device,
                        output.address(),
                        output.bytes
                    ),
                    "output buffer overlaps an input"
                );
            }
        }
        Ok(())
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
            self.execution_guard.clone(),
            &self.options,
        )
    }
}

struct NativeEngine {
    inputs: Vec<(Arc<plan::TensorSpec>, usize)>,
    output_alignments: HashMap<String, usize>,
    device: DLDevice,
    workspace: Mutex<Workspace>,
    execution_guard: Option<Arc<Mutex<()>>>,
    options: SessionOptions,
}

struct Workspace {
    // Prepared wrappers borrow slots. Drop all models before workspace storage.
    models: Vec<Box<dyn BackendPlugin>>,
    slots: Vec<plan::Slot>,
    plan: Arc<Plan>,
    // Session input position for each boundary slot; None for node outputs.
    input_positions: Vec<Option<usize>>,
    input_bytes: Vec<usize>,
    // Prepared calls whose backend disables input borrowing.
    staged_inputs: Vec<(usize, usize)>,
    calls: Vec<CallState>,
    dynamic_slots: Vec<usize>,
    output_bindings: Vec<(usize, DLTensor)>,
    transfers: Transfers,
}

// SAFETY: the mutex exclusively owns every handle and descriptor. Invocation
// bindings are cleared before returning; moving scratch capacity never accesses
// the caller's storage. All other workspace fields are independently Send.
unsafe impl Send for Workspace {}

// Select each node's execution function once, with no mode flag in the loop.
type Execute = fn(&mut Workspace, usize, &[(&str, BorrowedTensor<'_>)]) -> anyhow::Result<()>;

struct CallState {
    execute: Execute,
    // Input position within this call and its boundary position in Plan.inputs.
    external_inputs: Vec<InputRoute>,
    outputs: Vec<(usize, usize, String)>,
}

impl Engine for NativeEngine {
    fn new_session(self: Arc<Self>) -> Result<Arc<dyn Engine>, RuntimeError> {
        Ok(Arc::new(self.peer()?))
    }

    fn run(
        &self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        placement: &OutputPlacement,
        outputs: &mut HashMap<&str, OutputBuffer<'_>>,
    ) -> Result<IndexMap<String, SharedTensor>, RuntimeError> {
        let mut workspace = self.workspace.lock().map_err(|_| {
            RuntimeError::new(
                "execute",
                crate::error::Internal("native workspace lock poisoned".into()).into(),
            )
        })?;

        self.validate(inputs, outputs, &mut workspace.input_bytes)
            .map_err(|error| backends::error("execute", error))?;

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
        let result = workspace.run(inputs, placement, outputs);
        // Scratch capacity is reusable; invocation pointers must not survive return.
        workspace.output_bindings.clear();

        // No dynamic value from a failed/previous call can feed a later call.
        let workspace = &mut *workspace;
        for &id in &workspace.dynamic_slots {
            workspace.slots[id].clear_dynamic();
        }

        result.map_err(|e| backends::error("execute", e))
    }

    fn copy(
        &self,
        input: BorrowedTensor<'_>,
        target: MemoryLocation,
    ) -> Result<SharedTensor, RuntimeError> {
        let mut workspace = self.workspace.lock().map_err(|_| {
            RuntimeError::new("copy", anyhow::anyhow!("native workspace lock poisoned"))
        })?;
        workspace
            .transfers
            .copy(input, target)
            .map_err(|e| backends::error("copy", e))
    }
}

impl Workspace {
    fn run_prepared(
        &mut self,
        index: usize,
        inputs: &[(&str, BorrowedTensor<'_>)],
    ) -> anyhow::Result<()> {
        // SAFETY: run retains every supplied output and validates it before dispatch.
        unsafe {
            self.models[self.plan.calls[index].model].run(
                inputs,
                &self.calls[index].external_inputs,
                &self.output_bindings,
            )
        }
    }

    fn run_dynamic(
        &mut self,
        index: usize,
        inputs: &[(&str, BorrowedTensor<'_>)],
    ) -> anyhow::Result<()> {
        let call = &self.plan.calls[index];
        let inputs = call
            .inputs
            .iter()
            .map(|&id| match self.input_positions[id] {
                Some(position) => Ok(inputs[position].1),
                None => self.slots[id].view(),
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        // SAFETY: Session inputs, forwarded values and writable outputs remain
        // alive through backend completion, including errors.
        let outputs = unsafe {
            self.models[call.model].run_dynamic(&inputs, &self.output_bindings, &mut self.transfers)
        }?;
        for (&id, value) in call.outputs.iter().zip(outputs) {
            if let Some(value) = value {
                self.slots[id].assign_owned(value, &mut self.transfers)?;
            }
        }
        Ok(())
    }

    fn run(
        &mut self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        placement: &OutputPlacement,
        buffers: &mut HashMap<&str, OutputBuffer<'_>>,
    ) -> anyhow::Result<IndexMap<String, SharedTensor>> {
        // Copy session inputs for prepared backends that disable borrowing.
        // Dynamic calls read Session inputs directly. Initialization records
        // each (workspace slot, session input index) pair in staged_inputs.
        for &(slot, order) in &self.staged_inputs {
            // SAFETY: NativeEngine::validate checked public inputs; Session owns
            // validated private defaults/feedback. input_bytes has current lengths.
            unsafe {
                self.slots[slot].assign_validated(
                    inputs[order].1,
                    self.input_bytes[order],
                    &mut self.transfers,
                )?;
            }
        }

        // Caller-provided buffers set output destinations; placement selects
        // locations for outputs returned in newly owned storage.
        // Own fresh result buffers by slot ID until publication.
        let mut allocated = HashMap::new();
        for index in 0..self.calls.len() {
            // Rebuild destination overrides, keeping other prepared bindings intact.
            // TODO: Investigate reusing output bindings to reduce per-run rebuilding.
            self.output_bindings.clear();
            // Public/feedback outputs only, unlike plan.calls[index].outputs.
            // Tuple: (node output index, workspace slot ID, qualified port name).
            for &(position, id, ref name) in &self.calls[index].outputs {
                if let Some(buffer) = buffers.get(name.as_str()) {
                    // The caller supplied a destination for this named output.
                    self.output_bindings.push((position, *buffer.descriptor()));
                } else if let Some(fixed) = &self.slots[id].fixed {
                    // No caller destination; this slot has a fixed-shape tensor.
                    // Write into independent result storage when locations match;
                    // otherwise transfer the result after execution.
                    if placement.location(name) == fixed.buffer.location {
                        let buffer = Arc::new(Buffer::allocate_for_copy(
                            fixed.buffer.len,
                            fixed.buffer.location,
                        )?);
                        let mut tensor = *fixed.view().descriptor();
                        tensor.data = buffer.ptr.as_ptr().cast();
                        self.output_bindings.push((position, tensor));
                        allocated.insert(id, buffer);
                    }
                }
            }
            (self.calls[index].execute)(self, index, inputs)
                .with_context(|| format!("execute node {}", self.plan.calls[index].name))?;
            // Copy overridden outputs back for downstream consumers.
            for binding in &self.output_bindings {
                let id = self.plan.calls[index].outputs[binding.0];
                if !self.slots[id].terminal {
                    // SAFETY: the completed backend initialized the validated target.
                    let view = unsafe { BorrowedTensor::new(&binding.1) };
                    let len = unsafe {
                        crate::engines::memory::dlpack::layout_unchecked(*view.descriptor())
                    }
                    .1;
                    unsafe { self.slots[id].assign_validated(view, len, &mut self.transfers) }?;
                }
            }
        }

        // Collect owned results; caller-provided destinations are already filled.
        self.plan
            .outputs
            .iter()
            .filter(|(name, _)| !buffers.contains_key(name.as_str()))
            .map(|(name, &id)| {
                let value = if let Some(buffer) = allocated.remove(&id) {
                    // SAFETY: successful execution initialized this complete allocation.
                    unsafe {
                        crate::engines::memory::dlpack::from_buffer(
                            crate::engines::memory::dlpack::dl_type(self.slots[id].spec.dtype),
                            self.slots[id].spec.shape.clone(),
                            buffer,
                        )
                    }?
                } else {
                    self.slots[id].snapshot(placement.location(name), &mut self.transfers)?
                };
                Ok((name.clone(), value))
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
