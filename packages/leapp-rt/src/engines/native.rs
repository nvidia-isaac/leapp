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
        for target in options
            .output_placement
            .default_location
            .iter()
            .chain(options.output_placement.overrides.values().flatten())
        {
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
                    options.cuda_graph_replay.contains(name),
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
        let mut locations = vec![location; plan.slots.len()];
        for call in &plan.calls {
            for (position, &id) in call.outputs.iter().enumerate() {
                locations[id] = models[call.model]
                    .output_location(position)
                    .unwrap_or(location);
            }
        }
        let slots: Vec<_> = plan
            .slots
            .iter()
            .enumerate()
            .map(|(id, spec)| {
                plan::Slot::new(
                    spec.clone(),
                    plan.terminal_outputs.contains(&id),
                    locations[id],
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .map_err(|e| RuntimeError::new("prepare", e))?;
        let input_bytes = plan
            .inputs
            .values()
            .map(|&id| slots[id].fixed.as_ref().map_or(0, |fixed| fixed.buffer.len))
            .collect();
        let mut input_positions = vec![None; plan.slots.len()];
        let mut staged_inputs = Vec::new();
        for (position, &slot) in plan.inputs.values().enumerate() {
            if plan.feedback_inputs.contains(&slot) && slots[slot].fixed.is_some() {
                // Keep feedback bound to its existing workspace allocation. Session
                // retains committed values; only the slot contents change per call.
                staged_inputs.push((slot, position));
            } else {
                input_positions[slot] = Some(position);
            }
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
            shape_values: vec![None; plan.symbols.len()],
            models,
            slots,
            plan,
            location,
            input_positions,
            input_bytes,
            staged_inputs,
            calls,
            replay_calls: Vec::new(),
            dynamic_slots,
            output_bindings: Vec::with_capacity(binding_capacity),
            forwarded_bindings: HashMap::new(),
            allocated_outputs: HashMap::new(),
            transfers: Transfers::new(location, options)
                .map_err(|e| backends::error("prepare", e))?,
        };
        // Workspace's field order also protects buffers on failed preparation.
        for call in &workspace.plan.calls {
            let replay = options.cuda_graph_replay.contains(&call.name)
                || options
                    .onnx_tensorrt
                    .get(&call.name)
                    .map(crate::session::onnx_tensorrt::replay_requested)
                    .transpose()
                    .map_err(|e| RuntimeError::new("prepare", e))?
                    .unwrap_or(false);
            let mut external_inputs = Vec::new();
            let mut forwarded_inputs = Vec::new();
            let mut staged_inputs = Vec::new();
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
                // Preserve producer storage for public outputs; consumers bind
                // private execution-device slots only where a transfer is needed.
                let mut input_slots = call.inputs.clone();
                for id in &mut input_slots {
                    if workspace.slots[*id].location() != location {
                        let slot = workspace.slots.len();
                        workspace.slots.push(
                            plan::Slot::new(workspace.slots[*id].spec.clone(), false, location)
                                .map_err(|e| backends::error("prepare", e))?,
                        );
                        staged_inputs.push((*id, slot));
                        *id = slot;
                    }
                }
                let inputs = input_slots
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
            if replay && !prepared {
                return Err(RuntimeError::new(
                    "prepare",
                    anyhow::anyhow!(
                        "node {}: replay requires fixed prepared bindings",
                        call.name
                    ),
                ));
            }
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
            if prepared {
                for (position, &id) in call.inputs.iter().enumerate() {
                    if workspace.input_positions[id].is_none() {
                        if input_alignment != 0 && !replay {
                            forwarded_inputs.push((position, id));
                        } else if workspace.slots[id].location() == location {
                            // These consumers require their original prepared
                            // address (replay or backend input isolation).
                            staged_inputs.push((id, id));
                        }
                    }
                }
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
            if replay {
                workspace.replay_calls.push(workspace.calls.len());
            }
            workspace.calls.push(CallState {
                execute: if replay {
                    Workspace::run_replay_first
                } else if !prepared {
                    Workspace::run_dynamic
                } else if staged_inputs.is_empty() {
                    Workspace::run_prepared
                } else {
                    Workspace::run_prepared_staged
                },
                staged_inputs,
                forwarded_inputs,
                input_alignment,
                output_routes: call
                    .outputs
                    .iter()
                    .map(|&id| {
                        if workspace.slots[id].terminal {
                            OutputRoute::Terminal
                        } else {
                            OutputRoute::Forward(id)
                        }
                    })
                    .collect(),
                replay: replay.then(Vec::new),
                replay_failed: false,
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
                    // Replay keeps private feedback in its prepared output slots.
                    .filter(|(_, _, name)| {
                        !replay
                            || workspace.plan.outputs.get_index_of(name).unwrap()
                                < workspace.plan.public_outputs
                    })
                    .collect(),
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
        workspace: &mut Workspace,
    ) -> anyhow::Result<()> {
        let symbols = &workspace.plan.symbols;
        let (lengths, values) = (&mut workspace.input_bytes, &mut workspace.shape_values);
        // Bind public dimensions before any model or feedback staging runs.
        values.fill(None);
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
            *bytes = plan::check(spec, *value, self.device, symbols, values)?;
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
    shape_values: Vec<Option<i64>>,
    location: MemoryLocation,
    // Direct Session input positions; fixed feedback and node outputs use slots.
    input_positions: Vec<Option<usize>>,
    input_bytes: Vec<usize>,
    // Fixed feedback, plus prepared calls whose backend disables input borrowing.
    staged_inputs: Vec<(usize, usize)>,
    calls: Vec<CallState>,
    // Preflight skips ordinary nodes, including all nodes when replay is disabled.
    replay_calls: Vec<usize>,
    dynamic_slots: Vec<usize>,
    output_bindings: Vec<(usize, DLTensor)>,
    // Completed output overrides borrowed only until this invocation finishes.
    forwarded_bindings: HashMap<usize, DLTensor>,
    // Keep invocation allocations alive until forwarded backend views are released,
    // including downstream failures and partially collected public results.
    allocated_outputs: HashMap<usize, Arc<Buffer>>,
    transfers: Transfers,
}

// SAFETY: the mutex exclusively owns every handle and descriptor. Invocation
// bindings are cleared before returning; moving scratch capacity never accesses
// the caller's storage. All other workspace fields are independently Send.
unsafe impl Send for Workspace {}

// Dispatch uses a preselected function, with no mode flag in the loop.
type Execute = fn(&mut Workspace, usize, &[(&str, BorrowedTensor<'_>)]) -> anyhow::Result<()>;

// Destination handling for completed overrides, resolved at session setup.
enum OutputRoute {
    Terminal,
    Forward(usize),
}

struct CallState {
    execute: Execute,
    // Producer slot and the consumer's execution-device staging slot.
    staged_inputs: Vec<(usize, usize)>,
    output_routes: Vec<OutputRoute>,
    // Prepared argument positions that can borrow completed internal outputs.
    forwarded_inputs: Vec<(usize, usize)>,
    input_alignment: usize,
    // Input position within this call and its boundary position in Plan.inputs.
    external_inputs: Vec<InputRoute>,
    outputs: Vec<(usize, usize, String)>,
    // Only direct boundary addresses can change; other bindings use fixed slots.
    replay: Option<Vec<usize>>,
    replay_failed: bool,
}

impl CallState {
    /// Effective data addresses, independent of temporary DLPack/ORT wrappers.
    fn addresses<'a>(
        &'a self,
        inputs: &'a [(&str, BorrowedTensor<'_>)],
        outputs: impl Iterator<Item = usize> + 'a,
    ) -> impl Iterator<Item = usize> + 'a {
        self.external_inputs
            .iter()
            .map(|route| {
                let tensor = inputs[route.boundary].1.descriptor();
                tensor.data as usize + tensor.byte_offset as usize
            })
            .chain(outputs)
    }
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

        self.validate(inputs, outputs, &mut workspace)
            .map_err(|error| backends::error("execute", error))?;

        // Check every replay node before staging or advancing any backend.
        workspace
            .validate_replay(inputs, outputs)
            .map_err(|e| backends::error("execute", e))?;

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
        // Release invocation descriptors; replay retains only effective addresses.
        workspace.output_bindings.clear();
        workspace.forwarded_bindings.clear();
        for model in &mut workspace.models {
            model.release_outputs();
        }
        workspace.allocated_outputs.clear();

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
    fn forwarded_input<'a>(
        slots: &'a [plan::Slot],
        bindings: &'a HashMap<usize, DLTensor>,
        id: usize,
    ) -> anyhow::Result<BorrowedTensor<'a>> {
        match bindings.get(&id) {
            // SAFETY: completed, validated overrides remain alive until run returns.
            Some(tensor) => Ok(unsafe { BorrowedTensor::new(tensor) }),
            None => slots[id].view(),
        }
    }

    /// Reject changed replay bindings before any node can change session state.
    fn validate_replay(
        &self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        buffers: &HashMap<&str, OutputBuffer<'_>>,
    ) -> anyhow::Result<()> {
        for &index in &self.replay_calls {
            let call = &self.calls[index];
            let spec = &self.plan.calls[index];
            let saved = call.replay.as_ref().unwrap();
            ensure!(
                !call.replay_failed,
                "node {}: replay backend failed; close and reopen the session",
                spec.name
            );
            for (_, id, name) in &call.outputs {
                ensure!(buffers.contains_key(name.as_str()), "replay output {name} requires a stable CUDA buffer supplied through run_into()");
                debug_assert!(self.slots[*id].fixed.is_some());
            }
            if !saved.is_empty() {
                let outputs = call
                    .outputs
                    .iter()
                    .map(|(_, _, name)| buffers[name.as_str()].address());
                ensure!(saved.iter().copied().eq(call.addresses(inputs, outputs)),
                    "node {}: CUDA replay input/output addresses changed; retain buffers or open a new session", spec.name);
            }
        }
        Ok(())
    }

    /// Record bindings once, then use the ordinary prepared dispatcher on every replay.
    fn run_replay_first(
        &mut self,
        index: usize,
        inputs: &[(&str, BorrowedTensor<'_>)],
    ) -> anyhow::Result<()> {
        let outputs = self
            .output_bindings
            .iter()
            .map(|(_, tensor)| tensor.data as usize + tensor.byte_offset as usize);
        self.calls[index].replay = Some(self.calls[index].addresses(inputs, outputs).collect());
        let execute: Execute = if self.calls[index].staged_inputs.is_empty() {
            Self::run_prepared
        } else {
            Self::run_prepared_staged
        };
        self.calls[index].execute = execute;
        execute(self, index, inputs)
    }

    /// Refresh private consumer slots only for calls selected during preparation.
    fn run_prepared_staged(
        &mut self,
        index: usize,
        inputs: &[(&str, BorrowedTensor<'_>)],
    ) -> anyhow::Result<()> {
        for &(source, slot) in &self.calls[index].staged_inputs {
            if source == slot && !self.forwarded_bindings.contains_key(&source) {
                continue;
            }
            let tensor =
                *Self::forwarded_input(&self.slots, &self.forwarded_bindings, source)?.descriptor();
            if self.calls[index].input_alignment != 0
                && self.calls[index].replay.is_none()
                && crate::engines::memory::dlpack::location(tensor.device)? == self.location
            {
                // A supplied output can already be on the consumer's device,
                // even when its producer's normal output location differs.
                continue;
            }
            let target = &mut self.slots[slot];
            let fixed = target.fixed.as_ref().unwrap();
            if tensor.data == fixed.view().descriptor().data && tensor.byte_offset == 0 {
                continue;
            }
            let len = fixed.buffer.len;
            // SAFETY: completed producer storage lives through the graph. Ports
            // match, and the staging destination is separate from this view.
            unsafe {
                target.assign_validated(BorrowedTensor::new(&tensor), len, &mut self.transfers)?;
            }
        }
        self.run_prepared(index, inputs)
    }

    fn run_prepared(
        &mut self,
        index: usize,
        inputs: &[(&str, BorrowedTensor<'_>)],
    ) -> anyhow::Result<()> {
        let call = &self.calls[index];
        // Most fixed calls retain their existing wrappers. Extend invocation
        // routes only when a producer actually supplied a different address.
        let mut forwarded = call
            .forwarded_inputs
            .iter()
            .filter_map(|&(position, id)| {
                self.forwarded_bindings
                    .get(&id)
                    .filter(|tensor| {
                        crate::engines::memory::dlpack::location(tensor.device).ok()
                            == Some(self.location)
                    })
                    .map(|tensor| (position, tensor))
            })
            .peekable();
        let model = &mut self.models[self.plan.calls[index].model];
        if forwarded.peek().is_none() {
            // SAFETY: invocation owners and validated destinations remain alive.
            unsafe {
                model.run(inputs, &call.external_inputs, &self.output_bindings)?;
            }
        } else {
            let mut values = inputs.to_vec();
            let mut routes = call.external_inputs.clone();
            for (position, tensor) in forwarded {
                ensure!(
                    (tensor.data as usize + tensor.byte_offset as usize)
                        .is_multiple_of(call.input_alignment),
                    "forwarded input requires a {}-byte aligned address",
                    call.input_alignment
                );
                routes.push(InputRoute {
                    position,
                    boundary: values.len(),
                });
                // SAFETY: producer owners and metadata live through this graph.
                values.push(("", unsafe { BorrowedTensor::new(tensor) }));
            }
            // SAFETY: completed forwarded views have the prepared port layouts.
            unsafe {
                model.run(&values, &routes, &self.output_bindings)?;
            }
        }
        // None means run wrote the prepared destination.
        for position in 0..self.plan.calls[index].outputs.len() {
            if !self.output_bindings.iter().any(|(p, _)| *p == position) {
                let call = &self.plan.calls[index];
                let id = call.outputs[position];
                if let Some(tensor) = self.models[call.model].forwarded_output(position) {
                    self.bind_lent(id, tensor)?;
                }
            }
        }
        Ok(())
    }

    /// Validate a result a backend lent instead of copying, and route it to
    /// consumers and publication. It stays valid until release_outputs.
    fn bind_lent(&mut self, id: usize, tensor: DLTensor) -> anyhow::Result<()> {
        // SAFETY: the backend retains storage and metadata until release_outputs,
        // after all downstream calls and public snapshots.
        let view = unsafe { BorrowedTensor::new(&tensor) };
        let spec = &self.slots[id].spec;
        plan::check(
            spec,
            view,
            tensor.device,
            &self.plan.symbols,
            &mut self.shape_values,
        )?;
        self.forwarded_bindings.insert(id, tensor);
        Ok(())
    }

    fn run_dynamic(
        &mut self,
        index: usize,
        inputs: &[(&str, BorrowedTensor<'_>)],
    ) -> anyhow::Result<()> {
        let call = &self.plan.calls[index];
        let staged = call
            .inputs
            .iter()
            .map(|&id| {
                if self.input_positions[id].is_none() {
                    let value = Self::forwarded_input(&self.slots, &self.forwarded_bindings, id)?;
                    if crate::engines::memory::dlpack::location(value.descriptor().device)?
                        != self.location
                    {
                        return self.transfers.copy(value, self.location).map(Some);
                    }
                }
                Ok(None)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let inputs = call
            .inputs
            .iter()
            .zip(&staged)
            .map(|(&id, staged)| match staged {
                // SAFETY: these private copies stay alive through backend completion.
                Some(value) => Ok(unsafe { value.borrow() }),
                None => match self.input_positions[id] {
                    Some(position) => Ok(inputs[position].1),
                    None => Self::forwarded_input(&self.slots, &self.forwarded_bindings, id),
                },
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let symbols = &self.plan.symbols;
        for (position, spec) in &call.checked_inputs {
            let input = inputs[*position];
            plan::check(
                spec,
                input,
                input.descriptor().device,
                symbols,
                &mut self.shape_values,
            )?;
        }
        // Session checked supplied destinations; bind their bounded dimensions.
        for &(position, tensor) in &self.output_bindings {
            let spec = &self.slots[call.outputs[position]].spec;
            if spec.shape.is_bounded() {
                // SAFETY: caller destinations have passed layout and ownership checks.
                let value = unsafe { BorrowedTensor::new(&tensor) };
                plan::check(spec, value, tensor.device, symbols, &mut self.shape_values)?;
            }
        }
        // SAFETY: Session inputs, forwarded values and writable outputs remain
        // alive through backend completion, including errors.
        let model = call.model;
        let outputs = unsafe {
            self.models[model].run_dynamic(&inputs, &self.output_bindings, &mut self.transfers)
        }?;
        for (position, value) in outputs.into_iter().enumerate() {
            let id = self.plan.calls[index].outputs[position];
            match value {
                Some(value) => {
                    // Backends guarantee rank and fixed sizes; check bounds and
                    // bind symbols before any consumer runs.
                    let spec = &self.slots[id].spec;
                    if spec.shape.is_bounded() {
                        // SAFETY: the view ends before value moves into its slot.
                        let view = unsafe { value.borrow() };
                        let device = view.descriptor().device;
                        let symbols = &self.plan.symbols;
                        plan::check(spec, view, device, symbols, &mut self.shape_values)?;
                    }
                    self.slots[id].assign_owned(value, &mut self.transfers)?;
                }
                None if self.output_bindings.iter().any(|(p, _)| *p == position) => {}
                None => {
                    let tensor =
                        self.models[model]
                            .forwarded_output(position)
                            .with_context(|| {
                                format!("backend returned no result for output {position}")
                            })?;
                    self.bind_lent(id, tensor)?;
                }
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
        // Refresh existing slots for fixed feedback and backends that disable
        // borrowing. Session keeps committed values, so a failed invocation is
        // restored from those values before the next attempt, including reset.
        for &(slot, order) in &self.staged_inputs {
            if order >= self.plan.public_inputs {
                self.input_bytes[order] = plan::check(
                    &self.slots[slot].spec,
                    inputs[order].1,
                    self.slots[slot].view()?.descriptor().device,
                    &self.plan.symbols,
                    &mut self.shape_values,
                )?;
            }
            // SAFETY: NativeEngine::validate checked public inputs; private
            // copies were checked above. input_bytes has current lengths.
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
        // Own fresh result buffers by slot ID through forwarding cleanup.
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
                    if placement.location(name).unwrap_or(fixed.buffer.location)
                        == fixed.buffer.location
                    {
                        let buffer = Arc::new(Buffer::allocate_for_copy(
                            fixed.buffer.len,
                            fixed.buffer.location,
                        )?);
                        let mut tensor = *fixed.view().descriptor();
                        tensor.data = buffer.ptr.as_ptr().cast();
                        self.output_bindings.push((position, tensor));
                        self.allocated_outputs.insert(id, buffer);
                    }
                }
            }
            (self.calls[index].execute)(self, index, inputs).with_context(|| {
                if self.calls[index].replay.is_some() {
                    // A failed capture may retain partial provider state. Never retry it.
                    self.calls[index].replay_failed = true;
                }
                format!("execute node {}", self.plan.calls[index].name)
            })?;
            // Forward completed overrides to downstream consumers.
            for binding in &self.output_bindings {
                match self.calls[index].output_routes[binding.0] {
                    OutputRoute::Terminal => {}
                    OutputRoute::Forward(id) => {
                        self.forwarded_bindings.insert(id, binding.1);
                    }
                }
            }
        }

        // Collect owned results; caller-provided destinations are already filled.
        self.plan
            .outputs
            .iter()
            .filter(|(name, _)| !buffers.contains_key(name.as_str()))
            .map(|(name, &id)| {
                let value = if let Some(buffer) = self.allocated_outputs.get(&id) {
                    // SAFETY: successful execution initialized this complete allocation.
                    unsafe {
                        crate::engines::memory::dlpack::from_buffer(
                            crate::engines::memory::dlpack::dl_type(self.slots[id].spec.dtype),
                            self.slots[id].spec.shape.dims.clone(),
                            buffer.clone(),
                        )
                    }?
                } else if let Some(tensor) = self.forwarded_bindings.get(&id) {
                    // SAFETY: retain backend owners until this independent public
                    // result is complete; forwarded results may alias inputs/state.
                    let view = unsafe { BorrowedTensor::new(tensor) };
                    self.transfers.copy(
                        view,
                        placement
                            .location(name)
                            .unwrap_or(self.slots[id].location()),
                    )?
                } else {
                    self.slots[id].snapshot(
                        placement
                            .location(name)
                            .unwrap_or_else(|| self.slots[id].location()),
                        &mut self.transfers,
                    )?
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
