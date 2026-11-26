// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Indexed routing and stable workspace storage. Feedback stays in Session.

use crate::{
    bundle::shapes::{ExecutionShape, Symbols},
    dlpack::{BorrowedTensor, DLDevice, DLTensor, SharedTensor},
    engines::memory::{dlpack, Buffer},
    session::graph::{port_ref, Graph, InputSource},
    Bundle, DataType, MemoryLocation, PortSpec,
};
use anyhow::{ensure, Context, Result};
use indexmap::IndexMap;
use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

pub(super) struct TensorSpec {
    pub name: String,
    pub dtype: DataType,
    pub shape: ExecutionShape,
}

impl TensorSpec {
    fn new(port: &PortSpec, shape: &ExecutionShape) -> Arc<Self> {
        Arc::new(Self {
            name: port.name.clone(),
            dtype: port.dtype,
            shape: shape.clone(),
        })
    }
}

pub(super) struct Call {
    pub name: String,
    pub model: usize,
    pub inputs: Vec<usize>,
    /// (position, consumer spec) for inputs whose contract neither the boundary
    /// nor the producer already enforces. Usually empty.
    pub checked_inputs: Vec<(usize, Arc<TensorSpec>)>,
    pub outputs: Vec<usize>,
}

pub(super) struct Plan {
    pub calls: Vec<Call>,
    pub inputs: IndexMap<String, usize>,
    pub public_inputs: usize,
    pub public_outputs: usize,
    pub feedback_inputs: BTreeSet<usize>,
    pub outputs: IndexMap<String, usize>,
    pub slots: Vec<Arc<TensorSpec>>,
    pub terminal_outputs: BTreeSet<usize>,
    pub symbols: Symbols,
}

impl Plan {
    pub fn new(bundle: &Bundle, graph: &Graph<'_>) -> Result<Self> {
        let models = &bundle.0.description.models;
        ensure!(!models.is_empty(), "native execution requires a model");

        // Reuse the old compiler's stable dependency order; graph validation
        // already checked ports, conflicting sources and forward cycles.
        let mut next = vec![Vec::new(); models.len()];
        let mut indegree = vec![0; models.len()];
        for (source, targets) in &bundle.0.description.pipeline.data_flow {
            for target in targets {
                let from = models
                    .get_index_of(&source.node)
                    .context("missing source")?;
                let to = models
                    .get_index_of(&target.node)
                    .context("missing target")?;
                next[from].push(to);
                indegree[to] += 1;
            }
        }

        let mut ready: BTreeSet<_> = (0..models.len()).filter(|&i| indegree[i] == 0).collect();
        let mut order = Vec::new();
        while let Some(node) = ready.pop_first() {
            order.push(node);
            for &target in &next[node] {
                indegree[target] -= 1;
                if indegree[target] == 0 {
                    ready.insert(target);
                }
            }
        }
        ensure!(order.len() == models.len(), "forward cycle");

        let mut plan = Self {
            calls: Vec::new(),
            inputs: IndexMap::new(),
            public_inputs: bundle
                .pipeline_inputs()
                .iter()
                .map(|port| port.name())
                .collect::<BTreeSet<_>>()
                .len(),
            public_outputs: bundle
                .pipeline_outputs()
                .iter()
                .map(|port| port.name())
                .collect::<BTreeSet<_>>()
                .len(),
            feedback_inputs: BTreeSet::new(),
            outputs: IndexMap::new(),
            slots: Vec::new(),
            terminal_outputs: BTreeSet::new(),
            symbols: bundle.0.description.symbols.clone(),
        };
        let mut outputs = HashMap::new();
        for (name, model) in models {
            for (spec, shape) in model.output_ports() {
                outputs.insert(port_ref(name, spec), plan.slots.len());
                plan.slots.push(TensorSpec::new(spec, shape));
            }
        }

        for model_id in order {
            let (name, model) = models.get_index(model_id).unwrap();
            let mut inputs = Vec::new();
            let mut checked_inputs = Vec::new();
            for (position, (spec, shape)) in model.input_ports().enumerate() {
                let port = port_ref(name, spec);
                let source = graph.sources.get(&port);
                let (id, checked) = match source {
                    Some(InputSource::Forward(source)) => {
                        let producer = &plan.slots[outputs[source]].shape;
                        // Bundle validation proved fixed producers against unbounded
                        // consumers; an identical contract was checked when produced.
                        let checked = producer != shape
                            && (producer.dims.contains(&-1) || shape.is_bounded());
                        (outputs[source], checked)
                    }
                    // Both private defaults and feedback are supplied by Session.
                    // NativeEngine::validate checks public inputs; feedback is never
                    // bounded, and private defaults are checked where they are used.
                    _ => {
                        let id = plan.slots.len();
                        plan.slots.push(TensorSpec::new(spec, shape));
                        plan.inputs.insert(port.name(), id);
                        (id, source.is_none() && shape.is_bounded())
                    }
                };
                if checked {
                    checked_inputs.push((position, TensorSpec::new(spec, shape)));
                }
                if matches!(source, Some(InputSource::Feedback(_))) {
                    plan.feedback_inputs.insert(id);
                }
                inputs.push(id);
            }

            plan.calls.push(Call {
                name: name.clone(),
                model: model_id,
                inputs,
                checked_inputs,
                outputs: model
                    .outputs
                    .iter()
                    .map(|s| outputs[&port_ref(name, s)])
                    .collect(),
            });
        }

        let pipeline = &bundle.0.description.pipeline;
        // Session supplies public arguments first, then defaults/feedback in
        // model declaration order. Resolve that order once, independently of
        // the topological execution order.
        let boundary = bundle.pipeline_inputs().iter().cloned().chain(
            models
                .iter()
                .flat_map(|(name, model)| model.inputs.iter().map(move |spec| port_ref(name, spec)))
                .filter(|port| {
                    matches!(
                        graph.sources.get(port),
                        None | Some(InputSource::Feedback(_))
                    )
                }),
        );
        plan.inputs = boundary
            .map(|port| {
                let name = port.name();
                let id = plan.inputs[&name];
                (name, id)
            })
            .collect();
        for port in bundle
            .pipeline_outputs()
            .iter()
            .chain(pipeline.feedback_flow.keys())
        {
            plan.outputs.insert(port.name(), outputs[port]);
        }

        // Terminal slots keep backend-owned results until invocation cleanup.
        for (name, model) in models {
            for spec in &model.outputs {
                let port = port_ref(name, spec);
                if pipeline.data_flow.get(&port).is_none_or(Vec::is_empty) {
                    plan.terminal_outputs.insert(outputs[&port]);
                }
            }
        }

        Ok(plan)
    }
}

/// Checks layout, dtype and shape contract, binding the port's shared symbols
/// in this invocation's `values`. Returns the byte length.
pub(super) fn check(
    spec: &TensorSpec,
    value: BorrowedTensor<'_>,
    device: DLDevice,
    symbols: &Symbols,
    values: &mut [Option<i64>],
) -> Result<usize> {
    let (shape, len) = dlpack::layout_on(value, device)?;
    ensure!(
        dlpack::data_type(value.descriptor().dtype)? == spec.dtype && spec.shape.matches(shape),
        "port {} does not match declared dtype/shape",
        spec.name
    );
    spec.shape
        .bind(shape, symbols, values)
        .with_context(|| format!("port {} shape contract", spec.name))?;
    Ok(len)
}

pub(super) struct Slot {
    pub terminal: bool,
    pub spec: Arc<TensorSpec>,
    pub fixed: Option<FixedTensor>,
    dynamic: Option<SharedTensor>,
    location: MemoryLocation,
}

impl Slot {
    pub fn new(spec: Arc<TensorSpec>, terminal: bool, location: MemoryLocation) -> Result<Self> {
        let fixed = if spec.shape.dims.contains(&-1) {
            None
        } else {
            Some(FixedTensor::new(&spec, location)?)
        };
        Ok(Self {
            terminal,
            spec,
            fixed,
            dynamic: None,
            location,
        })
    }

    pub fn view(&self) -> Result<BorrowedTensor<'_>> {
        match (&self.fixed, &self.dynamic) {
            (Some(value), _) => Ok(value.view()),
            (_, Some(value)) => {
                // SAFETY: private completed values have never been exposed to writers.
                Ok(unsafe { value.borrow() })
            }
            _ => anyhow::bail!("uninitialized slot {}", self.spec.name),
        }
    }

    pub fn assign_owned(
        &mut self,
        value: SharedTensor,
        transfers: &mut super::Transfers,
    ) -> Result<()> {
        if self.terminal || self.fixed.is_none() {
            self.dynamic = Some(value);
            Ok(())
        } else {
            let len = self.fixed.as_ref().unwrap().buffer.len;
            // SAFETY: the plugin guarantees valid storage matching the
            // producer's declared dtype/shape, hence this fixed allocation.
            unsafe { self.assign_validated(value.borrow(), len, transfers) }
        }
    }

    /// # Safety
    /// The immutable value must have passed matches for this slot, with byte size len.
    pub unsafe fn assign_validated(
        &mut self,
        value: BorrowedTensor<'_>,
        len: usize,
        transfers: &mut super::Transfers,
    ) -> Result<()> {
        if let Some(fixed) = &mut self.fixed {
            // SAFETY: the validated value fits this exclusively owned workspace slot.
            unsafe {
                transfers.write(value, &fixed.buffer)?;
            }
        } else {
            // SAFETY: the caller checked the layout and retains it through this copy.
            self.dynamic = Some(unsafe { transfers.copy_validated(value, self.location, len)? });
        }
        Ok(())
    }

    pub fn clear_dynamic(&mut self) {
        self.dynamic = None;
    }

    pub fn location(&self) -> MemoryLocation {
        self.dynamic
            .as_ref()
            .map_or(self.location, SharedTensor::location)
    }

    pub fn snapshot(
        &self,
        target: MemoryLocation,
        transfers: &mut super::Transfers,
    ) -> Result<SharedTensor> {
        if let Some(value) = &self.dynamic {
            // Independently owned backend results can be returned without another copy.
            if target == value.location() {
                return Ok(value.clone());
            }
            // SAFETY: private, completed backend storage remains alive through the copy.
            return unsafe {
                let view = value.borrow();
                let (_, len) = dlpack::layout_unchecked(*view.descriptor());
                transfers.copy_validated(view, target, len)
            };
        }
        let fixed = self
            .fixed
            .as_ref()
            .with_context(|| format!("uninitialized slot {}", self.spec.name))?;
        // SAFETY: initialization validated the fixed descriptor/allocation;
        // execution completed before snapshotting.
        unsafe { transfers.copy_validated(fixed.view(), target, fixed.buffer.len) }
    }
}

/// Owned metadata and storage for a fixed workspace slot.
pub(super) struct FixedTensor {
    descriptor: DLTensor,
    _shape: Vec<i64>,
    pub buffer: Buffer,
}

// SAFETY: allocations and metadata are owned, accesses are exclusive under the
// workspace mutex, and moving the struct does not move its heap allocations.
unsafe impl Send for FixedTensor {}
impl FixedTensor {
    fn new(spec: &TensorSpec, location: MemoryLocation) -> Result<Self> {
        let dtype = dlpack::dl_type(spec.dtype);
        let buffer = Buffer::allocate(dlpack::byte_size(dtype, &spec.shape.dims)?, location)?;
        let mut shape = spec.shape.dims.clone();
        let descriptor = DLTensor {
            data: buffer.ptr.as_ptr().cast(),
            device: dlpack::device(buffer.location)?,
            ndim: i32::try_from(shape.len())?,
            dtype,
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        Ok(Self {
            descriptor,
            _shape: shape,
            buffer,
        })
    }

    pub fn view(&self) -> BorrowedTensor<'_> {
        // SAFETY: owned metadata/storage outlive the borrow; only synchronous
        // engine operations under its mutex may use or mutate this private slot.
        unsafe { BorrowedTensor::new(&self.descriptor) }
    }
}
