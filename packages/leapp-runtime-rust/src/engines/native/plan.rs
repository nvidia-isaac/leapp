// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Indexed routing and stable workspace storage. Feedback stays in Session.

use crate::{
    dlpack::{BorrowedTensor, DLTensor, SharedTensor},
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
    pub shape: Vec<i64>,
}

impl From<&PortSpec> for TensorSpec {
    fn from(port: &PortSpec) -> Self {
        Self {
            name: port.name.clone(),
            dtype: port.dtype,
            shape: port.shape.clone(),
        }
    }
}

pub(super) struct Call {
    pub name: String,
    pub model: usize,
    pub inputs: Vec<usize>,
    pub input_specs: Vec<TensorSpec>,
    pub outputs: Vec<usize>,
}

pub(super) struct Plan {
    pub calls: Vec<Call>,
    pub inputs: IndexMap<String, usize>,
    pub outputs: IndexMap<String, usize>,
    pub slots: Vec<Arc<TensorSpec>>,
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
            outputs: IndexMap::new(),
            slots: Vec::new(),
        };
        let mut outputs = HashMap::new();
        for (name, model) in models {
            for spec in &model.outputs {
                let id = plan.slots.len();
                plan.slots.push(Arc::new(spec.into()));
                outputs.insert(port_ref(name, spec), id);
            }
        }

        for model_id in order {
            let (name, model) = models.get_index(model_id).unwrap();
            let mut inputs = Vec::new();
            for spec in &model.inputs {
                let port = port_ref(name, spec);
                let id = match graph.sources.get(&port) {
                    Some(InputSource::Forward(source)) => outputs[source],
                    // Both private defaults and feedback are supplied by Session.
                    _ => {
                        let id = plan.slots.len();
                        plan.slots.push(Arc::new(spec.into()));
                        plan.inputs.insert(port.name(), id);
                        id
                    }
                };
                inputs.push(id);
            }

            plan.calls.push(Call {
                name: name.clone(),
                model: model_id,
                inputs,
                input_specs: model.inputs.iter().map(Into::into).collect(),
                outputs: model
                    .outputs
                    .iter()
                    .map(|s| outputs[&port_ref(name, s)])
                    .collect(),
            });
        }

        let pipeline = &bundle.0.description.pipeline;
        for port in bundle
            .pipeline_outputs()
            .iter()
            .chain(pipeline.feedback_flow.keys())
        {
            plan.outputs.insert(port.name(), outputs[port]);
        }

        // Match Triton's private terminal publication and branch execution.
        for (name, model) in models {
            for spec in &model.outputs {
                let port = port_ref(name, spec);
                if pipeline.data_flow.get(&port).is_none_or(Vec::is_empty) {
                    plan.outputs.insert(port.name(), outputs[&port]);
                }
            }
        }

        Ok(plan)
    }
}

pub(super) fn matches(spec: &TensorSpec, value: BorrowedTensor<'_>) -> Result<()> {
    let (shape, _) = dlpack::layout(value)?;
    ensure!(
        dlpack::data_type(value.descriptor().dtype)? == spec.dtype
            && shape.len() == spec.shape.len()
            && shape
                .iter()
                .zip(&spec.shape)
                .all(|(&actual, &expected)| expected == -1 || expected == actual),
        "port {} does not match declared dtype/shape",
        spec.name
    );
    Ok(())
}

pub(super) struct Slot {
    pub spec: Arc<TensorSpec>,
    pub fixed: Option<FixedTensor>,
    dynamic: Option<SharedTensor>,
    location: MemoryLocation,
}

impl Slot {
    pub fn new(spec: Arc<TensorSpec>, location: MemoryLocation) -> Result<Self> {
        let fixed = if spec.shape.contains(&-1) {
            None
        } else {
            Some(FixedTensor::new(&spec, location)?)
        };
        Ok(Self {
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

    pub fn assign(
        &mut self,
        value: BorrowedTensor<'_>,
        transfers: &mut super::Transfers,
    ) -> Result<()> {
        matches(&self.spec, value)?;
        if let Some(fixed) = &mut self.fixed {
            // SAFETY: the workspace mutex exclusively owns this disjoint workspace slot.
            unsafe {
                transfers.write(value, &fixed.buffer)?;
            }
        } else {
            self.dynamic = Some(transfers.copy(value, self.location)?);
        }
        Ok(())
    }

    pub fn clear_dynamic(&mut self) {
        self.dynamic = None;
    }
}

/// Metadata and addresses remain stable while prepared backend bindings borrow them.
pub(super) struct FixedTensor {
    descriptor: DLTensor,
    _shape: Vec<i64>,
    buffer: Buffer,
}

// SAFETY: allocations and metadata are owned, accesses are exclusive under the
// workspace mutex, and moving the struct does not move its heap allocations.
unsafe impl Send for FixedTensor {}
impl FixedTensor {
    fn new(spec: &TensorSpec, location: MemoryLocation) -> Result<Self> {
        let dtype = dlpack::dl_type(spec.dtype);
        let buffer = Buffer::allocate(dlpack::byte_size(dtype, &spec.shape)?, location)?;
        let mut shape = spec.shape.clone();
        let descriptor = DLTensor {
            data: buffer.ptr.as_ptr().cast(),
            device: dlpack::device(location)?,
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
