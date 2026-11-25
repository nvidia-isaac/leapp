// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Temporary workaround for an ONNX Runtime TensorRT provider bug with integer
//! shape inputs of rank two or more. Everything for it, including its test, is
//! in this file; `onnx.rs` calls it from lines marked
//! `ORT TensorRT shape-input workaround`.
//!
//! # Bug
//!
//! TensorRT treats an integer input as a shape tensor when its values set other
//! tensors' shapes, such as a `[1, 3]` `image_grid_thw`. ORT's provider takes the
//! value count from the first dimension instead of the element count
//! (`tensorrt_execution_provider.cc` lines 638, 678 and 914 in ORT 1.24.4), so
//! the engine build fails with "number of elements in shape tensor ... to be 1",
//! or a range profile computes corrupt shapes. With ORT 1.24.4 and TensorRT
//! 10.16.1, only inputs whose declared dims are all static became shape tensors;
//! inputs with a dynamic dimension work without this workaround.
//!
//! # Workaround
//!
//! On TensorRT nodes, each int32/int64 input with a fixed, nonempty shape of
//! rank two or more is replaced in the in-memory graph by a rank-1 input
//! (`PREFIX<name>`) plus a Reshape back to `<name>`, and bound flat without
//! copying. Its `trt_profile_*_shapes` entries are renamed, and an entry equal
//! to the declared dims is redundant and dropped. Fixed integer inputs that
//! aren't shape tensors are rewritten too, which only costs an engine rebuild.
//!
//! # Limits
//!
//! - Triton's ONNX Runtime backend has the same bug and isn't covered.
//! - ORT also swaps OPT and MAX for int64 shape-tensor profiles with TensorRT
//!   newer than 10.10 (lines 546-547, 762-763); range profiles need OPT = MAX.
//! - Inferred profiles widen each value's range independently (lines 716-740).
//!   Values no earlier input dominates, such as `[1, 3, 8]` then `[1, 4, 4]`,
//!   can make TensorRT reject the profile ("kOPT values ... violate shape
//!   constraints"); use an explicit range profile for those inputs.
//! - Inputs the bundle declares bounded but the model declares static aren't
//!   rewritten.
//!
//! # Removal
//!
//! Once every supported ORT TensorRT provider counts shape-tensor elements
//! correctly, delete this file and the marked lines in `onnx.rs`, then check a
//! model with a static `[1, 3]` shape input under TensorRT without a profile.

use crate::{bundle::schema::NodeSpec, dlpack::DLTensor, DataType};
use anyhow::{ensure, Context, Result};
use ort::{
    editor::{Graph, Model, Node, Opset, ONNX_DOMAIN},
    memory::Allocator,
    session::{builder::SessionBuilder, Session},
    value::{Outlet, Shape, SymbolicDimensions, Tensor, ValueType},
    AsPointer,
};
use std::{collections::HashMap, fs::File, io::Read, path::Path, ptr};

/// Reserved prefix for rewritten inputs and their helper values.
const PREFIX: &str = "__leapp_trt_shape_input_v1_";

const PROFILE_KEYS: [&str; 3] = [
    "trt_profile_min_shapes",
    "trt_profile_opt_shapes",
    "trt_profile_max_shapes",
];

fn private(name: &str) -> String {
    format!("{PREFIX}{name}")
}

/// Rewritten inputs of one ONNX node; empty when the workaround doesn't apply.
#[derive(Default)]
pub(super) struct ShapeInputs {
    /// Element count by bundle input position, for rewritten inputs.
    counts: Vec<Option<i64>>,
    /// Declared dims by public input name, for rewritten inputs.
    dims: HashMap<String, Vec<i64>>,
}

impl ShapeInputs {
    pub fn select(model: &NodeSpec, tensorrt: bool) -> Self {
        let mut selected = Self::default();
        if !tensorrt {
            return selected;
        }
        for (port, shape) in model.input_ports() {
            let count = (matches!(port.dtype, DataType::Int32 | DataType::Int64)
                && shape.dims.len() > 1)
                .then(|| {
                    shape.dims.iter().try_fold(1i64, |count, &dim| {
                        (dim > 0).then(|| count.checked_mul(dim)).flatten()
                    })
                })
                .flatten();
            if count.is_some() {
                selected.dims.insert(port.name.clone(), shape.dims.clone());
            }
            selected.counts.push(count);
        }
        selected
    }

    /// Element count of a rewritten input, which is bound as a rank-1 tensor.
    pub fn count(&self, position: usize) -> Option<i64> {
        self.counts.get(position).copied().flatten()
    }

    /// Calls `wrap` with the input presented flat when it is rewritten. Inputs
    /// reaching the backend are validated and contiguous, and ORT copies the
    /// shape when wrapping, so the flat shape only needs to outlive this call.
    pub fn with_input<R>(
        &self,
        position: usize,
        descriptor: &DLTensor,
        wrap: impl FnOnce(&DLTensor) -> R,
    ) -> R {
        let Some(count) = self.count(position) else {
            return wrap(descriptor);
        };
        let mut shape = [count];
        let mut flat = *descriptor;
        flat.ndim = 1;
        flat.shape = shape.as_mut_ptr();
        flat.strides = ptr::null_mut();
        wrap(&flat)
    }

    /// Points binding names of rewritten inputs at their private graph inputs.
    pub fn rename(&self, names: &mut [String]) {
        for (position, name) in names.iter_mut().enumerate() {
            if self.count(position).is_some() {
                *name = private(name);
            }
        }
    }

    /// Renames or drops profile entries for rewritten inputs, as described in
    /// the module documentation. A profile left empty is removed.
    pub fn provider_options(
        &self,
        mut options: HashMap<String, String>,
    ) -> HashMap<String, String> {
        if self.dims.is_empty() {
            return options;
        }
        for key in PROFILE_KEYS {
            let Some(value) = options.get(key).map(|value| self.profile(value)) else {
                continue;
            };
            if value.is_empty() {
                options.remove(key);
            } else {
                options.insert(key.to_owned(), value);
            }
        }
        options
    }

    fn profile(&self, value: &str) -> String {
        value
            .split(',')
            .filter_map(|entry| {
                let Some((name, values)) = entry.rsplit_once(':') else {
                    return Some(entry.to_owned());
                };
                let Some(dims) = self.dims.get(name) else {
                    return Some(entry.to_owned());
                };
                let declared = dims.iter().map(i64::to_string).collect::<Vec<_>>();
                (values != declared.join("x")).then(|| format!("{}:{values}", private(name)))
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Loads the model, rewriting selected inputs in memory when there are any.
    /// ort keeps the original input metadata on the session, so port checks
    /// still see public names; `rename` maps the binding names.
    pub fn commit(&self, builder: &mut SessionBuilder, path: &Path) -> Result<Session> {
        if self.dims.is_empty() {
            return Ok(builder.commit_from_file(path)?);
        }
        check_names(path)?;
        let mut session = builder
            .edit_from_file(path)
            .context("TensorRT shape-input workaround: open the ONNX model for editing")?;
        let mut opset = 0;
        // SAFETY: the editor owns the open session; the result is a scalar.
        unsafe {
            ort::Error::result_from_status((ort::editor::editor_api()?.SessionGetOpsetForDomain)(
                session.ptr(),
                c"".as_ptr(),
                &mut opset,
            ))?;
        }
        ensure!(
            opset >= 5,
            "TensorRT shape-input workaround requires ONNX opset 5 or newer"
        );
        let mut graph = Graph::new()?;
        let inputs = session
            .inputs()
            .iter()
            .map(|input| match (self.dims.get(input.name()), input.dtype()) {
                (Some(dims), ValueType::Tensor { ty, .. }) => Outlet::new(
                    private(input.name()),
                    ValueType::Tensor {
                        ty: *ty,
                        shape: Shape::new([dims.iter().product::<i64>()]),
                        dimension_symbols: SymbolicDimensions::empty(1),
                    },
                ),
                _ => Outlet::new(input.name(), input.dtype().clone()),
            })
            .collect::<Vec<_>>();
        graph.set_inputs(inputs)?;
        for (name, dims) in &self.dims {
            let flat = private(name);
            let target = format!("{flat}_shape");
            let mut shape = Tensor::<i64>::new(&Allocator::default(), [dims.len() as i64])?;
            shape.extract_tensor_mut().1.copy_from_slice(dims);
            graph.add_initializer(&target, shape, false)?;
            graph.add_node(Node::new(
                "Reshape",
                ONNX_DOMAIN,
                format!("{flat}_restore"),
                [&flat, &target],
                [name],
                [],
            )?)?;
        }
        let mut update = Model::new([Opset::new(ONNX_DOMAIN, opset as u32)?])?;
        update.add_graph(graph)?;
        session.apply_model(&update)?;
        Ok(session.into_session()?)
    }
}

/// The model editor can replace values with the same name, so reject models
/// that already use the reserved prefix. Scans the model file, not external
/// tensor data, in bounded memory, including across read boundaries.
fn check_names(path: &Path) -> Result<()> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0; 65536 + PREFIX.len()];
    let mut retained = 0;
    loop {
        let read = file.read(&mut buffer[retained..])?;
        if read == 0 {
            return Ok(());
        }
        let length = retained + read;
        ensure!(
            !buffer[..length]
                .windows(PREFIX.len())
                .any(|window| window == PREFIX.as_bytes()),
            "ONNX model uses the reserved TensorRT shape-input prefix {PREFIX}"
        );
        retained = (PREFIX.len() - 1).min(length);
        buffer.copy_within(length - retained..length, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_and_bindings_follow_rewritten_inputs() {
        let inputs = ShapeInputs {
            counts: vec![Some(3), None, Some(101)],
            dims: HashMap::from([
                ("grid".into(), vec![1, 3]),
                ("ids".into(), vec![1, 101]),
                ("scope:grid".into(), vec![2, 3]),
            ]),
        };
        let options = HashMap::from([
            (
                "trt_profile_min_shapes".into(),
                "data:12,grid:1x16x22,ids:1x101,scope:grid:1x2x6x1x2x6".into(),
            ),
            ("trt_profile_opt_shapes".into(), "ids:1x101".into()),
            ("trt_fp16_enable".into(), "1".into()),
        ]);
        let options = inputs.provider_options(options);
        assert_eq!(
            options["trt_profile_min_shapes"],
            format!("data:12,{PREFIX}grid:1x16x22,{PREFIX}scope:grid:1x2x6x1x2x6")
        );
        assert!(!options.contains_key("trt_profile_opt_shapes"));
        assert_eq!(options["trt_fp16_enable"], "1");

        let mut names = vec!["grid".to_owned(), "data".to_owned(), "ids".to_owned()];
        inputs.rename(&mut names);
        assert_eq!(names, [private("grid"), "data".into(), private("ids")]);
        assert_eq!(
            ShapeInputs::default().provider_options(options.clone()),
            options
        );
    }
}
