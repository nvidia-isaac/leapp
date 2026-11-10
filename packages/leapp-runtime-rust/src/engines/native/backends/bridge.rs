// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private lifecycle and tensor transport shared by the native C++ backends.

mod ffi;

use super::{BackendPlugin, InputRoute};
use crate::{
    bundle::schema::NodeSpec,
    dlpack::{BorrowedTensor, DLTensor, SharedTensor},
    engines::memory::dlpack,
    DataType, MemoryLocation, PortSpec,
};
use anyhow::{ensure, Result};
use std::{
    ffi::{c_void, CString},
    path::PathBuf,
    ptr,
    sync::Arc,
};

pub(in crate::engines::native) struct Model {
    loaded: Arc<Loaded>,
    session: *mut c_void,
    prepared: *mut c_void,
    input_views: Views,
}

// SAFETY: each workspace mutex exclusively owns its execution handles. The bridge selects
// and restores its CUDA device for execution and destruction on any thread.
unsafe impl Send for Model {}

struct Loaded {
    library: ffi::Library,
    model: *mut c_void,
    guarded: bool,
    location: MemoryLocation,
    output_alignment: usize,
}

// SAFETY: the backend permits concurrent shared-model access or requires the
// native graph guard. Execution contexts belong to individual workspaces.
// Session construction reads only immutable loaded metadata; private handles
// belong exclusively to one workspace.
unsafe impl Send for Loaded {}
unsafe impl Sync for Loaded {}

impl Drop for Loaded {
    fn drop(&mut self) {
        // SAFETY: final loaded owner; all sessions/calls have already been dropped.
        unsafe {
            (self.library.api.destroy_model)(self.model);
        }
    }
}

impl Model {
    pub fn load(
        spec: &NodeSpec,
        location: MemoryLocation,
        stream: Option<crate::CudaStream>,
        backend: &str,
        library: &str,
        guarded: bool,
    ) -> Result<Self> {
        let device = match location {
            MemoryLocation::Cpu => -1,
            MemoryLocation::Cuda { device_id } => i32::try_from(device_id)?,
            MemoryLocation::PinnedCpu => unreachable!("workspace execution uses CPU or CUDA"),
        };

        let directory = std::env::var_os("LEAPP_NATIVE_BACKENDS")
            .map(PathBuf::from)
            .unwrap_or_else(|| "/opt/leapp/backends".into());
        let mut loaded = Loaded {
            library: ffi::Library::open(&directory.join(backend).join(library), backend)?,
            model: ptr::null_mut(),
            guarded,
            location,
            output_alignment: if backend == "tensorrt" { 256 } else { 1 },
        };
        let path = CString::new(spec.parameters.model_path.as_os_str().as_encoded_bytes())?;
        let inputs: Vec<_> = spec.inputs.iter().map(port).collect();
        let outputs: Vec<_> = spec.outputs.iter().map(port).collect();
        let api = &loaded.library.api;

        // SAFETY: load borrows valid path/port metadata only for this call. Drop
        // releases any model already created if session creation later fails.
        unsafe {
            api.check((api.load_model)(
                path.as_ptr(),
                device,
                stream.map_or(ptr::null_mut(), crate::CudaStream::as_raw),
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_ptr(),
                outputs.len(),
                &mut loaded.model,
            ))?;
            ffi::nonnull(loaded.model)?;
        }

        Self::with_loaded(Arc::new(loaded))
    }

    fn with_loaded(loaded: Arc<Loaded>) -> Result<Self> {
        let mut value = Self {
            loaded,
            session: ptr::null_mut(),
            prepared: ptr::null_mut(),
            input_views: Views::default(),
        };
        let api = &value.loaded.library.api;
        // SAFETY: the shared loaded handle outlives every private session.
        unsafe {
            api.check((api.create_session)(value.loaded.model, &mut value.session))?;
            ffi::nonnull(value.session)?;
        }
        Ok(value)
    }
}

impl BackendPlugin for Model {
    fn needs_guard(&self) -> bool {
        self.loaded.guarded
    }

    fn new_session(&self) -> Result<Box<dyn BackendPlugin>> {
        Ok(Box::new(Self::with_loaded(self.loaded.clone())?))
    }

    fn can_prepare(&self) -> bool {
        // SAFETY: model is live; this query only reads immutable loaded metadata.
        !unsafe { (self.loaded.library.api.requires_dynamic)(self.loaded.model) }
    }

    fn is_prepared(&self) -> bool {
        !self.prepared.is_null()
    }

    fn input_alignment(&self) -> usize {
        // SAFETY: only live prepared handles reach the bridge query.
        if self.prepared.is_null() {
            0
        } else {
            unsafe { (self.loaded.library.api.input_alignment)(self.prepared) }
        }
    }

    fn output_alignment(&self) -> usize {
        self.loaded.output_alignment
    }

    /// # Safety
    /// Fixed storage remains exclusively owned and alive until this model drops.
    unsafe fn prepare(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        outputs: &[BorrowedTensor<'_>],
    ) -> Result<()> {
        ensure!(
            !self.is_prepared(),
            "native backend model is already prepared"
        );

        self.input_views = Views::new(inputs, self.loaded.location)?;
        let output_views = Views::new(outputs, self.loaded.location)?;
        let api = &self.loaded.library.api;

        // SAFETY: the backend copies temporary output metadata; input metadata
        // and all fixed workspace storage outlive the prepared call.
        unsafe {
            api.check((api.prepare)(
                self.session,
                self.input_views.values.as_ptr(),
                self.input_views.values.len(),
                output_views.values.as_ptr(),
                output_views.values.len(),
                &mut self.prepared,
            ))?;
        }

        ffi::nonnull(self.prepared)?;

        Ok(())
    }

    unsafe fn run(
        &mut self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        routes: &[InputRoute],
        outputs: &[(usize, DLTensor)],
    ) -> Result<()> {
        let outputs = unsafe { output_bindings(outputs, self.loaded.location) }?;
        let views = BorrowedViews {
            views: &mut self.input_views,
            routes,
        };
        for route in routes {
            let descriptor = inputs[route.boundary].1.descriptor();
            views.views.values[route.position].data =
                descriptor.data as usize + descriptor.byte_offset as usize;
        }
        let api = &self.loaded.library.api;
        // SAFETY: owners remain borrowed until completion, including failures.
        // The backend restores bindings before releasing the borrow.
        unsafe {
            api.check((api.run)(
                self.prepared,
                views.views.values.as_ptr(),
                views.views.values.len(),
                routes.as_ptr(),
                routes.len(),
                outputs.as_ptr(),
                outputs.len(),
            ))?;
        }
        Ok(())
    }

    unsafe fn run_dynamic(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        bindings: &[(usize, DLTensor)],
        transfers: &mut crate::engines::native::Transfers,
    ) -> Result<Vec<Option<SharedTensor>>> {
        let targets = unsafe { output_bindings(bindings, self.loaded.location) }?;
        let inputs = inputs
            .iter()
            .map(|&value| {
                // SAFETY: the engine validated and routed every input before dispatch.
                unsafe { view_validated(value.descriptor(), self.loaded.location) }
            })
            .collect::<Result<Vec<_>>>()?;
        let api = &self.loaded.library.api;
        let mut output = Outputs {
            value: ptr::null_mut(),
            api,
        };

        // SAFETY: input storage is retained by the caller. The output guard
        // releases foreign buffers after independent copies, including errors.
        unsafe {
            api.check((api.run_dynamic)(
                self.session,
                inputs.as_ptr(),
                inputs.len(),
                targets.as_ptr(),
                targets.len(),
                &mut output.value,
            ))?;
            ffi::nonnull(output.value)?;

            let mut count = 0;
            let values = (api.views)(output.value, &mut count);
            ensure!(
                count <= isize::MAX as usize / std::mem::size_of::<ffi::View>(),
                "native backend output count overflow"
            );
            ensure!(
                count == 0 || !values.is_null(),
                "native backend returned null output metadata"
            );

            let values = if count == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(values, count)
            };

            values
                .iter()
                .enumerate()
                .map(|(position, value)| {
                    if bindings.iter().any(|output| output.0 == position) {
                        return Ok(None);
                    }
                    let location = if value.device == -1 {
                        MemoryLocation::Cpu
                    } else {
                        MemoryLocation::Cuda {
                            device_id: u32::try_from(value.device)?,
                        }
                    };

                    let descriptor = DLTensor {
                        data: value.data as *mut c_void,
                        device: dlpack::device(location)?,
                        ndim: i32::try_from(value.port.rank)?,
                        dtype: dlpack::dl_type(native_type(value.port.dtype)?),
                        shape: value.port.shape.cast_mut(),
                        strides: ptr::null_mut(),
                        byte_offset: 0,
                    };

                    let borrowed = BorrowedTensor::new(&descriptor);
                    let len = dlpack::layout(borrowed)?.1;
                    ensure!(
                        len == value.bytes,
                        "native backend output byte size mismatch"
                    );
                    // SAFETY: this SDK boundary validated the complete layout;
                    // copy into independent execution-device storage without rechecking it.
                    transfers
                        .copy_validated(borrowed, self.loaded.location, len)
                        .map(Some)
                })
                .collect()
        }
    }

    #[cfg(test)]
    fn execution_identity(&self) -> Option<usize> {
        Some(self.session as usize)
    }

    #[cfg(test)]
    fn loaded_identity(&self) -> usize {
        self.loaded.model as usize
    }

    #[cfg(test)]
    fn prepared_identity(&self) -> Option<Vec<usize>> {
        self.is_prepared().then(|| {
            vec![
                self.loaded.model as usize,
                self.session as usize,
                self.prepared as usize,
            ]
        })
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: nullable, exclusive handles belong to this still-loaded library.
        // Destruction drains any pending work before releasing contexts/storage.
        unsafe {
            let api = &self.loaded.library.api;
            (api.destroy_call)(self.prepared);
            (api.destroy_session)(self.session);
        }
    }
}

struct Outputs<'a> {
    value: *mut c_void,
    api: &'a ffi::Api,
}

/// Owned fixed metadata for initialization and invocation-scoped input binding.
/// Prepare this once instead of revalidating each borrowed input.
#[derive(Default)]
struct Views {
    values: Vec<ffi::View>,
    _shapes: Vec<Vec<i64>>,
}

impl Views {
    fn new(tensors: &[BorrowedTensor<'_>], location: MemoryLocation) -> Result<Self> {
        let mut values = tensors
            .iter()
            .map(|&tensor| view(tensor, location))
            .collect::<Result<Vec<_>>>()?;
        let shapes: Vec<_> = values
            .iter()
            .map(|value| {
                if value.port.rank == 0 {
                    Vec::new()
                } else {
                    // SAFETY: view checked the descriptor's complete layout above.
                    unsafe { std::slice::from_raw_parts(value.port.shape, value.port.rank) }
                        .to_vec()
                }
            })
            .collect();
        for (value, shape) in values.iter_mut().zip(&shapes) {
            value.port.shape = shape.as_ptr();
        }
        Ok(Self {
            values,
            _shapes: shapes,
        })
    }
}

struct BorrowedViews<'a> {
    views: &'a mut Views,
    routes: &'a [InputRoute],
}

impl Drop for BorrowedViews<'_> {
    fn drop(&mut self) {
        for route in self.routes {
            // C++ restored its prepared bindings. Keep no caller addresses in Rust.
            self.views.values[route.position].data = 0;
        }
    }
}

impl Drop for Outputs<'_> {
    fn drop(&mut self) {
        // SAFETY: this guard owns the output handle and outlives its borrowed views.
        unsafe { (self.api.destroy_outputs)(self.value) };
    }
}

fn port(spec: &PortSpec) -> ffi::Port {
    ffi::Port {
        name: spec.name.as_ptr().cast(),
        name_size: spec.name.len(),
        dtype: dtype(spec.dtype),
        shape: spec.shape.as_ptr(),
        rank: spec.shape.len(),
    }
}

/// # Safety
/// Output metadata and execution placement must be validated and retained through completion.
unsafe fn output_bindings(
    outputs: &[(usize, DLTensor)],
    location: MemoryLocation,
) -> Result<Vec<ffi::OutputBinding>> {
    outputs
        .iter()
        .map(|output| {
            Ok(ffi::OutputBinding {
                position: output.0,
                view: unsafe { view_validated(&output.1, location) }?,
            })
        })
        .collect()
}

fn view(value: BorrowedTensor<'_>, location: MemoryLocation) -> Result<ffi::View> {
    dlpack::layout(value)?;
    ensure!(
        dlpack::location(value.descriptor().device)? == location,
        "native backend workspace device mismatch"
    );
    // SAFETY: initialization checked the complete layout and placement above.
    unsafe { view_validated(value.descriptor(), location) }
}

/// # Safety
/// The tensor metadata and execution placement have already been validated.
unsafe fn view_validated(descriptor: &DLTensor, location: MemoryLocation) -> Result<ffi::View> {
    // SAFETY: the caller retains the validated input through backend completion.
    let (shape, bytes) = unsafe { dlpack::layout_unchecked(*descriptor) };
    Ok(ffi::View {
        port: ffi::Port {
            name: ptr::null(),
            name_size: 0,
            dtype: dtype(dlpack::data_type(descriptor.dtype)?),
            shape: shape.as_ptr(),
            rank: shape.len(),
        },
        data: descriptor.data as usize + descriptor.byte_offset as usize,
        bytes,
        device: match location {
            MemoryLocation::Cuda { device_id } => device_id as i32,
            _ => -1,
        },
    })
}

fn dtype(value: DataType) -> u8 {
    match value {
        DataType::Bool => 0,
        DataType::Uint8 => 1,
        DataType::Uint16 => 2,
        DataType::Uint32 => 3,
        DataType::Uint64 => 4,
        DataType::Int8 => 5,
        DataType::Int16 => 6,
        DataType::Int32 => 7,
        DataType::Int64 => 8,
        DataType::Float16 => 9,
        DataType::Float32 => 10,
        DataType::Float64 => 11,
        DataType::Bfloat16 => 12,
    }
}

fn native_type(value: u8) -> Result<DataType> {
    Ok(match value {
        0 => DataType::Bool,
        1 => DataType::Uint8,
        2 => DataType::Uint16,
        3 => DataType::Uint32,
        4 => DataType::Uint64,
        5 => DataType::Int8,
        6 => DataType::Int16,
        7 => DataType::Int32,
        8 => DataType::Int64,
        9 => DataType::Float16,
        10 => DataType::Float32,
        11 => DataType::Float64,
        12 => DataType::Bfloat16,
        _ => anyhow::bail!("unsupported native backend output dtype {value}"),
    })
}
