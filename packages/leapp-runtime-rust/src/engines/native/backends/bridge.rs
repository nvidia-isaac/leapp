// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private lifecycle and tensor transport shared by the native C++ backends.

mod ffi;

use super::BackendPlugin;
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
    location: MemoryLocation,
}

// SAFETY: each workspace mutex exclusively owns its execution handles. The bridge selects
// and restores its CUDA device for execution and destruction on any thread.
unsafe impl Send for Model {}

struct Loaded {
    library: ffi::Library,
    model: *mut c_void,
    guarded: bool,
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
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_ptr(),
                outputs.len(),
                &mut loaded.model,
            ))?;
            ffi::nonnull(loaded.model)?;
        }

        Self::with_loaded(Arc::new(loaded), location)
    }

    fn with_loaded(loaded: Arc<Loaded>, location: MemoryLocation) -> Result<Self> {
        let mut value = Self {
            loaded,
            session: ptr::null_mut(),
            prepared: ptr::null_mut(),
            location,
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
        Ok(Box::new(Self::with_loaded(
            self.loaded.clone(),
            self.location,
        )?))
    }

    fn can_prepare(&self) -> bool {
        // SAFETY: model is live; this query only reads immutable loaded metadata.
        !unsafe { (self.loaded.library.api.requires_dynamic)(self.loaded.model) }
    }

    fn is_prepared(&self) -> bool {
        !self.prepared.is_null()
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

        let inputs = inputs
            .iter()
            .map(|&value| view(value, self.location))
            .collect::<Result<Vec<_>>>()?;
        let outputs = outputs
            .iter()
            .map(|&value| view(value, self.location))
            .collect::<Result<Vec<_>>>()?;
        let api = &self.loaded.library.api;

        // SAFETY: metadata is temporary; the bridge caches the fixed storage
        // addresses guaranteed by the caller and configures its context once.
        unsafe {
            api.check((api.prepare)(
                self.session,
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_ptr(),
                outputs.len(),
                &mut self.prepared,
            ))?;
        }

        ffi::nonnull(self.prepared)?;

        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        ensure!(self.is_prepared(), "unprepared native backend model");

        // SAFETY: prepared handles and workspace buffers remain exclusively
        // owned. The bridge drains CUDA work on success and failure.
        unsafe {
            self.loaded
                .library
                .api
                .check((self.loaded.library.api.run)(self.prepared))
        }
    }

    fn run_dynamic(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        transfers: &mut crate::engines::native::Transfers,
    ) -> Result<Vec<SharedTensor>> {
        let inputs = inputs
            .iter()
            .map(|&value| view(value, self.location))
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
                .map(|value| {
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
                    ensure!(
                        dlpack::layout(borrowed)?.1 == value.bytes,
                        "native backend output byte size mismatch"
                    );

                    transfers.copy(borrowed, self.location)
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

fn view(value: BorrowedTensor<'_>, location: MemoryLocation) -> Result<ffi::View> {
    let descriptor = value.descriptor();
    ensure!(
        dlpack::location(descriptor.device)? == location,
        "native backend workspace device mismatch"
    );

    let (shape, bytes) = dlpack::layout(value)?;

    Ok(ffi::View {
        port: ffi::Port {
            name: ptr::null(),
            name_size: 0,
            dtype: dtype(dlpack::data_type(descriptor.dtype)?),
            shape: descriptor.shape,
            rank: shape.len(),
        },
        data: descriptor.data as usize + usize::try_from(descriptor.byte_offset)?,
        bytes,
        device: match location {
            MemoryLocation::Cpu | MemoryLocation::PinnedCpu => -1,
            MemoryLocation::Cuda { device_id } => i32::try_from(device_id)?,
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
