// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust host for the documented C++ BackendPlugin interface.

mod ffi;

use super::{BackendPlugin, InputRoute};
use crate::{
    bundle::schema::NodeSpec,
    dlpack::{BorrowedTensor, DLTensor, SharedTensor},
    engines::memory::dlpack,
    MemoryLocation, PortSpec,
};
use anyhow::{ensure, Result};
use std::{ffi::c_void, path::PathBuf, ptr, sync::Arc};

pub(super) fn load(
    spec: &NodeSpec,
    location: MemoryLocation,
    stream: Option<crate::CudaStream>,
    backend: &str,
    library: &str,
) -> Result<Box<dyn BackendPlugin>> {
    let directory = std::env::var_os("LEAPP_NATIVE_BACKENDS")
        .map(PathBuf::from)
        .unwrap_or_else(|| "/opt/leapp/backends".into());
    Ok(Box::new(Model::load_library(
        spec,
        location,
        stream,
        ffi::Library::open(&directory.join(backend).join(library), backend)?,
    )?))
}

struct Model {
    library: Arc<ffi::Library>,
    instance: *mut c_void,
    location: MemoryLocation,
    output_count: usize,
}

// SAFETY: each workspace exclusively owns its instance. Plugins permit moving
// execution/destruction between threads and synchronize shared model access.
unsafe impl Send for Model {}

impl Model {
    fn load_library(
        spec: &NodeSpec,
        location: MemoryLocation,
        stream: Option<crate::CudaStream>,
        library: ffi::Library,
    ) -> Result<Self> {
        let mut value = Self {
            library: Arc::new(library),
            instance: ptr::null_mut(),
            location,
            output_count: spec.outputs.len(),
        };
        let device = match location {
            MemoryLocation::Cpu => -1,
            MemoryLocation::Cuda { device_id } => i32::try_from(device_id)?,
            MemoryLocation::PinnedCpu => unreachable!("workspace execution uses CPU or CUDA"),
        };
        let path = spec.parameters.model_path.as_os_str().as_encoded_bytes();
        let inputs: Vec<_> = spec.inputs.iter().map(port).collect();
        let outputs: Vec<_> = spec.outputs.iter().map(port).collect();
        let args = ffi::LoadArgs {
            path: path.as_ptr().cast(),
            path_size: path.len(),
            device,
            stream: stream.map_or(ptr::null_mut(), crate::CudaStream::as_raw),
            inputs: inputs.as_ptr(),
            input_count: inputs.len(),
            outputs: outputs.as_ptr(),
            output_count: outputs.len(),
        };
        // SAFETY: metadata is borrowed for this call. Drop releases a partially
        // published instance before its still-loaded library.
        unsafe {
            (value.library.api.load.unwrap())(&args, &mut value.instance).check()?;
        }
        ffi::nonnull(value.instance)?;
        ensure!(!value.is_prepared(), "loaded backend must start unprepared");
        Ok(value)
    }
}

impl BackendPlugin for Model {
    fn new_session(&self) -> Result<Box<dyn BackendPlugin>> {
        let mut peer = Self {
            library: self.library.clone(),
            instance: ptr::null_mut(),
            location: self.location,
            output_count: self.output_count,
        };
        // SAFETY: peer construction reads immutable shared metadata or synchronizes
        // with execution on other peers internally.
        unsafe {
            (self.library.api.new_session.unwrap())(self.instance, &mut peer.instance).check()?;
        }
        ffi::nonnull(peer.instance)?;
        ensure!(
            !peer.is_prepared(),
            "new backend session must start unprepared"
        );
        ensure!(
            peer.needs_guard() == self.needs_guard(),
            "backend changed its peer guard policy"
        );
        ensure!(
            peer.output_alignment() == self.output_alignment(),
            "backend changed its peer output alignment"
        );
        Ok(Box::new(peer))
    }

    fn needs_guard(&self) -> bool {
        // SAFETY: query a live instance without modifying shared state.
        unsafe { (self.library.api.needs_guard.unwrap())(self.instance) }
    }

    fn can_prepare(&self) -> bool {
        // SAFETY: query immutable loaded metadata.
        unsafe { (self.library.api.can_prepare.unwrap())(self.instance) }
    }

    fn is_prepared(&self) -> bool {
        // SAFETY: this instance is exclusively owned by its workspace.
        unsafe { (self.library.api.is_prepared.unwrap())(self.instance) }
    }

    fn input_alignment(&self) -> usize {
        // SAFETY: live instance, queried only during initialization.
        unsafe { (self.library.api.input_alignment.unwrap())(self.instance) }
    }

    fn output_alignment(&self) -> usize {
        // SAFETY: live instance with a stable declared output alignment.
        unsafe { (self.library.api.output_alignment.unwrap())(self.instance) }
    }

    unsafe fn prepare(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        outputs: &[BorrowedTensor<'_>],
    ) -> Result<()> {
        ensure!(
            !self.is_prepared(),
            "native backend model is already prepared"
        );
        let inputs: Vec<_> = inputs.iter().map(|value| *value.descriptor()).collect();
        let outputs: Vec<_> = outputs.iter().map(|value| *value.descriptor()).collect();
        // SAFETY: fixed storage remains alive through instance destruction,
        // including failed preparation. The plugin copies retained metadata.
        unsafe {
            (self.library.api.prepare.unwrap())(
                self.instance,
                inputs.as_ptr(),
                inputs.len(),
                outputs.as_ptr(),
                outputs.len(),
            )
            .check()?;
        }
        ensure!(
            self.is_prepared(),
            "successful preparation left backend unprepared"
        );
        Ok(())
    }

    unsafe fn run(
        &mut self,
        inputs: &[(&str, BorrowedTensor<'_>)],
        routes: &[InputRoute],
        outputs: &[(usize, DLTensor)],
    ) -> Result<()> {
        let inputs: Vec<_> = inputs
            .iter()
            .map(|(name, value)| ffi::NamedTensor {
                name: name.as_ptr().cast(),
                name_size: name.len(),
                tensor: *value.descriptor(),
            })
            .collect();
        let outputs = output_bindings(outputs);
        // SAFETY: validated owners live through synchronous completion. The plugin
        // restores fixed bindings on success, errors and unwinding.
        unsafe {
            (self.library.api.run.unwrap())(
                self.instance,
                inputs.as_ptr(),
                inputs.len(),
                routes.as_ptr(),
                routes.len(),
                outputs.as_ptr(),
                outputs.len(),
            )
            .check()
        }
    }

    unsafe fn run_dynamic(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        bindings: &[(usize, DLTensor)],
        transfers: &mut crate::engines::native::Transfers,
    ) -> Result<Vec<Option<SharedTensor>>> {
        let inputs: Vec<_> = inputs.iter().map(|value| *value.descriptor()).collect();
        let targets = output_bindings(bindings);
        let mut context = CopyContext {
            transfers,
            location: self.location,
        };
        let host = ffi::Host {
            context: (&mut context as *mut CopyContext<'_>).cast(),
            copy,
        };
        let api = &self.library.api;
        let mut outputs = Outputs {
            value: ptr::null_mut(),
            api,
        };
        // SAFETY: all metadata, owners, targets and host services remain alive.
        // Synchronous host callbacks only use the exclusively borrowed transfers.
        unsafe {
            (api.run_dynamic.unwrap())(
                self.instance,
                inputs.as_ptr(),
                inputs.len(),
                targets.as_ptr(),
                targets.len(),
                &host,
                &mut outputs.value,
            )
            .check()?;
            ffi::nonnull(outputs.value)?;
            ensure!(
                (api.output_count.unwrap())(outputs.value) == self.output_count,
                "native backend output count mismatch"
            );
            let mut values = Vec::with_capacity(self.output_count);
            for position in 0..self.output_count {
                let mut handle = (api.take_output.unwrap())(outputs.value, position);
                if bindings.iter().any(|output| output.0 == position) {
                    ensure!(
                        handle.value.is_null(),
                        "backend returned an owned result for a supplied output"
                    );
                    values.push(None);
                } else {
                    ffi::nonnull(handle.value)?;
                    ensure!(
                        handle.destroy.is_some_and(|destroy| std::ptr::fn_addr_eq(
                            destroy,
                            destroy_tensor as unsafe extern "C" fn(*mut c_void)
                        )),
                        "backend returned a foreign tensor owner"
                    );
                    // Only Transfers creates populated C++ owners. Consume this
                    // host-issued Box exactly once, without copying its storage.
                    let tensor = *Box::from_raw(handle.value.cast::<SharedTensor>());
                    handle.value = ptr::null_mut();
                    values.push(Some(tensor));
                }
            }
            Ok(values)
        }
    }

    #[cfg(test)]
    fn loaded_identity(&self) -> usize {
        Arc::as_ptr(&self.library) as usize
    }

    #[cfg(test)]
    fn prepared_identity(&self) -> Option<Vec<usize>> {
        self.is_prepared().then(|| vec![self.instance as usize])
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // SAFETY: the adapter accepts null after failures and drains work before
        // releasing borrowed workspace storage; the library is still loaded.
        unsafe { (self.library.api.destroy.unwrap())(self.instance) };
    }
}

struct Outputs<'a> {
    value: *mut c_void,
    api: &'a ffi::Api,
}

impl Drop for Outputs<'_> {
    fn drop(&mut self) {
        // SAFETY: own the result container and every tensor not yet consumed.
        unsafe { (self.api.destroy_outputs.unwrap())(self.value) };
    }
}

struct CopyContext<'a> {
    transfers: &'a mut crate::engines::native::Transfers,
    location: MemoryLocation,
}

unsafe extern "C" fn destroy_tensor(value: *mut c_void) {
    // SAFETY: exclusively owned host-issued Box, released or adopted once.
    unsafe { drop(Box::from_raw(value.cast::<SharedTensor>())) };
}

unsafe extern "C" fn destroy_message(value: *mut c_void) {
    // SAFETY: diagnostic created in copy, released by the C++ status guard.
    unsafe { drop(Box::from_raw(value.cast::<String>())) };
}

unsafe extern "C" fn copy(
    context: *mut c_void,
    source: *const DLTensor,
    output: *mut ffi::TensorHandle,
) -> ffi::Failure {
    // SAFETY: callback is synchronous inside run_dynamic, with exclusive transfers
    // and ready source storage retained by its C++ owner.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        let context = &mut *context.cast::<CopyContext<'_>>();
        let value = context
            .transfers
            .copy(BorrowedTensor::new(&*source), context.location)?;
        let value = Box::new(value);
        output.write(ffi::TensorHandle {
            value: Box::into_raw(value).cast(),
            destroy: Some(destroy_tensor),
        });
        Ok::<_, anyhow::Error>(())
    }));
    let error = match result {
        Ok(Ok(())) => return ffi::Failure::ok(),
        Ok(Err(error)) => super::error("copy", error),
        Err(_) => {
            const MESSAGE: &str = "panic in native transfer callback";
            return ffi::Failure {
                code: 5,
                message: MESSAGE.as_ptr().cast(),
                message_size: MESSAGE.len(),
                owner: ptr::null_mut(),
                destroy: None,
            };
        }
    };
    let message = Box::new(error.to_string());
    ffi::Failure {
        code: error.code() as i32,
        message: message.as_ptr().cast(),
        message_size: message.len(),
        owner: Box::into_raw(message).cast(),
        destroy: Some(destroy_message),
    }
}

fn port(spec: &PortSpec) -> ffi::Port {
    ffi::Port {
        name: spec.name.as_ptr().cast(),
        name_size: spec.name.len(),
        dtype: dlpack::dl_type(spec.dtype),
        shape: spec.shape.as_ptr(),
        rank: spec.shape.len(),
    }
}

fn output_bindings(outputs: &[(usize, DLTensor)]) -> Vec<ffi::OutputBinding> {
    outputs
        .iter()
        .map(|output| ffi::OutputBinding {
            position: output.0,
            tensor: output.1,
        })
        .collect()
}

#[cfg(test)]
#[path = "../../../../tests/unit_tests/native/cxx_plugin.rs"]
mod tests;
