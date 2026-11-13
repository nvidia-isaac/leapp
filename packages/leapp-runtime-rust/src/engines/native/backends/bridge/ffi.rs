// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private transport paired with ../include/leapp/native/detail/abi.h.

use super::super::InputRoute;
use crate::{
    dlpack::{DLDataType, DLTensor},
    engines::error::{Error, ErrorKind},
};
use anyhow::{ensure, Context, Result};
use std::{
    ffi::{c_char, c_void},
    path::Path,
};

#[repr(C)]
pub(super) struct Failure {
    pub code: i32,
    pub message: *const c_char,
    pub message_size: usize,
    pub owner: *mut c_void,
    pub destroy: Option<unsafe extern "C" fn(*mut c_void)>,
}

impl Failure {
    pub fn ok() -> Self {
        Self {
            code: 0,
            message: std::ptr::null(),
            message_size: 0,
            owner: std::ptr::null_mut(),
            destroy: None,
        }
    }

    /// # Safety
    /// The diagnostic and its owner must originate from this ABI and remain live.
    pub unsafe fn check(self) -> Result<()> {
        if self.code == 0 {
            return Ok(());
        }
        ensure!(
            self.message_size <= isize::MAX as usize,
            "backend diagnostic overflow"
        );
        ensure!(
            self.message_size == 0 || !self.message.is_null(),
            "null backend diagnostic"
        );
        let bytes = if self.message_size == 0 {
            &[]
        } else {
            // SAFETY: the ABI retains the diagnostic until this guard is dropped.
            unsafe { std::slice::from_raw_parts(self.message.cast(), self.message_size) }
        };
        let kind = match self.code {
            1 => ErrorKind::InvalidArgument,
            2 => ErrorKind::NotFound,
            3 => ErrorKind::Unsupported,
            4 => ErrorKind::Io,
            _ => ErrorKind::Internal,
        };
        Err(Error::new(kind, "backend", String::from_utf8_lossy(bytes)).into())
    }
}

impl Drop for Failure {
    fn drop(&mut self) {
        if let Some(destroy) = self.destroy {
            // SAFETY: each ABI failure owns one diagnostic with its matching deallocator.
            unsafe { destroy(self.owner) };
        }
    }
}

#[repr(C)]
pub(super) struct TensorHandle {
    pub value: *mut c_void,
    pub destroy: Option<unsafe extern "C" fn(*mut c_void)>,
}

impl Drop for TensorHandle {
    fn drop(&mut self) {
        if !self.value.is_null() {
            if let Some(destroy) = self.destroy {
                // SAFETY: the producer transfers one handle and its matching deleter.
                unsafe { destroy(self.value) };
            }
        }
    }
}

#[repr(C)]
pub(super) struct Port {
    pub name: *const c_char,
    pub name_size: usize,
    pub dtype: DLDataType,
    pub shape: *const i64,
    pub rank: usize,
}

#[repr(C)]
pub(super) struct LoadArgs {
    pub path: *const c_char,
    pub path_size: usize,
    pub device: i32,
    pub stream: *mut c_void,
    pub inputs: *const Port,
    pub input_count: usize,
    pub outputs: *const Port,
    pub output_count: usize,
}

#[repr(C)]
pub(super) struct NamedTensor {
    pub name: *const c_char,
    pub name_size: usize,
    pub tensor: DLTensor,
}

#[repr(C)]
pub(super) struct OutputBinding {
    pub position: usize,
    pub tensor: DLTensor,
}

#[repr(C)]
pub(super) struct Host {
    pub context: *mut c_void,
    pub copy: unsafe extern "C" fn(*mut c_void, *const DLTensor, *mut TensorHandle) -> Failure,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Header {
    pub version: u32,
    pub size: u32,
}

macro_rules! api {
    ($($field:ident: $type:ty),* $(,)?) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub(super) struct Api {
            pub header: Header,
            $(pub $field: Option<$type>,)*
        }

        impl Api {
            fn validate(&self) -> Result<()> {
                $(ensure!(self.$field.is_some(), "native backend is missing {}", stringify!($field));)*
                Ok(())
            }
        }
    };
}

api! {
    load: unsafe extern "C" fn(*const LoadArgs, *mut *mut c_void) -> Failure,
    destroy: unsafe extern "C" fn(*mut c_void),
    new_session: unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> Failure,
    needs_guard: unsafe extern "C" fn(*const c_void) -> bool,
    can_prepare: unsafe extern "C" fn(*const c_void) -> bool,
    is_prepared: unsafe extern "C" fn(*const c_void) -> bool,
    input_alignment: unsafe extern "C" fn(*const c_void) -> usize,
    output_alignment: unsafe extern "C" fn(*const c_void) -> usize,
    prepare: unsafe extern "C" fn(
        *mut c_void, *const DLTensor, usize, *const DLTensor, usize,
    ) -> Failure,
    run: unsafe extern "C" fn(
        *mut c_void, *const NamedTensor, usize, *const InputRoute, usize,
        *const OutputBinding, usize,
    ) -> Failure,
    run_dynamic: unsafe extern "C" fn(
        *mut c_void, *const DLTensor, usize, *const OutputBinding, usize,
        *const Host, *mut *mut c_void,
    ) -> Failure,
    output_count: unsafe extern "C" fn(*const c_void) -> usize,
    take_output: unsafe extern "C" fn(*mut c_void, usize) -> TensorHandle,
    destroy_outputs: unsafe extern "C" fn(*mut c_void),
}

pub(super) struct Library {
    pub api: Api,
    _library: libloading::Library,
}

impl Library {
    pub fn open(path: &Path, backend: &str) -> Result<Self> {
        let load = || -> Result<Self> {
            // SAFETY: only registered trusted plugins are loaded. Check the stable
            // header before reading callbacks; retain the library through all handles.
            unsafe {
                let library = libloading::Library::new(path)?;
                let symbol = format!("leapp_native_{backend}_v1\0");
                let entry =
                    library.get::<unsafe extern "C" fn() -> *const Header>(symbol.as_bytes())?;
                let descriptor = entry();
                ensure!(
                    !descriptor.is_null(),
                    "{backend} backend returned a null API"
                );
                let header = *descriptor;
                ensure!(
                    header.version == 1 && header.size as usize >= std::mem::size_of::<Api>(),
                    "incompatible {backend} backend ABI {} ({} bytes)",
                    header.version,
                    header.size
                );
                let api = *descriptor.cast::<Api>();
                api.validate()?;
                Ok(Self {
                    api,
                    _library: library,
                })
            }
        };
        load().map_err(|error| {
            Error::new(
                ErrorKind::Unsupported,
                "load",
                format!(
                    "load {backend} backend {}: {error:#}; build and install the native \
                     {backend} backend from the matching runtime revision",
                    path.display()
                ),
            )
            .into()
        })
    }
}

pub(super) fn nonnull(value: *mut c_void) -> Result<*mut c_void> {
    std::ptr::NonNull::new(value)
        .map(|p| p.as_ptr())
        .context(crate::error::Internal(
            "native backend returned a null handle".into(),
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_backend_has_an_actionable_unsupported_error() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("torchscript/libleapp_torchscript.so");
        let error = Library::open(&path, "torchscript").err().unwrap();
        assert_eq!(
            error.downcast_ref::<Error>().unwrap().kind(),
            ErrorKind::Unsupported
        );
        assert!(error.to_string().contains(path.to_str().unwrap()));
        assert!(error.to_string().contains("build and install"));
    }
}
