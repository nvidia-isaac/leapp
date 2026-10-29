// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private C++ backend ABI. Keep field order and signatures paired with api.h.

use crate::engines::error::{Error, ErrorKind};
use anyhow::{Context, Result};
use std::{
    ffi::{c_char, c_void, CStr},
    path::Path,
};

#[repr(C)]
pub(super) struct Port {
    pub name: *const c_char,
    pub name_size: usize,
    pub dtype: u8,
    pub shape: *const i64,
    pub rank: usize,
}

#[repr(C)]
pub(super) struct View {
    pub port: Port,
    pub data: usize,
    pub bytes: usize,
    pub device: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Api {
    pub load_model: unsafe extern "C" fn(
        *const c_char,
        i32,
        *mut c_void,
        *const Port,
        usize,
        *const Port,
        usize,
        *mut *mut c_void,
    ) -> *mut c_char,
    pub destroy_model: unsafe extern "C" fn(*mut c_void),
    pub requires_dynamic: unsafe extern "C" fn(*const c_void) -> bool,
    pub create_session: unsafe extern "C" fn(*mut c_void, *mut *mut c_void) -> *mut c_char,
    pub destroy_session: unsafe extern "C" fn(*mut c_void),
    pub prepare: unsafe extern "C" fn(
        *mut c_void,
        *const View,
        usize,
        *const View,
        usize,
        *mut *mut c_void,
    ) -> *mut c_char,
    pub destroy_call: unsafe extern "C" fn(*mut c_void),
    pub run: unsafe extern "C" fn(*mut c_void) -> *mut c_char,
    pub run_dynamic:
        unsafe extern "C" fn(*mut c_void, *const View, usize, *mut *mut c_void) -> *mut c_char,
    pub destroy_outputs: unsafe extern "C" fn(*mut c_void),
    pub views: unsafe extern "C" fn(*const c_void, *mut usize) -> *const View,
    pub free_error: unsafe extern "C" fn(*mut c_char),
}

pub(super) struct Library {
    pub api: Api,
    _library: libloading::Library,
}

impl Library {
    pub fn open(path: &Path, backend: &str) -> Result<Self> {
        let load = || -> Result<Self> {
            // SAFETY: this is the matching, installed backend bridge. It has no
            // Rust callbacks; the library is retained through every native handle.
            unsafe {
                let library = libloading::Library::new(path)?;
                let symbol = format!("leapp_native_{backend}\0");
                let entry =
                    library.get::<unsafe extern "C" fn() -> *const Api>(symbol.as_bytes())?;
                let api = entry();
                anyhow::ensure!(
                    !api.is_null(),
                    "{backend} backend returned a null API"
                );

                Ok(Self {
                    api: *api,
                    _library: library,
                })
            }
        };

        load().map_err(|error| {
            let message = format!(
                "load {backend} backend {}: {error:#}; build and install the native {backend} backend",
                path.display()
            );
            Error::new(
                ErrorKind::Unsupported,
                "load",
                message,
            )
            .into()
        })
    }
}

impl Api {
    /// # Safety
    /// message must be null or a diagnostic returned by this API instance.
    pub unsafe fn check(&self, message: *mut c_char) -> Result<()> {
        if message.is_null() {
            return Ok(());
        }

        // SAFETY: the bridge returns owned NUL-terminated messages and its deallocator.
        let text = unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned();
        unsafe { (self.free_error)(message) };

        let kind = if text.starts_with("invalid argument:") {
            ErrorKind::InvalidArgument
        } else if text.starts_with("backend unsupported:") {
            ErrorKind::Unsupported
        } else {
            ErrorKind::Internal
        };

        Err(Error::new(kind, "backend", text).into())
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
        let path = directory.path().join("tensorrt/libleapp_tensorrt.so");
        let error = Library::open(&path, "tensorrt").err().unwrap();

        assert_eq!(
            error.downcast_ref::<Error>().unwrap().kind(),
            ErrorKind::Unsupported
        );
        assert!(error.to_string().contains(path.to_str().unwrap()));
        assert!(error.to_string().contains("build and install"));
    }
}
