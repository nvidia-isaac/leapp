// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::sys::TritonApi;
use crate::engines::error::{Error, ErrorKind, Result};
use std::{
    path::PathBuf,
    sync::{Mutex, OnceLock},
};

// Callbacks and native handles may outlive individual sessions. Retain the
// library for the process lifetime, and synchronize only first initialization.
static TRITON_API: OnceLock<TritonApi> = OnceLock::new();
static INITIALIZE: Mutex<()> = Mutex::new(());

pub(in crate::engines::triton) fn load() -> Result<&'static TritonApi> {
    if let Some(api) = TRITON_API.get() {
        return Ok(api);
    }
    let _initializing = INITIALIZE.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(api) = TRITON_API.get() {
        return Ok(api);
    }
    let candidates = match std::env::var_os("TRITON_ROOT") {
        Some(root) => vec![PathBuf::from(root).join("lib/libtritonserver.so")],
        None => vec![
            PathBuf::from("libtritonserver.so"),
            PathBuf::from("/opt/tritonserver/lib/libtritonserver.so"),
        ],
    };
    let mut failures = Vec::new();
    for path in candidates {
        // SAFETY: these are trusted Triton installation paths. GLOBAL preserves
        // server symbol visibility for backend libraries loaded by Triton.
        let library = unsafe {
            libloading::os::unix::Library::open(
                Some(&path),
                libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
            )
        };
        let library = match library {
            Ok(library) => library.into(),
            Err(error) => {
                failures.push(format!(
                    "{}: {:#}",
                    path.display(),
                    anyhow::Error::new(error)
                ));
                continue;
            }
        };
        // SAFETY: signatures are generated from the supported Triton C header.
        // Every required symbol is resolved before the table is published.
        let api = unsafe { TritonApi::load(library) }.map_err(|error| {
            unavailable(format!(
                "{}: {:#}",
                path.display(),
                anyhow::Error::new(error)
            ))
        })?;
        return Ok(TRITON_API.get_or_init(|| api));
    }
    Err(unavailable(failures.join("; ")))
}

fn unavailable(detail: String) -> Error {
    Error::new(
        ErrorKind::Unsupported,
        "load",
        format!("load Triton: {detail}; install Triton 2.69.0 and its runtime dependencies, or set TRITON_ROOT"),
    )
}
