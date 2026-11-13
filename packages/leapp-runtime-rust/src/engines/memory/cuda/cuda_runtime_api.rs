// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::engines::error::{Error, ErrorKind, Result};
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
};

// Keep CUDA loaded through retained tensors, streams, and process teardown.
static CUDA_RUNTIME_API: OnceLock<CudaRuntimeApi> = OnceLock::new();
static INITIALIZE: Mutex<()> = Mutex::new(());

// Declare each signature once, pairing its field with its exported symbol.
macro_rules! functions {
    ($(fn $name:ident($($argument:ident: $ty:ty),* $(,)?) -> $result:ty;)*) => {
        #[allow(non_snake_case)]
        pub(super) struct CudaRuntimeApi {
            $(pub $name: unsafe extern "C" fn($($ty),*) -> $result,)*
            _library: libloading::Library,
        }

        impl CudaRuntimeApi {
            unsafe fn resolve(
                library: libloading::Library,
            ) -> std::result::Result<Self, libloading::Error> {
                // SAFETY: these declarations match the supported CUDA runtime ABI;
                // the table owns its library for as long as any symbol can be used.
                unsafe {
                    Ok(Self {
                        $($name: *library.get(concat!(stringify!($name), "\0").as_bytes())?,)*
                        _library: library,
                    })
                }
            }
        }
    };
}

functions! {
    fn cudaGetErrorString(error: c_int) -> *const c_char;
    fn cudaGetDeviceCount(count: *mut c_int) -> c_int;
    fn cudaGetDevice(device: *mut c_int) -> c_int;
    fn cudaSetDevice(device: c_int) -> c_int;
    fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> c_int;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        size: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    fn cudaMemcpyPeerAsync(
        dst: *mut c_void,
        dst_device: c_int,
        src: *const c_void,
        src_device: c_int,
        size: usize,
        stream: *mut c_void,
    ) -> c_int;
    fn cudaMemcpyPeer(
        dst: *mut c_void,
        dst_device: c_int,
        src: *const c_void,
        src_device: c_int,
        size: usize,
    ) -> c_int;
    fn cudaMemset(ptr: *mut c_void, value: c_int, count: usize) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaStreamCreateWithFlags(stream: *mut *mut c_void, flags: u32) -> c_int;
    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
    fn cudaStreamDestroy(stream: *mut c_void) -> c_int;
    fn cudaEventCreateWithFlags(event: *mut *mut c_void, flags: u32) -> c_int;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> c_int;
    fn cudaEventDestroy(event: *mut c_void) -> c_int;
}

impl CudaRuntimeApi {
    pub(super) fn check(&self, code: c_int) -> Result<()> {
        if code == 0 {
            return Ok(());
        }
        // SAFETY: CUDA returns a static NUL-terminated error description.
        let message = unsafe { CStr::from_ptr((self.cudaGetErrorString)(code)) }
            .to_string_lossy()
            .into_owned();
        Err(Error::new(ErrorKind::Internal, "buffer", message))
    }
}

pub(super) fn loaded() -> Option<&'static CudaRuntimeApi> {
    CUDA_RUNTIME_API.get()
}

pub(super) fn load() -> Result<&'static CudaRuntimeApi> {
    if let Some(api) = loaded() {
        return Ok(api);
    }
    let _initialize = INITIALIZE.lock().unwrap_or_else(|error| error.into_inner());
    if let Some(api) = loaded() {
        return Ok(api);
    }
    let api = open(std::env::var_os("CUDA_ROOT").map(PathBuf::from).as_deref())?;
    Ok(CUDA_RUNTIME_API.get_or_init(|| api))
}

fn open(root: Option<&Path>) -> Result<CudaRuntimeApi> {
    const NAMES: [&str; 3] = ["libcudart.so.13", "libcudart.so.12", "libcudart.so"];
    let installation = |root: &Path| {
        ["lib64", "lib"]
            .into_iter()
            .flat_map(|directory| NAMES.map(|name| root.join(directory).join(name)))
            .collect::<Vec<_>>()
    };
    let candidates = match root {
        Some(root) => installation(root),
        None => NAMES
            .map(PathBuf::from)
            .into_iter()
            .chain(installation(Path::new("/usr/local/cuda")))
            .collect(),
    };
    let mut errors = Vec::new();
    for candidate in candidates {
        // SAFETY: load the user-selected CUDA runtime, resolving its dependencies
        // immediately. Keep its symbols private to avoid replacing other SDK APIs.
        let library = unsafe {
            libloading::os::unix::Library::open(
                Some(&candidate),
                libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_LOCAL,
            )
        };
        match library {
            Ok(library) => {
                // A found library must provide the complete API. Do not hide an
                // incompatible installation by trying another CUDA version.
                return unsafe { CudaRuntimeApi::resolve(library.into()) }.map_err(|error| {
                    unavailable(format!(
                        "{}: {:#}",
                        candidate.display(),
                        anyhow::Error::new(error)
                    ))
                });
            }
            Err(error) => {
                errors.push(format!(
                    "{}: {:#}",
                    candidate.display(),
                    anyhow::Error::new(error)
                ));
                if candidate.is_file() {
                    break;
                }
            }
        }
    }
    Err(unavailable(errors.join("; ")))
}

fn unavailable(message: String) -> Error {
    Error::new(
        ErrorKind::Unsupported,
        "load",
        format!("load CUDA runtime: {message}; install its runtime dependencies or set CUDA_ROOT"),
    )
}
