// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{check, cstring, Handle},
    request, sys, Error, ErrorKind, MemoryLocation, Result, Tensor, TensorView,
};
use std::{collections::HashMap, os::unix::ffi::OsStrExt, path::Path, ptr, sync::Arc};

/// Exact output locations, with optional overrides by Triton tensor name.
#[derive(Debug, Clone, Default)]
pub struct OutputPlacement {
    pub default_location: MemoryLocation,
    pub overrides: HashMap<String, MemoryLocation>,
}

impl OutputPlacement {
    pub(crate) fn location(&self, name: &str) -> MemoryLocation {
        self.overrides
            .get(name)
            .copied()
            .unwrap_or(self.default_location)
    }

    fn validate(&self) -> Result<()> {
        self.default_location.validate()?;
        for location in self.overrides.values() {
            location.validate()?;
        }
        Ok(())
    }
}

/// Owns an in-process Triton instance. Repository files must remain available.
/// Inference is synchronous; each call has independent callback state.
pub struct Runtime {
    // Rust drops fields in declaration order: Triton stops before its allocator.
    server: Handle<sys::TRITONSERVER_Server>,
    allocator: Handle<sys::TRITONSERVER_ResponseAllocator>,
    output_pool: Arc<crate::output_pool::OutputPool>,
}

// SAFETY: Triton permits concurrent requests. The instance is immutable after
// creation; callbacks operate on request-local state. &self borrows prevent Drop
// while inference is active, and native shutdown precedes allocator destruction.
unsafe impl Send for Runtime {}

unsafe impl Sync for Runtime {}

impl Runtime {
    /// Load the supplied repository using Triton 2.69.0 and compatible backends.
    /// Keep the repository and linked artifacts alive until this object is dropped.
    /// Each runtime shares a bounded output-storage pool across its requests.
    pub fn open(repository: &Path, backend_directory: &Path) -> Result<Self> {
        let repository = cstring(repository.as_os_str().as_bytes(), "load")?;
        let backend = cstring(backend_directory.as_os_str().as_bytes(), "load")?;
        let mut options = Handle::empty(sys::TRITONSERVER_ServerOptionsDelete);
        let mut runtime = Self {
            server: Handle::empty(sys::TRITONSERVER_ServerDelete),
            allocator: Handle::empty(sys::TRITONSERVER_ResponseAllocatorDelete),
            output_pool: Arc::new(crate::output_pool::OutputPool::default()),
        };
        // SAFETY: output pointers are live, strings outlive the calls, and each
        // created object immediately has an owning handle, including error paths.
        unsafe {
            let (mut major, mut minor) = (0, 0);
            check(sys::TRITONSERVER_ApiVersion(&mut major, &mut minor), "load")?;
            if major != sys::TRITONSERVER_API_VERSION_MAJOR
                || minor < sys::TRITONSERVER_API_VERSION_MINOR
            {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "load",
                    format!("incompatible Triton C API {major}.{minor}"),
                ));
            }
            check(sys::TRITONSERVER_ServerOptionsNew(&mut options.ptr), "load")?;
            check(
                sys::TRITONSERVER_ServerOptionsSetModelRepositoryPath(
                    options.ptr,
                    repository.as_ptr(),
                ),
                "load",
            )?;
            check(
                sys::TRITONSERVER_ServerOptionsSetBackendDirectory(options.ptr, backend.as_ptr()),
                "load",
            )?;
            check(
                sys::TRITONSERVER_ServerOptionsSetModelControlMode(
                    options.ptr,
                    sys::TRITONSERVER_MODEL_CONTROL_NONE,
                ),
                "load",
            )?;
            check(
                sys::TRITONSERVER_ServerOptionsSetStrictModelConfig(options.ptr, true),
                "load",
            )?;
            check(
                sys::TRITONSERVER_ServerOptionsSetModelLoadRetryCount(options.ptr, 0),
                "load",
            )?;
            check(
                sys::TRITONSERVER_ServerOptionsSetExitOnError(options.ptr, true),
                "load",
            )?;
            check(
                sys::TRITONSERVER_ServerOptionsSetLogInfo(options.ptr, false),
                "load",
            )?;
            check(
                sys::TRITONSERVER_ResponseAllocatorNew(
                    &mut runtime.allocator.ptr,
                    Some(request::allocate),
                    Some(request::release_buffer),
                    None,
                ),
                "load",
            )?;
            check(
                sys::TRITONSERVER_ResponseAllocatorSetQueryFunction(
                    runtime.allocator.ptr,
                    Some(request::query),
                ),
                "load",
            )?;
            check(
                sys::TRITONSERVER_ServerNew(&mut runtime.server.ptr, options.ptr),
                "load",
            )?;
            // The C API version alone does not identify a Triton release.
            let mut metadata = Handle::empty(sys::TRITONSERVER_MessageDelete);
            check(
                sys::TRITONSERVER_ServerMetadata(runtime.server.ptr, &mut metadata.ptr),
                "load",
            )?;
            let mut json = ptr::null();
            let mut size = 0;
            check(
                sys::TRITONSERVER_MessageSerializeToJson(metadata.ptr, &mut json, &mut size),
                "load",
            )?;
            // Triton owns these bytes until the metadata handle is dropped.
            validate_server_version(std::slice::from_raw_parts(json.cast::<u8>(), size))?;
            let mut ready = false;
            check(
                sys::TRITONSERVER_ServerIsReady(runtime.server.ptr, &mut ready),
                "load",
            )?;
            if !ready {
                return Err(Error::new(
                    ErrorKind::Internal,
                    "load",
                    "Triton repository is not ready",
                ));
            }
        }
        Ok(runtime)
    }

    /// Run a numeric, single-response model, requesting every output.
    /// Input storage remains borrowed until request release. Returned outputs
    /// own their storage and can outlive this runtime. Placement never falls back.
    pub fn infer(
        &self,
        model_name: &str,
        inputs: &[(&str, TensorView<'_>)],
        placement: &OutputPlacement,
    ) -> Result<Vec<(String, Tensor)>> {
        placement.validate()?;
        let model = cstring(model_name, "execute")?;
        let names = inputs
            .iter()
            .map(|(name, _)| cstring(name, "execute"))
            .collect::<Result<Vec<_>>>()?;
        let mut request = Handle::empty(sys::TRITONSERVER_InferenceRequestDelete);
        let state = Box::new(request::State::new(
            placement.clone(),
            self.output_pool.clone(),
        ));
        let userp = (&*state as *const request::State).cast_mut().cast();

        const SELECTED_VERSION: i64 = -1;

        // SAFETY: native handles are owned, input borrows remain live through
        // completion, and the boxed callback state has a stable address.
        unsafe {
            let mut flags = 0;
            check(
                sys::TRITONSERVER_ServerModelTransactionProperties(
                    self.server.ptr,
                    model.as_ptr(),
                    SELECTED_VERSION,
                    &mut flags,
                    ptr::null_mut(),
                ),
                "execute",
            )?;
            if flags & sys::TRITONSERVER_TXN_DECOUPLED != 0 {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "execute",
                    "decoupled models are not supported",
                ));
            }
            check(
                sys::TRITONSERVER_InferenceRequestNew(
                    &mut request.ptr,
                    self.server.ptr,
                    model.as_ptr(),
                    SELECTED_VERSION,
                ),
                "execute",
            )?;
            for ((_, input), name) in inputs.iter().zip(&names) {
                let (memory, device) = input.location.native();
                check(
                    sys::TRITONSERVER_InferenceRequestAddInput(
                        request.ptr,
                        name.as_ptr(),
                        input.dtype as u32,
                        input.shape.as_ptr(),
                        input.shape.len() as u64,
                    ),
                    "execute",
                )?;
                check(
                    sys::TRITONSERVER_InferenceRequestAppendInputData(
                        request.ptr,
                        name.as_ptr(),
                        input.ptr,
                        input.len,
                        memory,
                        device,
                    ),
                    "execute",
                )?;
            }
            check(
                sys::TRITONSERVER_InferenceRequestSetReleaseCallback(
                    request.ptr,
                    Some(request::release_request),
                    userp,
                ),
                "execute",
            )?;
            check(
                sys::TRITONSERVER_InferenceRequestSetResponseCallback(
                    request.ptr,
                    self.allocator.ptr,
                    userp,
                    Some(request::receive_response),
                    userp,
                ),
                "execute",
            )?;
            // Callbacks can finish during submission. Transfer our request owner
            // first, reclaiming it only when submission fails to take ownership.
            let raw = request.release();
            if let Err(error) = check(
                sys::TRITONSERVER_ServerInferAsync(self.server.ptr, raw, ptr::null_mut()),
                "execute",
            ) {
                request.ptr = raw;
                return Err(error);
            }
        }
        // No early return may release input borrows or callback state before this wait.
        let response = state.wait()?;
        request::outputs(&response, placement)
    }
}

// Keep the supported release aligned with proto/model_config.proto (r26.05).
const TRITON_VERSION: &str = "2.69.0";

fn validate_server_version(json: &[u8]) -> Result<()> {
    let metadata: serde_json::Value = serde_json::from_slice(json).map_err(|error| {
        Error::new(
            ErrorKind::Internal,
            "load",
            format!("invalid Triton server metadata: {error}"),
        )
    })?;
    let version = metadata.get("version").and_then(serde_json::Value::as_str);
    if version != Some(TRITON_VERSION) {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "load",
            format!(
                "Triton {TRITON_VERSION} (r26.05) is required to match the bundled model configuration schema; loaded version: {}",
                version.unwrap_or("missing or invalid")
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_the_exact_schema_release() {
        validate_server_version(br#"{"version":"2.69.0","extensions":[]}"#).unwrap();
        for version in ["2.68.0", "2.69.1", "2.70.0", "2.69.0-custom", ""] {
            let json = serde_json::json!({"version": version}).to_string();
            let error = validate_server_version(json.as_bytes()).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert_eq!(error.phase(), "load");
            assert!(error.to_string().contains("2.69.0 (r26.05) is required"));
            assert!(error.to_string().ends_with(version));
        }
    }

    #[test]
    fn rejects_missing_invalid_or_malformed_versions() {
        for json in [
            b"{}".as_slice(),
            br#"{"version":269}"#,
            br#"{"version":null}"#,
        ] {
            let error = validate_server_version(json).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::Unsupported);
            assert!(error.to_string().ends_with("missing or invalid"));
        }
        let error = validate_server_version(b"not JSON").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Internal);
        assert_eq!(error.phase(), "load");
    }
}
