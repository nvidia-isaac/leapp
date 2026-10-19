// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    bundle::BundleHandle,
    ffi,
    tensor::{self, TensorHandle},
};
use leapp_runtime::{ExecutionDevice, MemoryLocation, RunOptions, RunResult, RuntimeOptions};
use std::{
    ffi::{c_void, OsStr},
    os::unix::ffi::OsStrExt,
};

pub struct RuntimeHandle(Option<leapp_runtime_bindings::RuntimeHandle>);

pub struct SessionHandle(Option<leapp_runtime_bindings::SessionHandle>);

pub struct ResultHandle(Option<RunResult>);

pub fn runtime_new() -> Box<RuntimeHandle> {
    Box::new(RuntimeHandle(None))
}

pub fn session_new() -> Box<SessionHandle> {
    Box::new(SessionHandle(None))
}

pub fn result_new() -> Box<ResultHandle> {
    Box::new(ResultHandle(None))
}

pub fn runtime_open(
    bundle: &BundleHandle,
    device: ffi::Location,
    backends: &[u8],
    output: &mut RuntimeHandle,
) -> ffi::StatusData {
    if output.0.is_some() {
        return crate::invalid("runtime destination is not empty");
    }
    let Some(bundle) = &bundle.0 else {
        return crate::invalid("bundle is empty");
    };
    let device = match tensor::location(device) {
        Ok(MemoryLocation::Cpu) => ExecutionDevice::Cpu,
        Ok(MemoryLocation::Cuda { device_id }) => ExecutionDevice::Cuda { device_id },
        Ok(_) => return crate::invalid("pinned memory is not an execution device"),
        Err(e) => return e,
    };
    let options = RuntimeOptions {
        device,
        backend_directory: OsStr::from_bytes(backends).into(),
    };
    crate::binding_status(
        leapp_runtime_bindings::RuntimeHandle::open(bundle, options).map(|runtime| {
            output.0 = Some(runtime);
        }),
    )
}

pub fn runtime_close(runtime: &RuntimeHandle) -> ffi::StatusData {
    match &runtime.0 {
        Some(r) => crate::binding_status(r.close()),
        None => crate::success(),
    }
}

pub fn session_create(runtime: &RuntimeHandle, output: &mut SessionHandle) -> ffi::StatusData {
    if output.0.is_some() {
        return crate::invalid("session destination is not empty");
    }
    let Some(runtime) = &runtime.0 else {
        return crate::invalid("runtime is empty");
    };
    crate::binding_status(runtime.create_session().map(|s| {
        output.0 = Some(s);
    }))
}

pub fn session_reset(session: &SessionHandle) -> ffi::StatusData {
    let Some(session) = &session.0 else {
        return crate::invalid("session is empty");
    };
    crate::binding_status(session.reset())
}

/// # Safety
/// Inputs must reference live, ready, immutable storage at their declared locations.
pub unsafe fn session_run(
    session: &SessionHandle,
    inputs: &[ffi::Input],
    location: ffi::Location,
    overrides: &[ffi::Placement],
    output: &mut ResultHandle,
) -> ffi::StatusData {
    let Some(session) = &session.0 else {
        return crate::invalid("session is empty");
    };
    let mut options = RunOptions::default();
    options.output_placement.default_location = match tensor::location(location) {
        Ok(v) => v,
        Err(e) => return e,
    };
    for p in overrides {
        let location = match tensor::location(ffi::Location {
            memory: p.location.memory,
            device: p.location.device,
        }) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let name = match std::str::from_utf8(&p.name) {
            Ok(v) => v.to_owned(),
            Err(_) => return crate::invalid("output name is not UTF-8"),
        };
        options.output_placement.overrides.insert(name, location);
    }
    let views = inputs
        .iter()
        .map(|input| {
            let location = tensor::location(ffi::Location {
                memory: input.location.memory,
                device: input.location.device,
            })?;
            let dtype = tensor::dtype(input.dtype)?;
            // SAFETY: the C++ facade retains descriptors for this call, and its caller
            // guarantees storage lifetime and completion of producer work.
            let view = unsafe {
                leapp_triton::TensorView::from_raw_buffer(
                    input.address as *const c_void,
                    input.byte_size,
                    dtype.into(),
                    &input.shape,
                    location,
                )
            }
            .map_err(crate::native_error)?;
            let name = std::str::from_utf8(&input.name)
                .map_err(|_| crate::invalid("input name is not UTF-8"))?;
            Ok((name, view.into()))
        })
        .collect::<Result<Vec<_>, ffi::StatusData>>();
    let views = match views {
        Ok(v) => v,
        Err(e) => return e,
    };
    crate::binding_status(session.run(&views, &options).map(|r| {
        output.0 = Some(r);
    }))
}

pub fn result_names(result: &ResultHandle) -> Vec<String> {
    result
        .0
        .as_ref()
        .map(|r| r.iter().map(|(n, _)| n.into()).collect())
        .unwrap_or_default()
}

pub fn result_tensor(
    result: &ResultHandle,
    name: &[u8],
    output: &mut TensorHandle,
) -> ffi::StatusData {
    let Some(result) = &result.0 else {
        return crate::invalid("result is empty");
    };
    let name = match std::str::from_utf8(name) {
        Ok(v) => v,
        Err(_) => return crate::invalid("output name is not UTF-8"),
    };
    match result.get(name) {
        Some(t) => {
            output.0 = Some(t.native().clone());
            crate::success()
        }
        None => crate::error(
            leapp_runtime::ErrorCode::NotFound,
            "execute",
            format!("output {name} not found"),
        ),
    }
}
