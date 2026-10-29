// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{bundle::BundleHandle, ffi};
use lrt::{EngineKind, ExecutionDevice, MemoryLocation, RunResult, SessionOptions};

pub struct SessionHandle(Option<leapp_runtime_bindings::SessionHandle>);

pub struct ResultHandle(Option<RunResult>);

pub fn engine_default() -> ffi::EngineKind {
    match EngineKind::default() {
        EngineKind::Native => ffi::EngineKind::Native,
        EngineKind::Triton => ffi::EngineKind::Triton,
    }
}

pub fn session_new() -> Box<SessionHandle> {
    Box::new(SessionHandle(None))
}

pub fn result_new() -> Box<ResultHandle> {
    Box::new(ResultHandle(None))
}

pub fn session_open(
    bundle: &BundleHandle,
    engine: ffi::EngineKind,
    max_entries: usize,
    device: ffi::Location,
    output_location: ffi::Location,
    overrides: &[ffi::Placement],
    output: &mut SessionHandle,
) -> ffi::StatusData {
    if output.0.is_some() {
        return crate::invalid("session destination is not empty");
    }
    let Some(bundle) = &bundle.0 else {
        return crate::invalid("bundle is empty");
    };
    let device = match location(device) {
        Ok(MemoryLocation::Cpu) => ExecutionDevice::Cpu,
        Ok(MemoryLocation::Cuda { device_id }) => ExecutionDevice::Cuda { device_id },
        Ok(_) => return crate::invalid("pinned memory is not an execution device"),
        Err(e) => return e,
    };
    let mut options = SessionOptions {
        engine: match engine {
            ffi::EngineKind::Triton => EngineKind::Triton,
            ffi::EngineKind::Native => EngineKind::Native,
            _ => return crate::invalid("unsupported engine; expected Triton or Native"),
        },
        ..SessionOptions::default()
    };
    options.output_pool.max_entries = max_entries;
    options.output_placement.default_location = match location(output_location) {
        Ok(v) => v,
        Err(e) => return e,
    };
    for p in overrides {
        let location = match location(ffi::Location {
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
    options.device = device;
    crate::binding_status(
        leapp_runtime_bindings::SessionHandle::open(bundle, options).map(|runtime| {
            output.0 = Some(runtime);
        }),
    )
}

pub fn session_close(session: &SessionHandle) -> ffi::StatusData {
    match &session.0 {
        Some(r) => crate::binding_status(r.close()),
        None => crate::success(),
    }
}

pub fn session_duplicate(session: &SessionHandle, output: &mut SessionHandle) -> ffi::StatusData {
    if output.0.is_some() {
        return crate::invalid("session destination is not empty");
    }
    let Some(session) = &session.0 else {
        return crate::invalid("session is empty");
    };
    crate::binding_status(session.new_session().map(|s| {
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
    output: &mut ResultHandle,
) -> ffi::StatusData {
    let Some(session) = &session.0 else {
        return crate::invalid("session is empty");
    };
    let views = inputs
        .iter()
        .map(|input| {
            let ptr = input.descriptor as *const lrt::dlpack::DLTensor;
            if ptr.is_null() {
                return Err(crate::invalid("null DLPack input"));
            }
            // SAFETY: Session::Run requires valid descriptors, metadata, and ready
            // immutable storage until this synchronous call completes.
            let view = unsafe { lrt::dlpack::BorrowedTensor::new(&*ptr) };
            let name = std::str::from_utf8(&input.name)
                .map_err(|_| crate::invalid("input name is not UTF-8"))?;
            Ok((name, view))
        })
        .try_fold(
            std::collections::HashMap::with_capacity(inputs.len()),
            |mut map, item| {
                let (name, value) = item?;
                if map.insert(name, value).is_some() {
                    return Err(crate::invalid("duplicate input name"));
                }
                Ok(map)
            },
        );
    let views = match views {
        Ok(v) => v,
        Err(e) => return e,
    };
    crate::binding_status(session.run(&views).map(|r| {
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

pub fn result_tensor(result: &ResultHandle, name: &[u8], output: &mut usize) -> ffi::StatusData {
    let Some(result) = &result.0 else {
        return crate::invalid("result is empty");
    };
    let name = match std::str::from_utf8(name) {
        Ok(v) => v,
        Err(_) => return crate::invalid("output name is not UTF-8"),
    };
    match result.get(name) {
        Some(t) => {
            *output = t.export(true).into_raw().0 as usize;
            crate::success()
        }
        None => crate::error(
            lrt::ErrorCode::NotFound,
            "execute",
            format!("output {name} not found"),
        ),
    }
}

pub fn location(v: ffi::Location) -> Result<MemoryLocation, ffi::StatusData> {
    match v.memory {
        ffi::Memory::Cpu if v.device == 0 => Ok(MemoryLocation::Cpu),
        ffi::Memory::Pinned if v.device == 0 => Ok(MemoryLocation::PinnedCpu),
        ffi::Memory::Cuda => Ok(MemoryLocation::Cuda {
            device_id: v.device,
        }),
        _ => Err(crate::invalid("invalid memory location")),
    }
}
