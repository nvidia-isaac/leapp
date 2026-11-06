// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python capsule protocol only. Tensor interpretation belongs to the selected engine.
use lrt::dlpack::{ManagedTensor, SharedTensor};
use pyo3::{
    exceptions::{PyBufferError, PyTypeError},
    ffi,
    prelude::*,
    types::PyDict,
};
use std::{cell::RefCell, ffi::c_char, sync::Arc};

const LEGACY: *const c_char = c"dltensor".as_ptr();
const VERSIONED: *const c_char = c"dltensor_versioned".as_ptr();
const USED_LEGACY: *const c_char = c"used_dltensor".as_ptr();
const USED_VERSIONED: *const c_char = c"used_dltensor_versioned".as_ptr();

thread_local! {
    static STREAMS: RefCell<Vec<(u32, Arc<lrt::dlpack::HandoffStream>)>> = const { RefCell::new(Vec::new()) };
}
fn stream(device: u32) -> Result<Arc<lrt::dlpack::HandoffStream>, lrt::RuntimeError> {
    STREAMS.with(|streams| {
        let mut streams = streams.borrow_mut();
        if let Some((_, stream)) = streams.iter().find(|(id, _)| *id == device) {
            return Ok(stream.clone());
        }
        let stream = Arc::new(lrt::dlpack::HandoffStream::new(device)?);
        streams.push((device, stream.clone()));
        Ok(stream)
    })
}

pub fn import_items<'py>(
    py: Python<'py>,
    items: &Bound<'py, PyAny>,
) -> PyResult<Vec<(String, ManagedTensor)>> {
    // Retain producers and capsules until every queued dependency has completed,
    // including failed imports. Declare owners last so their deleters run first.
    let mut retained = Vec::new();
    let mut handoffs = Vec::new();
    let mut owned = Vec::new();
    let imported = (|| {
        for item in items.try_iter()? {
            let (name, producer): (String, Bound<'_, PyAny>) = item?.extract()?;
            retained.push(producer);
            let (capsule, declared) =
                request_capsule(py, retained.last().unwrap(), &mut handoffs)?;
            retained.push(capsule);
            owned.push((name, consume(retained.last().unwrap())?));
            let descriptor = owned
                .last()
                .unwrap()
                .1
                .descriptor()
                .map_err(PyBufferError::new_err)?;
            let actual = (descriptor.device.device_type, descriptor.device.device_id);
            // Pinned producers may conservatively export ordinary CPU descriptors.
            if actual != declared && !(declared == (3, 0) && actual == (1, 0)) {
                return Err(PyBufferError::new_err(
                    "DLPack device descriptor changed during import",
                ));
            }
        }
        Ok(())
    })();
    let mut ready = Ok(());
    for handoff in handoffs {
        // Always drain every device, even if an earlier import or wait failed.
        let result = py
            .detach(|| handoff.synchronize())
            .map_err(|e| crate::execution_error(py, e));
        ready = ready.and(result);
    }
    imported?;
    ready?;
    Ok(owned)
}

fn request_capsule<'py>(
    py: Python<'py>,
    producer: &Bound<'py, PyAny>,
    handoffs: &mut Vec<Arc<lrt::dlpack::HandoffStream>>,
) -> PyResult<(Bound<'py, PyAny>, (i32, i32))> {
    let declared: (i32, i32) = producer
        .getattr("__dlpack_device__")
        .map_err(|_| PyTypeError::new_err("expected a DLPack producer"))?
        .call0()?
        .extract()?;
    if !matches!(declared, (1, 0) | (3, 0) | (2, 0..)) {
        return Err(PyBufferError::new_err("unsupported DLPack device"));
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("max_version", (1, 0))?;
    let handoff = if declared.0 == 2 {
        let handoff = py
            .detach(|| stream(declared.1 as u32))
            .map_err(|e| crate::execution_error(py, e))?;
        kwargs.set_item("stream", handoff.address())?;
        if !handoffs.iter().any(|stream| Arc::ptr_eq(stream, &handoff)) {
            handoffs.push(handoff.clone());
        }
        Some(handoff)
    } else {
        None
    };
    let export = || match producer.call_method("__dlpack__", (), Some(&kwargs)) {
        Ok(c) => Ok(c),
        Err(e) if e.is_instance_of::<PyTypeError>(py) => {
            // Compatibility with producers predating max_version; no framework probing.
            kwargs.del_item("max_version")?;
            producer.call_method("__dlpack__", (), Some(&kwargs))
        }
        Err(e) => Err(e),
    };
    let capsule = if let Some(handoff) = handoff {
        handoff
            .with_device(export)
            .map_err(|e| crate::execution_error(py, e))??
    } else {
        export()?
    };
    Ok((capsule, declared))
}

fn consume(capsule: &Bound<'_, PyAny>) -> PyResult<ManagedTensor> {
    // SAFETY: validate capsule name before reading; rename before adopting the lease.
    unsafe {
        let (name, used, versioned) =
            if ffi::PyCapsule_IsValid(capsule.as_ptr(), VERSIONED) != 0 {
                (VERSIONED, USED_VERSIONED, true)
            } else if ffi::PyCapsule_IsValid(capsule.as_ptr(), LEGACY) != 0 {
                (LEGACY, USED_LEGACY, false)
            } else {
                return Err(PyBufferError::new_err(
                    "invalid or already consumed DLPack capsule",
                ));
            };
        let ptr = ffi::PyCapsule_GetPointer(capsule.as_ptr(), name);
        if ptr.is_null() || ffi::PyCapsule_SetName(capsule.as_ptr(), used) != 0 {
            return Err(PyErr::fetch(capsule.py()));
        }
        Ok(if versioned {
            ManagedTensor::from_versioned(ptr.cast())
        } else {
            ManagedTensor::from_legacy(ptr.cast())
        }
        .unwrap())
    }
}
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: only unconsumed capsules own a lease; consumed capsules do nothing.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, LEGACY) != 0 {
            drop(ManagedTensor::from_legacy(
                ffi::PyCapsule_GetPointer(capsule, LEGACY).cast(),
            ));
        } else if ffi::PyCapsule_IsValid(capsule, VERSIONED) != 0 {
            drop(ManagedTensor::from_versioned(
                ffi::PyCapsule_GetPointer(capsule, VERSIONED).cast(),
            ));
        }
    }
}

/// Protocol exporter, deliberately without a tensor convenience API.
#[pyclass(name = "DLPackOutput", frozen, module = "lrt")]
pub struct Output {
    pub inner: SharedTensor,
}
#[pymethods]
impl Output {
    fn __dlpack_device__(&self) -> (i32, i32) {
        let device = self.inner.descriptor().device;
        (device.device_type, device.device_id)
    }
    #[pyo3(signature = (*, stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &self,
        py: Python<'_>,
        stream: Option<i64>,
        max_version: Option<(u32, u32)>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let device = self.__dlpack_device__();
        if copy == Some(true) || dl_device.is_some_and(|d| d != device) {
            return Err(PyBufferError::new_err(
                "materialize copies and device transfers in the consuming framework",
            ));
        }
        if stream == Some(0)
            || stream.is_some_and(|s| s < -1)
            || (device.0 != 2 && stream.is_some())
        {
            return Err(PyBufferError::new_err("invalid DLPack stream"));
        }
        // Synchronous inference/copies finish before publication, on every consumer stream.
        let (pointer, versioned) = self
            .inner
            .export(max_version.is_some_and(|v| v >= (1, 0)))
            .into_raw();
        // SAFETY: the capsule takes the fresh lease, or failure immediately releases it.
        unsafe {
            let capsule = ffi::PyCapsule_New(
                pointer,
                if versioned { VERSIONED } else { LEGACY },
                Some(capsule_destructor),
            );
            if capsule.is_null() {
                if versioned {
                    drop(ManagedTensor::from_versioned(pointer.cast()));
                } else {
                    drop(ManagedTensor::from_legacy(pointer.cast()));
                }
                return Err(PyErr::fetch(py));
            }
            Ok(Bound::from_owned_ptr(py, capsule).unbind())
        }
    }
}
