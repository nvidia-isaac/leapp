// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    bundle::PyBundle,
    dlpack::{import_items, Output},
};
use leapp_runtime_bindings::{Error, SessionHandle};
use lrt::dlpack::BorrowedTensor;
use lrt::{EngineKind, ExecutionDevice, RunResult, SessionOptions};
use pyo3::{
    exceptions::{PyKeyError, PyTypeError, PyValueError},
    prelude::*,
    types::PyTuple,
};
use std::collections::HashMap;

fn binding_error(py: Python<'_>, error: Error) -> PyErr {
    crate::diagnostic(
        py,
        crate::runtime_error(&error),
        error.code(),
        error.phase(),
    )
}

#[pyclass(name = "Session", frozen, module = "lrt")]
pub struct PySession {
    inner: SessionHandle,
}

#[pymethods]
impl PySession {
    #[staticmethod]
    #[pyo3(signature = (bundle, options=None))]
    /// Load a bundle for synchronous execution using SessionOptions.
    fn open(
        py: Python<'_>,
        bundle: &PyBundle,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut opts = SessionOptions::default();
        if let Some(options) = options {
            opts.output_pool.max_entries = options
                .getattr("output_pool")?
                .getattr("max_entries")?
                .extract()?;
            let engine: String = options.getattr("engine")?.extract()?;
            opts.engine = match engine.as_str() {
                "triton" => EngineKind::Triton,
                "native" => EngineKind::Native,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "unknown engine '{engine}'; expected 'triton' or 'native'"
                    )))
                }
            };
            let device: String = options.getattr("device")?.extract()?;
            opts.device = match crate::location(&device)? {
                lrt::MemoryLocation::Cpu => ExecutionDevice::Cpu,
                lrt::MemoryLocation::Cuda { device_id } => ExecutionDevice::Cuda { device_id },
                _ => {
                    return Err(PyTypeError::new_err(
                        "pinned is a buffer location, not an execution device",
                    ))
                }
            };
            let output_device: String = options.getattr("output_device")?.extract()?;
            opts.output_placement.default_location = crate::location(&output_device)?;
            let overrides: HashMap<String, String> =
                options.getattr("output_overrides")?.extract()?;
            for (name, device) in overrides {
                opts.output_placement
                    .overrides
                    .insert(name, crate::location(&device)?);
            }
        }
        let inner = py
            .detach(|| SessionHandle::open(&bundle.inner, opts))
            .map_err(|e| binding_error(py, e))?;
        Ok(Self { inner })
    }

    /// Return the bundle snapshot retained at open.
    fn info(&self) -> PyBundle {
        PyBundle {
            inner: self.inner.info().clone(),
        }
    }

    /// Create an independent feedback state and execution session.
    fn new_session(&self, py: Python<'_>) -> PyResult<PySession> {
        let inner = py
            .detach(|| self.inner.new_session())
            .map_err(|e| binding_error(py, e))?;
        Ok(PySession { inner })
    }

    /// Wait for active calls and close this session.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.close())
            .map_err(|e| binding_error(py, e))
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        let inner = &slf.inner;
        if !slf
            .py()
            .detach(|| inner.is_open())
            .map_err(|e| binding_error(slf.py(), e))?
        {
            return Err(crate::runtime_error("session is closed"));
        }
        Ok(slf)
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _ty: &Bound<'_, PyAny>,
        _value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.close(py)
    }
    /// Run with all public inputs; retain input owners until completion.
    /// Inputs must remain unmodified during the call.
    fn run(&self, py: Python<'_>, inputs: &Bound<'_, PyAny>) -> PyResult<PyRunResult> {
        let items = inputs.call_method0("items").map_err(|_| {
            PyTypeError::new_err("inputs must be a mapping of qualified names to tensors")
        })?;
        let owned = import_items(py, &items)?;
        let views = owned
            .iter()
            .map(|(name, value)| -> PyResult<_> {
                let descriptor = value
                    .descriptor()
                    .map_err(pyo3::exceptions::PyBufferError::new_err)?;
                // SAFETY: the producer lease retains metadata/storage; import drained CUDA work.
                // The caller must not mutate inputs until run returns.
                Ok((name.as_str(), unsafe { BorrowedTensor::new(descriptor) }))
            })
            .try_fold(HashMap::with_capacity(owned.len()), |mut views, item| {
                let (name, tensor) = item?;
                if views.insert(name, tensor).is_some() {
                    return Err(PyValueError::new_err("duplicate input name"));
                }
                Ok(views)
            })?;
        // Owners stay on this attached stack until inference releases every input.
        let result = py
            .detach(|| self.inner.run(&views))
            .map_err(|e| binding_error(py, e))?;
        Ok(PyRunResult { inner: result })
    }

    /// Restore this session's explicit feedback state to its initial values.
    fn reset(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.reset())
            .map_err(|e| binding_error(py, e))
    }
}

#[pyclass(name = "RunResult", frozen, module = "lrt")]
pub struct PyRunResult {
    inner: RunResult,
}

#[pymethods]
impl PyRunResult {
    fn __getitem__(&self, name: &str) -> PyResult<Output> {
        self.inner
            .get(name)
            .map(|t| Output { inner: t.clone() })
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.iter().count()
    }

    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            self.inner.iter().map(|(name, _)| name).collect::<Vec<_>>(),
        )
    }
}
