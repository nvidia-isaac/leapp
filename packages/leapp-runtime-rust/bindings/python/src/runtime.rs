// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{bundle::PyBundle, dlpack::ImportedTensor, tensor::PyTensor};
use leapp_runtime::{ExecutionDevice, RunOptions, RunResult, RuntimeOptions};
use leapp_runtime_bindings::{Error, RuntimeHandle, SessionHandle};
use pyo3::{
    exceptions::{PyKeyError, PyTypeError},
    prelude::*,
    types::PyTuple,
};
use std::{collections::HashMap, path::PathBuf};

fn binding_error(py: Python<'_>, error: Error) -> PyErr {
    crate::diagnostic(
        py,
        crate::runtime_error(&error),
        error.code(),
        error.phase(),
    )
}

#[pyclass(name = "Runtime", frozen, module = "leapp_runtime")]
pub struct PyRuntime {
    inner: RuntimeHandle,
}

#[pymethods]
impl PyRuntime {
    #[staticmethod]
    #[pyo3(signature = (bundle, options=None))]
    /// Load a bundle for synchronous execution using RuntimeOptions.
    fn open(
        py: Python<'_>,
        bundle: &PyBundle,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut opts = RuntimeOptions::default();
        if let Some(options) = options {
            let device: String = options.getattr("device")?.extract()?;
            opts.device = match crate::location(&device)? {
                leapp_runtime::MemoryLocation::Cpu => ExecutionDevice::Cpu,
                leapp_runtime::MemoryLocation::Cuda { device_id } => {
                    ExecutionDevice::Cuda { device_id }
                }
                _ => {
                    return Err(PyTypeError::new_err(
                        "pinned is a buffer location, not an execution device",
                    ))
                }
            };
            opts.backend_directory = options.getattr("backend_directory")?.extract::<PathBuf>()?;
        }
        let inner = py
            .detach(|| RuntimeHandle::open(&bundle.inner, opts))
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
    fn create_session(&self, py: Python<'_>) -> PyResult<PySession> {
        let inner = py
            .detach(|| self.inner.create_session())
            .map_err(|e| binding_error(py, e))?;
        Ok(PySession { inner })
    }

    /// Wait for active calls and close this runtime and all its sessions.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.close())
            .map_err(|e| binding_error(py, e))
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        let runtime = slf.inner.clone();
        if !slf
            .py()
            .detach(|| runtime.is_open())
            .map_err(|e| binding_error(slf.py(), e))?
        {
            return Err(binding_error(slf.py(), Error::Closed));
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
}

#[pyclass(name = "Session", frozen, module = "leapp_runtime")]
pub struct PySession {
    inner: SessionHandle,
}

#[pymethods]
impl PySession {
    #[pyo3(signature = (inputs, *, output_device="cpu", output_overrides=None))]
    /// Run with all public inputs; retain input owners until completion.
    /// Inputs must remain unmodified during the call.
    fn run(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyAny>,
        output_device: &str,
        output_overrides: Option<HashMap<String, String>>,
    ) -> PyResult<PyRunResult> {
        let mut options = RunOptions::default();
        options.output_placement.default_location = crate::location(output_device)?;
        for (name, device) in output_overrides.unwrap_or_default() {
            options
                .output_placement
                .overrides
                .insert(name, crate::location(&device)?);
        }
        let items = inputs.call_method0("items").map_err(|_| {
            PyTypeError::new_err("inputs must be a mapping of qualified names to tensors")
        })?;
        let owned = ImportedTensor::from_items(py, &items)?;
        let views = owned
            .iter()
            .map(|(name, value)| Ok((name.as_str(), value.view()?)))
            .collect::<PyResult<Vec<_>>>()?;
        // Owners stay on this attached stack until inference releases every input.
        let result = py
            .detach(|| self.inner.run(&views, &options))
            .map_err(|e| binding_error(py, e))?;
        Ok(PyRunResult { inner: result })
    }

    /// Restore this session's explicit feedback state to its initial values.
    fn reset(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.reset())
            .map_err(|e| binding_error(py, e))
    }
}

#[pyclass(name = "RunResult", frozen, module = "leapp_runtime")]
pub struct PyRunResult {
    inner: RunResult,
}

#[pymethods]
impl PyRunResult {
    fn __getitem__(&self, name: &str) -> PyResult<PyTensor> {
        self.inner
            .get(name)
            .map(|t| PyTensor {
                inner: t.native().clone(),
            })
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

    /// Return a retained, read-only NumPy view of a host output.
    fn numpy(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        Py::new(py, self.__getitem__(name)?)?
            .bind(py)
            .call_method0("numpy")
            .map(Bound::unbind)
    }
}
