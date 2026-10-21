// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_op_in_unsafe_fn)]

mod bundle;
mod dlpack;
mod session;

use leapp_runtime::{ErrorCode, MemoryLocation};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyValueError},
    prelude::*,
};

create_exception!(leapp_runtime, BundleError, PyException);

create_exception!(leapp_runtime, RuntimeError, PyException);

fn diagnostic(py: Python<'_>, error: PyErr, code: ErrorCode, phase: &str) -> PyErr {
    let value = error.value(py);
    let result = (|| -> PyResult<()> {
        let code = py
            .import("leapp_runtime")?
            .getattr("ErrorCode")?
            .call1((code as i32,))?;
        value.setattr("code", code)?;
        value.setattr("phase", phase)?;
        value.setattr("message", value.str()?)?;
        Ok(())
    })();
    result.err().unwrap_or(error)
}

fn runtime_error(error: impl std::fmt::Display) -> PyErr {
    RuntimeError::new_err(error.to_string())
}

fn execution_error(py: Python<'_>, error: leapp_runtime::RuntimeError) -> PyErr {
    diagnostic(py, runtime_error(&error), error.code(), error.phase())
}

fn location(text: &str) -> PyResult<MemoryLocation> {
    match text {
        "cpu" => Ok(MemoryLocation::Cpu),
        "pinned" => Ok(MemoryLocation::PinnedCpu),
        _ => text
            .strip_prefix("cuda:")
            .and_then(|s| s.parse().ok())
            .map(|device_id| MemoryLocation::Cuda { device_id })
            .ok_or_else(|| PyValueError::new_err("expected cpu, pinned, or cuda:<device_id>")),
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BundleError", m.py().get_type::<BundleError>())?;
    m.add("RuntimeError", m.py().get_type::<RuntimeError>())?;
    m.add_class::<bundle::PyBundle>()?;
    m.add_class::<bundle::PortRef>()?;
    m.add_class::<bundle::PortSpec>()?;
    m.add_class::<bundle::MetadataMapping>()?;
    m.add_class::<session::PySession>()?;
    m.add_class::<session::PyRunResult>()?;
    m.add_class::<dlpack::Output>()?;
    Ok(())
}
