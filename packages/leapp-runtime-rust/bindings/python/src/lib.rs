// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(unsafe_op_in_unsafe_fn)]

mod bundle;

use leapp_runtime::ErrorCode;
use pyo3::{create_exception, exceptions::PyException, prelude::*};

create_exception!(leapp_runtime, BundleError, PyException);

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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BundleError", m.py().get_type::<BundleError>())?;
    m.add_class::<bundle::PyBundle>()?;
    m.add_class::<bundle::PortRef>()?;
    m.add_class::<bundle::PortSpec>()?;
    m.add_class::<bundle::TaggedValue>()?;
    m.add_class::<bundle::MetadataMapping>()?;
    Ok(())
}
