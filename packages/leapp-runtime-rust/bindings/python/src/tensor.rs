// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use leapp_triton::{DataType, MemoryLocation, Tensor};
use pyo3::{
    exceptions::PyBufferError,
    prelude::*,
    types::{PyDict, PyTuple},
};

#[pyclass(name = "Tensor", frozen, module = "leapp_runtime")]
pub struct PyTensor {
    pub inner: Tensor,
}

#[pyclass(frozen, module = "leapp_runtime")]
pub struct DLPackView {
    pub inner: Tensor,
}

fn typestr(dtype: DataType) -> PyResult<&'static str> {
    use DataType::*;

    Ok(match dtype {
        Bool => "|b1",
        Uint8 => "|u1",
        Int8 => "|i1",
        Uint16 => "=u2",
        Int16 => "=i2",
        Uint32 => "=u4",
        Int32 => "=i4",
        Uint64 => "=u8",
        Int64 => "=i8",
        Float16 => "=f2",
        Float32 => "=f4",
        Float64 => "=f8",
        Bfloat16 => {
            return Err(PyBufferError::new_err(
                "NumPy has no native bfloat16 dtype; use DLPack",
            ))
        }
    })
}

#[pymethods]
impl PyTensor {
    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.dtype()).to_lowercase()
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.shape().iter().copied())
    }

    #[getter]
    fn device(&self) -> String {
        crate::device_name(self.inner.location())
    }

    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if matches!(self.inner.location(), MemoryLocation::Cuda { .. }) {
            return Err(PyBufferError::new_err(
                "CUDA tensors require an explicit host transfer",
            ));
        }
        let result = PyDict::new(py);
        result.set_item("version", 3)?;
        result.set_item("shape", self.shape(py)?)?;
        result.set_item("typestr", typestr(self.inner.dtype())?)?;
        result.set_item("data", (self.inner.data_ptr() as usize, true))?;
        Ok(result)
    }

    /// Return a zero-copy, read-only host view retaining this tensor's storage.
    fn numpy(slf: Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let numpy = slf.py().import("numpy")?;
        // NumPy retains this object as its base and honors the read-only data flag.
        Ok(numpy.call_method1("asarray", (slf,))?.unbind())
    }

    /// Explicit zero-copy export. Consumers must not mutate the shared allocation.
    fn dlpack_view(&self) -> DLPackView {
        DLPackView {
            inner: self.inner.clone(),
        }
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        crate::dlpack::device(self.inner.location())
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
        if copy == Some(false) {
            return Err(PyBufferError::new_err(
                "use tensor.dlpack_view() to explicitly request shared read-only storage",
            ));
        }
        crate::dlpack::export(py, &self.inner, false, stream, max_version, dl_device)
    }
}

#[pymethods]
impl DLPackView {
    fn __dlpack_device__(&self) -> (i32, i32) {
        crate::dlpack::device(self.inner.location())
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
        crate::dlpack::export(
            py,
            &self.inner,
            copy != Some(true),
            stream,
            max_version,
            dl_device,
        )
    }
}
