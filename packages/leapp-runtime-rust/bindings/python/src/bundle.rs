// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use leapp_runtime::{Bundle, Metadata, MetadataValue};
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};
use std::path::PathBuf;

#[pyclass(name = "PortRef", frozen, get_all, module = "leapp_runtime")]
pub struct PortRef {
    node: String,
    port: String,
}

#[pymethods]
impl PortRef {
    fn name(&self) -> String {
        format!("{}/{}", self.node, self.port)
    }

    fn __repr__(&self) -> String {
        format!("PortRef({:?}, {:?})", self.node, self.port)
    }
}

#[pyclass(frozen, get_all, module = "leapp_runtime")]
pub struct PortSpec {
    name: String,
    dtype: String,
    shape: Vec<i64>,
    tensor_type: String,
    metadata: Py<PyAny>,
}

/// Lossless fallback for YAML mappings containing keys Python cannot hash.
#[pyclass(frozen, get_all, module = "leapp_runtime")]
pub struct MetadataMapping {
    entries: Py<PyAny>,
}

pub fn value(py: Python<'_>, v: &MetadataValue) -> PyResult<Py<PyAny>> {
    Ok(match v {
        MetadataValue::Null => py.None(),
        MetadataValue::Bool(v) => v.into_pyobject(py)?.to_owned().into_any().unbind(),
        MetadataValue::Number(v) => {
            if let Some(n) = v.as_i64() {
                n.into_pyobject(py)?.into_any().unbind()
            } else if let Some(n) = v.as_u64() {
                n.into_pyobject(py)?.into_any().unbind()
            } else {
                v.as_f64()
                    .expect("YAML number")
                    .into_pyobject(py)?
                    .into_any()
                    .unbind()
            }
        }
        MetadataValue::String(v) => v.into_pyobject(py)?.into_any().unbind(),
        MetadataValue::Sequence(v) => {
            let items = v
                .iter()
                .map(|v| value(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any().unbind()
        }
        MetadataValue::Mapping(v) => {
            let entries = v
                .iter()
                .map(|(k, v)| Ok((value(py, k)?, value(py, v)?)))
                .collect::<PyResult<Vec<_>>>()?;
            // Do not stringify or discard YAML keys that have no Python dict equivalent.
            if entries.iter().any(|(k, _)| k.bind(py).hash().is_err()) {
                let entries = PyList::new(py, entries)?.into_any().unbind();
                Py::new(py, MetadataMapping { entries })?.into_any()
            } else {
                let dict = PyDict::new(py);
                let mut collision = false;
                for (key, val) in &entries {
                    collision |= dict.contains(key)?;
                    dict.set_item(key, val)?;
                }
                if collision {
                    let entries = PyList::new(py, entries)?.into_any().unbind();
                    Py::new(py, MetadataMapping { entries })?.into_any()
                } else {
                    dict.into_any().unbind()
                }
            }
        }
        MetadataValue::Tagged(_) => unreachable!("bundle loading rejects YAML tags"),
    })
}

pub fn metadata(py: Python<'_>, data: &Metadata) -> PyResult<Py<PyAny>> {
    let result = PyDict::new(py);
    for (key, val) in data {
        result.set_item(key, value(py, val)?)?;
    }
    Ok(result.into_any().unbind())
}

fn spec(py: Python<'_>, v: Option<&leapp_runtime::PortSpec>) -> PyResult<Option<PortSpec>> {
    v.map(|v| {
        Ok(PortSpec {
            name: v.name.clone(),
            dtype: format!("{:?}", v.dtype).to_lowercase(),
            shape: v.shape.clone(),
            tensor_type: v.tensor_type.clone(),
            metadata: metadata(py, &v.metadata)?,
        })
    })
    .transpose()
}

#[pyclass(name = "Bundle", frozen, module = "leapp_runtime")]
pub struct PyBundle {
    pub inner: Bundle,
}

#[pymethods]
impl PyBundle {
    #[staticmethod]
    /// Parse and validate a bundle without loading its model backends.
    fn open(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| Bundle::open(&path)).map_err(|e| {
            let error = super::diagnostic(
                py,
                super::BundleError::new_err(e.message()),
                e.code(),
                e.phase(),
            );
            if let Err(e) = error.value(py).setattr("path", e.path().to_path_buf()) {
                return e;
            }
            error
        })?;
        Ok(Self { inner })
    }

    fn yaml_path(&self) -> PathBuf {
        self.inner.yaml_path().into()
    }

    fn model_names(&self) -> Vec<String> {
        self.inner
            .model_names()
            .into_iter()
            .map(String::from)
            .collect()
    }

    /// Ordered mapping of qualified input names to detached port descriptors.
    fn pipeline_inputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for port in self.inner.pipeline_inputs() {
            result.set_item(
                port.name(),
                spec(py, self.inner.input_info(&port.node, &port.port))?,
            )?;
        }
        Ok(result)
    }

    /// Ordered mapping of qualified output names to detached port descriptors.
    fn pipeline_outputs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for port in self.inner.pipeline_outputs() {
            result.set_item(
                port.name(),
                spec(py, self.inner.output_info(&port.node, &port.port))?,
            )?;
        }
        Ok(result)
    }

    fn pipeline_configs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        metadata(py, self.inner.pipeline_configs())
    }

    fn pipeline_metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        metadata(py, self.inner.pipeline_metadata())
    }

    fn metadata(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        metadata(py, self.inner.metadata())
    }

    fn system_information(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        metadata(py, self.inner.system_information())
    }

    fn model_parameters(&self, py: Python<'_>, node: &str) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .model_parameters(node)
            .map(|v| metadata(py, v))
            .transpose()
    }

    fn model_metadata(&self, py: Python<'_>, node: &str) -> PyResult<Option<Py<PyAny>>> {
        self.inner
            .model_metadata(node)
            .map(|v| metadata(py, v))
            .transpose()
    }

    fn input_info(&self, py: Python<'_>, node: &str, port: &str) -> PyResult<Option<PortSpec>> {
        spec(py, self.inner.input_info(node, port))
    }

    fn output_info(&self, py: Python<'_>, node: &str, port: &str) -> PyResult<Option<PortSpec>> {
        spec(py, self.inner.output_info(node, port))
    }
}
