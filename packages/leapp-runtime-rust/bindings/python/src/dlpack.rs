// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DLPack ABI adapters. Capsule owners remain alive until synchronous input release.
use crate::tensor::PyTensor;
use leapp_triton::{DataType, MemoryLocation, Tensor, TensorView};
use pyo3::{
    exceptions::{PyBufferError, PyTypeError},
    ffi,
    prelude::*,
    types::PyDict,
};
use std::{
    cell::RefCell,
    ffi::{c_char, c_void},
    ptr,
    sync::Arc,
};

thread_local! {
    static HANDOFF_STREAMS: RefCell<Vec<(u32, Arc<leapp_triton::HandoffStream>)>> = const { RefCell::new(Vec::new()) };
}

fn handoff_stream(device: u32) -> leapp_triton::Result<Arc<leapp_triton::HandoffStream>> {
    HANDOFF_STREAMS.with(|streams| {
        let mut streams = streams.borrow_mut();
        if let Some((_, stream)) = streams.iter().find(|(id, _)| *id == device) {
            return Ok(stream.clone());
        }
        let stream = Arc::new(leapp_triton::HandoffStream::new(device)?);
        streams.push((device, stream.clone()));
        Ok(stream)
    })
}

const CPU: i32 = 1;
const CUDA: i32 = 2;
const PINNED: i32 = 3;
const READ_ONLY: u64 = 1;
const IS_COPIED: u64 = 2;
const LEGACY: *const c_char = c"dltensor".as_ptr();
const VERSIONED: *const c_char = c"dltensor_versioned".as_ptr();
const USED_LEGACY: *const c_char = c"used_dltensor".as_ptr();
const USED_VERSIONED: *const c_char = c"used_dltensor_versioned".as_ptr();

#[repr(C)]
#[derive(Clone, Copy)]
struct Device {
    kind: i32,
    id: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Dtype {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DlTensor {
    data: *mut c_void,
    device: Device,
    ndim: i32,
    dtype: Dtype,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct Legacy {
    tensor: DlTensor,
    context: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut Legacy)>,
}

#[repr(C)]
struct Version {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct Versioned {
    version: Version,
    context: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut Versioned)>,
    flags: u64,
    tensor: DlTensor,
}

enum Managed {
    Legacy(*mut Legacy),
    Versioned(*mut Versioned),
}

impl Managed {
    fn tensor(&self) -> PyResult<DlTensor> {
        // SAFETY: pointers come from named capsules supplied by conforming producers.
        unsafe {
            match self {
                Self::Legacy(p) => Ok((**p).tensor),
                Self::Versioned(p) => {
                    if (**p).version.major != 1 {
                        return Err(PyBufferError::new_err("unsupported DLPack major version"));
                    }
                    Ok((**p).tensor)
                }
            }
        }
    }
}

impl Drop for Managed {
    fn drop(&mut self) {
        // SAFETY: consumption renamed the capsule. This guard exclusively owns
        // its managed tensor and invokes its producer deleter exactly once.
        unsafe {
            match self {
                Self::Legacy(p) => {
                    if let Some(f) = (**p).deleter {
                        f(*p);
                    }
                }
                Self::Versioned(p) => {
                    if let Some(f) = (**p).deleter {
                        f(*p);
                    }
                }
            }
        }
    }
}

pub fn device(location: MemoryLocation) -> (i32, i32) {
    match location {
        MemoryLocation::Cpu => (CPU, 0),
        MemoryLocation::PinnedCpu => (PINNED, 0),
        MemoryLocation::Cuda { device_id } => (CUDA, device_id as i32),
    }
}

fn dtype(d: Dtype) -> PyResult<DataType> {
    if d.lanes != 1 {
        return Err(PyBufferError::new_err(
            "vector-lane DLPack dtypes are unsupported",
        ));
    }

    use DataType::*;

    Ok(match (d.code, d.bits) {
        (0, 8) => Int8,
        (0, 16) => Int16,
        (0, 32) => Int32,
        (0, 64) => Int64,
        (1, 8) => Uint8,
        (1, 16) => Uint16,
        (1, 32) => Uint32,
        (1, 64) => Uint64,
        (2, 16) => Float16,
        (2, 32) => Float32,
        (2, 64) => Float64,
        (4, 16) => Bfloat16,
        (6, 8) => Bool,
        _ => return Err(PyBufferError::new_err("unsupported DLPack dtype")),
    })
}

fn dl_dtype(d: DataType) -> Dtype {
    use DataType::*;

    let code = match d {
        Int8 | Int16 | Int32 | Int64 => 0,
        Uint8 | Uint16 | Uint32 | Uint64 => 1,
        Float16 | Float32 | Float64 => 2,
        Bfloat16 => 4,
        Bool => 6,
    };
    Dtype {
        code,
        bits: (d.byte_width() * 8) as u8,
        lanes: 1,
    }
}

fn memory((kind, id): (i32, i32)) -> PyResult<MemoryLocation> {
    match (kind, id) {
        (CPU, 0) => Ok(MemoryLocation::Cpu),
        (PINNED, 0) => Ok(MemoryLocation::PinnedCpu),
        (CUDA, id) if id >= 0 => Ok(MemoryLocation::Cuda {
            device_id: id as u32,
        }),
        _ => Err(PyBufferError::new_err("unsupported DLPack device")),
    }
}

// These fields intentionally only retain owners, including their cleanup behavior.
#[allow(dead_code)]
enum Owner {
    Python(Py<PyAny>),
    Native(Tensor),
    Managed(Managed),
}

pub struct ImportedTensor {
    shape: Vec<i64>,
    dtype: DataType,
    ptr: *const c_void,
    location: MemoryLocation,
    _owner: Owner,
}

type ProducerStream<'py> = (u32, usize, Option<Bound<'py, PyAny>>);

#[derive(Default)]
struct ImportBatch<'py> {
    torch_checked: bool,
    torch: Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    streams: Vec<ProducerStream<'py>>,
}

impl ImportedTensor {
    /// Import the entire mapping before waiting, so contiguous copies created by
    /// later inputs finish too. Returned tensors are ready for synchronous use.
    pub fn from_items(py: Python<'_>, items: &Bound<'_, PyAny>) -> PyResult<Vec<(String, Self)>> {
        let mut batch = ImportBatch::default();
        let mut owned = Vec::new();
        let imported = (|| {
            for item in items.try_iter()? {
                let (name, value): (String, Bound<'_, PyAny>) = item?.extract()?;
                owned.push((name, Self::new(py, &value, &mut batch)?));
            }
            Ok::<(), PyErr>(())
        })();
        // Drain every recorded stream even if a later import or wait fails.
        // Keep earlier capsule owners alive until all materializations finish.
        let mut ready = Ok(());
        for (device, address, stream) in batch.streams {
            let result = if let Some(stream) = stream {
                stream.call_method0("synchronize").map(|_| ())
            } else {
                // SAFETY: Torch owns this stream and inputs retain its storage
                // through the synchronous wait on the producer's device.
                py.detach(|| unsafe { leapp_triton::synchronize_external_stream(device, address) })
                    .map_err(crate::native_error)
            };
            if ready.is_ok() {
                ready = result;
            }
        }
        imported?;
        ready?;

        Ok(owned)
    }

    fn new<'py>(
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
        batch: &mut ImportBatch<'py>,
    ) -> PyResult<Self> {
        if let Ok(t) = obj.extract::<PyRef<'_, PyTensor>>() {
            let tensor = t.inner.clone();
            // This borrow completes synchronously; it does not expose a pointer
            // to a foreign consumer whose work could outlive the input owner.
            return Ok(Self {
                shape: tensor.shape().into(),
                dtype: tensor.dtype(),
                ptr: ptr::null(),
                location: tensor.location(),
                _owner: Owner::Native(tensor),
            });
        }
        if !batch.torch_checked {
            let modules = py.import("sys")?.getattr("modules")?;
            if let Some(torch) = modules.cast::<PyDict>()?.get_item("torch")? {
                if !torch.is_none() {
                    let tensor_type = torch.getattr("Tensor")?;
                    batch.torch = Some((torch, tensor_type));
                }
            }
            batch.torch_checked = true;
        }
        if let Some((torch, tensor_type)) = &batch.torch {
            if obj.get_type().is(tensor_type) {
                let torch = torch.clone();
                return Self::torch(py, obj, &torch, batch);
            }
        }
        let numpy = py.import("numpy")?;
        if obj.is_instance(&numpy.getattr("ndarray")?)? {
            let array = if c_contiguous(obj)? {
                obj.clone()
            } else {
                numpy.call_method1("ascontiguousarray", (obj,))?
            };
            return Self::numpy(&array);
        }
        // Recognize loaded array libraries without making them dependencies.
        // Generic DLPack producers have no standard contiguous-copy operation.
        let modules = py.import("sys")?.getattr("modules")?;
        let modules = modules.cast::<PyDict>()?;
        if let Some(torch) = modules.get_item("torch")? {
            if !torch.is_none() && obj.is_instance(&torch.getattr("Tensor")?)? {
                let tensor = if obj.call_method0("is_contiguous")?.extract::<bool>()? {
                    obj.clone()
                } else {
                    obj.call_method0("contiguous")?
                };
                if tensor.get_type().is(&torch.getattr("Tensor")?) {
                    return Self::torch(py, &tensor, &torch, batch);
                }
                return Self::dlpack(py, &tensor);
            }
        }
        if let Some(cupy) = modules.get_item("cupy")? {
            if !cupy.is_none() && obj.is_instance(&cupy.getattr("ndarray")?)? {
                let array = if c_contiguous(obj)? {
                    obj.clone()
                } else {
                    cupy.call_method1("ascontiguousarray", (obj,))?
                };
                return Self::dlpack(py, &array);
            }
        }
        Self::dlpack(py, obj)
    }

    fn numpy(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dt = obj.getattr("dtype")?;
        if !dt.getattr("isnative")?.extract::<bool>()? {
            return Err(PyBufferError::new_err(
                "non-native byte order is unsupported",
            ));
        }
        let name: String = dt.getattr("name")?.extract()?;

        use DataType::*;

        let dtype = match name.as_str() {
            "bool" => Bool,
            "int8" => Int8,
            "int16" => Int16,
            "int32" => Int32,
            "int64" => Int64,
            "uint8" => Uint8,
            "uint16" => Uint16,
            "uint32" => Uint32,
            "uint64" => Uint64,
            "float16" => Float16,
            "float32" => Float32,
            "float64" => Float64,
            "bfloat16" => Bfloat16,
            _ => return Err(PyBufferError::new_err("unsupported NumPy dtype")),
        };
        let info = obj.getattr("__array_interface__")?;
        let (address, _readonly): (usize, bool) = info.get_item("data")?.extract()?;
        Ok(Self {
            shape: obj.getattr("shape")?.extract()?,
            dtype,
            ptr: address as *const c_void,
            location: MemoryLocation::Cpu,
            _owner: Owner::Python(obj.clone().unbind()),
        })
    }

    fn torch<'py>(
        py: Python<'py>,
        obj: &Bound<'py, PyAny>,
        torch: &Bound<'py, PyAny>,
        batch: &mut ImportBatch<'py>,
    ) -> PyResult<Self> {
        // Only exact torch.Tensor objects use the capsule API directly. Tensor
        // subclasses retain their __dlpack__ dispatch and producer semantics.
        if obj.getattr("requires_grad")?.extract::<bool>()? {
            return Err(PyBufferError::new_err(
                "Can't export tensors that require gradient, use tensor.detach()",
            ));
        }
        let capsule = torch
            .getattr("utils")?
            .getattr("dlpack")?
            .call_method1("to_dlpack", (obj,))?;
        let managed = consume(py, &capsule)?;
        // Conjugate views are complex. Other supported dtypes cannot carry this
        // bit, so they need no additional Python metadata call.
        if managed.tensor()?.dtype.code == 5 && obj.call_method0("is_conj")?.extract::<bool>()? {
            return Err(PyBufferError::new_err(
                "Can't export tensors with the conjugate bit set",
            ));
        }
        // Inspect native stride metadata once. Only strided inputs need a Python
        // contiguous() call and a second capsule; generic producers still reject them.
        let (imported, contiguous) = Self::parse(managed, None)?;

        if let MemoryLocation::Cuda { device_id } = imported.location {
            // The capsule helper does not transfer stream dependencies. Record
            // every producer stream and wait once per stream after the full batch.
            // Avoid constructing a Python Stream wrapper when Torch exposes the
            // native handle. Older Torch versions retain the public API fallback.
            let (address, stream) = match torch.getattr("_C")?.getattr("_cuda_getCurrentRawStream")
            {
                Ok(current) => (current.call1((device_id,))?.extract::<usize>()?, None),
                Err(error) if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => {
                    let stream = torch
                        .getattr("cuda")?
                        .call_method1("current_stream", (device_id,))?;
                    (stream.getattr("cuda_stream")?.extract()?, Some(stream))
                }
                Err(error) => return Err(error),
            };
            if !batch
                .streams
                .iter()
                .any(|(id, ptr, _)| *id == device_id && *ptr == address)
            {
                batch.streams.push((device_id, address, stream));
            }
        }
        let imported = if contiguous {
            imported
        } else {
            let contiguous = obj.call_method0("contiguous")?;
            let capsule = torch
                .getattr("utils")?
                .getattr("dlpack")?
                .call_method1("to_dlpack", (&contiguous,))?;
            Self::from_capsule(py, &capsule, None)?
        };

        Ok(imported)
    }

    fn dlpack(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let declared: (i32, i32) = obj
            .call_method0("__dlpack_device__")
            .map_err(|_| {
                PyTypeError::new_err("expected NumPy, LEAPP Tensor, or a DLPack producer")
            })?
            .extract()?;
        let location = memory(declared)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("max_version", (1, 0))?;

        let handoff = if let MemoryLocation::Cuda { device_id } = location {
            let stream = py
                .detach(|| handoff_stream(device_id))
                .map_err(crate::native_error)?;
            kwargs.set_item("stream", stream.address())?;
            Some(stream)
        } else {
            None
        };

        let export = || match obj.call_method("__dlpack__", (), Some(&kwargs)) {
            Ok(c) => Ok(c),
            Err(e) if e.is_instance_of::<PyTypeError>(py) => {
                // Older producers accept stream but not max_version.
                kwargs.del_item("max_version")?;
                obj.call_method("__dlpack__", (), Some(&kwargs))
            }
            Err(e) => Err(e),
        };

        let capsule = match &handoff {
            Some(stream) => {
                let exported = stream.with_device(export).map_err(crate::native_error)?;
                // A producer may enqueue work before returning an error.
                let ready = py
                    .detach(|| stream.synchronize())
                    .map_err(crate::native_error);
                let capsule = exported?;
                ready?;
                capsule
            }
            None => export()?,
        };

        Self::from_capsule(py, &capsule, Some(declared))
    }

    fn from_capsule(
        py: Python<'_>,
        capsule: &Bound<'_, PyAny>,
        declared: Option<(i32, i32)>,
    ) -> PyResult<Self> {
        let (imported, contiguous) = Self::parse(consume(py, capsule)?, declared)?;
        if !contiguous {
            return Err(PyBufferError::new_err("generic DLPack inputs must be C-contiguous; materialize a contiguous array before calling run"));
        }
        Ok(imported)
    }

    fn parse(managed: Managed, declared: Option<(i32, i32)>) -> PyResult<(Self, bool)> {
        let t = managed.tensor()?;
        let actual = (t.device.kind, t.device.id);
        // PyTorch can advertise pinned host storage but export a CPU capsule.
        // Use the capsule's conservative CPU location, not the stronger claim.
        if let Some(declared) = declared {
            if actual != declared && !(declared == (PINNED, 0) && actual == (CPU, 0)) {
                return Err(PyBufferError::new_err(
                    "DLPack device descriptor changed during import",
                ));
            }
        }
        let location = memory(actual)?;
        if t.ndim < 0 || (t.ndim != 0 && t.shape.is_null()) {
            return Err(PyBufferError::new_err("invalid DLPack shape descriptor"));
        }
        let dtype = dtype(t.dtype)?;
        // SAFETY: a conforming capsule contains ndim readable shape/stride entries.
        let shape = if t.ndim == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(t.shape, t.ndim as usize).to_vec() }
        };
        let mut elements = 1usize;
        for &d in &shape {
            let d = usize::try_from(d)
                .map_err(|_| PyBufferError::new_err("negative tensor dimension"))?;
            elements = elements
                .checked_mul(d)
                .ok_or_else(|| PyBufferError::new_err("tensor size overflow"))?;
        }
        let mut contiguous = true;
        if !t.strides.is_null() && elements != 0 {
            let mut stride = 1i64;
            for i in (0..shape.len()).rev() {
                // SAFETY: stride metadata is valid for ndim entries.
                if shape[i] > 1 && unsafe { *t.strides.add(i) } != stride {
                    contiguous = false;
                }
                stride = stride
                    .checked_mul(shape[i])
                    .ok_or_else(|| PyBufferError::new_err("stride overflow"))?;
            }
        }
        let size = elements
            .checked_mul(dtype.byte_width())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| PyBufferError::new_err("tensor byte size overflow"))?;
        let offset = usize::try_from(t.byte_offset)
            .map_err(|_| PyBufferError::new_err("byte offset overflow"))?;
        let address = (t.data as usize)
            .checked_add(offset)
            .and_then(|p| p.checked_add(size).map(|_| p))
            .ok_or_else(|| PyBufferError::new_err("tensor address overflow"))?;
        if size != 0 && t.data.is_null() {
            return Err(PyBufferError::new_err("null tensor storage"));
        }
        Ok((
            Self {
                shape,
                dtype,
                ptr: address as *const c_void,
                location,
                _owner: Owner::Managed(managed),
            },
            contiguous,
        ))
    }

    pub fn view(&self) -> PyResult<leapp_runtime::TensorView<'_>> {
        if let Owner::Native(tensor) = &self._owner {
            return Ok(tensor.view().into());
        }
        // SAFETY: Python owners and capsule guards retain storage; import waited
        // for producer work. Callers must not mutate inputs before run returns.
        unsafe { TensorView::from_raw_parts(self.ptr, self.dtype, &self.shape, self.location) }
            .map(Into::into)
            .map_err(crate::native_error)
    }
}

fn c_contiguous(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.getattr("flags")?.getattr("c_contiguous")?.extract()
}

fn consume(py: Python<'_>, capsule: &Bound<'_, PyAny>) -> PyResult<Managed> {
    // SAFETY: only inspect pointers after checking the capsule's exact ABI name.
    unsafe {
        let (name, used, versioned) = if ffi::PyCapsule_IsValid(capsule.as_ptr(), VERSIONED) != 0 {
            (VERSIONED, USED_VERSIONED, true)
        } else if ffi::PyCapsule_IsValid(capsule.as_ptr(), LEGACY) != 0 {
            (LEGACY, USED_LEGACY, false)
        } else {
            return Err(PyBufferError::new_err(
                "invalid or already consumed DLPack capsule",
            ));
        };
        let ptr = ffi::PyCapsule_GetPointer(capsule.as_ptr(), name);
        if ptr.is_null() {
            return Err(PyErr::fetch(py));
        }
        if ffi::PyCapsule_SetName(capsule.as_ptr(), used) != 0 {
            return Err(PyErr::fetch(py));
        }
        Ok(if versioned {
            Managed::Versioned(ptr.cast())
        } else {
            Managed::Legacy(ptr.cast())
        })
    }
}

unsafe extern "C" fn delete_legacy(ptr: *mut Legacy) {
    // SAFETY: this deleter receives the Box created by export, exactly once.
    unsafe {
        let owned = Box::from_raw(ptr);
        drop(Box::from_raw(owned.context.cast::<Tensor>()));
    }
}

unsafe extern "C" fn delete_versioned(ptr: *mut Versioned) {
    // SAFETY: this deleter receives the Box created by export, exactly once.
    unsafe {
        let owned = Box::from_raw(ptr);
        drop(Box::from_raw(owned.context.cast::<Tensor>()));
    }
}

unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: called by Python with a live capsule; consumed capsules are owned by
    // their consumer and must not be deleted here.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, LEGACY) != 0 {
            delete_legacy(ffi::PyCapsule_GetPointer(capsule, LEGACY).cast());
        } else if ffi::PyCapsule_IsValid(capsule, VERSIONED) != 0 {
            delete_versioned(ffi::PyCapsule_GetPointer(capsule, VERSIONED).cast());
        }
    }
}

pub fn export(
    py: Python<'_>,
    tensor: &Tensor,
    shared: bool,
    stream: Option<i64>,
    max_version: Option<(u32, u32)>,
    target: Option<(i32, i32)>,
) -> PyResult<Py<PyAny>> {
    let native_device = device(tensor.location());
    if target.is_some_and(|d| d != native_device) {
        return Err(PyBufferError::new_err(
            "cross-device DLPack export is unsupported",
        ));
    }
    if stream == Some(0)
        || (native_device.0 != CUDA && stream.is_some())
        || stream.is_some_and(|s| s < -1)
    {
        return Err(PyBufferError::new_err("invalid DLPack stream"));
    }
    // Native inference and deep_copy both complete before publishing data, so
    // the returned allocation is ready on any consumer stream.
    let owned = if shared {
        tensor.clone()
    } else {
        py.detach(|| tensor.deep_copy())
            .map_err(crate::native_error)?
    };
    let owned = Box::new(owned);
    let descriptor = DlTensor {
        data: owned.data_ptr().cast_mut(),
        device: Device {
            kind: native_device.0,
            id: native_device.1,
        },
        ndim: i32::try_from(owned.shape().len())
            .map_err(|_| PyBufferError::new_err("too many tensor dimensions"))?,
        dtype: dl_dtype(owned.dtype()),
        shape: owned.shape().as_ptr().cast_mut(),
        strides: ptr::null_mut(),
        byte_offset: 0,
    };
    let context = Box::into_raw(owned).cast();
    let versioned = max_version.is_some_and(|v| v >= (1, 0));
    let (pointer, name) = if versioned {
        (
            Box::into_raw(Box::new(Versioned {
                version: Version { major: 1, minor: 0 },
                context,
                deleter: Some(delete_versioned),
                flags: if shared { READ_ONLY } else { IS_COPIED },
                tensor: descriptor,
            }))
            .cast::<c_void>(),
            VERSIONED,
        )
    } else {
        (
            Box::into_raw(Box::new(Legacy {
                tensor: descriptor,
                context,
                deleter: Some(delete_legacy),
            }))
            .cast::<c_void>(),
            LEGACY,
        )
    };
    // SAFETY: the capsule owns the managed tensor until consumed; its context
    // retains both allocation and shape metadata independently of Python objects.
    unsafe {
        let capsule = ffi::PyCapsule_New(pointer, name, Some(capsule_destructor));
        if capsule.is_null() {
            if versioned {
                delete_versioned(pointer.cast());
            } else {
                delete_legacy(pointer.cast());
            }
            return Err(PyErr::fetch(py));
        }
        Ok(Bound::from_owned_ptr(py, capsule).unbind())
    }
}
