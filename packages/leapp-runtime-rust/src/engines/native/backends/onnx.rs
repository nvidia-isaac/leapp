// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ONNX loading and cached CPU/CUDA bindings, refactored from the native prototype.

use super::BackendPlugin;
use crate::{
    bundle::schema::NodeSpec,
    dlpack::{BorrowedTensor, DLTensor, SharedTensor},
    engines::memory::{cuda, dlpack},
    DataType, MemoryLocation, PortSpec,
};
use anyhow::{ensure, Context, Result};
use ort::{
    memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
    session::{builder::GraphOptimizationLevel, IoBinding, Session},
    value::{DynTensorValueType, DynValue, Outlet, TensorElementType, ValueType},
    AsPointer,
};
use std::{
    collections::HashMap,
    ffi::{c_char, CString},
    path::PathBuf,
    ptr::{self, NonNull},
    sync::{Arc, Mutex},
};

// ORT's library/environment are process-wide; execution contexts are not.
// Keep the environment with the dynamically loaded library. In particular,
// ort's late ELF finalizer must not release it after CUDA's C++ globals exit.
static LIBRARY: Mutex<Option<(PathBuf, Arc<ort::environment::Environment>)>> = Mutex::new(None);

fn initialize(directory: &std::path::Path) -> Result<()> {
    let path = std::env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| directory.join("onnxruntime/libonnxruntime.so"));
    let path = path
        .canonicalize()
        .with_context(|| format!("ONNX Runtime library {}", path.display()))?;
    let mut loaded = LIBRARY
        .lock()
        .map_err(|_| anyhow::anyhow!("ORT initialization lock poisoned"))?;
    if let Some((previous, _environment)) = loaded.as_ref() {
        ensure!(
            *previous == path,
            "ONNX Runtime already loaded from {}; cannot load {}",
            previous.display(),
            path.display()
        );
    } else {
        ort::init_from(&path)?.with_name("leapp").commit();
        *loaded = Some((path, ort::environment::Environment::current()?));
    }
    Ok(())
}

// Bindings must drop before the SDK session and the workspace they reference.
pub(in crate::engines::native) struct Model {
    prepared: Option<Prepared>,
    loaded: Arc<Loaded>,
}

struct Loaded {
    session: Session,
    custom_ops: bool,
    // Workspace arguments/results follow bundle order, not ONNX graph order.
    input_names: Vec<String>,
    output_names: Vec<String>,
    location: MemoryLocation,
}

impl Model {
    pub fn load(
        model: &NodeSpec,
        location: MemoryLocation,
        stream: Option<crate::CudaStream>,
    ) -> Result<Self> {
        let directory = std::env::var_os("LEAPP_NATIVE_BACKENDS")
            .map(PathBuf::from)
            .unwrap_or_else(|| "/opt/leapp/backends".into());
        initialize(&directory)?;
        location.validate()?;
        let _device = select_device(location)?;
        let mut builder = Session::builder()?
            .with_no_environment_execution_providers()
            .map_err(ort::Error::<()>::from)?
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(ort::Error::<()>::from)?
            .with_parallel_execution(false)
            .map_err(ort::Error::<()>::from)?
            .with_independent_thread_pool()
            .map_err(ort::Error::<()>::from)?;
        if let MemoryLocation::Cuda { device_id } = location {
            let mut provider = ort::ep::CUDA::default()
                .with_device_id(i32::try_from(device_id)?)
                .with_tf32(false);
            if let Some(stream) = stream {
                // SAFETY: initialization validated this stream. CudaStream's
                // caller retains it through every peer sharing the ORT session.
                provider = unsafe { provider.with_compute_stream(stream.as_raw().cast()) };
            }
            builder = builder
                .with_execution_providers([provider.build().error_on_failure()])
                .map_err(ort::Error::<()>::from)?;
        }
        // Reuse the optional Warp custom-op adapter installed for ONNX Runtime.
        let custom = directory.join("onnxruntime/libleapp_wrp_onnx_custom_op.so");
        let custom_ops = match custom.symlink_metadata() {
            Ok(_) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("inspect ONNX custom operators {}", custom.display())
                });
            }
        };
        if custom_ops {
            builder = builder
                .with_operator_library(&custom)
                .map_err(ort::Error::<()>::from)
                .with_context(|| format!("register ONNX custom operators {}", custom.display()))?;
        }
        let session = builder
            .commit_from_file(&model.parameters.model_path)
            .with_context(|| {
                format!("load ONNX model {}", model.parameters.model_path.display())
            })?;
        let input_names = check_ports(&model.inputs, session.inputs()).context("ONNX inputs")?;
        let output_names =
            check_ports(&model.outputs, session.outputs()).context("ONNX outputs")?;
        Ok(Self {
            prepared: None,
            loaded: Arc::new(Loaded {
                session,
                custom_ops,
                input_names,
                output_names,
                location,
            }),
        })
    }
}

impl BackendPlugin for Model {
    fn needs_guard(&self) -> bool {
        self.loaded.custom_ops
    }

    fn new_session(&self) -> Result<Box<dyn BackendPlugin>> {
        Ok(Box::new(Self {
            prepared: None,
            loaded: self.loaded.clone(),
        }))
    }

    fn can_prepare(&self) -> bool {
        true
    }

    fn is_prepared(&self) -> bool {
        self.prepared.is_some()
    }

    #[cfg(test)]
    fn loaded_identity(&self) -> usize {
        self.loaded.session.ptr() as usize
    }

    #[cfg(test)]
    fn prepared_identity(&self) -> Option<Vec<usize>> {
        let prepared = self.prepared.as_ref()?;
        Some(
            prepared
                ._values
                .iter()
                .map(|value| value.ptr() as usize)
                .chain(prepared.name_ptrs.iter().map(|&name| name as usize))
                .chain([
                    prepared.inputs.as_ptr() as usize,
                    prepared.outputs.as_ptr() as usize,
                ])
                .collect(),
        )
    }

    /// # Safety
    /// Storage must remain fixed and exclusively owned until this model is dropped.
    unsafe fn prepare(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        outputs: &[BorrowedTensor<'_>],
    ) -> Result<()> {
        let _device = select_device(self.loaded.location)?;
        let prepare = || -> Result<Prepared> {
            let session = &self.loaded.session;
            ensure!(
                inputs.len() == session.inputs().len() && outputs.len() == session.outputs().len(),
                "ONNX port count mismatch"
            );
            let names = self
                .loaded
                .input_names
                .iter()
                .chain(&self.loaded.output_names)
                .map(|name| CString::new(name.as_str()).map_err(Into::into))
                .collect::<Result<Vec<_>>>()?;
            let mut values = Vec::with_capacity(names.len());
            for &input in inputs {
                // SAFETY: the workspace keeps this read-only region alive until
                // the prepared call is dropped, before releasing its allocations.
                values.push(unsafe { borrowed_value(input)? });
            }
            for &writable in outputs {
                // SAFETY: the caller retains exclusive fixed output storage.
                values.push(unsafe { borrowed_value(writable)? });
            }
            let count = inputs.len();
            let name_ptrs = names.iter().map(|n| n.as_ptr()).collect();
            let inputs = values[..count].iter().map(|v| v.ptr()).collect();
            let outputs = values[count..].iter_mut().map(|v| v.ptr_mut()).collect();
            let binding = if matches!(self.loaded.location, MemoryLocation::Cuda { .. }) {
                let mut binding = session.create_binding()?;
                for (name, value) in names[count..].iter().zip(&values[count..]) {
                    // SAFETY: values and names are retained by Prepared; bindings
                    // are destroyed before these wrappers and workspace storage.
                    unsafe {
                        ort::Error::result_from_status((ort::api().BindOutput)(
                            binding.ptr_mut(),
                            name.as_ptr(),
                            value.ptr(),
                        ))?;
                    }
                }
                Some(binding)
            } else {
                None
            };
            Ok(Prepared {
                binding,
                _values: values,
                _names: names,
                name_ptrs,
                inputs,
                outputs,
            })
        };
        self.prepared = Some(prepare()?);
        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        let mut run = || -> Result<()> {
            let prepared = self.prepared.as_mut().context("unprepared ONNX model")?;
            let session = &self.loaded.session;
            let mut completion = Completion::new(self.loaded.location)?;
            // SAFETY: workspace storage is exclusively owned under its workspace mutex,
            // all pointer arrays/bindings are workspace-private. ORT supports concurrent
            // runs of this shared session (CUDA graph capture remains disabled).
            // Default run options synchronize providers before successful return.
            let result = unsafe {
                if let Some(binding) = &mut prepared.binding {
                    for (name, value) in prepared.name_ptrs.iter().zip(&prepared.inputs) {
                        // ORT can stage CPU-only controls during BindInput; a
                        // stable source address does not imply unchanged contents.
                        ort::Error::result_from_status((ort::api().BindInput)(
                            binding.ptr_mut(),
                            *name,
                            *value,
                        ))?;
                    }
                    ort::Error::result_from_status((ort::api().RunWithBinding)(
                        session.ptr().cast_mut(),
                        ptr::null(),
                        binding.ptr(),
                    ))
                } else {
                    ort::Error::result_from_status((ort::api().Run)(
                        session.ptr().cast_mut(),
                        ptr::null(),
                        prepared.name_ptrs.as_ptr(),
                        prepared.inputs.as_ptr(),
                        prepared.inputs.len(),
                        prepared.name_ptrs[prepared.inputs.len()..].as_ptr(),
                        prepared.outputs.len(),
                        prepared.outputs.as_mut_ptr(),
                    ))
                }
            };
            completion.finish(result)
        };
        run()
    }

    fn run_dynamic(
        &mut self,
        inputs: &[BorrowedTensor<'_>],
        transfers: &mut crate::engines::native::Transfers,
    ) -> Result<Vec<SharedTensor>> {
        let mut run = || -> Result<Vec<SharedTensor>> {
            let session = &self.loaded.session;
            ensure!(
                inputs.len() == session.inputs().len(),
                "ONNX input count mismatch"
            );
            // The executor owns these already-routed inputs through completion.
            let values = inputs
                .iter()
                .map(|&view| {
                    // SAFETY: the invocation retains each owner until the SDK binding drops.
                    unsafe { borrowed_value(view) }
                })
                .collect::<Result<Vec<_>>>()?;
            let mut binding = session.create_binding()?;
            // This guard drops before bindings, wrappers and input allocations,
            // including when binding or inference fails or unwinds.
            let mut completion = Completion::new(self.loaded.location)?;
            for (name, value) in self.loaded.input_names.iter().zip(&values) {
                binding.bind_input(name.as_str(), value)?;
            }
            let info = memory(self.loaded.location)?;
            for name in &self.loaded.output_names {
                binding.bind_output_to_device(name.as_str(), &info)?;
            }
            // SAFETY: each invocation owns its binding; shared ORT sessions support
            // concurrent RunWithBinding with distinct bindings and tensor storage.
            let result = unsafe {
                ort::Error::result_from_status((ort::api().RunWithBinding)(
                    session.ptr().cast_mut(),
                    ptr::null(),
                    binding.ptr(),
                ))
            };
            completion.finish(result)?;
            let outputs = bound_outputs(session, &binding)?;
            self.loaded
                .output_names
                .iter()
                .zip(&outputs)
                .map(|(_name, value)| {
                    let tensor = value.downcast_ref::<DynTensorValueType>()?;
                    let ValueType::Tensor { ty, shape, .. } = tensor.dtype() else {
                        unreachable!()
                    };
                    let info = tensor.memory_info();
                    let location = if info.is_cpu_accessible() {
                        MemoryLocation::Cpu
                    } else {
                        ensure!(
                            info.allocation_device() == AllocationDevice::CUDA,
                            "unsupported ORT output allocation"
                        );
                        MemoryLocation::Cuda {
                            device_id: u32::try_from(info.device_id())?,
                        }
                    };

                    // SAFETY: ORT owns this ready value until the independent native
                    // copy completes; subsequent runs cannot mutate the returned owner.
                    let descriptor = DLTensor {
                        data: tensor.data_ptr().cast_mut(),
                        device: dlpack::device(location)?,
                        ndim: i32::try_from(shape.len())?,
                        dtype: dlpack::dl_type(native_type(*ty)?),
                        shape: shape.as_ptr().cast_mut(),
                        strides: ptr::null_mut(),
                        byte_offset: 0,
                    };
                    // SAFETY: ORT owns the completed output until this copy finishes.
                    transfers.copy(
                        unsafe { BorrowedTensor::new(&descriptor) },
                        self.loaded.location,
                    )
                })
                .collect()
        };
        run()
    }
}

fn bound_outputs(session: &Session, binding: &IoBinding) -> Result<Vec<DynValue>> {
    let mut values = ptr::null_mut();
    let mut count = 0;
    // SAFETY: a completed private binding returns owned OrtValues and an array
    // allocated by the supplied live allocator. Free the array after adoption.
    unsafe {
        ort::Error::result_from_status((ort::api().GetBoundOutputValues)(
            binding.ptr(),
            session.allocator().ptr().cast_mut(),
            &mut values,
            &mut count,
        ))?;
        let outputs = if count == 0 {
            Vec::new()
        } else {
            std::slice::from_raw_parts(values, count)
                .iter()
                .map(|&value| {
                    DynValue::from_ptr(
                        NonNull::new(value).expect("ORT returned null output"),
                        Some(session.inner()),
                    )
                })
                .collect()
        };
        session.allocator().free(values);
        Ok(outputs)
    }
}

fn check_ports(specs: &[PortSpec], outlets: &[Outlet]) -> Result<Vec<String>> {
    ensure!(specs.len() == outlets.len(), "model tensor count mismatch");
    let mut remaining: HashMap<_, _> = outlets.iter().map(|port| (port.name(), port)).collect();
    let mut names = Vec::with_capacity(specs.len());
    for spec in specs {
        let outlet = remaining
            .remove(spec.name.as_str())
            .with_context(|| format!("unknown or duplicate model tensor {}", spec.name))?;
        let ValueType::Tensor { ty, shape, .. } = outlet.dtype() else {
            anyhow::bail!("only tensor ports are supported");
        };
        ensure!(
            *ty == ort_type(spec.dtype)
                && shape.len() == spec.shape.len()
                && shape
                    .iter()
                    .zip(&spec.shape)
                    .all(|(&a, &b)| a < 0 || b == -1 || a == b),
            "ONNX port {} has incompatible dtype or shape",
            outlet.name()
        );
        names.push(spec.name.clone());
    }
    Ok(names)
}

struct Prepared {
    binding: Option<IoBinding>,
    _values: Vec<DynValue>,
    _names: Vec<CString>,
    name_ptrs: Vec<*const c_char>,
    inputs: Vec<*const ort::sys::OrtValue>,
    outputs: Vec<*mut ort::sys::OrtValue>,
}

// SAFETY: pointers refer to owned wrappers/names and fixed workspace storage.
// The workspace mutex exclusively owns these handles and all referenced buffers.
unsafe impl Send for Prepared {}

fn synchronize(location: MemoryLocation) -> Result<()> {
    if let MemoryLocation::Cuda { device_id } = location {
        return Ok(cuda::synchronize(device_id)?);
    }
    Ok(location.validate()?)
}

fn select_device(location: MemoryLocation) -> Result<Option<cuda::Device>> {
    match location {
        MemoryLocation::Cuda { device_id } => Ok(Some(cuda::Device::select(device_id)?)),
        _ => Ok(None),
    }
}

struct Completion {
    _device: Option<cuda::Device>,
    location: MemoryLocation,
    finished: bool,
}

impl Completion {
    fn new(location: MemoryLocation) -> Result<Self> {
        Ok(Self {
            _device: select_device(location)?,
            location,
            finished: false,
        })
    }

    fn finish<T>(&mut self, result: ort::Result<T>) -> Result<T> {
        if result.is_err() {
            synchronize(self.location)?;
        }
        self.finished = true;
        Ok(result?)
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        if !self.finished {
            let _ = synchronize(self.location);
        }
    }
}

// The OrtValue owns a wrapper only. The caller must retain the ready storage
// through every use of the value and drain native work before releasing it.
unsafe fn borrowed_value(view: BorrowedTensor<'_>) -> Result<DynValue> {
    let descriptor = view.descriptor();
    let (shape, bytes) = dlpack::layout(view)?;
    let info = memory(dlpack::location(descriptor.device)?)?;
    let mut raw = ptr::null_mut();
    // SAFETY: caller supplies the storage lifetime, exclusivity for outputs, and completion.
    unsafe {
        ort::Error::result_from_status((ort::api().CreateTensorWithDataAsOrtValue)(
            info.ptr(),
            descriptor
                .data
                .cast::<u8>()
                .wrapping_add(descriptor.byte_offset as usize)
                .cast(),
            bytes,
            shape.as_ptr(),
            shape.len(),
            ort_type(dlpack::data_type(descriptor.dtype)?).into(),
            &mut raw,
        ))?;
        Ok(DynValue::from_ptr(
            NonNull::new(raw).context("ORT returned null tensor")?,
            None,
        ))
    }
}

fn memory(location: MemoryLocation) -> Result<MemoryInfo<'static>> {
    let (device, id) = match location {
        MemoryLocation::Cpu => (AllocationDevice::CPU, 0),
        MemoryLocation::PinnedCpu => (AllocationDevice::CUDA_PINNED, 0),
        MemoryLocation::Cuda { device_id } => (AllocationDevice::CUDA, i32::try_from(device_id)?),
    };
    Ok(MemoryInfo::new(
        device,
        id,
        AllocatorType::Device,
        MemoryType::Default,
    )?)
}

fn ort_type(dtype: DataType) -> TensorElementType {
    match dtype {
        DataType::Bool => TensorElementType::Bool,
        DataType::Uint8 => TensorElementType::Uint8,
        DataType::Uint16 => TensorElementType::Uint16,
        DataType::Uint32 => TensorElementType::Uint32,
        DataType::Uint64 => TensorElementType::Uint64,
        DataType::Int8 => TensorElementType::Int8,
        DataType::Int16 => TensorElementType::Int16,
        DataType::Int32 => TensorElementType::Int32,
        DataType::Int64 => TensorElementType::Int64,
        DataType::Float16 => TensorElementType::Float16,
        DataType::Bfloat16 => TensorElementType::Bfloat16,
        DataType::Float32 => TensorElementType::Float32,
        DataType::Float64 => TensorElementType::Float64,
    }
}

fn native_type(dtype: TensorElementType) -> Result<DataType> {
    Ok(match dtype {
        TensorElementType::Bool => DataType::Bool,
        TensorElementType::Uint8 => DataType::Uint8,
        TensorElementType::Uint16 => DataType::Uint16,
        TensorElementType::Uint32 => DataType::Uint32,
        TensorElementType::Uint64 => DataType::Uint64,
        TensorElementType::Int8 => DataType::Int8,
        TensorElementType::Int16 => DataType::Int16,
        TensorElementType::Int32 => DataType::Int32,
        TensorElementType::Int64 => DataType::Int64,
        TensorElementType::Float16 => DataType::Float16,
        TensorElementType::Bfloat16 => DataType::Bfloat16,
        TensorElementType::Float32 => DataType::Float32,
        TensorElementType::Float64 => DataType::Float64,
        _ => anyhow::bail!("unsupported ONNX tensor dtype {dtype}"),
    })
}
