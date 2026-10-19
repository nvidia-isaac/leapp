// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{
    error::{check, Handle},
    sys,
    tensor::{byte_size, Buffer},
    DataType, Error, ErrorKind, OutputPlacement, Result, Tensor,
};
use std::{
    ffi::{c_char, c_void, CStr, CString},
    panic::{catch_unwind, UnwindSafe},
    ptr,
    sync::{Arc, Condvar, Mutex, MutexGuard},
};

pub(crate) struct Response(Handle<sys::TRITONSERVER_InferenceResponse>);

// SAFETY: response ownership is transferred from a callback to the waiting
// thread under the completion mutex. Only that owner accesses or deletes it.
unsafe impl Send for Response {}

#[derive(Default)]
struct Completion {
    final_response: bool,
    request_released: bool,
    response: Option<Response>,
    error: Option<Error>,
}

pub(crate) struct State {
    placement: OutputPlacement,
    output_pool: Arc<crate::output_pool::OutputPool>,
    completion: Mutex<Completion>,
    ready: Condvar,
}

impl State {
    pub(crate) fn new(
        placement: OutputPlacement,
        output_pool: Arc<crate::output_pool::OutputPool>,
    ) -> Self {
        Self {
            placement,
            output_pool,
            completion: Mutex::new(Completion::default()),
            ready: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Completion> {
        self.completion.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn released(&self, result: Result<()>) {
        let mut completion = self.lock();
        if let Err(error) = result {
            completion.error = Some(error);
        }
        completion.request_released = true;
        self.ready.notify_all();
    }

    pub(crate) fn wait(&self) -> Result<Response> {
        let mut completion = self.lock();
        while !completion.final_response || !completion.request_released {
            completion = self
                .ready
                .wait(completion)
                .unwrap_or_else(|e| e.into_inner());
        }
        if let Some(error) = completion.error.take() {
            return Err(error);
        }
        completion
            .response
            .take()
            .ok_or_else(|| internal("Triton returned no response"))
    }
}

fn internal(message: &str) -> Error {
    Error::new(ErrorKind::Internal, "execute", message)
}

// Convert callback failures to owned C errors; no Rust unwind crosses the ABI.
fn callback(operation: impl FnOnce() -> Result<()> + UnwindSafe) -> *mut sys::TRITONSERVER_Error {
    let error = match catch_unwind(operation) {
        Ok(Ok(())) => return ptr::null_mut(),
        Ok(Err(error)) => error,
        Err(_) => internal("panic in Triton allocation callback"),
    };
    let message = CString::new(error.to_string().replace('\0', "\\0")).expect("NULs escaped");
    // SAFETY: Triton copies the diagnostic before CString is dropped.
    unsafe { sys::TRITONSERVER_ErrorNew(sys::TRITONSERVER_ERROR_INTERNAL, message.as_ptr()) }
}

pub(crate) unsafe extern "C" fn query(
    _: *mut sys::TRITONSERVER_ResponseAllocator,
    userp: *mut c_void,
    name: *const c_char,
    _: *mut usize,
    memory: *mut u32,
    device: *mut i64,
) -> *mut sys::TRITONSERVER_Error {
    callback(|| {
        // SAFETY: Triton supplies our live State and writable ABI output parameters.
        unsafe {
            let state = &*userp.cast::<State>();
            let name = if name.is_null() {
                ""
            } else {
                CStr::from_ptr(name)
                    .to_str()
                    .map_err(|_| internal("output name is not UTF-8"))?
            };
            (*memory, *device) = state.placement.location(name).native();
        }
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn allocate(
    _: *mut sys::TRITONSERVER_ResponseAllocator,
    name: *const c_char,
    len: usize,
    _: u32,
    _: i64,
    userp: *mut c_void,
    data: *mut *mut c_void,
    owner: *mut *mut c_void,
    memory: *mut u32,
    device: *mut i64,
) -> *mut sys::TRITONSERVER_Error {
    callback(|| {
        // SAFETY: Triton supplies our live State, a NUL-terminated name and
        // writable output parameters. The boxed Arc lives until release_buffer.
        unsafe {
            *data = ptr::null_mut();
            *owner = ptr::null_mut();
            let state = &*userp.cast::<State>();
            let name = CStr::from_ptr(name)
                .to_str()
                .map_err(|_| internal("output name is not UTF-8"))?;
            let location = state.placement.location(name);
            (*memory, *device) = location.native();
            // Triton need not call release for null, zero-length allocations.
            if len != 0 {
                let buffer = state.output_pool.allocate(len, location)?;
                *data = buffer.ptr.as_ptr().cast();
                *owner = Box::into_raw(Box::new(buffer)).cast();
            }
        }
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn release_buffer(
    _: *mut sys::TRITONSERVER_ResponseAllocator,
    _: *mut c_void,
    owner: *mut c_void,
    _: usize,
    _: u32,
    _: i64,
) -> *mut sys::TRITONSERVER_Error {
    callback(|| {
        if !owner.is_null() {
            // SAFETY: allocate transferred this box exactly once to Triton.
            unsafe {
                drop(Box::from_raw(owner.cast::<Arc<Buffer>>()));
            }
        }
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn release_request(
    request: *mut sys::TRITONSERVER_InferenceRequest,
    flags: u32,
    userp: *mut c_void,
) {
    if flags & sys::TRITONSERVER_REQUEST_RELEASE_ALL == 0 {
        return;
    }
    // SAFETY: RELEASE_ALL transfers ownership and guarantees no more input access.
    // State remains alive until both completion flags have been observed.
    unsafe {
        let result =
            catch_unwind(|| check(sys::TRITONSERVER_InferenceRequestDelete(request), "release"));
        let state = &*userp.cast::<State>();
        state.released(result.unwrap_or_else(|_| Err(internal("panic releasing Triton request"))));
    }
}

pub(crate) unsafe extern "C" fn receive_response(
    response: *mut sys::TRITONSERVER_InferenceResponse,
    flags: u32,
    userp: *mut c_void,
) {
    // SAFETY: the callback receives sole ownership of response. State remains
    // alive until completion is signaled after the last callback access.
    unsafe {
        let state = &*userp.cast::<State>();
        let mut handle = Handle::empty(sys::TRITONSERVER_InferenceResponseDelete);
        handle.ptr = response;
        let result = catch_unwind(|| {
            if !response.is_null() {
                let mut completion = state.lock();
                if completion.response.is_some() {
                    completion.error =
                        Some(internal("multiple responses from a single-response model"));
                }
                completion.response = Some(Response(handle));
            }
        });
        let mut completion = state.lock();
        if result.is_err() {
            completion.error = Some(internal("panic receiving Triton response"));
        }
        if flags & sys::TRITONSERVER_RESPONSE_COMPLETE_FINAL != 0 {
            completion.final_response = true;
        }
        state.ready.notify_all();
    }
}

pub(crate) fn outputs(
    response: &Response,
    placement: &OutputPlacement,
) -> Result<Vec<(String, Tensor)>> {
    let mut result = Vec::new();
    // SAFETY: the response is owned and complete; metadata and allocator userp
    // remain live until it is dropped after all output owners are retained.
    unsafe {
        // Triton r26.05 returns a newly allocated status error here (the shipped
        // header's borrowed-error comment differs from its implementation).
        check(
            sys::TRITONSERVER_InferenceResponseError(response.0.ptr),
            "execute",
        )?;
        let mut count = 0;
        check(
            sys::TRITONSERVER_InferenceResponseOutputCount(response.0.ptr, &mut count),
            "execute",
        )?;
        for index in 0..count {
            let (mut name, mut shape, mut data) = (ptr::null(), ptr::null(), ptr::null());
            let (mut dtype, mut dimensions, mut len, mut memory, mut device) = (0, 0, 0, 0, 0);
            let mut owner = ptr::null_mut();
            check(
                sys::TRITONSERVER_InferenceResponseOutput(
                    response.0.ptr,
                    index,
                    &mut name,
                    &mut dtype,
                    &mut shape,
                    &mut dimensions,
                    &mut data,
                    &mut len,
                    &mut memory,
                    &mut device,
                    &mut owner,
                ),
                "execute",
            )?;
            let name = CStr::from_ptr(name)
                .to_str()
                .map_err(|_| internal("output name is not UTF-8"))?
                .to_owned();
            let location = placement.location(&name);
            if (memory, device) != location.native() {
                return Err(internal("output memory does not match requested placement"));
            }
            let dtype = DataType::from_native(dtype)?;
            let dimensions = usize::try_from(dimensions)
                .map_err(|_| internal("invalid output dimension count"))?;
            if dimensions > isize::MAX as usize / std::mem::size_of::<i64>() {
                return Err(internal("invalid output dimension count"));
            }
            let shape = if dimensions == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(shape, dimensions).to_vec()
            };
            if byte_size(dtype, &shape)? != len {
                return Err(internal("output byte count does not match dtype and shape"));
            }
            let buffer = if len == 0 {
                Arc::new(Buffer::allocate(0, location)?)
            } else {
                if owner.is_null() {
                    return Err(internal("output has no allocation owner"));
                }
                let buffer = &*owner.cast::<Arc<Buffer>>();
                if buffer.ptr.as_ptr().cast::<c_void>() != data.cast_mut()
                    || buffer.len != len
                    || buffer.location != location
                {
                    return Err(internal("output allocation does not match its owner"));
                }
                Arc::clone(buffer)
            };
            result.push((
                name,
                Tensor {
                    dtype,
                    shape,
                    buffer,
                },
            ));
        }
    }
    for name in placement.overrides.keys() {
        if !result.iter().any(|(output, _)| output == name) {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "execute",
                format!("unknown output placement: {name}"),
            ));
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, thread, time::Duration};

    #[test]
    fn completion_requires_both_callbacks_in_either_order() {
        for release_first in [false, true] {
            let state = Arc::new(pooled_state());
            let (sender, receiver) = mpsc::channel();
            let worker_state = state.clone();
            let worker = thread::spawn(move || {
                sender.send(worker_state.wait().is_err()).unwrap();
            });
            let finish_response = || {
                // SAFETY: state outlives the callback; a null final response is allowed by Triton.
                unsafe {
                    receive_response(
                        ptr::null_mut(),
                        sys::TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                        Arc::as_ptr(&state).cast_mut().cast(),
                    );
                }
            };
            if release_first {
                state.released(Ok(()));
            } else {
                finish_response();
            }
            assert!(receiver.recv_timeout(Duration::from_millis(20)).is_err());
            if release_first {
                finish_response();
            } else {
                state.released(Ok(()));
            }
            assert!(receiver.recv_timeout(Duration::from_secs(2)).unwrap());
            worker.join().unwrap();
        }
    }

    #[test]
    fn callbacks_can_finish_before_the_wait_begins() {
        let state = pooled_state();
        state.released(Ok(()));
        // SAFETY: state is live, and null is allowed for the final response.
        unsafe {
            receive_response(
                ptr::null_mut(),
                sys::TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                (&state as *const State).cast_mut().cast(),
            );
        }
        assert!(state.wait().is_err());
    }

    #[test]
    fn allocation_panics_become_native_errors() {
        let error = callback(|| panic!("simulated allocator panic"));
        // SAFETY: callback returns an owned C error.
        let error = unsafe { check(error, "execute") }.unwrap_err();
        assert!(error
            .to_string()
            .contains("panic in Triton allocation callback"));
    }

    struct Allocation {
        data: *mut c_void,
        owner: *mut c_void,
        len: usize,
    }

    impl Allocation {
        fn new(state: &State, len: usize) -> Result<Self> {
            let mut data = ptr::NonNull::<u8>::dangling().as_ptr().cast();
            let mut owner = data;
            let (mut memory, mut device) = (u32::MAX, -1);
            // SAFETY: state, name, and output parameters remain live throughout
            // the callback; its boxed owner is released exactly once by Drop.
            let result = unsafe {
                check(
                    allocate(
                        ptr::null_mut(),
                        c"output".as_ptr(),
                        len,
                        sys::TRITONSERVER_MEMORY_GPU,
                        99,
                        (state as *const State).cast_mut().cast(),
                        &mut data,
                        &mut owner,
                        &mut memory,
                        &mut device,
                    ),
                    "allocate",
                )
            };
            if result.is_err() || len == 0 {
                assert!(data.is_null());
                assert!(owner.is_null());
            }
            result?;
            assert_eq!(
                (memory, device),
                state.placement.location("output").native()
            );
            Ok(Self { data, owner, len })
        }

        fn tensor(&self) -> Tensor {
            assert!(!self.owner.is_null());
            // SAFETY: this helper retains the same boxed Arc as outputs(); the
            // allocation remains live until the release callback in Drop.
            let buffer = unsafe { Arc::clone(&*self.owner.cast::<Arc<Buffer>>()) };
            assert!(buffer.can_reuse());
            Tensor {
                dtype: DataType::Uint8,
                shape: vec![self.len as i64],
                buffer,
            }
        }
    }

    impl Drop for Allocation {
        fn drop(&mut self) {
            // SAFETY: allocate supplied this owner, which has not been released.
            unsafe {
                check(
                    release_buffer(
                        ptr::null_mut(),
                        self.data,
                        self.owner,
                        self.len,
                        sys::TRITONSERVER_MEMORY_CPU,
                        0,
                    ),
                    "release",
                )
                .unwrap();
            }
        }
    }

    fn pooled_state() -> State {
        State::new(
            OutputPlacement::default(),
            Arc::new(crate::output_pool::OutputPool::default()),
        )
    }

    #[test]
    fn callback_storage_is_reserved_until_all_returned_tensor_owners_drop() {
        let state = pooled_state();
        let allocation = Allocation::new(&state, 16).unwrap();
        let address = allocation.data;
        assert_eq!(allocation.tensor().as_cpu_bytes().unwrap(), &[0; 16]);
        let concurrent = Allocation::new(&state, 16).unwrap();
        assert_ne!(address, concurrent.data);
        drop(concurrent);
        // SAFETY: the callback exclusively reserved these CPU bytes; no tensor
        // retains a read and the complete output is written.
        unsafe { ptr::write_bytes(allocation.data.cast::<u8>(), 0x11, allocation.len) };
        let first = allocation.tensor();
        let retained = first.clone();
        drop(allocation);
        let second = Allocation::new(&state, 16).unwrap();
        assert_ne!(address, second.data);
        // SAFETY: second is a distinct reserved allocation with no tensor owner.
        unsafe { ptr::write_bytes(second.data.cast::<u8>(), 0x22, second.len) };
        assert_eq!(first.as_cpu_bytes().unwrap(), &[0x11; 16]);
        drop(first);
        let third = Allocation::new(&state, 16).unwrap();
        assert_ne!(address, third.data);
        assert_eq!(retained.as_cpu_bytes().unwrap(), &[0x11; 16]);
        drop(retained);
        let reused = Allocation::new(&state, 16).unwrap();
        assert_eq!(address, reused.data);
        // Internal callback pointer access permits reuse, without re-zeroing.
        assert_eq!(reused.tensor().as_cpu_bytes().unwrap(), &[0x11; 16]);
    }

    #[test]
    fn callback_pointer_exposure_excludes_reuse_and_tensor_outlives_pool() {
        let state = pooled_state();
        let allocation = Allocation::new(&state, 16).unwrap();
        let tensor = allocation.tensor();
        let address = tensor.clone().data_ptr();
        drop((allocation, tensor));
        let next = Allocation::new(&state, 16).unwrap();
        assert_ne!(address, next.data.cast_const());
        let retained = next.tensor();
        drop(next);
        drop(state);
        assert_eq!(retained.as_cpu_bytes().unwrap(), &[0; 16]);
    }

    #[test]
    fn callback_allocation_error_and_empty_output_leave_pool_usable() {
        let state = pooled_state();
        let first = Allocation::new(&state, 16).unwrap();
        let address = first.data;
        drop(first);
        // Layout rejects this size before any allocation is attempted.
        assert!(Allocation::new(&state, usize::MAX).is_err());
        drop(Allocation::new(&state, 0).unwrap());
        assert_eq!(address, Allocation::new(&state, 16).unwrap().data);
    }
}
