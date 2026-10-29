// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use std::{sync::mpsc, thread, time::Duration};

#[test]
fn completion_requires_both_callbacks_in_either_order() {
    for release_first in [false, true] {
        let state = Arc::new(test_state());
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
    let state = test_state();
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

    fn buffer(&self) -> Arc<Buffer> {
        assert!(!self.owner.is_null());
        unsafe { Arc::clone(&*self.owner.cast::<Arc<Buffer>>()) }
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

fn test_state() -> State {
    State::new(
        OutputPlacement::default(),
        Arc::new(OutputPool::new(crate::OutputPoolOptions::default(), 0)),
    )
}

#[test]
fn callback_release_preserves_retained_storage() {
    let state = test_state();
    let first = Allocation::new(&state, 16).unwrap();
    let pointer = first.data;
    let retained = first.buffer();
    drop(first);
    let second = Allocation::new(&state, 16).unwrap();
    assert_ne!(second.data, pointer);
    drop(retained);
    assert_eq!(Allocation::new(&state, 16).unwrap().data, pointer);
}

#[test]
fn failed_and_empty_allocations_leave_callbacks_usable() {
    let state = test_state();
    assert!(Allocation::new(&state, usize::MAX).is_err());
    drop(Allocation::new(&state, 0).unwrap());
    drop(Allocation::new(&state, 16).unwrap());
}
