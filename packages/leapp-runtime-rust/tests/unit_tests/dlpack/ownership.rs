// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use lrt::dlpack::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Context {
    dropped: Arc<AtomicUsize>,
    values: [f32; 2],
    shape: [i64; 1],
}
unsafe extern "C" fn delete(p: *mut DLManagedTensorVersioned) {
    unsafe {
        let p = Box::from_raw(p);
        let context = Box::from_raw(p.manager_ctx.cast::<Context>());
        context.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
fn producer() -> (ManagedTensor, Arc<AtomicUsize>) {
    let dropped = Arc::new(AtomicUsize::new(0));
    let context = Box::new(Context {
        dropped: dropped.clone(),
        values: [2., 3.],
        shape: [2],
    });
    let dl_tensor = *BorrowedTensor::cpu(&context.values, &context.shape)
        .unwrap()
        .descriptor();
    let p = Box::into_raw(Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: Box::into_raw(context).cast(),
        deleter: Some(delete),
        flags: 0,
        dl_tensor,
    }));
    (
        unsafe { ManagedTensor::from_versioned(p).unwrap() },
        dropped,
    )
}
#[test]
fn leases_preserve_ownership_and_exposure_across_threads() {
    for versioned in [false, true] {
        let (owner, dropped) = producer();
        // SAFETY: the fixture owns ready CPU storage and can be deleted on any thread.
        let shared = unsafe { SharedTensor::new(owner, None, None).unwrap() };
        let saved = shared.to_cpu_vec::<f32>().unwrap();
        assert_eq!(saved, [2., 3.]);
        let descriptor = shared.clone().descriptor().data;
        // SAFETY: this CPU fixture has no outstanding readers or foreign consumers.
        unsafe { *descriptor.cast::<f32>() = 99. };
        assert_eq!(saved, [2., 3.]);
        assert!(shared.to_cpu_vec::<f32>().is_err());
        let released = dropped.clone();
        std::thread::spawn(move || {
            let sibling = shared.export(!versioned);
            let (pointer, actual) = shared.export(versioned).into_raw();
            assert_eq!(actual, versioned);
            drop(shared);
            drop(sibling);
            assert_eq!(released.load(Ordering::SeqCst), 0);
            // Adopt the transferred lease exactly once, on the consumer thread.
            let consumer = unsafe {
                if versioned {
                    ManagedTensor::from_versioned(pointer.cast())
                } else {
                    ManagedTensor::from_legacy(pointer.cast())
                }
            }
            .unwrap();
            let tensor = consumer.descriptor().unwrap();
            assert_eq!(unsafe { *tensor.shape }, 2);
            assert_eq!(
                unsafe { std::slice::from_raw_parts(tensor.data.cast::<f32>(), 2) },
                [99., 3.]
            );
        })
        .join()
        .unwrap();
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
#[test]
fn transfer_failures_preserve_the_owner_and_allow_retry() {
    let (owner, dropped) = producer();
    let attempts = Arc::new(AtomicUsize::new(0));
    let copies = attempts.clone();
    // SAFETY: the fixture is ready; this synchronous callback copies its values
    // without retaining the destination or exposing any source pointers.
    let shared = unsafe {
        SharedTensor::new(
            owner,
            None,
            Some(Box::new(move |destination| {
                if copies.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Err("simulated device transfer failure".into());
                }
                destination[..4].copy_from_slice(&2f32.to_ne_bytes());
                destination[4..].copy_from_slice(&3f32.to_ne_bytes());
                Ok(())
            })),
        )
        .unwrap()
    };
    assert_eq!(
        shared.to_cpu_vec::<f32>().unwrap_err(),
        "simulated device transfer failure"
    );
    assert_eq!(shared.to_cpu_vec::<f32>().unwrap(), [2., 3.]);
    drop(shared.export(true));
    assert!(shared.to_cpu_vec::<f32>().unwrap_err().contains("exposed"));
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    drop(shared);
    assert_eq!(dropped.load(Ordering::SeqCst), 1);
}
