// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::dlpack::{DLDataType, SharedTensor};
use std::collections::HashSet;

fn locations() -> Vec<MemoryLocation> {
    vec![
        MemoryLocation::Cpu,
        MemoryLocation::PinnedCpu,
        MemoryLocation::Cuda { device_id: 0 },
    ]
}

fn lease(buffer: Arc<Buffer>) -> SharedTensor {
    let shape = vec![(buffer.len / 4) as i64];
    crate::engines::memory::dlpack::from_buffer(
        DLDataType {
            code: 2,
            bits: 32,
            lanes: 1,
        },
        shape,
        buffer,
    )
    .unwrap()
}

#[test]
fn retained_and_weak_owners_block_reuse_until_the_last_owner_drops() {
    for location in locations() {
        let pool = OutputPool::new(crate::OutputPoolOptions::default(), 0);
        let buffer = pool.allocate(16, location).unwrap();
        let ptr = buffer.ptr;
        let values = [1f32, 2., 3., 4.];
        // SAFETY: no tensor borrows this allocation yet; values has exactly 16 bytes.
        unsafe {
            buffer.copy_from(values.as_ptr().cast()).unwrap();
        }
        let weak = Arc::downgrade(&buffer);
        let first = lease(buffer);
        let retained = first.clone();
        let second = pool.allocate(16, location).unwrap();
        assert_ne!(ptr, second.ptr);
        drop(first);
        let third = pool.allocate(16, location).unwrap();
        assert_ne!(ptr, third.ptr);
        assert_eq!(retained.to_cpu_vec::<f32>().unwrap(), values);
        drop(retained);
        let fourth = pool.allocate(16, location).unwrap();
        assert_ne!(ptr, fourth.ptr);
        drop(weak);
        let reused = pool.allocate(16, location).unwrap();
        assert_eq!(ptr, reused.ptr);
        let output = lease(reused);
        drop(pool);
        assert_eq!(output.to_cpu_vec::<f32>().unwrap(), values);
        let copies = OutputPool::new(crate::OutputPoolOptions::default(), 0);
        let input = crate::dlpack::BorrowedTensor::cpu(&values, &[4]).unwrap();
        for _ in 0..3 {
            let copied = copies.copy(input, location).unwrap();
            assert_eq!(copied.to_cpu_vec::<f32>().unwrap(), values);
        }
        assert_eq!(copies.state.lock().unwrap().entries.len(), 1);
    }
}

#[test]
fn reuse_matches_byte_size_location_and_available_cuda_device_ids() {
    let pool = OutputPool::new(crate::OutputPoolOptions::default(), 0);
    let mut locations = locations();
    let second_device = MemoryLocation::Cuda { device_id: 1 };
    let second_device_available = second_device.validate().is_ok();
    if second_device_available {
        locations.push(second_device);
    }
    let mut identities = HashSet::new();
    let mut saved = Vec::new();
    for location in locations {
        let buffer = pool.allocate(16, location).unwrap();
        let identity = Arc::as_ptr(&buffer) as usize;
        assert!(
            identities.insert(identity),
            "reused storage in a different location"
        );
        assert_eq!(buffer.location, location);
        saved.push((location, identity));
        drop(buffer);
    }
    if !second_device_available {
        assert!(pool.allocate(16, second_device).is_err());
    }
    for (location, identity) in saved {
        let buffer = pool.allocate(16, location).unwrap();
        assert_eq!(Arc::as_ptr(&buffer) as usize, identity);
        drop(buffer);
        let different_size = pool.allocate(8, location).unwrap();
        assert_ne!(Arc::as_ptr(&different_size) as usize, identity);
    }
}

#[test]
fn descriptors_and_exports_permanently_exclude_shared_storage_from_reuse() {
    for location in locations() {
        for versioned in [None, Some(false), Some(true)] {
            let pool = OutputPool::new(crate::OutputPoolOptions::default(), 0);
            let buffer = pool.allocate(16, location).unwrap();
            let ptr = buffer.ptr;
            let owner = lease(buffer.clone());
            let alias = owner.clone();
            let mut exports = if let Some(versioned) = versioned {
                vec![alias.export(versioned), alias.export(versioned)]
            } else {
                assert_eq!(alias.descriptor().data, ptr.as_ptr().cast());
                Vec::new()
            };
            assert!(!buffer.can_reuse());
            assert!(owner.to_cpu_vec::<f32>().unwrap_err().contains("exposed"));
            drop((buffer, owner, alias));
            assert_ne!(pool.allocate(16, location).unwrap().ptr, ptr);
            while let Some(export) = exports.pop() {
                drop(export);
                // Exclusion survives the last external owner being released.
                assert_ne!(pool.allocate(16, location).unwrap().ptr, ptr);
            }
        }
    }
}

#[test]
fn configurable_limits_allow_large_and_dynamic_buffers() {
    for location in locations() {
        let pool = OutputPool::new(crate::OutputPoolOptions { max_entries: 2 }, 128);
        assert_eq!(pool.state.lock().unwrap().largest_buffer, 128);
        let first = pool.allocate(128, location).unwrap();
        let second = pool.allocate(64, location).unwrap();
        let uncached = pool.allocate(32, location).unwrap();
        assert_eq!(Arc::strong_count(&uncached), 1);
        assert_eq!(pool.state.lock().unwrap().entries.len(), 2);
        let weak = Arc::downgrade(&first);
        drop(first);
        // A weak owner also prevents eviction and reuse.
        assert_eq!(Arc::strong_count(&pool.allocate(256, location).unwrap()), 1);
        drop(weak);
        let larger = pool.allocate(256, location).unwrap();
        assert_eq!(Arc::strong_count(&larger), 2);
        assert_eq!(pool.state.lock().unwrap().largest_buffer, 256);
        drop(second);
        drop(larger);
        // Larger than the old 64 MiB cap; the same allocation must be reusable.
        let image_bytes = 65 * 1024 * 1024;
        let image = pool.allocate(image_bytes, location).unwrap();
        let identity = Arc::as_ptr(&image);
        drop(image);
        assert_eq!(
            Arc::as_ptr(&pool.allocate(image_bytes, location).unwrap()),
            identity
        );
        assert_eq!(pool.state.lock().unwrap().largest_buffer, image_bytes);
        let fresh = pool.fresh();
        assert_eq!(fresh.state.lock().unwrap().largest_buffer, 128);
        assert!(fresh.state.lock().unwrap().entries.is_empty());

        let disabled =
            OutputPool::new(crate::OutputPoolOptions { max_entries: 0 }, image_bytes);
        assert_eq!(
            Arc::strong_count(&disabled.allocate(16, location).unwrap()),
            1
        );
        assert!(disabled.state.lock().unwrap().entries.is_empty());
        pool.clear();
        assert!(pool.state.lock().unwrap().entries.is_empty());
    }
}

#[test]
fn invalid_copy_does_not_populate_the_cache() {
    let pool = OutputPool::new(crate::OutputPoolOptions::default(), 16);
    let values = [1_f32; 4];
    let mut descriptor = *BorrowedTensor::cpu(&values, &[4]).unwrap().descriptor();
    descriptor.data = std::ptr::null_mut();
    // SAFETY: null storage is rejected before any dereference; shape is static.
    assert!(pool
        .copy(
            unsafe { BorrowedTensor::new(&descriptor) },
            MemoryLocation::Cpu
        )
        .is_err());
    assert!(pool.state.lock().unwrap().entries.is_empty());
}

#[test]
fn concurrent_requests_reserve_distinct_cached_idle_allocations() {
    const THREADS: usize = 8;
    const ROUNDS: usize = 128;

    let pool = Arc::new(OutputPool::new(crate::OutputPoolOptions::default(), 0));
    let seed: Vec<_> = (0..THREADS)
        .map(|_| pool.allocate(32, MemoryLocation::Cpu).unwrap())
        .collect();
    let expected: HashSet<_> = seed.iter().map(|b| Arc::as_ptr(b) as usize).collect();
    drop(seed);
    let barrier = Arc::new(std::sync::Barrier::new(THREADS));
    let threads: Vec<_> = (0..THREADS)
        .map(|_| {
            let pool = pool.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut reservations = Vec::new();
                for _ in 0..ROUNDS {
                    barrier.wait();
                    let allocation = pool.allocate(32, MemoryLocation::Cpu).unwrap();
                    // Every allocation remains retained until all threads reserve.
                    barrier.wait();
                    reservations.push(Arc::as_ptr(&allocation) as usize);
                    drop(allocation);
                    // All allocations become idle before the next round starts.
                    barrier.wait();
                }
                reservations
            })
        })
        .collect();
    let reservations: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    for round in 0..ROUNDS {
        let actual: HashSet<_> = reservations.iter().map(|r| r[round]).collect();
        assert_eq!(
            actual, expected,
            "cached reservations differ in round {round}"
        );
    }
    assert_eq!(pool.state.lock().unwrap().entries.len(), THREADS);
}
