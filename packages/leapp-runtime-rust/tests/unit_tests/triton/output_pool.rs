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
    crate::engines::triton::memory::dlpack::from_buffer(
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
        let pool = OutputPool::default();
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
        let copies = OutputPool::default();
        let input = crate::dlpack::BorrowedTensor::cpu(&values, &[4]).unwrap();
        for _ in 0..3 {
            let copied = copies.copy(input, location).unwrap();
            assert_eq!(copied.to_cpu_vec::<f32>().unwrap(), values);
        }
        assert_eq!(copies.entries.lock().unwrap().len(), 1);
    }
}

#[test]
fn reuse_matches_byte_size_location_and_available_cuda_device_ids() {
    let pool = OutputPool::default();
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
            let pool = OutputPool::default();
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
fn capacity_limits_preserve_live_owners_and_leave_overflow_uncached() {
    let pool = OutputPool::default();
    for len in [0, MAX_BYTES + 1] {
        let buffer = pool.allocate(len, MemoryLocation::Cpu).unwrap();
        assert_eq!(buffer.len, len);
        assert_eq!(Arc::strong_count(&buffer), 1);
        assert!(pool.entries.lock().unwrap().is_empty());
    }
    let held: Vec<_> = (1..=128)
        .map(|n| pool.allocate(n, MemoryLocation::Cpu).unwrap())
        .collect();
    assert_eq!(pool.entries.lock().unwrap().len(), MAX_ENTRIES);
    for (index, buffer) in held.iter().enumerate() {
        assert_eq!(
            Arc::strong_count(buffer),
            if index < MAX_ENTRIES { 2 } else { 1 }
        );
    }
    drop(held);
    for size in 129..256 {
        let next = pool.allocate(size, MemoryLocation::Cpu).unwrap();
        let entries = pool.entries.lock().unwrap();
        assert_eq!(entries.len(), MAX_ENTRIES);
        assert!(entries.iter().any(|entry| Arc::ptr_eq(entry, &next)));
        assert!(entries.iter().map(|entry| entry.len).sum::<usize>() <= MAX_BYTES);
    }

    // Exercise the byte budget independently of the entry limit.
    let pool = OutputPool::default();
    let first = pool.allocate(MAX_BYTES, MemoryLocation::Cpu).unwrap();
    let weak = Arc::downgrade(&first);
    let second = pool.allocate(8, MemoryLocation::Cpu).unwrap();
    assert_eq!(Arc::strong_count(&second), 1);
    assert_eq!(pool.entries.lock().unwrap().len(), 1);
    assert_eq!(pool.entries.lock().unwrap()[0].len, MAX_BYTES);
    drop(first);
    let third = pool.allocate(16, MemoryLocation::Cpu).unwrap();
    assert_eq!(Arc::strong_count(&third), 1);
    assert_eq!(pool.entries.lock().unwrap()[0].len, MAX_BYTES);
    drop(weak);
    let cached = pool.allocate(32, MemoryLocation::Cpu).unwrap();
    assert_eq!(Arc::strong_count(&cached), 2);
    assert_eq!(pool.entries.lock().unwrap().len(), 1);
    assert_eq!(pool.entries.lock().unwrap()[0].len, 32);
}

#[test]
fn concurrent_requests_reserve_distinct_cached_idle_allocations() {
    const THREADS: usize = 8;
    const ROUNDS: usize = 128;

    let pool = Arc::new(OutputPool::default());
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
    assert_eq!(pool.entries.lock().unwrap().len(), THREADS);
}
