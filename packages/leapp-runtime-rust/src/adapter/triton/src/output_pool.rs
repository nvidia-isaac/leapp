// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{tensor::Buffer, MemoryLocation, Result};
use std::sync::{Arc, Mutex};

// Bound cached references and bytes across all requests on this runtime. The
// key is raw byte length and location, independent of shape, dtype, or model.
// Overflow allocations remain ordinary owned outputs.
const MAX_ENTRIES: usize = 64;
const MAX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Default)]
pub(crate) struct OutputPool {
    entries: Mutex<Vec<Arc<Buffer>>>,
}

impl OutputPool {
    pub(crate) fn allocate(&self, len: usize, location: MemoryLocation) -> Result<Arc<Buffer>> {
        {
            let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            for entry in entries.iter_mut() {
                if entry.len != len || entry.location != location {
                    continue;
                }
                // get_mut excludes weak references too. Holding this mutex until
                // cloning reserves the allocation before another request can use it.
                if Arc::get_mut(entry).is_some_and(|buffer| buffer.can_reuse()) {
                    return Ok(Arc::clone(entry));
                }
            }
        }
        // Allocation may synchronize CUDA. Do not hold the pool mutex across it.
        // A fresh buffer retains the existing zero-initialization contract; reused
        // buffers are already initialized. Successful backends fill the output.
        let buffer = Arc::new(Buffer::allocate(len, location)?);
        if len == 0 || len > MAX_BYTES {
            return Ok(buffer);
        }
        let mut retired = Vec::new();
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let mut bytes: usize = entries.iter().map(|entry| entry.len).sum();
        // Evict only when necessary. Dropping a cached Arc never invalidates an
        // output owner; in-use entries are left alone to avoid allocator churn.
        let mut index = 0;
        while index < entries.len() && (entries.len() >= MAX_ENTRIES || bytes + len > MAX_BYTES) {
            if Arc::get_mut(&mut entries[index]).is_some() {
                let entry = entries.swap_remove(index);
                bytes -= entry.len;
                retired.push(entry);
            } else {
                index += 1;
            }
        }
        if entries.len() < MAX_ENTRIES && bytes + len <= MAX_BYTES {
            entries.push(Arc::clone(&buffer));
        }
        drop(entries);
        // cudaFree can wait for device work, including foreign reads of exposed
        // buffers. Never hold the pool mutex while destroying those allocations.
        drop(retired);
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataType, Tensor};
    use std::collections::HashSet;

    fn locations() -> Vec<MemoryLocation> {
        vec![
            MemoryLocation::Cpu,
            MemoryLocation::PinnedCpu,
            MemoryLocation::Cuda { device_id: 0 },
        ]
    }

    fn tensor(buffer: Arc<Buffer>) -> Tensor {
        Tensor {
            dtype: DataType::Float32,
            shape: vec![(buffer.len / 4) as i64],
            buffer,
        }
    }

    #[test]
    fn retained_and_weak_owners_block_reuse_until_the_last_owner_drops() {
        for location in locations() {
            let pool = OutputPool::default();
            let first = tensor(pool.allocate(16, location).unwrap());
            let ptr = first.buffer.ptr;
            let retained = first.clone();
            let weak = Arc::downgrade(&first.buffer);
            let second = pool.allocate(16, location).unwrap();
            assert_ne!(ptr, second.ptr);
            drop(first);
            let third = pool.allocate(16, location).unwrap();
            assert_ne!(ptr, third.ptr);
            assert_eq!(retained.copy_to_cpu::<f32>().unwrap(), [0.; 4]);
            drop(retained);
            let fourth = pool.allocate(16, location).unwrap();
            assert_ne!(ptr, fourth.ptr);
            drop(weak);
            assert_eq!(ptr, pool.allocate(16, location).unwrap().ptr);
        }
    }

    #[test]
    fn reuse_matches_exact_bytes_but_not_shape_or_dtype() {
        for location in locations() {
            let pool = OutputPool::default();
            let first = tensor(pool.allocate(16, location).unwrap());
            let ptr = first.buffer.ptr;
            drop(first);
            let different_size = pool.allocate(8, location).unwrap();
            assert_ne!(ptr, different_size.ptr);
            let next = Tensor {
                dtype: DataType::Uint8,
                shape: vec![2, 8],
                buffer: pool.allocate(16, location).unwrap(),
            };
            assert_eq!(ptr, next.buffer.ptr);
            assert_eq!(next.copy_to_cpu::<u8>().unwrap(), [0; 16]);
            assert_eq!(next.location(), location);
        }
    }

    #[test]
    fn reuse_matches_exact_location_and_available_cuda_device_ids() {
        let pool = OutputPool::default();
        let mut locations = locations();
        let second_device = MemoryLocation::Cuda { device_id: 1 };
        let second_device_available = second_device.validate().is_ok();
        if second_device_available {
            locations.push(second_device);
        } else {
            eprintln!(
                "second CUDA device unavailable; positive two-device reuse case not exercised"
            );
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
            // An unavailable device must fail even with an idle same-sized GPU0
            // allocation cached; ignoring the device ID would incorrectly reuse it.
            assert!(pool.allocate(16, second_device).is_err());
        }
        for (location, identity) in saved {
            let buffer = pool.allocate(16, location).unwrap();
            assert_eq!(Arc::as_ptr(&buffer) as usize, identity);
        }
    }

    #[test]
    fn raw_pointer_exposure_is_permanent_and_shared_by_tensor_clones() {
        for location in locations() {
            let pool = OutputPool::default();
            let first = tensor(pool.allocate(16, location).unwrap());
            let ptr = first.buffer.ptr;
            let clone = first.clone();
            assert_eq!(clone.data_ptr(), ptr.as_ptr().cast());
            assert!(!first.buffer.can_reuse());
            drop(clone);
            drop(first);
            for _ in 0..4 {
                // No eviction is needed, so the exposed allocation is still cached.
                let next = pool.allocate(16, location).unwrap();
                assert_ne!(next.ptr, ptr);
            }
        }
    }

    #[test]
    fn internal_copies_keep_the_source_reusable_and_outputs_outlive_the_pool() {
        for location in locations() {
            let pool = OutputPool::default();
            let source = tensor(pool.allocate(16, location).unwrap());
            let ptr = source.buffer.ptr;
            let copy = source.deep_copy().unwrap();
            assert_ne!(copy.buffer.ptr, ptr);
            assert!(source.buffer.can_reuse());
            assert_eq!(source.copy_to_cpu::<f32>().unwrap(), [0.; 4]);
            assert_eq!(source.copy_to_cpu_bytes().unwrap(), [0; 16]);
            assert!(source.buffer.can_reuse());
            drop(source);
            let reused = tensor(pool.allocate(16, location).unwrap());
            assert_eq!(reused.buffer.ptr, ptr);
            drop(pool);
            assert_eq!(reused.copy_to_cpu::<f32>().unwrap(), [0.; 4]);
            assert_eq!(copy.copy_to_cpu::<f32>().unwrap(), [0.; 4]);
        }
    }

    #[test]
    fn fresh_storage_is_zeroed_but_reused_storage_is_not_zeroed_again() {
        let pool = OutputPool::default();
        let first = pool.allocate(16, MemoryLocation::Cpu).unwrap();
        let ptr = first.ptr;
        let first = tensor(first);
        assert_eq!(first.as_cpu_slice::<f32>().unwrap(), [0.; 4]);
        // SAFETY: the allocation is reserved by this test, has no other published
        // owners, and covers all sixteen initialized bytes being overwritten.
        unsafe { std::ptr::write_bytes(first.buffer.ptr.as_ptr(), 7, first.buffer.len) };
        drop(first);
        let reused = tensor(pool.allocate(16, MemoryLocation::Cpu).unwrap());
        assert_eq!(reused.buffer.ptr, ptr);
        assert_eq!(reused.as_cpu_bytes().unwrap(), [7; 16]);
    }

    #[test]
    fn zero_and_oversized_allocations_remain_uncached() {
        let pool = OutputPool::default();
        for len in [0, MAX_BYTES + 1] {
            let buffer = pool.allocate(len, MemoryLocation::Cpu).unwrap();
            assert_eq!(buffer.len, len);
            assert_eq!(Arc::strong_count(&buffer), 1);
            assert!(pool.entries.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn entry_limit_bounds_cached_references_with_retained_and_changing_outputs() {
        let pool = OutputPool::default();
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
    }

    #[test]
    fn byte_budget_evicts_only_after_strong_and_weak_owners_release() {
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
}
