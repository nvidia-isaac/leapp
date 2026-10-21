// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DLPack ABI, lifetime guards, borrowed CPU-slice access, and producer handoff.
#![deny(unsafe_op_in_unsafe_fn)]

#[path = "dlpack/cpu.rs"]
mod cpu;
#[path = "dlpack/sys.rs"]
pub mod sys;
pub use cpu::CpuElement;
use std::{
    marker::PhantomData,
    mem::ManuallyDrop,
    ptr::NonNull,
    sync::{Arc, Mutex},
};
pub use sys::*;

/// DLPack producer synchronization used by language bindings.
/// Native stream ownership stays in the compiled engine implementation.
#[doc(hidden)]
pub struct HandoffStream(crate::engines::HandoffStream);

impl HandoffStream {
    pub fn new(device: u32) -> Result<Self, crate::RuntimeError> {
        crate::engines::HandoffStream::new(device).map(Self)
    }

    pub fn with_device<T>(&self, producer: impl FnOnce() -> T) -> Result<T, crate::RuntimeError> {
        self.0.with_device(producer)
    }

    pub fn address(&self) -> usize {
        self.0.address()
    }

    pub fn synchronize(&self) -> Result<(), crate::RuntimeError> {
        self.0.synchronize()
    }
}

/// An exclusively owned producer lease. Foreign owners remain on their importing thread.
pub struct ManagedTensor(ManagedTensorPointer);
enum ManagedTensorPointer {
    Legacy(NonNull<DLManagedTensor>),
    Versioned(NonNull<DLManagedTensorVersioned>),
}

impl ManagedTensor {
    /// Adopt a producer lease, including responsibility for its deleter.
    /// # Safety
    /// `ptr` must be a live, exclusively owned DLPack managed descriptor. Its metadata
    /// and allocation must remain valid until its deleter is called. The producer's
    /// deleter must not unwind. No other owner may release this lease.
    pub unsafe fn from_legacy(ptr: *mut DLManagedTensor) -> Option<Self> {
        NonNull::new(ptr).map(|p| Self(ManagedTensorPointer::Legacy(p)))
    }

    /// Adopt a versioned lease. The same contract as `from_legacy` applies.
    /// # Safety
    /// The descriptor must satisfy the DLPack versioned ABI, including the stable
    /// version/context/deleter prefix when its major version is unsupported.
    pub unsafe fn from_versioned(ptr: *mut DLManagedTensorVersioned) -> Option<Self> {
        NonNull::new(ptr).map(|p| Self(ManagedTensorPointer::Versioned(p)))
    }

    /// Access metadata without dereferencing the allocation or shape/stride pointers.
    pub fn descriptor(&self) -> Result<&DLTensor, &'static str> {
        // SAFETY: adoption guarantees a live descriptor; inspect version first.
        unsafe {
            match &self.0 {
                ManagedTensorPointer::Legacy(p) => Ok(&p.as_ref().dl_tensor),
                ManagedTensorPointer::Versioned(p) => {
                    if (*p.as_ptr()).version.major != DLPACK_MAJOR_VERSION {
                        return Err("unsupported DLPack major version");
                    }
                    Ok(&p.as_ref().dl_tensor)
                }
            }
        }
    }

    /// Return the raw lease, transferring responsibility for invoking its deleter.
    pub fn into_raw(self) -> (*mut std::ffi::c_void, bool) {
        match &ManuallyDrop::new(self).0 {
            ManagedTensorPointer::Legacy(p) => (p.as_ptr().cast(), false),
            ManagedTensorPointer::Versioned(p) => (p.as_ptr().cast(), true),
        }
    }
}

impl Drop for ManagedTensor {
    fn drop(&mut self) {
        // SAFETY: this guard exclusively owns the producer lease, even on rejection.
        unsafe {
            match &self.0 {
                ManagedTensorPointer::Legacy(p) => {
                    if let Some(f) = (*p.as_ptr()).deleter {
                        f(p.as_ptr());
                    }
                }
                ManagedTensorPointer::Versioned(p) => {
                    if let Some(f) = (*p.as_ptr()).deleter {
                        f(p.as_ptr());
                    }
                }
            }
        }
    }
}

/// A descriptor borrowed together with its metadata and ready, immutable storage.
#[derive(Clone, Copy)]
pub struct BorrowedTensor<'a> {
    descriptor: DLTensor,
    lifetime: PhantomData<&'a DLTensor>,
}
impl<'a> BorrowedTensor<'a> {
    /// # Safety
    /// All descriptor pointers must be valid for the declared sizes and lifetime.
    /// Storage must be ready on every execution stream, readable on the declared
    /// device, and unmodified for this borrow, including concurrent consumers.
    pub unsafe fn new(descriptor: &'a DLTensor) -> Self {
        Self {
            descriptor: *descriptor,
            lifetime: PhantomData,
        }
    }
    pub fn descriptor(&self) -> &DLTensor {
        &self.descriptor
    }
}
// SAFETY: constructors require or retain immutable, ready storage for the borrow;
// it never transfers or invokes the producer's deleter.
unsafe impl Send for BorrowedTensor<'_> {}
unsafe impl Sync for BorrowedTensor<'_> {}

/// Shared ownership for engine-produced leases with thread-independent cleanup.
#[derive(Clone)]
pub struct SharedTensor(Arc<ThreadSafeTensorOwner>);
/// Synchronously copy the tensor's contiguous bytes into caller-owned CPU storage.
pub type CopyToCpu = dyn Fn(&mut [u8]) -> Result<(), String> + Send + Sync;

struct ThreadSafeTensorOwner {
    owner: ManagedTensor,
    exposed: Mutex<bool>,
    copy_to_cpu: Option<Box<CopyToCpu>>,
    on_expose: Option<Box<dyn Fn() + Send + Sync>>,
}
// SAFETY: SharedTensor::new requires thread-independent metadata lifetime and cleanup.
unsafe impl Send for ThreadSafeTensorOwner {}
unsafe impl Sync for ThreadSafeTensorOwner {}
impl SharedTensor {
    /// # Safety
    /// The producer must support metadata access and final deletion on any thread.
    /// The descriptor must use a supported ABI and remain unchanged. Its allocation
    /// must permit writable exports. Storage must be initialized and ready for
    /// reads, with no writes except through this owner's exposed descriptors or
    /// exports. Internal unsafe users must not race with CPU copies. This is not
    /// automatically true for foreign producers; imported leases stay `ManagedTensor`.
    /// `on_expose`, when present, runs before exposing pointers through descriptor
    /// access or export. It must be idempotent and must not modify the descriptor.
    /// `copy_to_cpu`, when supplied, must copy exactly the descriptor's contiguous
    /// bytes, complete all transfer work before returning, and retain its source.
    /// It must not expose the allocation or re-enter this owner. Without it, only
    /// host-accessible storage supports safe copies.
    pub unsafe fn new(
        owner: ManagedTensor,
        on_expose: Option<Box<dyn Fn() + Send + Sync>>,
        copy_to_cpu: Option<Box<CopyToCpu>>,
    ) -> Result<Self, &'static str> {
        owner.descriptor()?;
        Ok(Self(Arc::new(ThreadSafeTensorOwner {
            owner,
            exposed: Mutex::new(false),
            copy_to_cpu,
            on_expose,
        })))
    }

    /// Expose the descriptor, notifying the producer before releasing any pointers.
    pub fn descriptor(&self) -> &DLTensor {
        // Serialize the first exposure against safe CPU reads on every clone.
        *self.0.exposed.lock().unwrap_or_else(|e| e.into_inner()) = true;
        if let Some(notify) = &self.0.on_expose {
            notify();
        }
        self.0.owner.descriptor().expect("validated ABI")
    }

    /// Borrow storage for a bounded operation without exposing it externally.
    /// # Safety
    /// Storage must be ready and unmodified for the borrow. No pointers may escape
    /// it, and all work using them must finish before the borrow ends.
    pub unsafe fn borrow(&self) -> BorrowedTensor<'_> {
        // SAFETY: the caller supplies readiness, immutability, and completion;
        // this owner retains the descriptor, metadata, and allocation.
        unsafe { BorrowedTensor::new(self.0.owner.descriptor().expect("validated ABI")) }
    }

    /// Produce a fresh lease retaining this allocation. The consumer owns this lease.
    /// Flags deliberately do not claim a private copy: multiple exports may alias.
    pub fn export(&self, versioned: bool) -> ManagedTensor {
        let descriptor = *self.descriptor();
        let context = Box::into_raw(Box::new(self.clone())).cast();
        if versioned {
            let p = Box::into_raw(Box::new(DLManagedTensorVersioned {
                version: DLPackVersion { major: 1, minor: 0 },
                manager_ctx: context,
                deleter: Some(delete_versioned),
                flags: 0,
                dl_tensor: descriptor,
            }));
            // SAFETY: this new descriptor owns the context retaining its metadata/storage.
            unsafe { ManagedTensor::from_versioned(p).unwrap() }
        } else {
            let p = Box::into_raw(Box::new(DLManagedTensor {
                dl_tensor: descriptor,
                manager_ctx: context,
                deleter: Some(delete_legacy),
            }));
            // SAFETY: as above, using the legacy layout.
            unsafe { ManagedTensor::from_legacy(p).unwrap() }
        }
    }
}
unsafe extern "C" fn delete_legacy(p: *mut DLManagedTensor) {
    // SAFETY: export allocated both boxes; this callback is called exactly once.
    unsafe {
        let p = Box::from_raw(p);
        drop(Box::from_raw(p.manager_ctx.cast::<SharedTensor>()));
    }
}
unsafe extern "C" fn delete_versioned(p: *mut DLManagedTensorVersioned) {
    // SAFETY: export allocated both boxes; this callback is called exactly once.
    unsafe {
        let p = Box::from_raw(p);
        drop(Box::from_raw(p.manager_ctx.cast::<SharedTensor>()));
    }
}
