// Generated from upstream DLPack v1.0 by generate_bindings.py. Do not edit.
#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]
pub type DLDeviceType = i32;
pub const DLPACK_MAJOR_VERSION: u32 = 1;
pub const DLPACK_MINOR_VERSION: u32 = 0;
pub const DLPACK_FLAG_BITMASK_READ_ONLY: u64 = 1;
pub const DLPACK_FLAG_BITMASK_IS_COPIED: u64 = 2;
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLPackVersion {
    pub major: u32,
    pub minor: u32,
}
pub const kDLCPU: u32 = 1;
pub const kDLCUDA: u32 = 2;
pub const kDLCUDAHost: u32 = 3;
pub const kDLOpenCL: u32 = 4;
pub const kDLVulkan: u32 = 7;
pub const kDLMetal: u32 = 8;
pub const kDLVPI: u32 = 9;
pub const kDLROCM: u32 = 10;
pub const kDLROCMHost: u32 = 11;
pub const kDLExtDev: u32 = 12;
pub const kDLCUDAManaged: u32 = 13;
pub const kDLOneAPI: u32 = 14;
pub const kDLWebGPU: u32 = 15;
pub const kDLHexagon: u32 = 16;
pub const kDLMAIA: u32 = 17;
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDevice {
    pub device_type: DLDeviceType,
    pub device_id: i32,
}
pub const kDLInt: u32 = 0;
pub const kDLUInt: u32 = 1;
pub const kDLFloat: u32 = 2;
pub const kDLOpaqueHandle: u32 = 3;
pub const kDLBfloat: u32 = 4;
pub const kDLComplex: u32 = 5;
pub const kDLBool: u32 = 6;
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLDataType {
    pub code: u8,
    pub bits: u8,
    pub lanes: u16,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLTensor {
    pub data: *mut std::ffi::c_void,
    pub device: DLDevice,
    pub ndim: i32,
    pub dtype: DLDataType,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub byte_offset: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLManagedTensor {
    pub dl_tensor: DLTensor,
    pub manager_ctx: *mut std::ffi::c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DLManagedTensorVersioned {
    pub version: DLPackVersion,
    pub manager_ctx: *mut std::ffi::c_void,
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    pub flags: u64,
    pub dl_tensor: DLTensor,
}
