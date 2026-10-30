// Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Generated subset of the Triton 2.69.0 C API; see generate_bindings.py.
#![allow(non_camel_case_types, non_snake_case, dead_code)]
#[repr(C)]
pub struct TRITONSERVER_ServerOptions {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TRITONSERVER_Server {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TRITONSERVER_ResponseAllocator {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TRITONSERVER_Message {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TRITONSERVER_InferenceTrace {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TRITONSERVER_InferenceResponse {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TRITONSERVER_InferenceRequest {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TRITONSERVER_Error {
    _private: [u8; 0],
}
pub const TRITONSERVER_API_VERSION_MAJOR: u32 = 1;
pub const TRITONSERVER_API_VERSION_MINOR: u32 = 34;
pub type TRITONSERVER_DataType = std::ffi::c_uint;
pub const TRITONSERVER_TYPE_INVALID: TRITONSERVER_DataType = 0;
pub const TRITONSERVER_TYPE_BOOL: TRITONSERVER_DataType = 1;
pub const TRITONSERVER_TYPE_UINT8: TRITONSERVER_DataType = 2;
pub const TRITONSERVER_TYPE_UINT16: TRITONSERVER_DataType = 3;
pub const TRITONSERVER_TYPE_UINT32: TRITONSERVER_DataType = 4;
pub const TRITONSERVER_TYPE_UINT64: TRITONSERVER_DataType = 5;
pub const TRITONSERVER_TYPE_INT8: TRITONSERVER_DataType = 6;
pub const TRITONSERVER_TYPE_INT16: TRITONSERVER_DataType = 7;
pub const TRITONSERVER_TYPE_INT32: TRITONSERVER_DataType = 8;
pub const TRITONSERVER_TYPE_INT64: TRITONSERVER_DataType = 9;
pub const TRITONSERVER_TYPE_FP16: TRITONSERVER_DataType = 10;
pub const TRITONSERVER_TYPE_FP32: TRITONSERVER_DataType = 11;
pub const TRITONSERVER_TYPE_FP64: TRITONSERVER_DataType = 12;
pub const TRITONSERVER_TYPE_BYTES: TRITONSERVER_DataType = 13;
pub const TRITONSERVER_TYPE_BF16: TRITONSERVER_DataType = 14;
pub type TRITONSERVER_MemoryType = std::ffi::c_uint;
pub const TRITONSERVER_MEMORY_CPU: TRITONSERVER_MemoryType = 0;
pub const TRITONSERVER_MEMORY_CPU_PINNED: TRITONSERVER_MemoryType = 1;
pub const TRITONSERVER_MEMORY_GPU: TRITONSERVER_MemoryType = 2;
pub type TRITONSERVER_Error_Code = std::ffi::c_uint;
pub const TRITONSERVER_ERROR_UNKNOWN: TRITONSERVER_Error_Code = 0;
pub const TRITONSERVER_ERROR_INTERNAL: TRITONSERVER_Error_Code = 1;
pub const TRITONSERVER_ERROR_NOT_FOUND: TRITONSERVER_Error_Code = 2;
pub const TRITONSERVER_ERROR_INVALID_ARG: TRITONSERVER_Error_Code = 3;
pub const TRITONSERVER_ERROR_UNAVAILABLE: TRITONSERVER_Error_Code = 4;
pub const TRITONSERVER_ERROR_UNSUPPORTED: TRITONSERVER_Error_Code = 5;
pub const TRITONSERVER_ERROR_ALREADY_EXISTS: TRITONSERVER_Error_Code = 6;
pub const TRITONSERVER_ERROR_CANCELLED: TRITONSERVER_Error_Code = 7;
pub type TRITONSERVER_ModelControlMode = std::ffi::c_uint;
pub const TRITONSERVER_MODEL_CONTROL_NONE: TRITONSERVER_ModelControlMode = 0;
pub const TRITONSERVER_MODEL_CONTROL_POLL: TRITONSERVER_ModelControlMode = 1;
pub const TRITONSERVER_MODEL_CONTROL_EXPLICIT: TRITONSERVER_ModelControlMode = 2;
pub type TRITONSERVER_RequestReleaseFlag = std::ffi::c_uint;
pub const TRITONSERVER_REQUEST_RELEASE_ALL: TRITONSERVER_RequestReleaseFlag = 1;
pub const TRITONSERVER_REQUEST_RELEASE_RESCHEDULE: TRITONSERVER_RequestReleaseFlag = 2;
pub type TRITONSERVER_ResponseCompleteFlag = std::ffi::c_uint;
pub const TRITONSERVER_RESPONSE_COMPLETE_FINAL: TRITONSERVER_ResponseCompleteFlag = 1;
pub type TRITONSERVER_ModelTxnPropertyFlag = std::ffi::c_uint;
pub const TRITONSERVER_TXN_ONE_TO_ONE: TRITONSERVER_ModelTxnPropertyFlag = 1;
pub const TRITONSERVER_TXN_DECOUPLED: TRITONSERVER_ModelTxnPropertyFlag = 2;
pub type TRITONSERVER_ResponseAllocatorAllocFn_t = Option<
    unsafe extern "C" fn(
        allocator: *mut TRITONSERVER_ResponseAllocator,
        tensor_name: *const std::ffi::c_char,
        byte_size: usize,
        memory_type: TRITONSERVER_MemoryType,
        memory_type_id: i64,
        userp: *mut std::ffi::c_void,
        buffer: *mut *mut std::ffi::c_void,
        buffer_userp: *mut *mut std::ffi::c_void,
        actual_memory_type: *mut TRITONSERVER_MemoryType,
        actual_memory_type_id: *mut i64,
    ) -> *mut TRITONSERVER_Error,
>;
pub type TRITONSERVER_ResponseAllocatorReleaseFn_t = Option<
    unsafe extern "C" fn(
        allocator: *mut TRITONSERVER_ResponseAllocator,
        buffer: *mut std::ffi::c_void,
        buffer_userp: *mut std::ffi::c_void,
        byte_size: usize,
        memory_type: TRITONSERVER_MemoryType,
        memory_type_id: i64,
    ) -> *mut TRITONSERVER_Error,
>;
pub type TRITONSERVER_ResponseAllocatorQueryFn_t = Option<
    unsafe extern "C" fn(
        allocator: *mut TRITONSERVER_ResponseAllocator,
        userp: *mut std::ffi::c_void,
        tensor_name: *const std::ffi::c_char,
        byte_size: *mut usize,
        memory_type: *mut TRITONSERVER_MemoryType,
        memory_type_id: *mut i64,
    ) -> *mut TRITONSERVER_Error,
>;
pub type TRITONSERVER_ResponseAllocatorStartFn_t = Option<
    unsafe extern "C" fn(
        allocator: *mut TRITONSERVER_ResponseAllocator,
        userp: *mut std::ffi::c_void,
    ) -> *mut TRITONSERVER_Error,
>;
pub type TRITONSERVER_InferenceRequestReleaseFn_t = Option<
    unsafe extern "C" fn(
        request: *mut TRITONSERVER_InferenceRequest,
        flags: u32,
        userp: *mut std::ffi::c_void,
    ) -> (),
>;
pub type TRITONSERVER_InferenceResponseCompleteFn_t = Option<
    unsafe extern "C" fn(
        response: *mut TRITONSERVER_InferenceResponse,
        flags: u32,
        userp: *mut std::ffi::c_void,
    ) -> (),
>;
extern "C" {
    pub fn TRITONSERVER_ApiVersion(major: *mut u32, minor: *mut u32) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerMetadata(
        server: *mut TRITONSERVER_Server,
        server_metadata: *mut *mut TRITONSERVER_Message,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerModelIndex(
        server: *mut TRITONSERVER_Server,
        flags: u32,
        model_index: *mut *mut TRITONSERVER_Message,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_MessageSerializeToJson(
        message: *mut TRITONSERVER_Message,
        base: *mut *const std::ffi::c_char,
        byte_size: *mut usize,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_MessageDelete(
        message: *mut TRITONSERVER_Message,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ErrorNew(
        code: TRITONSERVER_Error_Code,
        msg: *const std::ffi::c_char,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ErrorDelete(error: *mut TRITONSERVER_Error) -> ();
    pub fn TRITONSERVER_ErrorCode(error: *mut TRITONSERVER_Error) -> TRITONSERVER_Error_Code;
    pub fn TRITONSERVER_ErrorMessage(error: *mut TRITONSERVER_Error) -> *const std::ffi::c_char;
    pub fn TRITONSERVER_ServerOptionsNew(
        options: *mut *mut TRITONSERVER_ServerOptions,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerOptionsDelete(
        options: *mut TRITONSERVER_ServerOptions,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerOptionsSetModelRepositoryPath(
        options: *mut TRITONSERVER_ServerOptions,
        model_repository_path: *const std::ffi::c_char,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerOptionsSetBackendDirectory(
        options: *mut TRITONSERVER_ServerOptions,
        backend_dir: *const std::ffi::c_char,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerOptionsSetModelControlMode(
        options: *mut TRITONSERVER_ServerOptions,
        mode: TRITONSERVER_ModelControlMode,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerOptionsSetStrictModelConfig(
        options: *mut TRITONSERVER_ServerOptions,
        strict: bool,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerOptionsSetModelLoadRetryCount(
        options: *mut TRITONSERVER_ServerOptions,
        retry_count: std::ffi::c_uint,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerOptionsSetExitOnError(
        options: *mut TRITONSERVER_ServerOptions,
        exit: bool,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerOptionsSetLogInfo(
        options: *mut TRITONSERVER_ServerOptions,
        log: bool,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerOptionsSetLogWarn(
        options: *mut TRITONSERVER_ServerOptions,
        log: bool,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerNew(
        server: *mut *mut TRITONSERVER_Server,
        options: *mut TRITONSERVER_ServerOptions,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerDelete(server: *mut TRITONSERVER_Server) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerIsReady(
        server: *mut TRITONSERVER_Server,
        ready: *mut bool,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerModelTransactionProperties(
        server: *mut TRITONSERVER_Server,
        model_name: *const std::ffi::c_char,
        model_version: i64,
        txn_flags: *mut u32,
        voidp: *mut *mut std::ffi::c_void,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ResponseAllocatorNew(
        allocator: *mut *mut TRITONSERVER_ResponseAllocator,
        alloc_fn: TRITONSERVER_ResponseAllocatorAllocFn_t,
        release_fn: TRITONSERVER_ResponseAllocatorReleaseFn_t,
        start_fn: TRITONSERVER_ResponseAllocatorStartFn_t,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ResponseAllocatorDelete(
        allocator: *mut TRITONSERVER_ResponseAllocator,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ResponseAllocatorSetQueryFunction(
        allocator: *mut TRITONSERVER_ResponseAllocator,
        query_fn: TRITONSERVER_ResponseAllocatorQueryFn_t,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_InferenceRequestNew(
        inference_request: *mut *mut TRITONSERVER_InferenceRequest,
        server: *mut TRITONSERVER_Server,
        model_name: *const std::ffi::c_char,
        model_version: i64,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_InferenceRequestDelete(
        inference_request: *mut TRITONSERVER_InferenceRequest,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_InferenceRequestAddInput(
        inference_request: *mut TRITONSERVER_InferenceRequest,
        name: *const std::ffi::c_char,
        datatype: TRITONSERVER_DataType,
        shape: *const i64,
        dim_count: u64,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_InferenceRequestAppendInputData(
        inference_request: *mut TRITONSERVER_InferenceRequest,
        name: *const std::ffi::c_char,
        base: *const std::ffi::c_void,
        byte_size: usize,
        memory_type: TRITONSERVER_MemoryType,
        memory_type_id: i64,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_InferenceRequestSetReleaseCallback(
        inference_request: *mut TRITONSERVER_InferenceRequest,
        request_release_fn: TRITONSERVER_InferenceRequestReleaseFn_t,
        request_release_userp: *mut std::ffi::c_void,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_InferenceRequestSetResponseCallback(
        inference_request: *mut TRITONSERVER_InferenceRequest,
        response_allocator: *mut TRITONSERVER_ResponseAllocator,
        response_allocator_userp: *mut std::ffi::c_void,
        response_fn: TRITONSERVER_InferenceResponseCompleteFn_t,
        response_userp: *mut std::ffi::c_void,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_ServerInferAsync(
        server: *mut TRITONSERVER_Server,
        inference_request: *mut TRITONSERVER_InferenceRequest,
        trace: *mut TRITONSERVER_InferenceTrace,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_InferenceResponseDelete(
        inference_response: *mut TRITONSERVER_InferenceResponse,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_InferenceResponseError(
        inference_response: *mut TRITONSERVER_InferenceResponse,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_InferenceResponseOutputCount(
        inference_response: *mut TRITONSERVER_InferenceResponse,
        count: *mut u32,
    ) -> *mut TRITONSERVER_Error;
    pub fn TRITONSERVER_InferenceResponseOutput(
        inference_response: *mut TRITONSERVER_InferenceResponse,
        index: u32,
        name: *mut *const std::ffi::c_char,
        datatype: *mut TRITONSERVER_DataType,
        shape: *mut *const i64,
        dim_count: *mut u64,
        base: *mut *const std::ffi::c_void,
        byte_size: *mut usize,
        memory_type: *mut TRITONSERVER_MemoryType,
        memory_type_id: *mut i64,
        userp: *mut *mut std::ffi::c_void,
    ) -> *mut TRITONSERVER_Error;
}
