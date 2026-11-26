// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native tokenizer library shared by the ONNX Runtime and Torch adapters.
use std::{ffi::CString, panic::catch_unwind, ptr, sync::OnceLock};

use ort::{operator::OperatorDomain, AsPointer};
use ort_sys::{OrtApiBase, OrtErrorCode, OrtSessionOptions, OrtStatus};


// ORT sessions borrow the operator domain. Keep one domain alive for the plugin's
// lifetime, rather than leaking a domain per session. usize stores its address
// without claiming that arbitrary ORT pointer operations are thread-safe.
static DOMAIN: OnceLock<Result<usize, String>> = OnceLock::new();
static API_ADDRESS: OnceLock<usize> = OnceLock::new();

#[no_mangle]
/// Register LEAPP text operators with an ONNX Runtime host.
///
/// # Safety
/// Both pointers must be valid, non-null objects supplied by ONNX Runtime.
/// The host API library must remain loaded while the plugin or its sessions
/// can call that API, including during process shutdown.
pub unsafe extern "system" fn RegisterCustomOps(
    options: *mut OrtSessionOptions,
    base: *const OrtApiBase,
) -> *mut OrtStatus {
    let api = ((*base).GetApi)(23);
    if api.is_null() {
        let old = ((*base).GetApi)(17);
        return ((*old).CreateStatus)(
            OrtErrorCode::ORT_FAIL,
            c"LEAPP text operators require ONNX Runtime 1.23 or newer".as_ptr(),
        )
        .0;
    }
    let result = catch_unwind(|| -> Result<(), String> {
        let address = *API_ADDRESS.get_or_init(|| {
            ort::set_api(ptr::read(api));
            api as usize
        });
        if address != api as usize {
            return Err("LEAPP text plugin cannot share two ONNX Runtime API instances".into());
        }
        let domain = DOMAIN.get_or_init(|| {
            OperatorDomain::new("com.leapp.text")
                .and_then(|domain| domain.add(engine::HfTokenizerUtf8))
                .map(|domain| Box::into_raw(Box::new(domain)) as usize)
                .map_err(|error| error.to_string())
        });
        let address = domain.as_ref().map_err(Clone::clone)?;
        let domain = &*(*address as *const OperatorDomain);
        let status = ((*api).AddCustomOpDomain)(options, domain.ptr().cast_mut());
        if !status.0.is_null() {
            // Convert the host error into the common registration-error path.
            let message = std::ffi::CStr::from_ptr(((*api).GetErrorMessage)(status.0))
                .to_string_lossy()
                .into_owned();
            ((*api).ReleaseStatus)(status.0);
            return Err(message);
        }
        Ok(())
    });
    match result {
        Ok(Ok(())) => ptr::null_mut(),
        failure => {
            let message = match failure {
                Ok(Err(message)) => message,
                _ => "Panic while registering LEAPP text operators".into(),
            };
            let message = CString::new(message.replace('\0', " ")).unwrap();
            ((*api).CreateStatus)(OrtErrorCode::ORT_FAIL, message.as_ptr()).0
        }
    }
}

/// Shared tokenizer engine, ONNX adapter, and ownership-safe C ABI for Torch.
mod engine {
    use ort::{
        operator::{
            BoxedKernel, ComputeContext, KernelContext, Operator, OperatorInput, OperatorOutput,
        },
        value::TensorElementType,
        Error, Result,
    };
    use std::{
        cell::RefCell,
        ffi::{c_char, CString},
        panic::{catch_unwind, AssertUnwindSafe},
        ptr,
    };
    use tokenizers::{Encoding, Tokenizer};

    pub fn encode(
        tokenizer: &Tokenizer,
        bytes: &[u8],
        lengths: &[i64],
        capacity: usize,
        special: bool,
    ) -> std::result::Result<Vec<Encoding>, String> {
        if lengths.is_empty()
            || capacity == 0
            || bytes.len()
                != lengths
                    .len()
                    .checked_mul(capacity)
                    .ok_or("text size overflow")?
        {
            return Err("expected uint8[batch, capacity] and int64[batch], with nonempty batch and positive capacity".into());
        }
        let texts = lengths
            .iter()
            .enumerate()
            .map(|(row, &length)| {
                let length = usize::try_from(length).map_err(|_| "negative UTF-8 length")?;
                if length > capacity {
                    return Err("UTF-8 length exceeds capacity".into());
                }
                std::str::from_utf8(&bytes[row * capacity..row * capacity + length])
                    .map_err(|e| e.to_string())
            })
            .collect::<std::result::Result<Vec<_>, String>>()?;
        let encoded = tokenizer
            .encode_batch(texts, special)
            .map_err(|e| e.to_string())?;
        let width = encoded[0].len();
        if encoded.iter().any(|item| item.len() != width) {
            return Err("tokenizer configuration must enable batch padding".into());
        }
        Ok(encoded)
    }

    pub struct HfTokenizerUtf8;
    impl Operator for HfTokenizerUtf8 {
        type Kernel<'attr> = BoxedKernel<'attr>;
        fn name(&self) -> &str {
            "HfTokenizerUtf8"
        }
        fn versions(&self) -> std::ops::RangeInclusive<u32> {
            1..=1
        }
        fn inputs(&self) -> impl IntoIterator<Item = OperatorInput> {
            [
                OperatorInput::required(TensorElementType::Uint8),
                OperatorInput::required(TensorElementType::Int64),
            ]
        }
        fn outputs(&self) -> impl IntoIterator<Item = OperatorOutput> {
            std::iter::repeat_with(|| OperatorOutput::required(TensorElementType::Int64)).take(3)
        }
        fn create_kernel<'attr>(&self, context: &KernelContext<'attr>) -> Result<Self::Kernel<'attr>> {
            let json: String = context
                .get("tokenizer_json")
                .ok_or_else(|| Error::new("missing tokenizer_json"))?;
            let tokenizer =
                Tokenizer::from_bytes(json.as_bytes()).map_err(|e| Error::new(e.to_string()))?;
            let special = context.get::<i64>("add_special_tokens").unwrap_or(1) != 0;
            Ok(Box::new(move |context: &ComputeContext| {
                let input = context
                    .input(0)?
                    .ok_or_else(|| Error::new("missing UTF-8 bytes"))?;
                let lengths = context
                    .input(1)?
                    .ok_or_else(|| Error::new("missing UTF-8 lengths"))?;
                let (shape, bytes) = input.try_extract_tensor::<u8>()?;
                let (length_shape, lengths) = lengths.try_extract_tensor::<i64>()?;
                if shape.len() != 2 || length_shape.as_ref() != [shape[0]] {
                    return Err(Error::new("UTF-8 tensor shape mismatch"));
                }
                let encoded = encode(&tokenizer, bytes, lengths, shape[1] as usize, special)
                    .map_err(Error::new)?;
                for index in 0..3 {
                    let mut output = context
                        .output(index, [encoded.len() as i64, encoded[0].len() as i64])?
                        .ok_or_else(|| Error::new("missing token output"))?;
                    let (_, data) = output.try_extract_tensor_mut::<i64>()?;
                    for (row, item) in encoded.iter().enumerate() {
                        let source = match index {
                            0 => item.get_ids(),
                            1 => item.get_attention_mask(),
                            _ => item.get_type_ids(),
                        };
                        for (column, &value) in source.iter().enumerate() {
                            data[row * source.len() + column] = i64::from(value);
                        }
                    }
                }
                Ok(())
            }))
        }
    }

    thread_local! { static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default()); }
    fn ffi<T>(function: impl FnOnce() -> std::result::Result<T, String>) -> Option<T> {
        match catch_unwind(AssertUnwindSafe(function)) {
            Ok(Ok(value)) => Some(value),
            failure => {
                let error = match failure {
                    Ok(Err(error)) => error,
                    _ => "panic in native tokenizer".into(),
                };
                LAST_ERROR
                    .with(|slot| *slot.borrow_mut() = CString::new(error.replace('\0', " ")).unwrap());
                None
            }
        }
    }

    #[repr(C)]
    pub struct TokenBatch {
        pub rows: i64,
        pub width: i64,
        pub ids: *const i64,
        pub mask: *const i64,
        pub types: *const i64,
        owned: [Vec<i64>; 3],
    }

    #[no_mangle]
    pub extern "C" fn leapp_text_error() -> *const c_char {
        LAST_ERROR.with(|slot| slot.borrow().as_ptr())
    }

    /// # Safety
    /// `json` must reference `size` readable bytes for this call.
    /// Release a nonnull returned handle with `leapp_tokenizer_destroy`.
    #[no_mangle]
    pub unsafe extern "C" fn leapp_tokenizer_create(json: *const u8, size: usize) -> *mut Tokenizer {
        ffi(|| {
            Tokenizer::from_bytes(std::slice::from_raw_parts(json, size))
                .map(Box::new)
                .map(Box::into_raw)
                .map_err(|e| e.to_string())
        })
        .unwrap_or(ptr::null_mut())
    }
    /// # Safety
    /// `handle` must be null or an unreleased handle from `leapp_tokenizer_create`.
    #[no_mangle]
    pub unsafe extern "C" fn leapp_tokenizer_destroy(handle: *mut Tokenizer) {
        if !handle.is_null() {
            drop(Box::from_raw(handle));
        }
    }
    /// # Safety
    /// `batch` must be null or an unreleased result from `leapp_tokenize`.
    #[no_mangle]
    pub unsafe extern "C" fn leapp_token_batch_destroy(batch: *mut TokenBatch) {
        if !batch.is_null() {
            drop(Box::from_raw(batch));
        }
    }
    /// # Safety
    /// The handle must remain valid; bytes and lengths must reference respectively
    /// `rows * capacity` bytes and `rows` aligned int64 values throughout this call.
    /// Release a nonnull result with `leapp_token_batch_destroy`.
    #[no_mangle]
    pub unsafe extern "C" fn leapp_tokenize(
        handle: *const Tokenizer,
        bytes: *const u8,
        lengths: *const i64,
        rows: usize,
        capacity: usize,
        special: bool,
    ) -> *mut TokenBatch {
        ffi(|| {
            let size = rows.checked_mul(capacity).ok_or("UTF-8 size overflow")?;
            let encoded = encode(
                &*handle,
                std::slice::from_raw_parts(bytes, size),
                std::slice::from_raw_parts(lengths, rows),
                capacity,
                special,
            )?;
            let owned: [Vec<i64>; 3] = std::array::from_fn(|index| {
                encoded
                    .iter()
                    .flat_map(|item| match index {
                        0 => item.get_ids(),
                        1 => item.get_attention_mask(),
                        _ => item.get_type_ids(),
                    })
                    .map(|&value| i64::from(value))
                    .collect()
            });
            Ok(Box::into_raw(Box::new(TokenBatch {
                rows: rows as i64,
                width: encoded[0].len() as i64,
                ids: owned[0].as_ptr(),
                mask: owned[1].as_ptr(),
                types: owned[2].as_ptr(),
                owned,
            })))
        })
        .unwrap_or(ptr::null_mut())
    }
}
