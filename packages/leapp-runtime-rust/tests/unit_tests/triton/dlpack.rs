// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use super::*;

#[test]
fn contiguous_offsets_scalars_and_empty_storage() {
    let values = [1.0f32, 2., 3., 4.];
    let mut shape = [2i64];
    let stride = [1i64];
    let mut t = *BorrowedTensor::cpu(&values, &[4]).unwrap().descriptor();
    t.shape = shape.as_mut_ptr();
    t.strides = stride.as_ptr().cast_mut();
    t.byte_offset = 8;
    let borrowed = unsafe { BorrowedTensor::new(&t) };
    assert_eq!(layout(borrowed).unwrap(), (&[2][..], 8));
    let copy = copy(borrowed, None).unwrap();
    assert_eq!(copy.to_cpu_vec::<f32>().unwrap(), [3., 4.]);
    t.ndim = 0;
    t.shape = ptr::null_mut();
    t.strides = ptr::null_mut();
    assert_eq!(
        layout(unsafe { BorrowedTensor::new(&t) }).unwrap(),
        (&[][..], 4)
    );
    shape[0] = 0;
    t.ndim = 1;
    t.shape = shape.as_mut_ptr();
    t.data = ptr::null_mut();
    t.byte_offset = 0;
    assert_eq!(
        layout(unsafe { BorrowedTensor::new(&t) }).unwrap(),
        (&[0][..], 0)
    );
}
#[test]
fn unsupported_layouts_and_metadata_fail_before_execution() {
    let values = [0f32; 8];
    let stride = [2i64];
    let valid = *BorrowedTensor::cpu(&values, &[8]).unwrap().descriptor();
    let mut bad = valid;
    bad.strides = stride.as_ptr().cast_mut();
    assert!(layout(unsafe { BorrowedTensor::new(&bad) })
        .unwrap_err()
        .to_string()
        .contains("contiguous"));
    for scenario in 0..7 {
        let mut bad = valid;
        match scenario {
            0 => bad.ndim = -1,
            1 => bad.shape = ptr::null_mut(),
            2 => bad.data = ptr::null_mut(),
            3 => bad.dtype.lanes = 2,
            4 => bad.dtype.code = 255,
            5 => bad.byte_offset = u64::MAX,
            _ => bad.device.device_type = 99,
        }
        assert!(layout(unsafe { BorrowedTensor::new(&bad) }).is_err());
    }
    assert!(byte_size(valid.dtype, &[-1]).is_err());
    assert!(byte_size(valid.dtype, &[i64::MAX, 8]).is_err());
}
#[test]
fn native_dtypes_match_explicit_dlpack_codes_and_widths() {
    for (native, code, bits) in [
        (sys::TRITONSERVER_TYPE_BOOL, 6, 8),
        (sys::TRITONSERVER_TYPE_UINT8, 1, 8),
        (sys::TRITONSERVER_TYPE_UINT16, 1, 16),
        (sys::TRITONSERVER_TYPE_UINT32, 1, 32),
        (sys::TRITONSERVER_TYPE_UINT64, 1, 64),
        (sys::TRITONSERVER_TYPE_INT8, 0, 8),
        (sys::TRITONSERVER_TYPE_INT16, 0, 16),
        (sys::TRITONSERVER_TYPE_INT32, 0, 32),
        (sys::TRITONSERVER_TYPE_INT64, 0, 64),
        (sys::TRITONSERVER_TYPE_FP16, 2, 16),
        (sys::TRITONSERVER_TYPE_FP32, 2, 32),
        (sys::TRITONSERVER_TYPE_FP64, 2, 64),
        (sys::TRITONSERVER_TYPE_BF16, 4, 16),
    ] {
        let dtype = dtype_from_native(native).unwrap();
        assert_eq!((dtype.code, dtype.bits, dtype.lanes), (code, bits, 1));
        assert_eq!(
            dtype_to_native(DLDataType {
                code,
                bits,
                lanes: 1
            })
            .unwrap(),
            native
        );
    }
}
