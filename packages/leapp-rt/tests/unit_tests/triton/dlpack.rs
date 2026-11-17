// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

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
