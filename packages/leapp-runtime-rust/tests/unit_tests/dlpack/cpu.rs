// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use lrt::dlpack::*;

#[test]
fn cpu_slices_infer_dtype_and_validate_shape_without_copying() {
    let values = [1f32, 2.];
    let input = BorrowedTensor::cpu(&values, &[1, 2]).unwrap();
    let dtype = input.descriptor().dtype;
    assert_eq!((dtype.code, dtype.bits, dtype.lanes), (2, 32, 1));
    assert_eq!(
        input.as_cpu_slice::<f32>().unwrap().as_ptr(),
        values.as_ptr()
    );
    for (values, shape) in [(&[7i32][..], &[][..]), (&[][..], &[2, 0][..])] {
        assert_eq!(
            BorrowedTensor::cpu(values, shape)
                .unwrap()
                .as_cpu_slice::<i32>()
                .unwrap(),
            values
        );
    }
    for shape in [&[3][..], &[-1][..], &[i64::MAX, 8][..], &[0, -1][..]] {
        assert!(BorrowedTensor::cpu(&[1i32, 2], shape).is_err());
    }
}

#[test]
fn cpu_reads_check_layout_and_apply_byte_offsets() {
    let values = [1f32, 2., 3., 4.];
    let input = BorrowedTensor::cpu(&values, &[4]).unwrap();
    let mut descriptor = *input.descriptor();
    let shape = [2];
    let strides = [1];
    descriptor.shape = shape.as_ptr().cast_mut();
    descriptor.strides = strides.as_ptr().cast_mut();
    descriptor.byte_offset = 8;
    // SAFETY: descriptors below refer to these live, immutable arrays; error
    // cases are rejected before reading storage through incompatible metadata.
    let borrowed = unsafe { BorrowedTensor::new(&descriptor) };
    assert_eq!(borrowed.as_cpu_slice::<f32>().unwrap(), &[3., 4.]);
    assert!(borrowed.as_cpu_slice::<u32>().is_err());
    let strided = [2];
    for scenario in 0..4 {
        let mut bad = descriptor;
        match scenario {
            0 => bad.device.device_type = kDLCUDA as i32,
            1 => {
                bad.strides = strided.as_ptr().cast_mut();
                bad.byte_offset = 0;
            }
            2 => {
                // Offset a naturally aligned f32 pointer by one byte.
                bad.data = values.as_ptr().cast_mut().cast();
                bad.byte_offset = 1;
            }
            _ => {
                bad.dtype.lanes = 2;
            }
        }
        assert!(unsafe { BorrowedTensor::new(&bad) }
            .as_cpu_slice::<f32>()
            .is_err());
    }
}
