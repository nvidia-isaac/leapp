# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise DLPack ownership and flags independently of consumer conventions."""

import ctypes as c
import gc

import numpy as np
import pytest
from test_runtime import CUDA, runtime, step


class Device(c.Structure):
    _fields_ = [("kind", c.c_int32), ("id", c.c_int32)]


class Dtype(c.Structure):
    _fields_ = [("code", c.c_uint8), ("bits", c.c_uint8), ("lanes", c.c_uint16)]


class Tensor(c.Structure):
    _fields_ = [
        ("data", c.c_void_p),
        ("device", Device),
        ("ndim", c.c_int32),
        ("dtype", Dtype),
        ("shape", c.POINTER(c.c_int64)),
        ("strides", c.POINTER(c.c_int64)),
        ("byte_offset", c.c_uint64),
    ]


class Legacy(c.Structure):
    pass


Delete = c.CFUNCTYPE(None, c.POINTER(Legacy))
Legacy._fields_ = [("tensor", Tensor), ("context", c.c_void_p), ("deleter", Delete)]


class Version(c.Structure):
    _fields_ = [("major", c.c_uint32), ("minor", c.c_uint32)]


class Versioned(c.Structure):
    _fields_ = [
        ("version", Version),
        ("context", c.c_void_p),
        ("deleter", c.c_void_p),
        ("flags", c.c_uint64),
        ("tensor", Tensor),
    ]


VersionedDelete = c.CFUNCTYPE(None, c.POINTER(Versioned))

capsule_new = c.pythonapi.PyCapsule_New
capsule_new.restype = c.py_object
capsule_new.argtypes = [c.c_void_p, c.c_char_p, c.c_void_p]
capsule_pointer = c.pythonapi.PyCapsule_GetPointer
capsule_pointer.restype = c.c_void_p
capsule_pointer.argtypes = [c.py_object, c.c_char_p]


class Producer:
    def __init__(
        self, *, stride=1, offset=0, declared=(1, 0), actual=(1, 0), versioned=False
    ):
        self.source = np.arange(8, dtype=np.float32)
        self.shape = (c.c_int64 * 1)(4)
        self.strides = (c.c_int64 * 1)(stride)
        self.deleted = 0

        self.declared = declared

        def delete(_):
            self.deleted += 1

        self.delete = (VersionedDelete if versioned else Delete)(delete)
        tensor = Tensor(
            self.source.ctypes.data,
            Device(*actual),
            1,
            Dtype(2, 32, 1),
            self.shape,
            self.strides,
            offset,
        )
        if versioned:
            self.managed = Versioned(
                Version(1, 0), None, c.cast(self.delete, c.c_void_p), 0, tensor
            )
        else:
            self.managed = Legacy(tensor, None, self.delete)
        # The test object retains the backing structures. A real producer would
        # release its allocation from its deleter.
        name = b"dltensor_versioned" if versioned else b"dltensor"
        self.capsule = capsule_new(c.addressof(self.managed), name, None)

    def __dlpack_device__(self):
        return self.declared

    def __dlpack__(self, *, max_version=None, stream=None):
        return self.capsule


@pytest.mark.parametrize("stride", [1, 2])
def test_consumed_capsule_deleted_once_including_failure(stride):
    with runtime("dynamic") as rt:
        session = rt.create_session()
        producer = Producer(stride=stride, offset=8)
        if stride == 1:
            result = session.run({"dynamic/input": producer})
            np.testing.assert_array_equal(result.numpy("dynamic/output"), [2, 3, 4, 5])
        else:
            with pytest.raises(BufferError, match="contiguous"):
                session.run({"dynamic/input": producer})
        assert producer.deleted == 1
        with pytest.raises(BufferError, match="consumed"):
            session.run({"dynamic/input": producer})
        assert producer.deleted == 1
        del producer.capsule
        gc.collect()
        assert producer.deleted == 1


def test_versioned_flags_and_unconsumed_cleanup():
    with runtime() as rt:
        tensor = step(rt.create_session())
        for producer, kwargs, expected in [
            (tensor, {}, 2),
            (tensor.dlpack_view(), {}, 1),
            (tensor.dlpack_view(), {"copy": True}, 2),
        ]:
            capsule = producer.__dlpack__(max_version=(1, 0), **kwargs)
            ptr = capsule_pointer(capsule, b"dltensor_versioned")
            managed = c.cast(ptr, c.POINTER(Versioned)).contents
            assert managed.version.major == 1
            assert managed.flags == expected
            assert managed.tensor.device.kind == 1
            assert managed.tensor.shape[0] == 4
            del capsule
        for _ in range(100):
            tensor.__dlpack__()
            tensor.__dlpack__(max_version=(1, 0))
        gc.collect()
        np.testing.assert_array_equal(tensor.numpy(), [1] * 4)


@pytest.mark.skipif(not CUDA, reason="set LEAPP_TEST_CUDA=1")
@pytest.mark.parametrize("versioned", [False, True])
@pytest.mark.parametrize(
    "declared,actual,accepted",
    [
        ((3, 0), (1, 0), True),
        ((1, 0), (2, 0), False),
        ((2, 0), (1, 0), False),
        ((2, 0), (2, 1), False),
    ],
)
def test_import_device_normalization_and_ownership(
    versioned, declared, actual, accepted
):
    producer = Producer(declared=declared, actual=actual, versioned=versioned)
    with runtime("dynamic") as rt:
        session = rt.create_session()
        if accepted:
            result = session.run({"dynamic/input": producer})
            np.testing.assert_array_equal(result.numpy("dynamic/output"), [0, 1, 2, 3])
        else:
            with pytest.raises(BufferError, match="device descriptor"):
                session.run({"dynamic/input": producer})
        assert producer.deleted == 1
        del producer.capsule
        gc.collect()
        assert producer.deleted == 1
