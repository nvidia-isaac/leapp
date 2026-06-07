import pytest
wp = pytest.importorskip("warp")
torch = pytest.importorskip("torch")
from leapp.backends import warp_dtypes as wd


def test_scalar_roundtrip():
    assert wd.warp_dtype_to_str(wp.float32) == "float32"
    assert wd.str_to_warp_dtype("float32") is wp.float32


def test_struct_dtype_metadata():
    assert wd.warp_dtype_to_str(wp.vec3f) == "vec3f"
    assert wd.str_to_warp_dtype("vec3f") is wp.vec3f
    assert wd.scalar_base_str("vec3f") == "float32"
    assert wd.scalar_count("vec3f") == 3
    assert wd.trailing_shape("vec3f") == (3,)


def test_transform_and_matrix():
    assert wd.scalar_count("transformf") == 7
    assert wd.trailing_shape("transformf") == (7,)
    assert wd.scalar_count("mat33f") == 9
    assert wd.trailing_shape("mat33f") == (3, 3)


def test_unknown_dtype_raises():
    with pytest.raises(KeyError):
        wd.str_to_warp_dtype("not_a_dtype")


def test_torch_dtype_to_warp_str():
    assert wd.torch_dtype_to_warp_str(torch.float32) == "float32"
    assert wd.torch_dtype_to_warp_str(torch.int32) == "int32"
    with pytest.raises(KeyError):
        wd.torch_dtype_to_warp_str(torch.complex64)
