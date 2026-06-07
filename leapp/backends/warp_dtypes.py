"""Single source of truth for warp dtype <-> string, including structured dtypes.

Each entry records: the warp dtype object, its scalar base (e.g. float32), the number of
scalars, and the trailing torch-view shape. Scalars have count 1 and trailing shape ().
The torch *view* of a structured-dtype port is always its scalar base with the trailing
shape appended to the per-element (batch) dims.
"""
import torch
import warp as wp

# name -> (warp dtype, scalar base name, scalar count, trailing torch-view shape)
_REGISTRY = {
    "float16": (wp.float16, "float16", 1, ()),
    "float32": (wp.float32, "float32", 1, ()),
    "float64": (wp.float64, "float64", 1, ()),
    "int8":  (wp.int8,  "int8",  1, ()),
    "int16": (wp.int16, "int16", 1, ()),
    "int32": (wp.int32, "int32", 1, ()),
    "int64": (wp.int64, "int64", 1, ()),
    "uint8": (wp.uint8, "uint8", 1, ()),
    "bool":  (wp.bool,  "bool",  1, ()),
    "vec2f": (wp.vec2f, "float32", 2, (2,)),
    "vec3f": (wp.vec3f, "float32", 3, (3,)),
    "vec4f": (wp.vec4f, "float32", 4, (4,)),
    "quatf": (wp.quatf, "float32", 4, (4,)),
    "transformf": (wp.transformf, "float32", 7, (7,)),
    "mat33f": (wp.mat33f, "float32", 9, (3, 3)),
    "mat44f": (wp.mat44f, "float32", 16, (4, 4)),
}
_WARP_TO_NAME = {entry[0]: name for name, entry in _REGISTRY.items()}


def warp_dtype_to_str(dtype) -> str:
    name = _WARP_TO_NAME.get(dtype)
    if name is None:
        raise KeyError(f"unsupported warp dtype {dtype!r}")
    return name


def str_to_warp_dtype(name: str):
    return _REGISTRY[name][0]


def scalar_base_str(name: str) -> str:
    return _REGISTRY[name][1]


def scalar_count(name: str) -> int:
    return _REGISTRY[name][2]


def trailing_shape(name: str) -> tuple:
    return _REGISTRY[name][3]


def is_structured(name: str) -> bool:
    return _REGISTRY[name][2] > 1


_TORCH_TO_WARP_NAME = {
    torch.float16: "float16", torch.float32: "float32", torch.float64: "float64",
    torch.int8: "int8", torch.int16: "int16", torch.int32: "int32", torch.int64: "int64",
    torch.uint8: "uint8", torch.bool: "bool",
}


def torch_dtype_to_warp_str(torch_dtype) -> str:
    """Map a torch scalar dtype to the warp SCALAR dtype name (used when wp.from_torch is
    called without an explicit warp dtype — warp then infers a scalar array of this dtype)."""
    name = _TORCH_TO_WARP_NAME.get(torch_dtype)
    if name is None:
        raise KeyError(f"unsupported torch dtype {torch_dtype!r} for warp bridge")
    return name
