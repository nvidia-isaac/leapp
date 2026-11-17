# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the pinned DLPack ABI with clang; normal builds need no generator."""

import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent
header = root / "include/dlpack/dlpack.h"
source = header.read_text()
ast = json.loads(
    subprocess.check_output(
        ["clang", "-x", "c", "-Xclang", "-ast-dump=json", "-fsyntax-only", str(header)]
    )
)
records = {n["id"]: n for n in ast["inner"] if n["kind"] == "RecordDecl"}
types = {
    "int32_t": "i32",
    "int64_t": "i64",
    "uint8_t": "u8",
    "uint16_t": "u16",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "void": "std::ffi::c_void",
}
selected = {
    "DLPackVersion",
    "DLDevice",
    "DLDataType",
    "DLTensor",
    "DLManagedTensor",
    "DLManagedTensorVersioned",
}


def rust_type(value):
    if value.endswith(" *"):
        return "*mut " + rust_type(value[:-2])
    value = value.removeprefix("struct ")
    return types.get(value, value)


out = [
    "// Generated from upstream DLPack v1.0 by generate_bindings.py. Do not edit.",
    "#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]",
    "pub type DLDeviceType = i32;",
]
for suffix in ["MAJOR_VERSION", "MINOR_VERSION"]:
    name = f"DLPACK_{suffix}"
    value = re.search(rf"#define {name} (\d+)", source)[1]
    out.append(f"pub const {name}: u32 = {value};")
for suffix in ["READ_ONLY", "IS_COPIED"]:
    name = f"DLPACK_FLAG_BITMASK_{suffix}"
    shift = re.search(rf"#define {name} \(1UL << (\d+)UL\)", source)[1]
    out.append(f"pub const {name}: u64 = {1 << int(shift)};")

for node in ast["inner"]:
    if node["kind"] == "EnumDecl":
        for item in node.get("inner", []):
            if item["kind"] != "EnumConstantDecl":
                continue
            value_node = item["inner"][0]
            while "value" not in value_node:
                value_node = value_node["inner"][0]
            out.append(f"pub const {item['name']}: u32 = {value_node['value']};")
    if node["kind"] not in {"TypedefDecl", "RecordDecl"}:
        continue
    if node.get("name") not in selected:
        continue
    # In v1.0 the versioned descriptor is a named struct without a C typedef.
    if node["kind"] == "RecordDecl" and node["name"] != "DLManagedTensorVersioned":
        continue
    name = node["name"]
    if node["kind"] == "RecordDecl":
        record = node
    else:
        declaration = next(n for n in node["inner"] if n["kind"] == "ElaboratedType")
        record = records[declaration["ownedTagDecl"]["id"]]
    out.extend(["#[repr(C)]", "#[derive(Clone, Copy, Debug)]", f"pub struct {name} {{"])
    for field in record["inner"]:
        if field["kind"] != "FieldDecl":
            continue
        value = (
            f'Option<unsafe extern "C" fn(*mut {name})>'
            if field["name"] == "deleter"
            else rust_type(field["type"]["qualType"])
        )
        out.append(f"    pub {field['name']}: {value},")
    out.append("}")
(root / "sys.rs").write_text("\n".join(out) + "\n")
subprocess.run(["rustfmt", str(root / "sys.rs")], check=True)
