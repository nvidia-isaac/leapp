# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate sys.rs from the pinned Triton release's tritonserver.h.

When upgrading Triton, update the release label below, ../convert/model_config.proto
from the matching triton-inference-server/common tag, and TRITON_VERSION in
../native.rs together. Run this script with the new SDK's include/triton/core/
tritonserver.h, then rustfmt --edition 2021 sys.rs. Review callback ownership
changes and run the native CPU/CUDA and binding suites.
"""

from pathlib import Path
import re
import argparse
parser = argparse.ArgumentParser(description="Extract the used declarations from the pinned Triton C header")
parser.add_argument("header", type=Path)
parser.add_argument("--output", type=Path, default=Path(__file__).parent / "sys.rs")
arguments = parser.parse_args()
h=arguments.header.read_text()
# Strip comments before parsing the selected, simple C declarations.
s=re.sub(r'/\*.*?\*/|//[^\n]*','',h,flags=re.S)
functions='''ApiVersion ServerMetadata MessageSerializeToJson MessageDelete ErrorNew ErrorDelete ErrorCode ErrorMessage ServerOptionsNew ServerOptionsDelete ServerOptionsSetModelRepositoryPath ServerOptionsSetBackendDirectory ServerOptionsSetModelControlMode ServerOptionsSetStrictModelConfig ServerOptionsSetModelLoadRetryCount ServerOptionsSetExitOnError ServerOptionsSetLogInfo ServerOptionsSetLogWarn ServerNew ServerDelete ServerIsReady ServerModelTransactionProperties ResponseAllocatorNew ResponseAllocatorDelete ResponseAllocatorSetQueryFunction InferenceRequestNew InferenceRequestDelete InferenceRequestAddInput InferenceRequestAppendInputData InferenceRequestSetReleaseCallback InferenceRequestSetResponseCallback ServerInferAsync InferenceResponseDelete InferenceResponseError InferenceResponseOutputCount InferenceResponseOutput'''.split()
callbacks='''ResponseAllocatorAllocFn_t ResponseAllocatorReleaseFn_t ResponseAllocatorQueryFn_t ResponseAllocatorStartFn_t InferenceRequestReleaseFn_t InferenceResponseCompleteFn_t'''.split()
types={'char':'std::ffi::c_char','void':'std::ffi::c_void','uint32_t':'u32','uint64_t':'u64','int64_t':'i64','size_t':'usize','bool':'bool','int':'std::ffi::c_int','unsigned int':'std::ffi::c_uint'}
used=set()


def ty(c):
    c=c.strip().replace('struct ','')
    parts=c.split('*'); base=parts[0].replace('const','').strip()
    rust=types.get(base,base)
    if base.startswith('TRITONSERVER_'):used.add(base)
    const='const' in parts[0]
    for suffix in parts[1:]:
        rust=('*const ' if const else '*mut ')+rust
        const='const' in suffix
    return '()' if c=='void' else rust


def args(c):
    if not c.strip() or c.strip()=='void':return ''
    result=[]
    for arg in c.split(','):
        m=re.fullmatch(r'\s*(.+?)([A-Za-z_]\w*)\s*',arg,flags=re.S)
        name=m[2]; name = 'r#'+name if name in {'type','in','ref'} else name
        result.append(name+': '+ty(m[1]))
    return ', '.join(result)


out=['// Generated subset of the Triton 2.69.0 C API; see generate_bindings.py.','#![allow(non_camel_case_types, non_snake_case, dead_code)]']
for name in ['MAJOR','MINOR']:
    value=re.search(r'#define TRITONSERVER_API_VERSION_'+name+r' (\d+)',h)[1]
    out.append('pub const TRITONSERVER_API_VERSION_'+name+': u32 = '+value+';')
for name in ['DataType','MemoryType','Error_Code','ModelControlMode','RequestReleaseFlag','ResponseCompleteFlag','ModelTxnPropertyFlag']:
    match=re.search(r'typedef enum \w*\s*\{([^}]+)\}\s*TRITONSERVER_'+name+r'\s*;',s)
    if not match:
        if name=='ModelTxnPropertyFlag':continue
        raise ValueError(name)
    out.append('pub type TRITONSERVER_'+name+' = std::ffi::c_uint;')
    value=0
    for item in match[1].split(','):
        item=item.strip()
        if not item:continue
        if '=' in item:
            key,raw=map(str.strip,item.split('='));value=int(raw)
        else:key=item
        out.append(f'pub const {key}: TRITONSERVER_{name} = {value};');value+=1

for short in callbacks:
    name='TRITONSERVER_'+short
    m=re.search(r'typedef\s+([^;]+?)\(\*'+name+r'\)\s*\((.*?)\);',s,flags=re.S)
    assert m,name
    out.append(f'pub type {name} = Option<unsafe extern "C" fn({args(m[2])}) -> {ty(m[1])}>;')
out.append('extern "C" {')
for short in functions:
    name='TRITONSERVER_'+short
    m=re.search(r'TRITONSERVER_DECLSPEC\s+([^;]+?)\b'+name+r'\s*\((.*?)\);',s,flags=re.S)
    assert m,name
    out.append(f'pub fn {name}({args(m[2])}) -> {ty(m[1])};')
out.append('}')
for name in sorted(used):
    if re.search(r'^struct '+name+r';',s,re.M):
        out.insert(2, f'#[repr(C)] pub struct {name} {{ _private: [u8; 0] }}')
license=h[:h.index('#pragma once')]
arguments.output.write_text(license+'\n'+'\n'.join(out)+'\n')
