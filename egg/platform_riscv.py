# Copyright (c) 2021 Agenium Scale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file gives the implementation for the RISC-V platform.
# This script tries to be as readable as possible. It implements VMX and VSX.

# Documentation found from:
# https://github.com/riscv/rvv-intrinsic-doc
# https://github.com/riscv/riscv-v-spec/blob/master/v-spec.adoc
# https://github.com/riscv/rvv-intrinsic-doc
# https://llvm.org/devmtg/2019-04/slides/TechTalk-Kruppe-Espasa-RISC-V_Vectors_and_LLVM.pdf
# https://riscv.org/wp-content/uploads/2019/06/17.40-Vector_RISCV-20190611-Vectors.pdf

import common

fmtspec = {}
sew = 1
lmul = 8

# -----------------------------------------------------------------------------
# Helpers

def get_lmul(simd_ext):
    return simd_ext[-1]

# Returns the riscv type corresponding to the nsimd type
def native_type(typ, simd_ext):
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    prefix = {
        'u8': 'uint8',
        'u16': 'uint16',
        'u32': 'uint32',
        'u64': 'uint64',
        'i8': 'int8',
        'i16': 'int16',
        'i32': 'int32',
        'i64': 'int64',
        'f16': 'float16',
        'f32': 'float32',
        'f64': 'float64'
    }
    return 'v{}m{}_t'.format(prefix[typ], simd_ext[-1])

# Returns the logical rrv type corresponding to the nsimd type
def native_typel(typ, simd_ext):
    # n = SEW / LMUL
    lmul = int(get_lmul(simd_ext))
    sew = 1
    if typ in ['i8', 'u8']:
        sew = 8
    elif typ in ['i16', 'u16', 'f16']:
        nsew = 16
    elif typ in ['i32', 'u32', 'f32']:
        sew = 32
    elif typ in ['f64', 'i64', 'f64']:
        sew = 64
    else:
        raise ValueError('Type "{}" not supported'.format(typ))
    n = sew / lmul
    return 'vbool{}_t'.format(n)

# -----------------------------------------------------------------------------
# Implementation of mandatory functions for this module

def emulate_fp16(simd_ext):
    return False

def get_simd_exts():
    return ['rvv1', 'rvv2', 'rvv4' 'rvv8']

def get_type(opts, simd_ext, typ, nsimd_typ):
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    else:
        struct = native_type(typ, simd_ext)
    return 'typedef {} {};'.format(struct, nsimd_typ)

def get_logical_type(opts, simd_ext, typ, nsimd_typ):
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    else:
        struct = native_typel(typ, simd_ext)
    return 'typedef {} {};'.format(struct, nsimd_typ)

def get_nb_registers(simd_ext):
    if simd_ext in get_simd_exts():
        return '64'
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def has_compatible_SoA_types(simd_ext):
    if simd_ext in get_simd_exts():
        return False
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_additional_include(func, platform, simd_ext):
    ret = ''
    return ret

# -----------------------------------------------------------------------------
