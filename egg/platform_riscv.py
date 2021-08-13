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
    if typ not in prefix:
        raise ValueError('Type "{}" not supported'.format(simd_ext))
    else:
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
    elif typ in ['i64', 'u64', 'f64']:
        sew = 64
    else:
        raise ValueError('Type "{}" not supported'.format(typ))
    n = sew // lmul
    return 'vbool{}_t'.format(n)

# -----------------------------------------------------------------------------
# Implementation of mandatory functions for this module

def emulate_fp16(simd_ext):
    return False

def get_simd_exts():
    return ['rvv1', 'rvv2', 'rvv4', 'rvv8']

def get_prev_simd_ext(simd_ext):
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    else:
        'cpu'

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
    ret = '''#include <nsimd/cpu/cpu/{}.h>
            '''.format(func)
    return ret

# -----------------------------------------------------------------------------
# Generic simple functions used by get_impl

def simple_op1(op, simd_ext, typ, add_suffix = False):
    suffix = 'u' if typ[0] == 'u' and add_suffix else ''
    return 'return v{op}({in0}, {vlmax});'.format(op=op, **fmtspec)

def simple_op2(op, simd_ext, typ, add_suffix = False):
    suffix = 'u' if typ[0] == 'u' and add_suffix else ''
    return 'return v{op}({in0}, {in1}, {vlmax});'.format(op=op, **fmtspec)

def simple_opf3(op, simd_ext, typ, add_suffix = False):
    suffix = 'u' if typ[0] == 'u' and add_suffix else ''
    if typ not in ['f16', 'f32', 'f64']:
        return 'return v{op}{suffix}({in0}, {in1}, {vlmax});'.format(op=op, **fmtspec)
    else:
        return 'return vf{op}{suffix}({in0}, {in1}, {vlmax});'.format(op=op, **fmtspec)

# -----------------------------------------------------------------------------
# Compariosn Functions

def cmp2(op, simd_ext, typ):
    suffix = 'u' if typ[0] == 'u' else ''
    if typ not in ['f16', 'f32', 'f64']:
        return 'return vms{op}{suffix}({in0}, {in1}, {vlmax});'.format(op=op, **fmtspec)
    else:
        return 'return vmf{op}{suffix}({in0}, {in1}, {vlmax});'.format(op=op, **fmtspec)

# -----------------------------------------------------------------------------
# Abs Functions

def abs1(simd_ext, typ):
    if typ in ['f16', 'f32', 'f64']:
        return 'return vfabs({in0}, {vlmax});'.format(**fmtspec)
    else:
        pass

# -----------------------------------------------------------------------------
# Length

def len1(simd_ext):
        return 'return (int)vsetvl_e{typnbits}_m{lmul}({vlmax});'. \
               format(**fmtspec)

# -----------------------------------------------------------------------------
# Square-root Functions

def sqrt1(simd_ext, typ):
    if typ in ['f16', 'f32', 'f64']:
        return 'return vfsqrt({in0}, {vlmax});'.format(**fmtspec)
    else:
        pass
        #return emulate_op1('abs', simd_ext, typ)

# -----------------------------------------------------------------------------
# Binary operator
def binop2(op, simd_ext2, from_typ):
    pass

# -----------------------------------------------------------------------------
# Logical operator

def lop2(op, simd_ext2, from_typ):
    pass

# -----------------------------------------------------------------------------

def get_impl(opts, func, simd_ext, from_typ, to_typ):
    global fmtspec

    fmtspec = {
        'simd_ext': simd_ext,
        'typ': from_typ,
        'styp': get_type(opts, simd_ext, from_typ, to_typ),
        'from_typ': from_typ,
        'to_typ': to_typ,
        'in0': common.in0,
        'in1': common.in1,
        'in2': common.in2,
        'in3': common.in3,
        'in4': common.in4,
        'in5': common.in5,
        'typnbits': from_typ[1:],
        'lmul': simd_ext[-1],
        'vlmax': 'vsetvlmax_e{}_m()'.format(from_typ[1:], simd_ext[-1]),
    }

    impls = {
        #'andnotb': 'binop2("andnotb", simd_ext2, from_typ)',
        'andb': 'binop2("andb", simd_ext2, from_typ)',
        'xorb': 'binop2("xorb", simd_ext2, from_typ)',
        'orb':  'binop2("orb", simd_ext2, from_typ)',
        'notb': 'binop2("not", simd_ext2, from_typ)',
        #'andnotl': 'lop2(opts, "andnotl", simd_ext2, from_typ)',
        'andl': 'lop2(opts, "andl", simd_ext2, from_typ)',
        'xorl': 'lop2(opts, "xorl", simd_ext2, from_typ)',
        'orl':  'lop2(opts, "orl", simd_ext2, from_typ)',
        'notl': 'lop2(opts, "notl", simd_ext2, from_typ)',

        'add': 'simple_op2("add", simd_ext, from_typ)',
        'sub': 'simple_op2("sub", simd_ext, from_typ)',
        'mul': 'simple_op2("mul", simd_ext, from_typ, True)',
        'div': 'simple_opf3("div", simd_ext, from_typ, True)',
        'sqrt': 'sqrt1(simd_ext, from_typ)',
        'len': 'len1(simd_ext)',
        'eq': 'cmp2("eq", simd_ext, from_typ)',
        'lt': 'cmp2("lt", simd_ext, from_typ)',
        'le': 'cmp2("le", simd_ext, from_typ)',
        'gt': 'cmp2("gt", simd_ext, from_typ)',
        'ge': 'cmp2("ge", simd_ext, from_typ)',
        'ne': 'cmp2("ne", simd_ext, from_typ)',
        'min': 'simple_opf3("min", simd_ext, from_typ, True)',
        'max': 'simple_opf3("max", simd_ext, from_typ, True)',
        'abs': 'simple_op1("abs", simd_ext, from_typ)',
    }
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if not from_typ in common.types:
        raise ValueError('Unknown type "{}"'.format(from_typ))
    if not func in impls:
        return common.NOT_IMPLEMENTED
    else:
        return eval(impls[func])

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Function prefixes

def pre(simd_ext):
    return 'rvv{}_'.format(simd_ext[-1])