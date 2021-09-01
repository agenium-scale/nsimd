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

# Returns the logical rvv type corresponding to the nsimd type
def native_typel(typ, simd_ext):
    # n = SEW / LMUL
    lmul = int(get_lmul(simd_ext))
    sew = 1
    if typ in ['i8', 'u8']:
        sew = 8
    elif typ in ['i16', 'u16', 'f16']:
        sew = 16
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
    return 'return v{op}{suffix}({in0}, {vlmax});'.format(op=op, suffix=suffix, **fmtspec)

def simple_op2(op, simd_ext, typ, add_suffix = False):
    suffix = 'u' if typ[0] == 'u' and add_suffix else ''
    return 'return v{op}{suffix}({in0}, {in1}, {vlmax});'.format(op=op, suffix=suffix, **fmtspec)

def simple_opf3(op, simd_ext, typ, add_suffix = False):
    suffix = 'u' if typ[0] == 'u' and add_suffix else ''
    if typ not in ['f16', 'f32', 'f64']:
        return 'return v{op}{suffix}({in0}, {in1}, {vlmax});'.format(op=op, suffix=suffix, **fmtspec)
    else:
        return 'return vf{op}{suffix}({in0}, {in1}, {vlmax});'.format(op=op, suffix=suffix, **fmtspec)

# -----------------------------------------------------------------------------
# Set1

def set1(simd_ext, typ):
    pass 
    # TODO

# -----------------------------------------------------------------------------
# Compariosn Functions

def cmp2(op, simd_ext, typ):
    suffix = 'u' if typ[0] == 'u' else ''
    if typ not in ['f16', 'f32', 'f64']:
        return 'return vms{op}{suffix}({in0}, {in1}, {vlmax});'.format(op=op, suffix=suffix, **fmtspec)
    else:
        return 'return vmf{op}{suffix}({in0}, {in1}, {vlmax});'.format(op=op, suffix=suffix, **fmtspec)

# -----------------------------------------------------------------------------
# Negation

def neg1(simd_ext, typ):
    if typ in common.ftypes:
        return 'vneg({in0}, {vlmax});'.format(**fmtspec)
    return 'vneg({in0}, {vlmax});'.format(**fmtspec)
# -----------------------------------------------------------------------------
# Abs Functions

def abs1(simd_ext, typ):
    if typ in common.utypes:
        return 'return {in0};'.format(**fmtspec)
    if typ in ['f16', 'f32', 'f64']:
        return 'return vfabs({in0}, {vlmax});'.format(**fmtspec)
    elif typ in ['i8']:
        pass
        # TODO Emulate operation
    else:
        return 'return vfabs(vreinterpret_f{typnbits}m{lmul}({in0}), {vlmax});'.format(**fmtspec)

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
        # TODO for itypes and utypes

# -----------------------------------------------------------------------------
# Binary operator
def binop2(op, simd_ext2, from_typ):
    pass
    # TODO

# -----------------------------------------------------------------------------
# Logical operator

def lop2(op, simd_ext2, from_typ):
    if from_typ not in ['f16', 'f32', 'f64']:
        if op == 'not':
            return '''return vreinterpret_f{typnbits}m{lmul}(vnot(vreinterpret_i{typnbits}({in0}), {vlmax}));''' \
                .format(**fmtspec)
        else:
            return '''return vreinterpret_f{typnbits}m{lmul}(
                v{op}(vreinterpret_i{typnbits}m{lmul}({in0}), vreinterpret_i{typnbits}m{lmul}({in1}), {vlmax})
            );
            '''.format(op=op, **fmtspec)
    else:
        if op == 'not':
            return 'return vnot({in0}, {vlmax});'.format(op=op, **fmtspec)
        else:
            return 'return v{op}({in0}, {in1}, {vlmax});'.format(op=op, **fmtspec)

# -----------------------------------------------------------------------------
# Reinterpret

def reinterpret1(simd_ext, from_typ, to_typ):
    return 'vreinterpret_{to_typ}m{lmul}({in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Convert

def convert1(simd_ext, from_typ, to_typ):
    if (from_typ[1:] == to_typ[1:]) and (from_typ in common.ftypes or to_typ in common.ftypes):
        return 'return vfcvt_x({in0}, {vlmax});'.format(**fmtspec)        
    return 'return vneg({in0}, {vlmax});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Vector Single-Width Saturating Add and Subtract Functions

def add_sub_s(op, simd_ext, typ):
    suffix = 'u' if typ[0] == 'u' else ''
    if typ in common.ftypes:
        return 'return nsimd_{op}_{simd_ext}_{typ}({in0}, {in1});'. \
               format(op=op, **fmtspec)
    return 'return vs{op}{suffix}({in0}, {in1}, {vlmax});'.format(op=op, suffix=suffix, **fmtspec)

# -----------------------------------------------------------------------------
# Reciprocal

def rec_8_11(simd_ext, typ):
    if typ in common.ftypes:
        func_div = "vfdiv"
    elif typ in common.utypes:
        func_div = "vdivu"
    else:
        func_div= "vdiv"
    return 'return {func_div}(nsimd_set1_{simd_ext}_{typ}(({typ})1), {in0}, {vlmax});'.format(func_div=func_div,**fmtspec)

# -----------------------------------------------------------------------------
# Rsqrt

def rsqrt_8_11(simd_ext, typ):
    fun_rsqrt = "vfsqrt"
    if typ in common.ftypes:
        return 'return vfdiv(nsimd_set1_{simd_ext}_{typ}(({typ})1), {in0}, {vlmax});'.format(**fmtspec)
    elif typ in common.utypes:
        func_div = "vdivu"
    else:
        func_div= "vdiv"
    return 'return {func_div}(nsimd_set1_{simd_ext}_{typ}(({typ})1), nsimd_sqrt_{simd_ext}_{typ}({in0}), {vlmax});'.format(func_div=func_div,**fmtspec)

# -----------------------------------------------------------------------------
# FMA

def fma(op, simd_ext, typ):
    if typ in common.ftypes:
        ppcop = { 'fma': 'vfmadd', 'fms': 'vfmsub', 'fnms': 'vfnmsub',
                  'fnma': 'vfnmadd' }
    else:
        ppcop = { 'fma': 'vmadd', 'fms': 'vmsub', 'fnms': 'vnmsub',
                  'fnma': 'vnmadd' }
    return 'return {ppcop}({in0}, {in1}, {in2}, {vlmax});'. \
               format(ppcop=ppcop[op], **fmtspec)

# -----------------------------------------------------------------------------
# Shift

def shl_shr(op, simd_ext, typ):
    pass

def shra(simd_ext, typ):
    pass

# -----------------------------------------------------------------------------
# to_logical

def to_logical1(simd_ext, typ):
    if typ in common.iutypes:
        return '''return nsimd_ne_{simd_ext}_{typ}(
                           {in0}, nsimd_set1_{simd_ext}_{typ}(({typ})0));'''.format(**fmtspec)
    elif typ in ['f32', 'f64']:
        pass
        # TODO
    else:
        return '''nsimd_{simd_ext}_vlf16 ret;
                  ret.v0 = nsimd_to_logical_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_to_logical_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(**fmtspec)

# -----------------------------------------------------------------------------

def reverse1(simd_ext, typ):
    pass
    # TODO

# -----------------------------------------------------------------------------

def gather(simd_ext, typ):
    pass
    # TODO


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
        #'andb': 'binop2("andb", simd_ext2, from_typ)',
        #'xorb': 'binop2("xorb", simd_ext2, from_typ)',
        #'orb':  'binop2("orb", simd_ext2, from_typ)',
        #'notb': 'binop2("not", simd_ext2, from_typ)',
        #'andnotl': 'lop2("andnotl", simd_ext2, from_typ)',
        'andl': 'lop2("and", simd_ext, from_typ)',
        'xorl': 'lop2("xor", simd_ext, from_typ)',
        'orl':  'lop2("or",  simd_ext, from_typ)',
        'notl': 'lop2("not", simd_ext, from_typ)',

        'add': 'simple_op2("add", simd_ext, from_typ)',
        'sub': 'simple_op2("sub", simd_ext, from_typ)',
        'mul': 'simple_op2("mul", simd_ext, from_typ, True)',
        'div': 'simple_opf3("div", simd_ext, from_typ, True)',
        'adds': 'add_sub_s("add", simd_ext, from_typ)',
        'subs': 'add_sub_s("sub", simd_ext, from_typ)',
        'sqrt': 'sqrt1(simd_ext, from_typ)',
        'len': 'len1(simd_ext)',
        #'shl': 'shl_shr("shl", simd_ext, from_typ)',
        #'shr': 'shl_shr("shr", simd_ext, from_typ)',
        #'shra': 'shra(simd_ext, from_typ)',
        'eq': 'cmp2("eq", simd_ext, from_typ)',
        'lt': 'cmp2("lt", simd_ext, from_typ)',
        'le': 'cmp2("le", simd_ext, from_typ)',
        'gt': 'cmp2("gt", simd_ext, from_typ)',
        'ge': 'cmp2("ge", simd_ext, from_typ)',
        'ne': 'cmp2("ne", simd_ext, from_typ)',
        'min': 'simple_opf3("min", simd_ext, from_typ, True)',
        'max': 'simple_opf3("max", simd_ext, from_typ, True)',
        'abs': 'simple_op1("abs", simd_ext, from_typ)',
        'reinterpret': 'reinterpret1(simd_ext, from_typ, to_typ)',
        'neg': 'neg1(simd_ext, from_typ)',
        'rec': 'rec_8_11(simd_ext, from_typ)',
        'rec8': 'rec_8_11(simd_ext, from_typ)',
        'rec11': 'rec_8_11(simd_ext, from_typ)',
        'rsqrt8': 'rsqrt_8_11(simd_ext, from_typ)',
        'rsqrt11': 'rsqrt_8_11(simd_ext, from_typ)',
        'fma': 'fma("fma", simd_ext, from_typ)',
        'fnma': 'fma("fnma", simd_ext, from_typ)',
        'fms': 'fma("fms", simd_ext, from_typ)',
        'fnms': 'fma("fnms", simd_ext, from_typ)',
        #'ceil': round1("cei", simd_ext, from_typ),
        #'floor': round1("floor", simd_ext, from_typ),'nbtrue': 'nbtrue1(simd_ext, from_typ)',
        #'reverse': 'reverse1(simd_ext, from_typ)',
        #'addv': 'addv(simd_ext, from_typ)',
        #'upcvt': 'upcvt1(simd_ext, from_typ, to_typ)',
        #'downcvt': 'downcvt1(simd_ext, from_typ, to_typ)',
        #'iota': 'iota(simd_ext, from_typ)',
        #'to_logical': 'to_logical(simd_ext, from_typ)',
        #'mask_for_loop_tail': 'mask_for_loop_tail(simd_ext, from_typ)',
        #'masko_loadu1': 'maskoz_load("o", simd_ext, from_typ)',
        #'maskz_loadu1': 'maskoz_load("z", simd_ext, from_typ)',
        #'masko_loada1': 'maskoz_load("o", simd_ext, from_typ)',
        #'maskz_loada1': 'maskoz_load("z", simd_ext, from_typ)',
        #'mask_storea1': 'mask_store(simd_ext, from_typ)',
        #'mask_storeu1': 'mask_store(simd_ext, from_typ)',
        #'gather': 'gather(simd_ext, from_typ)',
        #'scatter': 'scatter(simd_ext, from_typ)',
        #'gather_linear': 'gather_linear(simd_ext, from_typ)',
        #'scatter_linear': 'scatter_linear(simd_ext, from_typ)',
        #'to_mask': 'to_mask(simd_ext, from_typ)',
        #'ziplo': 'zip("ziplo", simd_ext, from_typ)',
        #'ziphi': 'zip("ziphi", simd_ext, from_typ)',
        #'zip': 'zip_unzip_basic("zip", simd_ext, from_typ)',
        #'unzip': 'zip_unzip_basic("unzip", simd_ext, from_typ)',
        #'unziplo': 'unzip("unziplo", simd_ext, from_typ)',
        #'unziphi': 'unzip("unziphi", simd_ext, from_typ)'
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