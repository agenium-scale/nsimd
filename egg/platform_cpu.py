# Copyright (c) 2019 Agenium Scale
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

# This file gives the implementation of platform CPU, i.e. scalar emulation.
# Reading this file is straightforward. For each function, e.g. the addition,
# code looks like:
#
#     return 'return {} + {};'.format(common.in0, common.in1)
#
# with an 'if' before to handle the FP16 special case.

import common

# -----------------------------------------------------------------------------
# Implementation of mandatory functions for this module

def get_simd_exts():
    return ['cpu']

def get_simd_strings(simd_ext):
    if simd_ext == 'cpu':
        return ['cpu']
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def emulate_fp16(simd_ext):
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    return True

def get_type(simd_ext, typ):
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if typ in common.types:
        if typ == 'f16':
            return 'struct { f32 f; }'
        return '{}'.format(typ)
    else:
        raise ValueError('Unknown type "{}"'.format(typ))

def get_logical_type(simd_ext, typ):
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if typ in common.types:
        if typ == 'f16':
            return 'struct { u32 u; }'
        return '{}'.format(common.bitfield_type[typ])
    else:
        raise ValueError('Unknown type "{}"'.format(typ))

def get_nb_registers(simd_ext):
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    return '1'

def has_compatible_SoA_types(simd_ext):
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    return False

def get_additional_include(func, platform, simd_ext):
    if func in ['sqrt', 'ceil', 'floor', 'trunc']:
        return '''#if NSIMD_CXX > 0
                    #include <cmath>
                  #else
                    #include <math.h>
                  #endif'''
    return ''

# -----------------------------------------------------------------------------
# Returns C code for func

fmtspec = {}

def op2(op, typ):
    if typ == 'f16':
        return '''nsimd_cpu_vf16 ret;
                  ret.f = {}.f {} {}.f;
                  return ret;'''.format(common.in0, op, common.in1)
    return 'return ({})({} {} {});'.format(typ, common.in0, op, common.in1)

def lop2(op, typ):
    if typ == 'f16':
        return '''nsimd_cpu_vlf16 ret;
                  ret.u = {}.u {} {}.u;
                  return ret;'''.format(common.in0, op, common.in1)
    return 'return {} {} {};'.format(common.in0, op, common.in1)

def bitwise2(op, typ):
    if typ in common.utypes:
        return op2(op, typ)
    elif typ == 'f16':
        return '''union {{ u32 u; f32 f; }} tmp0, tmp1;
                  nsimd_cpu_vf16 ret;
                  tmp0.f = {in0}.f;
                  tmp1.f = {in1}.f;
                  tmp0.u = (u32)(tmp0.u {op} tmp1.u);
                  ret.f = tmp0.f;
                  return ret;'''.format(in0 = common.in0,
                                        in1 = common.in1,
                                        op = op)
    else:
        return '''union {{ {T} f; {uT} u; }} tmp0, tmp1;
                  tmp0.f = {in0};
                  tmp1.f = {in1};
                  tmp0.u = ({uT})(tmp0.u {op} tmp1.u);
                  return tmp0.f;'''.format(T = typ,
                                           uT = common.bitfield_type[typ],
                                           in0 = common.in0,
                                           in1 = common.in1,
                                           op = op)

def andnot2(typ):
    if typ == 'f16':
        return '''union {{ f32 f; u32 u; }} tmp0, tmp1;
                  nsimd_cpu_vf16 ret;
                  tmp0.f = {in0}.f;
                  tmp1.f = {in1}.f;
                  tmp0.u = tmp0.u & (~tmp1.u);
                  ret.f = tmp0.f;
                  return ret;'''.format(in0=common.in0, in1=common.in1)
    if typ in ['f32', 'f64', 'i8', 'i16', 'i32', 'i64']:
        return '''union {{ {T} f; {uT} u; }} tmp0, tmp1;
                  tmp0.f = {in0};
                  tmp1.f = {in1};
                  tmp0.u = ({uT})(tmp0.u & (~tmp1.u));
                  return tmp0.f;'''.format(T=typ, uT=common.bitfield_type[typ],
                                           in0=common.in0, in1=common.in1)
    return 'return ({})({} & (~{}));'.format(typ, common.in0, common.in1)

def landnot2(typ):
    if typ == 'f16':
        return '''nsimd_cpu_vlf16 ret;
                  ret.u = {in0}.u & (~{in1}.u);
                  return ret;'''.format(in0=common.in0, in1=common.in1)
    else:
        return 'return ({})({} & (~{}));'.format(common.bitfield_type[typ],
                                                 common.in0, common.in1)

def lnot1(typ):
    if typ == 'f16':
        return '''nsimd_cpu_vlf16 ret;
                  ret.u = ~{}.u;
                  return ret;'''.format(common.in0)
    return 'return ({})(~{});'.format(common.bitfield_type[typ], common.in0)

def not1(typ):
    if typ in common.utypes:
        return lnot1(typ)
    elif typ == 'f16':
        return '''union {{ u32 u; f32 f; }} tmp0;
                  nsimd_cpu_vf16 ret;
                  tmp0.f = {in0}.f;
                  tmp0.u = ~tmp0.u;
                  ret.f = tmp0.f;
                  return ret;'''.format(in0 = common.in0)
    else:
        return '''union {{ {T} f; {uT} u; }} tmp0;
                  tmp0.f = {in0};
                  tmp0.u = ({uT})(~tmp0.u);
                  return tmp0.f;'''.format(T = typ,
                                           uT = common.bitfield_type[typ],
                                           in0 = common.in0)

def minmax2(minmax, typ):
    op = '<' if minmax == 'min' else '>'
    if typ == 'f16':
        return 'return ({in0}.f {op} {in1}.f ? {in0} : {in1});'. \
               format(op=op, **fmtspec)
    return 'return ({in0} {op} {in1} ? {in0} : {in1});'. \
           format(op=op, **fmtspec)

def libm_op1(func, typ, until_cpp11 = False, c89_code = ''):
    cxx_version = '> 0' if not until_cpp11 else '>= 2011'
    comment = '''/* {func} is not available in C89 but is given by POSIX 2001
                    and C99. But we do not want to pollute the user includes
                    and POSIX value if set so we play dirty. */'''
    if typ == 'f16':
        if c89_code != '':
            c89_code = '{}\n{}'.format(comment, c89_code)
        else:
            c89_code = '{comment}\nret.f = (f32){func}((f64){in0}.f);'. \
                       format(comment=comment, func=func, **fmtspec);
        return '''  nsimd_cpu_vf16 ret;
                  #if defined(NSIMD_IS_MSVC) && _MSC_VER <= 1800 /* VS 2012 */
                    {c89_code}
                  #else
                    #if NSIMD_CXX {cxx_version}
                      ret.f = std::{func}({in0}.f);
                    #elif NSIMD_C >= 1999 || _POSIX_C_SOURCE >= 200112L
                      ret.f = {func}f({in0}.f);
                    #else
                      {c89_code}
                    #endif
                  #endif
                    return ret;'''. \
                    format(comment=comment, func=func, cxx_version=cxx_version,
                           c89_code=c89_code, **fmtspec)
    if typ == 'f32':
        if c89_code != '':
            c89_code = '{}\n{}'.format(comment, c89_code)
        else:
            c89_code = '{comment}\nreturn (f32){func}((f64){in0});'. \
                       format(comment=comment, func=func, **fmtspec);
        return '''#if defined(NSIMD_IS_MSVC) && _MSC_VER <= 1800 /* VS 2012 */
                    {c89_code}
                  #else
                    #if NSIMD_CXX {cxx_version}
                      return std::{func}({in0});
                    #elif NSIMD_C >= 1999 || _POSIX_C_SOURCE >= 200112L
                      return {func}f({in0});
                    #else
                      {c89_code}
                    #endif
                  #endif'''.format(cxx_version=cxx_version, c89_code=c89_code,
                                   func=func, **fmtspec)
    elif typ == 'f64':
        if c89_code != '':
            c89_code = '{}\n{}'.format(comment, c89_code)
        else:
            c89_code = '{comment}\nreturn {func}({in0});'. \
                       format(comment=comment, func=func, **fmtspec);
        return '''#if NSIMD_CXX {cxx_version}
                    return std::{func}({in0});
                  #else
                    {c89_code}
                  #endif'''.format(cxx_version=cxx_version, func=func,
                                   c89_code=c89_code, **fmtspec)

def sqrt1(typ):
    return libm_op1('sqrt', typ)

def ceil1(typ):
    if typ in ['f16', 'f32', 'f64']:
        return libm_op1('ceil', typ)
    return 'return {in0};'.format(**fmtspec)

def floor1(typ):
    if typ in ['f16', 'f32', 'f64']:
        return libm_op1('floor', typ)
    return 'return {in0};'.format(**fmtspec)

def trunc1(typ):
    if typ in ['f16', 'f32', 'f64']:
        if typ == 'f16':
            code = '''ret.f = {in0}.f > 0 ? nsimd_floor_cpu_f32({in0}.f)
                                          : nsimd_ceil_cpu_f32({in0}.f);'''. \
                                          format(**fmtspec)
        elif typ == 'f32':
            code = '''return {in0} > 0 ? nsimd_floor_cpu_f32({in0})
                                       : nsimd_ceil_cpu_f32({in0});'''. \
                                       format(**fmtspec)
        else:
            code = '''return {in0} > 0 ? nsimd_floor_cpu_f64({in0})
                                       : nsimd_ceil_cpu_f64({in0});'''. \
                                       format(**fmtspec)
        return libm_op1('trunc', typ, True, code)
    return 'return {in0};'.format(**fmtspec)

def round_to_even1(typ):
    code = \
    '''{T} fl = nsimd_floor_cpu_{T}({in0}{member});
       {T} ce = nsimd_ceil_cpu_{T}({in0}{member});
       {T} fl_p_half = fl + 0.5{suffix};
       if (fl == {in0}{member}) {{
         {ret}{in0}{member};
       }}
       if ({in0}{member} == fl_p_half) {{
         {T} flo2 = fl * 0.5{suffix};
         if (nsimd_floor_cpu_{T}(flo2) == flo2) {{
           {ret}fl;
         }} else {{
           {ret}ce;
         }}
       }} else if ({in0}{member} > fl_p_half) {{
         {ret}ce;
       }} else {{
         {ret}fl;
       }}'''.format(T = 'f32' if typ in ['f16', 'f32'] else 'f64',
                    suffix = 'f' if typ in ['f16', 'f32'] else '',
                    member = '.f' if typ == 'f16' else '',
                    ret = 'ret.f = ' if typ == 'f16' else 'return ',
                    **fmtspec)
    if typ in ['f16', 'f32', 'f64']:
        return libm_op1('nearbyint', typ, True, code)
    return 'return {in0};'.format(**fmtspec)

def bitwise1_param(op, typ):
    if typ in common.utypes:
        return 'return ({T})({in0} {op} {param0});' \
               .format(T = typ, in0 = common.in0, op = op,
                       param0 = common.in1)
    elif typ == 'f16':
        return '''nsimd_cpu_vf16 ret;
                  union {{ u32 u; f32 f; }} tmp0;
                  tmp0.f = {in0}.f;
                  tmp0.u = (u32)(tmp0.u {op} {param0});
                  ret.f = tmp0.f;
                  return ret;'''.format(in0 = common.in0,
                                        op = op,
                                        param0 = common.in1)
    else:
        return '''union {{ {T} f; {uT} u; }} tmp0;
                  tmp0.f = {in0};
                  tmp0.u = ({uT})(tmp0.u {op} {param0});
                  return tmp0.f;'''.format(T = typ,
                                           uT = common.bitfield_type[typ],
                                           in0 = common.in0,
                                           op = op,
                                           param0 = common.in1)

def cmp2(op, typ):
    if typ == 'f16':
        return '''nsimd_cpu_vlf16 ret;
                  ret.u = ({in0}.f {op} {in1}.f ? (u32)-1 : (u32)0);
                  return ret;
                  '''.format(in0=common.in0, in1=common.in1, op=op)
    return 'return ({in0} {op} {in1} ? ({uT})-1 : ({uT})0);'. \
           format(in0 = common.in0,
                  op = op,
                  in1 = common.in1,
                  uT = common.bitfield_type[typ])

def set1(typ):
    if typ == 'f16':
        return '''nsimd_cpu_vf16 ret;
                  ret.f = nsimd_f16_to_f32({});
                  return ret;'''.format(common.in0)
    return 'return {};'.format(common.in0)

def load(typ):
    if typ == 'f16':
        return '''nsimd_cpu_vf16 ret;
                  ret.f = nsimd_u16_to_f32(*(u16*){});
                  return ret;'''.format(common.in0)
    return 'return *{};'.format(common.in0)

def load_deg234(typ, deg):
    if typ == 'f16':
        code = '\n'.join( \
               ['ret.v{i}.f = nsimd_u16_to_f32(*((u16 *){in0} + {i}));'. \
               format(i=i, **fmtspec) for i in range(0, deg)])
        return '''nsimd_{simd_ext}_vf16x{deg} ret;
                  {code}
                  return ret;'''.format(deg=deg, code=code, **fmtspec)
    code = '\n'.join(['ret.v{i} = {in0}[{i}];'.format(i=i, **fmtspec) \
                      for i in range(0, deg)])
    return '''nsimd_{simd_ext}_v{typ}x{deg} ret;
              {code}
              return ret;'''.format(deg=deg, code=code, **fmtspec)

def store_deg234(typ, deg):
    if typ == 'f16':
        return '\n'.join( \
               ['*((u16 *){in0} + {i}) = nsimd_f32_to_u16(a{ip1}.f);'. \
               format(i=i, ip1=i + 1, **fmtspec) for i in range(0, deg)])
    return '\n'.join(['{in0}[{i}] = a{ip1};'.format(i=i, ip1=i + 1, **fmtspec) \
                      for i in range(0, deg)])

def loadl(typ):
    if typ == 'f16':
        return '''nsimd_cpu_vlf16 ret;
                  ret.u = nsimd_u16_to_f32(*(u16*){}) == 0.0f ? (u32)0
                                                              : (u32)-1;
                  return ret;'''.format(common.in0)
    return 'return *{in0} == ({T})0 ? ({uT})0 : ({uT})-1;'. \
           format(in0=common.in0, T=typ, uT=common.bitfield_type[typ])

def store(typ):
    if typ == 'f16':
        return '*(u16*){} = nsimd_f32_to_u16({}.f);'. \
               format(common.in0, common.in1)
    return '*{} = {};'.format(common.in0, common.in1)

def storel(typ):
    if typ == 'f16':
        return '''*(u16*){} = {}.u ? nsimd_f32_to_u16(1.0f)
                                   : nsimd_f32_to_u16(0.0f);'''. \
               format(common.in0, common.in1)
    return '*{} = {} ? ({})1 : ({})0;'.format(common.in0, common.in1, typ, typ)

def if_else1(typ):
    if typ == 'f16':
        return 'return {}.u ? {} : {};'.format(common.in0, common.in1,
                                               common.in2)
    return 'return {} ? {} : {};'.format(common.in0, common.in1, common.in2)

def abs1(typ):
    if typ == 'f16':
        return '''nsimd_cpu_vf16 ret;
                  ret.f = {in0}.f >= 0.0f ? {in0}.f : -{in0}.f;
                  return ret;'''.format(in0=common.in0, typ=typ)
    if typ in ['u8', 'u16', 'u32', 'u64']:
        return 'return {};'.format(common.in0)
    return 'return ({typ})({in0} >= ({typ})0 ? {in0} : -{in0});'. \
           format(in0=common.in0, typ=typ)

def fma_fms(func, typ):
    op = '+' if func in ['fma', 'fnma'] else '-'
    neg = '-' if func in ['fnma', 'fnms'] else ''
    if typ == 'f16':
        return '''nsimd_cpu_vf16 ret;
                  ret.f = {neg}({in0}.f * {in1}.f) {op} {in2}.f;
                  ret.f = {neg}({in0}.f * {in1}.f) {op} {in2}.f;
                  return ret;'''.format(neg=neg, op=op, **fmtspec)
    return 'return ({typ})({neg}({in0} * {in1}) {op} {in2});'. \
           format(op=op, neg=neg, **fmtspec)

def all_any(typ):
    if typ == 'f16':
        return 'return {in0}.u != 0u;'.format(**fmtspec)
    return 'return {in0} != ({utyp})0;'.format(**fmtspec)

def reinterpret1(from_typ, to_typ):
    if from_typ == to_typ:
        return 'return {in0};'.format(**fmtspec)
    if from_typ == 'f16':
        if to_typ == 'u16':
            return 'return nsimd_f32_to_u16({in0}.f);'.format(**fmtspec)
        return '''union {{ u16 from; i16 to; }} buf;
                  buf.from = nsimd_f32_to_u16({in0}.f);
                  return buf.to;'''.format(**fmtspec)
    if to_typ == 'f16':
        if from_typ == 'u16':
            return '''nsimd_cpu_vf16 ret;
                      ret.f = nsimd_u16_to_f32({in0});
                      return ret;'''.format(**fmtspec)
        return '''union {{ i16 from; u16 to; }} buf;
                  nsimd_cpu_vf16 ret;
                  buf.from = {in0};
                  ret.f = nsimd_u16_to_f32(buf.to);
                  return ret;'''.format(**fmtspec)
    return '''union {{ {from_typ} from; {to_typ} to; }} buf;
              buf.from = {in0};
              return buf.to;'''.format(**fmtspec)

def reinterpretl1(from_typ, to_typ):
    if from_typ == to_typ:
        return 'return {in0};'.format(**fmtspec)
    if to_typ == 'f16':
        return '''nsimd_cpu_vlf16 ret;
                  ret.u = {in0} ? -1u : 0u;
                  return ret;'''.format(**fmtspec)
    if from_typ == 'f16':
        return 'return {in0}.u ? (u16)-1 : (u16)0;'.format(**fmtspec)
    return 'return {in0};'.format(**fmtspec)

def convert1(from_typ, to_typ):
    if to_typ == from_typ:
        return 'return {in0};'.format(**fmtspec)
    if to_typ == 'f16':
        return '''nsimd_cpu_vf16 ret;
                  ret.f = (f32){in0};
                  return ret;'''.format(**fmtspec)
    if from_typ == 'f16':
        return 'return ({to_typ}){in0}.f;'.format(**fmtspec)
    return 'return ({to_typ}){in0};'.format(**fmtspec)

def rec_rec11(typ):
    one = '1.0f' if typ in ['f16', 'f32'] else '1.0'
    if typ == 'f16':
        return '''nsimd_cpu_vf16 ret;
                  ret.f = {one} / {in0}.f;
                  return ret;'''.format(one=one, **fmtspec)
    return 'return {one} / {in0};'.format(one=one, **fmtspec)

def rsqrt11(typ):
    one = '1.0f' if typ in ['f16', 'f32'] else '1.0'
    if typ == 'f16':
        return '''nsimd_cpu_vf16 ret;
                  ret.f = {one} / nsimd_sqrt_{simd_ext}_f32({in0}.f);
                  return ret;'''.format(one=one, **fmtspec)
    return 'return {one} / nsimd_sqrt_{simd_ext}_{typ}({in0});'. \
           format(one=one, **fmtspec)

def neg1(typ):
    if typ == 'f16':
        return '''nsimd_cpu_vf16 ret;
                  ret.f = -{in0}.f;
                  return ret;'''.format(**fmtspec)
    return 'return ({typ})(-{in0});'.format(**fmtspec)

def nbtrue1(typ):
    return 'return {in0}{member} ? 1 : 0;'. \
           format(member='.u' if typ == 'f16' else '', **fmtspec)

def reverse1(typ):
    return 'return {in0};'.format(**fmtspec)

def addv(typ):
    if typ == 'f16':
        return '''\
                return nsimd_f32_to_f16({in0}.f);
                '''.format(**fmtspec);
    else:
        return 'return {in0};' .format(**fmtspec)

def get_impl(func, simd_ext, from_typ, to_typ=''):

    global fmtspec
    fmtspec = {
      'simd_ext': simd_ext,
      'typ': from_typ,
      'from_typ': from_typ,
      'to_typ': to_typ,
      'utyp': common.bitfield_type[from_typ],
      'in0': common.in0,
      'in1': common.in1,
      'in2': common.in2,
      'typnbits': from_typ[1:]
    }

    impls = {
        'loada': load(from_typ),
        'load2a': load_deg234(from_typ, 2),
        'load3a': load_deg234(from_typ, 3),
        'load4a': load_deg234(from_typ, 4),
        'loadu': load(from_typ),
        'load2u': load_deg234(from_typ, 2),
        'load3u': load_deg234(from_typ, 3),
        'load4u': load_deg234(from_typ, 4),
        'storea': store(from_typ),
        'store2a': store_deg234(from_typ, 2),
        'store3a': store_deg234(from_typ, 3),
        'store4a': store_deg234(from_typ, 4),
        'storeu': store(from_typ),
        'store2u': store_deg234(from_typ, 2),
        'store3u': store_deg234(from_typ, 3),
        'store4u': store_deg234(from_typ, 4),
        'loadla': loadl(from_typ),
        'loadlu': loadl(from_typ),
        'storela': storel(from_typ),
        'storelu': storel(from_typ),
        'add': op2('+', from_typ),
        'mul': op2('*', from_typ),
        'div': op2('/', from_typ),
        'sub': op2('-', from_typ),
        'orb': bitwise2('|', from_typ),
        'orl': lop2('|', from_typ),
        'andb': bitwise2('&', from_typ),
        'andnotb': andnot2(from_typ),
        'andnotl': landnot2(from_typ),
        'andl': lop2('&', from_typ),
        'xorb': bitwise2('^', from_typ),
        'xorl': lop2('^', from_typ),
        'min': minmax2('min', from_typ),
        'max': minmax2('max', from_typ),
        'notb': not1(from_typ),
        'notl': lnot1(from_typ),
        'sqrt': sqrt1(from_typ),
        'set1': set1(from_typ),
        'shr': bitwise1_param('>>', from_typ),
        'shl': bitwise1_param('<<', from_typ),
        'eq': cmp2('==', from_typ),
        'ne': cmp2('!=', from_typ),
        'gt': cmp2('>', from_typ),
        'ge': cmp2('>=', from_typ),
        'lt': cmp2('<', from_typ),
        'le': cmp2('<=', from_typ),
        'len': 'return 1;',
        'if_else1': if_else1(from_typ),
        'abs': abs1(from_typ),
        'fma': fma_fms('fma', from_typ),
        'fnma': fma_fms('fnma', from_typ),
        'fms': fma_fms('fms', from_typ),
        'fnms': fma_fms('fnms', from_typ),
        'ceil': ceil1(from_typ),
        'floor': floor1(from_typ),
        'trunc': trunc1(from_typ),
        'round_to_even': round_to_even1(from_typ),
        'all': all_any(from_typ),
        'any': all_any(from_typ),
        'reinterpret': reinterpret1(from_typ, to_typ),
        'reinterpretl': reinterpretl1(from_typ, to_typ),
        'cvt': convert1(from_typ, to_typ),
        'rec11': rec_rec11(from_typ),
        'rsqrt11': rsqrt11(from_typ),
        'rec': rec_rec11(from_typ),
        'neg': neg1(from_typ),
        'nbtrue': nbtrue1(from_typ),
        'reverse': reverse1(from_typ),
        'addv': addv(from_typ)
    }
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if not from_typ in common.types:
        raise ValueError('Unknown from_type "{}"'.format(from_typ))
    if not func in impls:
        return common.NOT_IMPLEMENTED
    return impls[func]
