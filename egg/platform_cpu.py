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
# Emulation parameters
#
# When emulating, we need to choose a vector length to fit the philosophy of
# SIMD. By default we choose 64 bits. It must be a multiple of 64 bits.

NBITS = common.CPU_NBITS

def get_nb_el(typ):
    return NBITS // int(typ[1:])

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
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    typ2 = typ if typ != 'f16' else 'f32'
    members = '\n'.join('{} v{};'.format(typ2, i) \
                        for i in range(0, get_nb_el(typ)))
    return 'struct {{ {} }}'.format(members)

def get_logical_type(simd_ext, typ):
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    members = '\n'.join('unsigned int v{};'.format(i) \
                        for i in range(0, get_nb_el(typ)))
    return 'struct {{ {} }}'.format(members)

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
    elif func in ['']:
        return '''#include <nsimd/cpu/cpu/reinterpret.h>
                  '''
    return ''

# -----------------------------------------------------------------------------
# Returns C code for func

fmtspec = {}

def repeat_stmt(fmt, typ):
    return '\n'.join(fmt.format(i=i) for i in range(0, get_nb_el(typ)))

# -----------------------------------------------------------------------------

def func_body(fmt, typ2, logical = False):
    return '''nsimd_cpu_v{logical}{typ2} ret;
              {content}
              return ret;'''.format(logical='l' if logical else '', typ2=typ2,
                                    content=repeat_stmt(fmt, typ2), **fmtspec)

# -----------------------------------------------------------------------------

def op2(op, typ):
    return func_body('ret.v{{i}} = {cast}({in0}.v{{i}} {op} {in1}.v{{i}});'. \
                     format(cast='({})'.format(typ) if typ in common.iutypes \
                            else '', op=op, **fmtspec), typ)

# -----------------------------------------------------------------------------

def lop2(op, typ):
    return func_body('ret.v{{i}} = {in0}.v{{i}} {op} {in1}.v{{i}};'. \
                     format(op=op, **fmtspec), typ, True)

# -----------------------------------------------------------------------------

def bitwise2(op, typ):
    if typ in common.utypes:
        return op2(op, typ)
    utyp2 = 'u32' if typ == 'f16' else common.bitfield_type[typ]
    typ2 = 'f32' if typ == 'f16' else typ
    return '''nsimd_cpu_v{typ} ret;
              union {{ {utyp2} u; {typ2} f; }} buf0, buf1;
              {content}
              return ret;'''.format(content=repeat_stmt(
              '''buf0.f = {in0}.v{{i}};
                 buf1.f = {in1}.v{{i}};
                 buf0.u = ({utyp2})(buf0.u {op} buf1.u);
                 ret.v{{i}} = buf0.f;'''.format(utyp2=utyp2, op=op, **fmtspec),
                 typ), utyp2=utyp2, typ2=typ2, **fmtspec)

# -----------------------------------------------------------------------------

def andnot2(typ):
    if typ in common.utypes:
        return '''nsimd_cpu_v{typ} ret;
                  {content}
                  return ret;'''.format(content=repeat_stmt(
                  'ret.v{{i}} = ({typ})({in0}.v{{i}} & (~{in1}.v{{i}}));'. \
                  format(**fmtspec), typ), **fmtspec)
    utyp2 = 'u32' if typ == 'f16' else common.bitfield_type[typ]
    typ2 = 'f32' if typ == 'f16' else typ
    return '''nsimd_cpu_v{typ} ret;
              union {{ {utyp2} u; {typ2} f; }} buf0, buf1;
              {content}
              return ret;'''.format(content=repeat_stmt(
              '''buf0.f = {in0}.v{{i}};
                 buf1.f = {in1}.v{{i}};
                 buf0.u = ({utyp2})(buf0.u & (~buf1.u));
                 ret.v{{i}} = buf0.f;'''.format(utyp2=utyp2, **fmtspec), typ),
                 utyp2=utyp2, typ2=typ2, **fmtspec)

# -----------------------------------------------------------------------------

def landnot2(typ):
    return func_body('ret.v{{i}} = {in0}.v{{i}} & (~{in1}.v{{i}});'.\
                     format(**fmtspec), typ, True)

# -----------------------------------------------------------------------------

def lnot1(typ):
    return func_body('ret.v{{i}} = ~{in0}.v{{i}};'.\
                     format(**fmtspec), typ, True)

# -----------------------------------------------------------------------------

def not1(typ):
    if typ in common.utypes:
        return func_body('ret.v{{i}} = ({typ})(~{in0}.v{{i}});'. \
                         format(**fmtspec), typ)
    utyp2 = 'u32' if typ == 'f16' else common.bitfield_type[typ]
    typ2 = 'f32' if typ == 'f16' else typ
    return '''nsimd_cpu_v{typ} ret;
              union {{ {utyp2} u; {typ2} f; }} buf0;
              {content}
              return ret;'''.format(content=repeat_stmt(
              '''buf0.f = {in0}.v{{i}};
                 buf0.u = ({utyp2})(~buf0.u);
                 ret.v{{i}} = buf0.f;'''.format(utyp2=utyp2, **fmtspec), typ),
                 utyp2=utyp2, typ2=typ2, **fmtspec)

# -----------------------------------------------------------------------------

def minmax2(minmax, typ):
    op = '<' if minmax == 'min' else '>'
    return func_body('''ret.v{{i}} = {in0}.v{{i}} {op} {in1}.v{{i}} ?
                                     {in0}.v{{i}} : {in1}.v{{i}};'''. \
                                     format(op=op, **fmtspec), typ)

# -----------------------------------------------------------------------------

def libm_op1(func, typ, until_cpp11 = False, c89_code = ''):
    cxx_version = '> 0' if not until_cpp11 else '>= 2011'
    comment = \
    '''/* {func} is not available in C89 but is given by POSIX 2001 */
       /* and C99. But we do not want to pollute the user includes  */
       /* and POSIX value if set so we play dirty.                  */'''. \
       format(func=func)
    if c89_code != '':
        c89_code = repeat_stmt(c89_code, typ)
    if typ in ['f16', 'f32']:
        c99_code = repeat_stmt('ret.v{{i}} = {func}f({in0}.v{{i}});'. \
                               format(func=func, **fmtspec), typ)
        if c89_code == '':
            c89_code = repeat_stmt(
                       'ret.v{{i}} = (f32){func}((f64){in0}.v{{i}});'. \
                       format(func=func, **fmtspec), typ)
        return \
        '''  {comment}
             nsimd_cpu_v{typ} ret;
           #if defined(NSIMD_IS_MSVC) && _MSC_VER <= 1800 /* VS 2012 */
             {c89_code}
           #else
             #if NSIMD_CXX {cxx_version} || NSIMD_C >= 1999 || \
                 _POSIX_C_SOURCE >= 200112L
               {c99_code}
             #else
               {c89_code}
             #endif
           #endif
             return ret;'''. \
             format(comment=comment, func=func, cxx_version=cxx_version,
                    c89_code=c89_code, c99_code=c99_code, **fmtspec)
    else:
        c99_code = repeat_stmt('ret.v{{i}} = {func}({in0}.v{{i}});'. \
                               format(func=func, **fmtspec), typ)
        if c89_code == '':
            return '''nsimd_cpu_vf64 ret;
                      {c99_code}
                      return ret;'''.format(c99_code=c99_code)
        return \
        '''  {comment}
             nsimd_cpu_vf64 ret;
           #if NSIMD_CXX {cxx_version} || NSIMD_C >= 1999 || \
               _POSIX_C_SOURCE >= 200112L
             {c99_code}
           #else
             {c89_code}
           #endif
           return ret;'''. \
           format(comment=comment, c89_code=c89_code, c99_code=c99_code,
                  cxx_version=cxx_version, **fmtspec)

# -----------------------------------------------------------------------------

def sqrt1(typ):
    return libm_op1('sqrt', typ)

# -----------------------------------------------------------------------------

def ceil1(typ):
    if typ in ['f16', 'f32', 'f64']:
        return libm_op1('ceil', typ)
    return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------

def floor1(typ):
    if typ in ['f16', 'f32', 'f64']:
        return libm_op1('floor', typ)
    return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------

def trunc1(typ):
    if typ == 'f16':
        c89_code = '''ret = {in0}.v{{i}} >= 0.0f
                                 ? nsimd_floor_cpu_{typ}({in0})
                                 : nsimd_ceil_cpu_{typ}({in0});'''. \
                                 format(**fmtspec)
        return libm_op1('trunc', typ, True, c89_code)
    elif typ in common.ftypes:
        c89_code = '''ret = {in0}.v{{i}} >= ({typ})0
                                 ? nsimd_floor_cpu_{typ}({in0})
                                 : nsimd_ceil_cpu_{typ}({in0});'''. \
                                 format(**fmtspec)
        return libm_op1('trunc', typ, True, c89_code)
    return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------

def round_to_even1(typ):
    if typ in common.iutypes:
        return 'return {in0};'.format(**fmtspec)
    stmt = '''{{{{
              {typ2} fl_p_half = fl.v{{i}} + 0.5{suffix};
              if (fl.v{{i}} == {in0}.v{{i}}) {{{{
                ret.v{{i}} = {in0}.v{{i}};
              }}}}
              if ({in0}.v{{i}} == fl_p_half) {{{{
                f64 flo2 = (f64)(fl.v{{i}} * 0.5{suffix});
                if (floor(flo2) == flo2) {{{{
                  ret.v{{i}} = fl.v{{i}};
                }}}} else {{{{
                  ret.v{{i}} = ce.v{{i}};
                }}}}
              }}}} else if ({in0}.v{{i}} > fl_p_half) {{{{
                ret.v{{i}} = ce.v{{i}};
              }}}} else {{{{
                ret.v{{i}} = fl.v{{i}};
              }}}}
              }}}}'''.format(typ2 = 'f32' if typ in ['f16', 'f32'] else 'f64',
                             suffix = 'f' if typ in ['f16', 'f32'] else '',
                             **fmtspec)
    return \
    '''nsimd_cpu_v{typ} fl = nsimd_floor_cpu_{typ}({in0});
       nsimd_cpu_v{typ} ce = nsimd_ceil_cpu_{typ}({in0});
       nsimd_cpu_v{typ} ret;
       '''.format(**fmtspec) + \
       repeat_stmt(stmt, typ) + '\n' + \
       'return ret;'

# -----------------------------------------------------------------------------

def bitwise1_param(op, typ):
    if typ in common.utypes:
        return func_body('ret.v{{i}} = ({typ})({in0}.v{{i}} {op} {in1});'. \
                         format(op=op, **fmtspec), typ)
    else:
        return '''nsimd_cpu_v{typ} ret;
                  union {{ {typ} i; {utyp} u; }} buf;
                  {content}
                  return ret;'''. \
                  format(content=repeat_stmt(
                  '''buf.i = {in0}.v{{i}};
                     buf.u = ({utyp})(buf.u {op} {in1});
                     ret.v{{i}} = buf.i;'''.format(op=op, **fmtspec), typ),
                     **fmtspec)

# -----------------------------------------------------------------------------

def cmp2(op, typ):
    return '''nsimd_cpu_vl{typ} ret;
              {content}
              return ret;'''.format(content=repeat_stmt(
              '''ret.v{{i}} = ({in0}.v{{i}} {op} {in1}.v{{i}}
                            ? (u32)-1 : (u32)0);'''. \
                            format(op=op, **fmtspec), typ), **fmtspec)

# -----------------------------------------------------------------------------

def set1(typ):
    if typ == 'f16':
        content = repeat_stmt('ret.v{{i}} = nsimd_f16_to_f32({in0});'. \
                              format(**fmtspec), typ)
    else:
        content = repeat_stmt('ret.v{{i}} = {in0};'.format(**fmtspec), typ)
    return '''nsimd_cpu_v{typ} ret;
              {content}
              return ret;'''.format(content=content, **fmtspec)

# -----------------------------------------------------------------------------

def load(typ):
    if typ == 'f16':
        content = repeat_stmt(
                  'ret.v{{i}} = nsimd_u16_to_f32(((u16 *){in0})[{{i}}]);'. \
                  format(**fmtspec), typ)
    else:
        content = repeat_stmt('ret.v{{i}} = {in0}[{{i}}];'.format(**fmtspec),
                  typ)
    return '''nsimd_cpu_v{typ} ret;
              {content}
              return ret;'''.format(content=content, **fmtspec)

# -----------------------------------------------------------------------------

def load_deg234(typ, deg):
    if typ == 'f16':
        buf = repeat_stmt(
              '''ret.v{{{{j}}}}.v{{i}} =
                     nsimd_u16_to_f32(
                       ((u16 *){in0})[{deg} * {{i}} + {{{{j}}}}]);'''. \
                       format(deg=deg, **fmtspec), typ)
    else:
        buf = repeat_stmt(
              'ret.v{{{{j}}}}.v{{i}} = {in0}[{deg} * {{i}} + {{{{j}}}}];'. \
              format(deg=deg, **fmtspec), typ)
    content = '\n'.join(buf.format(j=j) for j in range(0, deg))
    return '''nsimd_cpu_v{typ}x{deg} ret;
              {content}
              return ret;'''.format(deg=deg, content=content, **fmtspec)

# -----------------------------------------------------------------------------

def store_deg234(typ, deg):
    content = ''
    for i in range(0, get_nb_el(typ)):
        for j in range(1, deg + 1):
            arg = fmtspec['in{}'.format(j)]
            if typ == 'f16':
                content += \
                '''((u16 *){in0})[{deg} * {i} + {j}] =
                       nsimd_f32_to_u16({arg}.v{i});\n'''. \
                       format(deg=deg, i=i, j=j - 1, arg=arg, **fmtspec)
            else:
                content += \
                '{in0}[{deg} * {i} + {j}] = {arg}.v{i};\n'. \
                format(deg=deg, i=i, j=j - 1, arg=arg, **fmtspec)
    return content[:-1]

# -----------------------------------------------------------------------------

def loadl(typ):
    if typ == 'f16':
        content = repeat_stmt(
                  '''ret.v{{i}} = nsimd_u16_to_f32(
                                    ((u16 *){in0})[{{i}}]) == 0.0f
                                ? (u32)0 : (u32)-1;'''.format(**fmtspec), typ)
    else:
        content = repeat_stmt(
                  '''ret.v{{i}} = {in0}[{{i}}] == ({typ})0
                                ? (u32)0 : (u32)-1;'''. \
                                format(**fmtspec), typ)
    return '''nsimd_cpu_vl{typ} ret;
              {content}
              return ret;'''.format(content=content, **fmtspec)

# -----------------------------------------------------------------------------

def store(typ):
    if typ == 'f16':
        content = repeat_stmt(
                  '((u16*){in0})[{{i}}] = nsimd_f32_to_u16({in1}.v{{i}});'. \
                  format(**fmtspec), typ)
    else:
        content = repeat_stmt('{in0}[{{i}}] = {in1}.v{{i}};'. \
                              format(**fmtspec), typ)
    return content

# -----------------------------------------------------------------------------

def storel(typ):
    if typ == 'f16':
        content = repeat_stmt(
                  '''((u16*){in0})[{{i}}] = {in1}.v{{i}} == (u32)0
                                          ? nsimd_f32_to_u16(0.0f)
                                          : nsimd_f32_to_u16(1.0f);'''. \
                                          format(**fmtspec), typ)
    else:
        content = repeat_stmt('''{in0}[{{i}}] = {in1}.v{{i}} == (u32)0
                                              ? ({typ})0 : ({typ})1;'''. \
                                              format(**fmtspec), typ)
    return content

# -----------------------------------------------------------------------------

def if_else1(typ):
    return func_body('''ret.v{{i}} = {in0}.v{{i}} != (u32)0
                                   ? {in1}.v{{i}} : {in2}.v{{i}};'''. \
                                   format(**fmtspec), typ)

# -----------------------------------------------------------------------------

def abs1(typ):
    if typ in common.utypes:
        return func_body('ret.v{{i}} = {in0}.v{{i}};'.format(**fmtspec), typ)
    typ2 = 'f32' if typ == 'f16' else typ
    return func_body('''ret.v{{i}} = ({typ2})({in0}.v{{i}} < ({typ2})0
                                   ? -{in0}.v{{i}} : {in0}.v{{i}});'''. \
                                   format(typ2=typ2, **fmtspec), typ)

# -----------------------------------------------------------------------------

def fma_fms(func, typ):
    op = '+' if func in ['fma', 'fnma'] else '-'
    neg = '-' if func in ['fnma', 'fnms'] else ''
    typ2 = 'f32' if typ == 'f16' else typ
    return func_body(
           '''ret.v{{i}} = ({typ2})({neg}({in0}.v{{i}} * {in1}.v{{i}})
                         {op} {in2}.v{{i}});'''.format(op=op, neg=neg,
                         typ2=typ2, **fmtspec), typ)

# -----------------------------------------------------------------------------

def all_any(typ, func):
    op = '&&' if func == 'all' else '||'
    if get_nb_el(typ) == 1:
        cond = '{in0}.v0 == (u32)-1'.format(**fmtspec)
    else:
        cond = op.join('({in0}.v{i} == (u32)-1)'.format(i=i, **fmtspec) \
                       for i in range(0, get_nb_el(typ)))
    return '''if ({cond}) {{
                return -1;
              }} else {{
                return 0;
              }}'''.format(cond=cond)

# -----------------------------------------------------------------------------

def reinterpret1(from_typ, to_typ):
    if from_typ == to_typ:
        return func_body('ret.v{{i}} = {in0}.v{{i}};'.format(**fmtspec),
                         to_typ)
    return '''char buf[{len}];
              nsimd_storeu_cpu_{from_typ}(({from_typ} *)buf, {in0});
              return nsimd_loadu_cpu_{to_typ}(({to_typ} *)buf);'''. \
              format(len=NBITS // 8, **fmtspec)

# -----------------------------------------------------------------------------

def reinterpretl1(from_typ, to_typ):
    return func_body('ret.v{{i}} = {in0}.v{{i}};'.format(**fmtspec), to_typ,
                     True);

# -----------------------------------------------------------------------------

def convert1(from_typ, to_typ):
    if to_typ == from_typ:
        return func_body('ret.v{{i}} = {in0}.v{{i}};'.format(**fmtspec),
                         to_typ)
    typ2 = 'f32' if to_typ == 'f16' else to_typ
    return func_body('ret.v{{i}} = ({typ2}){in0}.v{{i}};'. \
                     format(typ2=typ2, **fmtspec), to_typ)

# -----------------------------------------------------------------------------

def rec_rec11(typ):
    one = '1.0f' if typ in ['f16', 'f32'] else '1.0'
    return func_body('ret.v{{i}} = {one} / {in0}.v{{i}};'. \
                     format(one=one, **fmtspec), typ)

# -----------------------------------------------------------------------------

def rsqrt11(typ):
    if typ == 'f64':
        return func_body('ret.v{{i}} = 1.0 / sqrt({in0}.v{{i}});'. \
                         format(**fmtspec), typ)
    else:
        return func_body(
               'ret.v{{i}} = (f32)(1.0 / sqrt((f64){in0}.v{{i}}));'. \
               format(**fmtspec), typ)

# -----------------------------------------------------------------------------

def neg1(typ):
    typ2 = 'f32' if typ == 'f16' else typ
    return func_body('ret.v{{i}} = ({typ2})(-{in0}.v{{i}});'. \
                     format(typ2=typ2, **fmtspec), typ)

# -----------------------------------------------------------------------------

def nbtrue1(typ):
    acc_code = repeat_stmt('acc += {in0}.v{{i}} == (u32)-1 ? 1 : 0;'. \
                           format(**fmtspec), typ)
    return '''int acc = 0;
              {acc_code}
              return acc;'''.format(acc_code=acc_code)

# -----------------------------------------------------------------------------

def reverse1(typ):
    n = get_nb_el(typ)
    content = '\n'.join('ret.v{i} = {in0}.v{j}'. \
                        format(i=i, j=n - i, **fmtspec) \
                        for i in range(0, n))
    return '''nsimd_cpu_v{typ} ret;
              {content}
              return ret;'''.format(content=content, **fmtspec)

# -----------------------------------------------------------------------------

def addv1(typ):
    content = '+'.join('{in0}.v{i}'.format(i=i, **fmtspec) \
                       for i in range(0, get_nb_el(typ)))
    if typ == 'f16':
        return 'return nsimd_f32_to_f16({});'.format(content)
    else:
        return 'return {};'.format(content)

# -----------------------------------------------------------------------------

def upcvt1(from_typ, to_typ):
    n = get_nb_el(to_typ)
    to_typ2 = 'f32' if to_typ == 'f16' else to_typ
    lower_half = '\n'.join('ret.v0.v{i} = ({to_typ2}){in0}.v{i};'. \
                           format(i=i, to_typ2=to_typ2, **fmtspec) \
                           for i in range(0, n))
    upper_half = '\n'.join('ret.v1.v{i} = ({to_typ2}){in0}.v{j};'. \
                           format(i=i, j=i + n, to_typ2=to_typ2, **fmtspec) \
                           for i in range(0, n))
    return '''nsimd_cpu_v{to_typ}x2 ret;
              {lower_half}
              {upper_half}
              return ret;'''.format(lower_half=lower_half,
                                    upper_half=upper_half, **fmtspec)

# -----------------------------------------------------------------------------

def downcvt2(from_typ, to_typ):
    n = get_nb_el(from_typ)
    to_typ2 = 'f32' if to_typ == 'f16' else to_typ
    lower_half = '\n'.join('ret.v{i} = ({to_typ2}){in0}.v{i};'. \
                           format(i=i, to_typ2=to_typ2, **fmtspec) \
                           for i in range(0, n))
    upper_half = '\n'.join('ret.v{j} = ({to_typ2}){in1}.v{i};'. \
                           format(i=i, j=i + n, to_typ2=to_typ2, **fmtspec) \
                           for i in range(0, n))
    return '''nsimd_cpu_v{to_typ} ret;
              {lower_half}
              {upper_half}
              return ret;'''.format(lower_half=lower_half,
                                    upper_half=upper_half, **fmtspec)

# -----------------------------------------------------------------------------

def len1(typ):
    return 'return {};'.format(get_nb_el(typ))

# -----------------------------------------------------------------------------

def to_logical1(typ):
    unsigned_to_logical = \
        'ret.v{{i}} = ({in0}.v{{i}} == ({utyp})0 ? (u32)0 : (u32)-1);'. \
        format(**fmtspec)
    if typ in common.utypes:
        return func_body(unsigned_to_logical, typ, True)
    else:
        unsigned_to_logical = \
            'ret.v{{i}} = (buf.v{{i}} == ({utyp})0 ? (u32)0 : (u32)-1);'. \
            format(**fmtspec)
        return '''nsimd_cpu_vl{typ} ret;
                  nsimd_cpu_vu{typnbits} buf;
                  buf = nsimd_reinterpret_cpu_u{typnbits}_{typ}({in0});
                  {unsigned_to_logical}
                  return ret;'''. \
                  format(unsigned_to_logical=repeat_stmt(unsigned_to_logical,
                                                         typ), **fmtspec)

# -----------------------------------------------------------------------------

def to_mask1(typ):
    logical_to_unsigned = \
        'ret.v{{i}} = ({in0}.v{{i}} ? ({utyp})-1 : ({utyp})0);'. \
        format(**fmtspec)
    if typ in common.utypes:
        return func_body(logical_to_unsigned, typ)
    elif typ == 'f16':
        return '''union {{ f32 f; u32 u; }} buf;
                  nsimd_cpu_vf16 ret;
                  {u32_to_f32}
                  return ret;'''. \
                  format(u32_to_f32=repeat_stmt(
                      'buf.u = {in0}.v{{i}}; ret.v{{i}} = buf.f;'. \
                      format(**fmtspec), 'f16'), **fmtspec)
    else:
        return '''nsimd_cpu_vu{typnbits} ret;
                  {logical_to_unsigned}
                  return nsimd_reinterpret_cpu_{typ}_u{typnbits}(ret);'''. \
                  format(logical_to_unsigned=repeat_stmt(logical_to_unsigned,
                                                         typ), **fmtspec)

# -----------------------------------------------------------------------------

def zip_half(func, typ):
    n = get_nb_el(typ)
    if typ in ['i64', 'u64', 'f64']:
      return '''(void)({in1});
                return {in0};'''.format(**fmtspec)
    else:
      if func == "ziplo":
        content = '\n'.join('ret.v{j1} = {in0}.v{i}; ret.v{j2} = {in1}.v{i};'. \
                            format(i=i, j1=i*2, j2=i*2+1, **fmtspec) \
                            for i in range(0, int(n/2)))
      else :
        content = '\n'.join('ret.v{j1} = {in0}.v{i}; ret.v{j2} = {in1}.v{i};'. \
                            format(i=i+int(n/2), j1=i*2, j2=i*2+1, **fmtspec) \
                            for i in range(0, int(n/2)))

      return '''nsimd_cpu_v{typ} ret;
              {content}
              return ret;'''.format(content=content, **fmtspec)

# -----------------------------------------------------------------------------

def unzip(func, typ):
  n = get_nb_el(typ)
  content = ''
  if int(n/2) != 0:
    if func == "unziplo":
      content = '\n'.join('ret.v{i} = {in0}.v{j}; '. \
                  format(i=i, j=i*2, **fmtspec) \
                  for i in range(0, int(n/2)))
      content = content + '\n'.join('ret.v{i} = {in1}.v{j}; '. \
                  format(i=i, j=2*(i-int(n/2)), **fmtspec) \
                  for i in range(int(n/2), n))
    else :
      content = '\n'.join('ret.v{i} = {in0}.v{j}; '. \
                  format(i=i, j=i*2+1, **fmtspec) \
                  for i in range(0, int(n/2)))
      content = content + '\n'.join('ret.v{i} = {in1}.v{j}; '. \
                  format(i=i, j=2*(i-int(n/2))+1, **fmtspec)\
                  for i in range(int(n/2), n))

    return '''nsimd_cpu_v{typ} ret;
            {content}
            return ret;'''.format(content=content, **fmtspec)
  else:
    return '''(void)({in1});
              return {in0};'''.format(**fmtspec)


# -----------------------------------------------------------------------------

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
      'in3': common.in3,
      'in4': common.in4,
      'typnbits': from_typ[1:]
    }

    impls = {
        'loada': lambda: load(from_typ),
        'load2a': lambda: load_deg234(from_typ, 2),
        'load3a': lambda: load_deg234(from_typ, 3),
        'load4a': lambda: load_deg234(from_typ, 4),
        'loadu': lambda: load(from_typ),
        'load2u': lambda: load_deg234(from_typ, 2),
        'load3u': lambda: load_deg234(from_typ, 3),
        'load4u': lambda: load_deg234(from_typ, 4),
        'storea': lambda: store(from_typ),
        'store2a': lambda: store_deg234(from_typ, 2),
        'store3a': lambda: store_deg234(from_typ, 3),
        'store4a': lambda: store_deg234(from_typ, 4),
        'storeu': lambda: store(from_typ),
        'store2u': lambda: store_deg234(from_typ, 2),
        'store3u': lambda: store_deg234(from_typ, 3),
        'store4u': lambda: store_deg234(from_typ, 4),
        'loadla': lambda: loadl(from_typ),
        'loadlu': lambda: loadl(from_typ),
        'storela': lambda: storel(from_typ),
        'storelu': lambda: storel(from_typ),
        'add': lambda: op2('+', from_typ),
        'mul': lambda: op2('*', from_typ),
        'div': lambda: op2('/', from_typ),
        'sub': lambda: op2('-', from_typ),
        'orb': lambda: bitwise2('|', from_typ),
        'orl': lambda: lop2('|', from_typ),
        'andb': lambda: bitwise2('&', from_typ),
        'andnotb': lambda: andnot2(from_typ),
        'andnotl': lambda: landnot2(from_typ),
        'andl': lambda: lop2('&', from_typ),
        'xorb': lambda: bitwise2('^', from_typ),
        'xorl': lambda: lop2('^', from_typ),
        'min': lambda: minmax2('min', from_typ),
        'max': lambda: minmax2('max', from_typ),
        'notb': lambda: not1(from_typ),
        'notl': lambda: lnot1(from_typ),
        'sqrt': lambda: sqrt1(from_typ),
        'set1': lambda: set1(from_typ),
        'shr': lambda: bitwise1_param('>>', from_typ),
        'shl': lambda: bitwise1_param('<<', from_typ),
        'eq': lambda: cmp2('==', from_typ),
        'ne': lambda: cmp2('!=', from_typ),
        'gt': lambda: cmp2('>', from_typ),
        'ge': lambda: cmp2('>=', from_typ),
        'lt': lambda: cmp2('<', from_typ),
        'le': lambda: cmp2('<=', from_typ),
        'len': lambda: len1(from_typ),
        'if_else1': lambda: if_else1(from_typ),
        'abs': lambda: abs1(from_typ),
        'fma': lambda: fma_fms('fma', from_typ),
        'fnma': lambda: fma_fms('fnma', from_typ),
        'fms': lambda: fma_fms('fms', from_typ),
        'fnms': lambda: fma_fms('fnms', from_typ),
        'ceil': lambda: ceil1(from_typ),
        'floor': lambda: floor1(from_typ),
        'trunc': lambda: trunc1(from_typ),
        'round_to_even': lambda: round_to_even1(from_typ),
        'all': lambda: all_any(from_typ, 'all'),
        'any': lambda: all_any(from_typ, 'any'),
        'reinterpret': lambda: reinterpret1(from_typ, to_typ),
        'reinterpretl': lambda: reinterpretl1(from_typ, to_typ),
        'cvt': lambda: convert1(from_typ, to_typ),
        'rec11': lambda: rec_rec11(from_typ),
        'rec8': lambda: rec_rec11(from_typ),
        'rsqrt11': lambda: rsqrt11(from_typ),
        'rsqrt8': lambda: rsqrt11(from_typ),
        'rec': lambda: rec_rec11(from_typ),
        'neg': lambda: neg1(from_typ),
        'nbtrue': lambda: nbtrue1(from_typ),
        'reverse': lambda: reverse1(from_typ),
        'addv': lambda: addv1(from_typ),
        'upcvt': lambda: upcvt1(from_typ, to_typ),
        'downcvt': lambda: downcvt2(from_typ, to_typ),
        'to_logical': lambda: to_logical1(from_typ),
        'to_mask': lambda: to_mask1(from_typ),
        'ziplo': lambda: zip_half('ziplo', from_typ),
        'ziphi': lambda: zip_half('ziphi', from_typ),
        'unziplo': lambda: unzip('unziplo', from_typ),
        'unziphi': lambda: unzip('unziphi', from_typ)
    }
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if not from_typ in common.types:
        raise ValueError('Unknown from_type "{}"'.format(from_typ))
    if not func in impls:
        return common.NOT_IMPLEMENTED
    return impls[func]()
