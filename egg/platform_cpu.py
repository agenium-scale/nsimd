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

def get_type(opts, simd_ext, typ):
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    typ2 = typ if typ != 'f16' else 'f32'
    members = '\n'.join('{} v{};'.format(typ2, i) \
                        for i in range(0, get_nb_el(typ)))
    return 'struct {{ {} }}'.format(members)

def get_logical_type(opts, simd_ext, typ):
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
    elif func == 'adds':
        return  '''
                #include <nsimd/cpu/cpu/add.h>
                '''
    elif func == 'subs':
        return'''
                #include <nsimd/cpu/cpu/sub.h>
                #include <nsimd/cpu/cpu/adds.h>
                #include <nsimd/cpu/cpu/neg.h>
               '''
    elif func in ['']:
        return '''#include <nsimd/cpu/cpu/reinterpret.h>
                  '''
    elif func == 'zip':
        return '''#include <nsimd/cpu/cpu/ziplo.h>
                  #include <nsimd/cpu/cpu/ziphi.h>
                  '''
    elif func == 'unzip':
         return '''#include <nsimd/cpu/cpu/unziplo.h>
                   #include <nsimd/cpu/cpu/unziphi.h>
                  '''
    elif func == 'shra':
        return '''#include <nsimd/cpu/{simd_ext}/shr.h>
                  '''.format(simd_ext=simd_ext)
    elif func == 'clz':
        return  '''
                #include <nsimd/cpu/cpu/shrv.h>
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
    typ2 = 'f32' if typ == 'f16' else typ
    return func_body(
           '''ret.v{{i}} = ({typ2})({in0}.v{{i}} {op} {in1}.v{{i}} ?
                                    {in0}.v{{i}} : {in1}.v{{i}});'''. \
                                    format(typ2=typ2, op=op, **fmtspec), typ)

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
              '''ret.v{{i}} = (u32)({in0}.v{{i}} {op} {in1}.v{{i}}
                                    ? -1 : 0);'''. \
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
                  '''ret.v{{i}} = (u32)(nsimd_u16_to_f32(((u16 *){in0})[{{i}}])
                                      == 0.0f ? 0 : -1);'''. \
                                      format(**fmtspec), typ)
    else:
        content = repeat_stmt(
                  '''ret.v{{i}} = (u32)({in0}[{{i}}] == ({typ})0
                                        ? 0 : -1);'''. \
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
                  '''((u16*){in0})[{{i}}] = (u16)({in1}.v{{i}} == (u32)0
                                            ? nsimd_f32_to_u16(0.0f)
                                            : nsimd_f32_to_u16(1.0f));'''. \
                                            format(**fmtspec), typ)
    else:
        content = repeat_stmt(
                  '''{in0}[{{i}}] = ({typ})({in1}.v{{i}} == (u32)0
                                  ? ({typ})0 : ({typ})1);'''. \
                                  format(**fmtspec), typ)
    return content

# -----------------------------------------------------------------------------

def if_else1(typ):
    typ2 = 'f32' if typ == 'f16' else typ
    return func_body(
           '''ret.v{{i}} = ({typ2})({in0}.v{{i}} != (u32)0
                                    ? {in1}.v{{i}} : {in2}.v{{i}});'''. \
                                    format(typ2=typ2, **fmtspec), typ)

# -----------------------------------------------------------------------------

def abs1(typ):
    if typ in common.utypes:
        return func_body('ret.v{{i}} = {in0}.v{{i}};'.format(**fmtspec), typ)
    typ2 = 'f32' if typ == 'f16' else typ
    return func_body(
           '''ret.v{{i}} = ({typ2})({in0}.v{{i}} < ({typ2})0
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

def adds_itypes(assign_max, assign_add_if_ok):
    check_overflow = \
           'if (({in0}.v{{i}} > 0) && ({in1}.v{{i}} > {max} - {in0}.v{{i}}))'
    assign_max_if_overflow = check_overflow + '\n' + assign_max

    check_underflow = \
           'else if (({in0}.v{{i}} < 0) && ({in1}.v{{i}} < {min} - {in0}.v{{i}}))'
    assign_min = '{{{{ ret.v{{i}} = {min}; }}}}'
    assign_min_if_underflow = check_underflow + '\n' + assign_min

    algo = assign_max_if_overflow + '\n' + \
           assign_min_if_underflow + '\n' + \
           assign_add_if_ok

    return algo


def adds_utypes(assign_max, assign_add_if_ok):
    check_overflow = 'if ({in1}.v{{i}} > {max} - {in0}.v{{i}})'
    assign_max_if_overflow = check_overflow + '\n' + assign_max

    algo = assign_max_if_overflow + '\n' + \
           assign_add_if_ok

    return algo


def adds(from_typ):

    if from_typ in common.ftypes:
      return 'return nsimd_add_{simd_ext}_{from_typ}({in0}, {in1});'.format(**fmtspec)

    if not from_typ in common.limits.keys():
      raise ValueError('Type not implemented in platform_{simd_ext} adds({from_typ})'.format(from_typ))

    assign_max = '{{{{ ret.v{{i}} = {max}; }}}}'
    add = '{{{{ ret.v{{i}} = ({from_typ})({in0}.v{{i}} + {in1}.v{{i}}); }}}}'
    assign_add_if_ok = 'else ' + '\n' + add

    if from_typ in common.itypes:
        algo = adds_itypes(assign_max, assign_add_if_ok)

    else:
        algo = adds_utypes(assign_max, assign_add_if_ok)

    type_limits = common.limits[from_typ]
    content = repeat_stmt(algo.format(**type_limits, **fmtspec), from_typ)

    return '''nsimd_{simd_ext}_v{from_typ} ret;

              {content}

              return ret;'''.format(from_typ = from_typ, content = content, simd_ext=fmtspec['simd_ext'])

# -----------------------------------------------------------------------------

def subs(from_typ):

    if from_typ in common.itypes:
      return 'return nsimd_adds_{simd_ext}_{from_typ}({in0},nsimd_neg_{simd_ext}_{from_typ}({in1}));'.format(**fmtspec)

    if from_typ in common.ftypes:
      return 'return nsimd_sub_{simd_ext}_{from_typ}({in0}, {in1});'.format(**fmtspec)

    if from_typ not in common.utypes:
      raise ValueError('Type not implemented in platform_{simd_ext} adds({from_typ})'.format(from_typ))

    check_underflow = 'if ({in0}.v{{i}} < {in1}.v{{i}})'
    assign_min = '{{{{ ret.v{{i}} = ({from_typ}){min}; }}}}'
    assign_min_if_underflow = check_underflow + '\n' + assign_min
    sub = '{{{{ ret.v{{i}} = ({from_typ})({in0}.v{{i}} - {in1}.v{{i}}); }}}}'
    assign_sub_if_ok = 'else ' + '\n' + sub

    algo = assign_min_if_underflow + '\n' + \
           assign_sub_if_ok

    type_limits = common.limits[from_typ]
    content = repeat_stmt(algo.format(**type_limits, **fmtspec), from_typ)

    return '''nsimd_{simd_ext}_v{from_typ} ret;

               {content}

               return ret;'''.format(from_typ = from_typ, content = content, simd_ext=fmtspec['simd_ext'])

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
        'ret.v{{i}} = ({utyp})({in0}.v{{i}} ? -1 : 0);'. \
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

def unzip_half(func, typ):
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

def zip(from_typ):
    return '''\
    nsimd_{simd_ext}_v{typ}x2 ret;
    ret.v0 = nsimd_ziplo_cpu_{typ}({in0}, {in1});
    ret.v1 = nsimd_ziphi_cpu_{typ}({in0}, {in1});
    return ret;
    '''.format(**fmtspec)

def unzip(from_typ):
    return '''\
    nsimd_{simd_ext}_v{typ}x2 ret;
    ret.v0 = nsimd_unziplo_cpu_{typ}({in0}, {in1});
    ret.v1 = nsimd_unziphi_cpu_{typ}({in0}, {in1});
    return ret;
    '''.format(**fmtspec)

# -----------------------------------------------------------------------------
## shift right

def shra(typ):
    n = get_nb_el(typ)
    content = ''
    # Sign extension for a right shift on signed values is compiler-dependant.
    # There is no guarantee that the sign extension is performed, therefore,
    # we do the sign extension manually using a mask.
    content = '\n'.join('''\
    /* -------------------------------------------- */
    if(a1 >= {typnbits}) {{
      ret.v{i} = ({typ}) 0; 
    }} else {{
    val.i = {in0}.v{i};
    mask = (u{typnbits})((val.u >> ({typnbits} - 1)) * ~(u{typnbits})(0) << shift);
    ret.v{i} = ({typ})((val.u >> {in1}) | mask);}}'''.\
                    format(**fmtspec, i=i)for i in range(0, n))
    if typ in common.utypes:
        return '''return nsimd_shr_{simd_ext}_{typ}({in0}, {in1});'''. \
            format(**fmtspec)
    else:
        return '''\
        union {{i{typnbits} i; u{typnbits} u;}} val;
        nsimd_cpu_v{typ} ret;
        const int shift = {typnbits} - 1 - {in1};
        u{typnbits} mask;
        {content}
        return ret;'''.format(content=content, **fmtspec)

# -----------------------------------------------------------------------------

def clz_helper( depth ):
  compare = [ '0x3' , '0xF' , '0xFF' , '0xFFFF' , '0xFFFFFFFF' ]
  shift   = [ '1' , '2' , '3' , '4' , '5' ]
  maxes   = [ 1 , 3 , 7 , 15 , 31 , 63 ]
  nmx = str(maxes[depth])

  compare = compare[0:depth]
  shift   = shift  [0:depth]

  compare = compare[::-1]
  shift   = shift  [::-1]

  prelude = '''\
  nsimd_{simd_ext}_v{from_typ} ones   = nsimd_set1_{simd_ext}_{from_typ}(1);
  nsimd_{simd_ext}_v{from_typ} zeroes = nsimd_set1_{simd_ext}_{from_typ}(0);
  nsimd_{simd_ext}_vl{from_typ} q_gt, lt_zero;
  nsimd_{simd_ext}_v{from_typ} q, r;
  nsimd_{simd_ext}_v{from_typ} x = {in0};
  '''.format(**fmtspec)

  intro = '''\

  q_gt = nsimd_gt_{simd_ext}_{from_typ}( x , nsimd_set1_{simd_ext}_{from_typ}({comp}) );
  q    = nsimd_shl_{simd_ext}_{from_typ}( nsimd_if_else1_{simd_ext}_{from_typ}( q_gt , ones , zeroes ) , {sh} );
  x = nsimd_shrv_{simd_ext}_{from_typ}( x , q );
  r = q;
  '''.format(**fmtspec, comp=compare[0], sh=shift[0])

  compare = compare[1:]
  shift   = shift  [1:]

  body = ''
  for i in range( 0 , len(compare) ):
    body += '''\

    q_gt = nsimd_gt_{simd_ext}_{from_typ}( x , nsimd_set1_{simd_ext}_{from_typ}({comp}) );
    q    = nsimd_shl_{simd_ext}_{from_typ}( nsimd_if_else1_{simd_ext}_{from_typ}( q_gt , ones , zeroes ) , {sh} );
    x = nsimd_shrv_{simd_ext}_{from_typ}( x , q );
    r = nsimd_orb_{simd_ext}_{from_typ}( r , q );
    '''.format(**fmtspec, comp=compare[i], sh=shift[i])

  outro = '''\

  r = nsimd_orb_{simd_ext}_{from_typ}( r , nsimd_shr_{simd_ext}_{from_typ}( x , 1 ) );
  r = nsimd_sub_{simd_ext}_{from_typ}( nsimd_set1_{simd_ext}_{from_typ}({nearmax}) , r );
  lt_zero = nsimd_lt_{simd_ext}_{from_typ}( x , zeroes );
  r = nsimd_if_else1_{simd_ext}_{from_typ}( lt_zero , zeroes , r );
  return r;
  '''.format(**fmtspec, nearmax=nmx)

  return (prelude + intro + body + outro)


# Using algorithm from:
#   https://en.wikipedia.org/wiki/Find_first_set#CLZ - clz5
#   No benchmarking was done to compare performance of the different algorithms
# Note: builtins have different behaviour for 0, so we should avoid 0 in tests
def clz(from_typ):
  r = ''
  if from_typ in [ 'i8'  , 'u8'  ]:
    return clz_helper(2)
    r += '''\
    #if defined(NSIMD_IS_CLANG) || defined(NSIMD_IS_ICC)
      {clicc}
    #elif defined(NSIMD_IS_GCC)
      {gcc}
    #elif defined(NSIMD_IS_MSVC)
      {msvc}
    #else
    '''.format(**fmtspec
              , clicc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}}) - 8;'. \
                                format(op='__builtin_clzs', **fmtspec), from_typ)
              , gcc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}}) - 24;'. \
                                format(op='__builtin_clz', **fmtspec), from_typ)
              , msvc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}}) - 8;'. \
                                format(op='__builtin_lzcnt16', **fmtspec), from_typ)
              )
    r += clz_helper( 2 )
    r += '''
    #endif
    '''
    return r
  if from_typ in [ 'i16' , 'u16' ]:
    return clz_helper(3)
    r += '''\
    #if defined(NSIMD_IS_CLANG) || defined(NSIMD_IS_ICC)
      {clicc}
    #elif defined(NSIMD_IS_GCC)
      {gcc}
    #elif defined(NSIMD_IS_MSVC)
      {msvc}
    #else
    '''.format(**fmtspec
              , clicc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}});'. \
                                format(op='__builtin_clzs', **fmtspec), from_typ)
              , gcc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}}) - 16;'. \
                                format(op='__builtin_clz', **fmtspec), from_typ)
              , msvc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}});'. \
                                format(op='__builtin_lzcnt16', **fmtspec), from_typ)
              )
    r += clz_helper( 3 )
    r += '''
    #endif
    '''
    return r
  if from_typ in [ 'i32' , 'u32' ]:
    return clz_helper(4)
    r += '''\
    #if defined(NSIMD_IS_CLANG) || defined(NSIMD_IS_ICC)
      {clicc}
    #elif defined(NSIMD_IS_GCC)
      {gcc}
    #elif defined(NSIMD_IS_MSVC)
      {msvc}
    #else
    '''.format(**fmtspec
              , clicc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}});'. \
                                format(op='__builtin_clz', **fmtspec), from_typ)
              , gcc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}});'. \
                                format(op='__builtin_clz', **fmtspec), from_typ)
              , msvc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}});'. \
                                format(op='__builtin_lzcnt32', **fmtspec), from_typ)
              )
    r += clz_helper( 4 )
    r += '''
    #endif
    '''
    return r
  if from_typ in [ 'i64' , 'u64' ]:
    return clz_helper(5)
    r += '''\
    #if defined(NSIMD_IS_CLANG) || defined(NSIMD_IS_ICC)
      {clicc}
    #elif defined(NSIMD_IS_GCC)
      {gcc}
    #elif defined(NSIMD_IS_MSVC)
      {msvc}
    #else
    '''.format(**fmtspec
              , clicc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}});'. \
                                format(op='__builtin_clzl', **fmtspec), from_typ)
              , gcc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}});'. \
                                format(op='__builtin_clzl', **fmtspec), from_typ)
              , msvc=func_body('ret.v{{i}} = ({typ}){op}({in0}.v{{i}});'. \
                                format(op='__builtin_lzcnt64', **fmtspec), from_typ)
              )
    r += clz_helper( 5 )
    r += '''
    #endif
    '''
    return r

# -----------------------------------------------------------------------------

# Barely modified from shra
def shrv(typ):
    n = get_nb_el(typ)
    content = ''
    # Sign extension for a right shift has implementation-defined behaviour.
    # To be sure it is performed, we do it manually.
    content = '\n'.join('''\
    /* -------------------------------------------- */
    const int shift{i} = {typnbits} - {in1}.v{i};
    val.ival = {in0}.v{i};
    if(val.ival < 0){{
      sign.ival = -1;
      sign.uval = (u{typnbits})(sign.uval << shift{i});
    }} else {{
      sign.uval = 0u;
    }}
    shifted = (u{typnbits})(val.uval >> {in1}.v{i});
    ret.v{i} = (i{typnbits}) (shifted | sign.uval);'''.\
                    format(**fmtspec, i=i)for i in range(0, n))
    if typ in common.utypes:
        return func_body('ret.v{{i}} = ({typ})({in0}.v{{i}} {op} {in1}.v{{i}});'. \
                         format(op='>>', **fmtspec), typ)
    else:
        return '''\
        nsimd_cpu_v{typ} ret;
        union {{i{typnbits} ival; u{typnbits} uval;}} val;
        union {{i{typnbits} ival; u{typnbits} uval;}} sign;
        u{typnbits} shifted;

        {content}

        return ret;'''.format(content=content, **fmtspec)


def shlv(from_typ):
    if from_typ in common.utypes:
        return func_body('ret.v{{i}} = ({typ})({in0}.v{{i}} {op} {in1}.v{{i}});'. \
                         format(op='<<', **fmtspec), from_typ)
    else:
        return '''nsimd_cpu_v{typ} ret;
                  union {{ {typ} i; {utyp} u; }} buf;
                  {content}
                  return ret;'''. \
                  format(content=repeat_stmt(
                  '''buf.i = {in0}.v{{i}};
                     buf.u = ({utyp})(buf.u {op} {in1}.v{{i}});
                     ret.v{{i}} = buf.i;'''.format(op='<<', **fmtspec), typ=from_typ),
                     **fmtspec)


# -----------------------------------------------------------------------------

def get_impl(opts, func, simd_ext, from_typ, to_typ=''):

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
        'adds' : lambda: adds(from_typ),
        'subs' : lambda: subs(from_typ),
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
        'shra': lambda:shra(from_typ),
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
        'unziplo': lambda: unzip_half('unziplo', from_typ),
        'unziphi': lambda: unzip_half('unziphi', from_typ),
        'zip' : lambda : zip(from_typ),
        'unzip' : lambda : unzip(from_typ),
        'clz' : lambda : clz(from_typ),
        'shlv' : lambda : shlv(from_typ),
        'shrv' : lambda : shrv(from_typ)
    }
    if simd_ext != 'cpu':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if not from_typ in common.types:
        raise ValueError('Unknown from_type "{}"'.format(from_typ))
    if not func in impls:
        return common.NOT_IMPLEMENTED
    return impls[func]()
