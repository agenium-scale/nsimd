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

import os
import common
import operators

# -----------------------------------------------------------------------------

def uniop(func, typ):
    normal = 'return ({{typ}})({func});'.format(func=func.format(in0='{in0}'))
    if typ == 'f16':
        return \
        '''#ifdef NSIMD_FP16
             {normal}
           #else
             return nsimd_f32_to_f16({func});
           #endif'''.format(normal=normal, func=func.\
           format(in0='nsimd_f16_to_f32({in0})'))
    else:
        return normal

# -----------------------------------------------------------------------------

def binop(func, typ):
    normal = 'return ({{typ}})({func});'. \
             format(func=func.format(in0='{in0}', in1='{in1}'))
    if typ == 'f16':
        return \
        '''#ifdef NSIMD_FP16
             {normal}
           #else
             return nsimd_f32_to_f16({func});
           #endif'''.format(normal=normal, func=func.\
           format(in0='nsimd_f16_to_f32({in0})',
                  in1='nsimd_f16_to_f32({in1})'))
    else:
        return normal

# -----------------------------------------------------------------------------

def binop_bin(func, typ):
    in0 = '{in0}' if typ in common.utypes else \
          'nsimd_reinterpret_u{}_{}({{in0}})'.format(typ[1:], typ)
    in1 = '{in1}' if typ in common.utypes else \
          'nsimd_reinterpret_u{}_{}({{in1}})'.format(typ[1:], typ)
    if typ in common.utypes:
        return 'return ({typ})({func});'. \
               format(typ=typ, func=func.format(in0=in0, in1=in1))
    else:
        return \
        'return ({typ})nsimd_reinterpret_{typ}_u{typnbits}(({func}));'. \
        format(typ=typ, typnbits=typ[1:], func=func.format(in0=in0, in1=in1))

# -----------------------------------------------------------------------------

def libm_opn(func, arity, typ, until_cpp11, c89_code):
    cxx_version = '> 0' if not until_cpp11 else '>= 2011'
    comment = \
    '''/* {func} is not available in C89 but is given by POSIX 2001 */
       /* and C99. But we do not want to pollute the user includes  */
       /* and POSIX value if set so we play dirty.                  */'''. \
       format(func=func)
    args = ', '.join(['{{in{}}}'.format(i) for i in range(arity)])
    args_f16 = ', '.join(['nsimd_f16_to_f32({{in{}}})'.format(i) \
                          for i in range(arity)])
    args_f64 = ', '.join(['(f64){{in{}}}'.format(i) for i in range(arity)])
    args_f64_f16 = ', '.join(['(f64)nsimd_f16_to_f32({{in{}}})'.format(i) \
                              for i in range(arity)])
    if typ == 'f16':
        c99_code = 'return nsimd_f32_to_f16({}f({}));'.format(func, args_f16)
        if c89_code == '':
            c89_code = 'return nsimd_f32_to_f16((f32){}({}));'. \
                       format(func, args_f64_f16)
        return \
        '''  {comment}
           #if defined(NSIMD_IS_MSVC) && _MSC_VER <= 1800 /* VS 2012 */
             {c89_code}
           #else
             #if NSIMD_CXX {cxx_version} || NSIMD_C >= 1999 || \
                 _POSIX_C_SOURCE >= 200112L
               {c99_code}
             #else
               {c89_code}
             #endif
           #endif'''. \
           format(comment=comment, cxx_version=cxx_version, c89_code=c89_code,
                  c99_code=c99_code)
    elif typ == 'f32':
        c99_code = 'return {}f({});'.format(func, args)
        if c89_code == '':
            c89_code = 'return (f32){}({});'.format(func, args_f64)
        return \
        '''  {comment}
           #if defined(NSIMD_IS_MSVC) && _MSC_VER <= 1800 /* VS 2012 */
             {c89_code}
           #else
             #if NSIMD_CXX {cxx_version} || NSIMD_C >= 1999 || \
                 _POSIX_C_SOURCE >= 200112L
               {c99_code}
             #else
               {c89_code}
             #endif
           #endif'''. \
           format(comment=comment, cxx_version=cxx_version, c89_code=c89_code,
                  c99_code=c99_code)
    else:
        normal = 'return {}({});'.format(func, args)
        if c89_code == '':
            return normal
        return \
        '''  {comment}
           #if NSIMD_CXX {cxx_version} || NSIMD_C >= 1999 || \
               _POSIX_C_SOURCE >= 200112L
             {normal}
           #else
             {c89_code}
           #endif'''. \
           format(comment=comment, normal=normal, c89_code=c89_code,
                  cxx_version=cxx_version)

# -----------------------------------------------------------------------------

def round_to_even(typ):
    if typ in ['f32', 'f64']:
        return \
        '''{typ} fl = nsimd_floor_{typ}({{in0}});
           {typ} ce = nsimd_ceil_{typ}({{in0}});
           {typ} df = {{in0}} - fl; /* exactly representable in IEEE754 */
           {typ} dc = ce - {{in0}}; /* exactly representable in IEEE754 */
           if (df < dc) {{{{
             return fl;
           }}}} else if (df > dc) {{{{
             return ce;
           }}}} else {{{{
             {typ} fld2 = fl * 0.5{f}; /* exactly representable in IEEE754 */
             if (fld2 == nsimd_floor_{typ}(fld2)) {{{{
               return fl;
             }}}} else {{{{
               return ce;
             }}}}
           }}}}'''.format(typ=typ, f='f' if typ == 'f32' else '')
    else:
        return \
        '''f32 in0 = nsimd_f16_to_f32({in0});
           f32 fl = nsimd_floor_f32(in0);
           f32 ce = nsimd_ceil_f32(in0);
           f32 df = in0 - fl; /* exactly representable in IEEE754 */
           f32 dc = ce - in0; /* exactly representable in IEEE754 */
           if (df < dc) {{
             return nsimd_f32_to_f16(fl);
           }} else if (df > dc) {{
             return nsimd_f32_to_f16(ce);
           }} else {{
             f32 fld2 = fl * 0.5f; /* exactly representable in IEEE754 */
             if (fld2 == nsimd_floor_f32(fld2)) {{
               return nsimd_f32_to_f16(fl);
             }} else {{
               return nsimd_f32_to_f16(ce);
             }}
           }}'''

# -----------------------------------------------------------------------------

def get_impl(operator, totyp, typ, until_cpp11 = False, c89_code = ''):
    if operator.name == 'trunc':
        if typ == 'f16':
            return '''f32 buf = nsimd_f16_to_f32({in0});
                      return nsimd_f32_to_f16(buf >= 0.0f ?
                                              nsimd_floor_f32(buf) :
                                              nsimd_ceil_f32(buf));'''
        else:
            return '''return {{in0}} >= 0.0{f} ? nsimd_floor_{typ}({{in0}})
                             : nsimd_ceil_{typ}({{in0}});'''. \
                             format(typ=typ, f='f' if typ == 'f32' else '')
        return libm_opn('trunc', 3, typ, True, c89_code)
    if operator.name == 'round_to_even':
        return round_to_even(typ)
    if operator.name in ['floor', 'ceil', 'sqrt', 'fma']:
        return libm_opn(operator.name, 3 if operator.name == 'fma' else 1,
                        typ, False, '')
    #func = {
    #    'orb' = '{in0} | {in1}'
    #    'andb' = '{in0} & {in1}'
    #    'andnotb' = '{in0} & (~{in1})'
    #    'notb' = '~{in0}'
    #    'xorb' = '{in0} ^ {in1}'
    #    'add' = '{in0} + {in1}'
    #    'sub' = '{in0} - {in1}'
    #    'mul' = '{in0} * {in1}'
    #    'div' = '{in0} / {in1}'
    #    'neg' = '-{in0}'
    #    'min' = '{in0} < {in1} ? {in0} : {in1}'
    #    'max' = '{in0} > {in1} ? {in0} : {in1}'
    #    'abs' = '{in0}' if typ in common.utypes \
    #                    else '{in0} >= 0 ? {in0} : -{in0}'
    #    'fma' = 'return a0 * a1 + a2;'
    #    'fnma' = 'return -a0 * a1 + a2;'
    #    'fms' = 'return a0 * a1 - a2;'
    #    'fnms' = 'return -a0 * a1 - a2;'
    #    'ceil' = 'return std::ceil(a);'
    #    'floor'] = 'return std::floor(a);'
    #    impl['trunc'] = '''#if NSIMD_CXX >= 2011
    #                                  return std::trunc(a);
    #                                  #else
    #                                  return (a > 0) ? std::floor(a) : std::ceil(a);
    #                                  #endif
    #                               '''
    #    impl['round_to_even'] = '''#if NSIMD_CXX >= 2011
    #                               return std::round(a / 2) * 2;
    #                               #else
    #                               return std::floor((a / 2) + 0.5) * 2;
    #                               #endif
    #                            '''
    #    impl['reinterpret'] = 'return a;' # FIXME
    #    impl['cvt'] = 'return a;' # FIXME
    #    impl['rec'] = 'return 1 / a;'
    #    impl['rec11'] = 'return 1 / a;'
    #    impl['rec8'] = 'return 1 / a;'
    #    impl['sqrt'] = 'return std::sqrt(a);'
    #    impl['rsqrt'] = 'return 1 / std::sqrt(a);'
    #    impl['rsqrt11'] = 'return 1 / std::sqrt(a);'
    #    impl['rsqrt8'] = 'return 1 / std::sqrt(a);'

# -----------------------------------------------------------------------------

def reinterprets(opts):
    ret = ''

    tmpl_ne = \
    '''NSIMD_INLINE {totyp} nsimd_reinterpret_{totyp}_{typ}({typ} {in0}) {{
      union {{ {totyp} to; {typ} from; }} buf;
      buf.from = {in0};
      return ({totyp})buf.to;
    }}'''

    tmpl_eq = 'NSIMD_INLINE {typ} ' + \
              'nsimd_reinterpret_{typ}_{typ}({typ} {in0}) {{ return {in0}; }}'

    tmpl_cxx = 'NSIMD_INLINE {totyp} ' + \
               'reinterpret_{totyp}_{typ}({typ} {in0}) {{' + \
               ' return nsimd_reinterpret_{totyp}_{typ}({in0}); }}\n'

    for typ in common.types:
        totyp = 'u{}'.format(typ[1:])
        if typ == 'f16':
            ret += '''NSIMD_INLINE u16 nsimd_reinterpret_u16_f16(f16 {in0}) {{
                      #ifdef NSIMD_FP16
                        union {{ u16 to; f16 from; }} buf;
                        buf.from = {in0};
                        return buf.to;
                      #else
                        return {in0}.u;
                      #endif
                      }}

                      NSIMD_INLINE f16 nsimd_reinterpret_f16_u16(u16 {in0}) {{
                      #ifdef NSIMD_FP16
                        union {{ f16 to; u16 from; }} buf;
                        buf.from = {in0};
                        return buf.to;
                      #else
                        f16 ret;
                        ret.u = {in0};
                        return ret;
                      #endif
                      }}'''.format(in0=common.in0)
        else:
            if totyp != typ:
                ret += tmpl_ne.format(typ=typ, totyp=totyp, in0=common.in0) + \
                       '\n\n'
                ret += tmpl_ne.format(typ=totyp, totyp=typ, in0=common.in0) + \
                       '\n\n'
            else:
                ret += tmpl_eq.format(typ=typ, in0=common.in0) + '\n\n'

    ret += '''#if NSIMD_CXX > 0

    namespace nsimd {

    '''

    for typ in common.types:
        totyp = 'u{}'.format(typ[1:])
        if totyp != typ:
            ret += tmpl_cxx.format(typ=typ, totyp=totyp, in0=common.in0) + \
                   '\n\n'
            ret += tmpl_cxx.format(typ=totyp, totyp=typ, in0=common.in0) + \
                   '\n\n'
        else:
            ret += tmpl_cxx.format(totyp=typ, typ=typ, in0=common.in0) + '\n\n'

    ret += '''
    } // namespace nsimd

    #endif

    '''

    return ret

# -----------------------------------------------------------------------------

def libms(opts):
    fmtspec = {'in0': common.in0, 'in1': common.in1, 'in2': common.in2}
    ret = ''
    for op_name in ['floor', 'ceil', 'sqrt', 'fma', 'trunc', 'round_to_even']:
        for typ in common.ftypes:
            op = operators.operators[op_name]
            arity = len(op.args)
            args = ', '.join(['{} {{in{}}}'.format(typ, i).format(**fmtspec) \
                              for i in range(arity)])
            ret += 'NSIMD_INLINE {typ} nsimd_{op_name}_{typ}({args}) {{\n'. \
                   format(typ=typ, op_name=op_name, args=args, **fmtspec)
            ret += get_impl(op, typ, typ).format(**fmtspec)
            ret += '\n}\n\n'
            ret += '#if NSIMD_CXX > 0\n\nnamespace nsimd {\n\n'
            ret += 'NSIMD_INLINE {typ} {op_name}_{typ}({args}) {{\n'. \
                   format(typ=typ, op_name=op_name, args=args, **fmtspec)
            ret += '  return nsimd_{op_name}_{typ}({args});'. \
                   format(op_name=op_name, typ=typ, args = ', '.join(
                          ['{{in{}}}'.format(i).format(**fmtspec) \
                           for i in range(arity)]))
            ret += '\n}\n\n} // namespace nsimd\n\n#endif\n\n'
    return ret

# -----------------------------------------------------------------------------

def doit(opts):
    filename = os.path.join(opts.include_dir, 'scalar_utilities.h')
    if not common.can_create_filename(opts, filename):
        return
    print('-- Generating scalar utilities')
    with common.open_utf8(opts, filename) as out:
        out.write('''#ifndef NSIMD_SCALAR_UTILITIES_H
                     #define NSIMD_SCALAR_UTILITIES_H

                     #if NSIMD_CXX > 0
                     #include <cmath>
                     #else
                     #include <math.h>
                     #endif

                     ''')
        out.write(reinterprets(opts))
        out.write(libms(opts))
        out.write('\n\n#endif\n')
    common.clang_format(opts, filename)
