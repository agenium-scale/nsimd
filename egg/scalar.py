# Copyright (c) 2020 Agenium Scale
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

import common

fmtspec = dict()

# -----------------------------------------------------------------------------

def opnum(func, typ):
    normal = 'return ({typ})({func});'. \
             format(func=func.format(**fmtspec), **fmtspec)
    if typ == 'f16':
        return \
        '''#ifdef NSIMD_NATIVE_FP16
             {normal}
           #else
             return nsimd_f32_to_f16({func});
           #endif'''.format(normal=normal, func=func. \
           format(in0='nsimd_f16_to_f32({in0})',
                  in1='nsimd_f16_to_f32({in1})',
                  in2='nsimd_f16_to_f32({in2})').format(**fmtspec))
    else:
        return normal

# -----------------------------------------------------------------------------

def cmp(func, typ):
    normal = 'return ({func});'. \
             format(func=func.format(**fmtspec), **fmtspec)
    if typ == 'f16':
        return \
        '''#ifdef NSIMD_NATIVE_FP16
             {normal}
           #else
             return ({func});
           #endif'''.format(normal=normal, func=func. \
           format(in0='nsimd_f16_to_f32({in0})',
                  in1='nsimd_f16_to_f32({in1})',
                  in2='nsimd_f16_to_f32({in2})').format(**fmtspec))
    else:
        return normal

# -----------------------------------------------------------------------------

def opbit(func, typ):
    in0 = '{in0}'.format(**fmtspec) if typ in common.utypes else \
          'nsimd_scalar_reinterpret_u{typnbits}_{typ}({in0})'.format(**fmtspec)
    in1 = '{in1}'.format(**fmtspec) if typ in common.utypes else \
          'nsimd_scalar_reinterpret_u{typnbits}_{typ}({in1})'.format(**fmtspec)
    if typ in common.utypes:
        return 'return ({typ})({func});'. \
               format(func=func.format(in0=in0, in1=in1), **fmtspec)
    else:
        return '''return nsimd_scalar_reinterpret_{typ}_u{typnbits}(
                             (u{typnbits})({func}));'''.format(
                             func=func.format(in0=in0, in1=in1), **fmtspec)

# -----------------------------------------------------------------------------

def shift(func, typ):
    if func == 'shl':
        return 'return ({typ})({in0} << {in1});'.format(**fmtspec)
    # getting here means shr or shra
    if typ in common.utypes:
        return 'return ({typ})({in0} >> {in1});'.format(**fmtspec)
    # getting here means shr or shra on signed type
    utyp = common.bitfield_type[typ]
    if func == 'shr':
        return '''return nsimd_scalar_reinterpret_{typ}_{utyp}(
                           ({utyp})(nsimd_scalar_reinterpret_{utyp}_{typ}(
                             {in0}) >> {in1}));'''.format(utyp=utyp, **fmtspec)
    # getting here means shra on signed type
    return \
    '''if ({in1} == 0) {{
         return {in0};
       }}
       if ({in0} >= 0) {{
         return nsimd_scalar_reinterpret_{typ}_{utyp}(({utyp})(
                  nsimd_scalar_reinterpret_{utyp}_{typ}({in0}) >> {in1}));
       }} else {{
         {utyp} mask = ({utyp})((({utyp})-1) << ({typnbits} - {in1}));
         return nsimd_scalar_reinterpret_{typ}_{utyp}(({utyp})(mask |
                  ({utyp})(nsimd_scalar_reinterpret_{utyp}_{typ}(
                    {in0}) >> {in1})));
       }}'''.format(utyp=utyp, **fmtspec)

# -----------------------------------------------------------------------------

def libm_opn(func, arity, typ, until_cpp11, c89_code):
    cxx_version = '> 0' if not until_cpp11 else '>= 2011'
    comment = \
    '''/* {func} is not available in C89 but is given by POSIX 2001 */
       /* and C99. But we do not want to pollute the user includes  */
       /* and POSIX value if set so we play dirty.                  */'''. \
       format(func=func)
    args = ', '.join(['{{in{}}}'.format(i).format(**fmtspec) \
                      for i in range(arity)])
    args_f16 = ', '.join(['nsimd_f16_to_f32({{in{}}})'.format(i). \
                          format(**fmtspec) for i in range(arity)])
    args_f64 = ', '.join(['(f64){{in{}}}'.format(i).format(**fmtspec) \
                          for i in range(arity)])
    args_f64_f16 = ', '.join(['(f64)nsimd_f16_to_f32({{in{}}})'.format(i). \
                              format(**fmtspec) for i in range(arity)])
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
        '''{typ} fl = nsimd_scalar_floor_{typ}({in0});
           {typ} ce = nsimd_scalar_ceil_{typ}({in0});
           {typ} df = {in0} - fl; /* exactly representable in IEEE754 */
           {typ} dc = ce - {in0}; /* exactly representable in IEEE754 */
           if (df < dc) {{
             return fl;
           }} else if (df > dc) {{
             return ce;
           }} else {{
             {typ} fld2 = fl * 0.5{f}; /* exactly representable in IEEE754 */
             if (fld2 == nsimd_scalar_floor_{typ}(fld2)) {{
               return fl;
             }} else {{
               return ce;
             }}
           }}'''.format(f='f' if typ == 'f32' else '', **fmtspec)
    elif typ == 'f16':
        return \
        '''f32 in0 = nsimd_f16_to_f32({in0});
           f32 fl = nsimd_scalar_floor_f32(in0);
           f32 ce = nsimd_scalar_ceil_f32(in0);
           f32 df = in0 - fl; /* exactly representable in IEEE754 */
           f32 dc = ce - in0; /* exactly representable in IEEE754 */
           if (df < dc) {{
             return nsimd_f32_to_f16(fl);
           }} else if (df > dc) {{
             return nsimd_f32_to_f16(ce);
           }} else {{
             f32 fld2 = fl * 0.5f; /* exactly representable in IEEE754 */
             if (fld2 == nsimd_scalar_floor_f32(fld2)) {{
               return nsimd_f32_to_f16(fl);
             }} else {{
               return nsimd_f32_to_f16(ce);
             }}
           }}'''.format(**fmtspec)
    else:
        return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------

def reinterpret(totyp, typ):
    if totyp == typ:
        return 'return {in0};'.format(**fmtspec)
    via_union = '''union {{ {typ} from; {totyp} to; }} buf;
                   buf.from = {in0};
                   return buf.to;'''.format(**fmtspec)
    via_memcpy = '''{totyp} ret;
                    memcpy((void *)&ret, (void *)&{in0}, sizeof(ret));
                    return ret;'''.format(**fmtspec)
    if typ == 'f16':
        if totyp == 'u16':
            emulated = 'return {in0}.u;'.format(**fmtspec)
        else:
            emulated = 'return nsimd_scalar_reinterpret_i16_u16({in0}.u);'. \
                       format(**fmtspec)
        return \
        '''#if defined(NSIMD_NATIVE_FP16) && defined(NSIMD_IS_GCC)
             {via_union}
           #elif (defined(NSIMD_NATIVE_FP16) && !defined(NSIMD_IS_GCC)) || \
                 defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) || \
                 defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
             {via_memcpy}
           #else
             {emulated}
           #endif'''.format(via_union=via_union, via_memcpy=via_memcpy,
                            emulated=emulated)
    if totyp == 'f16':
        if typ == 'u16':
            emulated = '''f16 ret;
                          ret.u = {in0};
                          return ret;'''.format(**fmtspec)
        else:
            emulated = '''f16 ret;
                          ret.u = nsimd_scalar_reinterpret_u16_i16({in0});
                          return ret;'''.format(**fmtspec)
        return \
        '''#if defined(NSIMD_NATIVE_FP16) && defined(NSIMD_IS_GCC)
             {via_union}
           #elif (defined(NSIMD_NATIVE_FP16) && !defined(NSIMD_IS_GCC)) || \
                 defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) || \
                 defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
             {via_memcpy}
           #else
             {emulated}
           #endif'''.format(via_union=via_union, via_memcpy=via_memcpy,
                            emulated=emulated)
    return '''#ifdef NSIMD_IS_GCC
                {via_union}
              #else
                {via_memcpy}
              #endif'''.format(via_union=via_union, via_memcpy=via_memcpy)

# -----------------------------------------------------------------------------

def cvt(totyp, typ):
    if totyp == typ:
        return 'return {in0};'.format(**fmtspec)
    if typ == 'f16':
        return '''#ifdef NSIMD_NATIVE_FP16
                      return ({totyp}){in0};
                  #else
                      return ({totyp})nsimd_f16_to_f32({in0});
                  #endif'''.format(**fmtspec)
    if totyp == 'f16':
        return '''#ifdef NSIMD_NATIVE_FP16
                      return (f16){in0};
                  #else
                      return nsimd_f32_to_f16((f32){in0});
                  #endif'''.format(**fmtspec)
    return 'return ({totyp}){in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------

def adds(typ):
    if typ in common.ftypes:
        return opnum('{in0} + {in1}', typ)
    if typ in common.utypes:
        return '''{typ} tmp = ({typ})({in0} + {in1});
                  if (tmp < {in0} || tmp < {in1}) {{
                    return ({typ})-1;
                  }} else {{
                    return tmp;
                  }}
                  '''.format(**fmtspec)
    # Getting here means typ is signed
    int_max = 'NSIMD_' + typ.upper() + '_MAX'
    int_min = 'NSIMD_' + typ.upper() + '_MIN'
    return '''if (({in0} >= 0 && {in1} <= 0) || ({in0} <= 0 && {in1} >= 0)) {{
                return ({typ})({in0} + {in1});
              }} else {{
                if ({in0} > 0) {{
                  if ({in1} > {int_max} - {in0}) {{
                    return {int_max};
                  }} else {{
                    return ({typ})({in0} + {in1});
                  }}
                }} else {{
                  if ({in1} < {int_min} - {in0}) {{
                    return {int_min};
                  }} else {{
                    return ({typ})({in0} + {in1});
                  }}
                }}
              }}'''.format(int_min=int_min, int_max=int_max, **fmtspec)

# -----------------------------------------------------------------------------

def subs(typ):
    if typ in common.ftypes:
        return opnum('{in0} - {in1}', typ)
    if typ in common.utypes:
        return '''if ({in0} < {in1}) {{
                    return ({typ})0;
                  }} else {{
                    return ({typ})({in0} - {in1});
                  }}
                  '''.format(**fmtspec)
    # Getting here means typ is signed
    return 'return nsimd_scalar_adds_{typ}({in0}, ({typ})(-{in1}));'. \
           format(**fmtspec)

# -----------------------------------------------------------------------------

def get_impl(operator, totyp, typ):

    global fmtspec

    fmtspec = {
      'in0': common.in0,
      'in1': common.in1,
      'in2': common.in2,
      'typ': typ,
      'totyp': totyp,
      'typnbits': typ[1:]
    }

    if operator.name == 'trunc':
        if typ in common.iutypes:
            return 'return {in0};'.format(**fmtspec)
        elif typ == 'f16':
            c89_code = \
            '''f32 buf = nsimd_f16_to_f32({in0});
               return nsimd_f32_to_f16(buf >= 0.0f ?
                                       nsimd_scalar_floor_f32(buf) :
                                       nsimd_scalar_ceil_f32(buf));'''. \
                                       format(**fmtspec)
        else:
            c89_code = \
            '''return {in0} >= 0.0{f} ? nsimd_scalar_floor_{typ}({in0})
                      : nsimd_scalar_ceil_{typ}({in0});'''. \
                      format(f='f' if typ == 'f32' else '', **fmtspec)
        return libm_opn('trunc', 1, typ, True, c89_code)
    if operator.name == 'abs':
        if typ == 'f16':
            return '''f32 tmp = nsimd_f16_to_f32({in0});
                      return nsimd_f32_to_f16(tmp >= 0.0f ? tmp : -tmp);'''. \
                      format(**fmtspec)
        elif typ in common.utypes:
            return 'return {in0};'.format(**fmtspec)
        else:
            return 'return ({typ})({in0} >= ({typ})0 ? {in0} : -{in0});'. \
                   format(**fmtspec)
    if operator.name in ['min', 'max']:
        op = '<' if operator.name == 'min' else '>'
        if typ == 'f16':
            return '''f32 in0 = nsimd_f16_to_f32({in0});
                      f32 in1 = nsimd_f16_to_f32({in1});
                      return nsimd_f32_to_f16(in0 {op} in1 ? in0 : in1);'''. \
                      format(op=op, **fmtspec)
        else:
            return 'return {in0} {op} {in1} ? {in0} : {in1};'. \
                   format(op=op, **fmtspec)
    if operator.name == 'to_logical':
        if typ in common.iutypes:
            return 'return {in0} != ({typ})0;'.format(**fmtspec)
        else:
            return '''return nsimd_scalar_reinterpret_u{typnbits}_{typ}(
                               {in0}) != (u{typnbits})0;'''.format(**fmtspec)
    if operator.name == 'to_mask':
        if typ in common.utypes:
            return 'return ({typ})({in0} ? -1 : 0);'.format(**fmtspec)
        else:
            return '''return nsimd_scalar_reinterpret_{typ}_u{typnbits}((
                                 u{typnbits})({in0} ? -1 : 0));'''. \
                                 format(**fmtspec)
    if operator.name == 'round_to_even':
        return round_to_even(typ)
    if operator.name in ['floor', 'ceil', 'sqrt']:
        if typ in common.iutypes and operator.name != 'sqrt':
            return 'return {in0};'.format(**fmtspec)
        return libm_opn(operator.name, 1, typ, False, '')
    if operator.name == 'fma':
        if typ in common.iutypes:
            return 'return ({typ})({in0} * {in1} + {in2});'.format(**fmtspec)
        else:
            if typ == 'f16':
                c89_code = 'return nsimd_f32_to_f16(nsimd_f16_to_f32({in0}) ' \
                           '* nsimd_f16_to_f32({in1}) ' \
                           '+ nsimd_f16_to_f32({in2}));'.format(**fmtspec)
            else:
                c89_code = 'return {in0} * {in1} + {in2};'.format(**fmtspec)
            return libm_opn(operator.name, 3, typ, False, c89_code)
    if operator.name in ['fnma', 'fms', 'fnms']:
        neg = '-' if operator.name in ['fnms', 'fnma'] else ''
        op = '-' if operator.name in ['fms', 'fnms'] else '+'
        if typ in common.iutypes:
            return 'return ({typ})(({neg}{in0}) * {in1} {op} {in2});'. \
                   format(neg=neg, op=op, **fmtspec)
        else:
            typ2 = 'f32' if typ == 'f16' else typ
            return opnum(
            'nsimd_scalar_fma_{typ2}({neg}{{in0}}, {{in1}}, {op}{{in2}})'. \
            format(typ2=typ2, neg=neg, op=op, **fmtspec), typ)
    f = 'f' if typ in ['f16', 'f32'] else ''
    typ2 = 'f32' if typ == 'f16' else typ
    func = {
        'orb': lambda: opbit('{in0} | {in1}', typ),
        'andb': lambda: opbit('{in0} & {in1}', typ),
        'andnotb': lambda: opbit('{in0} & (~{in1})', typ),
        'notb': lambda: opbit('~{in0}', typ),
        'xorb': lambda: opbit('{in0} ^ {in1}', typ),
        'add': lambda: opnum('{in0} + {in1}', typ),
        'sub': lambda: opnum('{in0} - {in1}', typ),
        'mul': lambda: opnum('{in0} * {in1}', typ),
        'div': lambda: opnum('{in0} / {in1}', typ),
        'neg': lambda: opnum('-{in0}', typ),
        'lt': lambda: cmp('{in0} < {in1}', typ),
        'gt': lambda: cmp('{in0} > {in1}', typ),
        'le': lambda: cmp('{in0} <= {in1}', typ),
        'ge': lambda: cmp('{in0} >= {in1}', typ),
        'ne': lambda: cmp('{in0} != {in1}', typ),
        'eq': lambda: cmp('{in0} == {in1}', typ),
        'andl': lambda: 'return {in0} && {in1};'.format(**fmtspec),
        'orl': lambda: 'return {in0} || {in1};'.format(**fmtspec),
        'xorl': lambda: 'return {in0} ^ {in1};'.format(**fmtspec),
        'andnotl': lambda: 'return {in0} && (!{in1});'.format(**fmtspec),
        'notl': lambda: 'return !{in0};'.format(**fmtspec),
        'shl': lambda: shift('shl', typ),
        'shr': lambda: shift('shr', typ),
        'shra': lambda: shift('shra', typ),
        'reinterpret': lambda: reinterpret(totyp, typ),
        'cvt': lambda: cvt(totyp, typ),
        'adds': lambda: adds(typ),
        'subs': lambda: subs(typ),
        'rec': lambda: opnum('1.0{f} / {{in0}}'.format(f=f), typ),
        'rec8': lambda: opnum('1.0{f} / {{in0}}'.format(f=f), typ),
        'rec11': lambda: opnum('1.0{f} / {{in0}}'.format(f=f), typ),
        'rsqrt': lambda:
                 opnum('1.0{f} / nsimd_scalar_sqrt_{typ2}({{in0}})'. \
                 format(f=f, typ2=typ2), typ),
        'rsqrt8': lambda:
                  opnum('1.0{f} / nsimd_scalar_sqrt_{typ2}({{in0}})'. \
                  format(f=f, typ2=typ2), typ),
        'rsqrt11': lambda:
                   opnum('1.0{f} / nsimd_scalar_sqrt_{typ2}({{in0}})'. \
                   format(f=f, typ2=typ2), typ)
    }
    return func[operator.name]()

