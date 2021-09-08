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
import scalar

fmtspec = dict()

# -----------------------------------------------------------------------------
# NVIDIA doc on f16 can be found at
# https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html

def get_impl_f16(operator, totyp, typ):
    if operator.name == 'round_to_even':
        arch53_code = 'return hrint({in0});'.format(**fmtspec)
    elif operator.name in ['rec', 'rec8', 'rec11']:
        arch53_code = 'return hrcp({in0});'.format(**fmtspec)
    elif operator.name in ['rsqrt8', 'rsqrt11']:
        arch53_code = 'return hrsqrt({in0});'.format(**fmtspec)
    elif operator.name in ['fma', 'fms', 'fnma', 'fnms']:
        neg = '-' if operator.name in ['fnma, fnms'] else ''
        op = '-' if operator.name in ['fnms, fms'] else ''
        arch53_code = 'return __hfma({neg}{in0}, {in1}, {op}{in2});'. \
                      format(neg=neg, op=op, **fmtspec)
    elif operator.name in ['min', 'max']:
        intr = '__hlt' if operator.name == 'min' else '__hgt'
        arch53_code = '''if ({intr}) {{
                           return {in0};
                         }} else {{
                           return {in1};
                         }}'''.format(intr=intr, **fmtspec)
    elif operator.name in ['adds', 'subs']:
        arch53_code = 'return __h{op}({in0}, {in1});'. \
                      format(op=operator.name[:-1], **fmtspec)
    else:
        args = ', '.join(['{{in{}}}'.format(i).format(**fmtspec) \
                          for i in range(len(operator.params[1:]))])
        # Some f16 functions are not prefixed by `__`
        not_prefixed = ['ceil', 'floor', 'trunc', 'sqrt']
        if operator.name in not_prefixed:
            arch53_code = 'return h{}({});'.format(operator.name, args)
        else:
            arch53_code = 'return __h{}({});'.format(operator.name, args)
    args = ', '.join(['__half2float({{in{}}})'.format(i).format(**fmtspec) \
                      for i in range(len(operator.params[1:]))])
    if operator.params[0] == 'l':
        emul = 'return gpu_{}({});'.format(operator.name, args)
    else:
        emul = 'return __float2half(gpu_{}({}));'.format(operator.name, args)
    return '''#if __CUDA_ARCH__ >= 530
                {arch53_code}
              #else
                {emul}
              #endif'''.format(arch53_code=arch53_code, emul=emul)

# -----------------------------------------------------------------------------
# Reinterprets on CUDA have intrinsics

def reinterpret(totyp, typ):
    if typ == totyp:
        return 'return {in0};'.format(**fmtspec)
    cuda_typ = { 'i16': 'short',
                 'u16': 'ushort',
                 'f16': 'half',
                 'i32': 'int',
                 'u32': 'uint',
                 'f32': 'float',
                 'f64': 'double',
                 'i64': 'longlong' }
    if typ in cuda_typ and totyp in cuda_typ and \
       ((typ in common.ftypes and totyp in common.iutypes) or \
        (typ in common.iutypes and totyp in common.ftypes)):
        return 'return __{typ2}_as_{totyp2}({in0});'. \
               format(typ2=cuda_typ[typ], totyp2=cuda_typ[totyp], **fmtspec)
    else:
        return '''{totyp} ret;
                  memcpy((void *)&ret, (void *)&{in0}, sizeof({in0}));
                  return ret;'''.format(**fmtspec)

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

    # src operators
    if operator.src:
        cuda_ops = {
          'sin_u35': 'sin',
          'cos_u35': 'cos',
          'tan_u35': 'tan',
          'asin_u35': 'asin',
          'acos_u35': 'acos',
          'atan_u35': 'atan',
          'atan2_u35': 'atan2',
          'log_u35': 'log',
          'cbrt_u35': 'cbrt',
          'sin_u10': 'sin',
          'cos_u10': 'cos',
          'tan_u10': 'tan',
          'asin_u10': 'asin',
          'acos_u10': 'acos',
          'atan_u10': 'atan',
          'atan2_u10': 'atan2',
          'log_u10': 'log',
          'cbrt_u10': 'cbrt',
          'exp_u10': 'exp',
          'pow_u10': 'pow',
          'sinh_u10': 'sinh',
          'cosh_u10': 'cosh',
          'tanh_u10': 'tanh',
          'sinh_u35': 'sinh',
          'cosh_u35': 'cosh',
          'tanh_u35': 'tanh',
          'asinh_u10': 'asinh',
          'acosh_u10': 'acosh',
          'atanh_u10': 'atanh',
          'exp2_u10': 'exp2',
          'exp2_u35': 'exp2',
          'exp10_u10': 'exp10',
          'exp10_u35': 'exp10',
          'expm1_u10': 'expm1',
          'log10_u10': 'log10',
          'log2_u10': 'log2',
          'log2_u35': 'log2',
          'log1p_u10': 'log1p',
          'sinpi_u05': 'sinpi',
          'cospi_u05': 'cospi',
          'hypot_u05': 'hypot',
          'hypot_u35': 'hypot',
          'remainder': 'remainder',
          'fmod': 'fmod',
          'lgamma_u10': 'lgamma',
          'tgamma_u10': 'tgamma',
          'erf_u10': 'erf',
          'erfc_u15': 'erfc'
        }
        args = common.get_args(len(operator.params[1:]))
        cuda_op = cuda_ops[operator.name]
        if typ == 'f16':
            # For f16 CUDA offers only a few operator
            if cuda_op in ['cos', 'exp', 'exp10', 'exp2', 'log', 'log10',
                           'log2', 'sin']:
                return '''#if __CUDA_ARCH__ >= 530
                            return h{}({});
                          #else
                            return __float2half(gpu_{}(__half2float({})));
                          #endif'''.format(cuda_op, args, operator.name, args)
            else:
                args = ', '.join('__half2float({})'.format(common.get_arg(i)) \
                                 for i in range(len(operator.params[1:])))
                return 'return __float2half(gpu_{}({}));'. \
                       format(operator.name, args)
        elif typ == 'f32':
            return 'return {}f({});'.format(cuda_op, args)
        else:
            return 'return {}({});'.format(cuda_op, args)

    # bool first, no special treatment for f16's
    bool_operators = {
        'andl': 'return {in0} && {in1};',
        'orl': 'return {in0} || {in1};',
        'xorl': 'return {in0} ^ {in1};',
        'andnotl': 'return {in0} && (!{in1});',
        'notl': 'return !{in0};',
    }
    if operator.name in bool_operators:
        return bool_operators[operator.name].format(**fmtspec)
    # infix operators that needs type punning, no special treatment for f16's
    def pun_code(code, arity, typ):
        if typ in common.utypes:
            return 'return ' + code.format(**fmtspec) + ';'
        utyp = common.bitfield_type[typ]
        to_utyp = '\n'.join(
                  ['''{utyp} buf{i};
                      memcpy(&buf{i}, &{{in{i}}}, sizeof({{in{i}}}));'''. \
                      format(i=i, utyp=utyp).format(**fmtspec) \
                      for i in range(arity)])
        return '''{to_utyp}
                  {utyp} tmp = {code};
                  {typ} ret;
                  memcpy(&ret, &tmp, sizeof(tmp));
                  return ret;'''.format(to_utyp=to_utyp, utyp=utyp, typ=typ,
                                        code=code.format(in0='buf0',
                                                         in1='buf1'))
    pun_operators = {
        'orb': lambda: pun_code('{in0} | {in1}', 2, typ),
        'andb': lambda: pun_code('{in0} & {in1}', 2, typ),
        'andnotb': lambda: pun_code('{in0} & (~{in1})', 2, typ),
        'notb': lambda: pun_code('~{in0}', 1, typ),
        'xorb': lambda: pun_code('{in0} ^ {in1}', 2, typ),
    }
    if operator.name in pun_operators:
        return pun_operators[operator.name]()
    # reinterpret
    if operator.name == 'reinterpret':
        return reinterpret(totyp, typ)
    # cvt
    if operator.name == 'cvt':
        return 'return ({totyp}){in0};'.format(**fmtspec)
    # to_mask
    if operator.name == 'to_mask':
        if typ in common.utypes:
            return 'return ({typ})({in0} ? -1 : 0);'.format(**fmtspec)
        return 'return gpu_reinterpret({typ}(), ({utyp})({in0} ? -1 : 0));'. \
               format(utyp=common.bitfield_type[typ], **fmtspec)
    # to_logical
    if operator.name == 'to_logical':
        if typ in common.iutypes:
            return 'return {in0} == ({typ})0 ? false : true;'.format(**fmtspec)
        return '''return gpu_reinterpret({utyp}(), {in0}) == ({utyp})0
                         ? false : true ;'''. \
                         format(utyp=common.bitfield_type[typ], **fmtspec)
    # for all other operators, f16 has a special treatment
    if typ == 'f16':
        return get_impl_f16(operator, totyp, typ)
    # then deal with f32's operators
    # first infix operators
    c_operators = {
        'add': 'return ({typ})({in0} + {in1});',
        'sub': 'return ({typ})({in0} - {in1});',
        'mul': 'return ({typ})({in0} * {in1});',
        'div': 'return ({typ})({in0} / {in1});',
        'neg': 'return ({typ})(-{in0});',
        'rec': 'return 1.0{f} / {in0};',
        'rec8': 'return 1.0{f} / {in0};',
        'rec11': 'return 1.0{f} / {in0};',
        'lt': 'return {in0} < {in1};',
        'gt': 'return {in0} > {in1};',
        'le': 'return {in0} <= {in1};',
        'ge': 'return {in0} >= {in1};',
        'ne': 'return {in0} != {in1};',
        'eq': 'return {in0} == {in1};',
        'shl': 'return ({typ})({in0} << {in1});',
    }
    if operator.name in c_operators:
        return c_operators[operator.name]. \
               format(f='f' if typ == 'f32' else '', **fmtspec)
    # right shifts
    if operator.name in ['shr', 'shra']:
        if typ in common.utypes:
            return 'return ({typ})({in0} >> {in1});'.format(**fmtspec)
        if operator.name == 'shr':
            return \
            '''return gpu_reinterpret({typ}(), ({utyp})(
                          gpu_reinterpret({utyp}(), {in0}) >> {in1}));'''. \
                          format(utyp=common.bitfield_type[typ], **fmtspec)
        # getting here means shra on signed types
        return \
        '''if ({in1} == 0) {{
             return {in0};
           }}
           if ({in0} >= 0) {{
             return gpu_reinterpret({typ}(), ({utyp})(
                        gpu_reinterpret({utyp}(), {in0}) >> {in1}));
           }} else {{
             {utyp} mask = ({utyp})((({utyp})-1) << ({typnbits} - {in1}));
             return gpu_reinterpret({typ}(), (({utyp})(mask |
                      ({utyp})(gpu_reinterpret({utyp}(), {in0}) >> {in1}))));
           }}'''.format(utyp=common.bitfield_type[typ], **fmtspec)
    # adds
    if operator.name == 'adds':
        if typ in common.ftypes:
            return c_operators['add'].format(**fmtspec)
        else:
            return scalar.get_impl(operator, totyp, typ)
    # subs
    if operator.name == 'subs':
        if typ in common.ftypes:
            return c_operators['sub'].format(**fmtspec)
        elif typ in common.utypes:
            return scalar.get_impl(operator, totyp, typ)
        else:
            return 'return nsimd::gpu_adds({in0}, ({typ})(-{in1}));'. \
                   format(**fmtspec)
    # fma's
    if operator.name in ['fma', 'fms', 'fnma', 'fnms']:
        neg = '-' if operator.name in ['fnma, fnms'] else ''
        op = '-' if operator.name in ['fnms, fms'] else ''
        if typ in common.ftypes:
            return 'return fma{f}({neg}{in0}, {in1}, {op}{in2});'. \
                   format(f='f' if typ == 'f32' else '', neg=neg, op=op,
                          **fmtspec)
        else:
            return 'return {neg}{in0} * {in1} + ({op}{in2});'. \
                   format(neg=neg, op=op, **fmtspec)
    # other operators
    if typ in common.iutypes:
        if operator.name in ['round_to_even', 'ceil', 'floor', 'trunc']:
            return 'return {in0};'.format(**fmtspec)
        elif operator.name == 'min':
            return 'return ({typ})({in0} < {in1} ? {in0} : {in1});'. \
                   format(**fmtspec)
        elif operator.name == 'max':
            return 'return ({typ})({in0} > {in1} ? {in0} : {in1});'. \
                   format(**fmtspec)
        elif operator.name == 'abs':
            return 'return ({typ})({in0} > 0 ? {in0} : -{in0});'. \
                   format(**fmtspec)
    else:
        cuda_name = {
            'round_to_even': 'rint',
            'min': 'fmin',
            'max': 'fmax',
            'abs': 'fabs',
            'ceil': 'ceil',
            'floor': 'floor',
            'trunc': 'trunc',
            'rsqrt8': 'rsqrt',
            'rsqrt11': 'rsqrt'
        }
        args = ', '.join(['{{in{}}}'.format(i).format(**fmtspec) \
                          for i in range(len(operator.args))])
        return 'return {name}{f}({args});'. \
               format(name=cuda_name[operator.name] \
                      if operator.name in cuda_name else operator.name,
                      f='f' if typ == 'f32' else '', args=args)
