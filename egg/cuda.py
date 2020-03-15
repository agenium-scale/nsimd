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
# NVIDIA doc on f16 can be found at
# https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html

def get_impl_f16(operator, totyp, typ):
    if operator.name == 'round_to_even':
        return 'return __hrint({in0}, {in1});'.format(**fmtspec)
    elif operator.name in ['rec', 'rec8', 'rec11']:
        return 'return __hrcp({in0});'.format(**fmtspec)
    elif operator.name in ['rsqrt', 'rsqrt8', 'rsqrt11']:
        return 'return __hrsqrt({in0});'.format(**fmtspec)
    elif operator.name in ['fma', 'fms', 'fnma', 'fnms']:
        neg = '-' if operator.name in ['fnma, fnms'] else ''
        op = '-' if operator.name in ['fnms, fms'] else ''
        return 'return __hfma({neg}{in0}, {in1}, {op}{in2});'. \
               format(neg=neg, op=op, **fmtspec)
    elif operator.name in ['min', 'max']:
        intr = '__hlt' if operator.name == 'min' else '__hgt'
        return '''if ({intr}) {{
                    return {in0};
                  }} else {{
                    return {in1};
                  }}'''.format(intr=intr, **fmtspec)
    else:
        return 'return __h{}({in0}, {in1});'. \
               format(operator.name, **fmtspec)

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
    # infix operators that needs type punning
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
    # for all other operators, f16 has a special treatment
    if typ == 'f16':
        return get_impl_f16(operator, totyp, typ)
    # then deal with f32's operators
    # first infix operators
    c_operators = {
        'add': 'return {in0} + {in1};',
        'sub': 'return {in0} - {in1};',
        'mul': 'return {in0} * {in1};',
        'div': 'return {in0} / {in1};',
        'neg': 'return -{in0};',
        'rec': 'return 1.0{f} / {{in0}};',
        'rec8': 'return 1.0{f} / {{in0}};',
        'rec11': 'return 1.0{f} / {{in0}};',
        'lt': 'return {in0} < {in1};',
        'gt': 'return {in0} > {in1};',
        'le': 'return {in0} <= {in1};',
        'ge': 'return {in0} >= {in1};',
        'ne': 'return {in0} != {in1};',
        'eq': 'return {in0} == {in1};'
    }
    if operator.name in c_operators:
        return c_operators[operator.name]. \
               format(f='f' if typ == 'f32' else '', **fmtspec)
    # fma's
    if operator.name in ['fma', 'fms', 'fnma', 'fnms']:
        neg = '-' if operator.name in ['fnma, fnms'] else ''
        op = '-' if operator.name in ['fnms, fms'] else ''
        return 'return fma{f}({neg}{in0}, {in1}, {op}{in2});'. \
               format(f='f' if typ == 'f32' else '', neg=neg, op=op, **fmtspec)
    # other operators
    cuda_name = {
        'round_to_even': 'rint',
        'min': 'fmin',
        'max': 'fmax',
        'abs': 'fabs'
    }
    args = ', '.join(['{{in{}}}'.format(i).format(**fmtspec) \
                      for i in range(len(operator.args))])
    return 'return {name}{f}({args});'. \
           format(name=cuda_name[operator.name] \
                  if operator.name in cuda_name else operator.name,
                  f='f' if typ == 'f32' else '', args=args)
