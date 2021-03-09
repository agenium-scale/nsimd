
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

# Reference: book Data Parallel C++
# Table page 475: list of maths functions. float16 supported

import common
import scalar

fmtspec = dict()

# -----------------------------------------------------------------------------

def get_impl_f16(operator, totyp, typ):
    # TODO: what's the difference between rec and rec8,rec11?

    # Case 1: operators do not support sycl half types
    # do not use nsimd casts. Use nsimd_scalar_functions

    op_f16_use_nsimd_scalar = ['round_to_even', 'ceil', 'floor',
                              'trunc','rec8','rec11','rsqrt8',
                              'rsqrt11']

    # Case 2: operators do not support sycl half types
    # use nsimd casts, then sycl provided functions

    op_f16_cvrt_then_use_sycl_f32_op = ['fma', 'fms', 'fnma', 'fnms',
                                       'fmin', 'fmax']

    # Case 3: sycl provides functions supporting half type

    op_f16_half_type_sycl = ['rec']

    # Dispatch

    # Case 1
    if operator.name in op_f16_use_nsimd_scalar:
      code = 'return nsimd_scalar_{op}_f16({in0});'.\
      format(op=operator.name,**fmtspec)

    # Case 2
    elif operator.name in op_f16_cvrt_then_use_sycl_f32_op:
      if operator.name in ['fma', 'fms', 'fnma', 'fnms']:
        neg = '-' if operator.name in ['fnma, fnms'] else ''
        op = '-' if operator.name in ['fnms, fms'] else ''
        code = \
        '''f32 in0 = nsimd_f16_to_f32({in0});
	   f32 in1 = nsimd_f16_to_f32({in1});
	   f32 in2 = nsimd_f16_to_f32({in2});
           f32 res = sycl::fma({neg}{in0}, {in1}, {op}{in2});
           return nsimd_f32_to_f16(res);
        '''.format(neg=neg, op=op, **fmtspec)
      if operator.name in ['min','max']:
        op = 'fmin' if operator.name == 'min' else 'fmax'
        '''f32 in0 = nsimd_f16_to_f32({in0});
	   f32 in1 = nsimd_f16_to_f32({in1});
           f32 res = sycl::{op}({in0}, {in1});
           return nsimd_f32_to_f16(res);
        '''.format(op=op, **fmtspec)

    # Case 3
    elif operator.name in op_f16_half_type_sycl:
      if operator.name == 'rec':
        code = 'return sycl::recip({in0});'.format(**fmtspec)

    # ------------ below placeholder from CUDA (to update to oneAPI)-----------

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

# ------------------------ below draft to be updated ------------------

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

  # Duplicated code with cuda (bool_operators, infix_operators)
  # TODO: factorize

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

  if operator.name == 'trunc':
   return 'return sycl::trunc({in0});'.format(**fmtspec)

  if operator.name in ['min', 'max']:
   op = 'sycl::fmin' if operator.name == 'min' else 'sycl::fmax'
   return 'return {op}({in0}, {in1});'.format(op=op, **fmtspec)

  if operator.name in ['floor', 'ceil', 'sqrt']:
    return 'return sycl::{op}({in0});'.format(op=operator.name, **fmtspec)




