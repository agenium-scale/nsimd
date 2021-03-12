
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

# -----------------------------------------------------------------------------
# References:

# Functions: book:
# Data Parallel C++
# Mastering DPC++ for Programming of Heterogeneous Systems using
# C++ and SYCL - Apress Open
# Table page 475: list of maths functions. float16 supported

# sycl half type (f16) API:
# https://mmha.github.io/syclreference/libraries/types/half/
# -----------------------------------------------------------------------------

import common
import scalar

fmtspec = dict()

# -----------------------------------------------------------------------------

def get_impl_f16(operator, totyp, typ):

    # Case 1: no sycl function available for half type
    # sycl function available for f32
    # however do not use nsimd casts f32 to f16 with rounding functions
    # use nsimd_scalar_[operator]_f16

    no_sycl_avail_f16_rounding_use_nsimd_scalar = \
      ['round_to_even', 'ceil', 'floor', 'trunc']

    # Case 2: no sycl function available for half type
    # sycl function available for f32
    # use nsimd casts f32-->f16 + sycl function + f16-->f32

    # TODO: hould we use sycl::add_sat, sycl::add_sub
    # both available for f32?
    no_sycl_avail_f16_cast_use_sycl_f32 = \
      ['fma', 'fms', 'fnma', 'fnms', 'min', 'max', 'abs']

    # Case 3: sycl provides functions supporting half type

    sycl_avail_functions_f16 = \
      ['rec8', 'rec11', 'rec', 'rsqrt8', 'rsqrt11', 'rsqrt', 'sqrt']

    # Case 4: sycl half's type provided comparison operators
    # Note:
    # not documented in the book
    # source: sycl half type (f16) API:
    # https://mmha.github.io/syclreference/libraries/types/half/

    sycl_avail_cmp_op_f16 = {
        'lt': 'return {in0} < {in1};',
        'gt': 'return {in0} > {in1};',
        'le': 'return {in0} <= {in1};',
        'ge': 'return {in0} >= {in1};',
        'ne': 'return {in0} != {in1};',
        'eq': 'return {in0} == {in1};'
    }

    # Case 5: no sycl function available for any type
    # use nsimd_scalar_[operator]_f16

    # Dispatch

    # Case 1
    if operator.name in no_sycl_avail_f16_rounding_use_nsimd_scalar:
      return 'return nsimd_scalar_{op}_f16({in0});'.\
        format(op=operator.name,**fmtspec)

    # Case 2
    elif operator.name in no_sycl_avail_f16_cast_use_sycl_f32:
      if operator.name in ['fma', 'fms', 'fnma', 'fnms']:
        neg = '-' if operator.name in ['fnma, fnms'] else ''
        op = '-' if operator.name in ['fnms, fms'] else ''
        return '''f32 x0 = nsimd_f16_to_f32({in0});
	          f32 x1 = nsimd_f16_to_f32({in1});
	          f32 x2 = nsimd_f16_to_f32({in2});
                  f32 res = sycl::fma({neg}x0, x1, {op}x2);
                  return nsimd_f32_to_f16(res);
                  '''.format(neg=neg, op=op, **fmtspec)
      elif operator.name in ['min','max']:
        op = 'fmin' if operator.name == 'min' else 'fmax'
        return '''f32 x0 = nsimd_f16_to_f32({in0});
	          f32 x1 = nsimd_f16_to_f32({in1});
                  f32 res = sycl::{op}(x0, x1);
                  return nsimd_f32_to_f16(res);
                  '''.format(op=op, **fmtspec)
      elif operator.name == 'abs':
        return '''f32 x0 = nsimd_f16_to_f32({in0});
                  f32 res = sycl::fabs(x0);
                  return nsimd_f32_to_f16(res);
                  '''.format(**fmtspec)

    # Case 3
    elif operator.name in sycl_avail_functions_f16:
      if operator.name in ['rec8', 'rec11', 'rec']:
        return 'return sycl::recip({in0});'.format(**fmtspec)
      elif operator.name in ['rsqrt8', 'rsqrt11', 'rsqrt']:
        return 'return sycl::rsqrt({in0});'.format(**fmtspec)
      elif operator.name == 'sqrt':
        return 'return sycl::sqrt({in0});'.format(**fmtspec)

    # Case 4
    elif operator.name in sycl_avail_cmp_op_f16:
      return sycl_avail_cmp_op_f16[operator.name].\
        format(**fmtspec)

    # Case 5
    else:
      args = ', '.join(['{{in{}}}'.format(i).format(**fmtspec) \
                          for i in range(len(operator.params[1:]))])

      return 'return nsimd_scalar_{op}_f16({args});'.\
        format(op=operator.name, args=args)

# -----------------------------------------------------------------------------

def reinterpret(totyp, typ):
    if typ == totyp:
      return 'return {in0};'.format(**fmtspec)
    elif ((typ in common.ftypes and totyp in common.iutypes) or \
    (typ in common.iutypes and totyp in common.ftypes)):
      return 'return nsimd_scalar_reinterpret_{totyp}_{typ}({in0});'. \
           format(totyp=totyp, typ=typ, **fmtspec)
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

  # bool first, no special treatment for f16's
  bool_operators = [ 'andl', 'orl', 'xorl', 'andnotl', 'notl' ]
  if operator.name in bool_operators:
    return 'return nsimd_scalar_{op}({in0},{in1})'.\
            format(op=operator.name,**fmtspec)

  # infix operators no special treatment for f16's
  infix_operators = [ 'orb', 'andb', 'andnotb', 'notb', 'xorb' ]
  if operator.name in infix_operators:
    return 'return nsimd_scalar_{op}_{typ}({in0},{in1})'.\
            format(op=operator.name,**fmtspec)

  # reinterpret
  if operator.name == 'reinterpret':
    return reinterpret(totyp, typ)

  # cvt
  if operator.name == 'cvt':
    return 'return nsimd_scalar_cvt_{totyp}_{typ}({in0});'.format(**fmtspec)

  # to_mask
  if operator.name == 'to_mask':
    return 'nsimd_scalar_to_mask_{totyp}({in0});'.format(**fmtspec)

  # to_logical
  if operator.name == 'to_logical':
    return 'nsimd_scalar_to_logical_{typ}({in0});'.format(**fmtspec)

  # for all other operators, f16 has a special treatment
  if typ == 'f16':
      return get_impl_f16(operator, totyp, typ)

  # infix operators - rec - f32, f64
  infix_op_rec_f32_f64 = {
      'rec': 'return sycl::recip({in0});',
      'rec8': 'return sycl::recip({in0});',
      'rec11': 'return sycl::recip({in0});'
      }

  if typ in ['f32','f64'] and operator.name in infix_op_rec_f32_f64:
    return infix_op_rec_f32_f64[operator.name].\
     format(**fmtspec)

  # infix operators - cmp - f32, f64
  infix_op_cmp_f32_f64 = {
      'lt': 'return {cast_to_int}sycl::isless({in0},{in1});',
      'gt': 'return {cast_to_int}sycl::isgreater({in0},{in1});',
      'le': 'return {cast_to_int}sycl::islessequal({in0},{in1});',
      'ge': 'return {cast_to_int}sycl::isgreaterequal({in0},{in1});',
      'ne': 'return {cast_to_int}sycl::isnotequal({in0},{in1});',
      'eq': 'return {cast_to_int}sycl::isequal({in0},{in1});'
  }

  if typ in ['f32','f64'] and operator.name in infix_op_cmp_f32_f64:
    return infix_op_cmp_f32_f64[operator.name].\
     format(cast_to_int='(int)' if typ == 'f64' else '', **fmtspec)

  # infix operators - cmp - integer types
  infix_op_cmp_iutypes = [ 'lt', 'gt', 'le', 'ge', 'ne', 'eq' ]
  if operator.name in infix_op_cmp_iutypes:
    return 'return nsimd_scalar_{op}_{typ}({in0},{in1});'.\
      format(op=operator.name, **fmtspec)

  # infix operators f32, f64 + integers
  # ref: see Data Parallel C++ book, pages 480, 481, 482
  # TODO: do the functions below call instrinsics/built-in
  # functions on the device?
  # 'add': 'return std::plus<{typ}>()({in0}, {in1});',
  # 'sub': 'return std::minus<{typ}>()({in0}, {in1});',
  # 'mul': 'return std::multiplies<{typ}>()({in0}, {in1});',
  # 'div': 'return std::divides<{typ}>()({in0}, {in1});',

  infix_op_t = [ 'add', 'sub', 'mul', 'div' ]
  if operator.name in infix_op_t:
    return 'return nsimd_scalar_{op}_{typ}({in0},{in1})'.\
      format(op=operator.name, **fmtspec)

  # neg
  # ref: see Data Parallel C++ book, pages 480, 481, 482
  # TODO: does the function below call an instrinsic/built-in
  # function on the device?
  # 'neg': 'return std::negate<{typ}>()({in0});'

  if operator.name == 'neg':
    return 'return nsimd_scalar_{op}_{typ}({in0})'.\
      format(op=operator.name, **fmtspec)

  # shifts
  shifts_op_ui_t = [ 'shl', 'shr', 'shra' ]
  if operator.name in shifts_op_ui_t and typ in common.iutypes:
    return 'return nsimd_scalar_{op}_{typ}({in0}, {in1});'.\
      format(op=operator.name, **fmtspec)

  # adds
  if operator.name == 'adds':
    if typ in common.ftypes:
      return infix_op_t['add'].\
        format(**fmtspec)
    else:
      return 'return sycl::add_sat({in0},{in1});'.format(**fmtspec)

  # subs
  if operator.name == 'subs':
    if typ in common.ftypes:
      return infix_op_t['sub'].\
        format(**fmtspec)
    else:
      return 'return sycl::sub_sat({in0},{in1});'.format(**fmtspec)

  # fma's
  if operator.name in ['fma', 'fms', 'fnma', 'fnms']:
    if typ in common.ftypes:
      neg = '-' if operator.name in ['fnma, fnms'] else ''
      op = '-' if operator.name in ['fnms, fms'] else ''
      return 'return sycl::fma({neg}{in0}, {in1}, {op}{in2});'.\
        format(op=op, neg=neg, **fmtspec)
    else:
      return 'return nsimd_scalar_{op}_{typ}({in0}, {in1}, {in2})'.\
        format(op=operator.name, **fmtspec)

  # other operators
  # round_to_even, ceil, floor, trunc, min, max, abs, sqrt

  # round_to_even
  if operator.name == 'round_to_even':
    return 'return nsimd_scalar_round_to_even_{typ}({in0})'.\
      format(**fmtspec)

  # other rounding operators
  other_rounding_ops = ['ceil', 'floor', 'trunc']
  if operator.name in other_rounding_ops:
    if typ in common.iutypes:
      return 'return nsimd_scalar_{op}_{typ}({in0})'.\
        format(op=operator.name, **fmtspec)
    else:
      return 'sycl::{op}({in0})'.format(op=operator.name, **fmtspec)

  # min/max
  if operator.name in ['min','max']:
    if typ in common.iutypes:
      return 'return sycl::{op}({in0},{in1})'.\
        format(op=operator.name, **fmtspec)
    else:
      op = 'sycl::fmin' if operator.name == 'min' else 'sycl::fmax'
      return 'return {op}({in0}, {in1});'.format(op=op, **fmtspec)

  # abs
  if operator.name == 'abs':
    if typ in common.itypes:
      return 'return ({typ})sycl::abs({in0})'.\
        format(**fmtspec)
    elif typ in common.utypes:
      return 'return nsimd_scalar_abs_{typ}({in0})'.format(**fmtspec)
    else:
      return 'return sycl::fabs({in0})'.format(**fmtspec)

  # sqrt
  if operator.name == 'sqrt':
    if typ in common.ftypes:
      return 'return sycl::sqrt({in0})'.format(**fmtspec)

  # rsqrt
  if operator.name in ['rsqrt8', 'rsqrt11', 'rsqrt']:
    if typ in common.ftypes:
      return 'return sycl::rsqrt({in0});'.format(**fmtspec)

