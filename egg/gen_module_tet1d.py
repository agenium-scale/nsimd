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
import operators
import common

def doit(opts):
    print('-- Generating module tet1d')

    scalar_impl = {}
    scalar_impl['orb'] = 'a0 | a1'
    scalar_impl['andb'] = 'a0 & a1'
    scalar_impl['andnotb'] = 'a0 & ~a1'
    scalar_impl['notb'] = '~a'
    scalar_impl['xorb'] = 'a0 ^ a1'
    scalar_impl['add'] = 'a0 + a1'
    scalar_impl['sub'] = 'a0 - a1'
    scalar_impl['mul'] = 'a0 * a1'
    scalar_impl['div'] = 'a0 / a1'
    scalar_impl['neg'] = '-a'
    scalar_impl['min'] = 'std::min(a0, a1)'
    scalar_impl['max'] = 'std::max(a0, a1)'
    scalar_impl['abs'] = 'std::abs(a)'
    scalar_impl['fma'] = '(a0 * a1) + a2' # FIXME
    scalar_impl['fnma'] = '(a0 * a1) + a2' # FIXME
    scalar_impl['fms'] = '(a0 * a1) + a2' # FIXME
    scalar_impl['fnms'] = '(a0 * a1) + a2' # FIXME
    scalar_impl['ceil'] = 'std::ceil(a)'
    scalar_impl['floor'] = 'std::floor(a)'
    scalar_impl['trunc'] = 'std::trunc(a)'
    scalar_impl['round_to_even'] = 'std::round(a)' # FIXME
    scalar_impl['reinterpret'] = 'a' # FIXME
    scalar_impl['cvt'] = 'a' # FIXME
    scalar_impl['rec'] = 'a' # FIXME
    scalar_impl['rec11'] = 'a' # FIXME
    scalar_impl['rec8'] = 'a' # FIXME
    scalar_impl['sqrt'] = 'sqrt(a)'
    scalar_impl['rsqrt'] = 'sqrt(a)' # FIXME
    scalar_impl['rsqrt11'] = 'sqrt(a)' # FIXME
    scalar_impl['rsqrt8'] = 'sqrt(a)' # FIXME

    code = ''
    code += '''namespace tet1d
               {

            '''
    for op_name, operator in operators.operators.items():
      print('-- ', op_name, operator.signature)
      if op_name == 'downcvt': continue
      if op_name == 'ziplo': continue
      if op_name == 'ziphi': continue
      if op_name == 'unziplo': continue
      if op_name == 'unziphi': continue
      if (operator.params == ['v', 'v']):
        code += '''// {op_name}

                   struct {op_name}_t;

                   template <typename A>
                   struct op1_result_type_t<tet1d::{op_name}_t, A>
                   {{
                     typedef A result_type;
                     typedef A simd_pack_type; // TODO: Clean it
                   }};

                   struct {op_name}_t
                   {{
                     template <typename A>
                     typename tet1d::op1_result_type_t<tet1d::{op_name}_t, A>::result_type eval_scalar(A const & a) const {{
                       return {op_scalar_impl};
                     }}

                     template <typename A>
                     typename tet1d::op1_result_type_t<tet1d::{op_name}_t, A>::simd_pack_type eval_simd_i(A const & a) const {{
                       return nsimd::{op_name}(a);
                     }}
                   }};

                   template <typename A>
                   tet1d::op1_t<tet1d::{op_name}_t, A> {op_name}(A const & a)
                   {{
                     return tet1d::op1_t<tet1d::{op_name}_t, A>(tet1d::{op_name}_t(), a);
                   }};

                   '''.format(op_name=op_name, op_scalar_impl=scalar_impl[op_name])
        if (operator.cxx_operator != None):
          code += '''template <typename A>
                     tet1d::op1_t<tet1d::{op_name}_t, A> {op_cxx_op}(A const & a)
                     {{
                       return tet1d::{op_name}(a);
                     }};

                     '''.format(op_name=op_name, op_cxx_op=operator.cxx_operator)
      if (operator.params == ['v', 'v', 'v']):
        code += '''// {op_name}

                   struct {op_name}_t;

                   template <typename A0, typename A1>
                   struct op2_result_type_t<tet1d::{op_name}_t, A0, A1>
                   {{
                     typedef typename tet1d::common_type<A0, A1>::result_type result_type;
                     typedef typename tet1d::common_type<A0, A1>::result_type simd_pack_type; // TODO: Clean it
                   }};

                   struct {op_name}_t
                   {{
                     template <typename A0, typename A1>
                     typename tet1d::op2_result_type_t<tet1d::{op_name}_t, A0, A1>::result_type eval_scalar(A0 const & a0, A1 const & a1) const {{
                       return {op_scalar_impl};
                     }}

                     template <typename A0, typename A1>
                     typename tet1d::op2_result_type_t<tet1d::{op_name}_t, A0, A1>::simd_pack_type eval_simd_i(A0 const & a0, A1 const & a1) const {{
                       return nsimd::{op_name}(a0, a1);
                     }}
                   }};

                   template <typename A0, typename A1>
                   tet1d::op2_t<tet1d::{op_name}_t, A0, A1> {op_name}(A0 const & a0, A1 const & a1)
                   {{
                     return tet1d::op2_t<tet1d::{op_name}_t, A0, A1>(tet1d::{op_name}_t(), a0, a1);
                   }};

                   '''.format(op_name=op_name, op_scalar_impl=scalar_impl[op_name])
        if (operator.cxx_operator != None):
          code += '''template <typename A0, typename A1>
                     tet1d::op2_t<tet1d::{op_name}_t, A0, A1> {op_cxx_op}(A0 const & a0, A1 const & a1)
                     {{
                       return tet1d::{op_name}(a0, a1);
                     }};

                     '''.format(op_name=op_name, op_cxx_op=operator.cxx_operator)
      if (operator.params == ['v', 'v', 'v', 'v']):
        code += '''// {op_name}

                   struct {op_name}_t;

                   template <typename A0, typename A1, typename A2>
                   struct op3_result_type_t<tet1d::{op_name}_t, A0, A1, A2>
                   {{
                     typedef typename tet1d::common_type<typename tet1d::common_type<A0, A1>::result_type, A2>::result_type result_type;
                     typedef typename tet1d::common_type<typename tet1d::common_type<A0, A1>::result_type, A2>::result_type simd_pack_type; // TODO: Clean it
                   }};

                   struct {op_name}_t
                   {{
                     template <typename A0, typename A1, typename A2>
                     typename tet1d::op3_result_type_t<tet1d::{op_name}_t, A0, A1, A2>::result_type eval_scalar(A0 const & a0, A1 const & a1, A2 const & a2) const {{
                       return {op_scalar_impl};
                     }}

                     template <typename A0, typename A1, typename A2>
                     typename tet1d::op3_result_type_t<tet1d::{op_name}_t, A0, A1, A2>::simd_pack_type eval_simd_i(A0 const & a0, A1 const & a1, A2 const & a2) const {{
                       return nsimd::{op_name}(a0, a1, a2);
                     }}
                   }};

                   template <typename A0, typename A1, typename A2>
                   tet1d::op3_t<tet1d::{op_name}_t, A0, A1, A2> {op_name}(A0 const & a0, A1 const & a1, A2 const & a2)
                   {{
                     return tet1d::op3_t<tet1d::{op_name}_t, A0, A1, A2>(tet1d::{op_name}_t(), a0, a1, a2);
                   }};

                   '''.format(op_name=op_name, op_scalar_impl=scalar_impl[op_name])
    code += '}'
    dirname = os.path.join('include', 'nsimd', 'modules', 'tet1d')
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, 'functions.hpp')
    with common.open_utf8(opts, filename) as out:
      out.write(code)
    common.clang_format(opts, filename)
