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
    scalar_impl['abs'] = '(a < 0 ? -a : a)' # std::abs(const unsigned int&) is ambiguous
    scalar_impl['fma'] = 'a0 * a1 + a2'
    scalar_impl['fnma'] = '-a0 * a1 + a2'
    scalar_impl['fms'] = 'a0 * a1 - a2'
    scalar_impl['fnms'] = '-a0 * a1 - a2'
    scalar_impl['ceil'] = 'std::ceil(a)'
    scalar_impl['floor'] = 'std::floor(a)'
    scalar_impl['trunc'] = 'std::trunc(a)'
    scalar_impl['round_to_even'] = 'std::round(a)' # FIXME
    scalar_impl['reinterpret'] = 'a' # FIXME
    scalar_impl['cvt'] = 'a' # FIXME
    scalar_impl['rec'] = '1 / a'
    scalar_impl['rec11'] = '1 / a'
    scalar_impl['rec8'] = '1 / a'
    scalar_impl['sqrt'] = 'std::sqrt(a)'
    scalar_impl['rsqrt'] = '1 / std::sqrt(a)'
    scalar_impl['rsqrt11'] = '1 / std::sqrt(a)'
    scalar_impl['rsqrt8'] = '1 / std::sqrt(a)'

    # Code contains the functions as tet1d nodes
    code = ''
    code += '''namespace tet1d
               {

            '''

    # Generate code and tests
    for op_name, operator in operators.operators.items():
      #print('-- ', op_name, operator.signature)

      # Skip some operators
      if op_name == 'downcvt': continue
      if op_name == 'ziplo': continue
      if op_name == 'ziphi': continue
      if op_name == 'unziplo': continue
      if op_name == 'unziphi': continue

      # TODO
      if op_name == 'reinterpret': continue
      if op_name == 'cvt': continue
      if op_name == 'notb': continue
      if op_name == 'orb': continue
      if op_name == 'andb': continue
      if op_name == 'andnotb': continue
      if op_name == 'xorb': continue

      # Generate code
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

      # Generate the test
      test = ''
      test += '''#include <nsimd/modules/tet1d.hpp>

                #include <algorithm>
                #include <ctime>
                #include <iostream>
                #include <vector>

                template <typename C>
                void print_container(std::string const & header, C const & c, std::string const & footer) {
                  std::cout << header << "{";
                  for (size_t i = 0; i < c.size(); ++i) {
                    std::cout << " " << c[i];
                  }
                  std::cout << " }" << footer;
                }

                template <typename T>
                T rand_int(T const a, T const b) {
                  // https://stackoverflow.com/questions/5008804/generating-random-integer-from-a-range
                  return a + (rand() % (a - b + 1));
                }

                template <typename T>
                T rand_float(T const a, T const b) {
                  // https://stackoverflow.com/questions/686353/random-float-number-generation
                  return a + T(rand()) / (T(RAND_MAX) / (b - a));
                }

                struct compare_float_t {
                  template <typename T>
                  bool operator()(T const a, T const b) const {
                    return std::abs(a - b) < T(0.0002);
                  }
                };

                template <typename T, typename Fct = std::equal_to<typename T::value_type>>
                int compare_vector( std::string const & title
                                  , std::string const & v0_name, T const & v0
                                  , std::string const & v1_name, T const & v1
                                  , Fct const & fct = Fct()) {
                if (v0.size() != v1.size() ||
                    std::equal(v0.begin(), v0.end(), v1.begin(), fct) == false) {
                  std::cout << "ERROR: " << title <<  ": vectors are not the same:" << std::endl;
                  print_container("- " + v0_name + ":\\n  ", v0, "\\n");
                  print_container("- " + v1_name + ":\\n  ", v1, "\\n");
                  return 1;
                }
                return 0;
              }

              int main() {
                std::srand(std::time(0));

                int r = 0;

                size_t N = 16;

            '''
      write_test = False
      for element_type in ['f32', 'f64', 'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64']:
        if ('sqrt' in op_name or 'rec' in op_name) and element_type not in ['f32', 'f64']: continue
        rand_fct = ''
        compare_fct = ''
        if element_type == 'f32':
          rand_fct = 'rand_float(1.0f, 42.0f)'
          compare_fct = ', compare_float_t()'
        elif element_type == 'f64':
          rand_fct = 'rand_float(1.0, 42.0)'
          compare_fct = ', compare_float_t()'
        else:
          rand_fct = 'rand_int(' + element_type + '(1), ' + element_type + '(42))'
        test_T = ''
        if (operator.params == ['v', 'v']):
          write_test = True
          test_T += '''std::vector<{T}/*, nsimd::allocator<{T}> */> v(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> r_loop(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> r_scalar(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> r_simd(N);

                       for (size_t i = 0; i < v.size(); ++i) {{
                         v[i] = {rand_fct};
                       }}

                       for (size_t i = 0; i < r_loop.size(); ++i) {{
                         {T} const & a = v[i];
                         r_loop[i] = {op_scalar_impl};
                       }}

                       tet1d::out(r_scalar, tet1d::Scalar) = tet1d::{op_name}(tet1d::in(v));
                       r += compare_vector("{T}", "r_scalar", r_scalar, "r_loop", r_loop{compare_fct});

                       tet1d::out(r_simd, tet1d::Simd) = tet1d::{op_name}(tet1d::in(v));
                       r += compare_vector("{T}", "r_simd", r_simd, "r_loop", r_loop{compare_fct});

                    '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
          if (operator.cxx_operator != None):
            test_T += '''std::vector<{T}/*, nsimd::allocator<{T}> */> r_scalar_op(N);
                         std::vector<{T}/*, nsimd::allocator<{T}> */> r_simd_op(N);

                         tet1d::out(r_scalar_op, tet1d::Scalar) = {op_cxx_op}tet1d::in(v);
                         r += compare_vector("{T}", "r_scalar_op", r_scalar_op, "r_loop", r_loop{compare_fct});

                         tet1d::out(r_simd_op, tet1d::Simd) = {op_cxx_op}tet1d::in(v);
                         r += compare_vector("{T}", "r_simd_op", r_simd_op, "r_loop", r_loop{compare_fct});

                    '''.format(T=element_type, compare_fct=compare_fct, op_cxx_op=operator.cxx_operator[8:] if operator.cxx_operator.startswith('operator') else operator.cxx_operator[8:])
        elif (operator.params == ['v', 'v', 'v']):
          write_test = True
          test_T += '''std::vector<{T}/*, nsimd::allocator<{T}> */> v0(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> v1(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> r_loop(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> r_scalar(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> r_simd(N);

                       for (size_t i = 0; i < v0.size(); ++i) {{
                         v0[i] = {rand_fct};
                         v1[i] = {rand_fct};
                       }}

                       for (size_t i = 0; i < r_loop.size(); ++i) {{
                         {T} const & a0 = v0[i];
                         {T} const & a1 = v1[i];
                         r_loop[i] = {op_scalar_impl};
                       }}

                       tet1d::out(r_scalar, tet1d::Scalar) =
                         tet1d::{op_name}(tet1d::in(v0), tet1d::in(v1));
                       r += compare_vector("{T}", "r_scalar", r_scalar, "r_loop", r_loop{compare_fct});

                       tet1d::out(r_simd, tet1d::Simd) =
                         tet1d::{op_name}(tet1d::in(v0), tet1d::in(v1));
                       r += compare_vector("{T}", "r_simd", r_simd, "r_loop", r_loop{compare_fct});

                      '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
          if (operator.cxx_operator != None):
            test_T += '''std::vector<{T}/*, nsimd::allocator<{T}> */> r_scalar_op(N);
                         std::vector<{T}/*, nsimd::allocator<{T}> */> r_simd_op(N);

                         tet1d::out(r_scalar_op, tet1d::Scalar) =
                           tet1d::in(v0) {op_cxx_op} tet1d::in(v1);
                         r += compare_vector("{T}", "r_scalar_op", r_scalar_op, "r_loop", r_loop{compare_fct});

                         tet1d::out(r_simd_op, tet1d::Simd) =
                           tet1d::in(v0) {op_cxx_op} tet1d::in(v1);
                         r += compare_vector("{T}", "r_simd_op", r_simd_op, "r_loop", r_loop{compare_fct});

                    '''.format(T=element_type, compare_fct=compare_fct, op_cxx_op=operator.cxx_operator[8:] if operator.cxx_operator.startswith('operator') else operator.cxx_operator[8:])
        elif (operator.params == ['v', 'v', 'v', 'v']):
          write_test = True
          test_T += '''std::vector<{T}/*, nsimd::allocator<{T}> */> v0(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> v1(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> v2(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> r_loop(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> r_scalar(N);
                       std::vector<{T}/*, nsimd::allocator<{T}> */> r_simd(N);

                       for (size_t i = 0; i < v0.size(); ++i) {{
                         v0[i] = {rand_fct};
                         v1[i] = {rand_fct};
                         v2[i] = {rand_fct};
                       }}

                       for (size_t i = 0; i < r_loop.size(); ++i) {{
                         {T} const & a0 = v0[i];
                         {T} const & a1 = v1[i];
                         {T} const & a2 = v2[i];
                         r_loop[i] = {op_scalar_impl};
                       }}

                       tet1d::out(r_scalar, tet1d::Scalar) =
                         tet1d::{op_name}(tet1d::in(v0), tet1d::in(v1), tet1d::in(v2));
                       r += compare_vector("{T}", "r_scalar", r_scalar, "r_loop", r_loop{compare_fct});

                       tet1d::out(r_simd, tet1d::Simd) =
                         tet1d::{op_name}(tet1d::in(v0), tet1d::in(v1), tet1d::in(v2));
                       r += compare_vector("{T}", "r_simd", r_simd, "r_loop", r_loop{compare_fct});

                    '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
        else:
          continue
        test += '''// Test of {op_name} for {T}
                   {{
                     {test_T}
                   }}

                 '''.format(T=element_type, op_name=op_name, test_T=test_T)
      test += '''  return r;
                 }

              '''
      # Write the test
      if write_test:
        dirname = os.path.join('tests', 'modules', 'tet1d', 'functions')
        os.makedirs(dirname, exist_ok=True)
        filename = os.path.join(dirname, op_name + '.cpp')
        with common.open_utf8(opts, filename) as out:
          out.write(test)
        common.clang_format(opts, filename)

    # End of the code
    code += '}'

    # Write the code
    dirname = os.path.join('include', 'nsimd', 'modules', 'tet1d')
    os.makedirs(dirname, exist_ok=True)
    filename = os.path.join(dirname, 'functions.hpp')
    with common.open_utf8(opts, filename) as out:
      out.write(code)
    common.clang_format(opts, filename)
