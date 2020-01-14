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
    scalar_impl['orb'] = 'return a0 | a1;'
    scalar_impl['andb'] = 'return a0 & a1;'
    scalar_impl['andnotb'] = 'return a0 & ~a1;'
    scalar_impl['notb'] = 'return ~a;'
    scalar_impl['xorb'] = 'return a0 ^ a1;'
    scalar_impl['add'] = 'return a0 + a1;'
    scalar_impl['sub'] = 'return a0 - a1;'
    scalar_impl['mul'] = 'return a0 * a1;'
    scalar_impl['div'] = 'return a0 / a1;'
    scalar_impl['neg'] = 'return -a;'
    scalar_impl['min'] = 'return std::min(a0, a1);'
    scalar_impl['max'] = 'return std::max(a0, a1);'
    scalar_impl['abs'] = 'return tet1d::scalar_abs(a);'
    scalar_impl['fma'] = 'return a0 * a1 + a2;'
    scalar_impl['fnma'] = 'return -a0 * a1 + a2;'
    scalar_impl['fms'] = 'return a0 * a1 - a2;'
    scalar_impl['fnms'] = 'return -a0 * a1 - a2;'
    scalar_impl['ceil'] = 'return std::ceil(a);'
    scalar_impl['floor'] = 'return std::floor(a);'
    scalar_impl['trunc'] = '''#if NSIMD_CXX >= 2011
                              return std::trunc(a);
                              #else
                              return (a > 0) ? std::floor(a) : std::ceil(a);
                              #endif
                           '''
    scalar_impl['round_to_even'] = '''#if NSIMD_CXX >= 2011
                                      return std::round(a / 2) * 2;
                                      #else
                                      return std::floor((a / 2) + 0.5) * 2;
                                      #endif
                                   '''
    scalar_impl['reinterpret'] = 'return a;' # FIXME
    scalar_impl['cvt'] = 'return a;' # FIXME
    scalar_impl['rec'] = 'return 1 / a;'
    scalar_impl['rec11'] = 'return 1 / a;'
    scalar_impl['rec8'] = 'return 1 / a;'
    scalar_impl['sqrt'] = 'return std::sqrt(a);'
    scalar_impl['rsqrt'] = 'return 1 / std::sqrt(a);'
    scalar_impl['rsqrt11'] = 'return 1 / std::sqrt(a);'
    scalar_impl['rsqrt8'] = 'return 1 / std::sqrt(a);'

    cuda_impl = {}
    cuda_impl['orb'] = 'return a0 | a1;'
    cuda_impl['andb'] = 'return a0 & a1;'
    cuda_impl['andnotb'] = 'return a0 & ~a1;'
    cuda_impl['notb'] = 'return ~a;'
    cuda_impl['xorb'] = 'return a0 ^ a1;'
    cuda_impl['add'] = 'return a0 + a1;'
    cuda_impl['sub'] = 'return a0 - a1;'
    cuda_impl['mul'] = 'return a0 * a1;'
    cuda_impl['div'] = 'return a0 / a1;'
    cuda_impl['neg'] = 'return -a;'
    cuda_impl['min'] = 'return (a0 <= a1 ? a0 : a1);'
    cuda_impl['max'] = 'return (a0 >= a1 ? a0 : a1);'
    cuda_impl['abs'] = 'return a; // return (a < 0 ? -a : a);' # std::abs(const unsigned int&) is ambiguous
    cuda_impl['fma'] = 'return a0 * a1 + a2;'
    cuda_impl['fnma'] = 'return -a0 * a1 + a2;'
    cuda_impl['fms'] = 'return a0 * a1 - a2;'
    cuda_impl['fnms'] = 'return -a0 * a1 - a2;'
    cuda_impl['ceil'] = 'return tet1d::cuda_ceil(a);'
    cuda_impl['floor'] = 'return tet1d::cuda_floor(a);'
    cuda_impl['trunc'] = 'return tet1d::cuda_trunc(a);'
    cuda_impl['round_to_even'] = 'return tet1d::cuda_round(a / 2) * 2;'
    cuda_impl['reinterpret'] = 'return a;' # FIXME
    cuda_impl['cvt'] = 'return a;' # FIXME
    cuda_impl['rec'] = 'return 1 / a;'
    cuda_impl['rec11'] = 'return 1 / a;'
    cuda_impl['rec8'] = 'return 1 / a;'
    cuda_impl['sqrt'] = 'return ::sqrt(a);'
    cuda_impl['rsqrt'] = 'return 1 / ::sqrt(a);'
    cuda_impl['rsqrt11'] = 'return 1 / ::sqrt(a);'
    cuda_impl['rsqrt8'] = 'return 1 / ::sqrt(a);'

    # Code contains the functions as tet1d nodes
    code = ''
    code += '''#include <cmath>

               namespace tet1d
               {
                 // std::abs(const unsigned int&) is ambiguous
                 template <typename T>
                 T scalar_abs(T const & a) { return a < T(0) ? -a : a; }
                 u8 scalar_abs(u8 const a) { return a; }
                 u16 scalar_abs(u16 const a) { return a; }
                 u32 scalar_abs(u32 const a) { return a; }
                 u64 scalar_abs(u64 const a) { return a; }

                 #ifdef NSIMD_IS_NVCC

                 template <typename T>
                 __device__ T cuda_ceil(T const a) { return a; }
                 __device__ float cuda_ceil(float const a) { return ::ceil(a); }
                 __device__ double cuda_ceil(double const a) { return ::ceil(a); }

                 template <typename T>
                 __device__ T cuda_floor(T const a) { return a; }
                 __device__ float cuda_floor(float const a) { return ::floor(a); }
                 __device__ double cuda_floor(double const a) { return ::floor(a); }

                 template <typename T>
                 __device__ T cuda_trunc(T const a) { return a; }
                 __device__ float cuda_trunc(float const a) { return ::trunc(a); }
                 __device__ double cuda_trunc(double const a) { return ::trunc(a); }

                 template <typename T>
                 __device__ T cuda_round(T const a) { return a; }
                 __device__ float cuda_round(float const a) { return ::round(a); }
                 __device__ double cuda_round(double const a) { return ::round(a); }

                 #endif

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
                       {op_scalar_impl}
                     }}

                     template <typename A>
                     typename tet1d::op1_result_type_t<tet1d::{op_name}_t, A>::simd_pack_type eval_simd_i(A const & a) const {{
                       return nsimd::{op_name}(a);
                     }}

                     #ifdef NSIMD_IS_NVCC
                     template <typename A>
                     __device__ typename tet1d::op1_result_type_t<tet1d::{op_name}_t, A>::result_type eval_cuda_i(A const & a) const {{
                       {op_cuda_impl}
                     }}
                     #endif
                   }};

                   template <typename A>
                   tet1d::op1_t<tet1d::{op_name}_t, A> {op_name}(A const & a)
                   {{
                     return tet1d::op1_t<tet1d::{op_name}_t, A>(tet1d::{op_name}_t(), a);
                   }};

                '''.format(op_name=op_name, op_scalar_impl=scalar_impl[op_name], op_cuda_impl=cuda_impl[op_name])
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
                       {op_scalar_impl}
                     }}

                     template <typename A0, typename A1>
                     typename tet1d::op2_result_type_t<tet1d::{op_name}_t, A0, A1>::simd_pack_type eval_simd_i(A0 const & a0, A1 const & a1) const {{
                       return nsimd::{op_name}(a0, a1);
                     }}

                     #ifdef NSIMD_IS_NVCC
                     template <typename A0, typename A1>
                     __device__ typename tet1d::op2_result_type_t<tet1d::{op_name}_t, A0, A1>::result_type eval_cuda_i(A0 const & a0, A1 const & a1) const {{
                       {op_cuda_impl}
                     }}
                     #endif
                   }};

                   template <typename A0, typename A1>
                   tet1d::op2_t<tet1d::{op_name}_t, A0, A1> {op_name}(A0 const & a0, A1 const & a1)
                   {{
                     return tet1d::op2_t<tet1d::{op_name}_t, A0, A1>(tet1d::{op_name}_t(), a0, a1);
                   }};

                '''.format(op_name=op_name, op_scalar_impl=scalar_impl[op_name], op_cuda_impl=cuda_impl[op_name])
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
                       {op_scalar_impl}
                     }}

                     template <typename A0, typename A1, typename A2>
                     typename tet1d::op3_result_type_t<tet1d::{op_name}_t, A0, A1, A2>::simd_pack_type eval_simd_i(A0 const & a0, A1 const & a1, A2 const & a2) const {{
                       return nsimd::{op_name}(a0, a1, a2);
                     }}

                     #ifdef NSIMD_IS_NVCC
                     template <typename A0, typename A1, typename A2>
                     __device__ typename tet1d::op3_result_type_t<tet1d::{op_name}_t, A0, A1, A2>::result_type eval_cuda_i(A0 const & a0, A1 const & a1, A2 const & a2) const {{
                       {op_cuda_impl}
                     }}
                     #endif
                   }};

                   template <typename A0, typename A1, typename A2>
                   tet1d::op3_t<tet1d::{op_name}_t, A0, A1, A2> {op_name}(A0 const & a0, A1 const & a1, A2 const & a2)
                   {{
                     return tet1d::op3_t<tet1d::{op_name}_t, A0, A1, A2>(tet1d::{op_name}_t(), a0, a1, a2);
                   }};

                '''.format(op_name=op_name, op_scalar_impl=scalar_impl[op_name], op_cuda_impl=cuda_impl[op_name])

      # Generate the test
      for element_type in operator.types:
        if element_type == 'f16': continue # TODO
        test_header = '''#include <nsimd/modules/tet1d.hpp>

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

                         template <typename V0, typename V1, typename Fct>
                         int compare_vector( std::string const & title
                                           , std::string const & v0_name, V0 const & v0
                                           , std::string const & v1_name, V1 const & v1
                                           , Fct const & fct) {
                           if (v0.size() != v1.size() ||
                               std::equal(v0.begin(), v0.end(), v1.begin(), fct) == false) {
                             std::cout << "ERROR: " << title <<  ": vectors are not the same:" << std::endl;
                             print_container("- " + v0_name + ":\\n  ", v0, "\\n");
                             print_container("- " + v1_name + ":\\n  ", v1, "\\n");
                             return 1;
                           }
                           return 0;
                         }

                         template <typename V0, typename V1>
                         int compare_vector( std::string const & title
                                           , std::string const & v0_name, V0 const & v0
                                           , std::string const & v1_name, V1 const & v1) {
                           return compare_vector(title, v0_name, v0, v1_name, v1, std::equal_to<typename V0::value_type>());
                         }

                       int main() {
                         std::srand(std::time(0));
                         int r = 0;
                         size_t N = 16;

                     '''
        test_cpp = ''
        test_cu = ''
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
        if (operator.params == ['v', 'v']):
          test_common = '''std::vector<{T}, nsimd::allocator<{T}> > v(N);
                           std::vector<{T}/*, nsimd::allocator<{T}> */> r_loop(N);

                           for (size_t i = 0; i < v.size(); ++i) {{
                             v[i] = {rand_fct};
                           }}

                           for (size_t i = 0; i < r_loop.size(); ++i) {{
                             struct scalar_impl_t {{
                               {T} operator()({T} const & a) {{
                                 {op_scalar_impl}
                               }}
                             }};
                             r_loop[i] = scalar_impl_t()(v[i]);
                           }}

                           std::vector<{T}> r_scalar(N);

                           tet1d::out(r_scalar, tet1d::Scalar) = tet1d::{op_name}(tet1d::in(v));
                           r += compare_vector("{T}", "r_scalar", r_scalar, "r_loop", r_loop{compare_fct});

                        '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
          test_cpp += test_common
          test_cpp += '''std::vector<{T}, nsimd::allocator<{T}> > r_simd(N);

                         tet1d::out(r_simd, tet1d::Simd) = tet1d::{op_name}(tet1d::in(v));
                         r += compare_vector("{T}", "r_simd", r_simd, "r_loop", r_loop{compare_fct});

                      '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
          test_cu += test_common
          test_cu += '''std::vector<{T}, nsimd::cuda_allocator<{T}> > v_cuda(N);
                        if (cudaMemcpy(v_cuda.data(), v.data(), N * sizeof({T}), cudaMemcpyHostToDevice) != cudaSuccess) {{
                          throw std::runtime_error("cudaMemcpy fails");
                        }}
                        std::vector<{T}, nsimd::cuda_allocator<{T}> > r_cuda(N);

                        tet1d::out(r_cuda, tet1d::Cuda) = tet1d::{op_name}(tet1d::in(v_cuda));
                        std::vector<{T}> r_cuda_host(N);
                        if (cudaMemcpy(r_cuda_host.data(), r_cuda.data(), N * sizeof({T}), cudaMemcpyDeviceToHost) != cudaSuccess) {{
                          throw std::runtime_error("cudaMemcpy fails");
                        }}
                        r += compare_vector("{T}", "r_cuda_host", r_cuda_host, "r_loop", r_loop{compare_fct});

                     '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
          if (operator.cxx_operator != None):
            test_common = '''std::vector<{T}/*, nsimd::allocator<{T}> */> r_scalar_op(N);

                             tet1d::out(r_scalar_op, tet1d::Simd) = {op_cxx_op}tet1d::in(v);
                             r += compare_vector("{T}", "r_scalar_op", r_scalar_op, "r_loop", r_loop{compare_fct});

                          '''.format(T=element_type, compare_fct=compare_fct, op_cxx_op=operator.cxx_operator[8:] if operator.cxx_operator.startswith('operator') else operator.cxx_operator[8:])
            test_cpp += test_common
            test_cpp += '''std::vector<{T}/*, nsimd::allocator<{T}> */> r_simd_op(N);

                           tet1d::out(r_simd_op, tet1d::Simd) = {op_cxx_op}tet1d::in(v);
                           r += compare_vector("{T}", "r_simd_op", r_simd_op, "r_loop", r_loop{compare_fct});

                        '''.format(T=element_type, compare_fct=compare_fct, op_cxx_op=operator.cxx_operator[8:] if operator.cxx_operator.startswith('operator') else operator.cxx_operator[8:])
            test_cu += test_common
            test_cu += '''std::vector<{T}, nsimd::cuda_allocator<{T}> > r_cuda_op(N);

                           tet1d::out(r_cuda_op, tet1d::Cuda) = {op_cxx_op}tet1d::in(v_cuda);
                           std::vector<{T}> r_cuda_op_host(N);
                           if (cudaMemcpy(r_cuda_op_host.data(), r_cuda_op.data(), N * sizeof({T}), cudaMemcpyDeviceToHost) != cudaSuccess) {{
                             throw std::runtime_error("cudaMemcpy fails");
                           }}
                           r += compare_vector("{T}", "r_cuda_op_host", r_cuda_op_host, "r_loop", r_loop{compare_fct});

                        '''.format(T=element_type, compare_fct=compare_fct, op_cxx_op=operator.cxx_operator[8:] if operator.cxx_operator.startswith('operator') else operator.cxx_operator[8:])
        elif (operator.params == ['v', 'v', 'v']):
          write_test = True
          test_common = '''std::vector<{T}/*, nsimd::allocator<{T}> */> v0(N);
                           std::vector<{T}/*, nsimd::allocator<{T}> */> v1(N);
                           std::vector<{T}/*, nsimd::allocator<{T}> */> r_loop(N);

                           for (size_t i = 0; i < v0.size(); ++i) {{
                             v0[i] = {rand_fct};
                             v1[i] = {rand_fct};
                           }}

                           for (size_t i = 0; i < r_loop.size(); ++i) {{
                             struct scalar_impl_t {{
                               {T} operator()({T} const & a0, {T} const & a1) {{
                                 {op_scalar_impl}
                               }}
                             }};
                             r_loop[i] = scalar_impl_t()(v0[i], v1[i]);
                           }}

                           std::vector<{T}/*, nsimd::allocator<{T}> */> r_scalar(N);
                           tet1d::out(r_scalar, tet1d::Scalar) =
                             tet1d::{op_name}(tet1d::in(v0), tet1d::in(v1));
                           r += compare_vector("{T}", "r_scalar", r_scalar, "r_loop", r_loop{compare_fct});

                        '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
          test_cpp += test_common
          test_cpp += '''std::vector<{T}/*, nsimd::allocator<{T}> */> r_simd(N);

                         tet1d::out(r_simd, tet1d::Simd) =
                           tet1d::{op_name}(tet1d::in(v0), tet1d::in(v1));
                         r += compare_vector("{T}", "r_simd", r_simd, "r_loop", r_loop{compare_fct});

                      '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
          test_cu += test_common
          test_cu += '''std::vector<{T}, nsimd::cuda_allocator<{T}> > v0_cuda(N);
                        if (cudaMemcpy(v0_cuda.data(), v0.data(), N * sizeof({T}), cudaMemcpyHostToDevice) != cudaSuccess) {{
                          throw std::runtime_error("cudaMemcpy fails");
                        }}
                        std::vector<{T}, nsimd::cuda_allocator<{T}> > v1_cuda(N);
                        if (cudaMemcpy(v1_cuda.data(), v1.data(), N * sizeof({T}), cudaMemcpyHostToDevice) != cudaSuccess) {{
                          throw std::runtime_error("cudaMemcpy fails");
                        }}
                        std::vector<{T}, nsimd::cuda_allocator<{T}> > r_cuda(N);

                        tet1d::out(r_cuda, tet1d::Cuda) =
                          tet1d::{op_name}(tet1d::in(v0_cuda), tet1d::in(v1_cuda));
                        std::vector<{T}> r_cuda_host(N);
                        if (cudaMemcpy(r_cuda_host.data(), r_cuda.data(), N * sizeof({T}), cudaMemcpyDeviceToHost) != cudaSuccess) {{
                          throw std::runtime_error("cudaMemcpy fails");
                        }}
                        r += compare_vector("{T}", "r_cuda_host", r_cuda_host, "r_loop", r_loop{compare_fct});

                     '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
          if (operator.cxx_operator != None):
            test_common = '''std::vector<{T}/*, nsimd::allocator<{T}> */> r_scalar_op(N);

                             tet1d::out(r_scalar_op, tet1d::Scalar) =
                               tet1d::in(v0) {op_cxx_op} tet1d::in(v1);
                             r += compare_vector("{T}", "r_scalar_op", r_scalar_op, "r_loop", r_loop{compare_fct});

                          '''.format(T=element_type, compare_fct=compare_fct, op_cxx_op=operator.cxx_operator[8:] if operator.cxx_operator.startswith('operator') else operator.cxx_operator[8:])
            test_cpp += test_common
            test_cpp += '''std::vector<{T}/*, nsimd::allocator<{T}> */> r_simd_op(N);

                           tet1d::out(r_simd_op, tet1d::Simd) =
                             tet1d::in(v0) {op_cxx_op} tet1d::in(v1);
                           r += compare_vector("{T}", "r_simd_op", r_simd_op, "r_loop", r_loop{compare_fct});

                        '''.format(T=element_type, compare_fct=compare_fct, op_cxx_op=operator.cxx_operator[8:] if operator.cxx_operator.startswith('operator') else operator.cxx_operator[8:])
            test_cu += test_common
            test_cu += '''std::vector<{T}, nsimd::cuda_allocator<{T}> > r_cuda_op(N);

                          tet1d::out(r_cuda_op, tet1d::Cuda) =
                            tet1d::in(v0_cuda) {op_cxx_op} tet1d::in(v1_cuda);
                          std::vector<{T}> r_cuda_op_host(N);
                          if (cudaMemcpy(r_cuda_op_host.data(), r_cuda_op.data(), N * sizeof({T}), cudaMemcpyDeviceToHost) != cudaSuccess) {{
                            throw std::runtime_error("cudaMemcpy fails");
                          }}
                          r += compare_vector("{T}", "r_cuda_op_host", r_cuda_op_host, "r_loop", r_loop{compare_fct});

                       '''.format(T=element_type, compare_fct=compare_fct, op_cxx_op=operator.cxx_operator[8:] if operator.cxx_operator.startswith('operator') else operator.cxx_operator[8:])
        elif (operator.params == ['v', 'v', 'v', 'v']):
          write_test = True
          test_common = '''std::vector<{T}/*, nsimd::allocator<{T}> */> v0(N);
                           std::vector<{T}/*, nsimd::allocator<{T}> */> v1(N);
                           std::vector<{T}/*, nsimd::allocator<{T}> */> v2(N);
                           std::vector<{T}/*, nsimd::allocator<{T}> */> r_loop(N);

                           for (size_t i = 0; i < v0.size(); ++i) {{
                             v0[i] = {rand_fct};
                             v1[i] = {rand_fct};
                             v2[i] = {rand_fct};
                           }}

                           for (size_t i = 0; i < r_loop.size(); ++i) {{
                             struct scalar_impl_t {{
                               {T} operator()({T} const & a0, {T} const & a1, {T} const & a2) {{
                                 {op_scalar_impl}
                               }}
                             }};
                             r_loop[i] = scalar_impl_t()(v0[i], v1[i], v2[i]);
                           }}

                           std::vector<{T}/*, nsimd::allocator<{T}> */> r_scalar(N);

                           tet1d::out(r_scalar, tet1d::Scalar) =
                             tet1d::{op_name}(tet1d::in(v0), tet1d::in(v1), tet1d::in(v2));
                           r += compare_vector("{T}", "r_scalar", r_scalar, "r_loop", r_loop{compare_fct});

                        '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
          test_cpp += test_common
          test_cpp += '''std::vector<{T}/*, nsimd::allocator<{T}> */> r_simd(N);

                         tet1d::out(r_simd, tet1d::Simd) =
                           tet1d::{op_name}(tet1d::in(v0), tet1d::in(v1), tet1d::in(v2));
                         r += compare_vector("{T}", "r_simd", r_simd, "r_loop", r_loop{compare_fct});

                      '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
          test_cu += test_common
          test_cu += '''std::vector<{T}, nsimd::cuda_allocator<{T}> > v0_cuda(N);
                        if (cudaMemcpy(v0_cuda.data(), v0.data(), N * sizeof({T}), cudaMemcpyHostToDevice) != cudaSuccess) {{
                          throw std::runtime_error("cudaMemcpy fails");
                        }}
                        std::vector<{T}, nsimd::cuda_allocator<{T}> > v1_cuda(N);
                        if (cudaMemcpy(v1_cuda.data(), v1.data(), N * sizeof({T}), cudaMemcpyHostToDevice) != cudaSuccess) {{
                          throw std::runtime_error("cudaMemcpy fails");
                        }}
                        std::vector<{T}, nsimd::cuda_allocator<{T}> > v2_cuda(N);
                        if (cudaMemcpy(v2_cuda.data(), v2.data(), N * sizeof({T}), cudaMemcpyHostToDevice) != cudaSuccess) {{
                          throw std::runtime_error("cudaMemcpy fails");
                        }}
                        std::vector<{T}, nsimd::cuda_allocator<{T}> > r_cuda(N);

                        tet1d::out(r_cuda, tet1d::Cuda) =
                          tet1d::{op_name}(tet1d::in(v0_cuda), tet1d::in(v1_cuda), tet1d::in(v2_cuda));
                        std::vector<{T}> r_cuda_host(N);
                        if (cudaMemcpy(r_cuda_host.data(), r_cuda.data(), N * sizeof({T}), cudaMemcpyDeviceToHost) != cudaSuccess) {{
                          throw std::runtime_error("cudaMemcpy fails");
                        }}
                        r += compare_vector("{T}", "r_cuda_host", r_cuda_host, "r_loop", r_loop{compare_fct});

                     '''.format(T=element_type, rand_fct=rand_fct, compare_fct=compare_fct, op_name=op_name, op_scalar_impl=scalar_impl[op_name])
        else:
          continue
        test_footer = '''  return r;
                         }

                      '''
        test_cpp = '''{test_header}
                      // Test of {op_name} for {T}

                      {test_cpp}
                      {test_footer}
                   '''.format(T=element_type, op_name=op_name, test_header=test_header, test_cpp=test_cpp, test_footer=test_footer)
        test_cu = '''{test_header}
                     // Test of {op_name} for {T}

                     {test_cu}
                     {test_footer}
                  '''.format(T=element_type, op_name=op_name, test_header=test_header, test_cu=test_cu, test_footer=test_footer)
        # Write the tests
        if write_test:
          dirname = os.path.join('tests', 'modules', 'tet1d', 'functions')
          os.makedirs(dirname, exist_ok=True)
          # C++
          filename = os.path.join(dirname, op_name + '.' + element_type + '.cpp')
          with common.open_utf8(opts, filename) as out:
            out.write(test_cpp)
          common.clang_format(opts, filename)
          # CUDA
          filename = os.path.join(dirname, op_name + '.' + element_type + '.cu')
          with common.open_utf8(opts, filename) as out:
            out.write(test_cu)
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
