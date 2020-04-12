/*

Copyright (c) 2020 Agenium Scale

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#ifndef NSIMD_MODULES_SPMD_HPP
#define NSIMD_MODULES_SPMD_HPP

#include <nsimd/nsimd-all.hpp>

#include <cassert>
#include <vector>
#include <cstring>

namespace spmd {

// ----------------------------------------------------------------------------
// CUDA and ROCm

#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) ||                               \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)

// kernel definition
#define nsimd_kernel(name, ...)                                               \
  __device__ void name(int nsimd_current_thread_id, __VA_ARGS__)

// supported types
#define k_char i8
#define k_uchar u8

#define k_short i16
#define k_ushort u16
#define k_half f16

#define k_int i32
#define k_uint u32
#define k_float f32

#define k_long i64
#define k_ulong u64
#define k_double f64

// supported language tokens
#define k_if(cond) if (cond)
#define k_else else
#define k_endif
#define k_while(cond) while (cond)
#define k_break break
#define k_endwhile
#define k_return return

// loads and stores
#define k_store(addr, value)                                                  \
  do {                                                                        \
    *(addr + nsimd_current_thread_id) = value;                                \
  } while (0)
#define k_load(addr) (*(addr + nsimd_current_thread_id))

// ----------------------------------------------------------------------------
// SIMD and SCALAR: dispatch between the two is done on a type

#else

// type indicating SIMD or scalar kernel
struct KernelScalar {};
struct KernelSIMD {};

// kernel definition (generic)
#define nsimd_kernel(name, ...)                                               \
  template <typename KernelType>                                              \
  void name(KernelType, nsimd_nat spmd_i, __VA_ARGS__)

// supported types (generic)
template <typename KernelType, typename T> struct type_t {};

// supported types (scalar)
template <typename T> struct type_t<KernelScalar, T> { typedef T type; };

// supported types (SIMD)
template <typename T> struct type_t<KernelSIMD, T> {
  typedef nsimd::pack<T> type;
};

// supported types (generic)
#define k_char typename type_t<KernelType, i8>::type
#define k_uchar typename type_t<KernelType, u8>::type

#define k_short typename type_t<KernelType, i16>::type
#define k_ushort typename type_t<KernelType, u16>::type
#define k_half typename type_t<KernelType, f16>::type

#define k_int typename type_t<KernelType, i32>::type
#define k_uint typename type_t<KernelType, u32>::type
#define k_float typename type_t<KernelType, f32>::type

#define k_long typename type_t<KernelType, i64>::type
#define k_ulong typename type_t<KernelType, u64>::type
#define k_double typename type_t<KernelType, f64>::type

// loads and stores (generic)
template <typename KernelType> struct store_t {};
template <typename KernelType> struct load_t {};
#define k_store(addr, value) store_t<KernelType>::impl(addr, value)
#define k_load(addr) load_t<KernelType>::impl(addr)

// loads and stores (scalar)
template <> struct store_t<KernelScalar> {
  template <typename T, typename S> static void impl(T *addr, S value) {
    *addr = (T)value;
  }
};

// ----------------------------------------------------------------------------

#endif

} // namespace spmd

//#include <nsimd/modules/tet1d/functions.hpp>

#endif
