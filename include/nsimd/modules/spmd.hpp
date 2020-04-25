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
//   KernelType   SIMD or scalar (for loop tails)
//   ScalarBits   width in bits of types used in kernels (8, 16, 32 or 64)
//   N            unroll factor (number of threads per block for GPUs)
//   Other template parameters are deduced automatically from arguments
#define nsimd_kernel(name, ...)                                               \
  template <typename KernelType, int ScalarBits, int N, typename MaskType>    \
  inline void name(nsimd_nat spmd_i_, MaskType spmd_mask_, __VA_ARGS__)

#define nsimd_launch_kernel_1d(name, spmd_scalar_bits_, spmd_unroll_,         \
                               spmd_n_, ...)                                  \
  {                                                                           \
    typename spmd::type_t<spmd::KernelSIMD, spmd_scalar_bits_,                \
                          spmd_unroll_>::btype spmd_mask_(true);              \
    nsimd_nat spmd_i_;                                                        \
    nsimd_nat len =                                                           \
        nsimd::len(typename spmd::type_t<spmd::KernelSIMD, spmd_scalar_bits_, \
                                         spmd_unroll_>::itype());             \
    for (spmd_i_ = 0; spmd_i_ + len < spmd_n_; spmd_i_ += len) {              \
      name<spmd::KernelSIMD, spmd_scalar_bits_, spmd_unroll_>(                \
          spmd_i_, spmd_mask_, __VA_ARGS__);                                  \
    }                                                                         \
    for (; spmd_i_ < spmd_n_; spmd_i_++) {                                    \
      name<spmd::KernelScalar, spmd_scalar_bits_, spmd_unroll_>(              \
          spmd_i_, true, __VA_ARGS__);                                        \
    }                                                                         \
  }

// supported types (generic)
template <typename KernelType, int ScalarBits, int N> struct type_t {};

// supported types (scalar)
template <int N> struct type_t<KernelScalar, 8, N> {
  typedef i8 itype;
  typedef u8 utype;
  typedef bool btype;
};

template <int N> struct type_t<KernelScalar, 16, N> {
  typedef i16 itype;
  typedef u16 utype;
  typedef f16 ftype;
  typedef bool btype;
};

template <int N> struct type_t<KernelScalar, 32, N> {
  typedef i32 itype;
  typedef u32 utype;
  typedef f32 ftype;
  typedef bool btype;
};

template <int N> struct type_t<KernelScalar, 64, N> {
  typedef i64 itype;
  typedef u64 utype;
  typedef f64 ftype;
  typedef bool btype;
};

// supported types (SIMD)
template <int N> struct type_t<KernelSIMD, 8, N> {
  typedef nsimd::pack<i8, N> itype;
  typedef nsimd::pack<u8, N> utype;
  typedef nsimd::packl<i8, N> btype;
};

template <int N> struct type_t<KernelSIMD, 16, N> {
  typedef nsimd::pack<i16, N> itype;
  typedef nsimd::pack<u16, N> utype;
  typedef nsimd::pack<f16, N> ftype;
  typedef nsimd::packl<i16, N> btype;
};

template <int N> struct type_t<KernelSIMD, 32, N> {
  typedef nsimd::pack<i32, N> itype;
  typedef nsimd::pack<u32, N> utype;
  typedef nsimd::pack<f32, N> ftype;
  typedef nsimd::packl<i32, N> btype;
};

template <int N> struct type_t<KernelSIMD, 64, N> {
  typedef nsimd::pack<i64, N> itype;
  typedef nsimd::pack<u64, N> utype;
  typedef nsimd::pack<f64, N> ftype;
  typedef nsimd::packl<i64, N> btype;
};

// supported types (generic)
#define k_int typename type_t<KernelType, ScalarBits>::itype
#define k_uint typename type_t<KernelType, ScalarBits>::utype
#define k_float typename type_t<KernelType, ScalarBits>::ftype
#define k_bool typename type_t<KernelType, ScalarBits>::btype

// loads and stores (generic)
template <typename KernelType> struct store_helper {};
template <typename KernelType> struct load_helper {};
#define k_store(base_addr, value)                                             \
  spmd::store_helper<KernelType>::impl(spmd_mask_, &base_addr[spmd_i_], value)
#define k_load(base_addr)                                                     \
  spmd::load_helper<KernelType>::impl(spmd_mask_, &base_addr[spmd_i_])

// loads and stores (scalar)
template <> struct store_helper<KernelScalar> {
  template <typename T, typename S>
  static void impl(bool mask, T *addr, S value) {
    if (mask) {
      *addr = (T)value;
    }
  }
};

template <> struct load_helper<KernelScalar> {
  template <typename T>
  static T impl(bool mask, T *addr) {
    if (mask) {
      return *addr;
    } else {
      return T(0);
    }
  }
};

template <> struct store_helper<KernelSIMD> {
  template <typename T, typename S>
  static void impl(nsimd::packl<T> const &mask, S *addr,
                   nsimd::pack<S> const &value) {
    return nsimd::mask_storeu(mask, addr, value);
  }
};

template <> struct load_helper<KernelSIMD> {
  template <typename T, typename S>
  static nsimd::pack<S> impl(nsimd::packl<T> const &mask, S *addr) {
    return nsimd::maskz_loadu(mask, addr);
  }
};

// assignment statement
#define k_set(var, value) var = nsimd::if_else(spmd_mask_, value, var)

// while statement (k_while)
#define k_while(cond)                                                         \
  {                                                                           \
    k_bool spmd_middle_mask_ = spmd_mask_;                                    \
    {                                                                         \
      while (nsimd::any(cond)) {                                              \
        k_bool spmd_mask_ = spmd_middle_mask_ && (cond);

// continue statement
#define k_continue spmd_mask_ = false

// endwhile statement (k_endwhile)
#define k_endwhile                                                            \
  spmd_middle_mask_ =                                                         \
      nsimd::andnotl(spmd_middle_mask_, spmd_lanes_go_off_break_);            \
  }                                                                           \
  spmd_middle_mask_ =                                                         \
      nsimd::andnotl(spmd_middle_mask_, spmd_lanes_go_off_return_);           \
  }

// return statement (k_return)
#define k_return                                                              \
  spmd_lanes_go_off_return_ = spmd_mask_;                                     \
  spmd_mask_ = false;

// break statement (k_break)
#define k_break                                                               \
  spmd_lanes_go_off_break_ = spmd_mask_;                                      \
  spmd_mask_ = false;

// if statement (k_if)
#define k_if(cond)                                                            \
  {                                                                           \
    k_bool spmd_middle_mask_ = (cond) && spmd_mask_;                          \
    k_bool spmd_lanes_go_off_return_(false);                                  \
    k_bool spmd_lanes_go_off_break_(false);                                   \
    {                                                                         \
      k_bool spmd_mask_ = spmd_middle_mask_;


// endif statement (k_endif)
#define k_endif                                                               \
  spmd_middle_mask_ =                                                         \
      nsimd::andnotl(spmd_middle_mask_, spmd_lanes_go_off_return_);           \
  spmd_middle_mask_ =                                                         \
      nsimd::andnotl(spmd_middle_mask_, spmd_lanes_go_off_break_);            \
  }                                                                           \
  spmd_mask_ = spmd_middle_mask_;                                             \
  }

// ----------------------------------------------------------------------------

#endif

} // namespace spmd

//#include <nsimd/modules/tet1d/functions.hpp>

#endif
