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
#include <cstring>
#include <vector>

#if defined(NSIMD_ONEAPI)
#include <CL/sycl.hpp>
#endif

namespace spmd {

#if NSIMD_CXX < 2011 || NSIMD_C < 1999
#define NSIMD_VARIADIC_MACROS_IS_EXTENSION
#endif

#ifdef NSIMD_VARIADIC_MACROS_IS_EXTENSION
#if defined(NSIMD_IS_GCC)
/* Not emitting the warning -Wvariadic-macros is not possible with
   GCC <= 12. It is a bug. A workaround is to tell GCC to consider this
   header file as a system header file so that all warnings are not
   emitted. This is not satisfying but necessary for the moment.
   */
#pragma GCC system_header
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#elif defined(NSIMD_IS_CLANG)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvariadic-macros"
#endif
#endif

// ----------------------------------------------------------------------------
// CUDA ,ROCm and oneAPI

#if defined(NSIMD_CUDA) || defined(NSIMD_ROCM) || defined(NSIMD_ONEAPI)

#if defined(NSIMD_CUDA)

// 1d kernel definition
#define spmd_kernel_1d(name, ...)                                             \
  template <int spmd_ScalarBits_> __global__ void name(__VA_ARGS__, int n) {  \
    int spmd_i_ = threadIdx.x + blockIdx.x * blockDim.x;                      \
    if (spmd_i_ < n) {

// templated kernel definition
#define spmd_tmpl_kernel_1d(name, template_argument, ...)                     \
  template <typename template_argument, int spmd_ScalarBits_>                 \
  __global__ void name(__VA_ARGS__, int n) {                                  \
    int spmd_i_ = threadIdx.x + blockIdx.x * blockDim.x;                      \
    if (spmd_i_ < n) {

#elif defined(NSIMD_ROCM)

// 1d kernel definition
#define spmd_kernel_1d(name, ...)                                             \
  template <int spmd_ScalarBits_>                                             \
  __global__ void name(__VA_ARGS__, size_t n) {                               \
    size_t spmd_i_ = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;          \
    if (spmd_i_ < n) {

// templated kernel definition
#define spmd_tmpl_kernel_1d(name, template_argument, ...)                     \
  template <typename template_argument, int spmd_ScalarBits_>                 \
  __global__ void name(__VA_ARGS__, size_t n) {                               \
    size_t spmd_i_ = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;          \
    if (spmd_i_ < n) {

#elif defined(NSIMD_ONEAPI)

// 1d kernel definition
#define spmd_kernel_1d(name, ...)                                             \
  template <int spmd_ScalarBits_>                                             \
  inline void name(__VA_ARGS__, const size_t n, sycl::nd_item<1> item) {      \
    size_t spmd_i_ = item.get_global_id().get(0);                             \
    if (spmd_i_ < n) {

// templated kernel definition
#define spmd_tmpl_kernel_1d(name, template_argument, ...)                     \
  template <typename template_argument, int spmd_ScalarBits_>                 \
  inline void name(__VA_ARGS__, const size_t n, sycl::nd_item<1> item) {      \
    size_t spmd_i_ = item.get_global_id().get(0);                             \
    if (spmd_i_ < n) {

#endif

#define spmd_kernel_end                                                       \
  }                                                                           \
  }

#if defined(NSIMD_CUDA) || defined(NSIMD_ROCM)

// device function
#define spmd_dev_func(type_name, ...)                                         \
  template <int spmd_ScalarBits_> __device__ type_name(__VA_ARGS__) {

// templated device function
#define spmd_tmpl_dev_func(type_name, template_argument, ...)                 \
  template <typename template_argument, int spmd_ScalarBits_>                 \
  __device__ type_name(__VA_ARGS__) {

#elif defined(NSIMD_ONEAPI)

// device function
#define spmd_dev_func(type_name, ...)                                         \
  template <int spmd_ScalarBits_> type_name(__VA_ARGS__) {

// templated device function
#define spmd_tmpl_dev_func(type_name, template_argument, ...)                 \
  template <typename template_argument, int spmd_ScalarBits_>                 \
  type_name(__VA_ARGS__) {

#endif

#define spmd_dev_func_end }

// call spmd_dev_function
#define spmd_call_dev_func(name, ...) name<spmd_ScalarBits_>(__VA_ARGS__)

// call templated spmd_dev_function
#define spmd_call_tmpl_dev_func(name, template_argument, ...)                 \
  name<template_argument, spmd_ScalarBits_>(__VA_ARGS__)

#if defined(NSIMD_CUDA)

// launch 1d kernel CUDA
#define spmd_launch_kernel_1d(name, spmd_scalar_bits_, threads_per_block, n,  \
                              ...)                                            \
  name<spmd_scalar_bits_>                                                     \
      <<<(unsigned int)((n + threads_per_block - 1) / threads_per_block),     \
         (unsigned int)(threads_per_block)>>>(__VA_ARGS__, (int)n)

#elif defined(NSIMD_ROCM)

// launch 1d kernel ROCm
#define spmd_launch_kernel_1d(name, spmd_scalar_bits_, threads_per_block, n,  \
                              ...)                                            \
  hipLaunchKernelGGL(                                                         \
      (name<spmd_scalar_bits_>),                                              \
      (size_t)((n + threads_per_block - 1) / threads_per_block),              \
      (size_t)(threads_per_block), 0, NULL, __VA_ARGS__, (size_t)n)

#elif defined(NSIMD_ONEAPI)

// launch 1d kernel oneAPI
#define spmd_launch_kernel_1d(name, spmd_scalar_bits_, threads_per_block, n,  \
                              ...)                                            \
  const size_t total_num_threads =                                            \
      nsimd::compute_total_num_threads(n, threads_per_block);                 \
  sycl::queue q = nsimd::_get_global_queue();                                 \
  q.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_num_threads),         \
                                   sycl::range<1>(threads_per_block)),        \
                 [=](sycl::nd_item<1> item) {                                 \
                   name<spmd_scalar_bits_>(__VA_ARGS__, (size_t)n, item);     \
                 })                                                           \
      .wait();

#endif

// supported types (generic)
template <int ScalarBits> struct type_t {};

// supported types (scalar)
template <> struct type_t<8> {
  typedef i8 itype;
  typedef u8 utype;
  typedef bool btype;
};

template <> struct type_t<16> {
  typedef i16 itype;
  typedef u16 utype;
  typedef f16 ftype;
  typedef bool btype;
};

template <> struct type_t<32> {
  typedef i32 itype;
  typedef u32 utype;
  typedef f32 ftype;
  typedef bool btype;
};

template <> struct type_t<64> {
  typedef i64 itype;
  typedef u64 utype;
  typedef f64 ftype;
  typedef bool btype;
};

// supported types (generic)
#define k_int typename spmd::type_t<spmd_ScalarBits_>::itype
#define k_uint typename spmd::type_t<spmd_ScalarBits_>::utype
#define k_float typename spmd::type_t<spmd_ScalarBits_>::ftype
#define k_bool typename spmd::type_t<spmd_ScalarBits_>::btype

// loads and stores (generic)
#define k_store(base_addr, value)                                             \
  do {                                                                        \
    base_addr[spmd_i_] = value;                                               \
  } while (0)

#define k_unmasked_store(base_addr, value) k_store(base_addr, value)
#define k_load(base_addr) base_addr[spmd_i_]
#define k_unmasked_load(base_addr) k_load(base_addr)

// f32 <--> f16 conversions
#if defined(NSIMD_CUDA) || defined(NSIMD_ROCM)
#define k_f32_to_f16(a) __float2half(a)
#define k_f16_to_f32(a) __half2float(a)

#elif defined(NSIMD_ONEAPI)
// use sycl::half f32 --> f16 conversion operator: sycl::half(const float& RHS)
#define k_f32_to_f16(a) f16(a)
// use sycl::half::operator float() half --> f32
#define k_f16_to_f32(a) static_cast<f32>(a)
#endif

// assignment statement
#define k_set(var, value)                                                     \
  do {                                                                        \
    var = value;                                                              \
  } while (0)

#define k_unmasked_set(var, value) k_set(var, value)

// while statement (k_while)
#define k_while(cond) while (cond) {
#define k_endwhile }

// break statement (k_break)
#define k_break break

// continue statement (k_continue)
#define k_continue continue

// endwhile statement (k_endwhile)
#define k_endwhile }

// if statement (k_if)
#define k_if(cond) if (cond) {

// elseif statement (k_elseif)
#define k_elseif(cond)                                                        \
  }                                                                           \
  else if (cond) {

// else statement (k_else)
#define k_else                                                                \
  }                                                                           \
  else {

// endif statement (k_endif)
#define k_endif }

// ----------------------------------------------------------------------------
// SIMD and SCALAR: dispatch between the two is done on a type

#else

// helpers
template <typename T, int N> nsimd::pack<T, N> to_pack(T a) {
  return nsimd::pack<T, N>(a);
}

template <typename T, int N, typename SimdExt>
nsimd::pack<T, N, SimdExt> to_pack(nsimd::pack<T, N, SimdExt> const &a) {
  return a;
}

template <typename T, int N> nsimd::packl<T, N> to_packl(bool a) {
  return nsimd::packl<T, N>(int(a));
}

template <typename T, int N, typename Pack>
nsimd::packl<T, N> to_packl(Pack const &a) {
  return nsimd::reinterpretl<nsimd::packl<T, N> >(a);
}

template <typename T> struct base_type { typedef T type; };

template <typename T, int N, typename SimdExt>
struct base_type<nsimd::pack<T, N, SimdExt> > {
  typedef T type;
};

template <typename T, int N, typename SimdExt>
struct base_type<nsimd::packl<T, N, SimdExt> > {
  typedef T type;
};

// type indicating SIMD or scalar kernel
struct KernelScalar {};
struct KernelSIMD {};

// common to all function: mainly to avoid warnings
#define spmd_func_begin_                                                      \
  (void)spmd_i_;                                                              \
  (void)spmd_mask_;                                                           \
  k_bool spmd_off_lanes_return_(false);                                       \
  (void)spmd_off_lanes_return_;                                               \
  k_bool spmd_off_lanes_break_(false);                                        \
  (void)spmd_off_lanes_break_;                                                \
  k_bool spmd_off_lanes_continue_(false);                                     \
  (void)spmd_off_lanes_continue_;

// 1d kernel definition
#define spmd_kernel_1d(name, ...)                                             \
  template <typename spmd_KernelType_, int spmd_ScalarBits_, int spmd_N_,     \
            typename spmd_MaskType_>                                          \
  void name(nsimd_nat spmd_i_, spmd_MaskType_ spmd_mask_, __VA_ARGS__) {      \
    spmd_func_begin_

// templated kernel definition
#define spmd_tmpl_kernel_1d(name, template_argument, ...)                     \
  template <typename template_argument, typename spmd_KernelType_,            \
            int spmd_ScalarBits_, int spmd_N_, typename spmd_MaskType_>       \
  void name(nsimd_nat spmd_i_, spmd_MaskType_ spmd_mask_, __VA_ARGS__) {      \
    spmd_func_begin_

#define spmd_kernel_end }

// device function
#define spmd_dev_func(type_name, ...)                                         \
  template <typename spmd_KernelType_, int spmd_ScalarBits_, int spmd_N_,     \
            typename spmd_MaskType_>                                          \
  type_name(nsimd_nat spmd_i_, spmd_MaskType_ spmd_mask_, __VA_ARGS__) {      \
    spmd_func_begin_

// templated device function
#define spmd_tmpl_dev_func(type_name, template_argument, ...)                 \
  template <typename template_argument, typename spmd_KernelType_,            \
            int spmd_ScalarBits_, int spmd_N_, typename spmd_MaskType_>       \
  type_name(nsimd_nat spmd_i_, spmd_MaskType_ spmd_mask_, __VA_ARGS__) {      \
    spmd_func_begin_

#define spmd_dev_func_end }

// call spmd_dev_function
#define spmd_call_dev_func(name, ...)                                         \
  name<spmd_KernelType_, spmd_ScalarBits_, spmd_N_>(spmd_i_, spmd_mask_,      \
                                                    __VA_ARGS__)

// call templated spmd_dev_function
#define spmd_call_tmpl_dev_func(name, template_argument, ...)                 \
  name<template_argument, spmd_KernelType_, spmd_ScalarBits_, spmd_N_>(       \
      spmd_i_, spmd_mask_, __VA_ARGS__)

// launch 1d kernel
#define spmd_launch_kernel_1d(name, spmd_scalar_bits_, spmd_unroll_, spmd_n_, \
                              ...)                                            \
  {                                                                           \
    spmd::type_t<spmd::KernelSIMD, spmd_scalar_bits_, spmd_unroll_>::btype    \
        spmd_mask_(true);                                                     \
    nsimd_nat spmd_i_;                                                        \
    nsimd_nat len =                                                           \
        nsimd::len(spmd::type_t<spmd::KernelSIMD, spmd_scalar_bits_,          \
                                spmd_unroll_>::itype());                      \
    for (spmd_i_ = 0; spmd_i_ + len <= spmd_n_; spmd_i_ += len) {             \
      name<spmd::KernelSIMD, spmd_scalar_bits_, spmd_unroll_>(                \
          spmd_i_, spmd_mask_, __VA_ARGS__);                                  \
    }                                                                         \
    for (; spmd_i_ < spmd_n_; spmd_i_++) {                                    \
      name<spmd::KernelScalar, spmd_scalar_bits_, spmd_unroll_>(              \
          spmd_i_, true, __VA_ARGS__);                                        \
    }                                                                         \
  }

// launch 1d templated kernel
#define spmd_launch_tmpl_kernel_1d(                                           \
    name, template_argument, spmd_scalar_bits_, spmd_unroll_, spmd_n_, ...)   \
  {                                                                           \
    typename spmd::type_t<spmd::KernelSIMD, spmd_scalar_bits_,                \
                          spmd_unroll_>::btype spmd_mask_(true);              \
    nsimd_nat spmd_i_;                                                        \
    nsimd_nat len =                                                           \
        nsimd::len(typename spmd::type_t<spmd::KernelSIMD, spmd_scalar_bits_, \
                                         spmd_unroll_>::itype());             \
    for (spmd_i_ = 0; spmd_i_ + len <= spmd_n_; spmd_i_ += len) {             \
      name<template_argument, spmd::KernelSIMD, spmd_scalar_bits_,            \
           spmd_unroll_>(spmd_i_, spmd_mask_, __VA_ARGS__);                   \
    }                                                                         \
    for (; spmd_i_ < spmd_n_; spmd_i_++) {                                    \
      name<template_argument, spmd::KernelScalar, spmd_scalar_bits_,          \
           spmd_unroll_>(spmd_i_, true, __VA_ARGS__);                         \
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
#define k_int                                                                 \
  typename spmd::type_t<spmd_KernelType_, spmd_ScalarBits_, spmd_N_>::itype
#define k_uint                                                                \
  typename spmd::type_t<spmd_KernelType_, spmd_ScalarBits_, spmd_N_>::utype
#define k_float                                                               \
  typename spmd::type_t<spmd_KernelType_, spmd_ScalarBits_, spmd_N_>::ftype
#define k_bool                                                                \
  typename spmd::type_t<spmd_KernelType_, spmd_ScalarBits_, spmd_N_>::btype

// loads and stores (generic)
template <typename KernelType> struct store_helper {};
template <typename KernelType> struct load_helper {};
#define k_store(base_addr, value)                                             \
  spmd::store_helper<spmd_KernelType_>::impl(spmd_mask_, &base_addr[spmd_i_], \
                                             value)
#define k_unmasked_store(base_addr, value)                                    \
  spmd::store_helper<spmd_KernelType_>::unmasked_impl(&base_addr[spmd_i_],    \
                                                      value)

#define k_load(base_addr)                                                     \
  spmd::load_helper<spmd_KernelType_>::impl(spmd_mask_, &base_addr[spmd_i_])
#define k_unmasked_load(base_addr)                                            \
  spmd::load_helper<spmd_KernelType_>::template unmasked_impl<spmd_N_>(       \
      &base_addr[spmd_i_])

// loads and stores (scalar)
template <> struct store_helper<KernelScalar> {
  template <typename T, typename S>
  static void impl(bool mask, T *addr, S value) {
    if (mask) {
      *addr = nsimd::to<T>(value);
    }
  }

  template <typename T, typename S>
  static void unmasked_impl(T *addr, S value) {
    *addr = nsimd::to<T>(value);
  }
};

template <> struct load_helper<KernelScalar> {
  template <typename T> static T impl(bool mask, T *addr) {
    if (mask) {
      return *addr;
    } else {
      return nsimd::to<T>(0);
    }
  }

  template <int N, typename T> static T unmasked_impl(T *addr) {
    return *addr;
  }
};

template <> struct store_helper<KernelSIMD> {
  template <typename T, typename S, int N, typename SimdExt>
  static void impl(nsimd::packl<T, N, SimdExt> const &mask, S *addr,
                   nsimd::pack<S, N, SimdExt> const &value) {
    nsimd::mask_storeu(mask, addr, value);
  }

  template <typename T, typename S, typename U, int N, typename SimdExt>
  static void impl(nsimd::packl<T, N, SimdExt> const &mask, S *addr, U value) {
    nsimd::mask_storeu(mask, addr,
                       nsimd::pack<S, N, SimdExt>(nsimd::to<S>(value)));
  }

  template <typename T, int N, typename SimdExt>
  static void unmasked_impl(T *addr, nsimd::pack<T, N, SimdExt> const &value) {
    nsimd::storeu(addr, value);
  }

  template <typename T, typename S, int N, typename SimdExt>
  static void unmasked_impl(T *addr, S value) {
    nsimd::storeu(addr, nsimd::pack<T, N, SimdExt>(nsimd::to<T>(value)));
  }
};

template <> struct load_helper<KernelSIMD> {
  template <typename T, typename S, int N, typename SimdExt>
  static nsimd::pack<S, N, SimdExt>
  impl(nsimd::packl<T, N, SimdExt> const &mask, S *addr) {
    return nsimd::maskz_loadu(mask, addr);
  }

  template <int N, typename T>
  static nsimd::pack<T, N> unmasked_impl(T *addr) {
    return nsimd::loadu<nsimd::pack<T, N> >(addr);
  }
};

// f32 <--> f16 conversions
#define k_f32_to_f16(a) nsimd_f32_to_f16(a)
#define k_f16_to_f32(a) nsimd_f16_to_f32(a)

// Clear lanes
template <typename T, typename S, int N, typename SimdExt>
nsimd::packl<T, N, SimdExt>
clear_lanes(nsimd::packl<T, N, SimdExt> const &mask,
            nsimd::packl<S, N, SimdExt> const &lanes) {
  return nsimd::andnotl(mask, lanes);
}

inline bool clear_lanes(bool mask, bool lanes) { return lanes ? false : mask; }

// assignment statement
template <typename T, typename S> void k_set_(bool mask, T &var, S value) {
  if (mask) {
    var = nsimd::to<T>(value);
  }
}

template <typename T, typename S, int N, typename SimdExt, typename U>
void k_set_(nsimd::packl<T, N, SimdExt> const &mask,
            nsimd::pack<S, N, SimdExt> &var, U value) {
  var = nsimd::if_else(mask, nsimd::pack<S, N, SimdExt>(S(value)), var);
}

template <typename T, typename S, int N, typename SimdExt>
void k_set_(nsimd::packl<T, N, SimdExt> const &mask,
            nsimd::pack<S, N, SimdExt> &var,
            nsimd::pack<S, N, SimdExt> const &value) {
  var = nsimd::if_else(mask, value, var);
}

template <typename T, typename S, int N, typename SimdExt, typename U>
void k_set_(nsimd::packl<T, N, SimdExt> const &mask,
            nsimd::packl<S, N, SimdExt> &var, U value) {
  var = nsimd::reinterpretl<nsimd::packl<S, N, SimdExt> >(
      mask && nsimd::pack<S, N, SimdExt>(int(value)));
}

template <typename T, typename S, int N, typename SimdExt, typename U>
void k_set_(nsimd::packl<T, N, SimdExt> const &mask,
            nsimd::packl<S, N, SimdExt> &var,
            nsimd::packl<U, N, SimdExt> const &value) {
  var = nsimd::reinterpretl<nsimd::packl<S, N, SimdExt> >(mask && value);
}

#define k_set(var, value) spmd::k_set_(spmd_mask_, var, value)
#define k_unmasked_set(var, value)                                            \
  do {                                                                        \
    var = value;                                                              \
  } while (0)

template <typename T, int N, typename SimdExt>
bool any(nsimd::packl<T, N, SimdExt> const a) {
  return nsimd::any(a);
}

template <typename KernelType, int ScalarBits, int N, typename Packl>
typename type_t<KernelType, ScalarBits, N>::btype to_k_bool_(Packl const &a) {
  return nsimd::reinterpretl<
      typename type_t<KernelType, ScalarBits, N>::btype>(a);
}

template <typename KernelType, int ScalarBits, int N>
inline bool to_k_bool_(bool a) {
  return a;
}

#define k_to_bool(a)                                                          \
  spmd::to_k_bool_<spmd_KernelType_, spmd_ScalarBits_, spmd_N_>(a)

inline bool any(bool a) { return a; }

// while statement (k_while)
#define k_while(cond)                                                         \
  {                                                                           \
    k_bool spmd_middle_mask_ = spmd_mask_;                                    \
    k_bool spmd_off_lanes_break_(false);                                      \
    (void)spmd_off_lanes_break_;                                              \
    k_bool spmd_off_lanes_continue_(false);                                   \
    (void)spmd_off_lanes_continue_;                                           \
    {                                                                         \
      while (spmd::any(cond)) {                                               \
        k_bool spmd_cond_ =                                                   \
            spmd::to_k_bool_<spmd_KernelType_, spmd_ScalarBits_, spmd_N_>(    \
                cond);                                                        \
        {                                                                     \
          k_bool spmd_mask_ = spmd_cond_ && spmd_middle_mask_;                \
          spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_break_);  \
          spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_return_);

// break statement (k_break)
#define k_break                                                               \
  spmd_off_lanes_break_ = spmd_off_lanes_break_ || spmd_mask_;                \
  spmd_mask_ = false;

// continue statement (k_continue)
#define k_continue                                                            \
  spmd_off_lanes_continue_ = spmd_off_lanes_continue_ || spmd_mask_;          \
  spmd_mask_ = false;

// endwhile statement (k_endwhile)
#define k_endwhile                                                            \
  }                                                                           \
  }                                                                           \
  }                                                                           \
  }                                                                           \
  spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_return_);

// return statement (k_return)
#define k_return                                                              \
  spmd_off_lanes_return_ = spmd_off_lanes_return_ || spmd_mask_;              \
  spmd_mask_ = false;

// if statement (k_if)
#define k_if(cond)                                                            \
  {                                                                           \
    k_bool spmd_cond_ =                                                       \
        spmd::to_k_bool_<spmd_KernelType_, spmd_ScalarBits_, spmd_N_>(cond);  \
    k_bool spmd_middle_mask_ = spmd_mask_;                                    \
    {                                                                         \
      k_bool spmd_mask_ = spmd_cond_ && spmd_middle_mask_;

// elseif statement (k_elseif)
#define k_elseif(cond)                                                        \
  }                                                                           \
  spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_return_);         \
  spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_break_);          \
  spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_continue_);       \
  spmd_middle_mask_ = spmd::clear_lanes(spmd_middle_mask_, spmd_cond_);       \
  spmd_cond_ =                                                                \
      spmd::to_k_bool_<spmd_KernelType_, spmd_ScalarBits_, spmd_N_>(cond);    \
  {                                                                           \
    k_bool spmd_mask_ = spmd_cond_ && spmd_middle_mask_;

// else statement (k_else)
#define k_else                                                                \
  }                                                                           \
  spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_return_);         \
  spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_break_);          \
  spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_continue_);       \
  spmd_middle_mask_ = spmd::clear_lanes(spmd_middle_mask_, spmd_cond_);       \
  {                                                                           \
    k_bool spmd_mask_ = spmd_middle_mask_;

// endif statement (k_endif)
#define k_endif                                                               \
  }                                                                           \
  }                                                                           \
  spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_return_);         \
  spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_break_);          \
  spmd_mask_ = spmd::clear_lanes(spmd_mask_, spmd_off_lanes_continue_);

// ----------------------------------------------------------------------------

#endif

#ifdef NSIMD_VARIADIC_MACROS_IS_EXTENSION
#if defined(NSIMD_IS_GCC)
#pragma GCC diagnostic pop
#elif defined(NSIMD_IS_CLANG)
#pragma clang diagnostic pop
#endif
#endif

} // namespace spmd

#include <nsimd/modules/spmd/functions.hpp>

#endif
