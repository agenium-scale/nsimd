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

#if NSIMD_CXX < 2011 && NSIMD_C < 1999
  #define NSIMD_VARIADIC_MACROS_IS_EXTENSION
#endif

#ifdef NSIMD_VARIADIC_MACROS_IS_EXTENSION
  #if defined(NSIMD_IS_GCC)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wvariadic-macros"
  #elif defined(NSIMD_IS_CLANG)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wvariadic-macros"
  #endif
#endif

// ----------------------------------------------------------------------------
// CUDA and ROCm

#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) ||                               \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)


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
#define spmd_launch_tmpl_kernel_1d(                                      \
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
  static void impl(nsimd::packl<T, N, SimdExt> const &mask, S *addr,
                   U value) {
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
