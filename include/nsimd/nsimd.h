/*

Copyright (c) 2019 Agenium Scale

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

#ifndef NSIMD_H
#define NSIMD_H

/* ------------------------------------------------------------------------- */
/* Compiler detection (order matters https://stackoverflow.com/a/28166605) */

#if defined(_MSC_VER)
  #define NSIMD_IS_MSVC
#elif defined(__INTEL_COMPILER)
  #define NSIMD_IS_ICC
#elif defined(__clang__)
  #define NSIMD_IS_CLANG
#elif defined(__GNUC__) || defined(__GNUG__)
  #define NSIMD_IS_GCC
#elif defined(__NVCC__)
  #define NSIMD_IS_NVCC
#endif

/* ------------------------------------------------------------------------- */
/* Register size detection */

#if defined(__x86_64) || defined(__x86_64__) || defined(__amd64__) || \
    defined(__amd64) || defined(_M_AMD64) || defined(__aarch64__) || \
    defined (_M_ARM64) || defined(__PPC64__)
  #define NSIMD_WORD_SIZE 64
#else
  #define NSIMD_WORD_SIZE 32
#endif

/* ------------------------------------------------------------------------- */
/* Architecture detection */

#if defined(i386) || defined(__i386__) || defined(__i486__) || \
    defined(__i586__) || defined(__i686__) || defined(__i386) || \
    defined(_M_IX86) || defined(_X86_) || defined(__THW_INTEL__) || \
    defined(__I86__) || defined(__INTEL__) || defined(__x86_64) || \
    defined(__x86_64__) || defined(__amd64__) || defined(__amd64) || \
    defined(_M_X64)
  #define NSIMD_X86
#elif defined(__arm__) || defined(__arm64) || defined(__thumb__) || \
      defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB) || \
      defined(_M_ARM) || defined(_M_ARM64) || defined(__arch64__)
  #define NSIMD_ARM
#else
  #define NSIMD_CPU
#endif

/* ------------------------------------------------------------------------- */
/* C standard detection */

#ifdef NSIMD_IS_MSVC
  #define NSIMD_C 1999
#else
  #if __STDC_VERSION__ == 199901L
    #define NSIMD_C 1999
  #elif __STDC_VERSION__ >= 201112L
    #define NSIMD_C 2011
  #else
    #define NSIMD_C 1989
  #endif
#endif

/* ------------------------------------------------------------------------- */
/* C++ standard detection */

#ifdef NSIMD_IS_MSVC
  #define NSIMD__cplusplus _MSVC_LANG
#else
  #define NSIMD__cplusplus __cplusplus
#endif

#if NSIMD__cplusplus > 0 && NSIMD__cplusplus < 201103L
  #define NSIMD_CXX 1998
#elif NSIMD__cplusplus >= 201103L && NSIMD__cplusplus < 201402L
  #define NSIMD_CXX 2011
#elif NSIMD__cplusplus >= 201402L && NSIMD__cplusplus < 201703L
  #define NSIMD_CXX 2014
#elif NSIMD__cplusplus >= 201703L
  #define NSIMD_CXX 2017
#else
  #define NSIMD_CXX 0
#endif

/* ------------------------------------------------------------------------- */
/* Microsoft DLL specifics */

#ifdef NSIMD_IS_MSVC
  #define NSIMD_DLLEXPORT __declspec(dllexport)
  #define NSIMD_DLLIMPORT __declspec(dllimport)
#else
  #define NSIMD_DLLEXPORT
  #define NSIMD_DLLIMPORT extern
#endif

/* ------------------------------------------------------------------------- */
/* DLL specifics when inside/outside the library */

#ifdef NSIMD_INSIDE
  #define NSIMD_DLLSPEC NSIMD_DLLEXPORT
#else
  #define NSIMD_DLLSPEC NSIMD_DLLIMPORT
#endif

/* ------------------------------------------------------------------------- */
/* inline in nsimd is ONLY useful for linkage */

#if NSIMD_CXX > 0 || NSIMD_C > 1989
  #if NSIMD_C > 0 && defined(NSIMD_IS_MSVC)
    #define NSIMD_INLINE static __inline
  #else
    #define NSIMD_INLINE static inline
  #endif
#else
  #if defined(NSIMD_IS_GCC) || defined(NSIMD_IS_CLANG)
    #define NSIMD_INLINE __extension__ __inline__
  #else
    #define NSIMD_INLINE
  #endif
#endif

/* ------------------------------------------------------------------------- */
/* Pre-processor */

#define NSIMD_PP_CAT_2_e(a, b) a##b
#define NSIMD_PP_CAT_2(a, b) NSIMD_PP_CAT_2_e(a, b)

#define NSIMD_PP_CAT_3_e(a, b, c) a##b##c
#define NSIMD_PP_CAT_3(a, b, c) NSIMD_PP_CAT_3_e(a, b, c)

#define NSIMD_PP_CAT_4_e(a, b, c, d) a##b##c##d
#define NSIMD_PP_CAT_4(a, b, c, d) NSIMD_PP_CAT_4_e(a, b, c, d)

#define NSIMD_PP_CAT_5_e(a, b, c, d, e) a##b##c##d##e
#define NSIMD_PP_CAT_5(a, b, c, d, e) NSIMD_PP_CAT_5_e(a, b, c, d, e)

#define NSIMD_PP_CAT_6_e(a, b, c, d, e, f) a##b##c##d##e##f
#define NSIMD_PP_CAT_6(a, b, c, d, e, f) NSIMD_PP_CAT_6_e(a, b, c, d, e, f)

#define NSIMD_PP_EXPAND_e(a) a
#define NSIMD_PP_EXPAND(a) NSIMD_PP_EXPAND_e(a)

/* ------------------------------------------------------------------------- */
/* Detect architecture/SIMD */

/* Intel */

#if defined(SSE2) && !defined(NSIMD_SSE2)
  #define NSIMD_SSE2
#endif

#if defined(SSE42) && !defined(NSIMD_SSE42)
  #define NSIMD_SSE42
#endif

#if defined(AVX) && !defined(NSIMD_AVX)
  #define NSIMD_AVX
#endif

#if defined(AVX2) && !defined(NSIMD_AVX2)
  #define NSIMD_AVX2
#endif

#if defined(AVX512_KNL) && !defined(NSIMD_AVX512_KNL)
  #define NSIMD_AVX512_KNL
#endif

#if defined(AVX512_SKYLAKE) && !defined(NSIMD_AVX512_SKYLAKE)
  #define NSIMD_AVX512_SKYLAKE
#endif

#if defined(FP16) && !defined(NSIMD_FP16)
  #define NSIMD_FP16
#endif

#if defined(FMA) && !defined(NSIMD_FMA)
  #define NSIMD_FMA
#endif

/* ARM */

#if defined(NEON128) && !defined(NSIMD_NEON128)
  #define NSIMD_NEON128
#endif

#if defined(AARCH64) && !defined(NSIMD_AARCH64)
  #define NSIMD_AARCH64
#endif

#if defined(SVE) && !defined(NSIMD_SVE)
  #define NSIMD_SVE
#endif

/* PPC */

#if (defined(VMX) || defined(ALTIVEC)) && !defined(NSIMD_VMX)
  #define NSIMD_VMX
#endif

#if defined(VSX) && !defined(NSIMD_VSX)
  #define NSIMD_VSX
#endif

/* CUDA */

#if defined(CUDA) && !defined(NSIMD_CUDA)
  #define NSIMD_CUDA
#endif

/* ------------------------------------------------------------------------- */
/* Set NSIMD_SIMD and NSIMD_PLATFORM macro, include the correct header. */

#if defined(NSIMD_SSE2)

  #define NSIMD_PLATFORM x86
  #define NSIMD_SIMD sse2
  #include <emmintrin.h>
  #if defined(NSIMD_FMA) || defined(NSIMD_FP16)
    #include <immintrin.h>
  #endif
  /* For some reason MSVC <= 2015 has intrinsics defined in another header */
  #ifdef NSIMD_IS_MSVC
    #include <intrin.h>
  #endif

#elif defined(NSIMD_SSE42)

  #define NSIMD_PLATFORM x86
  #define NSIMD_SIMD sse42
  #include <nmmintrin.h>
  #if defined(NSIMD_FMA) || defined(NSIMD_FP16)
    #include <immintrin.h>
  #endif
  /* For some reason MSVC <= 2015 has intrinsics defined in another header */
  #ifdef NSIMD_IS_MSVC
    #include <intrin.h>
  #endif

#elif defined(NSIMD_AVX)

  #define NSIMD_PLATFORM x86
  #define NSIMD_SIMD avx
  #include <immintrin.h>
  /* For some reason MSVC <= 2015 has intrinsics defined in another header */
  #ifdef NSIMD_IS_MSVC
    #include <intrin.h>
  #endif

#elif defined(NSIMD_AVX2)

  #define NSIMD_PLATFORM x86
  #define NSIMD_SIMD avx2
  #include <immintrin.h>
  /* For some reason MSVC <= 2015 has intrinsics defined in another header */
  #ifdef NSIMD_IS_MSVC
    #include <intrin.h>
  #endif

#elif defined(NSIMD_AVX512_KNL)

  #define NSIMD_PLATFORM x86
  #define NSIMD_SIMD avx512_knl
  #include <immintrin.h>

#elif defined(NSIMD_AVX512_SKYLAKE)

  #define NSIMD_PLATFORM x86
  #define NSIMD_SIMD avx512_skylake
  #include <immintrin.h>

#elif defined(NSIMD_NEON128)

  #define NSIMD_PLATFORM arm
  #define NSIMD_SIMD neon128
  #include <arm_neon.h>

#elif defined(NSIMD_AARCH64)

  #define NSIMD_PLATFORM arm
  #define NSIMD_SIMD aarch64
  #include <arm_neon.h>

#elif defined(NSIMD_SVE)

  #define NSIMD_PLATFORM arm
  #define NSIMD_SIMD sve
  #include <arm_neon.h>
  #include <arm_sve.h>

#else

  #define NSIMD_SIMD cpu
  #define NSIMD_PLATFORM cpu

#endif

/* ------------------------------------------------------------------------- */
/* Shorter typedefs for integers */

#ifdef NSIMD_IS_MSVC
  typedef unsigned __int8  u8;
  typedef   signed __int8  i8;
  typedef unsigned __int16 u16;
  typedef   signed __int16 i16;
  typedef unsigned __int32 u32;
  typedef   signed __int32 i32;
  typedef unsigned __int64 u64;
  typedef   signed __int64 i64;
#else
  typedef unsigned char  u8;
  typedef   signed char  i8;
  typedef unsigned short u16;
  typedef   signed short i16;
  typedef unsigned int   u32;
  typedef   signed int   i32;
  #if NSIMD_WORD_SIZE == 64
    typedef unsigned long u64;
    typedef   signed long i64;
  #else
    #if defined(NSIMD_IS_GCC) || defined(NSIMD_IS_CLANG)
      __extension__ typedef unsigned long long u64;
      __extension__ typedef   signed long long i64;
    #else
      typedef unsigned long long u64;
      typedef   signed long long i64;
    #endif
  #endif
#endif

/* ------------------------------------------------------------------------- */
/* Sorter typedefs for floatting point types */

#if ((defined(NSIMD_NEON128) || defined(NSIMD_AARCH64)) && \
     defined(NSIMD_FP16)) || defined(NSIMD_SVE)
  #define NSIMD_NATIVE_FP16
#endif

#ifdef NSIMD_NATIVE_FP16
  typedef __fp16 f16;
#else
  typedef struct {
    u16 u;
  } f16;
#endif

typedef float  f32;
typedef double f64;

/* ------------------------------------------------------------------------- */
/* Native register size (for now only 32 and 64 bits) types */

#if NSIMD_WORD_SIZE == 64
  typedef i64 nat;
#else
  typedef i32 nat;
#endif

/* ------------------------------------------------------------------------- */
/* POPCNT: GCC and Clang have intrinsics */

#if defined(NSIMD_IS_GCC) || defined(NSIMD_IS_CLANG)

NSIMD_INLINE int nsimd_popcnt32_(u32 a) {
  return __builtin_popcount(a);
}

NSIMD_INLINE int nsimd_popcnt64_(u64 a) {
#if __SIZEOF_LONG__ == 4
  return __builtin_popcountl((u32)(a & 0xFFFFFFFF)) +
         __builtin_popcountl((u32)(a >> 32));
#else
  return __builtin_popcountl(a);
#endif
}

/* ------------------------------------------------------------------------- */
/* POPCNT: MSVC has also an intrinsic for that */

#elif defined(NSIMD_IS_MSVC)

#include <intrin.h>

NSIMD_INLINE int nsimd_popcnt32_(u32 a) {
  return (int)__popcnt(a);
}

NSIMD_INLINE int nsimd_popcnt64_(u64 a) {
  return (int)__popcnt64(a);
}

/* ------------------------------------------------------------------------- */
/* POPCNT: Default naive implementation */

#else

NSIMD_INLINE int nsimd_popcnt32_(u32 a) {
  int i, ret = 0;
  for (i = 0; i < 32; i++) {
    ret += (a >> i) & 1;
  }
  return ret;
}

NSIMD_INLINE int nsimd_popcnt64_(u64 a) {
  int i, ret = 0;
  for (i = 0; i < 64; i++) {
    ret += (a >> i) & 1;
  }
  return ret;
}

#endif

/* ------------------------------------------------------------------------- */
/* Macro to automatically include function depending on detected
   platform/SIMD */

#define NSIMD_AUTO_INCLUDE(path) <nsimd/NSIMD_PLATFORM/NSIMD_SIMD/path>

/* ------------------------------------------------------------------------- */
/* Standard includes */

#if NSIMD_CXX > 0
  #include <cerrno>
  #include <cstdlib>
#else
  #include <errno.h>
  #include <stdlib.h>
#endif

/* ------------------------------------------------------------------------- */
/* Now includes detected SIMD types */

#if NSIMD_CXX > 0

namespace nsimd {
template <typename T, typename SimdExt> struct simd_traits {};
} // namespace nsimd

#endif

#include NSIMD_AUTO_INCLUDE(types.h)

/* ------------------------------------------------------------------------- */
/* Macro/typedefs for SIMD infos */

#define vec(T) NSIMD_PP_CAT_4(nsimd_, NSIMD_SIMD, _v, T)
#define vecl(T) NSIMD_PP_CAT_4(nsimd_, NSIMD_SIMD, _vl, T)

#define vecx2(T) NSIMD_PP_CAT_5(nsimd_, NSIMD_SIMD, _v, T, x2)
#define vecx3(T) NSIMD_PP_CAT_5(nsimd_, NSIMD_SIMD, _v, T, x3)
#define vecx4(T) NSIMD_PP_CAT_5(nsimd_, NSIMD_SIMD, _v, T, x4)

typedef vec(i8) vi8;
typedef vec(i8) vu8;
typedef vec(i16) vi16;
typedef vec(i16) vu16;
typedef vec(i32) vi32;
typedef vec(i32) vu32;
typedef vec(i64) vi64;
typedef vec(i64) vu64;
typedef vec(f16) vf16;
typedef vec(f32) vf32;
typedef vec(f64) vf64;

typedef vecx2(i8) vi8x2;
typedef vecx2(i8) vu8x2;
typedef vecx2(i16) vi16x2;
typedef vecx2(i16) vu16x2;
typedef vecx2(i32) vi32x2;
typedef vecx2(i32) vu32x2;
typedef vecx2(i64) vi64x2;
typedef vecx2(i64) vu64x2;
typedef vecx2(f16) vf16x2;
typedef vecx2(f32) vf32x2;
typedef vecx2(f64) vf64x2;

typedef vecx3(i8) vi8x3;
typedef vecx3(i8) vu8x3;
typedef vecx3(i16) vi16x3;
typedef vecx3(i16) vu16x3;
typedef vecx3(i32) vi32x3;
typedef vecx3(i32) vu32x3;
typedef vecx3(i64) vi64x3;
typedef vecx3(i64) vu64x3;
typedef vecx3(f16) vf16x3;
typedef vecx3(f32) vf32x3;
typedef vecx3(f64) vf64x3;

typedef vecx4(i8) vi8x4;
typedef vecx4(i8) vu8x4;
typedef vecx4(i16) vi16x4;
typedef vecx4(i16) vu16x4;
typedef vecx4(i32) vi32x4;
typedef vecx4(i32) vu32x4;
typedef vecx4(i64) vi64x4;
typedef vecx4(i64) vu64x4;
typedef vecx4(f16) vf16x4;
typedef vecx4(f32) vf32x4;
typedef vecx4(f64) vf64x4;

typedef vecl(i8) vli8;
typedef vecl(i8) vlu8;
typedef vecl(i16) vli16;
typedef vecl(i16) vlu16;
typedef vecl(i32) vli32;
typedef vecl(i32) vlu32;
typedef vecl(i64) vli64;
typedef vecl(i64) vlu64;
typedef vecl(f16) vlf16;
typedef vecl(f32) vlf32;
typedef vecl(f64) vlf64;

#define vec_a(T, simd_ext) NSIMD_PP_CAT_4(nsimd_, simd_ext, _v, T)
#define vecl_a(T, simd_ext) NSIMD_PP_CAT_4(nsimd_, simd_ext, _vl, T)

#if NSIMD_CXX > 0

namespace nsimd {

/* Alignment tags */
struct aligned {};
struct unaligned {};

#if NSIMD_CXX >= 2011

template <typename T>
using simd_vector = typename simd_traits<T, NSIMD_SIMD>::simd_vector;

template <typename T>
using simd_vectorl = typename simd_traits<T, NSIMD_SIMD>::simd_vectorl;

#endif

} // namespace nsimd

#endif

#if defined(NSIMD_X86)
  #define NSIMD_MAX_ALIGNMENT 64
#elif defined(NSIMD_ARM)
  #define NSIMD_MAX_ALIGNMENT 256
#else
  #define NSIMD_MAX_ALIGNMENT 16
#endif

#define NSIMD_NB_REGISTERS  NSIMD_PP_CAT_3(NSIMD_, NSIMD_SIMD, _NB_REGISTERS)

#define NSIMD_MAX_LEN_BIT  2048

#define NSIMD_MAX_LEN_i8   (NSIMD_MAX_LEN_BIT / 8)
#define NSIMD_MAX_LEN_u8   (NSIMD_MAX_LEN_BIT / 8)
#define NSIMD_MAX_LEN_i16  (NSIMD_MAX_LEN_BIT / 16)
#define NSIMD_MAX_LEN_u16  (NSIMD_MAX_LEN_BIT / 16)
#define NSIMD_MAX_LEN_f16  (NSIMD_MAX_LEN_BIT / 16)
#define NSIMD_MAX_LEN_i32  (NSIMD_MAX_LEN_BIT / 32)
#define NSIMD_MAX_LEN_u32  (NSIMD_MAX_LEN_BIT / 32)
#define NSIMD_MAX_LEN_f32  (NSIMD_MAX_LEN_BIT / 32)
#define NSIMD_MAX_LEN_i64  (NSIMD_MAX_LEN_BIT / 64)
#define NSIMD_MAX_LEN_u64  (NSIMD_MAX_LEN_BIT / 64)
#define NSIMD_MAX_LEN_f64  (NSIMD_MAX_LEN_BIT / 64)

#define NSIMD_MAX_LEN_e(typ) NSIMD_MAX_LEN_ ## typ
#define NSIMD_MAX_LEN(typ) NSIMD_MAX_LEN_e(typ)

#if NSIMD_CXX > 0
namespace nsimd {

template <typename T> struct max_len_t {};

template <> struct max_len_t<i8> {
  static const int value = NSIMD_MAX_LEN_BIT / 8;
};
template <> struct max_len_t<u8> {
  static const int value = NSIMD_MAX_LEN_BIT / 8;
};
template <> struct max_len_t<i16> {
  static const int value = NSIMD_MAX_LEN_BIT / 16;
};
template <> struct max_len_t<u16> {
  static const int value = NSIMD_MAX_LEN_BIT / 16;
};
template <> struct max_len_t<f16> {
  static const int value = NSIMD_MAX_LEN_BIT / 16;
};
template <> struct max_len_t<i32> {
  static const int value = NSIMD_MAX_LEN_BIT / 32;
};
template <> struct max_len_t<u32> {
  static const int value = NSIMD_MAX_LEN_BIT / 32;
};
template <> struct max_len_t<f32> {
  static const int value = NSIMD_MAX_LEN_BIT / 32;
};
template <> struct max_len_t<i64> {
  static const int value = NSIMD_MAX_LEN_BIT / 64;
};
template <> struct max_len_t<u64> {
  static const int value = NSIMD_MAX_LEN_BIT / 64;
};
template <> struct max_len_t<f64> {
  static const int value = NSIMD_MAX_LEN_BIT / 64;
};

#if NSIMD_CXX >= 2014
template <typename T> constexpr int max_len = max_len_t<T>::value;
#endif

} // namespace nsimd
#endif

/* ------------------------------------------------------------------------- */
/* Memory functions */

#if NSIMD_CXX > 0
#include <cstddef>
#include <new>
#endif

/* ------------------------------------------------------------------------- */

#if NSIMD_CXX > 0
extern "C" {
#endif

NSIMD_DLLSPEC void *nsimd_aligned_alloc(nat);
NSIMD_DLLSPEC void nsimd_aligned_free(void *);

#if NSIMD_CXX > 0
} // extern "C"
#endif

/* ------------------------------------------------------------------------- */
/* C++ templated functions */

#if NSIMD_CXX > 0
namespace nsimd {

NSIMD_DLLSPEC void *aligned_alloc(nat);
NSIMD_DLLSPEC void aligned_free(void *);

template <typename T> T *aligned_alloc_for(nat n) {
  return (T *)aligned_alloc(n * (nat)sizeof(T));
}

template <typename T> void aligned_free_for(void *ptr) {
  return aligned_free((T *)ptr);
}

} // namespace nsimd
#endif

/* ------------------------------------------------------------------------- */
/* C++ <11 allocator */

#if NSIMD_CXX > 0 && NSIMD_CXX < 2011
namespace nsimd {

template <typename T> class allocator {
public:
  typedef T value_type;
  typedef value_type *pointer;
  typedef const value_type *const_pointer;
  typedef value_type &reference;
  typedef const value_type &const_reference;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;

public:
  template <typename U> struct rebind { typedef allocator<U> other; };

public:
  inline allocator() {}
  inline ~allocator() {}
  inline allocator(allocator const &) {}

  template <typename U> inline explicit allocator(allocator<U> const &) {}

  inline pointer address(reference r) { return &r; }
  inline const_pointer address(const_reference r) { return &r; }

  inline pointer allocate(size_type n) {
    return reinterpret_cast<pointer>(aligned_alloc_for<T>((nat)n));
  }

  inline pointer allocate(size_type n, const void *) {
    return allocate(n);
  }

  inline void deallocate(pointer p, size_type) {
    aligned_free_for<T>(p);
  }

  inline size_type max_size() const {
    return size_type(-1) / sizeof(T);
  }

  inline void construct(pointer p, const T &t) { new (p) T(t); }
  inline void destroy(pointer p) { p->~T(); }

  inline bool operator==(allocator const &) { return true; }
  inline bool operator!=(allocator const &a) { return !operator==(a); }
};

} // namespace nsimd
#endif

/* ------------------------------------------------------------------------- */
/* C++ >=11 allocator */

#if NSIMD_CXX >= 2011
namespace nsimd {

template <typename T> struct allocator {
  using value_type = T;

  allocator() = default;

  template <typename S> allocator(allocator<S> const &) {}

  T *allocate(std::size_t n) {
    if (n > std::size_t(-1) / sizeof(T)) {
      throw std::bad_alloc();
    }
    T *ptr = aligned_alloc_for<T>((nat)n);
    if (ptr != NULL) {
      return ptr;
    }
    throw std::bad_alloc();
  }

  void deallocate(T *ptr, std::size_t) { nsimd::aligned_free(ptr); }
};

template <class T, class S>
bool operator==(allocator<T> const &, allocator<S> const &) {
  return true;
}

template <class T, class S>
bool operator!=(allocator<T> const &, allocator<S> const &) {
  return false;
}

} // namespace nsimd
#endif

/* ------------------------------------------------------------------------- */
/* Conversion functions f16 <---> f32 for C */

#if NSIMD_CXX > 0
extern "C" {
#endif

NSIMD_DLLSPEC u16 nsimd_f32_to_u16(f32);
NSIMD_DLLSPEC f32 nsimd_u16_to_f32(u16);

#ifdef NSIMD_NATIVE_FP16
NSIMD_INLINE f16 nsimd_f32_to_f16(f32 a) { return (f16)a; }
NSIMD_INLINE f32 nsimd_f16_to_f32(f16 a) { return (f32)a; }
#else
NSIMD_DLLSPEC f16 nsimd_f32_to_f16(f32);
NSIMD_DLLSPEC f32 nsimd_f16_to_f32(f16);
#endif

#if NSIMD_CXX > 0
} // extern "C"
#endif

/* ------------------------------------------------------------------------- */
/* Conversion functions f16 <---> f32 for C++ */

#if NSIMD_CXX > 0
namespace nsimd {
  NSIMD_DLLSPEC u16 f32_to_u16(f32);
  NSIMD_DLLSPEC f32 u16_to_f32(u16);

  #ifdef NSIMD_NATIVE_FP16
    NSIMD_INLINE f16 f32_to_f16(f32 a) { return (f16)a; }
    NSIMD_INLINE f32 f16_to_f32(f16 a) { return (f32)a; }
  #else
    NSIMD_DLLSPEC f16 f32_to_f16(f32);
    NSIMD_DLLSPEC f32 f16_to_f32(f16);
  #endif
} // namespace nsimd
#endif

/* ------------------------------------------------------------------------- */
/* SIMD-related functions */

#ifdef NSIMD_IS_MSVC
  #pragma warning(push)
  /* We do not want MSVC to warn us about unary minus on an unsigned type.
     It is well defined in standards: unsigned arithmetic is done modulo
     2^n. */
  #pragma warning(disable : 4146)
#endif

#include <nsimd/functions.h>

#ifdef NSIMD_IS_MSVC
  #pragma warning(pop)
#endif

/* ------------------------------------------------------------------------- */
/* If_else cannot be auto-generated */

#define vif_else(a0, a1, a2, typel, type)                                      \
  NSIMD_PP_CAT_4(nsimd_if_else1_, NSIMD_SIMD, _, type)                         \
  (NSIMD_PP_CAT_6(nsimd_vreinterpretl_, NSIMD_SIMD, _, type, _, typel)(a0),    \
   a1, a2)

#define vif_else_e(a0, a1, a2, typel, type, simd_ext)                          \
  NSIMD_PP_CAT_4(nsimd_if_else1_, simd_ext, _, type)                           \
  (NSIMD_PP_CAT_6(nsimd_vreinterpretl_, simd_ext, _, type, _, typel)(a0), a1,  \
   a2)

#if NSIMD_CXX > 0
namespace nsimd {

template <typename A0, typename A1, typename A2, typename L, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vector if_else(A0 a0, A1 a1, A2 a2, L,
                                                         T) {
  return if_else1(reinterpretl(a0, L(), T(), NSIMD_SIMD()), a1, a2, T(),
                  NSIMD_SIMD());
}

template <typename A0, typename A1, typename A2, typename L, typename T,
          typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vector if_else(A0 a0, A1 a1, A2 a2, L,
                                                      T, SimdExt) {
  return if_else1(reinterpretl(a0, L(), T(), SimdExt()), a1, a2, T(),
                  SimdExt());
}

} // namespace nsimd
#endif

/* ------------------------------------------------------------------------- */
/* Loads/stores can be parametrized/templated by the alignment */

#define NSIMD_ALIGNED a
#define NSIMD_UNALIGNED u

#define vload(a0, type, alignment)                                             \
  NSIMD_PP_CAT_6(nsimd_load, alignment, _, NSIMD_SIMD, _, type)(a0)

#define vload_e(a0, type, simd_ext, alignment)                                 \
  NSIMD_PP_CAT_6(nsimd_load, alignment, _, simd_ext, _, type)(a0)

#define vload2(a0, type, alignment)                                            \
  NSIMD_PP_CAT_6(nsimd_load2, alignment, _, NSIMD_SIMD, _, type)(a0)

#define vload2_e(a0, type, simd_ext, alignment)                                \
  NSIMD_PP_CAT_6(nsimd_load2, alignment, _, simd_ext, _, type)(a0)

#define vload3(a0, type, alignment)                                            \
  NSIMD_PP_CAT_6(nsimd_load3, alignment, _, NSIMD_SIMD, _, type)(a0)

#define vload3_e(a0, type, simd_ext, alignment)                                \
  NSIMD_PP_CAT_6(nsimd_load3, alignment, _, simd_ext, _, type)(a0)

#define vload4(a0, type, alignment)                                            \
  NSIMD_PP_CAT_6(nsimd_load4, alignment, _, NSIMD_SIMD, _, type)(a0)

#define vload4_e(a0, type, simd_ext, alignment)                                \
  NSIMD_PP_CAT_6(nsimd_load4, alignment, _, simd_ext, _, type)(a0)

#define vloadl(a0, type, alignment)                                            \
  NSIMD_PP_CAT_6(nsimd_loadl, alignment_, NSIMD_SIMD, _, type)(a0)

#define vloadl_e(a0, type, simd_ext, alignment)                                \
  NSIMD_PP_CAT_6(nsimd_loadl, alignment_, simd_ext, _, type)(a0)

#define vstore(a0, a1, type, alignment)                                        \
  NSIMD_PP_CAT_6(nsimd_store, alignment, _, NSIMD_SIMD, _, type)(a0, a1)

#define vstore_e(a0, a1, type, simd_ext, alignment)                            \
  NSIMD_PP_CAT_6(nsimd_store, alignment, _, simd_ext, _, type)(a0, a1)

#define vstore2(a0, a1, a2, type, alignment)                                   \
  NSIMD_PP_CAT_4(nsimd_store2, alignment, _, NSIMD_SIMD, _, type)(a0, a1, a2)

#define vstore2_e(a0, a1, a2, type, simd_ext, alignment)                       \
  NSIMD_PP_CAT_4(nsimd_store2, alignment, _, simd_ext, _, type)(a0, a1, a2)

#define vstore3(a0, a1, a2, a3, type, alignment)                               \
  NSIMD_PP_CAT_4(nsimd_store3, alignment, _, NSIMD_SIMD, _, type)              \
  (a0, a1, a2, a3)

#define vstore3_e(a0, a1, a2, a3, type, simd_ext, alignment)                   \
  NSIMD_PP_CAT_4(nsimd_store3, alignment, _, simd_ext, _, type)(a0, a1, a2, a3)

#define vstore4(a0, a1, a2, a3, a4, type, alignment)                           \
  NSIMD_PP_CAT_4(nsimd_store3, alignment, _, NSIMD_SIMD, _, type)              \
  (a0, a1, a2, a3, a4)

#define vstore4_e(a0, a1, a2, a3, a4, type, simd_ext, alignment)               \
  NSIMD_PP_CAT_4(nsimd_store3, alignment, _, simd_ext, _, type)                \
  (a0, a1, a2, a3, a4)

#define vstorel(a0, a1, type, alignment)                                       \
  NSIMD_PP_CAT_6(nsimd_storel, alignment, _, NSIMD_SIMD, _, type)(a0, a1)

#define vstorel_e(a0, a1, type, simd_ext, alignment)                           \
  NSIMD_PP_CAT_6(nsimd_storel, alignment, _, simd_ext, _, type)(a0, a1)

#if NSIMD_CXX > 0
namespace nsimd {

template <typename A0, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vector load(A0 a0, T, aligned) {
  return loada(a0, T(), NSIMD_SIMD());
}

template <typename A0, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vector load(A0 a0, T, unaligned) {
  return loadu(a0, T(), NSIMD_SIMD());
}

template <typename A0, typename T, typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vector load(A0 a0, T, SimdExt,
                                                      aligned) {
  return loada(a0, T(), SimdExt());
}

template <typename A0, typename T, typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vector load(A0 a0, T, SimdExt,
                                                      unaligned) {
  return loadu(a0, T(), SimdExt());
}

template <typename A0, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vectorx2 load2(A0 a0, T, aligned) {
  return load2a(a0, T(), NSIMD_SIMD());
}

template <typename A0, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vectorx2 load2(A0 a0, T, unaligned) {
  return load2u(a0, T(), NSIMD_SIMD());
}

template <typename A0, typename T, typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vectorx2 load2(A0 a0, T, SimdExt,
                                                         aligned) {
  return load2a(a0, T(), SimdExt());
}

template <typename A0, typename T, typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vectorx2 load2(A0 a0, T, SimdExt,
                                                         unaligned) {
  return load2u(a0, T(), SimdExt());
}

template <typename A0, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vectorx3 load3(A0 a0, T, aligned) {
  return load3a(a0, T(), NSIMD_SIMD());
}

template <typename A0, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vectorx3 load3(A0 a0, T, unaligned) {
  return load3u(a0, T(), NSIMD_SIMD());
}

template <typename A0, typename T, typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vectorx3 load3(A0 a0, T, SimdExt,
                                                         aligned) {
  return load3a(a0, T(), SimdExt());
}

template <typename A0, typename T, typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vectorx3 load3(A0 a0, T, SimdExt,
                                                         unaligned) {
  return load3u(a0, T(), SimdExt());
}

template <typename A0, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vectorx4 load4(A0 a0, T, aligned) {
  return load4a(a0, T(), NSIMD_SIMD());
}

template <typename A0, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vectorx4 load4(A0 a0, T, unaligned) {
  return load4u(a0, T(), NSIMD_SIMD());
}

template <typename A0, typename T, typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vectorx4 load4(A0 a0, T, SimdExt,
                                                         aligned) {
  return load4a(a0, T(), SimdExt());
}

template <typename A0, typename T, typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vectorx4 load4(A0 a0, T, SimdExt,
                                                         unaligned) {
  return load4u(a0, T(), SimdExt());
}

template <typename A0, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vectorl loadlu(A0 a0, T, aligned) {
  return loadla(a0, T(), NSIMD_SIMD());
}

template <typename A0, typename T>
typename simd_traits<T, NSIMD_SIMD>::simd_vectorl loadlu(A0 a0, T, unaligned) {
  return loadlu(a0, T(), NSIMD_SIMD());
}

template <typename A0, typename T, typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vectorl loadlu(A0 a0, T, SimdExt,
                                                      aligned) {
  return loadla(a0, T(), SimdExt());
}

template <typename A0, typename T, typename SimdExt>
typename simd_traits<T, SimdExt>::simd_vectorl loadlu(A0 a0, T, SimdExt,
                                                      unaligned) {
  return loadlu(a0, T(), SimdExt());
}

template <typename A0, typename A1, typename T>
void store(A0 a0, A1 a1, T, aligned) {
  storea(a0, a1, T(), NSIMD_SIMD());
}

template <typename A0, typename A1, typename T>
void store(A0 a0, A1 a1, T, unaligned) {
  storeu(a0, a1, T(), NSIMD_SIMD());
}

template <typename A0, typename A1, typename T, typename SimdExt>
void store(A0 a0, A1 a1, T, SimdExt, aligned) {
  storea(a0, a1, T(), SimdExt());
}

template <typename A0, typename A1, typename T, typename SimdExt>
void store(A0 a0, A1 a1, T, SimdExt, unaligned) {
  storeu(a0, a1, T(), SimdExt());
}

template <typename A0, typename A1, typename A2, typename T>
void store2(A0 a0, A1 a1, A2 a2, T, aligned) {
  store2a(a0, a1, a2, T(), NSIMD_SIMD());
}

template <typename A0, typename A1, typename A2, typename T>
void store2(A0 a0, A1 a1, A2 a2, T, unaligned) {
  store2u(a0, a1, a2, T(), NSIMD_SIMD());
}

template <typename A0, typename A1, typename A2, typename T, typename SimdExt>
void store2(A0 a0, A1 a1, A2 a2, T, SimdExt, aligned) {
  store2a(a0, a1, a2, T(), SimdExt());
}

template <typename A0, typename A1, typename A2, typename T, typename SimdExt>
void store2(A0 a0, A1 a1, A2 a2, T, SimdExt, unaligned) {
  store2u(a0, a1, a2, T(), SimdExt());
}

template <typename A0, typename A1, typename A2, typename A3, typename T>
void store3(A0 a0, A1 a1, A2 a2, A3 a3, T, aligned) {
  store3a(a0, a1, a2, a3, T(), NSIMD_SIMD());
}

template <typename A0, typename A1, typename A2, typename A3, typename T>
void store3(A0 a0, A1 a1, A2 a2, A3 a3, T, unaligned) {
  store3u(a0, a1, a2, a3, T(), NSIMD_SIMD());
}

template <typename A0, typename A1, typename A2, typename A3, typename T,
          typename SimdExt>
void store3(A0 a0, A1 a1, A2 a2, A3 a3, T, SimdExt, aligned) {
  store3a(a0, a1, a2, a3, T(), SimdExt());
}

template <typename A0, typename A1, typename A2, typename A3, typename T,
          typename SimdExt>
void store3(A0 a0, A1 a1, A2 a2, A3 a3, T, SimdExt, unaligned) {
  store3u(a0, a1, a2, a3, T(), SimdExt());
}

template <typename A0, typename A1, typename A2, typename A3, typename A4,
          typename T>
void store4(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, T, aligned) {
  store4a(a0, a1, a2, a3, a4, T(), NSIMD_SIMD());
}

template <typename A0, typename A1, typename A2, typename A3, typename A4,
          typename T>
void store4(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, T, unaligned) {
  store4u(a0, a1, a2, a3, a4, T(), NSIMD_SIMD());
}

template <typename A0, typename A1, typename A2, typename A3, typename A4,
          typename T, typename SimdExt>
void store4(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, T, SimdExt, aligned) {
  store4a(a0, a1, a2, a3, a4, T(), SimdExt());
}

template <typename A0, typename A1, typename A2, typename A3, typename A4,
          typename T, typename SimdExt>
void store4(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4, T, SimdExt, unaligned) {
  store4u(a0, a1, a2, a3, a4, T(), SimdExt());
}

template <typename A0, typename A1, typename T>
void storel(A0 a0, A1 a1, T, aligned) {
  storela(a0, a1, T(), NSIMD_SIMD());
}

template <typename A0, typename A1, typename T>
void storel(A0 a0, A1 a1, T, unaligned) {
  storelu(a0, a1, T(), NSIMD_SIMD());
}

template <typename A0, typename A1, typename T, typename SimdExt>
void storel(A0 a0, A1 a1, T, SimdExt, aligned) {
  storela(a0, a1, T(), SimdExt());
}

template <typename A0, typename A1, typename T, typename SimdExt>
void storel(A0 a0, A1 a1, T, SimdExt, unaligned) {
  storelu(a0, a1, T(), SimdExt());
}

} // namespace nsimd
#endif

/* ------------------------------------------------------------------------- */
/* Endianess */
/* TODO */
#define ENDIAN_LITTLE_BYTE 1
/* ------------------------------------------------------------------------- */

#endif
