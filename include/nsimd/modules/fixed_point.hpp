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

#ifndef NSIMD_MODULES_FIXED_POINT_HPP
#define NSIMD_MODULES_FIXED_POINT_HPP

#include <nsimd/nsimd.h>

#include "nsimd/modules/fixed_point/fixed.hpp"
#include "nsimd/modules/fixed_point/simd.hpp"
#include "nsimd/modules/fixed_point/simd_math.hpp"

namespace nsimd {
namespace fixed_point {
// -----------------------------------------------------------------------------
// ------------------------ Types definitions and len --------------------------
// -----------------------------------------------------------------------------

template <typename T> NSIMD_STRUCT pack;

template <typename T> int len(const T &) { return fpsimd_n(T()); }

template <typename T> int len(const nsimd::fixed_point::pack<T> &) {
  return fpsimd_n(fpsimd_t<T::lf, T::rt>());
}

template <typename T> NSIMD_STRUCT pack {
  static const uint8_t lf = T::lf;
  static const uint8_t rt = T::rt;
  typedef fp_t<lf, rt> value_type;
  fpsimd_t<lf, rt> val;

  friend std::ostream &operator<<(std::ostream &os, pack<T> &a0) {
    T *buf = new T[nsimd::fixed_point::len(a0)];
    nsimd::fixed_point::simd_storeu( buf , a0.val );
    os << "{ ";
    int n = nsimd::fixed_point::len(a0);
    for (int i = 0; i < n; i++) {
      os << buf[i];
      if (i < n - 1) {
        os << ", ";
      }
    }
    os << " }";
    delete[] buf;
    return os;
  }
};

template <typename T> NSIMD_STRUCT packl {
  static const uint8_t lf = T::lf;
  static const uint8_t rt = T::rt;
  typedef typename fp_t<lf, rt>::logical_type value_type;
  fpsimdl_t<lf, rt> val;
};

// -----------------------------------------------------------------------------
// ------------------- Basic arithmetic operators ------------------------------
// -----------------------------------------------------------------------------

template <typename T>
NSIMD_INLINE pack<T> add(const pack<T> &a0, const pack<T> &a1) {
  pack<T> res;
  res.val = simd_add<T::lf, T::rt>(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> operator+(const pack<T> &a0, const pack<T> &a1) {
  return add( a0 , a1 );
}

template <typename T>
NSIMD_INLINE pack<T> sub(const pack<T> &a0, const pack<T> &a1) {
  pack<T> res;
  res.val = simd_sub<T::lf, T::rt>(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> operator-(const pack<T> &a0, const pack<T> &a1) {
  return sub( a0 , a1 );
}

template <typename T>
NSIMD_INLINE pack<T> mul(const pack<T> &a0, const pack<T> &a1) {
  pack<T> res;
  res.val = simd_mul<T::lf, T::rt>(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> operator*(const pack<T> &a0, const pack<T> &a1) {
  return mul( a0 , a1 );
}

template <typename T>
NSIMD_INLINE pack<T> div(const pack<T> &a0, const pack<T> &a1) {
  pack<T> res;
  res.val = simd_div<T::lf, T::rt>(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> operator/(const pack<T> &a0, const pack<T> &a1) {
  return div( a0 , a1 );
}

template <typename T>
NSIMD_INLINE pack<T> fma(const pack<T> &a0, const pack<T> &a1,
                         const pack<T> &a2) {
  pack<T> res;
  res.val = simd_fma<T::lf, T::rt>(a0.val, a1.val, a2.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> min(const pack<T> &a0, const pack<T> &a1) {
  pack<T> res;
  res.val = simd_min(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> max(const pack<T> &a0, const pack<T> &a1) {
  pack<T> res;
  res.val = simd_max(a0.val, a1.val);
  return res;
}

// -----------------------------------------------------------------------------
// ------------------- Comparison operators ------------------------------------
// -----------------------------------------------------------------------------

template <typename T>
NSIMD_INLINE packl<T> eq(const pack<T> &a0, const pack<T> &a1) {
  packl<T> res;
  res.val = simd_eq(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> operator==(const pack<T> &a0, const pack<T> &a1) {
  return eq( a0 , a1 );
}

template <typename T>
NSIMD_INLINE packl<T> ne(const pack<T> &a0, const pack<T> &a1) {
  packl<T> res;
  res.val = simd_ne(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> operator!=(const pack<T> &a0, const pack<T> &a1) {
  return ne( a0 , a1 );
}

template <typename T>
NSIMD_INLINE packl<T> le(const pack<T> &a0, const pack<T> &a1) {
  packl<T> res;
  res.val = simd_le(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> operator<=(const pack<T> &a0, const pack<T> &a1) {
  return le( a0 , a1 );
}

template <typename T>
NSIMD_INLINE packl<T> lt(const pack<T> &a0, const pack<T> &a1) {
  packl<T> res;
  res.val = simd_lt(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> operator<(const pack<T> &a0, const pack<T> &a1) {
  return lt( a0 , a1 );
}

template <typename T>
NSIMD_INLINE packl<T> ge(const pack<T> &a0, const pack<T> &a1) {
  packl<T> res;
  res.val = simd_ge(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> operator>=(const pack<T> &a0, const pack<T> &a1) {
  return ge( a0 , a1 );
}

template <typename T>
NSIMD_INLINE packl<T> gt(const pack<T> &a0, const pack<T> &a1) {
  packl<T> res;
  res.val = simd_gt(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> operator>(const pack<T> &a0, const pack<T> &a1) {
  return gt( a0 , a1 );
}

template <typename T>
NSIMD_INLINE pack<T> if_else1(const packl<T> &a0, const pack<T> &a1,
                              const pack<T> &a2) {
  pack<T> res;
  res.val = simd_if_else1(a0.val, a1.val, a2.val);
  return res;
}

// -----------------------------------------------------------------------------
// ------------------- Bitwise operators  --------------------------------------
// -----------------------------------------------------------------------------

template <typename T>
NSIMD_INLINE pack<T> andb(const pack<T> &a0, const pack<T> &a1) {
  pack<T> res;
  res.val = simd_andb(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> andl(const packl<T> &a0, const packl<T> &a1) {
  packl<T> res;
  res.val = simd_andl(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> andnotb(const pack<T> &a0, const pack<T> &a1) {
  pack<T> res;
  res.val = simd_andnotb(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> andnotl(const packl<T> &a0, const packl<T> &a1) {
  packl<T> res;
  res.val = simd_andnotl(a0.val, a1.val);
  return res;
}

template <typename T> NSIMD_INLINE pack<T> notb(pack<T> a0) {
  pack<T> res;
  res.val = simd_notb(a0.val);
  return res;
}

template <typename T> NSIMD_INLINE packl<T> notl(packl<T> a0) {
  packl<T> res;
  res.val = simd_notl(a0.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> orb(const pack<T> &a0, const pack<T> &a1) {
  pack<T> res;
  res.val = simd_orb(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> orl(const packl<T> &a0, const packl<T> &a1) {
  packl<T> res;
  res.val = simd_orl(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> xorb(const pack<T> &a0, const pack<T> &a1) {
  pack<T> res;
  res.val = simd_xorb(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> xorl(const packl<T> &a0, const packl<T> &a1) {
  packl<T> res;
  res.val = simd_xorl(a0.val, a1.val);
  return res;
}

// -----------------------------------------------------------------------------
// ------------------- Math functions ------------------------------------------
// -----------------------------------------------------------------------------

template <typename T> NSIMD_INLINE pack<T> abs(pack<T> a0) {
  pack<T> res;
  res.val = simd_abs(a0.val);
  return res;
}

template <typename T> NSIMD_INLINE pack<T> rec(pack<T> a0) {
  pack<T> res;
  res.val = simd_rec(a0.val);
  return res;
}

// -----------------------------------------------------------------------------
// -------------------- Load functions -----------------------------------------
// -----------------------------------------------------------------------------

template <typename T> NSIMD_INLINE T set1(typename T::value_type a0) {
  T res;
  res.val = simd_set1<T::lf, T::rt>(a0);
  return res;
}

template <typename T> NSIMD_INLINE T loadu(typename T::value_type *p) {
  T res;
  res.val = simd_loadu<T::lf, T::rt>(p);
  return res;
}

template <typename T> NSIMD_INLINE T loada(typename T::value_type *p) {
  T res;
  res.val = simd_loada<T::lf, T::rt>(p);
  return res;
}

template <typename T> NSIMD_INLINE T loadlu(typename T::value_type *p) {
  T res;
  res.val = simd_loadlu<T::lf, T::rt>(p);
  return res;
}

template <typename T> NSIMD_INLINE T loadla(typename T::value_type *p) {
  T res;
  res.val = simd_loadla<T::lf, T::rt>(p);
  return res;
}

// -----------------------------------------------------------------------------
// -------------------- Store functions ----------------------------------------
// -----------------------------------------------------------------------------

template <typename T>
NSIMD_INLINE void storeu(typename T::value_type *p, T v) {
  simd_storeu<T::lf, T::rt>(p, v.val);
}

template <typename T>
NSIMD_INLINE void storea(typename T::value_type *p, T v) {
  simd_storea<T::lf, T::rt>(p, v.val);
}

template <typename T>
NSIMD_INLINE void storelu(typename T::value_type *p, T v) {
  simd_storelu<T::lf, T::rt>(p, v.val);
}

template <typename T>
NSIMD_INLINE void storela(typename T::value_type *p, T v) {
  simd_storela<T::lf, T::rt>(p, v.val);
}

} // namespace fixed_point

} // namespace nsimd

#endif
