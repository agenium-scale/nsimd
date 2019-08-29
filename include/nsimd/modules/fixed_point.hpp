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

namespace nsimd
{
namespace fixed_point
{
// -----------------------------------------------------------------------------
// ------------------------ Types definitions and len --------------------------
// -----------------------------------------------------------------------------

template <typename T>
struct pack
{
  static constexpr uint8_t lf = T::lf;
  static constexpr uint8_t rt = T::rt;
  using scalar_type = fp_t<lf, rt>;
  using base_type = fp_t<lf, rt>;
  fpsimd_t<lf, rt> val;
};

template <typename T>
struct packl
{
  static constexpr uint8_t lf = T::lf;
  static constexpr uint8_t rt = T::rt;
  using scalar_type = fp_t<lf, rt>;
  using base_type = typename fp_t<lf, rt>::logical_type;
  fpsimdl_t<lf, rt> val;
};

template <typename T>
constexpr size_t len(const T &)
{
  return fpsimd_n(T());
}

template <typename T>
constexpr size_t len(const pack<T> &)
{
  return fpsimd_n(fpsimd_t<T::lf, T::rt>());
}

// -----------------------------------------------------------------------------
// ------------------- Basic arithmetic operators ------------------------------
// -----------------------------------------------------------------------------

template <typename T>
NSIMD_INLINE pack<T> add(pack<T> a0, pack<T> a1)
{
  pack<T> res;
  res.val = simd_add<T::lf, T::rt>(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> sub(pack<T> a0, pack<T> a1)
{
  pack<T> res;
  res.val = simd_sub<T::lf, T::rt>(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> mul(pack<T> a0, pack<T> a1)
{
  pack<T> res;
  res.val = simd_mul<T::lf, T::rt>(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> div(pack<T> a0, pack<T> a1)
{
  pack<T> res;
  res.val = simd_div<T::lf, T::rt>(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> fma(pack<T> a0, pack<T> a1, pack<T> a2)
{
  pack<T> res;
  res.val = simd_fma<T::lf, T::rt>(a0.val, a1.val, a2.val);
  return res;
}

// -----------------------------------------------------------------------------
// ------------------- Comparison operators ------------------------------------
// -----------------------------------------------------------------------------

template <typename T>
NSIMD_INLINE packl<T> eq(pack<T> a0, pack<T> a1)
{
  packl<T> res;
  res.val = simd_eq(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> ne(pack<T> a0, pack<T> a1)
{
  packl<T> res;
  res.val = simd_ne(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> le(pack<T> a0, pack<T> a1)
{
  packl<T> res;
  res.val = simd_le(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> lt(pack<T> a0, pack<T> a1)
{
  packl<T> res;
  res.val = simd_lt(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> ge(pack<T> a0, pack<T> a1)
{
  packl<T> res;
  res.val = simd_ge(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> gt(pack<T> a0, pack<T> a1)
{
  packl<T> res;
  res.val = simd_gt(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> if_else1(packl<T> a0, pack<T> a1, pack<T> a2)
{
  pack<T> res;
  res.val = simd_if_else1(a0.val, a1.val, a2.val);
  return res;
}

// -----------------------------------------------------------------------------
// ------------------- Bitwise operators  --------------------------------------
// -----------------------------------------------------------------------------

template <typename T>
NSIMD_INLINE pack<T> andb(pack<T> a0, pack<T> a1)
{
  pack<T> res;
  res.val = simd_andb(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> andl(packl<T> a0, packl<T> a1)
{
  packl<T> res;
  res.val = simd_andl(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> andnotb(pack<T> a0, pack<T> a1)
{
  pack<T> res;
  res.val = simd_andnotb(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> andnotl(packl<T> a0, packl<T> a1)
{
  packl<T> res;
  res.val = simd_andnotl(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> notb(pack<T> a0)
{
  pack<T> res;
  res.val = simd_notb(a0.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> notl(packl<T> a0)
{
  packl<T> res;
  res.val = simd_notl(a0.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> orb(pack<T> a0, pack<T> a1)
{
  pack<T> res;
  res.val = simd_orb(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> orl(packl<T> a0, packl<T> a1)
{
  packl<T> res;
  res.val = simd_orl(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> xorb(pack<T> a0, pack<T> a1)
{
  pack<T> res;
  res.val = simd_xorb(a0.val, a1.val);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> xorl(packl<T> a0, packl<T> a1)
{
  packl<T> res;
  res.val = simd_xorl(a0.val, a1.val);
  return res;
}

// -----------------------------------------------------------------------------
// ------------------- Math functions ------------------------------------------
// -----------------------------------------------------------------------------

template <typename T>
NSIMD_INLINE pack<T> rec(pack<T> a0)
{
  pack<T> res;
  res.val = simd_rec(a0.val);
  return res;
}

// -----------------------------------------------------------------------------
// -------------------- Load functions -----------------------------------------
// -----------------------------------------------------------------------------

template <typename T>
NSIMD_INLINE pack<T> set1(typename pack<T>::base_type a0)
{
  pack<T> res;
  res.val = simd_set1<T::lf, T::rt>(a0);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> loadu(typename pack<T>::base_type *p)
{
  pack<T> res;
  res.val = simd_loadu<T::lf, T::rt>(p);
  return res;
}

template <typename T>
NSIMD_INLINE pack<T> loada(typename pack<T>::base_type *p)
{
  pack<T> res;
  res.val = simd_loada<T::lf, T::rt>(p);
  return res;
}
template <typename T>
NSIMD_INLINE packl<T> loadlu(typename packl<T>::base_type *p)
{
  packl<T> res;
  res.val = simd_loadlu<T::lf, T::rt>(p);
  return res;
}

template <typename T>
NSIMD_INLINE packl<T> loadla(typename packl<T>::base_type *p)
{
  packl<T> res;
  res.val = simd_loadla<T::lf, T::rt>(p);
  return res;
}

// -----------------------------------------------------------------------------
// -------------------- Store functions ----------------------------------------
// -----------------------------------------------------------------------------

template <typename T>
NSIMD_INLINE void storeu(typename pack<T>::base_type *p, pack<T> &v)
{
  simd_storeu<T::lf, T::rt>(p, v.val);
}

template <typename T>
NSIMD_INLINE void storea(typename pack<T>::base_type *p, pack<T> &v)
{
  simd_storea<T::lf, T::rt>(p, v.val);
}

template <typename T>
NSIMD_INLINE void storelu(typename packl<T>::base_type *p, packl<T> v)
{
  simd_storelu<T::lf, T::rt>(p, v.val);
}

template <typename T>
NSIMD_INLINE void storela(typename packl<T>::base_type *p, packl<T> v)
{
  simd_storela<T::lf, T::rt>(p, v.val);
}

} // namespace fixed_point
} // namespace nsimd

#endif
