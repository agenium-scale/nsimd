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

// *****************************************************************************
//
// Implementation of a fixed-point arithmetic for nsimd. All the functions and
// types defines in this modules are under the namespace nsimd::fixed_point.
//
// A fixed point number N is defined by two values :
//   - lf : number of bits of the integer part of N
//   - rd : number of bits of the fractional part of N
// This module defines the nsimd packs for the fp_t types defined in fixed_point,
// for more information about the fp_t and its usage, refer to
// fixed_point/README.md.
//
// ------------------------------- Usage ---------------------------------------
//
// To use it, just include nsimd/modules/fixed_point.hpp, and be sure that
// the folder nsimd/modules/fixed_point/ is in your include path.
// A new pack of fp_t<lf, rt> elements is declared as follow:
//   nsimd::fixed_point::pack<lf, rt> v;
// And in the same way, all the nsimd operators defined in can be used by
// writing nsimd::fixed_point::op<lf, rt>(..., ...);
// Operators prototypes are the same as the original nsimd operators.
// Here is a minimal example :
//
// #include <iostream>
// #include <nsimd/modules/fixed_point.hpp>
//
// -----------------------------------------------------------------------------
//
// namespace fp = nsimd::fixed_point;
//
// int main() {
//   using fp_t = nsimd::fixed_point::fp_t<8, 8>;
//   using vec_t = nsimd::fixed_point::pack<8, 8>;
//   using raw_t = nsimd::fixed_point::pack<8, 8>::value_type;
//
//   const size_t v_size = nsimd::fixed_point::len(fp_t());
//
//   fp_t tab0[v_size];
//   fp_t tab1[v_size];
//   fp_t res[v_size];
//
//   for (size_t i = 0; i < v_size; i++) {
//     tab0[i] = (fp_t)i;
//     tab1[i] = (fp_t)i;
//   }
//
//   vec_t v0 = nsimd::fixed_point::loadu<vec_t>(tab0);
//   vec_t v1 = nsimd::fixed_point::loadu<vec_t>(tab1);
//   vec_t sum = nsimd::fixed_point::add(v0, v1);
//   nsimd::fixed_point::storeu(res, sum);
//
//   std::cout << "Output vector : [";
//   for (size_t i = 0; i < v_size; i++) {
//     std::cout << " " << res[i];
//   }
//   std::cout << "]" << std::endl;
//   return 0;
// }
//
// *****************************************************************************

#include "fixed_point/fixed.hpp"
#include "fixed_point/simd.hpp"
#include "fixed_point/simd_math.hpp"

#include <nsimd/nsimd.h>

namespace nsimd
{
namespace fixed_point
{
// -----------------------------------------------------------------------------
// ------------------------ Types definitions and len --------------------------
// -----------------------------------------------------------------------------

template <uint8_t lf, uint8_t rt>
struct pack
{
  using scalar_type = fp_t<lf, rt>;
  using value_type = typename fp_t<lf, rt>::value_type;
  fpsimd_t<lf, rt> val;
};

template <uint8_t lf, uint8_t rt>
struct packl
{
  using scalar_type = fp_t<lf, rt>;
  using value_type = typename fp_t<lf, rt>::value_type;
  fpsimdl_t<lf, rt> val;
};

template <uint8_t lf, uint8_t rt>
constexpr size_t len(const fp_t<lf, rt> &)
{
  return fpsimd_n(fp_t<lf, rt>());
}

template <uint8_t lf, uint8_t rt>
constexpr size_t len(const pack<lf, rt> &)
{
  return fpsimd_n(fpsimd_t<lf, rt>());
}

// -----------------------------------------------------------------------------
// ------------------- Basic arithmetic operators ------------------------------
// -----------------------------------------------------------------------------

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> add(pack<lf, rt> a0, pack<lf, rt> a1)
{
  pack<lf, rt> res;
  res.val = simd_add<lf, rt>(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> sub(pack<lf, rt> a0, pack<lf, rt> a1)
{
  pack<lf, rt> res;
  res.val = simd_sub<lf, rt>(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> mul(pack<lf, rt> a0, pack<lf, rt> a1)
{
  pack<lf, rt> res;
  res.val = simd_mul<lf, rt>(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> rec(pack<lf, rt> a0, pack<lf, rt> a1)
{
  pack<lf, rt> res;
  res.val = simd_rec<lf, rt>(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> div(pack<lf, rt> a0, pack<lf, rt> a1)
{
  pack<lf, rt> res;
  res.val = simd_div<lf, rt>(a0.val, a1.val);
  return res;
}

// -----------------------------------------------------------------------------
// ------------------- Comparison operators ------------------------------------
// -----------------------------------------------------------------------------

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> eq(pack<lf, rt> a0, pack<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_eq(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> ne(pack<lf, rt> a0, pack<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_ne(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> le(pack<lf, rt> a0, pack<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_le(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> lt(pack<lf, rt> a0, pack<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_lt(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> ge(pack<lf, rt> a0, pack<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_ge(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> gt(pack<lf, rt> a0, pack<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_gt(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> if_else1(packl<lf, rt> a0, pack<lf, rt> a1, pack<lf, rt> a2)
{
  pack<lf, rt> res;
  res.cal = simd_if_else1(a0.val, a1.val, a2.val);
  return res;
}

// -----------------------------------------------------------------------------
// ------------------- Bitwise operators  --------------------------------------
// -----------------------------------------------------------------------------

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> andb(pack<lf, rt> a0, pack<lf, rt> a1)
{
  pack<lf, rt> res;
  res.val = simd_andb(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> andnotb(pack<lf, rt> a0, pack<lf, rt> a1)
{
  pack<lf, rt> res;
  res.val = simd_andnotb(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> notb(pack<lf, rt> a0, pack<lf, rt> a1)
{
  pack<lf, rt> res;
  res.val = simd_notb(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> orb(pack<lf, rt> a0, pack<lf, rt> a1)
{
  pack<lf, rt> res;
  res.val = simd_orb(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> xorb(pack<lf, rt> a0, pack<lf, rt> a1)
{
  pack<lf, rt> res;
  res.val = simd_xorb(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> andl(packl<lf, rt> a0, packl<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_andl(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> andnotl(packl<lf, rt> a0, packl<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_andnotl(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> notl(packl<lf, rt> a0, packl<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_notl(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> orl(packl<lf, rt> a0, packl<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_orl(a0.val, a1.val);
  return res;
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> xorl(packl<lf, rt> a0, packl<lf, rt> a1)
{
  packl<lf, rt> res;
  res.val = simd_xorl(a0.val, a1.val);
  return res;
}

// -----------------------------------------------------------------------------
// -------------------- Load functions -----------------------------------------
// -----------------------------------------------------------------------------

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> loadu(fp_t<lf, rt> *p)
{
  pack<lf, rt> res;
  res.val = simd_loadu<lf, rt>(p);
  return res;
}

template <typename vec_t>
NSIMD_INLINE vec_t loadu(typename vec_t::scalar_type *p)
{
  return loadu(p);
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> loadlu(typename pack<lf, rt>::value_type *p)
{
  packl<lf, rt> res;
  res.val = simd_loadlu<lf, rt>(p);
  return res;
}

template <typename vecl_t>
NSIMD_INLINE vecl_t loadlu(typename vecl_t::scalar_type *p)
{
  return loadlu(p);
}

// -----------------------------------------------------------------------------
// -------------------- Store functions ----------------------------------------
// -----------------------------------------------------------------------------

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE void storeu(fp_t<lf, rt> *p, pack<lf, rt> &v)
{
  simd_storeu<lf, rt>(p, v.val);
}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE void storelu(typename packl<lf, rt>::value_type *p, fp_t<lf, rt> v)
{
  simd_storelu<lf, rt>(p, v.val);
}

} // namespace fixed_point
} // namespace nsimd

#endif
