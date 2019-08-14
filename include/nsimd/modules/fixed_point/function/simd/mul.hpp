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

#ifndef NSIMD_MODULES_FUNCTION_SIMD_MUL_HPP
#define NSIMD_MODULES_FUNCTION_SIMD_MUL_HPP

#include "fixed_point/simd.hpp"
// #include "helper/bit_printer.hpp"

namespace nsimd
{
namespace fixed_point
{
// For use in other functions where you may have repeated mul/div operations
template <unsigned char _lf, unsigned char _rt, typename T>
inline T simd_mul_ungrouped(const T &a_half, const T &b_half, const fpsimd_t<_lf, _rt>)
{
  using up_t = typename fp_t<_lf, _rt>::value_up;
  constexpr int half_size = 4 * sizeof(up_t);
  constexpr int shift = half_size - _lf;

  return ((a_half * b_half) >> shift);
}

template <unsigned char _lf, unsigned char _rt>
inline fpsimd_t<_lf, _rt>
simd_mul(const fpsimd_t<_lf, _rt> &a, const fpsimd_t<_lf, _rt> &b)
{
  using up_t = typename fp_t<_lf, _rt>::value_up;
  using val_t = typename fp_t<_lf, _rt>::value_type;
  using simd_up_t = typename fp_t<_lf, _rt>::simd_up;
  constexpr int half_size = 4 * sizeof(up_t);
  constexpr int shift = half_size - _lf;

  auto a_big = nsimd::upcvt(a);
  auto b_big = nsimd::upcvt(b);
  a_big = ((a_big * b_big) >> shift);

  fpsimd_t<_lf, _rt> res;
  res._raw = nsimd::downcvt(a_big);

  return res;
}

// Operator overload with base type compatibility
template <unsigned char _lf, unsigned char _rt>
inline fpsimd_t<_lf, _rt>
operator*(const fpsimd_t<_lf, _rt> &a, const fpsimd_t<_lf, _rt> &b)
{
  return simd_mul(a, b);
}

} // namespace fixed_point
} // namespace nsimd

#endif
