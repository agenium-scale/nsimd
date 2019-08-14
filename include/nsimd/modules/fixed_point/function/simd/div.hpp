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

#ifndef NSIMD_MODULES_FUNCTION_SIMD_DIV_HPP
#define NSIMD_MODULES_FUNCTION_SIMD_DIV_HPP

#include "fixed_point/simd.hpp"

namespace nsimd
{
namespace fixed_point
{
// For use in other functions where you may have repeated mul/div operations
template <unsigned char _lf, unsigned char _rt, typename T>
inline T simd_div_ungrouped(const T &a_half, const T &b_half, const fpsimd_t<_lf, _rt>)
{
  using val_t = typename fp_t<_lf, _rt>::value_type;
  constexpr int extra = 8 * sizeof(val_t) - _lf - _rt;
  constexpr int shift = _rt + extra;

  T tmp = a_half << shift;
  return tmp / b_half; // nsimd does not have simd integer division...
}

template <unsigned char _lf, unsigned char _rt>
inline fpsimd_t<_lf, _rt>
simd_div(const fpsimd_t<_lf, _rt> &a, const fpsimd_t<_lf, _rt> &b)
{
  // Slower, but allows decimal output
  using up_t = typename fp_t<_lf, _rt>::value_up;
  using val_t = typename fp_t<_lf, _rt>::value_type;
  using simd_up_t = typename fp_t<_lf, _rt>::simd_up;
  constexpr int extra = 8 * sizeof(val_t) - _lf - _rt;
  constexpr int shift = _rt + extra;

  simd_up_t a_lo = nsimd::split_low(a._raw, val_t());
  simd_up_t b_lo = nsimd::split_low(b._raw, val_t());

  simd_up_t a_hi = nsimd::split_high(a._raw, val_t());
  simd_up_t b_hi = nsimd::split_high(b._raw, val_t());

  fpsimd_t<_lf, _rt> res;
  res._raw = nsimd::group(
      simd_div_ungrouped(a_lo, b_lo, res), simd_div_ungrouped(a_hi, b_hi, res), up_t());

  return res;
}

// Operator overload with base type compatibility
template <unsigned char _lf, unsigned char _rt>
inline fpsimd_t<_lf, _rt>
operator/(const fpsimd_t<_lf, _rt> &a, const fpsimd_t<_lf, _rt> &b)
{
  return simd_div(a, b);
}

} // namespace fixed_point
} // namespace nsimd

#endif
