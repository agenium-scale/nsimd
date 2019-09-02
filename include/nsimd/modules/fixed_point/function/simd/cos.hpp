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

#ifndef NSIMD_MODULES_FIXED_POINT_FUNCTION_SIMD_COS_HPP
#define NSIMD_MODULES_FIXED_POINT_FUNCTION_SIMD_COS_HPP

#include "nsimd/modules/fixed_point/constants.hpp"
#include "nsimd/modules/fixed_point/fixed.hpp"

namespace nsimd
{
namespace fixed_point
{
// Calculate cos(x) using Taylor series up to x^8 (error is of the order x^10)
// Limits input to range[0,pi/2] for best precision
// -- range reduction is not trivially vectorizable though...
template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fpsimd_t<_lf, _rt> simd_safe_cos(const fpsimd_t<_lf, _rt> &b)
{
  fpsimd_t<_lf, _rt> b2 = b * b;
  fpsimd_t<_lf, _rt> one(constants::one<_lf, _rt>());
  fpsimd_t<_lf, _rt> res = one;
  // TODO: Choose error according to precision of input/output
  res =
      (one - ((b2 * fpsimd_t<_lf, _rt>(fp_t<_lf, _rt>(1. / 56.))) * (res))); // 56 = 7 * 8
  res =
      (one - ((b2 * fpsimd_t<_lf, _rt>(fp_t<_lf, _rt>(1. / 30.))) * (res))); // 30 = 5 * 6
  res =
      (one - ((b2 * fpsimd_t<_lf, _rt>(fp_t<_lf, _rt>(1. / 12.))) * (res))); // 12 = 3 * 4
  res =
      (one - ((b2 * fpsimd_t<_lf, _rt>(fp_t<_lf, _rt>(1. / 2.))) * (res))); // 2  = 1 * 2
  res = (res);

  return res;
}

template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fpsimd_t<_lf, _rt> simd_cos(const fpsimd_t<_lf, _rt> &a)
{
  typedef typename fp_t<_lf, _rt>::value_type val_t;
  typedef typename fp_t<_lf, _rt>::simd_logical log_t;
  fpsimd_t<_lf, _rt> b = a;
  // Constants to use
  // TODO: See if dividing the twopi constant when using pi and half pi is
  // faster
  //       -- It could save registers and avoid extra splats
  fpsimd_t<_lf, _rt> mul(constants::one<_lf, _rt>());
  fpsimd_t<_lf, _rt> pi(constants::pi<_lf, _rt>());
  fpsimd_t<_lf, _rt> halfpi(constants::halfpi<_lf, _rt>());
  fpsimd_t<_lf, _rt> twopi(constants::twopi<_lf, _rt>());

  // Reduce to range [0,inf]
  b = simd_abs(b);

  // Reduce to range [0,2pi]
  b = b - twopi * simd_floor(b / twopi);

  // Reduce to range [0,pi]
  log_t gt_pi = (b._raw > pi._raw);
  fpsimd_t<_lf, _rt> b_gt = b - pi;
  fpsimd_t<_lf, _rt> mul_gt;
  mul_gt._raw = ~(mul._raw);
  b._raw = nsimd::if_else(gt_pi, b_gt._raw, b._raw, val_t(), val_t());
  mul._raw = nsimd::if_else(gt_pi, mul_gt._raw, mul._raw, val_t(), val_t());

  // Reduce to range [-pi/2,pi/2] thanks to: cos(x) = sin(x+pi/2)
  b = b + halfpi;

  return mul * simd_safe_sin(b);
}

} // namespace fixed_point
} // namespace nsimd

#endif
