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

#ifndef NSIMD_MODULES_FIXED_POINT_FUNCTION_SIMD_TAN_HPP
#define NSIMD_MODULES_FIXED_POINT_FUNCTION_SIMD_TAN_HPP

#include "fixed_point/constants.hpp"
#include "fixed_point/fixed.hpp"

namespace nsimd
{
namespace fixed_point
{
template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fpsimd_t<_lf, _rt> simd_tan(const fpsimd_t<_lf, _rt> &a)
{
  using val_t = typename fp_t<_lf, _rt>::value_type;
  using log_t = typename fp_t<_lf, _rt>::simd_logical;
  fpsimd_t<_lf, _rt> b = a;
  // Constants to use
  // TODO: See if dividing the twopi constant when using pi and half pi is
  // faster
  //       -- It could save registers and avoid extra splats
  fpsimd_t<_lf, _rt> mul(constants::one<_lf, _rt>());
  fpsimd_t<_lf, _rt> pi(constants::pi<_lf, _rt>());
  fpsimd_t<_lf, _rt> halfpi(constants::halfpi<_lf, _rt>());
  fpsimd_t<_lf, _rt> zero(constants::zero<_lf, _rt>());

  // Reduce to range [0,inf]
  log_t lt_0 = (b._raw < zero._raw);
  fpsimd_t<_lf, _rt> b_pos;
  b_pos._raw = ~(b._raw);
  fpsimd_t<_lf, _rt> mul_pos;
  mul_pos._raw = ~(mul._raw);
  b._raw = nsimd::if_else(lt_0, b_pos._raw, b._raw, val_t(), val_t());
  mul._raw = nsimd::if_else(lt_0, mul_pos._raw, mul._raw, val_t(), val_t());

  // Reduce to range [0,pi]
  b = b - pi * simd_floor(b / pi);

  // Reduce to range [-pi/2,pi/2] by shifting the range [pi/2,pi] to [-pi/2,0]
  log_t gt_pi = (b._raw > halfpi._raw);
  fpsimd_t<_lf, _rt> b_gt = b - pi;
  b._raw = nsimd::if_else(gt_pi, b_gt._raw, b._raw, val_t(), val_t());

  return mul * simd_safe_sin(b) / simd_safe_cos(b);
}

} // namespace fixed_point
} // namespace nsimd

#endif
