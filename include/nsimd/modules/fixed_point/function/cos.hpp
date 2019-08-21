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

#ifndef NSIMD_MODULES_FIXED_POINT_FUNCTION_COS_HPP
#define NSIMD_MODULES_FIXED_POINT_FUNCTION_COS_HPP

#include "fixed/constants.hpp"
#include "fixed_point/fixed.hpp"

namespace nsimd
{
namespace fixed_point
{
// Calculate cos(x) using Taylor series up to x^8 (error is of the order x^10)
// Limits input to range[0,pi/2] for best precision
// -- range reduction is not trivially vectorizable though...
template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fp_t<_lf, _rt> safe_cos(const fp_t<_lf, _rt> &b)
{
  fp_t<_lf, _rt> b2 = b * b;
  fp_t<_lf, _rt> one = constants::one<_lf, _rt>();
  fp_t<_lf, _rt> res(1);
  // TODO: Choose error according to precision of input/output
  res = (one - ((b2 * (1. / 56.)) * (res))); // 56 = 7 * 8
  res = (one - ((b2 * (1. / 30.)) * (res))); // 30 = 5 * 6
  res = (one - ((b2 * (1. / 12.)) * (res))); // 12 = 3 * 4
  res = (one - ((b2 * (1. / 2.)) * (res)));  // 2  = 1 * 2
  res = (res);

  return res;
}

template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fp_t<_lf, _rt> cos(const fp_t<_lf, _rt> &a)
{
  fp_t<_lf, _rt> b = a;
  // Reduce to range [0,inf]
  fp_t<_lf, _rt> mul = constants::one<_lf, _rt>();
  if(b < fp_t<_lf, _rt>(0))
  {
    b = -1 * b;
  }

  // Reduce to range [0,2pi]
  b = b - constants::twopi<_lf, _rt>() * floor(b / constants::twopi<_lf, _rt>());

  fp_t<_lf, _rt> frac = b / constants::twopi<_lf, _rt>();
  // Reduce to range [0,pi]
  if(frac > fp_t<_lf, _rt>(0.5))
  {
    b = b - constants::pi<_lf, _rt>();
    mul = -1 * mul;
    frac = frac - 0.5;
  }
  // Reduce to range [-pi/2,pi/2] thanks to: cos(x) = -sin(x-pi/2)
  b = b + constants::pi<_lf, _rt>() / 2;

  return mul * safe_sin(b);
}

} // namespace fixed_point
} // namespace nsimd

#endif
