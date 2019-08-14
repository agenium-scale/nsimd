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

#ifndef NSIMD_MODULES_FUNCTION_TAN_HPP
#define NSIMD_MODULES_FUNCTION_TAN_HPP

#include "fixed_point/constants.hpp"
#include "fixed_point/fixed.hpp"
#include "fixed_point/function/cos.hpp"
#include "fixed_point/function/sin.hpp"

namespace nsimd
{
namespace fixed_point
{
// Calculate tan(x) using the identity tan = sin/cos
// Limits input to range[0,pi/2] for best precision
// -- range reduction is not trivially vectorizable though...
template <unsigned char _lf, unsigned char _rt>
inline fp_t<_lf, _rt> safe_tan(const fp_t<_lf, _rt> &a)
{
  return safe_sin(a) / safe_cos(a);
}

template <unsigned char _lf, unsigned char _rt>
inline fp_t<_lf, _rt> tan(const fp_t<_lf, _rt> &a)
{
  fp_t<_lf, _rt> b = a;
  // Reduce to range [0,inf]
  fp_t<_lf, _rt> mul = constants::one<_lf, _rt>();
  if(b < fp_t<_lf, _rt>(0))
  {
    b = -1 * b;
    mul = mul * -1;
  }

  // Reduce to range [0,pi]
  b = b - constants::pi<_lf, _rt>() * floor(b / constants::pi<_lf, _rt>());

  // Reduce to range [-pi/2,pi/2]
  fp_t<_lf, _rt> frac = b / constants::pi<_lf, _rt>();
  if(frac > fp_t<_lf, _rt>(0.5))
  {
    b = b - constants::pi<_lf, _rt>();
  }

  return mul * safe_tan(b);
}

} // namespace fixed_point
} // namespace nsimd

#endif
