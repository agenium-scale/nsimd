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

#ifndef NSIMD_MODULES_FIXED_POINT_FUNCTION_LOG_HPP
#define NSIMD_MODULES_FIXED_POINT_FUNCTION_LOG_HPP

#include "fixed_point/constants.hpp"
#include "fixed_point/fixed.hpp"

#include <iostream>

namespace nsimd
{
namespace fixed_point
{
// TODO: This iterative method is not vectorizable...
template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fp_t<_lf, _rt> log2(const fp_t<_lf, _rt> &a)
{
  if(a._raw <= 0)
  {
    std::cout << "Error: cannot take log of negative number\n";
    return fp_t<_lf, _rt>(-1);
  }

  using raw_type = typename fp_t<_lf, _rt>::value_type;
  const fp_t<_lf, _rt> two(2);

  // First calculate integer portion
  // It is safe to use shifts because we know the input is positive at this
  // point
  const int shift_size = 8 * sizeof(raw_type) - _lf;
  raw_type tmp = a._raw >> shift_size;
  tmp = tmp >> 1;
  char I = 0;
  fp_t<_lf, _rt> pow2(1);
  fp_t<_lf, _rt> tmp2(1);
  if(0 != tmp)
  {
    while(tmp)
    {
      tmp = tmp >> 1;
      ++I;
      pow2 = two * pow2;
    }
    // If integer power of 2, stop here
    if(constants::one<_lf, _rt>() == (a / pow2))
    {
      return fp_t<_lf, _rt>(I);
    }
  }
  else
  {
    // Opposite of above - multiply by two and divide pow2 until tmp > 1
    fp_t<_lf, _rt> tmp2 = a;
    while(tmp2._raw < constants::one<_lf, _rt>()._raw)
    {
      tmp2 = tmp2 << 1;
      --I;
      pow2 = pow2 / 2;
    }
    // If integer power of 2, stop here
    if(a == pow2)
    {
      return fp_t<_lf, _rt>(I);
    }
  }
  fp_t<_lf, _rt> res(I);

  // Remaining fractional part is bound between 1 and 2
  // Iteratively approach
  fp_t<_lf, _rt> y0 = a / pow2;
  fp_t<_lf, _rt> z;
  pow2 = 1;
  // TODO: Look into choosing a better stopping condition
  for(int i = 0; i < _rt + _lf; ++i)
  {
    z = y0;
    while(z._raw < two._raw)
    {
      z = z * z;
      pow2 = pow2 / two;
      ++i;
    }
    res = res + pow2;
    y0 = z / two;
  }

  return res;
}

template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fp_t<_lf, _rt> log(const fp_t<_lf, _rt> &a)
{
  fp_t<_lf, _rt> res = log2(a) / fixed::constants::log2_e<_lf, _rt>();
  return res;
}

template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fp_t<_lf, _rt> log10(const fp_t<_lf, _rt> &a)
{
  fp_t<_lf, _rt> res = log2(a) / fixed::constants::log2_10<_lf, _rt>();
  return res;
}

} // namespace fixed_point
} // namespace nsimd

#endif
