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

#ifndef NSIMD_MODULES_FUNCTION_DIV_HPP
#define NSIMD_MODULES_FUNCTION_DIV_HPP

#include "nsimd/modules/fixed_point/fixed.hpp"

namespace nsimd
{
namespace fixed_point
{
template <unsigned char _lf, unsigned char _rt>
inline fp_t<_lf, _rt> div(const fp_t<_lf, _rt> &a, const fp_t<_lf, _rt> &b)
{
  fp_t<_lf, _rt> res;

  //// Fastest, but only gives integer output
  // res._raw = a._raw / b._raw;
  // res._raw = res._raw << _rt;

  // Slower, but allows decimal output
  using up_t = typename fp_t<_lf, _rt>::value_up;
  using val_t = typename fp_t<_lf, _rt>::value_type;
  constexpr int extra = 8 * sizeof(val_t) - _lf - _rt;
  constexpr int shift = _rt + extra;

  up_t tmp;
  tmp = (up_t(a._raw) << shift);
  tmp /= b._raw;
  res._raw = tmp;

  return res;
}

// Compatibility with base types
template <unsigned char _lf, unsigned char _rt, typename T>
inline fp_t<_lf, _rt> div(const fp_t<_lf, _rt> &a, const T &b)
{
  return div(a, fp_t<_lf, _rt>(b));
}

template <unsigned char _lf, unsigned char _rt, typename T>
inline fp_t<_lf, _rt> div(const T &b, const fp_t<_lf, _rt> &a)
{
  return div(fp_t<_lf, _rt>(b), a);
}

// Operator overload with base type compatibility
template <unsigned char _lf, unsigned char _rt>
inline fp_t<_lf, _rt> operator/(const fp_t<_lf, _rt> &a, const fp_t<_lf, _rt> &b)
{
  return div(a, b);
}

template <unsigned char _lf, unsigned char _rt, typename T>
inline fp_t<_lf, _rt> operator/(const fp_t<_lf, _rt> &a, const T &b)
{
  return div(a, fp_t<_lf, _rt>(b));
}

template <unsigned char _lf, unsigned char _rt, typename T>
inline fp_t<_lf, _rt> operator/(const T &b, const fp_t<_lf, _rt> &a)
{
  return div(fp_t<_lf, _rt>(b), a);
}

} // namespace fixed_point
} // namespace nsimd

#endif
