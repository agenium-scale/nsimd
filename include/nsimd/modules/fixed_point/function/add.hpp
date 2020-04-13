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

#ifndef NSIMD_MODULES_FIXED_POINT_FUNCTION_ADD_HPP
#define NSIMD_MODULES_FIXED_POINT_FUNCTION_ADD_HPP

#include "nsimd/modules/fixed_point/fixed.hpp"

namespace nsimd {
namespace fixed_point {
template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fp_t<_lf, _rt> add(const fp_t<_lf, _rt> &a,
                                const fp_t<_lf, _rt> &b) {
  fp_t<_lf, _rt> res;
  res._raw = a._raw + b._raw;

  return res;
}

// Compatibility with base types
template <unsigned char _lf, unsigned char _rt, typename T>
NSIMD_INLINE fp_t<_lf, _rt> add(const fp_t<_lf, _rt> &a, const T &b) {
  return add(a, fp_t<_lf, _rt>(b));
}

template <unsigned char _lf, unsigned char _rt, typename T>
NSIMD_INLINE fp_t<_lf, _rt> add(const T &b, const fp_t<_lf, _rt> &a) {
  return add(a, fp_t<_lf, _rt>(b));
}

// Operator overload with base type compatibility
template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fp_t<_lf, _rt> operator+(const fp_t<_lf, _rt> &a,
                                      const fp_t<_lf, _rt> &b) {
  return add(a, b);
}

template <unsigned char _lf, unsigned char _rt, typename T>
NSIMD_INLINE fp_t<_lf, _rt> operator+(const fp_t<_lf, _rt> &a, const T &b) {
  return add(a, fp_t<_lf, _rt>(b));
}

template <unsigned char _lf, unsigned char _rt, typename T>
NSIMD_INLINE fp_t<_lf, _rt> operator+(const T &b, const fp_t<_lf, _rt> &a) {
  return add(a, fp_t<_lf, _rt>(b));
}

} // namespace fixed_point
} // namespace nsimd

#endif
