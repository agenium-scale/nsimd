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

#ifndef NSIMD_MODULES_FIXED_POINT_FUNCTION_MUL_HPP
#define NSIMD_MODULES_FIXED_POINT_FUNCTION_MUL_HPP

#include "nsimd/modules/fixed_point/fixed.hpp"

namespace nsimd {
namespace fixed_point {
template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fp_t<_lf, _rt> mul(const fp_t<_lf, _rt> &a,
                                const fp_t<_lf, _rt> &b) {
  typedef typename fp_t<_lf, _rt>::value_type raw_t;
  typedef typename fp_t<_lf, _rt>::positive_type pos_t;

  const size_t storage_size = 8 * sizeof(raw_t);
  static const raw_t sign_val =
      (raw_t)((pos_t)-1 << (storage_size - _rt));

  fp_t<_lf, _rt> res;

  raw_t tmp = a._raw * b._raw;
  raw_t sign = tmp < 0 ? sign_val : 0;
  tmp = tmp >> _rt;
  res._raw = tmp | sign;

  return res;
}

// Compatibility with base types
template <unsigned char _lf, unsigned char _rt, typename T>
NSIMD_INLINE fp_t<_lf, _rt> mul(const fp_t<_lf, _rt> &a, const T &b) {
  return mul(a, fp_t<_lf, _rt>(b));
}

template <unsigned char _lf, unsigned char _rt, typename T>
NSIMD_INLINE fp_t<_lf, _rt> mul(const T &b, const fp_t<_lf, _rt> &a) {
  return mul(a, fp_t<_lf, _rt>(b));
}

// Operator overload with base type compatibility
template <unsigned char _lf, unsigned char _rt>
NSIMD_INLINE fp_t<_lf, _rt> operator*(const fp_t<_lf, _rt> &a,
                                      const fp_t<_lf, _rt> &b) {
  return mul(a, b);
}

template <unsigned char _lf, unsigned char _rt, typename T>
NSIMD_INLINE fp_t<_lf, _rt> operator*(const fp_t<_lf, _rt> &a, const T &b) {
  return mul(a, fp_t<_lf, _rt>(b));
}

template <unsigned char _lf, unsigned char _rt, typename T>
NSIMD_INLINE fp_t<_lf, _rt> operator*(const T &b, const fp_t<_lf, _rt> &a) {
  return mul(a, fp_t<_lf, _rt>(b));
}

} // namespace fixed_point
} // namespace nsimd

#endif
