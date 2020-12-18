/*

Copyright (c) 2020 Agenium Scale

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

#ifndef NSIMD_MODULES_FIXED_POINT_FIXED_HPP
#define NSIMD_MODULES_FIXED_POINT_FIXED_HPP

#include <iostream>
#include <limits>
#include <stdint.h>

#include <nsimd/nsimd.h>

namespace nsimd {
namespace fixed_point {

template <typename T, u8 rt> static inline T float2int(const float val) {
  return (T)roundf(val * (float)(1 << rt));
}

template <typename T, u8 rt> static inline float int2float(const T val) {
  return (float)val / (float)(1 << rt);
}

//------------------------------------------------------------------------------
// Helper class to outsource the _raw size evaluation
//------------------------------------------------------------------------------

template <bool B, typename T = void> struct enable_if {};

template <typename T> struct enable_if<true, T> { typedef T type; };

template <u8 _lf, u8 _rt, typename T = void> struct fp_types {};

template <u8 _lf, u8 _rt>
struct fp_types<_lf, _rt,
                typename enable_if<(((_lf + 2 * _rt) > 0)) &&
                                   ((_lf + 2 * _rt) <= 8)>::type> {
  typedef i8 value_type;
  typedef i8 logical_type;
  typedef u8 positive_type;
  typedef vi8 simd_type;
  typedef vli8 simd_logical;
};

template <u8 _lf, u8 _rt>
struct fp_types<_lf, _rt,
                typename enable_if<(((_lf + 2 * _rt) > 8)) &&
                                   ((_lf + 2 * _rt) <= 16)>::type> {
  typedef i16 value_type;
  typedef i16 logical_type;
  typedef u16 positive_type;
  typedef vi16 simd_type;
  typedef vli16 simd_logical;
};

template <u8 _lf, u8 _rt>
struct fp_types<_lf, _rt,
                typename enable_if<(((_lf + 2 * _rt) > 16)) &&
                                   ((_lf + 2 * _rt) <= 32)>::type> {
  typedef i32 value_type;
  typedef i32 logical_type;
  typedef u32 positive_type;
  typedef vi32 simd_type;
  typedef vli32 simd_logical;
};

template <u8 _lf, u8 _rt>
struct fp_types<_lf, _rt,
                typename enable_if<(((_lf + 2 * _rt) > 32)) &&
                                   ((_lf + 2 * _rt) <= 64)>::type> {
  typedef i64 value_type;
  typedef i64 logical_type;
  typedef u64 positive_type;
  typedef vi64 simd_type;
  typedef vli64 simd_logical;
};


template <u8 _lf, u8 _rt> struct fp_t {
  typedef typename fp_types<_lf, _rt>::value_type value_type;
  typedef typename fp_types<_lf, _rt>::logical_type logical_type;
  typedef typename fp_types<_lf, _rt>::positive_type positive_type;
  typedef typename fp_types<_lf, _rt>::simd_type simd_type;
  typedef typename fp_types<_lf, _rt>::simd_logical simd_logical;

  static const u8 lf = _lf;
  static const u8 rt = _rt;

  value_type _raw;

  fp_t() {}

  fp_t(const fp_t<_lf, _rt> &cp) { _raw = cp._raw; }

  inline fp_t(const float &in) { _raw = float2int<value_type, rt>(in); }

  template <typename T> inline fp_t(const T &in) {
    _raw = float2int<value_type, rt>((float)in);
  }

  fp_t &operator=(const fp_t<_lf, _rt> &cp) {
    _raw = cp._raw;
    return *this;
  }

  fp_t &operator+=(const fp_t<_lf, _rt> &pl) {
    *this = *this + pl;
    return *this;
  }

  fp_t &operator-=(const fp_t<_lf, _rt> &mi) {
    *this = *this - mi;
    return *this;
  }

  fp_t &operator*=(const fp_t<_lf, _rt> &ti) {
    *this = *this * ti;
    return *this;
  }

  fp_t &operator/=(const fp_t<_lf, _rt> &di) {
    *this = *this / di;
    return *this;
  }

  template <typename T> operator T() const {
    return T(int2float<value_type, _rt>(_raw));
  }

  friend std::ostream& operator<<( std::ostream& stream
                                 , const nsimd::fixed_point::fp_t<_lf,_rt> &in
                                 ) {
    stream << float(in);
    return stream;
  }

};

//------------------------------------------------------------------------------
// For examining available precision
//------------------------------------------------------------------------------

template <u8 _lf, u8 _rt> u8 left(const fp_t<_lf, _rt>) {
  return _lf;
}

template <u8 _lf, u8 _rt> u8 right(const fp_t<_lf, _rt>) {
  return _rt;
}

} // namespace fixed_point
} // namespace nsimd

#include "nsimd/modules/fixed_point/fixed_math.hpp"

#endif
