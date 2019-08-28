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

#ifndef NSIMD_MODULES_FIXED_POINT_SIMD_HPP
#define NSIMD_MODULES_FIXED_POINT_SIMD_HPP

#include <nsimd/nsimd.h>

#include "nsimd/modules/fixed_point/fixed.hpp"
#include "nsimd/modules/fixed_point/fixed_math.hpp"
#include "nsimd/modules/fixed_point/fixed_vector.hpp"

namespace nsimd
{
namespace fixed_point
{
// This implementation means that we must cast to the appropriate pointer type
// to load/store values
template <uint8_t _lf, uint8_t _rt>
class fpsimd_t
{
public:
  using base_type = typename fp_t<_lf, _rt>::value_type;
  using value_type = typename fp_t<_lf, _rt>::simd_type;
  using value_up = typename fp_t<_lf, _rt>::simd_up;
  using logic = typename fp_t<_lf, _rt>::simd_logical;
  value_type _raw;

  fpsimd_t()
  {
  }
  
  fpsimd_t(const fp_t<_lf, _rt> &cp)
  {
    _raw = nsimd::set1(cp._raw, base_type());
  }

  // Splats
  // NSIMD_INLINE fpsimd_t( const float  &in ) { float2fixed<value_type,_lf,_rt>( _raw
  // , in ); } NSIMD_INLINE fpsimd_t( const double &in ) {
  // float2fixed<value_type,_lf,_rt>( _raw , in ); } template<typename T>
  // fpsimd_t( const T &in ) { integer_convert<_lf,_rt>( _raw , in ); }

  fpsimd_t &operator=(const fpsimd_t<_lf, _rt> &cp)
  {
    _raw = cp._raw;
    return *this;
  }
};

template <uint8_t _lf, uint8_t _rt>
class fpsimdl_t
{
public:
  using base_type = typename fp_t<_lf, _rt>::logical_type;
  using value_type = typename fp_t<_lf, _rt>::simd_type;
  using value_up = typename fp_t<_lf, _rt>::simd_up;
  using logic = typename fp_t<_lf, _rt>::simd_logical;
  logic _raw;

  fpsimdl_t()
  {
  }

  fpsimdl_t &operator=(const fpsimdl_t<_lf, _rt> &cp)
  {
    _raw = cp._raw;
    return *this;
  }
};

// Number of elements that fit into a SIMD register
template <uint8_t _lf, uint8_t _rt>
constexpr size_t fpsimd_n()
{
  using raw_t = typename fp_t<_lf, _rt>::value_type;
  return nsimd::len(raw_t());
}

// Number of elements that fit into a SIMD register
template <uint8_t _lf, uint8_t _rt>
constexpr size_t fpsimd_n(const fp_t<_lf, _rt> &)
{
  return fpsimd_n<_lf, _rt>();
}

// Number of elements that fit into a SIMD register
template <uint8_t _lf, uint8_t _rt>
constexpr size_t fpsimd_n(const fpv_t<_lf, _rt> &)
{
  return fpsimd_n<_lf, _rt>();
}

// Number of elements that fit into a SIMD register
template <uint8_t _lf, uint8_t _rt>
constexpr size_t fpsimd_n(const fpsimd_t<_lf, _rt> &)
{
  return fpsimd_n<_lf, _rt>();
}

} // namespace fixed_point
} // namespace nsimd

#endif
