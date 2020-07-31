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

#ifndef NSIMD_MODULES_FUNCTION_SIMD_REC_HPP
#define NSIMD_MODULES_FUNCTION_SIMD_REC_HPP

#include "nsimd/modules/fixed_point/function/simd/div.hpp"
#include "nsimd/modules/fixed_point/simd.hpp"
#include <nsimd/nsimd.h>

namespace nsimd {
namespace fixed_point {
//template <uint8_t _lf, uint8_t _rt>
//NSIMD_INLINE fpsimd_t<_lf, _rt> simd_rec(const fpsimd_t<_lf, _rt> &a0) {
//  fpsimd_t<_lf, _rt> one(fp_t<_lf, _rt>(1));
//  return simd_div<_lf, _rt>(one, a0);
//}

// Calculate 1/a via newton-raphson
template <unsigned char _lf, unsigned char _rt>
inline fpsimd_t<_lf, _rt> simd_rec(const fpsimd_t<_lf, _rt> &a)
{
  typedef typename fp_t<_lf, _rt>::value_type val_t;
  typedef typename fp_t<_lf, _rt>::simd_logical log_t;
  typedef typename fp_t<_lf, _rt>::simd_type simd_t;

  fpsimd_t<_lf, _rt> one;
  one._raw = nsimd::set1( val_t(1) , val_t() );
  fpsimd_t<_lf, _rt> two(fp_t<_lf, _rt>(2));

  fpsimd_t<_lf, _rt> zero;
  zero._raw = nsimd::xorb( zero._raw , zero._raw , val_t() );
  log_t negative = nsimd::lt( a._raw , zero._raw , val_t() );
  fpsimd_t<_lf, _rt> abs = simd_abs(a);

  fpsimd_t<_lf, _rt> guess;
  val_t offset = 2*_rt - 8*val_t(sizeof(val_t));
  simd_t z = nsimd::clz(abs._raw, val_t());
  z = nsimd::add( z , nsimd::set1( offset , val_t() ) , val_t() );
  guess._raw = nsimd::shlv( one._raw , z , val_t() );

  int iter = 10;
  for(int i = 0; i < iter; ++i)
  {
    guess = guess * (two - abs * guess);
  }

  fpsimd_t<_lf,_rt> neg = zero - guess;
  guess._raw = nsimd::if_else1( negative , neg._raw , guess._raw , val_t() );

  return guess;
}

} // namespace fixed_point
} // namespace nsimd

#endif
