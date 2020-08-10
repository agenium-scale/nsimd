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


template <uint8_t _lf, uint8_t _rt>
NSIMD_INLINE fpsimd_t<_lf, _rt> simd_rec(const fpsimd_t<_lf, _rt> &a0) {
  typedef typename fp_t<_lf, _rt>::value_type raw_t;
  typedef typename fpsimd_t<_lf, _rt>::value_type simd_t;

  simd_t z = nsimd::clz( a0._raw , raw_t() );
  raw_t total_size = 8 * sizeof(raw_t);
  z = nsimd::sub( z , nsimd::set1(raw_t(total_size - 2*_rt - 1) , raw_t() ) , raw_t() );

  // Initial seed = ulp << ( total bits - clz )
  fpsimd_t<_lf,_rt> x;
  x._raw = nsimd::set1( raw_t(1) , raw_t() );
  x._raw = nsimd::shlv( x._raw , z , raw_t() );

  // Newton raphson iterations
  fpsimd_t<_lf,_rt> two( fp_t<_lf,_rt>(2) );
  for ( int i = 0 ; i < 10 ; ++i ) {
    x = x * ( two - ( x * a0 ) );
  }

  return x;
}

} // namespace fixed_point
} // namespace nsimd

#endif
