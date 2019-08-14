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

#ifndef NSIMD_MODULES_FUNCTION_SIMD_EXP_HPP
#define NSIMD_MODULES_FUNCTION_SIMD_EXP_HPP

#include "fixed_point/constants.hpp"
#include "fixed_point/fixed.hpp"
#include "fixed_point/function/simd/floor.hpp"

#include <iostream>

namespace nsimd
{
namespace fixed_point
{
// For integer exponents, use exponentiation by squaring
template <unsigned char _lf, unsigned char _rt>
inline fpsimd_t<_lf, _rt> simd_exp(const fpsimd_t<_lf, _rt> &a, const int &b)
{
  if(b == 0)
    return fpsimd_t<_lf, _rt>(fp_t<_lf, _rt>(1));
  fpsimd_t<_lf, _rt> one(constants::one<_lf, _rt>());

  fpsimd_t<_lf, _rt> val = a;
  int e = b;
  if(e < 0)
  {
    val = one / val;
    e = -e;
  }
  fpsimd_t<_lf, _rt> res = one;
  while(e > 1)
  {
    if(e % 2)
    { // odd
      res = res * val;
      val = val * val;
      e = (e - 1) / 2;
    }
    else
    {
      val = val * val;
      e = e / 2;
    }
  }
  return res * val;
}

//// For floating point exponents, use Taylor series
// template< unsigned char _lf
//        , unsigned char _rt
//        >
// inline
// fpsimd_t< _lf , _rt >
// simd_exp( const fpsimd_t< _lf , _rt > &a
//   , const fpsimd_t< _lf , _rt > &b
//   ) {
//  using raw_type = typename fpsimd_t<_lf,_rt>::value_type;
//
//  // Separate integer and fractional portions for better accuracy
//  // - Taylor series is centered around x = 0
//  const raw_type integer = a._raw >> ( 8*sizeof(raw_type) - _lf );
//  const fpsimd_t<_lf,_rt> rem = b - fpsimd_t<_lf,_rt>(integer);
//
//  if ( 0 == rem._raw ) { return exp( a , integer ); }
//
//  fpsimd_t<_lf,_rt> fact = constants::one<_lf,_rt>();
//  fpsimd_t<_lf,_rt> res  = constants::one<_lf,_rt>();
//  fpsimd_t<_lf,_rt> log_eval = constants::one<_lf,_rt>();
//  fpsimd_t<_lf,_rt> log_init = rem * log( a );
//  // TODO: choose better stopping condition
//  // For   x  , error is of order 1/(3*sizeof(x))!
//  //     int8_t         ,   log2( 1/(6!)  ) = 9.5  bits precision
//  //    int16_t         ,   log2( 1/(8!)  ) = 15.3 bits precision
//  //    int32_t         ,   log2( 1/(12!) ) = 28.8 bits precision
//  constexpr int stop = 4 + 2 * sizeof(raw_type);
//  for ( int i = 1 ; i < stop ; ++i ) {
//    fact = fact / i;
//    log_eval = log_eval * log_init;
//    res = res + ( fact * log_eval );
//  }
//  res = res * exp( a , integer );
//
//  return res;
//}
//
// template< unsigned char _lf
//        , unsigned char _rt
//        >
// inline
// fpsimd_t< _lf , _rt >
// simd_exp( const fpsimd_t< _lf , _rt > &a
//   , const float &b
//   ) {
//  return exp( a , fpsimd_t<_lf,_rt>(b) );
//}
//
// template< unsigned char _lf
//        , unsigned char _rt
//        >
// inline
// fpsimd_t< _lf , _rt >
// simd_exp( const int &b ) {
//  return exp( constants::e<_lf,_rt>() , b );
//}

} // namespace fixed_point
} // namespace nsimd

#endif
