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

#include <nsimd/nsimd.h>
#include "fixed_point/simd.hpp"
#include "fixed_point/function/simd/div.hpp"

namespace nsimd
{
namespace fixed_point
{
template <uint8_t _lf, uint8_t _rt>
NSIMD_INLINE fpsimd_t<_lf, _rt> simd_rec(const fpsimd_t<_lf, _rt> &a0)
{
   using val_t = typename fp_t<_lf, _rt>::value_type;
   fpsimd_t<_lf, _rt> one(fp_t<_lf, _rt>(1));

   return simd_div<_lf, _rt>(one, a0);
}

// // Calculate 1/a via newton-raphson
// template <unsigned char _lf, unsigned char _rt>
// inline fpsimd_t<_lf, _rt> simd_rec(const fpsimd_t<_lf, _rt> &a)
// {
//   using val_t = typename fp_t<_lf, _rt>::value_type;
//   using log_t = typename fp_t<_lf, _rt>::simd_logical;

//   fpsimd_t<_lf, _rt> one(fp_t<_lf, _rt>(1));
//   fpsimd_t<_lf, _rt> two(fp_t<_lf, _rt>(2));
//   fpsimd_t<_lf, _rt> guess(fp_t<_lf, _rt>(1));

//   log_t negative = (a._raw < fpsimd_t<_lf, _rt>(0)._raw);
//   fpsimd_t<_lf, _rt> abs = simd_abs(a);

//   // fpsimd_t<_lf, _rt> z = nsimd::clz(abs._raw, val_t());
//   // log_t gt1 = (a._raw > one);
//   // guess._raw = one << z - _lf

//   int iter = 10;
//   fpsimd_t<_lf, _rt> res0 = guess;
//   // fpsimd_t<_lf,_rt> res1 = guess;
//   // fpsimd_t<_lf,_rt> tmp0;
//   // fpsimd_t<_lf,_rt> tmp1;
//   // fpsimd_t<_lf,_rt> tmp2;
//   // fpsimd_t<_lf,_rt> tmp3;
//   for(int i = 0; i < iter; ++i)
//   {
//     // tmp0 = a * res0;
//     // tmp1 = one - tmp0;
//     // tmp2 = tmp1 * res0;
//     // res1 = tmp2 + res0;
//     // res0 = res1;
    
//     res0 = res0 * (one + (one - abs * res0));
//     // res0 = res0 * (two - abs * res0);
//   }
//   return res0;
// }

} // namespace fixed_point
} // namespace nsimd

#endif
