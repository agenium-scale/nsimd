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

#ifndef NSIMD_MODULES_FIXED_POINT_FUNCTION_SIMD_FMA_HPP
#define NSIMD_MODULES_FIXED_POINT_FUNCTION_SIMD_FMA_HPP

#include <nsimd/nsimd.h>
#include "nsimd/modules/fixed_point/simd.hpp"
#include "nsimd/modules/fixed_point/function/simd/add.hpp"
#include "nsimd/modules/fixed_point/function/simd/mul.hpp"

namespace nsimd {
namespace fixed_point {
// FMA emulation for FP type
template <uint8_t _lf, uint8_t _rt>
NSIMD_INLINE fpsimd_t<_lf, _rt> simd_fma(const fpsimd_t<_lf, _rt> &a0,
                                         const fpsimd_t<_lf, _rt> &a1,
                                         const fpsimd_t<_lf, _rt> &a2) {
  fpsimd_t<_lf, _rt> res;
  res = simd_mul<_lf, _rt>(a0, a1);
  res = simd_add<_lf, _rt>(res, a2);
  return res;
}

} // namespace fixed_point
} // namespace nsimd

#endif
