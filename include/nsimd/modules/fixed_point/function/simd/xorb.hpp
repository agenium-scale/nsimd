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

#ifndef NSIMD_MODULES_FIXED_POINT_FUNCTION_SIMD_XORB_HPP
#define NSIMD_MODULES_FIXED_POINT_FUNCTION_SIMD_XORB_HPP

#include <nsimd/nsimd.h>
#include "fixed_point/simd.hpp"

namespace nsimd
{
namespace fixed_point
{
template <uint8_t _lf, uint8_t _rt>
NSIMD_INLINE fpsimd_t<_lf, _rt>
simd_xorb(const fpsimd_t<_lf, _rt> &a0, const fpsimd_t<_lf, _rt> &a1)
{
  using raw_t = typename fp_t<_lf, _rt>::value_type;
  fpsimd_t<_lf, _rt> res;
  res._raw = nsimd::xorb(a0._raw, a1._raw, raw_t());
  return res;
}

} // namespace fixed_point
} // namespace nsimd

#endif
