/*

Copyright (c) 2021 Agenium Scale

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

#include <nsimd/nsimd.h>

// ----------------------------------------------------------------------------
// Actual implementation

namespace nsimd {

template <int ExponentSize, int MantissaSize, typename UnsignedType,
          typename T>
int ufp(T a_, T b_) {
  UnsignedType a = nsimd::scalar_reinterpret(UnsignedType(), a_);
  UnsignedType b = nsimd::scalar_reinterpret(UnsignedType(), b_);
  UnsignedType exp_mask = ((UnsignedType)1 << ExponentSize) - 1;
  i64 ea = (i64)((a >> MantissaSize) & exp_mask);
  i64 eb = (i64)((b >> MantissaSize) & exp_mask);
  if (ea - eb > 1 || ea - eb < -1) {
    return 0;
  }
  UnsignedType man_mask = ((UnsignedType)1 << MantissaSize) - 1;
  i64 ma = (i64)(a & man_mask) | ((i64)1 << MantissaSize);
  i64 mb = (i64)(b & man_mask) | ((i64)1 << MantissaSize);
  i64 d = 0;

  if (ea == eb) {
    d = ma - mb;
  } else if (ea > eb) {
    d = 2 * ma - mb;
  } else {
    d = 2 * mb - ma;
  }
  d = (d >= 0 ? d : -d);
  int i = 0;
  for (; i <= MantissaSize + 1 && d >= ((i64)1 << i); i++)
    ;
  return (int)(MantissaSize + 1 - i);
}

} // namespace nsimd

// ----------------------------------------------------------------------------
// C ABI

extern "C" {

NSIMD_DLLSPEC int nsimd_ufp_f16(f16 a, f16 b) {
  return nsimd::ufp<5, 10, u16>(a, b);
}

NSIMD_DLLSPEC int nsimd_ufp_f32(f32 a, f32 b) {
  return nsimd::ufp<8, 23, u32>(a, b);
}

NSIMD_DLLSPEC int nsimd_ufp_f64(f64 a, f64 b) {
  return nsimd::ufp<11, 52, u64>(a, b);
}

} // extern "C"
