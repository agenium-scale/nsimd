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

#define NSIMD_INSIDE
#include <nsimd/nsimd.h>

// ----------------------------------------------------------------------------

static int nsimd_upper_log2(u64 a) {
  int l = 0;
  for (; ((u64)1 << l) < a; l++);
  return l;
}

// ----------------------------------------------------------------------------

extern "C" {

NSIMD_DLLEXPORT int nsimd_diff_in_logulps_f16(f16 a, f16 b) {
  int d = nsimd_scalar_reinterpret_i16_f16(a) -
          nsimd_scalar_reinterpret_i16_f16(b);
  return nsimd_upper_log2((u64)(d >= 0 ? d : -d));
}

NSIMD_DLLEXPORT int nsimd_diff_in_logulps_f32(f32 a, f32 b) {
  int d = nsimd_scalar_reinterpret_i32_f32(a) -
          nsimd_scalar_reinterpret_i32_f32(b);
  return nsimd_upper_log2((u64)(d >= 0 ? d : -d));
}

NSIMD_DLLEXPORT int nsimd_diff_in_logulps_f64(f64 a, f64 b) {
  i64 d = nsimd_scalar_reinterpret_i64_f64(a) -
          nsimd_scalar_reinterpret_i64_f64(b);
  return nsimd_upper_log2((u64)(d >= 0 ? d : -d));
}

} // extern "C"
