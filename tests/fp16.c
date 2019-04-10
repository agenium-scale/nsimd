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

#define _POSIX_C_SOURCE 200112L

#include <nsimd/nsimd.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/* ------------------------------------------------------------------------- */

#ifndef NSIMD_NO_IEEE754

int is_nan(float a) {
  union {
    u32 u;
    f32 f;
  } buf;
  buf.f = a;
  return ((buf.u & 0x7FFFFF) != 0u) && ((buf.u & 0x7F800000) == 0x7F800000);
}

#endif

/* ------------------------------------------------------------------------- */

float via_fp16(float a) {
  return nsimd_f16_to_f32(nsimd_f32_to_f16(a));
}

/* ------------------------------------------------------------------------- */

float mk_fp32(int mantissa, int exponent) {
  return (float)ldexp((double)mantissa, exponent);
}

/* ------------------------------------------------------------------------- */

#ifndef NSIMD_NO_IEEE754

float mk_fp32_bin(u32 a) {
  union {
    u32 u;
    f32 f;
  } buf;
  buf.u = a;
  return buf.f;
}

#endif

/* ------------------------------------------------------------------------- */

int main(void) {
#ifndef NSIMD_NO_IEEE754
  const float infty = mk_fp32_bin(0x7F800000);
  const float m_infty = mk_fp32_bin(0xFF800000);
  const float nan = mk_fp32_bin(0x7FC00000);
#endif
  int i;

  /* Some corner cases first. */
  if (via_fp16(0.0f) != 0.0f) {
    return EXIT_FAILURE;
  }
  if (via_fp16(1.0f) != 1.0f) {
    return EXIT_FAILURE;
  }
  if (via_fp16(-1.0f) != -1.0f) {
    return EXIT_FAILURE;
  }
#ifndef NSIMD_NO_IEEE754
  if (via_fp16(mk_fp32(1, 20)) != infty) {
    return EXIT_FAILURE;
  }
  if (via_fp16(mk_fp32(-1, 20)) != m_infty) {
    return EXIT_FAILURE;
  }
  if (!is_nan(via_fp16(nan))) {
    return EXIT_FAILURE;
  }
#endif

  /* Some random inputs */
  for (i = 0; i < 100; i++) {
    float a = (float)rand() / (float)RAND_MAX;
    if (fabsf(a - via_fp16(a)) > ldexpf(1.0, -9)) {
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
