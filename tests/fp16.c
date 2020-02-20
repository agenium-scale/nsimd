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

#include <math.h>
#include <nsimd/nsimd.h>
#include <stdio.h>
#include <stdlib.h>

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

float via_fp16(float a) { return nsimd_f16_to_f32(nsimd_f32_to_f16(a)); }

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

int test_f16_to_f32(u16 val, u32 expected) {
  f32 fexpected = *(f32 *)&expected;

  f32 res = nsimd_f16_to_f32(*(f16 *)&val);
  u32 ures = *(u32 *)&res;
  if (ures != expected) {
    fprintf(stdout,
            "Error, nsimd_f16_to_f32: expected %e(0x%x) but got %e(0x%x) \n",
            (f64)fexpected, expected, (f64)res, ures);
    fflush(stdout);
    return 1;
  }

  return 0;
}

/* ------------------------------------------------------------------------- */

int test_f32_to_f16(u32 val, u16 expected) {
  f16 fres = nsimd_f32_to_f16(*(f32 *)&val);
  u16 ures = *(u16 *)&fres;
  if (ures != expected) {
    fprintf(stdout, "Error, nsimd_f16_to_f32: expected 0x%x but got 0x%x \n",
            expected, ures);
    fflush(stdout);
    return 1;
  }

  return 0;
}

/* ------------------------------------------------------------------------- */

int main(void) {
#ifndef NSIMD_NO_IEEE754
  const float infty = mk_fp32_bin(0x7F800000);
  const float m_infty = mk_fp32_bin(0xFF800000);
  const float nan = mk_fp32_bin(0x7FC00000);
#endif
  int i;

  /* Some corner cases first. */
  if (test_f16_to_f32(0x0000, 0x0)) {
    return EXIT_FAILURE;
  }
  if (test_f16_to_f32(0x8000, 0x80000000)) {
    return EXIT_FAILURE;
  }
  if (test_f16_to_f32(0x3C00, 0x3f800000)) {
    return EXIT_FAILURE;
  }
  if (test_f16_to_f32(0x13e, 0x379F0000)) { // 1.8954277E-5
    return EXIT_FAILURE;
  }
  if (test_f16_to_f32(0x977e, 0xBAEFC000)) { // -1.8291473E-3
    return EXIT_FAILURE;
  }

  if (test_f32_to_f16(0xC7BDC4FC, 0xFC00)) { // -97161.97
    return EXIT_FAILURE;
  }

  if (test_f32_to_f16(0x37c3642c, 0x187)) { // 2.329246e-05
    return EXIT_FAILURE;
  }

  if (test_f32_to_f16(0xb314e840, 0x8001)) {
    return EXIT_FAILURE;
  }

  /* Test rounding when the input f32 is perfectly between 2 f16*/
  if (test_f32_to_f16(0xC66AD000, 0xf356)) {
    return EXIT_FAILURE;
  }

  /* Close to ±Inf */
  if (test_f32_to_f16(0x477fefff, 0x7bff)) {
    return EXIT_FAILURE;
  }
  if (test_f32_to_f16(0x477ff000, 0x7c00)) {
    return EXIT_FAILURE;
  }
  if (test_f32_to_f16(0xC77fefff, 0xfbff)) {
    return EXIT_FAILURE;
  }
  if (test_f32_to_f16(0xC77ff000, 0xfc00)) {
    return EXIT_FAILURE;
  }

  /* Close to ±0 */
  if (test_f32_to_f16(0x33000001, 0x0001)) {
    return EXIT_FAILURE;
  }
  if (test_f32_to_f16(0x33000000, 0x0000)) {
    return EXIT_FAILURE;
  }
  if (test_f32_to_f16(0xB3000001, 0x8001)) {
    return EXIT_FAILURE;
  }
  if (test_f32_to_f16(0xB3000000, 0x8000)) {
    return EXIT_FAILURE;
  }

  /* Close to the denormal limit */
  if (test_f32_to_f16(0x38800000, 0x0400)) {
    return EXIT_FAILURE;
  }
  if (test_f32_to_f16(0x387fffff, 0x0400)) {
    return EXIT_FAILURE;
  }

  /* NaN special value (Copy Intel intrinsics which set the MSB of the mantissa
   * of NaNs to 1 when converting f16 to f32). */
  if (test_f16_to_f32(0xfcf8, 0xff9f0000)) {
    return EXIT_FAILURE;
  }

#ifndef NSIMD_NO_IEEE754
  if (via_fp16(mk_fp32(1, 20)) != infty) {
    fprintf(stdout, "... Error, %i \n", __LINE__);
    fflush(stdout);
    return EXIT_FAILURE;
  }
  if (via_fp16(mk_fp32(-1, 20)) != m_infty) {
    fprintf(stdout, "... Error, %i \n", __LINE__);
    fflush(stdout);
    return EXIT_FAILURE;
  }
  if (!is_nan(via_fp16(nan))) {
    fprintf(stdout, "... Error, %i \n", __LINE__);
    fflush(stdout);
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

  fprintf(stdout, "... OK\n");
  fflush(stdout);
  return EXIT_SUCCESS;
}
