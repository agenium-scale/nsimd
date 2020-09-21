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

/*

We follow IEEE754-2008 for FP16 (= binary16) storage.
However IEEE754 compliance is not guaranteed by C/C++ standards
and therefore we propose two modes:

- IEEE754 mode with NaNs, INFs, ... (this is the default)
- non IEEE754 mode compatible with only C89 (no NaNs, INFs...)

FP16 format
-----------

    +---+--------+--------------+
    | S | E EEEE | MM MMMM MMMM |
    +---+--------+--------------+
     15  14   10   9          0

FP16 interpretation
-------------------

S = sign bit
E = exponent bits (offset is 15), emin = -14, emax = 15
M = mantissa bits

E == 0 and M != 0 => subnormal => (-1)^S x 2^(-14) x (0 + 2^(-10) x T)
32 > E >  0       =>    normal => (-1)^S x 2^(E - 15) x (1 + 2^(-10) x T)

FP32 format
-----------

    +---+-----------+------------------------------+
    | S | EEEE EEEE | MMM MMMM MMMM MMMM MMMM MMMM |
    +---+-----------+------------------------------+
     31  30      23  22                          0

FP32 interpretation
-------------------

S = sign bit
E = exponent bits (offset is 127), emin = -126, emax = 127
M = mantissa bits

E == 0 and M != 0 => subnormal => (-1)^S x 2^(-126) x (0 + 2^(-23) x T)
256 > E > 0       =>    normal => (-1)^S x 2^(E - 127) x (1 + 2^(-23) x T)

In both cases we treat subnormal numbers as zeros. Moreover the
implementation below was written so that it can easily be SIMD'ed.

*/

#define NSIMD_INSIDE
#include <nsimd/nsimd.h>

#ifdef NSIMD_NO_IEEE754
  #include <cmath>
#endif
#include <algorithm>

extern "C" {

/* Union used to manipulate bit in float numbers. */
typedef union {
  u32 u;
  f32 f;
} f32_u32;


// ----------------------------------------------------------------------------
// Convert a FP16 as an u16 to a float

NSIMD_DLLEXPORT float nsimd_u16_to_f32(u16 a) {
#ifdef NSIMD_NO_IEEE754
  float sign;
  int exponent, mantissa;

  sign = (a >> 15) == 1 ? -1.0f : 1.0f;
  exponent = (a >> 10) & 0x1F;
  mantissa = (float)(a & 0x3FF);

  if (exponent == 0) {
    return std::ldexp(sign * mantissa, -24);
  } else {
    return std::ldexp(sign * (0x400 | mantissa), exponent - 25);
  }
#else
  u32 sign, mantissa, exponent;

  sign = a & 0x8000;
  exponent = (a >> 10) & 0x1F;
  mantissa = (a & 0x3FF);

  if (exponent == 31) {
    /* We have a NaN of an INF. */
    exponent = 255;
    /* Force the first bit of the mantissa to 1 to be compatible with the way
     * Intel convert f16 to f32 */
    if (mantissa != 0) {
      //mantissa |= 0x200;
    }
  } else if (exponent == 0 && mantissa == 0) {
    /* Nothing to do */
  } else if (exponent == 0) {
    u32 mask = mantissa;
    /* Find the most significant bit of the mantissa (could use a better
     * algorithm) */
    int i = -1;
    do {
      ++i;
      mask <<= 1;
    } while ((mask & 0x400) == 0);

    /* Update the mantissa and the exponent */
    mantissa = (mask & 0x3ff);
    exponent += (u32)(112 - i);
  } else {
    /* the exponent must be recomputed -15 + 127 */
    exponent += 112;
  }
  /* We then rebuild the float */
  f32_u32 ret;
  ret.u = (sign << 16) | (((u32)exponent) << 23) | (mantissa << 13);
  return ret.f;
#endif
}

// ----------------------------------------------------------------------------
// Convert a FP16 to a float

#ifndef NSIMD_NATIVE_FP16
NSIMD_DLLEXPORT f32 nsimd_f16_to_f32(f16 a) { return nsimd_u16_to_f32(a.u); }
#endif

// ----------------------------------------------------------------------------
// Convert a float to a FP16 as an u16

NSIMD_DLLEXPORT u16 nsimd_f32_to_u16(f32 a) {
#ifdef NSIMD_NO_IEEE754
  double frac;
  int exponent;
  u32 sign, mantissa;

  /* Get mantissa (= fractional part) and exponent. */
  frac = std::frexp(a, &exponent);

  /* Get sign and make sure frac is positive. */
  if (frac < 0) {
    sign = 1u;
    frac = -frac;
  } else {
    sign = 0u;
  }

  /* Add 1 to the exponent to have the IEEE exponent: The mantissa here
     lives in [0.5, 1) whereas for IEEE it must live in [1, 2). */
  exponent++;

  if (exponent < -14) {
    /* We have a too small number, returns zero */
    return (u16)(sign << 15);
  } else if (exponent > 15) {
    /* We have a too big number, return INF */
    return (u16)((sign << 15) | 0x7C00);
  } else {
    /* We have a normal number. Get the mantissa:
       frac lives in [0.5, 1) and is of the form 0.1XXXXXXX, therefore
       to get the mantissa frac must be multiplied by 2^11 = 2048. Then
       it will be of the form 1XX XXXX XXXX.XXXXX, so we have to get rid
       of the leading bit. */
    mantissa = (u32)(frac * 2048.0) & 0x3FF;
    return (u16)((sign << 15) | ((u32)(exponent + 15) << 10) | mantissa);
  }
#else
  u32 sign, mantissa;
  int exponent;

  f32_u32 in;
  in.f = a;

  sign = in.u & 0x80000000;
  exponent = (int)((in.u >> 23) & 0xFF);
  mantissa = (in.u & 0x7FFFFF);

  if (exponent == 255 && mantissa != 0) {
    /* Nan */
    return (u16)(0xffff);
  }

  const f32_u32 biggest_f16 = {0x477ff000};
  if (in.f >= biggest_f16.f || in.f <= -biggest_f16.f) {
    /* Number is too big to be representable in half => return infinity */
    return (u16)(sign >> 16 | 0x1f << 10);
  }

  const f32_u32 smallest_f16 = {0x33000000};
  if (in.f <= smallest_f16.f && in.f >= -smallest_f16.f) {
    /* Number is too small to be representable in half => return ±0 */
    return (u16)(sign >> 16);
  }

  /* For FP32 exponent bias is 127, compute the real exponent. */
  exponent -= 127;

  /* Following algorithm taken from:
   * https://fgiesen.wordpress.com/2012/03/28/half-to-float-done-quic/ */
  const f32_u32 denormal_f16 = {0x38800000};
  if (in.f < denormal_f16.f && in.f > -denormal_f16.f) {
    /* Denormalized half */
    const f32_u32 magic = {((127 - 15) + (23 - 10) + 1) << 23};

    in.u &= ~0x80000000U;
    in.f += magic.f;
    in.u -= magic.u;

    return (u16)(sign >> 16 | in.u);
  }

  /* Normal half */
  in.u &= ~0x80000000U;
  u32 mant_odd = (in.u >> 13) & 1;
  in.u += ((u32)(15 - 127) << 23) + 0xfffU;
  in.u += mant_odd;

  return (u16)(sign >> 16 | in.u >> 13);
#endif
}

// ----------------------------------------------------------------------------
// Convert a float to a FP16

#ifndef NSIMD_NATIVE_FP16
NSIMD_DLLEXPORT f16 nsimd_f32_to_f16(f32 a) {
  f16 ret;
  ret.u = nsimd_f32_to_u16(a);
  return ret;
}
#endif

// ----------------------------------------------------------------------------

} // extern "C"

// ----------------------------------------------------------------------------
// C++ versions in namespace nsimd

namespace nsimd {

NSIMD_DLLEXPORT u16 f32_to_u16(f32 a) { return nsimd_f32_to_u16(a); }
NSIMD_DLLEXPORT f32 u16_to_f32(u16 a) { return nsimd_u16_to_f32(a); }
#ifndef NSIMD_NATIVE_FP16
NSIMD_DLLEXPORT f16 f32_to_f16(f32 a) { return nsimd_f32_to_f16(a); }
NSIMD_DLLEXPORT f32 f16_to_f32(f16 a) { return nsimd_f16_to_f32(a); }
#endif

} // namespace nsimd
