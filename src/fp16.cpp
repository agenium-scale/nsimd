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
  } else if (exponent == 0) {
    /* We have a denormal, treat it as zero. */
    mantissa = 0;
  } else {
    /* the exponent must be recomputed - 15 + 127 */
    exponent += 112;
  }
  /* We then rebuild the float */
  {
    union {
      u32 u;
      f32 f;
    } buf;

    buf.u = (sign << 16) | (((u32)exponent) << 23) | (mantissa << 13);
    return buf.f;
  }
#endif
}

// ----------------------------------------------------------------------------
// Convert a FP16 to a float

#ifndef NSIMD_NATIVE_FP16
NSIMD_DLLEXPORT f32 nsimd_f16_to_f32(f16 a) {
  return nsimd_u16_to_f32(a.u);
}
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
  u32 b, sign, mantissa;
  int exponent;

  {
    union {
      u32 u;
      f32 f;
    } buf;

    buf.f = a;
    b = buf.u;
  }

  sign = b & 0x80000000;
  exponent = (int)((b >> 23) & 0xFF);
  mantissa = (b & 0x7FFFFF);

  if (exponent < 113 || (exponent > 142 && exponent < 255)) {
    /* In this case we have to set the mantissa to zero because
       - exponent is two small or zero and we have either a too small
         number of a subnormal (or zero) and we treat all theses cases as
         zero. In this case E < 127 - 14 = (FP32 bias) + (FP16 + emin).
       - or an exponent encoding a number two large to be represented in a
         FP16 so we convert it to INF and for INF the mantissa must be zero.
         In this case E > 127 + 15 = (FP32 bias) + (FP16 + emax).
     */
    mantissa = 0;
  }

  /* For FP32 exponent bias is 127, compute the real exponent. */
  exponent -= 127;

  /* Now ensure that exponent is between -15 and +16. The exponent must be
     between -14 and 15. But when we add the bias we get an exponent between
     1 and 30. This does take into account zero (exponent is 0) and INF
     (exponent is 31). Therefore we ensure that exponent lives between
     -15 and 16. Indeed an exponent of -15 means a too small number for FP16
     and -15 + 15 = 0 is the good exponent for zero. An exponent of 16 means
     a too big number, so we produce an INF whose exponent is 31 = 16 + 15. */
  exponent = std::min(exponent, 16);
  exponent = std::max(exponent, -15);

  /* Finally rebuild the FP16. */
  return (u16)((sign >> 16) | (((u32)(exponent + 15)) << 10) |
               (mantissa >> 13));
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
