<!--

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

-->

# API

## Miscellaneous

- [Vector length (len)](api_len.md)
- [Value broadcast (set1)](api_set1.md)
- [Horizontal sum (addv)](api_addv.md)
- [Blend (if_else1)](api_if-else1.md)
- [Check all elements (all)](api_all.md)
- [Check for one true elements (any)](api_any.md)
- [Count true elements (nbtrue)](api_nbtrue.md)
- [Ziplo (ziplo)](api_ziplo.md)
- [Ziphi (ziphi)](api_ziphi.md)
- [Unziplo (unziplo)](api_unziplo.md)
- [Unziphi (unziphi)](api_unziphi.md)
- [Zip (zip)](api_zip.md)
- [Unzip (unzip)](api_unzip.md)
- [Gamma function (gamma)](api_gamma.md)
- [Logarithm of the gamma function (lgamma)](api_lgamma.md)
- [Stirling function (stirling)](api_stirling.md)
- [Error function (erf)](api_erf.md)
- [Complementary error function (erfc)](api_erfc.md)
- [Scaled complementary error function (erfcx)](api_erfcx.md)

## Loads & stores

- [Loadu (loadu)](api_loadu.md)
- [Load array of structure (load2u)](api_load2u.md)
- [Load array of structure (load3u)](api_load3u.md)
- [Load array of structure (load4u)](api_load4u.md)
- [Loada (loada)](api_loada.md)
- [Load array of structure (load2a)](api_load2a.md)
- [Load array of structure (load3a)](api_load3a.md)
- [Load array of structure (load4a)](api_load4a.md)
- [Load vector of logicals (loadlu)](api_loadlu.md)
- [Load vector of logicals (loadla)](api_loadla.md)
- [Storeu (storeu)](api_storeu.md)
- [Store into array of structures (store2u)](api_store2u.md)
- [Store into array of structures (store3u)](api_store3u.md)
- [Store into array of structures (store4u)](api_store4u.md)
- [Storea (storea)](api_storea.md)
- [Store into array of structures (store2a)](api_store2a.md)
- [Store into array of structures (store3a)](api_store3a.md)
- [Store into array of structures (store4a)](api_store4a.md)
- [Store vector of logicals (storelu)](api_storelu.md)
- [Store vector of logicals (storela)](api_storela.md)

## Bits manipulation operators

- [Bitwise or (orb)](api_orb.md)
- [Bitwise and (andb)](api_andb.md)
- [Bitwise andnot (andnotb)](api_andnotb.md)
- [Bitwise not (notb)](api_notb.md)
- [Bitwise xor (xorb)](api_xorb.md)
- [Right shift in zeros (shr)](api_shr.md)
- [Left shift (shl)](api_shl.md)
- [Arithmetic right shift (shra)](api_shra.md)

## Logicals operators

- [Logical or (orl)](api_orl.md)
- [Logical and (andl)](api_andl.md)
- [Logical andnot (andnotl)](api_andnotl.md)
- [Logical xor (xorl)](api_xorl.md)
- [Logical not (notl)](api_notl.md)
- [Build mask from logicals (to_mask)](api_to-mask.md)
- [Build logicals from data (to_logical)](api_to-logical.md)

## Basic arithmetic operators

- [Addition (add)](api_add.md)
- [Subtraction (sub)](api_sub.md)
- [Multiplication (mul)](api_mul.md)
- [Division (div)](api_div.md)
- [Opposite (neg)](api_neg.md)
- [Minimum (min)](api_min.md)
- [Maximum (max)](api_max.md)
- [Absolute value (abs)](api_abs.md)
- [Fused multiply-add (fma)](api_fma.md)
- [Fused negate-multiply-add (fnma)](api_fnma.md)
- [Fused multiply-substract (fms)](api_fms.md)
- [Fused negate-multiply-substract (fnms)](api_fnms.md)
- [Reciprocal (rec)](api_rec.md)
- [Reciprocal with relative error at most $2^{-11}$ (rec11)](api_rec11.md)
- [Reciprocal with relative error at most 2^{-8} (rec8)](api_rec8.md)
- [Square root (sqrt)](api_sqrt.md)
- [Square root with relative error at most $2^{-11}$ (rsqrt11)](api_rsqrt11.md)
- [Square root with relative error at most $2^{-8}$ (rsqrt8)](api_rsqrt8.md)
- [Power of absolute value (pow_abs)](api_pow-abs.md)
- [Power (pow)](api_pow.md)

## Comparison operators

- [Compare for equality (eq)](api_eq.md)
- [Compare for inequality (ne)](api_ne.md)
- [Compare for greater-than (gt)](api_gt.md)
- [Compare for greater-or-equal-than (ge)](api_ge.md)
- [Compare for lesser-than (lt)](api_lt.md)
- [Compare for lesser-or-equal-than (le)](api_le.md)

## Rounding functions

- [Rounding up to integer value (ceil)](api_ceil.md)
- [Rounding down to integer value (floor)](api_floor.md)
- [Rounding towards zero to integer value (trunc)](api_trunc.md)
- [Rounding to nearest integer value, tie to even (round_to_even)](api_round-to-even.md)

## Conversion operators

- [Reinterpret vector (reinterpret)](api_reinterpret.md)
- [Reinterpret vector of logicals (reinterpretl)](api_reinterpretl.md)
- [Convert vector (cvt)](api_cvt.md)
- [Convert vector to larger type (upcvt)](api_upcvt.md)
- [Convert vector to narrow type (downcvt)](api_downcvt.md)

## Trigonometric functions

- [Cosine (cos)](api_cos.md)
- [Cosine with input in degrees (cosd)](api_cosd.md)
- [Cosine with input in multiples of π (cospi)](api_cospi.md)
- [Arc-cosine (acos)](api_acos.md)
- [Arc-cosine with output in degrees (acosd)](api_acosd.md)
- [Arc-cosine with output in multiple of π (acospi)](api_acospi.md)
- [Sinus (sin)](api_sin.md)
- [Sinus with input in multiple of π (sinpi)](api_sinpi.md)
- [Sinus with input in degrees (sind)](api_sind.md)
- [Arc-sinus (asin)](api_asin.md)
- [Arc-sinus with output in degrees (asind)](api_asind.md)
- [Arc-sinus with output in multiples of π (asinpi)](api_asinpi.md)
- [Tangent (tan)](api_tan.md)
- [Tangent with input in degrees (tand)](api_tand.md)
- [Tangent with input multiples of π (tanpi)](api_tanpi.md)
- [Arc-tangent (atan)](api_atan.md)
- [Arc-tangent with output in degrees (atand)](api_atand.md)
- [Arc-tangent with output in multiples of π (atanpi)](api_atanpi.md)
- [Sinc (sinc)](api_sinc.md)
- [Sinc with input in mutiples of π (sincpi)](api_sincpi.md)
- [Arc-tangent (atan2)](api_atan2.md)
- [Arc-tangent with output in degrees (atan2d)](api_atan2d.md)
- [Arc-tangent with output in multiples of π (atan2pi)](api_atan2pi.md)
- [Cotangent (cot)](api_cot.md)
- [Cotangent with input in degrees (cotd)](api_cotd.md)
- [Cotangent with input in multiples of π (cotpi)](api_cotpi.md)
- [Arc-cotangent (acot)](api_acot.md)
- [Arc-cotangent with output in degrees (acotd)](api_acotd.md)
- [Arc-cotangent with output in multiples of π (acotpi)](api_acotpi.md)
- [Cosecant (csc)](api_csc.md)
- [Cosecant with input in degrees (cscd)](api_cscd.md)
- [Cosecant with input in multiples of π (cscpi)](api_cscpi.md)
- [Arc-cosecant (acsc)](api_acsc.md)
- [Arc-cosecant with output in degrees (acscd)](api_acscd.md)
- [Arc-cosecant with output in multiples of π (acscpi)](api_acscpi.md)
- [Secant (sec)](api_sec.md)
- [Secant with input in degrees (secd)](api_secd.md)
- [Secant with input in multiples of π (secpi)](api_secpi.md)
- [Arc-secant (asec)](api_asec.md)
- [Arc-secant with output in degrees (asecd)](api_asecd.md)
- [Arc-secant with output in multiples of π (asecpi)](api_asecpi.md)

## Hyperbolic functions

- [Hyperbolic cosine (cosh)](api_cosh.md)
- [Hyperbolic arc-cosine (acosh)](api_acosh.md)
- [Hyperbolic sinus (sinh)](api_sinh.md)
- [Hyperbolic arc-sinus (asinh)](api_asinh.md)
- [Hyperbolic tangent (tanh)](api_tanh.md)
- [Hyperbolic arc-tangent (atanh)](api_atanh.md)
- [Sinhc (sinhc)](api_sinhc.md)
- [Hyperbolic cotangent (coth)](api_coth.md)
- [Hyperbolic arc-cotangent (acoth)](api_acoth.md)
- [Hyperbolic cosecant (csch)](api_csch.md)
- [Hyperbolic arc-cosecant (acsch)](api_acsch.md)
- [Hyperbolic secant (sech)](api_sech.md)
- [Hyperbolic arc-secant (asech)](api_asech.md)

## Exponential and logarithmic functions

- [Exponential (exp)](api_exp.md)
- [Base 2 exponential (exp2)](api_exp2.md)
- [Base 10 exponential (exp10)](api_exp10.md)
- [Exponential minus one (expm1)](api_expm1.md)
- [$1 - e^{-1/x}$ (exprecnegc)](api_exprecnegc.md)
- [$e^{-1/x}$ (exprecneg)](api_exprecneg.md)
- [$e^{x^2}$ (expx2)](api_expx2.md)
- [Natural logarithm (log)](api_log.md)
- [Base 2 logarithm (log2)](api_log2.md)
- [Base 10 logarithm (log10)](api_log10.md)
- [$\log(1 + x)$ (log1p)](api_log1p.md)
