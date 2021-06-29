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

#ifdef ENABLE_NEON32
#include "renameneon32.h"
#define nsimd_vec_f64 nsimd_neon128_vf64
#endif

nsimd_vec_f64 xsin(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_sin_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}


nsimd_vec_f64 xcos(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_cos_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xtan(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_tan_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xasin(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_asin_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xacos(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_acos_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xatan(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_atan_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xatan2(nsimd_vec_f64 a0_, nsimd_vec_f64 a1_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, a1, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  a1.v0 = a1_.v0;
  a1.v1 = a1_.v1;
  ret = nsimd_atan2_u35_cpu_f64(a0, a1);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xlog(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_log_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xcbrt(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_cbrt_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xsin_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_sin_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xcos_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_cos_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xtan_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_tan_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xasin_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_asin_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xacos_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_acos_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xatan_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_atan_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xatan2_u1(nsimd_vec_f64 a0_, nsimd_vec_f64 a1_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, a1, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  a1.v0 = a1_.v0;
  a1.v1 = a1_.v1;
  ret = nsimd_atan2_u10_cpu_f64(a0, a1);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xlog_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_log_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xcbrt_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_cbrt_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xexp(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_exp_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xpow(nsimd_vec_f64 a0_, nsimd_vec_f64 a1_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, a1, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  a1.v0 = a1_.v0;
  a1.v1 = a1_.v1;
  ret = nsimd_pow_u10_cpu_f64(a0, a1);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xsinh(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_sinh_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xcosh(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_cosh_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xtanh(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_tanh_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xsinh_u35(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_sinh_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xcosh_u35(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_cosh_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xtanh_u35(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_tanh_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xasinh(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_asinh_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xacosh(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_acosh_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xatanh(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_atanh_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xexp2(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_exp2_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xexp2_u35(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_exp2_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xexp10(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_exp10_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xexp10_u35(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_exp10_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xexpm1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_expm1_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xlog10(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_log10_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xlog2(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_log2_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xlog2_u35(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_log2_u35_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xlog1p(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_log1p_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xsinpi_u05(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_sinpi_u05_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xcospi_u05(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_cospi_u05_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xhypot_u05(nsimd_vec_f64 a0_, nsimd_vec_f64 a1_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, a1, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  a1.v0 = a1_.v0;
  a1.v1 = a1_.v1;
  ret = nsimd_hypot_u05_cpu_f64(a0, a1);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xhypot_u35(nsimd_vec_f64 a0_, nsimd_vec_f64 a1_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, a1, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  a1.v0 = a1_.v0;
  a1.v1 = a1_.v1;
  ret = nsimd_hypot_u35_cpu_f64(a0, a1);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xfmod(nsimd_vec_f64 a0_, nsimd_vec_f64 a1_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, a1, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  a1.v0 = a1_.v0;
  a1.v1 = a1_.v1;
  ret = nsimd_fmod_cpu_f64(a0, a1);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xremainder(nsimd_vec_f64 a0_, nsimd_vec_f64 a1_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, a1, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  a1.v0 = a1_.v0;
  a1.v1 = a1_.v1;
  ret = nsimd_remainder_cpu_f64(a0, a1);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xlgamma_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_lgamma_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xtgamma_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_tgamma_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xerf_u1(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_erf_u10_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

nsimd_vec_f64 xerfc_u15(nsimd_vec_f64 a0_) {
  nsimd_vec_f64 ret_;
  nsimd_cpu_vf64 a0, ret;
  a0.v0 = a0_.v0;
  a0.v1 = a0_.v1;
  ret = nsimd_erfc_u15_cpu_f64(a0);
  ret_.v0 = ret.v0;
  ret_.v1 = ret.v1;
  return ret_;
}

