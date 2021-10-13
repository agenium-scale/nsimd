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

#ifdef ENABLE_VSX
#include "renamevsx.h"
#define nsimd_vec_f32 nsimd_vmx_vf32
#define get0(a) vec_extract(a, 0)
#define get1(a) vec_extract(a, 1)
#define get2(a) vec_extract(a, 2)
#define get3(a) vec_extract(a, 3)
#define set0(a, b) vec_splats(b)
#define set1(a, b) vec_insert(b, a, 1)
#define set2(a, b) vec_insert(b, a, 2)
#define set3(a, b) vec_insert(b, a, 3)
#endif

nsimd_vec_f32 xsinf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_sin_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}


nsimd_vec_f32 xcosf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_cos_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xtanf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_tan_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xasinf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_asin_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xacosf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_acos_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xatanf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_atan_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xatan2f(nsimd_vec_f32 a0_, nsimd_vec_f32 a1_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, a1, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  a1.v0 = get0(a1_);
  a1.v1 = get1(a1_);
  a1.v2 = get2(a1_);
  a1.v3 = get3(a1_);
  ret = nsimd_atan2_u35_cpu_f32(a0, a1);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xlogf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_log_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xcbrtf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_cbrt_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xsinf_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_sin_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xcosf_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_cos_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xtanf_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_tan_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xasinf_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_asin_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xacosf_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_acos_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xatanf_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_atan_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xatan2f_u1(nsimd_vec_f32 a0_, nsimd_vec_f32 a1_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, a1, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  a1.v0 = get0(a1_);
  a1.v1 = get1(a1_);
  a1.v2 = get2(a1_);
  a1.v3 = get3(a1_);
  ret = nsimd_atan2_u10_cpu_f32(a0, a1);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xlogf_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_log_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xcbrtf_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_cbrt_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xexpf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_exp_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xpowf(nsimd_vec_f32 a0_, nsimd_vec_f32 a1_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, a1, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  a1.v0 = get0(a1_);
  a1.v1 = get1(a1_);
  a1.v2 = get2(a1_);
  a1.v3 = get3(a1_);
  ret = nsimd_pow_u10_cpu_f32(a0, a1);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xsinhf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_sinh_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xcoshf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_cosh_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xtanhf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_tanh_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xsinhf_u35(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_sinh_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xcoshf_u35(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_cosh_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xtanhf_u35(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_tanh_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xasinhf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_asinh_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xacoshf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_acosh_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xatanhf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_atanh_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xexp2f(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_exp2_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xexp2f_u35(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_exp2_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xexp10f(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_exp10_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xexp10f_u35(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_exp10_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xexpm1f(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_expm1_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xlog10f(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_log10_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xlog2f(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_log2_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xlog2f_u35(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_log2_u35_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xlog1pf(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_log1p_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xsinpif_u05(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_sinpi_u05_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xcospif_u05(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_cospi_u05_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xhypotf_u05(nsimd_vec_f32 a0_, nsimd_vec_f32 a1_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, a1, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  a1.v0 = get0(a1_);
  a1.v1 = get1(a1_);
  a1.v2 = get2(a1_);
  a1.v3 = get3(a1_);
  ret = nsimd_hypot_u05_cpu_f32(a0, a1);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xhypotf_u35(nsimd_vec_f32 a0_, nsimd_vec_f32 a1_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, a1, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  a1.v0 = get0(a1_);
  a1.v1 = get1(a1_);
  a1.v2 = get2(a1_);
  a1.v3 = get3(a1_);
  ret = nsimd_hypot_u35_cpu_f32(a0, a1);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xfmodf(nsimd_vec_f32 a0_, nsimd_vec_f32 a1_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, a1, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  a1.v0 = get0(a1_);
  a1.v1 = get1(a1_);
  a1.v2 = get2(a1_);
  a1.v3 = get3(a1_);
  ret = nsimd_fmod_cpu_f32(a0, a1);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xremainderf(nsimd_vec_f32 a0_, nsimd_vec_f32 a1_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, a1, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  a1.v0 = get0(a1_);
  a1.v1 = get1(a1_);
  a1.v2 = get2(a1_);
  a1.v3 = get3(a1_);
  ret = nsimd_remainder_cpu_f32(a0, a1);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xlgammaf_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_lgamma_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xtgammaf_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_tgamma_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xerff_u1(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_erf_u10_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

nsimd_vec_f32 xerfcf_u15(nsimd_vec_f32 a0_) {
  nsimd_vec_f32 ret_;
  nsimd_cpu_vf32 a0, ret;
  a0.v0 = get0(a0_);
  a0.v1 = get1(a0_);
  a0.v2 = get2(a0_);
  a0.v3 = get3(a0_);
  ret = nsimd_erfc_u15_cpu_f32(a0);
  ret_ = set0(ret_, ret.v0);
  ret_ = set1(ret_, ret.v1);
  ret_ = set2(ret_, ret.v2);
  ret_ = set3(ret_, ret.v3);
  return ret_;
}

