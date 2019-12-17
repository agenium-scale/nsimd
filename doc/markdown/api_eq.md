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

# Compare for equality

## Description

Compare the inputs for equality.

## C base API (generic)

```c
#define veq(a0, a1, type)
#define veq_e(a0, a1, type, simd_ext)
```

## C++ base API (generic)

```c++
template <typename A0, typename A1, typename T> typename simd_traits<T, NSIMD_SIMD>::simd_vectorl eq(A0 a0, A1 a1, T);
```

## C++ advanced API

```c++
template <typename T, typename SimdExt> packl<T, 1, SimdExt> eq(pack<T, 1, SimdExt> const& a0, pack<T, 1, SimdExt> const& a1);
template <typename T, int N, typename SimdExt> packl<T, N, SimdExt> eq(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);
template <typename T, typename SimdExt> packl<T, 1, SimdExt> operator==(pack<T, 1, SimdExt> const& a0, pack<T, 1, SimdExt> const& a1);
template <typename T, int N, typename SimdExt> packl<T, N, SimdExt> operator==(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);
```

## C base API (architecture specifics)

### NEON128

```c
nsimd_neon128_vlf64 nsimd_eq_neon128_f64(nsimd_neon128_vf64 a0, nsimd_neon128_vf64 a1);
nsimd_neon128_vlf32 nsimd_eq_neon128_f32(nsimd_neon128_vf32 a0, nsimd_neon128_vf32 a1);
nsimd_neon128_vlf16 nsimd_eq_neon128_f16(nsimd_neon128_vf16 a0, nsimd_neon128_vf16 a1);
nsimd_neon128_vli64 nsimd_eq_neon128_i64(nsimd_neon128_vi64 a0, nsimd_neon128_vi64 a1);
nsimd_neon128_vli32 nsimd_eq_neon128_i32(nsimd_neon128_vi32 a0, nsimd_neon128_vi32 a1);
nsimd_neon128_vli16 nsimd_eq_neon128_i16(nsimd_neon128_vi16 a0, nsimd_neon128_vi16 a1);
nsimd_neon128_vli8 nsimd_eq_neon128_i8(nsimd_neon128_vi8 a0, nsimd_neon128_vi8 a1);
nsimd_neon128_vlu64 nsimd_eq_neon128_u64(nsimd_neon128_vu64 a0, nsimd_neon128_vu64 a1);
nsimd_neon128_vlu32 nsimd_eq_neon128_u32(nsimd_neon128_vu32 a0, nsimd_neon128_vu32 a1);
nsimd_neon128_vlu16 nsimd_eq_neon128_u16(nsimd_neon128_vu16 a0, nsimd_neon128_vu16 a1);
nsimd_neon128_vlu8 nsimd_eq_neon128_u8(nsimd_neon128_vu8 a0, nsimd_neon128_vu8 a1);
```

### AVX2

```c
nsimd_avx2_vlf64 nsimd_eq_avx2_f64(nsimd_avx2_vf64 a0, nsimd_avx2_vf64 a1);
nsimd_avx2_vlf32 nsimd_eq_avx2_f32(nsimd_avx2_vf32 a0, nsimd_avx2_vf32 a1);
nsimd_avx2_vlf16 nsimd_eq_avx2_f16(nsimd_avx2_vf16 a0, nsimd_avx2_vf16 a1);
nsimd_avx2_vli64 nsimd_eq_avx2_i64(nsimd_avx2_vi64 a0, nsimd_avx2_vi64 a1);
nsimd_avx2_vli32 nsimd_eq_avx2_i32(nsimd_avx2_vi32 a0, nsimd_avx2_vi32 a1);
nsimd_avx2_vli16 nsimd_eq_avx2_i16(nsimd_avx2_vi16 a0, nsimd_avx2_vi16 a1);
nsimd_avx2_vli8 nsimd_eq_avx2_i8(nsimd_avx2_vi8 a0, nsimd_avx2_vi8 a1);
nsimd_avx2_vlu64 nsimd_eq_avx2_u64(nsimd_avx2_vu64 a0, nsimd_avx2_vu64 a1);
nsimd_avx2_vlu32 nsimd_eq_avx2_u32(nsimd_avx2_vu32 a0, nsimd_avx2_vu32 a1);
nsimd_avx2_vlu16 nsimd_eq_avx2_u16(nsimd_avx2_vu16 a0, nsimd_avx2_vu16 a1);
nsimd_avx2_vlu8 nsimd_eq_avx2_u8(nsimd_avx2_vu8 a0, nsimd_avx2_vu8 a1);
```

### AVX512_KNL

```c
nsimd_avx512_knl_vlf64 nsimd_eq_avx512_knl_f64(nsimd_avx512_knl_vf64 a0, nsimd_avx512_knl_vf64 a1);
nsimd_avx512_knl_vlf32 nsimd_eq_avx512_knl_f32(nsimd_avx512_knl_vf32 a0, nsimd_avx512_knl_vf32 a1);
nsimd_avx512_knl_vlf16 nsimd_eq_avx512_knl_f16(nsimd_avx512_knl_vf16 a0, nsimd_avx512_knl_vf16 a1);
nsimd_avx512_knl_vli64 nsimd_eq_avx512_knl_i64(nsimd_avx512_knl_vi64 a0, nsimd_avx512_knl_vi64 a1);
nsimd_avx512_knl_vli32 nsimd_eq_avx512_knl_i32(nsimd_avx512_knl_vi32 a0, nsimd_avx512_knl_vi32 a1);
nsimd_avx512_knl_vli16 nsimd_eq_avx512_knl_i16(nsimd_avx512_knl_vi16 a0, nsimd_avx512_knl_vi16 a1);
nsimd_avx512_knl_vli8 nsimd_eq_avx512_knl_i8(nsimd_avx512_knl_vi8 a0, nsimd_avx512_knl_vi8 a1);
nsimd_avx512_knl_vlu64 nsimd_eq_avx512_knl_u64(nsimd_avx512_knl_vu64 a0, nsimd_avx512_knl_vu64 a1);
nsimd_avx512_knl_vlu32 nsimd_eq_avx512_knl_u32(nsimd_avx512_knl_vu32 a0, nsimd_avx512_knl_vu32 a1);
nsimd_avx512_knl_vlu16 nsimd_eq_avx512_knl_u16(nsimd_avx512_knl_vu16 a0, nsimd_avx512_knl_vu16 a1);
nsimd_avx512_knl_vlu8 nsimd_eq_avx512_knl_u8(nsimd_avx512_knl_vu8 a0, nsimd_avx512_knl_vu8 a1);
```

### AVX

```c
nsimd_avx_vlf64 nsimd_eq_avx_f64(nsimd_avx_vf64 a0, nsimd_avx_vf64 a1);
nsimd_avx_vlf32 nsimd_eq_avx_f32(nsimd_avx_vf32 a0, nsimd_avx_vf32 a1);
nsimd_avx_vlf16 nsimd_eq_avx_f16(nsimd_avx_vf16 a0, nsimd_avx_vf16 a1);
nsimd_avx_vli64 nsimd_eq_avx_i64(nsimd_avx_vi64 a0, nsimd_avx_vi64 a1);
nsimd_avx_vli32 nsimd_eq_avx_i32(nsimd_avx_vi32 a0, nsimd_avx_vi32 a1);
nsimd_avx_vli16 nsimd_eq_avx_i16(nsimd_avx_vi16 a0, nsimd_avx_vi16 a1);
nsimd_avx_vli8 nsimd_eq_avx_i8(nsimd_avx_vi8 a0, nsimd_avx_vi8 a1);
nsimd_avx_vlu64 nsimd_eq_avx_u64(nsimd_avx_vu64 a0, nsimd_avx_vu64 a1);
nsimd_avx_vlu32 nsimd_eq_avx_u32(nsimd_avx_vu32 a0, nsimd_avx_vu32 a1);
nsimd_avx_vlu16 nsimd_eq_avx_u16(nsimd_avx_vu16 a0, nsimd_avx_vu16 a1);
nsimd_avx_vlu8 nsimd_eq_avx_u8(nsimd_avx_vu8 a0, nsimd_avx_vu8 a1);
```

### AVX512_SKYLAKE

```c
nsimd_avx512_skylake_vlf64 nsimd_eq_avx512_skylake_f64(nsimd_avx512_skylake_vf64 a0, nsimd_avx512_skylake_vf64 a1);
nsimd_avx512_skylake_vlf32 nsimd_eq_avx512_skylake_f32(nsimd_avx512_skylake_vf32 a0, nsimd_avx512_skylake_vf32 a1);
nsimd_avx512_skylake_vlf16 nsimd_eq_avx512_skylake_f16(nsimd_avx512_skylake_vf16 a0, nsimd_avx512_skylake_vf16 a1);
nsimd_avx512_skylake_vli64 nsimd_eq_avx512_skylake_i64(nsimd_avx512_skylake_vi64 a0, nsimd_avx512_skylake_vi64 a1);
nsimd_avx512_skylake_vli32 nsimd_eq_avx512_skylake_i32(nsimd_avx512_skylake_vi32 a0, nsimd_avx512_skylake_vi32 a1);
nsimd_avx512_skylake_vli16 nsimd_eq_avx512_skylake_i16(nsimd_avx512_skylake_vi16 a0, nsimd_avx512_skylake_vi16 a1);
nsimd_avx512_skylake_vli8 nsimd_eq_avx512_skylake_i8(nsimd_avx512_skylake_vi8 a0, nsimd_avx512_skylake_vi8 a1);
nsimd_avx512_skylake_vlu64 nsimd_eq_avx512_skylake_u64(nsimd_avx512_skylake_vu64 a0, nsimd_avx512_skylake_vu64 a1);
nsimd_avx512_skylake_vlu32 nsimd_eq_avx512_skylake_u32(nsimd_avx512_skylake_vu32 a0, nsimd_avx512_skylake_vu32 a1);
nsimd_avx512_skylake_vlu16 nsimd_eq_avx512_skylake_u16(nsimd_avx512_skylake_vu16 a0, nsimd_avx512_skylake_vu16 a1);
nsimd_avx512_skylake_vlu8 nsimd_eq_avx512_skylake_u8(nsimd_avx512_skylake_vu8 a0, nsimd_avx512_skylake_vu8 a1);
```

### SVE

```c
nsimd_sve_vlf64 nsimd_eq_sve_f64(nsimd_sve_vf64 a0, nsimd_sve_vf64 a1);
nsimd_sve_vlf32 nsimd_eq_sve_f32(nsimd_sve_vf32 a0, nsimd_sve_vf32 a1);
nsimd_sve_vlf16 nsimd_eq_sve_f16(nsimd_sve_vf16 a0, nsimd_sve_vf16 a1);
nsimd_sve_vli64 nsimd_eq_sve_i64(nsimd_sve_vi64 a0, nsimd_sve_vi64 a1);
nsimd_sve_vli32 nsimd_eq_sve_i32(nsimd_sve_vi32 a0, nsimd_sve_vi32 a1);
nsimd_sve_vli16 nsimd_eq_sve_i16(nsimd_sve_vi16 a0, nsimd_sve_vi16 a1);
nsimd_sve_vli8 nsimd_eq_sve_i8(nsimd_sve_vi8 a0, nsimd_sve_vi8 a1);
nsimd_sve_vlu64 nsimd_eq_sve_u64(nsimd_sve_vu64 a0, nsimd_sve_vu64 a1);
nsimd_sve_vlu32 nsimd_eq_sve_u32(nsimd_sve_vu32 a0, nsimd_sve_vu32 a1);
nsimd_sve_vlu16 nsimd_eq_sve_u16(nsimd_sve_vu16 a0, nsimd_sve_vu16 a1);
nsimd_sve_vlu8 nsimd_eq_sve_u8(nsimd_sve_vu8 a0, nsimd_sve_vu8 a1);
```

### CPU

```c
nsimd_cpu_vlf64 nsimd_eq_cpu_f64(nsimd_cpu_vf64 a0, nsimd_cpu_vf64 a1);
nsimd_cpu_vlf32 nsimd_eq_cpu_f32(nsimd_cpu_vf32 a0, nsimd_cpu_vf32 a1);
nsimd_cpu_vlf16 nsimd_eq_cpu_f16(nsimd_cpu_vf16 a0, nsimd_cpu_vf16 a1);
nsimd_cpu_vli64 nsimd_eq_cpu_i64(nsimd_cpu_vi64 a0, nsimd_cpu_vi64 a1);
nsimd_cpu_vli32 nsimd_eq_cpu_i32(nsimd_cpu_vi32 a0, nsimd_cpu_vi32 a1);
nsimd_cpu_vli16 nsimd_eq_cpu_i16(nsimd_cpu_vi16 a0, nsimd_cpu_vi16 a1);
nsimd_cpu_vli8 nsimd_eq_cpu_i8(nsimd_cpu_vi8 a0, nsimd_cpu_vi8 a1);
nsimd_cpu_vlu64 nsimd_eq_cpu_u64(nsimd_cpu_vu64 a0, nsimd_cpu_vu64 a1);
nsimd_cpu_vlu32 nsimd_eq_cpu_u32(nsimd_cpu_vu32 a0, nsimd_cpu_vu32 a1);
nsimd_cpu_vlu16 nsimd_eq_cpu_u16(nsimd_cpu_vu16 a0, nsimd_cpu_vu16 a1);
nsimd_cpu_vlu8 nsimd_eq_cpu_u8(nsimd_cpu_vu8 a0, nsimd_cpu_vu8 a1);
```

### SSE2

```c
nsimd_sse2_vlf64 nsimd_eq_sse2_f64(nsimd_sse2_vf64 a0, nsimd_sse2_vf64 a1);
nsimd_sse2_vlf32 nsimd_eq_sse2_f32(nsimd_sse2_vf32 a0, nsimd_sse2_vf32 a1);
nsimd_sse2_vlf16 nsimd_eq_sse2_f16(nsimd_sse2_vf16 a0, nsimd_sse2_vf16 a1);
nsimd_sse2_vli64 nsimd_eq_sse2_i64(nsimd_sse2_vi64 a0, nsimd_sse2_vi64 a1);
nsimd_sse2_vli32 nsimd_eq_sse2_i32(nsimd_sse2_vi32 a0, nsimd_sse2_vi32 a1);
nsimd_sse2_vli16 nsimd_eq_sse2_i16(nsimd_sse2_vi16 a0, nsimd_sse2_vi16 a1);
nsimd_sse2_vli8 nsimd_eq_sse2_i8(nsimd_sse2_vi8 a0, nsimd_sse2_vi8 a1);
nsimd_sse2_vlu64 nsimd_eq_sse2_u64(nsimd_sse2_vu64 a0, nsimd_sse2_vu64 a1);
nsimd_sse2_vlu32 nsimd_eq_sse2_u32(nsimd_sse2_vu32 a0, nsimd_sse2_vu32 a1);
nsimd_sse2_vlu16 nsimd_eq_sse2_u16(nsimd_sse2_vu16 a0, nsimd_sse2_vu16 a1);
nsimd_sse2_vlu8 nsimd_eq_sse2_u8(nsimd_sse2_vu8 a0, nsimd_sse2_vu8 a1);
```

### AARCH64

```c
nsimd_aarch64_vlf64 nsimd_eq_aarch64_f64(nsimd_aarch64_vf64 a0, nsimd_aarch64_vf64 a1);
nsimd_aarch64_vlf32 nsimd_eq_aarch64_f32(nsimd_aarch64_vf32 a0, nsimd_aarch64_vf32 a1);
nsimd_aarch64_vlf16 nsimd_eq_aarch64_f16(nsimd_aarch64_vf16 a0, nsimd_aarch64_vf16 a1);
nsimd_aarch64_vli64 nsimd_eq_aarch64_i64(nsimd_aarch64_vi64 a0, nsimd_aarch64_vi64 a1);
nsimd_aarch64_vli32 nsimd_eq_aarch64_i32(nsimd_aarch64_vi32 a0, nsimd_aarch64_vi32 a1);
nsimd_aarch64_vli16 nsimd_eq_aarch64_i16(nsimd_aarch64_vi16 a0, nsimd_aarch64_vi16 a1);
nsimd_aarch64_vli8 nsimd_eq_aarch64_i8(nsimd_aarch64_vi8 a0, nsimd_aarch64_vi8 a1);
nsimd_aarch64_vlu64 nsimd_eq_aarch64_u64(nsimd_aarch64_vu64 a0, nsimd_aarch64_vu64 a1);
nsimd_aarch64_vlu32 nsimd_eq_aarch64_u32(nsimd_aarch64_vu32 a0, nsimd_aarch64_vu32 a1);
nsimd_aarch64_vlu16 nsimd_eq_aarch64_u16(nsimd_aarch64_vu16 a0, nsimd_aarch64_vu16 a1);
nsimd_aarch64_vlu8 nsimd_eq_aarch64_u8(nsimd_aarch64_vu8 a0, nsimd_aarch64_vu8 a1);
```

### SSE42

```c
nsimd_sse42_vlf64 nsimd_eq_sse42_f64(nsimd_sse42_vf64 a0, nsimd_sse42_vf64 a1);
nsimd_sse42_vlf32 nsimd_eq_sse42_f32(nsimd_sse42_vf32 a0, nsimd_sse42_vf32 a1);
nsimd_sse42_vlf16 nsimd_eq_sse42_f16(nsimd_sse42_vf16 a0, nsimd_sse42_vf16 a1);
nsimd_sse42_vli64 nsimd_eq_sse42_i64(nsimd_sse42_vi64 a0, nsimd_sse42_vi64 a1);
nsimd_sse42_vli32 nsimd_eq_sse42_i32(nsimd_sse42_vi32 a0, nsimd_sse42_vi32 a1);
nsimd_sse42_vli16 nsimd_eq_sse42_i16(nsimd_sse42_vi16 a0, nsimd_sse42_vi16 a1);
nsimd_sse42_vli8 nsimd_eq_sse42_i8(nsimd_sse42_vi8 a0, nsimd_sse42_vi8 a1);
nsimd_sse42_vlu64 nsimd_eq_sse42_u64(nsimd_sse42_vu64 a0, nsimd_sse42_vu64 a1);
nsimd_sse42_vlu32 nsimd_eq_sse42_u32(nsimd_sse42_vu32 a0, nsimd_sse42_vu32 a1);
nsimd_sse42_vlu16 nsimd_eq_sse42_u16(nsimd_sse42_vu16 a0, nsimd_sse42_vu16 a1);
nsimd_sse42_vlu8 nsimd_eq_sse42_u8(nsimd_sse42_vu8 a0, nsimd_sse42_vu8 a1);
```

## C++ base API (architecture specifics)

### NEON128

```c
nsimd_neon128_vlf64 eq(nsimd_neon128_vf64 a0, nsimd_neon128_vf64 a1, f64, neon128);
nsimd_neon128_vlf32 eq(nsimd_neon128_vf32 a0, nsimd_neon128_vf32 a1, f32, neon128);
nsimd_neon128_vlf16 eq(nsimd_neon128_vf16 a0, nsimd_neon128_vf16 a1, f16, neon128);
nsimd_neon128_vli64 eq(nsimd_neon128_vi64 a0, nsimd_neon128_vi64 a1, i64, neon128);
nsimd_neon128_vli32 eq(nsimd_neon128_vi32 a0, nsimd_neon128_vi32 a1, i32, neon128);
nsimd_neon128_vli16 eq(nsimd_neon128_vi16 a0, nsimd_neon128_vi16 a1, i16, neon128);
nsimd_neon128_vli8 eq(nsimd_neon128_vi8 a0, nsimd_neon128_vi8 a1, i8, neon128);
nsimd_neon128_vlu64 eq(nsimd_neon128_vu64 a0, nsimd_neon128_vu64 a1, u64, neon128);
nsimd_neon128_vlu32 eq(nsimd_neon128_vu32 a0, nsimd_neon128_vu32 a1, u32, neon128);
nsimd_neon128_vlu16 eq(nsimd_neon128_vu16 a0, nsimd_neon128_vu16 a1, u16, neon128);
nsimd_neon128_vlu8 eq(nsimd_neon128_vu8 a0, nsimd_neon128_vu8 a1, u8, neon128);
```

### AVX2

```c
nsimd_avx2_vlf64 eq(nsimd_avx2_vf64 a0, nsimd_avx2_vf64 a1, f64, avx2);
nsimd_avx2_vlf32 eq(nsimd_avx2_vf32 a0, nsimd_avx2_vf32 a1, f32, avx2);
nsimd_avx2_vlf16 eq(nsimd_avx2_vf16 a0, nsimd_avx2_vf16 a1, f16, avx2);
nsimd_avx2_vli64 eq(nsimd_avx2_vi64 a0, nsimd_avx2_vi64 a1, i64, avx2);
nsimd_avx2_vli32 eq(nsimd_avx2_vi32 a0, nsimd_avx2_vi32 a1, i32, avx2);
nsimd_avx2_vli16 eq(nsimd_avx2_vi16 a0, nsimd_avx2_vi16 a1, i16, avx2);
nsimd_avx2_vli8 eq(nsimd_avx2_vi8 a0, nsimd_avx2_vi8 a1, i8, avx2);
nsimd_avx2_vlu64 eq(nsimd_avx2_vu64 a0, nsimd_avx2_vu64 a1, u64, avx2);
nsimd_avx2_vlu32 eq(nsimd_avx2_vu32 a0, nsimd_avx2_vu32 a1, u32, avx2);
nsimd_avx2_vlu16 eq(nsimd_avx2_vu16 a0, nsimd_avx2_vu16 a1, u16, avx2);
nsimd_avx2_vlu8 eq(nsimd_avx2_vu8 a0, nsimd_avx2_vu8 a1, u8, avx2);
```

### AVX512_KNL

```c
nsimd_avx512_knl_vlf64 eq(nsimd_avx512_knl_vf64 a0, nsimd_avx512_knl_vf64 a1, f64, avx512_knl);
nsimd_avx512_knl_vlf32 eq(nsimd_avx512_knl_vf32 a0, nsimd_avx512_knl_vf32 a1, f32, avx512_knl);
nsimd_avx512_knl_vlf16 eq(nsimd_avx512_knl_vf16 a0, nsimd_avx512_knl_vf16 a1, f16, avx512_knl);
nsimd_avx512_knl_vli64 eq(nsimd_avx512_knl_vi64 a0, nsimd_avx512_knl_vi64 a1, i64, avx512_knl);
nsimd_avx512_knl_vli32 eq(nsimd_avx512_knl_vi32 a0, nsimd_avx512_knl_vi32 a1, i32, avx512_knl);
nsimd_avx512_knl_vli16 eq(nsimd_avx512_knl_vi16 a0, nsimd_avx512_knl_vi16 a1, i16, avx512_knl);
nsimd_avx512_knl_vli8 eq(nsimd_avx512_knl_vi8 a0, nsimd_avx512_knl_vi8 a1, i8, avx512_knl);
nsimd_avx512_knl_vlu64 eq(nsimd_avx512_knl_vu64 a0, nsimd_avx512_knl_vu64 a1, u64, avx512_knl);
nsimd_avx512_knl_vlu32 eq(nsimd_avx512_knl_vu32 a0, nsimd_avx512_knl_vu32 a1, u32, avx512_knl);
nsimd_avx512_knl_vlu16 eq(nsimd_avx512_knl_vu16 a0, nsimd_avx512_knl_vu16 a1, u16, avx512_knl);
nsimd_avx512_knl_vlu8 eq(nsimd_avx512_knl_vu8 a0, nsimd_avx512_knl_vu8 a1, u8, avx512_knl);
```

### AVX

```c
nsimd_avx_vlf64 eq(nsimd_avx_vf64 a0, nsimd_avx_vf64 a1, f64, avx);
nsimd_avx_vlf32 eq(nsimd_avx_vf32 a0, nsimd_avx_vf32 a1, f32, avx);
nsimd_avx_vlf16 eq(nsimd_avx_vf16 a0, nsimd_avx_vf16 a1, f16, avx);
nsimd_avx_vli64 eq(nsimd_avx_vi64 a0, nsimd_avx_vi64 a1, i64, avx);
nsimd_avx_vli32 eq(nsimd_avx_vi32 a0, nsimd_avx_vi32 a1, i32, avx);
nsimd_avx_vli16 eq(nsimd_avx_vi16 a0, nsimd_avx_vi16 a1, i16, avx);
nsimd_avx_vli8 eq(nsimd_avx_vi8 a0, nsimd_avx_vi8 a1, i8, avx);
nsimd_avx_vlu64 eq(nsimd_avx_vu64 a0, nsimd_avx_vu64 a1, u64, avx);
nsimd_avx_vlu32 eq(nsimd_avx_vu32 a0, nsimd_avx_vu32 a1, u32, avx);
nsimd_avx_vlu16 eq(nsimd_avx_vu16 a0, nsimd_avx_vu16 a1, u16, avx);
nsimd_avx_vlu8 eq(nsimd_avx_vu8 a0, nsimd_avx_vu8 a1, u8, avx);
```

### AVX512_SKYLAKE

```c
nsimd_avx512_skylake_vlf64 eq(nsimd_avx512_skylake_vf64 a0, nsimd_avx512_skylake_vf64 a1, f64, avx512_skylake);
nsimd_avx512_skylake_vlf32 eq(nsimd_avx512_skylake_vf32 a0, nsimd_avx512_skylake_vf32 a1, f32, avx512_skylake);
nsimd_avx512_skylake_vlf16 eq(nsimd_avx512_skylake_vf16 a0, nsimd_avx512_skylake_vf16 a1, f16, avx512_skylake);
nsimd_avx512_skylake_vli64 eq(nsimd_avx512_skylake_vi64 a0, nsimd_avx512_skylake_vi64 a1, i64, avx512_skylake);
nsimd_avx512_skylake_vli32 eq(nsimd_avx512_skylake_vi32 a0, nsimd_avx512_skylake_vi32 a1, i32, avx512_skylake);
nsimd_avx512_skylake_vli16 eq(nsimd_avx512_skylake_vi16 a0, nsimd_avx512_skylake_vi16 a1, i16, avx512_skylake);
nsimd_avx512_skylake_vli8 eq(nsimd_avx512_skylake_vi8 a0, nsimd_avx512_skylake_vi8 a1, i8, avx512_skylake);
nsimd_avx512_skylake_vlu64 eq(nsimd_avx512_skylake_vu64 a0, nsimd_avx512_skylake_vu64 a1, u64, avx512_skylake);
nsimd_avx512_skylake_vlu32 eq(nsimd_avx512_skylake_vu32 a0, nsimd_avx512_skylake_vu32 a1, u32, avx512_skylake);
nsimd_avx512_skylake_vlu16 eq(nsimd_avx512_skylake_vu16 a0, nsimd_avx512_skylake_vu16 a1, u16, avx512_skylake);
nsimd_avx512_skylake_vlu8 eq(nsimd_avx512_skylake_vu8 a0, nsimd_avx512_skylake_vu8 a1, u8, avx512_skylake);
```

### SVE

```c
nsimd_sve_vlf64 eq(nsimd_sve_vf64 a0, nsimd_sve_vf64 a1, f64, sve);
nsimd_sve_vlf32 eq(nsimd_sve_vf32 a0, nsimd_sve_vf32 a1, f32, sve);
nsimd_sve_vlf16 eq(nsimd_sve_vf16 a0, nsimd_sve_vf16 a1, f16, sve);
nsimd_sve_vli64 eq(nsimd_sve_vi64 a0, nsimd_sve_vi64 a1, i64, sve);
nsimd_sve_vli32 eq(nsimd_sve_vi32 a0, nsimd_sve_vi32 a1, i32, sve);
nsimd_sve_vli16 eq(nsimd_sve_vi16 a0, nsimd_sve_vi16 a1, i16, sve);
nsimd_sve_vli8 eq(nsimd_sve_vi8 a0, nsimd_sve_vi8 a1, i8, sve);
nsimd_sve_vlu64 eq(nsimd_sve_vu64 a0, nsimd_sve_vu64 a1, u64, sve);
nsimd_sve_vlu32 eq(nsimd_sve_vu32 a0, nsimd_sve_vu32 a1, u32, sve);
nsimd_sve_vlu16 eq(nsimd_sve_vu16 a0, nsimd_sve_vu16 a1, u16, sve);
nsimd_sve_vlu8 eq(nsimd_sve_vu8 a0, nsimd_sve_vu8 a1, u8, sve);
```

### CPU

```c
nsimd_cpu_vlf64 eq(nsimd_cpu_vf64 a0, nsimd_cpu_vf64 a1, f64, cpu);
nsimd_cpu_vlf32 eq(nsimd_cpu_vf32 a0, nsimd_cpu_vf32 a1, f32, cpu);
nsimd_cpu_vlf16 eq(nsimd_cpu_vf16 a0, nsimd_cpu_vf16 a1, f16, cpu);
nsimd_cpu_vli64 eq(nsimd_cpu_vi64 a0, nsimd_cpu_vi64 a1, i64, cpu);
nsimd_cpu_vli32 eq(nsimd_cpu_vi32 a0, nsimd_cpu_vi32 a1, i32, cpu);
nsimd_cpu_vli16 eq(nsimd_cpu_vi16 a0, nsimd_cpu_vi16 a1, i16, cpu);
nsimd_cpu_vli8 eq(nsimd_cpu_vi8 a0, nsimd_cpu_vi8 a1, i8, cpu);
nsimd_cpu_vlu64 eq(nsimd_cpu_vu64 a0, nsimd_cpu_vu64 a1, u64, cpu);
nsimd_cpu_vlu32 eq(nsimd_cpu_vu32 a0, nsimd_cpu_vu32 a1, u32, cpu);
nsimd_cpu_vlu16 eq(nsimd_cpu_vu16 a0, nsimd_cpu_vu16 a1, u16, cpu);
nsimd_cpu_vlu8 eq(nsimd_cpu_vu8 a0, nsimd_cpu_vu8 a1, u8, cpu);
```

### SSE2

```c
nsimd_sse2_vlf64 eq(nsimd_sse2_vf64 a0, nsimd_sse2_vf64 a1, f64, sse2);
nsimd_sse2_vlf32 eq(nsimd_sse2_vf32 a0, nsimd_sse2_vf32 a1, f32, sse2);
nsimd_sse2_vlf16 eq(nsimd_sse2_vf16 a0, nsimd_sse2_vf16 a1, f16, sse2);
nsimd_sse2_vli64 eq(nsimd_sse2_vi64 a0, nsimd_sse2_vi64 a1, i64, sse2);
nsimd_sse2_vli32 eq(nsimd_sse2_vi32 a0, nsimd_sse2_vi32 a1, i32, sse2);
nsimd_sse2_vli16 eq(nsimd_sse2_vi16 a0, nsimd_sse2_vi16 a1, i16, sse2);
nsimd_sse2_vli8 eq(nsimd_sse2_vi8 a0, nsimd_sse2_vi8 a1, i8, sse2);
nsimd_sse2_vlu64 eq(nsimd_sse2_vu64 a0, nsimd_sse2_vu64 a1, u64, sse2);
nsimd_sse2_vlu32 eq(nsimd_sse2_vu32 a0, nsimd_sse2_vu32 a1, u32, sse2);
nsimd_sse2_vlu16 eq(nsimd_sse2_vu16 a0, nsimd_sse2_vu16 a1, u16, sse2);
nsimd_sse2_vlu8 eq(nsimd_sse2_vu8 a0, nsimd_sse2_vu8 a1, u8, sse2);
```

### AARCH64

```c
nsimd_aarch64_vlf64 eq(nsimd_aarch64_vf64 a0, nsimd_aarch64_vf64 a1, f64, aarch64);
nsimd_aarch64_vlf32 eq(nsimd_aarch64_vf32 a0, nsimd_aarch64_vf32 a1, f32, aarch64);
nsimd_aarch64_vlf16 eq(nsimd_aarch64_vf16 a0, nsimd_aarch64_vf16 a1, f16, aarch64);
nsimd_aarch64_vli64 eq(nsimd_aarch64_vi64 a0, nsimd_aarch64_vi64 a1, i64, aarch64);
nsimd_aarch64_vli32 eq(nsimd_aarch64_vi32 a0, nsimd_aarch64_vi32 a1, i32, aarch64);
nsimd_aarch64_vli16 eq(nsimd_aarch64_vi16 a0, nsimd_aarch64_vi16 a1, i16, aarch64);
nsimd_aarch64_vli8 eq(nsimd_aarch64_vi8 a0, nsimd_aarch64_vi8 a1, i8, aarch64);
nsimd_aarch64_vlu64 eq(nsimd_aarch64_vu64 a0, nsimd_aarch64_vu64 a1, u64, aarch64);
nsimd_aarch64_vlu32 eq(nsimd_aarch64_vu32 a0, nsimd_aarch64_vu32 a1, u32, aarch64);
nsimd_aarch64_vlu16 eq(nsimd_aarch64_vu16 a0, nsimd_aarch64_vu16 a1, u16, aarch64);
nsimd_aarch64_vlu8 eq(nsimd_aarch64_vu8 a0, nsimd_aarch64_vu8 a1, u8, aarch64);
```

### SSE42

```c
nsimd_sse42_vlf64 eq(nsimd_sse42_vf64 a0, nsimd_sse42_vf64 a1, f64, sse42);
nsimd_sse42_vlf32 eq(nsimd_sse42_vf32 a0, nsimd_sse42_vf32 a1, f32, sse42);
nsimd_sse42_vlf16 eq(nsimd_sse42_vf16 a0, nsimd_sse42_vf16 a1, f16, sse42);
nsimd_sse42_vli64 eq(nsimd_sse42_vi64 a0, nsimd_sse42_vi64 a1, i64, sse42);
nsimd_sse42_vli32 eq(nsimd_sse42_vi32 a0, nsimd_sse42_vi32 a1, i32, sse42);
nsimd_sse42_vli16 eq(nsimd_sse42_vi16 a0, nsimd_sse42_vi16 a1, i16, sse42);
nsimd_sse42_vli8 eq(nsimd_sse42_vi8 a0, nsimd_sse42_vi8 a1, i8, sse42);
nsimd_sse42_vlu64 eq(nsimd_sse42_vu64 a0, nsimd_sse42_vu64 a1, u64, sse42);
nsimd_sse42_vlu32 eq(nsimd_sse42_vu32 a0, nsimd_sse42_vu32 a1, u32, sse42);
nsimd_sse42_vlu16 eq(nsimd_sse42_vu16 a0, nsimd_sse42_vu16 a1, u16, sse42);
nsimd_sse42_vlu8 eq(nsimd_sse42_vu8 a0, nsimd_sse42_vu8 a1, u8, sse42);
```