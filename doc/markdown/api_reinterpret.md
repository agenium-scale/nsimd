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

# Reinterpret vector

## Description

Reinterpret input vector into a different vector type preserving all bits.

## C base API (generic)

```c
#define vreinterpret(a0, from_type, to_type)
#define vreinterpret_e(a0, from_type, to_type, simd_ext)
```

## C++ base API (generic)

```c++
template <typename A0, typename F, typename T> typename simd_traits<T, NSIMD_SIMD>::simd_vector reinterpret(A0 a0, F, T);
```

## C++ advanced API

```c++
template <typename ToPackType, typename T, typename SimdExt> ToPackType reinterpret(ToPackType, pack<T, 1, SimdExt> const& a0);
template <typename ToPackType, typename T, int N, typename SimdExt> ToPackType reinterpret(ToPackType, pack<T, N, SimdExt> const& a0);
template <typename ToPackType, typename T, int N, typename SimdExt> ToPackType reinterpret(pack<T, N, SimdExt> const& a0);
```

## C base API (architecture specifics)

### NEON128

```c
nsimd_neon128_vf64 nsimd_reinterpret_neon128_f64_f64(nsimd_neon128_vf64 a0);
nsimd_neon128_vf32 nsimd_reinterpret_neon128_f32_f32(nsimd_neon128_vf32 a0);
nsimd_neon128_vf16 nsimd_reinterpret_neon128_f16_f16(nsimd_neon128_vf16 a0);
nsimd_neon128_vi64 nsimd_reinterpret_neon128_i64_i64(nsimd_neon128_vi64 a0);
nsimd_neon128_vi32 nsimd_reinterpret_neon128_i32_i32(nsimd_neon128_vi32 a0);
nsimd_neon128_vi16 nsimd_reinterpret_neon128_i16_i16(nsimd_neon128_vi16 a0);
nsimd_neon128_vi8 nsimd_reinterpret_neon128_i8_i8(nsimd_neon128_vi8 a0);
nsimd_neon128_vu64 nsimd_reinterpret_neon128_u64_u64(nsimd_neon128_vu64 a0);
nsimd_neon128_vu32 nsimd_reinterpret_neon128_u32_u32(nsimd_neon128_vu32 a0);
nsimd_neon128_vu16 nsimd_reinterpret_neon128_u16_u16(nsimd_neon128_vu16 a0);
nsimd_neon128_vu8 nsimd_reinterpret_neon128_u8_u8(nsimd_neon128_vu8 a0);
```

### AVX2

```c
nsimd_avx2_vf64 nsimd_reinterpret_avx2_f64_f64(nsimd_avx2_vf64 a0);
nsimd_avx2_vf32 nsimd_reinterpret_avx2_f32_f32(nsimd_avx2_vf32 a0);
nsimd_avx2_vf16 nsimd_reinterpret_avx2_f16_f16(nsimd_avx2_vf16 a0);
nsimd_avx2_vi64 nsimd_reinterpret_avx2_i64_i64(nsimd_avx2_vi64 a0);
nsimd_avx2_vi32 nsimd_reinterpret_avx2_i32_i32(nsimd_avx2_vi32 a0);
nsimd_avx2_vi16 nsimd_reinterpret_avx2_i16_i16(nsimd_avx2_vi16 a0);
nsimd_avx2_vi8 nsimd_reinterpret_avx2_i8_i8(nsimd_avx2_vi8 a0);
nsimd_avx2_vu64 nsimd_reinterpret_avx2_u64_u64(nsimd_avx2_vu64 a0);
nsimd_avx2_vu32 nsimd_reinterpret_avx2_u32_u32(nsimd_avx2_vu32 a0);
nsimd_avx2_vu16 nsimd_reinterpret_avx2_u16_u16(nsimd_avx2_vu16 a0);
nsimd_avx2_vu8 nsimd_reinterpret_avx2_u8_u8(nsimd_avx2_vu8 a0);
```

### AVX512_KNL

```c
nsimd_avx512_knl_vf64 nsimd_reinterpret_avx512_knl_f64_f64(nsimd_avx512_knl_vf64 a0);
nsimd_avx512_knl_vf32 nsimd_reinterpret_avx512_knl_f32_f32(nsimd_avx512_knl_vf32 a0);
nsimd_avx512_knl_vf16 nsimd_reinterpret_avx512_knl_f16_f16(nsimd_avx512_knl_vf16 a0);
nsimd_avx512_knl_vi64 nsimd_reinterpret_avx512_knl_i64_i64(nsimd_avx512_knl_vi64 a0);
nsimd_avx512_knl_vi32 nsimd_reinterpret_avx512_knl_i32_i32(nsimd_avx512_knl_vi32 a0);
nsimd_avx512_knl_vi16 nsimd_reinterpret_avx512_knl_i16_i16(nsimd_avx512_knl_vi16 a0);
nsimd_avx512_knl_vi8 nsimd_reinterpret_avx512_knl_i8_i8(nsimd_avx512_knl_vi8 a0);
nsimd_avx512_knl_vu64 nsimd_reinterpret_avx512_knl_u64_u64(nsimd_avx512_knl_vu64 a0);
nsimd_avx512_knl_vu32 nsimd_reinterpret_avx512_knl_u32_u32(nsimd_avx512_knl_vu32 a0);
nsimd_avx512_knl_vu16 nsimd_reinterpret_avx512_knl_u16_u16(nsimd_avx512_knl_vu16 a0);
nsimd_avx512_knl_vu8 nsimd_reinterpret_avx512_knl_u8_u8(nsimd_avx512_knl_vu8 a0);
```

### AVX

```c
nsimd_avx_vf64 nsimd_reinterpret_avx_f64_f64(nsimd_avx_vf64 a0);
nsimd_avx_vf32 nsimd_reinterpret_avx_f32_f32(nsimd_avx_vf32 a0);
nsimd_avx_vf16 nsimd_reinterpret_avx_f16_f16(nsimd_avx_vf16 a0);
nsimd_avx_vi64 nsimd_reinterpret_avx_i64_i64(nsimd_avx_vi64 a0);
nsimd_avx_vi32 nsimd_reinterpret_avx_i32_i32(nsimd_avx_vi32 a0);
nsimd_avx_vi16 nsimd_reinterpret_avx_i16_i16(nsimd_avx_vi16 a0);
nsimd_avx_vi8 nsimd_reinterpret_avx_i8_i8(nsimd_avx_vi8 a0);
nsimd_avx_vu64 nsimd_reinterpret_avx_u64_u64(nsimd_avx_vu64 a0);
nsimd_avx_vu32 nsimd_reinterpret_avx_u32_u32(nsimd_avx_vu32 a0);
nsimd_avx_vu16 nsimd_reinterpret_avx_u16_u16(nsimd_avx_vu16 a0);
nsimd_avx_vu8 nsimd_reinterpret_avx_u8_u8(nsimd_avx_vu8 a0);
```

### AVX512_SKYLAKE

```c
nsimd_avx512_skylake_vf64 nsimd_reinterpret_avx512_skylake_f64_f64(nsimd_avx512_skylake_vf64 a0);
nsimd_avx512_skylake_vf32 nsimd_reinterpret_avx512_skylake_f32_f32(nsimd_avx512_skylake_vf32 a0);
nsimd_avx512_skylake_vf16 nsimd_reinterpret_avx512_skylake_f16_f16(nsimd_avx512_skylake_vf16 a0);
nsimd_avx512_skylake_vi64 nsimd_reinterpret_avx512_skylake_i64_i64(nsimd_avx512_skylake_vi64 a0);
nsimd_avx512_skylake_vi32 nsimd_reinterpret_avx512_skylake_i32_i32(nsimd_avx512_skylake_vi32 a0);
nsimd_avx512_skylake_vi16 nsimd_reinterpret_avx512_skylake_i16_i16(nsimd_avx512_skylake_vi16 a0);
nsimd_avx512_skylake_vi8 nsimd_reinterpret_avx512_skylake_i8_i8(nsimd_avx512_skylake_vi8 a0);
nsimd_avx512_skylake_vu64 nsimd_reinterpret_avx512_skylake_u64_u64(nsimd_avx512_skylake_vu64 a0);
nsimd_avx512_skylake_vu32 nsimd_reinterpret_avx512_skylake_u32_u32(nsimd_avx512_skylake_vu32 a0);
nsimd_avx512_skylake_vu16 nsimd_reinterpret_avx512_skylake_u16_u16(nsimd_avx512_skylake_vu16 a0);
nsimd_avx512_skylake_vu8 nsimd_reinterpret_avx512_skylake_u8_u8(nsimd_avx512_skylake_vu8 a0);
```

### SVE

```c
nsimd_sve_vf64 nsimd_reinterpret_sve_f64_f64(nsimd_sve_vf64 a0);
nsimd_sve_vf32 nsimd_reinterpret_sve_f32_f32(nsimd_sve_vf32 a0);
nsimd_sve_vf16 nsimd_reinterpret_sve_f16_f16(nsimd_sve_vf16 a0);
nsimd_sve_vi64 nsimd_reinterpret_sve_i64_i64(nsimd_sve_vi64 a0);
nsimd_sve_vi32 nsimd_reinterpret_sve_i32_i32(nsimd_sve_vi32 a0);
nsimd_sve_vi16 nsimd_reinterpret_sve_i16_i16(nsimd_sve_vi16 a0);
nsimd_sve_vi8 nsimd_reinterpret_sve_i8_i8(nsimd_sve_vi8 a0);
nsimd_sve_vu64 nsimd_reinterpret_sve_u64_u64(nsimd_sve_vu64 a0);
nsimd_sve_vu32 nsimd_reinterpret_sve_u32_u32(nsimd_sve_vu32 a0);
nsimd_sve_vu16 nsimd_reinterpret_sve_u16_u16(nsimd_sve_vu16 a0);
nsimd_sve_vu8 nsimd_reinterpret_sve_u8_u8(nsimd_sve_vu8 a0);
```

### CPU

```c
nsimd_cpu_vf64 nsimd_reinterpret_cpu_f64_f64(nsimd_cpu_vf64 a0);
nsimd_cpu_vf32 nsimd_reinterpret_cpu_f32_f32(nsimd_cpu_vf32 a0);
nsimd_cpu_vf16 nsimd_reinterpret_cpu_f16_f16(nsimd_cpu_vf16 a0);
nsimd_cpu_vi64 nsimd_reinterpret_cpu_i64_i64(nsimd_cpu_vi64 a0);
nsimd_cpu_vi32 nsimd_reinterpret_cpu_i32_i32(nsimd_cpu_vi32 a0);
nsimd_cpu_vi16 nsimd_reinterpret_cpu_i16_i16(nsimd_cpu_vi16 a0);
nsimd_cpu_vi8 nsimd_reinterpret_cpu_i8_i8(nsimd_cpu_vi8 a0);
nsimd_cpu_vu64 nsimd_reinterpret_cpu_u64_u64(nsimd_cpu_vu64 a0);
nsimd_cpu_vu32 nsimd_reinterpret_cpu_u32_u32(nsimd_cpu_vu32 a0);
nsimd_cpu_vu16 nsimd_reinterpret_cpu_u16_u16(nsimd_cpu_vu16 a0);
nsimd_cpu_vu8 nsimd_reinterpret_cpu_u8_u8(nsimd_cpu_vu8 a0);
```

### SSE2

```c
nsimd_sse2_vf64 nsimd_reinterpret_sse2_f64_f64(nsimd_sse2_vf64 a0);
nsimd_sse2_vf32 nsimd_reinterpret_sse2_f32_f32(nsimd_sse2_vf32 a0);
nsimd_sse2_vf16 nsimd_reinterpret_sse2_f16_f16(nsimd_sse2_vf16 a0);
nsimd_sse2_vi64 nsimd_reinterpret_sse2_i64_i64(nsimd_sse2_vi64 a0);
nsimd_sse2_vi32 nsimd_reinterpret_sse2_i32_i32(nsimd_sse2_vi32 a0);
nsimd_sse2_vi16 nsimd_reinterpret_sse2_i16_i16(nsimd_sse2_vi16 a0);
nsimd_sse2_vi8 nsimd_reinterpret_sse2_i8_i8(nsimd_sse2_vi8 a0);
nsimd_sse2_vu64 nsimd_reinterpret_sse2_u64_u64(nsimd_sse2_vu64 a0);
nsimd_sse2_vu32 nsimd_reinterpret_sse2_u32_u32(nsimd_sse2_vu32 a0);
nsimd_sse2_vu16 nsimd_reinterpret_sse2_u16_u16(nsimd_sse2_vu16 a0);
nsimd_sse2_vu8 nsimd_reinterpret_sse2_u8_u8(nsimd_sse2_vu8 a0);
```

### AARCH64

```c
nsimd_aarch64_vf64 nsimd_reinterpret_aarch64_f64_f64(nsimd_aarch64_vf64 a0);
nsimd_aarch64_vf32 nsimd_reinterpret_aarch64_f32_f32(nsimd_aarch64_vf32 a0);
nsimd_aarch64_vf16 nsimd_reinterpret_aarch64_f16_f16(nsimd_aarch64_vf16 a0);
nsimd_aarch64_vi64 nsimd_reinterpret_aarch64_i64_i64(nsimd_aarch64_vi64 a0);
nsimd_aarch64_vi32 nsimd_reinterpret_aarch64_i32_i32(nsimd_aarch64_vi32 a0);
nsimd_aarch64_vi16 nsimd_reinterpret_aarch64_i16_i16(nsimd_aarch64_vi16 a0);
nsimd_aarch64_vi8 nsimd_reinterpret_aarch64_i8_i8(nsimd_aarch64_vi8 a0);
nsimd_aarch64_vu64 nsimd_reinterpret_aarch64_u64_u64(nsimd_aarch64_vu64 a0);
nsimd_aarch64_vu32 nsimd_reinterpret_aarch64_u32_u32(nsimd_aarch64_vu32 a0);
nsimd_aarch64_vu16 nsimd_reinterpret_aarch64_u16_u16(nsimd_aarch64_vu16 a0);
nsimd_aarch64_vu8 nsimd_reinterpret_aarch64_u8_u8(nsimd_aarch64_vu8 a0);
```

### SSE42

```c
nsimd_sse42_vf64 nsimd_reinterpret_sse42_f64_f64(nsimd_sse42_vf64 a0);
nsimd_sse42_vf32 nsimd_reinterpret_sse42_f32_f32(nsimd_sse42_vf32 a0);
nsimd_sse42_vf16 nsimd_reinterpret_sse42_f16_f16(nsimd_sse42_vf16 a0);
nsimd_sse42_vi64 nsimd_reinterpret_sse42_i64_i64(nsimd_sse42_vi64 a0);
nsimd_sse42_vi32 nsimd_reinterpret_sse42_i32_i32(nsimd_sse42_vi32 a0);
nsimd_sse42_vi16 nsimd_reinterpret_sse42_i16_i16(nsimd_sse42_vi16 a0);
nsimd_sse42_vi8 nsimd_reinterpret_sse42_i8_i8(nsimd_sse42_vi8 a0);
nsimd_sse42_vu64 nsimd_reinterpret_sse42_u64_u64(nsimd_sse42_vu64 a0);
nsimd_sse42_vu32 nsimd_reinterpret_sse42_u32_u32(nsimd_sse42_vu32 a0);
nsimd_sse42_vu16 nsimd_reinterpret_sse42_u16_u16(nsimd_sse42_vu16 a0);
nsimd_sse42_vu8 nsimd_reinterpret_sse42_u8_u8(nsimd_sse42_vu8 a0);
```

## C++ base API (architecture specifics)

### NEON128

```c
nsimd_neon128_vf64 reinterpret(nsimd_neon128_vf64 a0, f64, f64, neon128);
nsimd_neon128_vf32 reinterpret(nsimd_neon128_vf32 a0, f32, f32, neon128);
nsimd_neon128_vf16 reinterpret(nsimd_neon128_vf16 a0, f16, f16, neon128);
nsimd_neon128_vi64 reinterpret(nsimd_neon128_vi64 a0, i64, i64, neon128);
nsimd_neon128_vi32 reinterpret(nsimd_neon128_vi32 a0, i32, i32, neon128);
nsimd_neon128_vi16 reinterpret(nsimd_neon128_vi16 a0, i16, i16, neon128);
nsimd_neon128_vi8 reinterpret(nsimd_neon128_vi8 a0, i8, i8, neon128);
nsimd_neon128_vu64 reinterpret(nsimd_neon128_vu64 a0, u64, u64, neon128);
nsimd_neon128_vu32 reinterpret(nsimd_neon128_vu32 a0, u32, u32, neon128);
nsimd_neon128_vu16 reinterpret(nsimd_neon128_vu16 a0, u16, u16, neon128);
nsimd_neon128_vu8 reinterpret(nsimd_neon128_vu8 a0, u8, u8, neon128);
```

### AVX2

```c
nsimd_avx2_vf64 reinterpret(nsimd_avx2_vf64 a0, f64, f64, avx2);
nsimd_avx2_vf32 reinterpret(nsimd_avx2_vf32 a0, f32, f32, avx2);
nsimd_avx2_vf16 reinterpret(nsimd_avx2_vf16 a0, f16, f16, avx2);
nsimd_avx2_vi64 reinterpret(nsimd_avx2_vi64 a0, i64, i64, avx2);
nsimd_avx2_vi32 reinterpret(nsimd_avx2_vi32 a0, i32, i32, avx2);
nsimd_avx2_vi16 reinterpret(nsimd_avx2_vi16 a0, i16, i16, avx2);
nsimd_avx2_vi8 reinterpret(nsimd_avx2_vi8 a0, i8, i8, avx2);
nsimd_avx2_vu64 reinterpret(nsimd_avx2_vu64 a0, u64, u64, avx2);
nsimd_avx2_vu32 reinterpret(nsimd_avx2_vu32 a0, u32, u32, avx2);
nsimd_avx2_vu16 reinterpret(nsimd_avx2_vu16 a0, u16, u16, avx2);
nsimd_avx2_vu8 reinterpret(nsimd_avx2_vu8 a0, u8, u8, avx2);
```

### AVX512_KNL

```c
nsimd_avx512_knl_vf64 reinterpret(nsimd_avx512_knl_vf64 a0, f64, f64, avx512_knl);
nsimd_avx512_knl_vf32 reinterpret(nsimd_avx512_knl_vf32 a0, f32, f32, avx512_knl);
nsimd_avx512_knl_vf16 reinterpret(nsimd_avx512_knl_vf16 a0, f16, f16, avx512_knl);
nsimd_avx512_knl_vi64 reinterpret(nsimd_avx512_knl_vi64 a0, i64, i64, avx512_knl);
nsimd_avx512_knl_vi32 reinterpret(nsimd_avx512_knl_vi32 a0, i32, i32, avx512_knl);
nsimd_avx512_knl_vi16 reinterpret(nsimd_avx512_knl_vi16 a0, i16, i16, avx512_knl);
nsimd_avx512_knl_vi8 reinterpret(nsimd_avx512_knl_vi8 a0, i8, i8, avx512_knl);
nsimd_avx512_knl_vu64 reinterpret(nsimd_avx512_knl_vu64 a0, u64, u64, avx512_knl);
nsimd_avx512_knl_vu32 reinterpret(nsimd_avx512_knl_vu32 a0, u32, u32, avx512_knl);
nsimd_avx512_knl_vu16 reinterpret(nsimd_avx512_knl_vu16 a0, u16, u16, avx512_knl);
nsimd_avx512_knl_vu8 reinterpret(nsimd_avx512_knl_vu8 a0, u8, u8, avx512_knl);
```

### AVX

```c
nsimd_avx_vf64 reinterpret(nsimd_avx_vf64 a0, f64, f64, avx);
nsimd_avx_vf32 reinterpret(nsimd_avx_vf32 a0, f32, f32, avx);
nsimd_avx_vf16 reinterpret(nsimd_avx_vf16 a0, f16, f16, avx);
nsimd_avx_vi64 reinterpret(nsimd_avx_vi64 a0, i64, i64, avx);
nsimd_avx_vi32 reinterpret(nsimd_avx_vi32 a0, i32, i32, avx);
nsimd_avx_vi16 reinterpret(nsimd_avx_vi16 a0, i16, i16, avx);
nsimd_avx_vi8 reinterpret(nsimd_avx_vi8 a0, i8, i8, avx);
nsimd_avx_vu64 reinterpret(nsimd_avx_vu64 a0, u64, u64, avx);
nsimd_avx_vu32 reinterpret(nsimd_avx_vu32 a0, u32, u32, avx);
nsimd_avx_vu16 reinterpret(nsimd_avx_vu16 a0, u16, u16, avx);
nsimd_avx_vu8 reinterpret(nsimd_avx_vu8 a0, u8, u8, avx);
```

### AVX512_SKYLAKE

```c
nsimd_avx512_skylake_vf64 reinterpret(nsimd_avx512_skylake_vf64 a0, f64, f64, avx512_skylake);
nsimd_avx512_skylake_vf32 reinterpret(nsimd_avx512_skylake_vf32 a0, f32, f32, avx512_skylake);
nsimd_avx512_skylake_vf16 reinterpret(nsimd_avx512_skylake_vf16 a0, f16, f16, avx512_skylake);
nsimd_avx512_skylake_vi64 reinterpret(nsimd_avx512_skylake_vi64 a0, i64, i64, avx512_skylake);
nsimd_avx512_skylake_vi32 reinterpret(nsimd_avx512_skylake_vi32 a0, i32, i32, avx512_skylake);
nsimd_avx512_skylake_vi16 reinterpret(nsimd_avx512_skylake_vi16 a0, i16, i16, avx512_skylake);
nsimd_avx512_skylake_vi8 reinterpret(nsimd_avx512_skylake_vi8 a0, i8, i8, avx512_skylake);
nsimd_avx512_skylake_vu64 reinterpret(nsimd_avx512_skylake_vu64 a0, u64, u64, avx512_skylake);
nsimd_avx512_skylake_vu32 reinterpret(nsimd_avx512_skylake_vu32 a0, u32, u32, avx512_skylake);
nsimd_avx512_skylake_vu16 reinterpret(nsimd_avx512_skylake_vu16 a0, u16, u16, avx512_skylake);
nsimd_avx512_skylake_vu8 reinterpret(nsimd_avx512_skylake_vu8 a0, u8, u8, avx512_skylake);
```

### SVE

```c
nsimd_sve_vf64 reinterpret(nsimd_sve_vf64 a0, f64, f64, sve);
nsimd_sve_vf32 reinterpret(nsimd_sve_vf32 a0, f32, f32, sve);
nsimd_sve_vf16 reinterpret(nsimd_sve_vf16 a0, f16, f16, sve);
nsimd_sve_vi64 reinterpret(nsimd_sve_vi64 a0, i64, i64, sve);
nsimd_sve_vi32 reinterpret(nsimd_sve_vi32 a0, i32, i32, sve);
nsimd_sve_vi16 reinterpret(nsimd_sve_vi16 a0, i16, i16, sve);
nsimd_sve_vi8 reinterpret(nsimd_sve_vi8 a0, i8, i8, sve);
nsimd_sve_vu64 reinterpret(nsimd_sve_vu64 a0, u64, u64, sve);
nsimd_sve_vu32 reinterpret(nsimd_sve_vu32 a0, u32, u32, sve);
nsimd_sve_vu16 reinterpret(nsimd_sve_vu16 a0, u16, u16, sve);
nsimd_sve_vu8 reinterpret(nsimd_sve_vu8 a0, u8, u8, sve);
```

### CPU

```c
nsimd_cpu_vf64 reinterpret(nsimd_cpu_vf64 a0, f64, f64, cpu);
nsimd_cpu_vf32 reinterpret(nsimd_cpu_vf32 a0, f32, f32, cpu);
nsimd_cpu_vf16 reinterpret(nsimd_cpu_vf16 a0, f16, f16, cpu);
nsimd_cpu_vi64 reinterpret(nsimd_cpu_vi64 a0, i64, i64, cpu);
nsimd_cpu_vi32 reinterpret(nsimd_cpu_vi32 a0, i32, i32, cpu);
nsimd_cpu_vi16 reinterpret(nsimd_cpu_vi16 a0, i16, i16, cpu);
nsimd_cpu_vi8 reinterpret(nsimd_cpu_vi8 a0, i8, i8, cpu);
nsimd_cpu_vu64 reinterpret(nsimd_cpu_vu64 a0, u64, u64, cpu);
nsimd_cpu_vu32 reinterpret(nsimd_cpu_vu32 a0, u32, u32, cpu);
nsimd_cpu_vu16 reinterpret(nsimd_cpu_vu16 a0, u16, u16, cpu);
nsimd_cpu_vu8 reinterpret(nsimd_cpu_vu8 a0, u8, u8, cpu);
```

### SSE2

```c
nsimd_sse2_vf64 reinterpret(nsimd_sse2_vf64 a0, f64, f64, sse2);
nsimd_sse2_vf32 reinterpret(nsimd_sse2_vf32 a0, f32, f32, sse2);
nsimd_sse2_vf16 reinterpret(nsimd_sse2_vf16 a0, f16, f16, sse2);
nsimd_sse2_vi64 reinterpret(nsimd_sse2_vi64 a0, i64, i64, sse2);
nsimd_sse2_vi32 reinterpret(nsimd_sse2_vi32 a0, i32, i32, sse2);
nsimd_sse2_vi16 reinterpret(nsimd_sse2_vi16 a0, i16, i16, sse2);
nsimd_sse2_vi8 reinterpret(nsimd_sse2_vi8 a0, i8, i8, sse2);
nsimd_sse2_vu64 reinterpret(nsimd_sse2_vu64 a0, u64, u64, sse2);
nsimd_sse2_vu32 reinterpret(nsimd_sse2_vu32 a0, u32, u32, sse2);
nsimd_sse2_vu16 reinterpret(nsimd_sse2_vu16 a0, u16, u16, sse2);
nsimd_sse2_vu8 reinterpret(nsimd_sse2_vu8 a0, u8, u8, sse2);
```

### AARCH64

```c
nsimd_aarch64_vf64 reinterpret(nsimd_aarch64_vf64 a0, f64, f64, aarch64);
nsimd_aarch64_vf32 reinterpret(nsimd_aarch64_vf32 a0, f32, f32, aarch64);
nsimd_aarch64_vf16 reinterpret(nsimd_aarch64_vf16 a0, f16, f16, aarch64);
nsimd_aarch64_vi64 reinterpret(nsimd_aarch64_vi64 a0, i64, i64, aarch64);
nsimd_aarch64_vi32 reinterpret(nsimd_aarch64_vi32 a0, i32, i32, aarch64);
nsimd_aarch64_vi16 reinterpret(nsimd_aarch64_vi16 a0, i16, i16, aarch64);
nsimd_aarch64_vi8 reinterpret(nsimd_aarch64_vi8 a0, i8, i8, aarch64);
nsimd_aarch64_vu64 reinterpret(nsimd_aarch64_vu64 a0, u64, u64, aarch64);
nsimd_aarch64_vu32 reinterpret(nsimd_aarch64_vu32 a0, u32, u32, aarch64);
nsimd_aarch64_vu16 reinterpret(nsimd_aarch64_vu16 a0, u16, u16, aarch64);
nsimd_aarch64_vu8 reinterpret(nsimd_aarch64_vu8 a0, u8, u8, aarch64);
```

### SSE42

```c
nsimd_sse42_vf64 reinterpret(nsimd_sse42_vf64 a0, f64, f64, sse42);
nsimd_sse42_vf32 reinterpret(nsimd_sse42_vf32 a0, f32, f32, sse42);
nsimd_sse42_vf16 reinterpret(nsimd_sse42_vf16 a0, f16, f16, sse42);
nsimd_sse42_vi64 reinterpret(nsimd_sse42_vi64 a0, i64, i64, sse42);
nsimd_sse42_vi32 reinterpret(nsimd_sse42_vi32 a0, i32, i32, sse42);
nsimd_sse42_vi16 reinterpret(nsimd_sse42_vi16 a0, i16, i16, sse42);
nsimd_sse42_vi8 reinterpret(nsimd_sse42_vi8 a0, i8, i8, sse42);
nsimd_sse42_vu64 reinterpret(nsimd_sse42_vu64 a0, u64, u64, sse42);
nsimd_sse42_vu32 reinterpret(nsimd_sse42_vu32 a0, u32, u32, sse42);
nsimd_sse42_vu16 reinterpret(nsimd_sse42_vu16 a0, u16, u16, sse42);
nsimd_sse42_vu8 reinterpret(nsimd_sse42_vu8 a0, u8, u8, sse42);
```