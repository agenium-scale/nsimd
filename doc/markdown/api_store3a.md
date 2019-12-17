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

# Store into array of structures

## Description

Store 3 SIMD vectors as array of structures of 3 members into aligned memory.

## C base API (generic)

```c
#define vstore3a(a0, a1, a2, a3, type)
#define vstore3a_e(a0, a1, a2, a3, type, simd_ext)
```

## C++ base API (generic)

```c++
template <typename A0, typename A1, typename A2, typename A3, typename T> void store3a(A0 a0, A1 a1, A2 a2, A3 a3, T);
```

## C++ advanced API

```c++
template <typename T, typename SimdExt, typename A0> void store3a(A0 a0, pack<T, 1, SimdExt> const& a1, pack<T, 1, SimdExt> const& a2, pack<T, 1, SimdExt> const& a3);
template <typename T, int N, typename SimdExt, typename A0> void store3a(A0 a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2, pack<T, N, SimdExt> const& a3);
```

## C base API (architecture specifics)

### NEON128

```c
void nsimd_store3a_neon128_f64(f64* a0, nsimd_neon128_vf64 a1, nsimd_neon128_vf64 a2, nsimd_neon128_vf64 a3);
void nsimd_store3a_neon128_f32(f32* a0, nsimd_neon128_vf32 a1, nsimd_neon128_vf32 a2, nsimd_neon128_vf32 a3);
void nsimd_store3a_neon128_f16(f16* a0, nsimd_neon128_vf16 a1, nsimd_neon128_vf16 a2, nsimd_neon128_vf16 a3);
void nsimd_store3a_neon128_i64(i64* a0, nsimd_neon128_vi64 a1, nsimd_neon128_vi64 a2, nsimd_neon128_vi64 a3);
void nsimd_store3a_neon128_i32(i32* a0, nsimd_neon128_vi32 a1, nsimd_neon128_vi32 a2, nsimd_neon128_vi32 a3);
void nsimd_store3a_neon128_i16(i16* a0, nsimd_neon128_vi16 a1, nsimd_neon128_vi16 a2, nsimd_neon128_vi16 a3);
void nsimd_store3a_neon128_i8(i8* a0, nsimd_neon128_vi8 a1, nsimd_neon128_vi8 a2, nsimd_neon128_vi8 a3);
void nsimd_store3a_neon128_u64(u64* a0, nsimd_neon128_vu64 a1, nsimd_neon128_vu64 a2, nsimd_neon128_vu64 a3);
void nsimd_store3a_neon128_u32(u32* a0, nsimd_neon128_vu32 a1, nsimd_neon128_vu32 a2, nsimd_neon128_vu32 a3);
void nsimd_store3a_neon128_u16(u16* a0, nsimd_neon128_vu16 a1, nsimd_neon128_vu16 a2, nsimd_neon128_vu16 a3);
void nsimd_store3a_neon128_u8(u8* a0, nsimd_neon128_vu8 a1, nsimd_neon128_vu8 a2, nsimd_neon128_vu8 a3);
```

### AVX2

```c
void nsimd_store3a_avx2_f64(f64* a0, nsimd_avx2_vf64 a1, nsimd_avx2_vf64 a2, nsimd_avx2_vf64 a3);
void nsimd_store3a_avx2_f32(f32* a0, nsimd_avx2_vf32 a1, nsimd_avx2_vf32 a2, nsimd_avx2_vf32 a3);
void nsimd_store3a_avx2_f16(f16* a0, nsimd_avx2_vf16 a1, nsimd_avx2_vf16 a2, nsimd_avx2_vf16 a3);
void nsimd_store3a_avx2_i64(i64* a0, nsimd_avx2_vi64 a1, nsimd_avx2_vi64 a2, nsimd_avx2_vi64 a3);
void nsimd_store3a_avx2_i32(i32* a0, nsimd_avx2_vi32 a1, nsimd_avx2_vi32 a2, nsimd_avx2_vi32 a3);
void nsimd_store3a_avx2_i16(i16* a0, nsimd_avx2_vi16 a1, nsimd_avx2_vi16 a2, nsimd_avx2_vi16 a3);
void nsimd_store3a_avx2_i8(i8* a0, nsimd_avx2_vi8 a1, nsimd_avx2_vi8 a2, nsimd_avx2_vi8 a3);
void nsimd_store3a_avx2_u64(u64* a0, nsimd_avx2_vu64 a1, nsimd_avx2_vu64 a2, nsimd_avx2_vu64 a3);
void nsimd_store3a_avx2_u32(u32* a0, nsimd_avx2_vu32 a1, nsimd_avx2_vu32 a2, nsimd_avx2_vu32 a3);
void nsimd_store3a_avx2_u16(u16* a0, nsimd_avx2_vu16 a1, nsimd_avx2_vu16 a2, nsimd_avx2_vu16 a3);
void nsimd_store3a_avx2_u8(u8* a0, nsimd_avx2_vu8 a1, nsimd_avx2_vu8 a2, nsimd_avx2_vu8 a3);
```

### AVX512_KNL

```c
void nsimd_store3a_avx512_knl_f64(f64* a0, nsimd_avx512_knl_vf64 a1, nsimd_avx512_knl_vf64 a2, nsimd_avx512_knl_vf64 a3);
void nsimd_store3a_avx512_knl_f32(f32* a0, nsimd_avx512_knl_vf32 a1, nsimd_avx512_knl_vf32 a2, nsimd_avx512_knl_vf32 a3);
void nsimd_store3a_avx512_knl_f16(f16* a0, nsimd_avx512_knl_vf16 a1, nsimd_avx512_knl_vf16 a2, nsimd_avx512_knl_vf16 a3);
void nsimd_store3a_avx512_knl_i64(i64* a0, nsimd_avx512_knl_vi64 a1, nsimd_avx512_knl_vi64 a2, nsimd_avx512_knl_vi64 a3);
void nsimd_store3a_avx512_knl_i32(i32* a0, nsimd_avx512_knl_vi32 a1, nsimd_avx512_knl_vi32 a2, nsimd_avx512_knl_vi32 a3);
void nsimd_store3a_avx512_knl_i16(i16* a0, nsimd_avx512_knl_vi16 a1, nsimd_avx512_knl_vi16 a2, nsimd_avx512_knl_vi16 a3);
void nsimd_store3a_avx512_knl_i8(i8* a0, nsimd_avx512_knl_vi8 a1, nsimd_avx512_knl_vi8 a2, nsimd_avx512_knl_vi8 a3);
void nsimd_store3a_avx512_knl_u64(u64* a0, nsimd_avx512_knl_vu64 a1, nsimd_avx512_knl_vu64 a2, nsimd_avx512_knl_vu64 a3);
void nsimd_store3a_avx512_knl_u32(u32* a0, nsimd_avx512_knl_vu32 a1, nsimd_avx512_knl_vu32 a2, nsimd_avx512_knl_vu32 a3);
void nsimd_store3a_avx512_knl_u16(u16* a0, nsimd_avx512_knl_vu16 a1, nsimd_avx512_knl_vu16 a2, nsimd_avx512_knl_vu16 a3);
void nsimd_store3a_avx512_knl_u8(u8* a0, nsimd_avx512_knl_vu8 a1, nsimd_avx512_knl_vu8 a2, nsimd_avx512_knl_vu8 a3);
```

### AVX

```c
void nsimd_store3a_avx_f64(f64* a0, nsimd_avx_vf64 a1, nsimd_avx_vf64 a2, nsimd_avx_vf64 a3);
void nsimd_store3a_avx_f32(f32* a0, nsimd_avx_vf32 a1, nsimd_avx_vf32 a2, nsimd_avx_vf32 a3);
void nsimd_store3a_avx_f16(f16* a0, nsimd_avx_vf16 a1, nsimd_avx_vf16 a2, nsimd_avx_vf16 a3);
void nsimd_store3a_avx_i64(i64* a0, nsimd_avx_vi64 a1, nsimd_avx_vi64 a2, nsimd_avx_vi64 a3);
void nsimd_store3a_avx_i32(i32* a0, nsimd_avx_vi32 a1, nsimd_avx_vi32 a2, nsimd_avx_vi32 a3);
void nsimd_store3a_avx_i16(i16* a0, nsimd_avx_vi16 a1, nsimd_avx_vi16 a2, nsimd_avx_vi16 a3);
void nsimd_store3a_avx_i8(i8* a0, nsimd_avx_vi8 a1, nsimd_avx_vi8 a2, nsimd_avx_vi8 a3);
void nsimd_store3a_avx_u64(u64* a0, nsimd_avx_vu64 a1, nsimd_avx_vu64 a2, nsimd_avx_vu64 a3);
void nsimd_store3a_avx_u32(u32* a0, nsimd_avx_vu32 a1, nsimd_avx_vu32 a2, nsimd_avx_vu32 a3);
void nsimd_store3a_avx_u16(u16* a0, nsimd_avx_vu16 a1, nsimd_avx_vu16 a2, nsimd_avx_vu16 a3);
void nsimd_store3a_avx_u8(u8* a0, nsimd_avx_vu8 a1, nsimd_avx_vu8 a2, nsimd_avx_vu8 a3);
```

### AVX512_SKYLAKE

```c
void nsimd_store3a_avx512_skylake_f64(f64* a0, nsimd_avx512_skylake_vf64 a1, nsimd_avx512_skylake_vf64 a2, nsimd_avx512_skylake_vf64 a3);
void nsimd_store3a_avx512_skylake_f32(f32* a0, nsimd_avx512_skylake_vf32 a1, nsimd_avx512_skylake_vf32 a2, nsimd_avx512_skylake_vf32 a3);
void nsimd_store3a_avx512_skylake_f16(f16* a0, nsimd_avx512_skylake_vf16 a1, nsimd_avx512_skylake_vf16 a2, nsimd_avx512_skylake_vf16 a3);
void nsimd_store3a_avx512_skylake_i64(i64* a0, nsimd_avx512_skylake_vi64 a1, nsimd_avx512_skylake_vi64 a2, nsimd_avx512_skylake_vi64 a3);
void nsimd_store3a_avx512_skylake_i32(i32* a0, nsimd_avx512_skylake_vi32 a1, nsimd_avx512_skylake_vi32 a2, nsimd_avx512_skylake_vi32 a3);
void nsimd_store3a_avx512_skylake_i16(i16* a0, nsimd_avx512_skylake_vi16 a1, nsimd_avx512_skylake_vi16 a2, nsimd_avx512_skylake_vi16 a3);
void nsimd_store3a_avx512_skylake_i8(i8* a0, nsimd_avx512_skylake_vi8 a1, nsimd_avx512_skylake_vi8 a2, nsimd_avx512_skylake_vi8 a3);
void nsimd_store3a_avx512_skylake_u64(u64* a0, nsimd_avx512_skylake_vu64 a1, nsimd_avx512_skylake_vu64 a2, nsimd_avx512_skylake_vu64 a3);
void nsimd_store3a_avx512_skylake_u32(u32* a0, nsimd_avx512_skylake_vu32 a1, nsimd_avx512_skylake_vu32 a2, nsimd_avx512_skylake_vu32 a3);
void nsimd_store3a_avx512_skylake_u16(u16* a0, nsimd_avx512_skylake_vu16 a1, nsimd_avx512_skylake_vu16 a2, nsimd_avx512_skylake_vu16 a3);
void nsimd_store3a_avx512_skylake_u8(u8* a0, nsimd_avx512_skylake_vu8 a1, nsimd_avx512_skylake_vu8 a2, nsimd_avx512_skylake_vu8 a3);
```

### SVE

```c
void nsimd_store3a_sve_f64(f64* a0, nsimd_sve_vf64 a1, nsimd_sve_vf64 a2, nsimd_sve_vf64 a3);
void nsimd_store3a_sve_f32(f32* a0, nsimd_sve_vf32 a1, nsimd_sve_vf32 a2, nsimd_sve_vf32 a3);
void nsimd_store3a_sve_f16(f16* a0, nsimd_sve_vf16 a1, nsimd_sve_vf16 a2, nsimd_sve_vf16 a3);
void nsimd_store3a_sve_i64(i64* a0, nsimd_sve_vi64 a1, nsimd_sve_vi64 a2, nsimd_sve_vi64 a3);
void nsimd_store3a_sve_i32(i32* a0, nsimd_sve_vi32 a1, nsimd_sve_vi32 a2, nsimd_sve_vi32 a3);
void nsimd_store3a_sve_i16(i16* a0, nsimd_sve_vi16 a1, nsimd_sve_vi16 a2, nsimd_sve_vi16 a3);
void nsimd_store3a_sve_i8(i8* a0, nsimd_sve_vi8 a1, nsimd_sve_vi8 a2, nsimd_sve_vi8 a3);
void nsimd_store3a_sve_u64(u64* a0, nsimd_sve_vu64 a1, nsimd_sve_vu64 a2, nsimd_sve_vu64 a3);
void nsimd_store3a_sve_u32(u32* a0, nsimd_sve_vu32 a1, nsimd_sve_vu32 a2, nsimd_sve_vu32 a3);
void nsimd_store3a_sve_u16(u16* a0, nsimd_sve_vu16 a1, nsimd_sve_vu16 a2, nsimd_sve_vu16 a3);
void nsimd_store3a_sve_u8(u8* a0, nsimd_sve_vu8 a1, nsimd_sve_vu8 a2, nsimd_sve_vu8 a3);
```

### CPU

```c
void nsimd_store3a_cpu_f64(f64* a0, nsimd_cpu_vf64 a1, nsimd_cpu_vf64 a2, nsimd_cpu_vf64 a3);
void nsimd_store3a_cpu_f32(f32* a0, nsimd_cpu_vf32 a1, nsimd_cpu_vf32 a2, nsimd_cpu_vf32 a3);
void nsimd_store3a_cpu_f16(f16* a0, nsimd_cpu_vf16 a1, nsimd_cpu_vf16 a2, nsimd_cpu_vf16 a3);
void nsimd_store3a_cpu_i64(i64* a0, nsimd_cpu_vi64 a1, nsimd_cpu_vi64 a2, nsimd_cpu_vi64 a3);
void nsimd_store3a_cpu_i32(i32* a0, nsimd_cpu_vi32 a1, nsimd_cpu_vi32 a2, nsimd_cpu_vi32 a3);
void nsimd_store3a_cpu_i16(i16* a0, nsimd_cpu_vi16 a1, nsimd_cpu_vi16 a2, nsimd_cpu_vi16 a3);
void nsimd_store3a_cpu_i8(i8* a0, nsimd_cpu_vi8 a1, nsimd_cpu_vi8 a2, nsimd_cpu_vi8 a3);
void nsimd_store3a_cpu_u64(u64* a0, nsimd_cpu_vu64 a1, nsimd_cpu_vu64 a2, nsimd_cpu_vu64 a3);
void nsimd_store3a_cpu_u32(u32* a0, nsimd_cpu_vu32 a1, nsimd_cpu_vu32 a2, nsimd_cpu_vu32 a3);
void nsimd_store3a_cpu_u16(u16* a0, nsimd_cpu_vu16 a1, nsimd_cpu_vu16 a2, nsimd_cpu_vu16 a3);
void nsimd_store3a_cpu_u8(u8* a0, nsimd_cpu_vu8 a1, nsimd_cpu_vu8 a2, nsimd_cpu_vu8 a3);
```

### SSE2

```c
void nsimd_store3a_sse2_f64(f64* a0, nsimd_sse2_vf64 a1, nsimd_sse2_vf64 a2, nsimd_sse2_vf64 a3);
void nsimd_store3a_sse2_f32(f32* a0, nsimd_sse2_vf32 a1, nsimd_sse2_vf32 a2, nsimd_sse2_vf32 a3);
void nsimd_store3a_sse2_f16(f16* a0, nsimd_sse2_vf16 a1, nsimd_sse2_vf16 a2, nsimd_sse2_vf16 a3);
void nsimd_store3a_sse2_i64(i64* a0, nsimd_sse2_vi64 a1, nsimd_sse2_vi64 a2, nsimd_sse2_vi64 a3);
void nsimd_store3a_sse2_i32(i32* a0, nsimd_sse2_vi32 a1, nsimd_sse2_vi32 a2, nsimd_sse2_vi32 a3);
void nsimd_store3a_sse2_i16(i16* a0, nsimd_sse2_vi16 a1, nsimd_sse2_vi16 a2, nsimd_sse2_vi16 a3);
void nsimd_store3a_sse2_i8(i8* a0, nsimd_sse2_vi8 a1, nsimd_sse2_vi8 a2, nsimd_sse2_vi8 a3);
void nsimd_store3a_sse2_u64(u64* a0, nsimd_sse2_vu64 a1, nsimd_sse2_vu64 a2, nsimd_sse2_vu64 a3);
void nsimd_store3a_sse2_u32(u32* a0, nsimd_sse2_vu32 a1, nsimd_sse2_vu32 a2, nsimd_sse2_vu32 a3);
void nsimd_store3a_sse2_u16(u16* a0, nsimd_sse2_vu16 a1, nsimd_sse2_vu16 a2, nsimd_sse2_vu16 a3);
void nsimd_store3a_sse2_u8(u8* a0, nsimd_sse2_vu8 a1, nsimd_sse2_vu8 a2, nsimd_sse2_vu8 a3);
```

### AARCH64

```c
void nsimd_store3a_aarch64_f64(f64* a0, nsimd_aarch64_vf64 a1, nsimd_aarch64_vf64 a2, nsimd_aarch64_vf64 a3);
void nsimd_store3a_aarch64_f32(f32* a0, nsimd_aarch64_vf32 a1, nsimd_aarch64_vf32 a2, nsimd_aarch64_vf32 a3);
void nsimd_store3a_aarch64_f16(f16* a0, nsimd_aarch64_vf16 a1, nsimd_aarch64_vf16 a2, nsimd_aarch64_vf16 a3);
void nsimd_store3a_aarch64_i64(i64* a0, nsimd_aarch64_vi64 a1, nsimd_aarch64_vi64 a2, nsimd_aarch64_vi64 a3);
void nsimd_store3a_aarch64_i32(i32* a0, nsimd_aarch64_vi32 a1, nsimd_aarch64_vi32 a2, nsimd_aarch64_vi32 a3);
void nsimd_store3a_aarch64_i16(i16* a0, nsimd_aarch64_vi16 a1, nsimd_aarch64_vi16 a2, nsimd_aarch64_vi16 a3);
void nsimd_store3a_aarch64_i8(i8* a0, nsimd_aarch64_vi8 a1, nsimd_aarch64_vi8 a2, nsimd_aarch64_vi8 a3);
void nsimd_store3a_aarch64_u64(u64* a0, nsimd_aarch64_vu64 a1, nsimd_aarch64_vu64 a2, nsimd_aarch64_vu64 a3);
void nsimd_store3a_aarch64_u32(u32* a0, nsimd_aarch64_vu32 a1, nsimd_aarch64_vu32 a2, nsimd_aarch64_vu32 a3);
void nsimd_store3a_aarch64_u16(u16* a0, nsimd_aarch64_vu16 a1, nsimd_aarch64_vu16 a2, nsimd_aarch64_vu16 a3);
void nsimd_store3a_aarch64_u8(u8* a0, nsimd_aarch64_vu8 a1, nsimd_aarch64_vu8 a2, nsimd_aarch64_vu8 a3);
```

### SSE42

```c
void nsimd_store3a_sse42_f64(f64* a0, nsimd_sse42_vf64 a1, nsimd_sse42_vf64 a2, nsimd_sse42_vf64 a3);
void nsimd_store3a_sse42_f32(f32* a0, nsimd_sse42_vf32 a1, nsimd_sse42_vf32 a2, nsimd_sse42_vf32 a3);
void nsimd_store3a_sse42_f16(f16* a0, nsimd_sse42_vf16 a1, nsimd_sse42_vf16 a2, nsimd_sse42_vf16 a3);
void nsimd_store3a_sse42_i64(i64* a0, nsimd_sse42_vi64 a1, nsimd_sse42_vi64 a2, nsimd_sse42_vi64 a3);
void nsimd_store3a_sse42_i32(i32* a0, nsimd_sse42_vi32 a1, nsimd_sse42_vi32 a2, nsimd_sse42_vi32 a3);
void nsimd_store3a_sse42_i16(i16* a0, nsimd_sse42_vi16 a1, nsimd_sse42_vi16 a2, nsimd_sse42_vi16 a3);
void nsimd_store3a_sse42_i8(i8* a0, nsimd_sse42_vi8 a1, nsimd_sse42_vi8 a2, nsimd_sse42_vi8 a3);
void nsimd_store3a_sse42_u64(u64* a0, nsimd_sse42_vu64 a1, nsimd_sse42_vu64 a2, nsimd_sse42_vu64 a3);
void nsimd_store3a_sse42_u32(u32* a0, nsimd_sse42_vu32 a1, nsimd_sse42_vu32 a2, nsimd_sse42_vu32 a3);
void nsimd_store3a_sse42_u16(u16* a0, nsimd_sse42_vu16 a1, nsimd_sse42_vu16 a2, nsimd_sse42_vu16 a3);
void nsimd_store3a_sse42_u8(u8* a0, nsimd_sse42_vu8 a1, nsimd_sse42_vu8 a2, nsimd_sse42_vu8 a3);
```

## C++ base API (architecture specifics)

### NEON128

```c
void store3a(f64* a0, nsimd_neon128_vf64 a1, nsimd_neon128_vf64 a2, nsimd_neon128_vf64 a3, f64, neon128);
void store3a(f32* a0, nsimd_neon128_vf32 a1, nsimd_neon128_vf32 a2, nsimd_neon128_vf32 a3, f32, neon128);
void store3a(f16* a0, nsimd_neon128_vf16 a1, nsimd_neon128_vf16 a2, nsimd_neon128_vf16 a3, f16, neon128);
void store3a(i64* a0, nsimd_neon128_vi64 a1, nsimd_neon128_vi64 a2, nsimd_neon128_vi64 a3, i64, neon128);
void store3a(i32* a0, nsimd_neon128_vi32 a1, nsimd_neon128_vi32 a2, nsimd_neon128_vi32 a3, i32, neon128);
void store3a(i16* a0, nsimd_neon128_vi16 a1, nsimd_neon128_vi16 a2, nsimd_neon128_vi16 a3, i16, neon128);
void store3a(i8* a0, nsimd_neon128_vi8 a1, nsimd_neon128_vi8 a2, nsimd_neon128_vi8 a3, i8, neon128);
void store3a(u64* a0, nsimd_neon128_vu64 a1, nsimd_neon128_vu64 a2, nsimd_neon128_vu64 a3, u64, neon128);
void store3a(u32* a0, nsimd_neon128_vu32 a1, nsimd_neon128_vu32 a2, nsimd_neon128_vu32 a3, u32, neon128);
void store3a(u16* a0, nsimd_neon128_vu16 a1, nsimd_neon128_vu16 a2, nsimd_neon128_vu16 a3, u16, neon128);
void store3a(u8* a0, nsimd_neon128_vu8 a1, nsimd_neon128_vu8 a2, nsimd_neon128_vu8 a3, u8, neon128);
```

### AVX2

```c
void store3a(f64* a0, nsimd_avx2_vf64 a1, nsimd_avx2_vf64 a2, nsimd_avx2_vf64 a3, f64, avx2);
void store3a(f32* a0, nsimd_avx2_vf32 a1, nsimd_avx2_vf32 a2, nsimd_avx2_vf32 a3, f32, avx2);
void store3a(f16* a0, nsimd_avx2_vf16 a1, nsimd_avx2_vf16 a2, nsimd_avx2_vf16 a3, f16, avx2);
void store3a(i64* a0, nsimd_avx2_vi64 a1, nsimd_avx2_vi64 a2, nsimd_avx2_vi64 a3, i64, avx2);
void store3a(i32* a0, nsimd_avx2_vi32 a1, nsimd_avx2_vi32 a2, nsimd_avx2_vi32 a3, i32, avx2);
void store3a(i16* a0, nsimd_avx2_vi16 a1, nsimd_avx2_vi16 a2, nsimd_avx2_vi16 a3, i16, avx2);
void store3a(i8* a0, nsimd_avx2_vi8 a1, nsimd_avx2_vi8 a2, nsimd_avx2_vi8 a3, i8, avx2);
void store3a(u64* a0, nsimd_avx2_vu64 a1, nsimd_avx2_vu64 a2, nsimd_avx2_vu64 a3, u64, avx2);
void store3a(u32* a0, nsimd_avx2_vu32 a1, nsimd_avx2_vu32 a2, nsimd_avx2_vu32 a3, u32, avx2);
void store3a(u16* a0, nsimd_avx2_vu16 a1, nsimd_avx2_vu16 a2, nsimd_avx2_vu16 a3, u16, avx2);
void store3a(u8* a0, nsimd_avx2_vu8 a1, nsimd_avx2_vu8 a2, nsimd_avx2_vu8 a3, u8, avx2);
```

### AVX512_KNL

```c
void store3a(f64* a0, nsimd_avx512_knl_vf64 a1, nsimd_avx512_knl_vf64 a2, nsimd_avx512_knl_vf64 a3, f64, avx512_knl);
void store3a(f32* a0, nsimd_avx512_knl_vf32 a1, nsimd_avx512_knl_vf32 a2, nsimd_avx512_knl_vf32 a3, f32, avx512_knl);
void store3a(f16* a0, nsimd_avx512_knl_vf16 a1, nsimd_avx512_knl_vf16 a2, nsimd_avx512_knl_vf16 a3, f16, avx512_knl);
void store3a(i64* a0, nsimd_avx512_knl_vi64 a1, nsimd_avx512_knl_vi64 a2, nsimd_avx512_knl_vi64 a3, i64, avx512_knl);
void store3a(i32* a0, nsimd_avx512_knl_vi32 a1, nsimd_avx512_knl_vi32 a2, nsimd_avx512_knl_vi32 a3, i32, avx512_knl);
void store3a(i16* a0, nsimd_avx512_knl_vi16 a1, nsimd_avx512_knl_vi16 a2, nsimd_avx512_knl_vi16 a3, i16, avx512_knl);
void store3a(i8* a0, nsimd_avx512_knl_vi8 a1, nsimd_avx512_knl_vi8 a2, nsimd_avx512_knl_vi8 a3, i8, avx512_knl);
void store3a(u64* a0, nsimd_avx512_knl_vu64 a1, nsimd_avx512_knl_vu64 a2, nsimd_avx512_knl_vu64 a3, u64, avx512_knl);
void store3a(u32* a0, nsimd_avx512_knl_vu32 a1, nsimd_avx512_knl_vu32 a2, nsimd_avx512_knl_vu32 a3, u32, avx512_knl);
void store3a(u16* a0, nsimd_avx512_knl_vu16 a1, nsimd_avx512_knl_vu16 a2, nsimd_avx512_knl_vu16 a3, u16, avx512_knl);
void store3a(u8* a0, nsimd_avx512_knl_vu8 a1, nsimd_avx512_knl_vu8 a2, nsimd_avx512_knl_vu8 a3, u8, avx512_knl);
```

### AVX

```c
void store3a(f64* a0, nsimd_avx_vf64 a1, nsimd_avx_vf64 a2, nsimd_avx_vf64 a3, f64, avx);
void store3a(f32* a0, nsimd_avx_vf32 a1, nsimd_avx_vf32 a2, nsimd_avx_vf32 a3, f32, avx);
void store3a(f16* a0, nsimd_avx_vf16 a1, nsimd_avx_vf16 a2, nsimd_avx_vf16 a3, f16, avx);
void store3a(i64* a0, nsimd_avx_vi64 a1, nsimd_avx_vi64 a2, nsimd_avx_vi64 a3, i64, avx);
void store3a(i32* a0, nsimd_avx_vi32 a1, nsimd_avx_vi32 a2, nsimd_avx_vi32 a3, i32, avx);
void store3a(i16* a0, nsimd_avx_vi16 a1, nsimd_avx_vi16 a2, nsimd_avx_vi16 a3, i16, avx);
void store3a(i8* a0, nsimd_avx_vi8 a1, nsimd_avx_vi8 a2, nsimd_avx_vi8 a3, i8, avx);
void store3a(u64* a0, nsimd_avx_vu64 a1, nsimd_avx_vu64 a2, nsimd_avx_vu64 a3, u64, avx);
void store3a(u32* a0, nsimd_avx_vu32 a1, nsimd_avx_vu32 a2, nsimd_avx_vu32 a3, u32, avx);
void store3a(u16* a0, nsimd_avx_vu16 a1, nsimd_avx_vu16 a2, nsimd_avx_vu16 a3, u16, avx);
void store3a(u8* a0, nsimd_avx_vu8 a1, nsimd_avx_vu8 a2, nsimd_avx_vu8 a3, u8, avx);
```

### AVX512_SKYLAKE

```c
void store3a(f64* a0, nsimd_avx512_skylake_vf64 a1, nsimd_avx512_skylake_vf64 a2, nsimd_avx512_skylake_vf64 a3, f64, avx512_skylake);
void store3a(f32* a0, nsimd_avx512_skylake_vf32 a1, nsimd_avx512_skylake_vf32 a2, nsimd_avx512_skylake_vf32 a3, f32, avx512_skylake);
void store3a(f16* a0, nsimd_avx512_skylake_vf16 a1, nsimd_avx512_skylake_vf16 a2, nsimd_avx512_skylake_vf16 a3, f16, avx512_skylake);
void store3a(i64* a0, nsimd_avx512_skylake_vi64 a1, nsimd_avx512_skylake_vi64 a2, nsimd_avx512_skylake_vi64 a3, i64, avx512_skylake);
void store3a(i32* a0, nsimd_avx512_skylake_vi32 a1, nsimd_avx512_skylake_vi32 a2, nsimd_avx512_skylake_vi32 a3, i32, avx512_skylake);
void store3a(i16* a0, nsimd_avx512_skylake_vi16 a1, nsimd_avx512_skylake_vi16 a2, nsimd_avx512_skylake_vi16 a3, i16, avx512_skylake);
void store3a(i8* a0, nsimd_avx512_skylake_vi8 a1, nsimd_avx512_skylake_vi8 a2, nsimd_avx512_skylake_vi8 a3, i8, avx512_skylake);
void store3a(u64* a0, nsimd_avx512_skylake_vu64 a1, nsimd_avx512_skylake_vu64 a2, nsimd_avx512_skylake_vu64 a3, u64, avx512_skylake);
void store3a(u32* a0, nsimd_avx512_skylake_vu32 a1, nsimd_avx512_skylake_vu32 a2, nsimd_avx512_skylake_vu32 a3, u32, avx512_skylake);
void store3a(u16* a0, nsimd_avx512_skylake_vu16 a1, nsimd_avx512_skylake_vu16 a2, nsimd_avx512_skylake_vu16 a3, u16, avx512_skylake);
void store3a(u8* a0, nsimd_avx512_skylake_vu8 a1, nsimd_avx512_skylake_vu8 a2, nsimd_avx512_skylake_vu8 a3, u8, avx512_skylake);
```

### SVE

```c
void store3a(f64* a0, nsimd_sve_vf64 a1, nsimd_sve_vf64 a2, nsimd_sve_vf64 a3, f64, sve);
void store3a(f32* a0, nsimd_sve_vf32 a1, nsimd_sve_vf32 a2, nsimd_sve_vf32 a3, f32, sve);
void store3a(f16* a0, nsimd_sve_vf16 a1, nsimd_sve_vf16 a2, nsimd_sve_vf16 a3, f16, sve);
void store3a(i64* a0, nsimd_sve_vi64 a1, nsimd_sve_vi64 a2, nsimd_sve_vi64 a3, i64, sve);
void store3a(i32* a0, nsimd_sve_vi32 a1, nsimd_sve_vi32 a2, nsimd_sve_vi32 a3, i32, sve);
void store3a(i16* a0, nsimd_sve_vi16 a1, nsimd_sve_vi16 a2, nsimd_sve_vi16 a3, i16, sve);
void store3a(i8* a0, nsimd_sve_vi8 a1, nsimd_sve_vi8 a2, nsimd_sve_vi8 a3, i8, sve);
void store3a(u64* a0, nsimd_sve_vu64 a1, nsimd_sve_vu64 a2, nsimd_sve_vu64 a3, u64, sve);
void store3a(u32* a0, nsimd_sve_vu32 a1, nsimd_sve_vu32 a2, nsimd_sve_vu32 a3, u32, sve);
void store3a(u16* a0, nsimd_sve_vu16 a1, nsimd_sve_vu16 a2, nsimd_sve_vu16 a3, u16, sve);
void store3a(u8* a0, nsimd_sve_vu8 a1, nsimd_sve_vu8 a2, nsimd_sve_vu8 a3, u8, sve);
```

### CPU

```c
void store3a(f64* a0, nsimd_cpu_vf64 a1, nsimd_cpu_vf64 a2, nsimd_cpu_vf64 a3, f64, cpu);
void store3a(f32* a0, nsimd_cpu_vf32 a1, nsimd_cpu_vf32 a2, nsimd_cpu_vf32 a3, f32, cpu);
void store3a(f16* a0, nsimd_cpu_vf16 a1, nsimd_cpu_vf16 a2, nsimd_cpu_vf16 a3, f16, cpu);
void store3a(i64* a0, nsimd_cpu_vi64 a1, nsimd_cpu_vi64 a2, nsimd_cpu_vi64 a3, i64, cpu);
void store3a(i32* a0, nsimd_cpu_vi32 a1, nsimd_cpu_vi32 a2, nsimd_cpu_vi32 a3, i32, cpu);
void store3a(i16* a0, nsimd_cpu_vi16 a1, nsimd_cpu_vi16 a2, nsimd_cpu_vi16 a3, i16, cpu);
void store3a(i8* a0, nsimd_cpu_vi8 a1, nsimd_cpu_vi8 a2, nsimd_cpu_vi8 a3, i8, cpu);
void store3a(u64* a0, nsimd_cpu_vu64 a1, nsimd_cpu_vu64 a2, nsimd_cpu_vu64 a3, u64, cpu);
void store3a(u32* a0, nsimd_cpu_vu32 a1, nsimd_cpu_vu32 a2, nsimd_cpu_vu32 a3, u32, cpu);
void store3a(u16* a0, nsimd_cpu_vu16 a1, nsimd_cpu_vu16 a2, nsimd_cpu_vu16 a3, u16, cpu);
void store3a(u8* a0, nsimd_cpu_vu8 a1, nsimd_cpu_vu8 a2, nsimd_cpu_vu8 a3, u8, cpu);
```

### SSE2

```c
void store3a(f64* a0, nsimd_sse2_vf64 a1, nsimd_sse2_vf64 a2, nsimd_sse2_vf64 a3, f64, sse2);
void store3a(f32* a0, nsimd_sse2_vf32 a1, nsimd_sse2_vf32 a2, nsimd_sse2_vf32 a3, f32, sse2);
void store3a(f16* a0, nsimd_sse2_vf16 a1, nsimd_sse2_vf16 a2, nsimd_sse2_vf16 a3, f16, sse2);
void store3a(i64* a0, nsimd_sse2_vi64 a1, nsimd_sse2_vi64 a2, nsimd_sse2_vi64 a3, i64, sse2);
void store3a(i32* a0, nsimd_sse2_vi32 a1, nsimd_sse2_vi32 a2, nsimd_sse2_vi32 a3, i32, sse2);
void store3a(i16* a0, nsimd_sse2_vi16 a1, nsimd_sse2_vi16 a2, nsimd_sse2_vi16 a3, i16, sse2);
void store3a(i8* a0, nsimd_sse2_vi8 a1, nsimd_sse2_vi8 a2, nsimd_sse2_vi8 a3, i8, sse2);
void store3a(u64* a0, nsimd_sse2_vu64 a1, nsimd_sse2_vu64 a2, nsimd_sse2_vu64 a3, u64, sse2);
void store3a(u32* a0, nsimd_sse2_vu32 a1, nsimd_sse2_vu32 a2, nsimd_sse2_vu32 a3, u32, sse2);
void store3a(u16* a0, nsimd_sse2_vu16 a1, nsimd_sse2_vu16 a2, nsimd_sse2_vu16 a3, u16, sse2);
void store3a(u8* a0, nsimd_sse2_vu8 a1, nsimd_sse2_vu8 a2, nsimd_sse2_vu8 a3, u8, sse2);
```

### AARCH64

```c
void store3a(f64* a0, nsimd_aarch64_vf64 a1, nsimd_aarch64_vf64 a2, nsimd_aarch64_vf64 a3, f64, aarch64);
void store3a(f32* a0, nsimd_aarch64_vf32 a1, nsimd_aarch64_vf32 a2, nsimd_aarch64_vf32 a3, f32, aarch64);
void store3a(f16* a0, nsimd_aarch64_vf16 a1, nsimd_aarch64_vf16 a2, nsimd_aarch64_vf16 a3, f16, aarch64);
void store3a(i64* a0, nsimd_aarch64_vi64 a1, nsimd_aarch64_vi64 a2, nsimd_aarch64_vi64 a3, i64, aarch64);
void store3a(i32* a0, nsimd_aarch64_vi32 a1, nsimd_aarch64_vi32 a2, nsimd_aarch64_vi32 a3, i32, aarch64);
void store3a(i16* a0, nsimd_aarch64_vi16 a1, nsimd_aarch64_vi16 a2, nsimd_aarch64_vi16 a3, i16, aarch64);
void store3a(i8* a0, nsimd_aarch64_vi8 a1, nsimd_aarch64_vi8 a2, nsimd_aarch64_vi8 a3, i8, aarch64);
void store3a(u64* a0, nsimd_aarch64_vu64 a1, nsimd_aarch64_vu64 a2, nsimd_aarch64_vu64 a3, u64, aarch64);
void store3a(u32* a0, nsimd_aarch64_vu32 a1, nsimd_aarch64_vu32 a2, nsimd_aarch64_vu32 a3, u32, aarch64);
void store3a(u16* a0, nsimd_aarch64_vu16 a1, nsimd_aarch64_vu16 a2, nsimd_aarch64_vu16 a3, u16, aarch64);
void store3a(u8* a0, nsimd_aarch64_vu8 a1, nsimd_aarch64_vu8 a2, nsimd_aarch64_vu8 a3, u8, aarch64);
```

### SSE42

```c
void store3a(f64* a0, nsimd_sse42_vf64 a1, nsimd_sse42_vf64 a2, nsimd_sse42_vf64 a3, f64, sse42);
void store3a(f32* a0, nsimd_sse42_vf32 a1, nsimd_sse42_vf32 a2, nsimd_sse42_vf32 a3, f32, sse42);
void store3a(f16* a0, nsimd_sse42_vf16 a1, nsimd_sse42_vf16 a2, nsimd_sse42_vf16 a3, f16, sse42);
void store3a(i64* a0, nsimd_sse42_vi64 a1, nsimd_sse42_vi64 a2, nsimd_sse42_vi64 a3, i64, sse42);
void store3a(i32* a0, nsimd_sse42_vi32 a1, nsimd_sse42_vi32 a2, nsimd_sse42_vi32 a3, i32, sse42);
void store3a(i16* a0, nsimd_sse42_vi16 a1, nsimd_sse42_vi16 a2, nsimd_sse42_vi16 a3, i16, sse42);
void store3a(i8* a0, nsimd_sse42_vi8 a1, nsimd_sse42_vi8 a2, nsimd_sse42_vi8 a3, i8, sse42);
void store3a(u64* a0, nsimd_sse42_vu64 a1, nsimd_sse42_vu64 a2, nsimd_sse42_vu64 a3, u64, sse42);
void store3a(u32* a0, nsimd_sse42_vu32 a1, nsimd_sse42_vu32 a2, nsimd_sse42_vu32 a3, u32, sse42);
void store3a(u16* a0, nsimd_sse42_vu16 a1, nsimd_sse42_vu16 a2, nsimd_sse42_vu16 a3, u16, sse42);
void store3a(u8* a0, nsimd_sse42_vu8 a1, nsimd_sse42_vu8 a2, nsimd_sse42_vu8 a3, u8, sse42);
```