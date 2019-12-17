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

# Value broadcast

## Description

Returns a vector whose all elements are set to the given value.

## C base API (generic)

```c
#define vset1(a0, type)
#define vset1_e(a0, type, simd_ext)
```

## C++ base API (generic)

```c++
template <typename A0, typename T> typename simd_traits<T, NSIMD_SIMD>::simd_vector set1(A0 a0, T);
```

## C++ advanced API

```c++
template <typename T, typename SimdExt, typename A0> pack<T, 1, SimdExt> set1(pack<T, 1, SimdExt> const&, A0 a0);
template <typename T, int N, typename SimdExt, typename A0> pack<T, N, SimdExt> set1(pack<T, N, SimdExt> const&, A0 a0);
template <typename SimdVector, typename A0> SimdVector set1(A0 a0);
```

## C base API (architecture specifics)

### NEON128

```c
nsimd_neon128_vf64 nsimd_set1_neon128_f64(f64 a0);
nsimd_neon128_vf32 nsimd_set1_neon128_f32(f32 a0);
nsimd_neon128_vf16 nsimd_set1_neon128_f16(f16 a0);
nsimd_neon128_vi64 nsimd_set1_neon128_i64(i64 a0);
nsimd_neon128_vi32 nsimd_set1_neon128_i32(i32 a0);
nsimd_neon128_vi16 nsimd_set1_neon128_i16(i16 a0);
nsimd_neon128_vi8 nsimd_set1_neon128_i8(i8 a0);
nsimd_neon128_vu64 nsimd_set1_neon128_u64(u64 a0);
nsimd_neon128_vu32 nsimd_set1_neon128_u32(u32 a0);
nsimd_neon128_vu16 nsimd_set1_neon128_u16(u16 a0);
nsimd_neon128_vu8 nsimd_set1_neon128_u8(u8 a0);
```

### AVX2

```c
nsimd_avx2_vf64 nsimd_set1_avx2_f64(f64 a0);
nsimd_avx2_vf32 nsimd_set1_avx2_f32(f32 a0);
nsimd_avx2_vf16 nsimd_set1_avx2_f16(f16 a0);
nsimd_avx2_vi64 nsimd_set1_avx2_i64(i64 a0);
nsimd_avx2_vi32 nsimd_set1_avx2_i32(i32 a0);
nsimd_avx2_vi16 nsimd_set1_avx2_i16(i16 a0);
nsimd_avx2_vi8 nsimd_set1_avx2_i8(i8 a0);
nsimd_avx2_vu64 nsimd_set1_avx2_u64(u64 a0);
nsimd_avx2_vu32 nsimd_set1_avx2_u32(u32 a0);
nsimd_avx2_vu16 nsimd_set1_avx2_u16(u16 a0);
nsimd_avx2_vu8 nsimd_set1_avx2_u8(u8 a0);
```

### AVX512_KNL

```c
nsimd_avx512_knl_vf64 nsimd_set1_avx512_knl_f64(f64 a0);
nsimd_avx512_knl_vf32 nsimd_set1_avx512_knl_f32(f32 a0);
nsimd_avx512_knl_vf16 nsimd_set1_avx512_knl_f16(f16 a0);
nsimd_avx512_knl_vi64 nsimd_set1_avx512_knl_i64(i64 a0);
nsimd_avx512_knl_vi32 nsimd_set1_avx512_knl_i32(i32 a0);
nsimd_avx512_knl_vi16 nsimd_set1_avx512_knl_i16(i16 a0);
nsimd_avx512_knl_vi8 nsimd_set1_avx512_knl_i8(i8 a0);
nsimd_avx512_knl_vu64 nsimd_set1_avx512_knl_u64(u64 a0);
nsimd_avx512_knl_vu32 nsimd_set1_avx512_knl_u32(u32 a0);
nsimd_avx512_knl_vu16 nsimd_set1_avx512_knl_u16(u16 a0);
nsimd_avx512_knl_vu8 nsimd_set1_avx512_knl_u8(u8 a0);
```

### AVX

```c
nsimd_avx_vf64 nsimd_set1_avx_f64(f64 a0);
nsimd_avx_vf32 nsimd_set1_avx_f32(f32 a0);
nsimd_avx_vf16 nsimd_set1_avx_f16(f16 a0);
nsimd_avx_vi64 nsimd_set1_avx_i64(i64 a0);
nsimd_avx_vi32 nsimd_set1_avx_i32(i32 a0);
nsimd_avx_vi16 nsimd_set1_avx_i16(i16 a0);
nsimd_avx_vi8 nsimd_set1_avx_i8(i8 a0);
nsimd_avx_vu64 nsimd_set1_avx_u64(u64 a0);
nsimd_avx_vu32 nsimd_set1_avx_u32(u32 a0);
nsimd_avx_vu16 nsimd_set1_avx_u16(u16 a0);
nsimd_avx_vu8 nsimd_set1_avx_u8(u8 a0);
```

### AVX512_SKYLAKE

```c
nsimd_avx512_skylake_vf64 nsimd_set1_avx512_skylake_f64(f64 a0);
nsimd_avx512_skylake_vf32 nsimd_set1_avx512_skylake_f32(f32 a0);
nsimd_avx512_skylake_vf16 nsimd_set1_avx512_skylake_f16(f16 a0);
nsimd_avx512_skylake_vi64 nsimd_set1_avx512_skylake_i64(i64 a0);
nsimd_avx512_skylake_vi32 nsimd_set1_avx512_skylake_i32(i32 a0);
nsimd_avx512_skylake_vi16 nsimd_set1_avx512_skylake_i16(i16 a0);
nsimd_avx512_skylake_vi8 nsimd_set1_avx512_skylake_i8(i8 a0);
nsimd_avx512_skylake_vu64 nsimd_set1_avx512_skylake_u64(u64 a0);
nsimd_avx512_skylake_vu32 nsimd_set1_avx512_skylake_u32(u32 a0);
nsimd_avx512_skylake_vu16 nsimd_set1_avx512_skylake_u16(u16 a0);
nsimd_avx512_skylake_vu8 nsimd_set1_avx512_skylake_u8(u8 a0);
```

### SVE

```c
nsimd_sve_vf64 nsimd_set1_sve_f64(f64 a0);
nsimd_sve_vf32 nsimd_set1_sve_f32(f32 a0);
nsimd_sve_vf16 nsimd_set1_sve_f16(f16 a0);
nsimd_sve_vi64 nsimd_set1_sve_i64(i64 a0);
nsimd_sve_vi32 nsimd_set1_sve_i32(i32 a0);
nsimd_sve_vi16 nsimd_set1_sve_i16(i16 a0);
nsimd_sve_vi8 nsimd_set1_sve_i8(i8 a0);
nsimd_sve_vu64 nsimd_set1_sve_u64(u64 a0);
nsimd_sve_vu32 nsimd_set1_sve_u32(u32 a0);
nsimd_sve_vu16 nsimd_set1_sve_u16(u16 a0);
nsimd_sve_vu8 nsimd_set1_sve_u8(u8 a0);
```

### CPU

```c
nsimd_cpu_vf64 nsimd_set1_cpu_f64(f64 a0);
nsimd_cpu_vf32 nsimd_set1_cpu_f32(f32 a0);
nsimd_cpu_vf16 nsimd_set1_cpu_f16(f16 a0);
nsimd_cpu_vi64 nsimd_set1_cpu_i64(i64 a0);
nsimd_cpu_vi32 nsimd_set1_cpu_i32(i32 a0);
nsimd_cpu_vi16 nsimd_set1_cpu_i16(i16 a0);
nsimd_cpu_vi8 nsimd_set1_cpu_i8(i8 a0);
nsimd_cpu_vu64 nsimd_set1_cpu_u64(u64 a0);
nsimd_cpu_vu32 nsimd_set1_cpu_u32(u32 a0);
nsimd_cpu_vu16 nsimd_set1_cpu_u16(u16 a0);
nsimd_cpu_vu8 nsimd_set1_cpu_u8(u8 a0);
```

### SSE2

```c
nsimd_sse2_vf64 nsimd_set1_sse2_f64(f64 a0);
nsimd_sse2_vf32 nsimd_set1_sse2_f32(f32 a0);
nsimd_sse2_vf16 nsimd_set1_sse2_f16(f16 a0);
nsimd_sse2_vi64 nsimd_set1_sse2_i64(i64 a0);
nsimd_sse2_vi32 nsimd_set1_sse2_i32(i32 a0);
nsimd_sse2_vi16 nsimd_set1_sse2_i16(i16 a0);
nsimd_sse2_vi8 nsimd_set1_sse2_i8(i8 a0);
nsimd_sse2_vu64 nsimd_set1_sse2_u64(u64 a0);
nsimd_sse2_vu32 nsimd_set1_sse2_u32(u32 a0);
nsimd_sse2_vu16 nsimd_set1_sse2_u16(u16 a0);
nsimd_sse2_vu8 nsimd_set1_sse2_u8(u8 a0);
```

### AARCH64

```c
nsimd_aarch64_vf64 nsimd_set1_aarch64_f64(f64 a0);
nsimd_aarch64_vf32 nsimd_set1_aarch64_f32(f32 a0);
nsimd_aarch64_vf16 nsimd_set1_aarch64_f16(f16 a0);
nsimd_aarch64_vi64 nsimd_set1_aarch64_i64(i64 a0);
nsimd_aarch64_vi32 nsimd_set1_aarch64_i32(i32 a0);
nsimd_aarch64_vi16 nsimd_set1_aarch64_i16(i16 a0);
nsimd_aarch64_vi8 nsimd_set1_aarch64_i8(i8 a0);
nsimd_aarch64_vu64 nsimd_set1_aarch64_u64(u64 a0);
nsimd_aarch64_vu32 nsimd_set1_aarch64_u32(u32 a0);
nsimd_aarch64_vu16 nsimd_set1_aarch64_u16(u16 a0);
nsimd_aarch64_vu8 nsimd_set1_aarch64_u8(u8 a0);
```

### SSE42

```c
nsimd_sse42_vf64 nsimd_set1_sse42_f64(f64 a0);
nsimd_sse42_vf32 nsimd_set1_sse42_f32(f32 a0);
nsimd_sse42_vf16 nsimd_set1_sse42_f16(f16 a0);
nsimd_sse42_vi64 nsimd_set1_sse42_i64(i64 a0);
nsimd_sse42_vi32 nsimd_set1_sse42_i32(i32 a0);
nsimd_sse42_vi16 nsimd_set1_sse42_i16(i16 a0);
nsimd_sse42_vi8 nsimd_set1_sse42_i8(i8 a0);
nsimd_sse42_vu64 nsimd_set1_sse42_u64(u64 a0);
nsimd_sse42_vu32 nsimd_set1_sse42_u32(u32 a0);
nsimd_sse42_vu16 nsimd_set1_sse42_u16(u16 a0);
nsimd_sse42_vu8 nsimd_set1_sse42_u8(u8 a0);
```

## C++ base API (architecture specifics)

### NEON128

```c
nsimd_neon128_vf64 set1(f64 a0, f64, neon128);
nsimd_neon128_vf32 set1(f32 a0, f32, neon128);
nsimd_neon128_vf16 set1(f16 a0, f16, neon128);
nsimd_neon128_vi64 set1(i64 a0, i64, neon128);
nsimd_neon128_vi32 set1(i32 a0, i32, neon128);
nsimd_neon128_vi16 set1(i16 a0, i16, neon128);
nsimd_neon128_vi8 set1(i8 a0, i8, neon128);
nsimd_neon128_vu64 set1(u64 a0, u64, neon128);
nsimd_neon128_vu32 set1(u32 a0, u32, neon128);
nsimd_neon128_vu16 set1(u16 a0, u16, neon128);
nsimd_neon128_vu8 set1(u8 a0, u8, neon128);
```

### AVX2

```c
nsimd_avx2_vf64 set1(f64 a0, f64, avx2);
nsimd_avx2_vf32 set1(f32 a0, f32, avx2);
nsimd_avx2_vf16 set1(f16 a0, f16, avx2);
nsimd_avx2_vi64 set1(i64 a0, i64, avx2);
nsimd_avx2_vi32 set1(i32 a0, i32, avx2);
nsimd_avx2_vi16 set1(i16 a0, i16, avx2);
nsimd_avx2_vi8 set1(i8 a0, i8, avx2);
nsimd_avx2_vu64 set1(u64 a0, u64, avx2);
nsimd_avx2_vu32 set1(u32 a0, u32, avx2);
nsimd_avx2_vu16 set1(u16 a0, u16, avx2);
nsimd_avx2_vu8 set1(u8 a0, u8, avx2);
```

### AVX512_KNL

```c
nsimd_avx512_knl_vf64 set1(f64 a0, f64, avx512_knl);
nsimd_avx512_knl_vf32 set1(f32 a0, f32, avx512_knl);
nsimd_avx512_knl_vf16 set1(f16 a0, f16, avx512_knl);
nsimd_avx512_knl_vi64 set1(i64 a0, i64, avx512_knl);
nsimd_avx512_knl_vi32 set1(i32 a0, i32, avx512_knl);
nsimd_avx512_knl_vi16 set1(i16 a0, i16, avx512_knl);
nsimd_avx512_knl_vi8 set1(i8 a0, i8, avx512_knl);
nsimd_avx512_knl_vu64 set1(u64 a0, u64, avx512_knl);
nsimd_avx512_knl_vu32 set1(u32 a0, u32, avx512_knl);
nsimd_avx512_knl_vu16 set1(u16 a0, u16, avx512_knl);
nsimd_avx512_knl_vu8 set1(u8 a0, u8, avx512_knl);
```

### AVX

```c
nsimd_avx_vf64 set1(f64 a0, f64, avx);
nsimd_avx_vf32 set1(f32 a0, f32, avx);
nsimd_avx_vf16 set1(f16 a0, f16, avx);
nsimd_avx_vi64 set1(i64 a0, i64, avx);
nsimd_avx_vi32 set1(i32 a0, i32, avx);
nsimd_avx_vi16 set1(i16 a0, i16, avx);
nsimd_avx_vi8 set1(i8 a0, i8, avx);
nsimd_avx_vu64 set1(u64 a0, u64, avx);
nsimd_avx_vu32 set1(u32 a0, u32, avx);
nsimd_avx_vu16 set1(u16 a0, u16, avx);
nsimd_avx_vu8 set1(u8 a0, u8, avx);
```

### AVX512_SKYLAKE

```c
nsimd_avx512_skylake_vf64 set1(f64 a0, f64, avx512_skylake);
nsimd_avx512_skylake_vf32 set1(f32 a0, f32, avx512_skylake);
nsimd_avx512_skylake_vf16 set1(f16 a0, f16, avx512_skylake);
nsimd_avx512_skylake_vi64 set1(i64 a0, i64, avx512_skylake);
nsimd_avx512_skylake_vi32 set1(i32 a0, i32, avx512_skylake);
nsimd_avx512_skylake_vi16 set1(i16 a0, i16, avx512_skylake);
nsimd_avx512_skylake_vi8 set1(i8 a0, i8, avx512_skylake);
nsimd_avx512_skylake_vu64 set1(u64 a0, u64, avx512_skylake);
nsimd_avx512_skylake_vu32 set1(u32 a0, u32, avx512_skylake);
nsimd_avx512_skylake_vu16 set1(u16 a0, u16, avx512_skylake);
nsimd_avx512_skylake_vu8 set1(u8 a0, u8, avx512_skylake);
```

### SVE

```c
nsimd_sve_vf64 set1(f64 a0, f64, sve);
nsimd_sve_vf32 set1(f32 a0, f32, sve);
nsimd_sve_vf16 set1(f16 a0, f16, sve);
nsimd_sve_vi64 set1(i64 a0, i64, sve);
nsimd_sve_vi32 set1(i32 a0, i32, sve);
nsimd_sve_vi16 set1(i16 a0, i16, sve);
nsimd_sve_vi8 set1(i8 a0, i8, sve);
nsimd_sve_vu64 set1(u64 a0, u64, sve);
nsimd_sve_vu32 set1(u32 a0, u32, sve);
nsimd_sve_vu16 set1(u16 a0, u16, sve);
nsimd_sve_vu8 set1(u8 a0, u8, sve);
```

### CPU

```c
nsimd_cpu_vf64 set1(f64 a0, f64, cpu);
nsimd_cpu_vf32 set1(f32 a0, f32, cpu);
nsimd_cpu_vf16 set1(f16 a0, f16, cpu);
nsimd_cpu_vi64 set1(i64 a0, i64, cpu);
nsimd_cpu_vi32 set1(i32 a0, i32, cpu);
nsimd_cpu_vi16 set1(i16 a0, i16, cpu);
nsimd_cpu_vi8 set1(i8 a0, i8, cpu);
nsimd_cpu_vu64 set1(u64 a0, u64, cpu);
nsimd_cpu_vu32 set1(u32 a0, u32, cpu);
nsimd_cpu_vu16 set1(u16 a0, u16, cpu);
nsimd_cpu_vu8 set1(u8 a0, u8, cpu);
```

### SSE2

```c
nsimd_sse2_vf64 set1(f64 a0, f64, sse2);
nsimd_sse2_vf32 set1(f32 a0, f32, sse2);
nsimd_sse2_vf16 set1(f16 a0, f16, sse2);
nsimd_sse2_vi64 set1(i64 a0, i64, sse2);
nsimd_sse2_vi32 set1(i32 a0, i32, sse2);
nsimd_sse2_vi16 set1(i16 a0, i16, sse2);
nsimd_sse2_vi8 set1(i8 a0, i8, sse2);
nsimd_sse2_vu64 set1(u64 a0, u64, sse2);
nsimd_sse2_vu32 set1(u32 a0, u32, sse2);
nsimd_sse2_vu16 set1(u16 a0, u16, sse2);
nsimd_sse2_vu8 set1(u8 a0, u8, sse2);
```

### AARCH64

```c
nsimd_aarch64_vf64 set1(f64 a0, f64, aarch64);
nsimd_aarch64_vf32 set1(f32 a0, f32, aarch64);
nsimd_aarch64_vf16 set1(f16 a0, f16, aarch64);
nsimd_aarch64_vi64 set1(i64 a0, i64, aarch64);
nsimd_aarch64_vi32 set1(i32 a0, i32, aarch64);
nsimd_aarch64_vi16 set1(i16 a0, i16, aarch64);
nsimd_aarch64_vi8 set1(i8 a0, i8, aarch64);
nsimd_aarch64_vu64 set1(u64 a0, u64, aarch64);
nsimd_aarch64_vu32 set1(u32 a0, u32, aarch64);
nsimd_aarch64_vu16 set1(u16 a0, u16, aarch64);
nsimd_aarch64_vu8 set1(u8 a0, u8, aarch64);
```

### SSE42

```c
nsimd_sse42_vf64 set1(f64 a0, f64, sse42);
nsimd_sse42_vf32 set1(f32 a0, f32, sse42);
nsimd_sse42_vf16 set1(f16 a0, f16, sse42);
nsimd_sse42_vi64 set1(i64 a0, i64, sse42);
nsimd_sse42_vi32 set1(i32 a0, i32, sse42);
nsimd_sse42_vi16 set1(i16 a0, i16, sse42);
nsimd_sse42_vi8 set1(i8 a0, i8, sse42);
nsimd_sse42_vu64 set1(u64 a0, u64, sse42);
nsimd_sse42_vu32 set1(u32 a0, u32, sse42);
nsimd_sse42_vu16 set1(u16 a0, u16, sse42);
nsimd_sse42_vu8 set1(u8 a0, u8, sse42);
```