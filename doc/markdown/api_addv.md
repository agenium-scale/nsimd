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

# Horizontal sum

## Description

Returns the sum of all the elements contained in v

## C base API (generic)

```c
#define vaddv(a0, type)
#define vaddv_e(a0, type, simd_ext)
```

## C++ base API (generic)

```c++
template <typename A0, typename T> T addv(A0 a0, T);
```

## C++ advanced API

```c++
template <typename T, typename SimdExt> T addv(pack<T, 1, SimdExt> const& a0);
template <typename T, int N, typename SimdExt> T addv(pack<T, N, SimdExt> const& a0);
```

## C base API (architecture specifics)

### NEON128

```c
f64 nsimd_addv_neon128_f64(nsimd_neon128_vf64 a0);
f32 nsimd_addv_neon128_f32(nsimd_neon128_vf32 a0);
f16 nsimd_addv_neon128_f16(nsimd_neon128_vf16 a0);
```

### AVX2

```c
f64 nsimd_addv_avx2_f64(nsimd_avx2_vf64 a0);
f32 nsimd_addv_avx2_f32(nsimd_avx2_vf32 a0);
f16 nsimd_addv_avx2_f16(nsimd_avx2_vf16 a0);
```

### AVX512_KNL

```c
f64 nsimd_addv_avx512_knl_f64(nsimd_avx512_knl_vf64 a0);
f32 nsimd_addv_avx512_knl_f32(nsimd_avx512_knl_vf32 a0);
f16 nsimd_addv_avx512_knl_f16(nsimd_avx512_knl_vf16 a0);
```

### AVX

```c
f64 nsimd_addv_avx_f64(nsimd_avx_vf64 a0);
f32 nsimd_addv_avx_f32(nsimd_avx_vf32 a0);
f16 nsimd_addv_avx_f16(nsimd_avx_vf16 a0);
```

### AVX512_SKYLAKE

```c
f64 nsimd_addv_avx512_skylake_f64(nsimd_avx512_skylake_vf64 a0);
f32 nsimd_addv_avx512_skylake_f32(nsimd_avx512_skylake_vf32 a0);
f16 nsimd_addv_avx512_skylake_f16(nsimd_avx512_skylake_vf16 a0);
```

### SVE

```c
f64 nsimd_addv_sve_f64(nsimd_sve_vf64 a0);
f32 nsimd_addv_sve_f32(nsimd_sve_vf32 a0);
f16 nsimd_addv_sve_f16(nsimd_sve_vf16 a0);
```

### CPU

```c
f64 nsimd_addv_cpu_f64(nsimd_cpu_vf64 a0);
f32 nsimd_addv_cpu_f32(nsimd_cpu_vf32 a0);
f16 nsimd_addv_cpu_f16(nsimd_cpu_vf16 a0);
```

### SSE2

```c
f64 nsimd_addv_sse2_f64(nsimd_sse2_vf64 a0);
f32 nsimd_addv_sse2_f32(nsimd_sse2_vf32 a0);
f16 nsimd_addv_sse2_f16(nsimd_sse2_vf16 a0);
```

### AARCH64

```c
f64 nsimd_addv_aarch64_f64(nsimd_aarch64_vf64 a0);
f32 nsimd_addv_aarch64_f32(nsimd_aarch64_vf32 a0);
f16 nsimd_addv_aarch64_f16(nsimd_aarch64_vf16 a0);
```

### SSE42

```c
f64 nsimd_addv_sse42_f64(nsimd_sse42_vf64 a0);
f32 nsimd_addv_sse42_f32(nsimd_sse42_vf32 a0);
f16 nsimd_addv_sse42_f16(nsimd_sse42_vf16 a0);
```

## C++ base API (architecture specifics)

### NEON128

```c
f64 addv(nsimd_neon128_vf64 a0, f64, neon128);
f32 addv(nsimd_neon128_vf32 a0, f32, neon128);
f16 addv(nsimd_neon128_vf16 a0, f16, neon128);
```

### AVX2

```c
f64 addv(nsimd_avx2_vf64 a0, f64, avx2);
f32 addv(nsimd_avx2_vf32 a0, f32, avx2);
f16 addv(nsimd_avx2_vf16 a0, f16, avx2);
```

### AVX512_KNL

```c
f64 addv(nsimd_avx512_knl_vf64 a0, f64, avx512_knl);
f32 addv(nsimd_avx512_knl_vf32 a0, f32, avx512_knl);
f16 addv(nsimd_avx512_knl_vf16 a0, f16, avx512_knl);
```

### AVX

```c
f64 addv(nsimd_avx_vf64 a0, f64, avx);
f32 addv(nsimd_avx_vf32 a0, f32, avx);
f16 addv(nsimd_avx_vf16 a0, f16, avx);
```

### AVX512_SKYLAKE

```c
f64 addv(nsimd_avx512_skylake_vf64 a0, f64, avx512_skylake);
f32 addv(nsimd_avx512_skylake_vf32 a0, f32, avx512_skylake);
f16 addv(nsimd_avx512_skylake_vf16 a0, f16, avx512_skylake);
```

### SVE

```c
f64 addv(nsimd_sve_vf64 a0, f64, sve);
f32 addv(nsimd_sve_vf32 a0, f32, sve);
f16 addv(nsimd_sve_vf16 a0, f16, sve);
```

### CPU

```c
f64 addv(nsimd_cpu_vf64 a0, f64, cpu);
f32 addv(nsimd_cpu_vf32 a0, f32, cpu);
f16 addv(nsimd_cpu_vf16 a0, f16, cpu);
```

### SSE2

```c
f64 addv(nsimd_sse2_vf64 a0, f64, sse2);
f32 addv(nsimd_sse2_vf32 a0, f32, sse2);
f16 addv(nsimd_sse2_vf16 a0, f16, sse2);
```

### AARCH64

```c
f64 addv(nsimd_aarch64_vf64 a0, f64, aarch64);
f32 addv(nsimd_aarch64_vf32 a0, f32, aarch64);
f16 addv(nsimd_aarch64_vf16 a0, f16, aarch64);
```

### SSE42

```c
f64 addv(nsimd_sse42_vf64 a0, f64, sse42);
f32 addv(nsimd_sse42_vf32 a0, f32, sse42);
f16 addv(nsimd_sse42_vf16 a0, f16, sse42);
```