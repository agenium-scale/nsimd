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

# Arc-secant with output in degrees

## Description

Returns the arc-secant with output in degrees of the argument. Defined over $(-∞, -1] ⋃ [1, +∞)$.

## C base API (generic)

```c
#define vasecd(a0, type)
#define vasecd_e(a0, type, simd_ext)
```

## C++ base API (generic)

```c++
template <typename A0, typename T> typename simd_traits<T, NSIMD_SIMD>::simd_vector asecd(A0 a0, T);
```

## C++ advanced API

```c++
template <typename T, typename SimdExt> pack<T, 1, SimdExt> asecd(pack<T, 1, SimdExt> const& a0);
template <typename T, int N, typename SimdExt> pack<T, N, SimdExt> asecd(pack<T, N, SimdExt> const& a0);
```

## C base API (architecture specifics)

### NEON128

```c
nsimd_neon128_vf64 nsimd_asecd_neon128_f64(nsimd_neon128_vf64 a0);
nsimd_neon128_vf32 nsimd_asecd_neon128_f32(nsimd_neon128_vf32 a0);
nsimd_neon128_vf16 nsimd_asecd_neon128_f16(nsimd_neon128_vf16 a0);
```

### AVX2

```c
nsimd_avx2_vf64 nsimd_asecd_avx2_f64(nsimd_avx2_vf64 a0);
nsimd_avx2_vf32 nsimd_asecd_avx2_f32(nsimd_avx2_vf32 a0);
nsimd_avx2_vf16 nsimd_asecd_avx2_f16(nsimd_avx2_vf16 a0);
```

### AVX512_KNL

```c
nsimd_avx512_knl_vf64 nsimd_asecd_avx512_knl_f64(nsimd_avx512_knl_vf64 a0);
nsimd_avx512_knl_vf32 nsimd_asecd_avx512_knl_f32(nsimd_avx512_knl_vf32 a0);
nsimd_avx512_knl_vf16 nsimd_asecd_avx512_knl_f16(nsimd_avx512_knl_vf16 a0);
```

### AVX

```c
nsimd_avx_vf64 nsimd_asecd_avx_f64(nsimd_avx_vf64 a0);
nsimd_avx_vf32 nsimd_asecd_avx_f32(nsimd_avx_vf32 a0);
nsimd_avx_vf16 nsimd_asecd_avx_f16(nsimd_avx_vf16 a0);
```

### AVX512_SKYLAKE

```c
nsimd_avx512_skylake_vf64 nsimd_asecd_avx512_skylake_f64(nsimd_avx512_skylake_vf64 a0);
nsimd_avx512_skylake_vf32 nsimd_asecd_avx512_skylake_f32(nsimd_avx512_skylake_vf32 a0);
nsimd_avx512_skylake_vf16 nsimd_asecd_avx512_skylake_f16(nsimd_avx512_skylake_vf16 a0);
```

### SVE

```c
nsimd_sve_vf64 nsimd_asecd_sve_f64(nsimd_sve_vf64 a0);
nsimd_sve_vf32 nsimd_asecd_sve_f32(nsimd_sve_vf32 a0);
nsimd_sve_vf16 nsimd_asecd_sve_f16(nsimd_sve_vf16 a0);
```

### CPU

```c
nsimd_cpu_vf64 nsimd_asecd_cpu_f64(nsimd_cpu_vf64 a0);
nsimd_cpu_vf32 nsimd_asecd_cpu_f32(nsimd_cpu_vf32 a0);
nsimd_cpu_vf16 nsimd_asecd_cpu_f16(nsimd_cpu_vf16 a0);
```

### SSE2

```c
nsimd_sse2_vf64 nsimd_asecd_sse2_f64(nsimd_sse2_vf64 a0);
nsimd_sse2_vf32 nsimd_asecd_sse2_f32(nsimd_sse2_vf32 a0);
nsimd_sse2_vf16 nsimd_asecd_sse2_f16(nsimd_sse2_vf16 a0);
```

### AARCH64

```c
nsimd_aarch64_vf64 nsimd_asecd_aarch64_f64(nsimd_aarch64_vf64 a0);
nsimd_aarch64_vf32 nsimd_asecd_aarch64_f32(nsimd_aarch64_vf32 a0);
nsimd_aarch64_vf16 nsimd_asecd_aarch64_f16(nsimd_aarch64_vf16 a0);
```

### SSE42

```c
nsimd_sse42_vf64 nsimd_asecd_sse42_f64(nsimd_sse42_vf64 a0);
nsimd_sse42_vf32 nsimd_asecd_sse42_f32(nsimd_sse42_vf32 a0);
nsimd_sse42_vf16 nsimd_asecd_sse42_f16(nsimd_sse42_vf16 a0);
```

## C++ base API (architecture specifics)

### NEON128

```c
nsimd_neon128_vf64 asecd(nsimd_neon128_vf64 a0, f64, neon128);
nsimd_neon128_vf32 asecd(nsimd_neon128_vf32 a0, f32, neon128);
nsimd_neon128_vf16 asecd(nsimd_neon128_vf16 a0, f16, neon128);
```

### AVX2

```c
nsimd_avx2_vf64 asecd(nsimd_avx2_vf64 a0, f64, avx2);
nsimd_avx2_vf32 asecd(nsimd_avx2_vf32 a0, f32, avx2);
nsimd_avx2_vf16 asecd(nsimd_avx2_vf16 a0, f16, avx2);
```

### AVX512_KNL

```c
nsimd_avx512_knl_vf64 asecd(nsimd_avx512_knl_vf64 a0, f64, avx512_knl);
nsimd_avx512_knl_vf32 asecd(nsimd_avx512_knl_vf32 a0, f32, avx512_knl);
nsimd_avx512_knl_vf16 asecd(nsimd_avx512_knl_vf16 a0, f16, avx512_knl);
```

### AVX

```c
nsimd_avx_vf64 asecd(nsimd_avx_vf64 a0, f64, avx);
nsimd_avx_vf32 asecd(nsimd_avx_vf32 a0, f32, avx);
nsimd_avx_vf16 asecd(nsimd_avx_vf16 a0, f16, avx);
```

### AVX512_SKYLAKE

```c
nsimd_avx512_skylake_vf64 asecd(nsimd_avx512_skylake_vf64 a0, f64, avx512_skylake);
nsimd_avx512_skylake_vf32 asecd(nsimd_avx512_skylake_vf32 a0, f32, avx512_skylake);
nsimd_avx512_skylake_vf16 asecd(nsimd_avx512_skylake_vf16 a0, f16, avx512_skylake);
```

### SVE

```c
nsimd_sve_vf64 asecd(nsimd_sve_vf64 a0, f64, sve);
nsimd_sve_vf32 asecd(nsimd_sve_vf32 a0, f32, sve);
nsimd_sve_vf16 asecd(nsimd_sve_vf16 a0, f16, sve);
```

### CPU

```c
nsimd_cpu_vf64 asecd(nsimd_cpu_vf64 a0, f64, cpu);
nsimd_cpu_vf32 asecd(nsimd_cpu_vf32 a0, f32, cpu);
nsimd_cpu_vf16 asecd(nsimd_cpu_vf16 a0, f16, cpu);
```

### SSE2

```c
nsimd_sse2_vf64 asecd(nsimd_sse2_vf64 a0, f64, sse2);
nsimd_sse2_vf32 asecd(nsimd_sse2_vf32 a0, f32, sse2);
nsimd_sse2_vf16 asecd(nsimd_sse2_vf16 a0, f16, sse2);
```

### AARCH64

```c
nsimd_aarch64_vf64 asecd(nsimd_aarch64_vf64 a0, f64, aarch64);
nsimd_aarch64_vf32 asecd(nsimd_aarch64_vf32 a0, f32, aarch64);
nsimd_aarch64_vf16 asecd(nsimd_aarch64_vf16 a0, f16, aarch64);
```

### SSE42

```c
nsimd_sse42_vf64 asecd(nsimd_sse42_vf64 a0, f64, sse42);
nsimd_sse42_vf32 asecd(nsimd_sse42_vf32 a0, f32, sse42);
nsimd_sse42_vf16 asecd(nsimd_sse42_vf16 a0, f16, sse42);
```