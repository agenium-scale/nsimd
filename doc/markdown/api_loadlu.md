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

# Load vector of logicals

## Description

Load data from unaligned memory and interpret it as booleans. Zero is interpreted as False and nonzero as True.

## C base API (generic)

```c
#define vloadlu(a0, type)
#define vloadlu_e(a0, type, simd_ext)
```

## C++ base API (generic)

```c++
template <typename A0, typename T> typename simd_traits<T, NSIMD_SIMD>::simd_vectorl loadlu(A0 a0, T);
```

## C++ advanced API

```c++
template <typename T, typename SimdExt, typename A0> packl<T, 1, SimdExt> loadlu(packl<T, 1, SimdExt> const&, A0 a0);
template <typename T, int N, typename SimdExt, typename A0> packl<T, N, SimdExt> loadlu(packl<T, N, SimdExt> const&, A0 a0);
template <typename SimdVector, typename A0> SimdVector loadlu(A0 a0);
```

## C base API (architecture specifics)

### NEON128

```c
nsimd_neon128_vlf64 nsimd_loadlu_neon128_f64(f64 const* a0);
nsimd_neon128_vlf32 nsimd_loadlu_neon128_f32(f32 const* a0);
nsimd_neon128_vlf16 nsimd_loadlu_neon128_f16(f16 const* a0);
nsimd_neon128_vli64 nsimd_loadlu_neon128_i64(i64 const* a0);
nsimd_neon128_vli32 nsimd_loadlu_neon128_i32(i32 const* a0);
nsimd_neon128_vli16 nsimd_loadlu_neon128_i16(i16 const* a0);
nsimd_neon128_vli8 nsimd_loadlu_neon128_i8(i8 const* a0);
nsimd_neon128_vlu64 nsimd_loadlu_neon128_u64(u64 const* a0);
nsimd_neon128_vlu32 nsimd_loadlu_neon128_u32(u32 const* a0);
nsimd_neon128_vlu16 nsimd_loadlu_neon128_u16(u16 const* a0);
nsimd_neon128_vlu8 nsimd_loadlu_neon128_u8(u8 const* a0);
```

### AVX2

```c
nsimd_avx2_vlf64 nsimd_loadlu_avx2_f64(f64 const* a0);
nsimd_avx2_vlf32 nsimd_loadlu_avx2_f32(f32 const* a0);
nsimd_avx2_vlf16 nsimd_loadlu_avx2_f16(f16 const* a0);
nsimd_avx2_vli64 nsimd_loadlu_avx2_i64(i64 const* a0);
nsimd_avx2_vli32 nsimd_loadlu_avx2_i32(i32 const* a0);
nsimd_avx2_vli16 nsimd_loadlu_avx2_i16(i16 const* a0);
nsimd_avx2_vli8 nsimd_loadlu_avx2_i8(i8 const* a0);
nsimd_avx2_vlu64 nsimd_loadlu_avx2_u64(u64 const* a0);
nsimd_avx2_vlu32 nsimd_loadlu_avx2_u32(u32 const* a0);
nsimd_avx2_vlu16 nsimd_loadlu_avx2_u16(u16 const* a0);
nsimd_avx2_vlu8 nsimd_loadlu_avx2_u8(u8 const* a0);
```

### AVX512_KNL

```c
nsimd_avx512_knl_vlf64 nsimd_loadlu_avx512_knl_f64(f64 const* a0);
nsimd_avx512_knl_vlf32 nsimd_loadlu_avx512_knl_f32(f32 const* a0);
nsimd_avx512_knl_vlf16 nsimd_loadlu_avx512_knl_f16(f16 const* a0);
nsimd_avx512_knl_vli64 nsimd_loadlu_avx512_knl_i64(i64 const* a0);
nsimd_avx512_knl_vli32 nsimd_loadlu_avx512_knl_i32(i32 const* a0);
nsimd_avx512_knl_vli16 nsimd_loadlu_avx512_knl_i16(i16 const* a0);
nsimd_avx512_knl_vli8 nsimd_loadlu_avx512_knl_i8(i8 const* a0);
nsimd_avx512_knl_vlu64 nsimd_loadlu_avx512_knl_u64(u64 const* a0);
nsimd_avx512_knl_vlu32 nsimd_loadlu_avx512_knl_u32(u32 const* a0);
nsimd_avx512_knl_vlu16 nsimd_loadlu_avx512_knl_u16(u16 const* a0);
nsimd_avx512_knl_vlu8 nsimd_loadlu_avx512_knl_u8(u8 const* a0);
```

### AVX

```c
nsimd_avx_vlf64 nsimd_loadlu_avx_f64(f64 const* a0);
nsimd_avx_vlf32 nsimd_loadlu_avx_f32(f32 const* a0);
nsimd_avx_vlf16 nsimd_loadlu_avx_f16(f16 const* a0);
nsimd_avx_vli64 nsimd_loadlu_avx_i64(i64 const* a0);
nsimd_avx_vli32 nsimd_loadlu_avx_i32(i32 const* a0);
nsimd_avx_vli16 nsimd_loadlu_avx_i16(i16 const* a0);
nsimd_avx_vli8 nsimd_loadlu_avx_i8(i8 const* a0);
nsimd_avx_vlu64 nsimd_loadlu_avx_u64(u64 const* a0);
nsimd_avx_vlu32 nsimd_loadlu_avx_u32(u32 const* a0);
nsimd_avx_vlu16 nsimd_loadlu_avx_u16(u16 const* a0);
nsimd_avx_vlu8 nsimd_loadlu_avx_u8(u8 const* a0);
```

### AVX512_SKYLAKE

```c
nsimd_avx512_skylake_vlf64 nsimd_loadlu_avx512_skylake_f64(f64 const* a0);
nsimd_avx512_skylake_vlf32 nsimd_loadlu_avx512_skylake_f32(f32 const* a0);
nsimd_avx512_skylake_vlf16 nsimd_loadlu_avx512_skylake_f16(f16 const* a0);
nsimd_avx512_skylake_vli64 nsimd_loadlu_avx512_skylake_i64(i64 const* a0);
nsimd_avx512_skylake_vli32 nsimd_loadlu_avx512_skylake_i32(i32 const* a0);
nsimd_avx512_skylake_vli16 nsimd_loadlu_avx512_skylake_i16(i16 const* a0);
nsimd_avx512_skylake_vli8 nsimd_loadlu_avx512_skylake_i8(i8 const* a0);
nsimd_avx512_skylake_vlu64 nsimd_loadlu_avx512_skylake_u64(u64 const* a0);
nsimd_avx512_skylake_vlu32 nsimd_loadlu_avx512_skylake_u32(u32 const* a0);
nsimd_avx512_skylake_vlu16 nsimd_loadlu_avx512_skylake_u16(u16 const* a0);
nsimd_avx512_skylake_vlu8 nsimd_loadlu_avx512_skylake_u8(u8 const* a0);
```

### SVE

```c
nsimd_sve_vlf64 nsimd_loadlu_sve_f64(f64 const* a0);
nsimd_sve_vlf32 nsimd_loadlu_sve_f32(f32 const* a0);
nsimd_sve_vlf16 nsimd_loadlu_sve_f16(f16 const* a0);
nsimd_sve_vli64 nsimd_loadlu_sve_i64(i64 const* a0);
nsimd_sve_vli32 nsimd_loadlu_sve_i32(i32 const* a0);
nsimd_sve_vli16 nsimd_loadlu_sve_i16(i16 const* a0);
nsimd_sve_vli8 nsimd_loadlu_sve_i8(i8 const* a0);
nsimd_sve_vlu64 nsimd_loadlu_sve_u64(u64 const* a0);
nsimd_sve_vlu32 nsimd_loadlu_sve_u32(u32 const* a0);
nsimd_sve_vlu16 nsimd_loadlu_sve_u16(u16 const* a0);
nsimd_sve_vlu8 nsimd_loadlu_sve_u8(u8 const* a0);
```

### CPU

```c
nsimd_cpu_vlf64 nsimd_loadlu_cpu_f64(f64 const* a0);
nsimd_cpu_vlf32 nsimd_loadlu_cpu_f32(f32 const* a0);
nsimd_cpu_vlf16 nsimd_loadlu_cpu_f16(f16 const* a0);
nsimd_cpu_vli64 nsimd_loadlu_cpu_i64(i64 const* a0);
nsimd_cpu_vli32 nsimd_loadlu_cpu_i32(i32 const* a0);
nsimd_cpu_vli16 nsimd_loadlu_cpu_i16(i16 const* a0);
nsimd_cpu_vli8 nsimd_loadlu_cpu_i8(i8 const* a0);
nsimd_cpu_vlu64 nsimd_loadlu_cpu_u64(u64 const* a0);
nsimd_cpu_vlu32 nsimd_loadlu_cpu_u32(u32 const* a0);
nsimd_cpu_vlu16 nsimd_loadlu_cpu_u16(u16 const* a0);
nsimd_cpu_vlu8 nsimd_loadlu_cpu_u8(u8 const* a0);
```

### SSE2

```c
nsimd_sse2_vlf64 nsimd_loadlu_sse2_f64(f64 const* a0);
nsimd_sse2_vlf32 nsimd_loadlu_sse2_f32(f32 const* a0);
nsimd_sse2_vlf16 nsimd_loadlu_sse2_f16(f16 const* a0);
nsimd_sse2_vli64 nsimd_loadlu_sse2_i64(i64 const* a0);
nsimd_sse2_vli32 nsimd_loadlu_sse2_i32(i32 const* a0);
nsimd_sse2_vli16 nsimd_loadlu_sse2_i16(i16 const* a0);
nsimd_sse2_vli8 nsimd_loadlu_sse2_i8(i8 const* a0);
nsimd_sse2_vlu64 nsimd_loadlu_sse2_u64(u64 const* a0);
nsimd_sse2_vlu32 nsimd_loadlu_sse2_u32(u32 const* a0);
nsimd_sse2_vlu16 nsimd_loadlu_sse2_u16(u16 const* a0);
nsimd_sse2_vlu8 nsimd_loadlu_sse2_u8(u8 const* a0);
```

### AARCH64

```c
nsimd_aarch64_vlf64 nsimd_loadlu_aarch64_f64(f64 const* a0);
nsimd_aarch64_vlf32 nsimd_loadlu_aarch64_f32(f32 const* a0);
nsimd_aarch64_vlf16 nsimd_loadlu_aarch64_f16(f16 const* a0);
nsimd_aarch64_vli64 nsimd_loadlu_aarch64_i64(i64 const* a0);
nsimd_aarch64_vli32 nsimd_loadlu_aarch64_i32(i32 const* a0);
nsimd_aarch64_vli16 nsimd_loadlu_aarch64_i16(i16 const* a0);
nsimd_aarch64_vli8 nsimd_loadlu_aarch64_i8(i8 const* a0);
nsimd_aarch64_vlu64 nsimd_loadlu_aarch64_u64(u64 const* a0);
nsimd_aarch64_vlu32 nsimd_loadlu_aarch64_u32(u32 const* a0);
nsimd_aarch64_vlu16 nsimd_loadlu_aarch64_u16(u16 const* a0);
nsimd_aarch64_vlu8 nsimd_loadlu_aarch64_u8(u8 const* a0);
```

### SSE42

```c
nsimd_sse42_vlf64 nsimd_loadlu_sse42_f64(f64 const* a0);
nsimd_sse42_vlf32 nsimd_loadlu_sse42_f32(f32 const* a0);
nsimd_sse42_vlf16 nsimd_loadlu_sse42_f16(f16 const* a0);
nsimd_sse42_vli64 nsimd_loadlu_sse42_i64(i64 const* a0);
nsimd_sse42_vli32 nsimd_loadlu_sse42_i32(i32 const* a0);
nsimd_sse42_vli16 nsimd_loadlu_sse42_i16(i16 const* a0);
nsimd_sse42_vli8 nsimd_loadlu_sse42_i8(i8 const* a0);
nsimd_sse42_vlu64 nsimd_loadlu_sse42_u64(u64 const* a0);
nsimd_sse42_vlu32 nsimd_loadlu_sse42_u32(u32 const* a0);
nsimd_sse42_vlu16 nsimd_loadlu_sse42_u16(u16 const* a0);
nsimd_sse42_vlu8 nsimd_loadlu_sse42_u8(u8 const* a0);
```

## C++ base API (architecture specifics)

### NEON128

```c
nsimd_neon128_vlf64 loadlu(f64 const* a0, f64, neon128);
nsimd_neon128_vlf32 loadlu(f32 const* a0, f32, neon128);
nsimd_neon128_vlf16 loadlu(f16 const* a0, f16, neon128);
nsimd_neon128_vli64 loadlu(i64 const* a0, i64, neon128);
nsimd_neon128_vli32 loadlu(i32 const* a0, i32, neon128);
nsimd_neon128_vli16 loadlu(i16 const* a0, i16, neon128);
nsimd_neon128_vli8 loadlu(i8 const* a0, i8, neon128);
nsimd_neon128_vlu64 loadlu(u64 const* a0, u64, neon128);
nsimd_neon128_vlu32 loadlu(u32 const* a0, u32, neon128);
nsimd_neon128_vlu16 loadlu(u16 const* a0, u16, neon128);
nsimd_neon128_vlu8 loadlu(u8 const* a0, u8, neon128);
```

### AVX2

```c
nsimd_avx2_vlf64 loadlu(f64 const* a0, f64, avx2);
nsimd_avx2_vlf32 loadlu(f32 const* a0, f32, avx2);
nsimd_avx2_vlf16 loadlu(f16 const* a0, f16, avx2);
nsimd_avx2_vli64 loadlu(i64 const* a0, i64, avx2);
nsimd_avx2_vli32 loadlu(i32 const* a0, i32, avx2);
nsimd_avx2_vli16 loadlu(i16 const* a0, i16, avx2);
nsimd_avx2_vli8 loadlu(i8 const* a0, i8, avx2);
nsimd_avx2_vlu64 loadlu(u64 const* a0, u64, avx2);
nsimd_avx2_vlu32 loadlu(u32 const* a0, u32, avx2);
nsimd_avx2_vlu16 loadlu(u16 const* a0, u16, avx2);
nsimd_avx2_vlu8 loadlu(u8 const* a0, u8, avx2);
```

### AVX512_KNL

```c
nsimd_avx512_knl_vlf64 loadlu(f64 const* a0, f64, avx512_knl);
nsimd_avx512_knl_vlf32 loadlu(f32 const* a0, f32, avx512_knl);
nsimd_avx512_knl_vlf16 loadlu(f16 const* a0, f16, avx512_knl);
nsimd_avx512_knl_vli64 loadlu(i64 const* a0, i64, avx512_knl);
nsimd_avx512_knl_vli32 loadlu(i32 const* a0, i32, avx512_knl);
nsimd_avx512_knl_vli16 loadlu(i16 const* a0, i16, avx512_knl);
nsimd_avx512_knl_vli8 loadlu(i8 const* a0, i8, avx512_knl);
nsimd_avx512_knl_vlu64 loadlu(u64 const* a0, u64, avx512_knl);
nsimd_avx512_knl_vlu32 loadlu(u32 const* a0, u32, avx512_knl);
nsimd_avx512_knl_vlu16 loadlu(u16 const* a0, u16, avx512_knl);
nsimd_avx512_knl_vlu8 loadlu(u8 const* a0, u8, avx512_knl);
```

### AVX

```c
nsimd_avx_vlf64 loadlu(f64 const* a0, f64, avx);
nsimd_avx_vlf32 loadlu(f32 const* a0, f32, avx);
nsimd_avx_vlf16 loadlu(f16 const* a0, f16, avx);
nsimd_avx_vli64 loadlu(i64 const* a0, i64, avx);
nsimd_avx_vli32 loadlu(i32 const* a0, i32, avx);
nsimd_avx_vli16 loadlu(i16 const* a0, i16, avx);
nsimd_avx_vli8 loadlu(i8 const* a0, i8, avx);
nsimd_avx_vlu64 loadlu(u64 const* a0, u64, avx);
nsimd_avx_vlu32 loadlu(u32 const* a0, u32, avx);
nsimd_avx_vlu16 loadlu(u16 const* a0, u16, avx);
nsimd_avx_vlu8 loadlu(u8 const* a0, u8, avx);
```

### AVX512_SKYLAKE

```c
nsimd_avx512_skylake_vlf64 loadlu(f64 const* a0, f64, avx512_skylake);
nsimd_avx512_skylake_vlf32 loadlu(f32 const* a0, f32, avx512_skylake);
nsimd_avx512_skylake_vlf16 loadlu(f16 const* a0, f16, avx512_skylake);
nsimd_avx512_skylake_vli64 loadlu(i64 const* a0, i64, avx512_skylake);
nsimd_avx512_skylake_vli32 loadlu(i32 const* a0, i32, avx512_skylake);
nsimd_avx512_skylake_vli16 loadlu(i16 const* a0, i16, avx512_skylake);
nsimd_avx512_skylake_vli8 loadlu(i8 const* a0, i8, avx512_skylake);
nsimd_avx512_skylake_vlu64 loadlu(u64 const* a0, u64, avx512_skylake);
nsimd_avx512_skylake_vlu32 loadlu(u32 const* a0, u32, avx512_skylake);
nsimd_avx512_skylake_vlu16 loadlu(u16 const* a0, u16, avx512_skylake);
nsimd_avx512_skylake_vlu8 loadlu(u8 const* a0, u8, avx512_skylake);
```

### SVE

```c
nsimd_sve_vlf64 loadlu(f64 const* a0, f64, sve);
nsimd_sve_vlf32 loadlu(f32 const* a0, f32, sve);
nsimd_sve_vlf16 loadlu(f16 const* a0, f16, sve);
nsimd_sve_vli64 loadlu(i64 const* a0, i64, sve);
nsimd_sve_vli32 loadlu(i32 const* a0, i32, sve);
nsimd_sve_vli16 loadlu(i16 const* a0, i16, sve);
nsimd_sve_vli8 loadlu(i8 const* a0, i8, sve);
nsimd_sve_vlu64 loadlu(u64 const* a0, u64, sve);
nsimd_sve_vlu32 loadlu(u32 const* a0, u32, sve);
nsimd_sve_vlu16 loadlu(u16 const* a0, u16, sve);
nsimd_sve_vlu8 loadlu(u8 const* a0, u8, sve);
```

### CPU

```c
nsimd_cpu_vlf64 loadlu(f64 const* a0, f64, cpu);
nsimd_cpu_vlf32 loadlu(f32 const* a0, f32, cpu);
nsimd_cpu_vlf16 loadlu(f16 const* a0, f16, cpu);
nsimd_cpu_vli64 loadlu(i64 const* a0, i64, cpu);
nsimd_cpu_vli32 loadlu(i32 const* a0, i32, cpu);
nsimd_cpu_vli16 loadlu(i16 const* a0, i16, cpu);
nsimd_cpu_vli8 loadlu(i8 const* a0, i8, cpu);
nsimd_cpu_vlu64 loadlu(u64 const* a0, u64, cpu);
nsimd_cpu_vlu32 loadlu(u32 const* a0, u32, cpu);
nsimd_cpu_vlu16 loadlu(u16 const* a0, u16, cpu);
nsimd_cpu_vlu8 loadlu(u8 const* a0, u8, cpu);
```

### SSE2

```c
nsimd_sse2_vlf64 loadlu(f64 const* a0, f64, sse2);
nsimd_sse2_vlf32 loadlu(f32 const* a0, f32, sse2);
nsimd_sse2_vlf16 loadlu(f16 const* a0, f16, sse2);
nsimd_sse2_vli64 loadlu(i64 const* a0, i64, sse2);
nsimd_sse2_vli32 loadlu(i32 const* a0, i32, sse2);
nsimd_sse2_vli16 loadlu(i16 const* a0, i16, sse2);
nsimd_sse2_vli8 loadlu(i8 const* a0, i8, sse2);
nsimd_sse2_vlu64 loadlu(u64 const* a0, u64, sse2);
nsimd_sse2_vlu32 loadlu(u32 const* a0, u32, sse2);
nsimd_sse2_vlu16 loadlu(u16 const* a0, u16, sse2);
nsimd_sse2_vlu8 loadlu(u8 const* a0, u8, sse2);
```

### AARCH64

```c
nsimd_aarch64_vlf64 loadlu(f64 const* a0, f64, aarch64);
nsimd_aarch64_vlf32 loadlu(f32 const* a0, f32, aarch64);
nsimd_aarch64_vlf16 loadlu(f16 const* a0, f16, aarch64);
nsimd_aarch64_vli64 loadlu(i64 const* a0, i64, aarch64);
nsimd_aarch64_vli32 loadlu(i32 const* a0, i32, aarch64);
nsimd_aarch64_vli16 loadlu(i16 const* a0, i16, aarch64);
nsimd_aarch64_vli8 loadlu(i8 const* a0, i8, aarch64);
nsimd_aarch64_vlu64 loadlu(u64 const* a0, u64, aarch64);
nsimd_aarch64_vlu32 loadlu(u32 const* a0, u32, aarch64);
nsimd_aarch64_vlu16 loadlu(u16 const* a0, u16, aarch64);
nsimd_aarch64_vlu8 loadlu(u8 const* a0, u8, aarch64);
```

### SSE42

```c
nsimd_sse42_vlf64 loadlu(f64 const* a0, f64, sse42);
nsimd_sse42_vlf32 loadlu(f32 const* a0, f32, sse42);
nsimd_sse42_vlf16 loadlu(f16 const* a0, f16, sse42);
nsimd_sse42_vli64 loadlu(i64 const* a0, i64, sse42);
nsimd_sse42_vli32 loadlu(i32 const* a0, i32, sse42);
nsimd_sse42_vli16 loadlu(i16 const* a0, i16, sse42);
nsimd_sse42_vli8 loadlu(i8 const* a0, i8, sse42);
nsimd_sse42_vlu64 loadlu(u64 const* a0, u64, sse42);
nsimd_sse42_vlu32 loadlu(u32 const* a0, u32, sse42);
nsimd_sse42_vlu16 loadlu(u16 const* a0, u16, sse42);
nsimd_sse42_vlu8 loadlu(u8 const* a0, u8, sse42);
```