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

# Vector length

## Description

Returns the vector length of the argument. Defined over $$.

## C base API (generic)

```c
#define vlen(type)
#define vlen_e(type, simd_ext)
```

## C++ base API (generic)

```c++
template <typename T> int len(T);
```

## C++ advanced API

```c++
template <typename T, typename SimdExt> int len(packl<T, 1, SimdExt> const&);
template <typename T, int N, typename SimdExt> int len(packl<T, N, SimdExt> const&);
template <typename SimdVector> SimdVector len();
```

## C base API (architecture specifics)

### NEON128

```c
int nsimd_len_neon128_f64(void);
int nsimd_len_neon128_f32(void);
int nsimd_len_neon128_f16(void);
int nsimd_len_neon128_i64(void);
int nsimd_len_neon128_i32(void);
int nsimd_len_neon128_i16(void);
int nsimd_len_neon128_i8(void);
int nsimd_len_neon128_u64(void);
int nsimd_len_neon128_u32(void);
int nsimd_len_neon128_u16(void);
int nsimd_len_neon128_u8(void);
```

### AVX2

```c
int nsimd_len_avx2_f64(void);
int nsimd_len_avx2_f32(void);
int nsimd_len_avx2_f16(void);
int nsimd_len_avx2_i64(void);
int nsimd_len_avx2_i32(void);
int nsimd_len_avx2_i16(void);
int nsimd_len_avx2_i8(void);
int nsimd_len_avx2_u64(void);
int nsimd_len_avx2_u32(void);
int nsimd_len_avx2_u16(void);
int nsimd_len_avx2_u8(void);
```

### AVX512_KNL

```c
int nsimd_len_avx512_knl_f64(void);
int nsimd_len_avx512_knl_f32(void);
int nsimd_len_avx512_knl_f16(void);
int nsimd_len_avx512_knl_i64(void);
int nsimd_len_avx512_knl_i32(void);
int nsimd_len_avx512_knl_i16(void);
int nsimd_len_avx512_knl_i8(void);
int nsimd_len_avx512_knl_u64(void);
int nsimd_len_avx512_knl_u32(void);
int nsimd_len_avx512_knl_u16(void);
int nsimd_len_avx512_knl_u8(void);
```

### AVX

```c
int nsimd_len_avx_f64(void);
int nsimd_len_avx_f32(void);
int nsimd_len_avx_f16(void);
int nsimd_len_avx_i64(void);
int nsimd_len_avx_i32(void);
int nsimd_len_avx_i16(void);
int nsimd_len_avx_i8(void);
int nsimd_len_avx_u64(void);
int nsimd_len_avx_u32(void);
int nsimd_len_avx_u16(void);
int nsimd_len_avx_u8(void);
```

### AVX512_SKYLAKE

```c
int nsimd_len_avx512_skylake_f64(void);
int nsimd_len_avx512_skylake_f32(void);
int nsimd_len_avx512_skylake_f16(void);
int nsimd_len_avx512_skylake_i64(void);
int nsimd_len_avx512_skylake_i32(void);
int nsimd_len_avx512_skylake_i16(void);
int nsimd_len_avx512_skylake_i8(void);
int nsimd_len_avx512_skylake_u64(void);
int nsimd_len_avx512_skylake_u32(void);
int nsimd_len_avx512_skylake_u16(void);
int nsimd_len_avx512_skylake_u8(void);
```

### SVE

```c
int nsimd_len_sve_f64(void);
int nsimd_len_sve_f32(void);
int nsimd_len_sve_f16(void);
int nsimd_len_sve_i64(void);
int nsimd_len_sve_i32(void);
int nsimd_len_sve_i16(void);
int nsimd_len_sve_i8(void);
int nsimd_len_sve_u64(void);
int nsimd_len_sve_u32(void);
int nsimd_len_sve_u16(void);
int nsimd_len_sve_u8(void);
```

### CPU

```c
int nsimd_len_cpu_f64(void);
int nsimd_len_cpu_f32(void);
int nsimd_len_cpu_f16(void);
int nsimd_len_cpu_i64(void);
int nsimd_len_cpu_i32(void);
int nsimd_len_cpu_i16(void);
int nsimd_len_cpu_i8(void);
int nsimd_len_cpu_u64(void);
int nsimd_len_cpu_u32(void);
int nsimd_len_cpu_u16(void);
int nsimd_len_cpu_u8(void);
```

### SSE2

```c
int nsimd_len_sse2_f64(void);
int nsimd_len_sse2_f32(void);
int nsimd_len_sse2_f16(void);
int nsimd_len_sse2_i64(void);
int nsimd_len_sse2_i32(void);
int nsimd_len_sse2_i16(void);
int nsimd_len_sse2_i8(void);
int nsimd_len_sse2_u64(void);
int nsimd_len_sse2_u32(void);
int nsimd_len_sse2_u16(void);
int nsimd_len_sse2_u8(void);
```

### AARCH64

```c
int nsimd_len_aarch64_f64(void);
int nsimd_len_aarch64_f32(void);
int nsimd_len_aarch64_f16(void);
int nsimd_len_aarch64_i64(void);
int nsimd_len_aarch64_i32(void);
int nsimd_len_aarch64_i16(void);
int nsimd_len_aarch64_i8(void);
int nsimd_len_aarch64_u64(void);
int nsimd_len_aarch64_u32(void);
int nsimd_len_aarch64_u16(void);
int nsimd_len_aarch64_u8(void);
```

### SSE42

```c
int nsimd_len_sse42_f64(void);
int nsimd_len_sse42_f32(void);
int nsimd_len_sse42_f16(void);
int nsimd_len_sse42_i64(void);
int nsimd_len_sse42_i32(void);
int nsimd_len_sse42_i16(void);
int nsimd_len_sse42_i8(void);
int nsimd_len_sse42_u64(void);
int nsimd_len_sse42_u32(void);
int nsimd_len_sse42_u16(void);
int nsimd_len_sse42_u8(void);
```

## C++ base API (architecture specifics)

### NEON128

```c
int len(f64, neon128);
int len(f32, neon128);
int len(f16, neon128);
int len(i64, neon128);
int len(i32, neon128);
int len(i16, neon128);
int len(i8, neon128);
int len(u64, neon128);
int len(u32, neon128);
int len(u16, neon128);
int len(u8, neon128);
```

### AVX2

```c
int len(f64, avx2);
int len(f32, avx2);
int len(f16, avx2);
int len(i64, avx2);
int len(i32, avx2);
int len(i16, avx2);
int len(i8, avx2);
int len(u64, avx2);
int len(u32, avx2);
int len(u16, avx2);
int len(u8, avx2);
```

### AVX512_KNL

```c
int len(f64, avx512_knl);
int len(f32, avx512_knl);
int len(f16, avx512_knl);
int len(i64, avx512_knl);
int len(i32, avx512_knl);
int len(i16, avx512_knl);
int len(i8, avx512_knl);
int len(u64, avx512_knl);
int len(u32, avx512_knl);
int len(u16, avx512_knl);
int len(u8, avx512_knl);
```

### AVX

```c
int len(f64, avx);
int len(f32, avx);
int len(f16, avx);
int len(i64, avx);
int len(i32, avx);
int len(i16, avx);
int len(i8, avx);
int len(u64, avx);
int len(u32, avx);
int len(u16, avx);
int len(u8, avx);
```

### AVX512_SKYLAKE

```c
int len(f64, avx512_skylake);
int len(f32, avx512_skylake);
int len(f16, avx512_skylake);
int len(i64, avx512_skylake);
int len(i32, avx512_skylake);
int len(i16, avx512_skylake);
int len(i8, avx512_skylake);
int len(u64, avx512_skylake);
int len(u32, avx512_skylake);
int len(u16, avx512_skylake);
int len(u8, avx512_skylake);
```

### SVE

```c
int len(f64, sve);
int len(f32, sve);
int len(f16, sve);
int len(i64, sve);
int len(i32, sve);
int len(i16, sve);
int len(i8, sve);
int len(u64, sve);
int len(u32, sve);
int len(u16, sve);
int len(u8, sve);
```

### CPU

```c
int len(f64, cpu);
int len(f32, cpu);
int len(f16, cpu);
int len(i64, cpu);
int len(i32, cpu);
int len(i16, cpu);
int len(i8, cpu);
int len(u64, cpu);
int len(u32, cpu);
int len(u16, cpu);
int len(u8, cpu);
```

### SSE2

```c
int len(f64, sse2);
int len(f32, sse2);
int len(f16, sse2);
int len(i64, sse2);
int len(i32, sse2);
int len(i16, sse2);
int len(i8, sse2);
int len(u64, sse2);
int len(u32, sse2);
int len(u16, sse2);
int len(u8, sse2);
```

### AARCH64

```c
int len(f64, aarch64);
int len(f32, aarch64);
int len(f16, aarch64);
int len(i64, aarch64);
int len(i32, aarch64);
int len(i16, aarch64);
int len(i8, aarch64);
int len(u64, aarch64);
int len(u32, aarch64);
int len(u16, aarch64);
int len(u8, aarch64);
```

### SSE42

```c
int len(f64, sse42);
int len(f32, sse42);
int len(f16, sse42);
int len(i64, sse42);
int len(i32, sse42);
int len(i16, sse42);
int len(i8, sse42);
int len(u64, sse42);
int len(u32, sse42);
int len(u16, sse42);
int len(u8, sse42);
```