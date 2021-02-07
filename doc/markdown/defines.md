# Defines provided by NSIMD

NSIMD uses macros (not function macros) that we call defines to make choices
in its code at copmile time. Most of them can be of use to the end-user so
we list them here.

## Compiler detection

The compiler detection is automatically done by NSIMD as it is relatively
easy.

| Define              | Compiler                                          |
|---------------------|---------------------------------------------------|
| `NSIMD_IS_MSVC`     | Microsoft Visual C++                              |
| `NSIMD_IS_HIPCC`    | ROCm HIP compiler (warning, see below)            |
| `NSIMD_IS_NVCC`     | NVIDIA CUDA Compiler                              |
| `NSIMD_IS_ICC`      | Intel C++ Compiler                                |
| `NSIMD_IS_CLANG`    | Clang/LLVM                                        |
| `NSIMD_IS_GCC`      | GNU Compiler Collection                           |
| `NSIMD_IS_FCC`      | Fujitsu compiler                                  |

**Warning**: some HIP versions do not declare themselves at all so it
impossible to find out that HIP is the compiler. As HIP is based on clang,
without help NSIMD will detect Clang. It is up to the end-user to compile
with `-D__HIPCC__` for NSIMD to detect HIP.

Note that we do support the Armclang C and C++ compilers but for NSIMD there
is no need to have code different from Clang's specific code so we do no
provide a macro to detect this compiler in particular.

Note also that two of the above macros can be defined at the same time. This
happens typically when compiling for a device. For example when compiling for
NVIDIA CUDA with nvcc both `NSIMD_IS_NVCC` and `NSIMD_IS_GCC` (when the host
compiler is GCC).

## Compilation environment and contants

| Define            | Description           | Possible values                 |
|-------------------|-----------------------|---------------------------------|
| `NSIMD_C`         | C version             | 1989, 1999, 2011                |
| `NSIMD_CXX`       | C++ version           | 1998, 2011, 2014, 2017, 2020    |
| `NSIMD_WORD_SIZE` | Machine word size     | 32, 64                          |
| `NSIMD_U8_MIN`    | Minimum value for u8  | 0                               |
| `NSIMD_U8_MAX`    | Maximum value for u8  | 255                             |
| `NSIMD_I8_MIN`    | Minimum value for i8  | -128                            |
| `NSIMD_I8_MAX`    | Maximum value for i8  | 127                             |
| `NSIMD_U16_MIN`   | Minimum value for u16 | 0                               |
| `NSIMD_U16_MAX`   | Maximum value for u16 | 65535                           |
| `NSIMD_I16_MIN`   | Minimum value for i16 | -32768                          |
| `NSIMD_I16_MAX`   | Maximum value for i16 | 32767                           |
| `NSIMD_U32_MIN`   | Minimum value for u32 | 0                               |
| `NSIMD_U32_MAX`   | Maximum value for u32 | 4294967295                      |
| `NSIMD_I32_MIN`   | Minimum value for i32 | -2147483648                     |
| `NSIMD_I32_MAX`   | Maximum value for i32 | 2147483647                      |
| `NSIMD_U64_MIN`   | Minimum value for u64 | 0                               |
| `NSIMD_U64_MAX`   | Maximum value for u64 | 18446744073709551615            |
| `NSIMD_I64_MIN`   | Minimum value for i64 | -9223372036854775808            |
| `NSIMD_I64_MAX`   | Maximum value for i64 | 9223372036854775807             |
| `NSIMD_DLLSPEC`   | (Windows) DLL storage-class information | `__declspec(dllexport)` or `__declspec(dllimport)` |
| `NSIMD_DLLSPEC`   | (Unix) storage-class information        | `extern` or nothing |
| `NSIMD_C_LINKAGE_FOR_F16` | Indicate whether functions involving f16 have C linkage | defined or not |

## Targeted architecture detection

Contrary to the compiler detection, the targeted architecture is not done
autoamtically by NSIMD as is really hard and some compilers do not provide
the necessary informations. So in order to have a consistent way of targeting
an architecture this is up to the end-user to specify it using one of the
following defines.

| Define                 | Targeted architecture                             |
|------------------------|---------------------------------------------------|
| `NSIMD_CPU`            | Generic, no SIMD, emulation                       |
| `NSIMD_SSE2`           | Intel SSE2                                        |
| `NSIMD_SSE42`          | Intel SSE4.2                                      |
| `NSIMD_AVX`            | Intel AVX                                         |
| `NSIMD_AVX2`           | Intel AVX2                                        |
| `NSIMD_AVX512_KNL`     | Intel AVX-512 as found on KNLs                    |
| `NSIMD_AVX512_SKYLAKE` | Intel AVX-512 as found on Xeon Skylake            |
| `NSIMD_NEON128`        | Arm NEON 128 bits as found on 32-bits Arm chips   |
| `NSIMD_AARCH64`        | Arm NEON 128 bits as found on 64-bits Arm chips   |
| `NSIMD_SVE`            | Arm SVE (length agnostic)                         |
| `NSIMD_SVE128`         | Arm SVE (size known at compilation to 128 bits)   |
| `NSIMD_SVE256`         | Arm SVE (size known at compilation to 256 bits)   |
| `NSIMD_SVE512`         | Arm SVE (size known at compilation to 512 bits)   |
| `NSIMD_SVE1024`        | Arm SVE (size known at compilation to 1024 bits)  |
| `NSIMD_SVE2048`        | Arm SVE (size known at compilation to 2048 bits)  |
| `NSIMD_CUDA`           | Nvidia CUDA                                       |
| `NSIMD_ROCM`           | AMD ROCm architectures                            |
| `NSIMD_FP16`           | Architecture supports natively IEEE float16       |
| `NSIMD_FMA`            | Architecture supports natively FMAs               |

## Targeted architecture constants

| Define                | Description                                        |
|-----------------------|----------------------------------------------------|
| `NSIMD_NB_REGISTERS`  | Number of SIMD registers                           |
| `NSIMD_MAX_LEN_BIT`   | Maximum number of bits in a SIMD register          |
| `NSIMD_MAX_LEN_i8`    | Maximum number of i8's in a SIMD register          |
| `NSIMD_MAX_LEN_u8`    | Maximum number of u8's in a SIMD register          |
| `NSIMD_MAX_LEN_i16`   | Maximum number of i16's in a SIMD register         |
| `NSIMD_MAX_LEN_u16`   | Maximum number of u16's in a SIMD register         |
| `NSIMD_MAX_LEN_i32`   | Maximum number of i32's in a SIMD register         |
| `NSIMD_MAX_LEN_u32`   | Maximum number of u32's in a SIMD register         |
| `NSIMD_MAX_LEN_i64`   | Maximum number of i64's in a SIMD register         |
| `NSIMD_MAX_LEN_u64`   | Maximum number of u64's in a SIMD register         |

NSIMD provides a mean to write generic code by using the `NSIMD_MAX_LEN` macros
whose argument is one of { i8, u8, i16, u16, i32, u32, i64, u64 }.

```c++
#define T ??? // to be defined as a base type

int main(void) {
  T buf[NSIMD_MAX_LEN(T)]; // an array of T's for loading/storing
  ...
  return 0;
}
```

## Other useful macros

NSIMD provides macros to concatenate blobs so that generic programming in pure
C is possible.

- `#define NSIMD_PP_CAT_2(a, b)` concatenates `a` and `b`.
- `#define NSIMD_PP_CAT_3(a, b, c)` concatenates `a`, `b` and `c`.
- `#define NSIMD_PP_CAT_4(a, b, c, d)` concatenates `a`, `b`, `c` and `d`.
- `#define NSIMD_PP_CAT_5(a, b, c, d, e)` concatenates `a`, `b`, `c`, `d` and
  `e`.
- `#define NSIMD_PP_CAT_6(a, b, c, d, e, f)` concatenates `a`, `b`, `c`, `d`,
  `e` and `f`.
