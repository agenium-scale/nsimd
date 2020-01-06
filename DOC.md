## NSIMD scalar types

Their names follows the following pattern: `Sxx` where

- `S` is `i` for signed integers, `u` for unsigned integer and `f` for
  floatting point number.
- `xx` is the number of bits taken to represent the number.

Full list of scalar types:

- `f64`
- `f32`
- `f16`
- `i64`
- `i32`
- `i16`
- `i8`
- `u64`
- `u32`
- `u16`
- `u8`


## NSIMD SIMD vector types

Their names follows the following pattern: `vSCALAR` where `SCALAR` is a
one of scalar type listed above. For example `vi8` means a SIMD vector
containing `i8`'s.

Full list of SIMD vector types:

- `vf64`
- `vf32`
- `vf16`
- `vi64`
- `vi32`
- `vi16`
- `vi8`
- `vu64`
- `vu32`
- `vu16`
- `vu8`


## C/C++ base APIs

These come automatically when you include `nsimd/nsimd.h`. You do *not* need
to include a header file for having a function. In NSIMD, we call a platform
an architecture e.g. Intel, ARM, POWERPC. We call SIMD extension a set of
low-level functions and types provided to access a given SIDM extension.
Examples include SSE2, SSE42, AVX, ...

Here is a list of supported platforms and their corresponding SIMD extensions.

- Platform `x86`
  - `sse2`
  - `sse42`
  - `avx`
  - `avx2`
  - `avx512_knl`
  - `avx512_skylake`
- Platform `cpu`
  - `cpu`
- Platform `arm`
  - `neon128`
  - `aarch64`
  - `sve`

Each simd extension has its own set of SIMD types and functions. Types follow
the following pattern: `nsimd_SIMDEXT_vSCALAR` where

- `SIMDEXT` is the SIMD extensions.
- `SCALAR` is one of scalar types listed above.

There are also logical types associated to each SIMD vector type. These types
are used to represent the result of a comparison of SIMD vectors. They are
usually bit masks. Their name follow the following pattern:
`nsimd_SIMDEXT_vlSCALAR` where

- `SIMDEXT` is the SIMD extensions.
- `SCALAR` is one of scalar types listed above.

Note 1: Platform `cpu` is scalar fallback when no SIMD extension has been
specified.

Note 2: as all SIMD extensions of all platforms are different there is no
need to put the name of the platform in each identifier.

Function names follow the following pattern: `nsimd_SIMDEXT_FUNCNAME_SCALAR`
where

- `SIMDEXT` is the SIMD extensions.
- `FUNCNAME` is the name of a function e.g. `add` or `sub`.
- `SCALAR` is one of scalar types listed above.

### Generic identifier

In C, genericity is achieved using macros.

- `vec(SCALAR)` represents the SIMD vector type containing SCALAR elements.
  SCALAR must be one of scalar types listed above.
- `vecl(SCALAR)` represents the SIMD vector of logicals type containing SCALAR
  elements. SCALAR must be one of scalar types listed above.
- `vec_e(SCALAR)` represents the SIMD vector type containing SCALAR elements.
  SCALAR must be one of scalar types listed above.
- `vecl_e(SCALAR)` represents the SIMD vector of logicals type containing
  SCALAR elements. SCALAR must be one of scalar types listed above.
- `vFUNCNAME` is the macro name to access the function FUNCNAME e.g. `vadd`,
  `vsub`.
- `vFUNCNAME_e` is the macro name to access the function FUNCNAME e.g.
  `vadd_e`, `vsub_e`.

In C++98 and C++03, type traits are available.

- `nsimd::simd_traits<SCALAR, SIMDEXT>::vector` is the SIMD vector type for
  platform SIMDEXT containing SCALAR elements. SIMDEXT is one of SIMD
  extension listed above, SCALAR is one of scalar type listed above.
- `nsimd::simd_traits<SCALAR, SIMDEXT>::vectorl` is the SIMD vector of logicals
  type for platform SIMDEXT containing SCALAR elements. SIMDEXT is one of
  SIMD extensions listed above, SCALAR is one of scalar type listed above.

In C++11 and beyond, type traits are still available but typedefs are also
provided.

- `nsimd::vector<SCALAR, SIMDEXT>` is a typedef to
  `nsimd::simd_traits<SCALAR, SIMDEXT>::vector`.
- `nsimd::vectorl<SCALAR, SIMDEXT>` is a typedef to
  `nsimd::simd_traits<SCALAR, SIMDEXT>::vectorl`.

Note that all macro and functions available in plain C are still available in
C++.

### List of functions available for manipulation of SIMD vectors

For each FUNCNAME a C function (also available in C++)
named `nsimd_SIMDEXT_FUNCNAME_SCALAR` is available for each SCALAR type unless
specified otherwise.

For each FUNCNAME, a C macro (also available in C++) named `vFUNCNAME` is
available and takes as its last argument a SCALAR type.

For each FUNCNAME, a C macro (also available in C++) named `vFUNCNAME_a` is
available and takes as its two last argument a SCALAR type and a SIMDEXT.

For each FUNCNAME, a C++ function in namespace `nsimd` named `FUNCNAME` is
available. It takes as its last argument the SCALAR type and can optionnally
take the SIMDEXT as its last last argument.

For example, for the addition of two SIMD vectors `a` and `b` here are the
possibilities:

    c = nsimd_add_avx_f32(a, b); // use AVX
    c = nsimd::add(a, b, f32()); // use detected SIMDEXT
    c = nsimd::add(a, b, f32(), avx()); // force AVX even if detected SIMDEXT is not AVX
    c = vadd(a, b, f32); // use detected SIMDEXT
    c = vadd_e(a, b, f32, avx); // force AVX even if detected SIMDEXT is not AVX

Here is a list of available FUNCNAME.

- `int len();`
- `vSCALAR set1(SCALAR a0);`
  a0 ∈ ℝ
- `vSCALAR loadu(SCALAR const* a0);`
  a0 ∈ ℝ
- `vSCALARx2 load2u(SCALAR const* a0);`
  a0 ∈ ℝ
- `vSCALARx3 load3u(SCALAR const* a0);`
  a0 ∈ ℝ
- `vSCALARx4 load4u(SCALAR const* a0);`
  a0 ∈ ℝ
- `vSCALAR loada(SCALAR const* a0);`
  a0 ∈ ℝ
- `vSCALARx2 load2a(SCALAR const* a0);`
  a0 ∈ ℝ
- `vSCALARx3 load3a(SCALAR const* a0);`
  a0 ∈ ℝ
- `vSCALARx4 load4a(SCALAR const* a0);`
  a0 ∈ ℝ
- `vlSCALAR loadlu(SCALAR const* a0);`
  a0 ∈ ℝ
- `vlSCALAR loadla(SCALAR const* a0);`
  a0 ∈ ℝ
- `void storeu(SCALAR* a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ
- `void store2u(SCALAR* a0, vSCALAR a1, vSCALAR a2);`
  (a0, a1, a2) ∈ ℝ × ℝ
- `void store3u(SCALAR* a0, vSCALAR a1, vSCALAR a2, vSCALAR a3);`
  (a0, a1, a2, a3) ∈ ℝ × ℝ × ℝ
- `void store4u(SCALAR* a0, vSCALAR a1, vSCALAR a2, vSCALAR a3, vSCALAR a4);`
  (a0, a1, a2, a3, a4) ∈ ℝ × ℝ × ℝ × ℝ
- `void storea(SCALAR* a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ
- `void store2a(SCALAR* a0, vSCALAR a1, vSCALAR a2);`
  (a0, a1, a2) ∈ ℝ × ℝ
- `void store3a(SCALAR* a0, vSCALAR a1, vSCALAR a2, vSCALAR a3);`
  (a0, a1, a2, a3) ∈ ℝ × ℝ × ℝ
- `void store4a(SCALAR* a0, vSCALAR a1, vSCALAR a2, vSCALAR a3, vSCALAR a4);`
  (a0, a1, a2, a3, a4) ∈ ℝ × ℝ × ℝ × ℝ
- `void storelu(SCALAR* a0, vlSCALAR a1);`
  (a0, a1) ∈ ℝ
- `void storela(SCALAR* a0, vlSCALAR a1);`
  (a0, a1) ∈ ℝ
- `vSCALAR orb(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vSCALAR andb(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vSCALAR andnotb(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vSCALAR notb(vSCALAR a0);`
  a0 ∈ ℝ
- `vSCALAR xorb(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vlSCALAR orl(vlSCALAR a0, vlSCALAR a1);`
  (a0, a1) ∈ 𝔹 × 𝔹
- `vlSCALAR andl(vlSCALAR a0, vlSCALAR a1);`
  (a0, a1) ∈ 𝔹 × 𝔹
- `vlSCALAR andnotl(vlSCALAR a0, vlSCALAR a1);`
  (a0, a1) ∈ 𝔹 × 𝔹
- `vlSCALAR xorl(vlSCALAR a0, vlSCALAR a1);`
  (a0, a1) ∈ 𝔹 × 𝔹
- `vlSCALAR notl(vlSCALAR a0);`
  a0 ∈ 𝔹
- `vSCALAR add(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vSCALAR sub(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `SCALAR addv(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR mul(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vSCALAR div(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ∖{0}
- `vSCALAR neg(vSCALAR a0);`
  a0 ∈ ℝ
- `vSCALAR min(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vSCALAR max(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vSCALAR shr(vSCALAR a0, int a1);`
  (a0, a1) ∈ ℝ × ℕ
  Only available for i64, i32, i16, i8, u64, u32, u16, u8
- `vSCALAR shl(vSCALAR a0, int a1);`
  (a0, a1) ∈ ℝ × ℕ
  Only available for i64, i32, i16, i8, u64, u32, u16, u8
- `vlSCALAR eq(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vlSCALAR ne(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vlSCALAR gt(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vlSCALAR ge(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vlSCALAR lt(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vlSCALAR le(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
- `vSCALAR if_else1(vlSCALAR a0, vSCALAR a1, vSCALAR a2);`
  (a0, a1, a2) ∈ 𝔹 × ℝ × ℝ
- `vSCALAR abs(vSCALAR a0);`
  a0 ∈ ℝ
- `vSCALAR fma(vSCALAR a0, vSCALAR a1, vSCALAR a2);`
  (a0, a1, a2) ∈ ℝ × ℝ × ℝ
- `vSCALAR fnma(vSCALAR a0, vSCALAR a1, vSCALAR a2);`
  (a0, a1, a2) ∈ ℝ × ℝ × ℝ
- `vSCALAR fms(vSCALAR a0, vSCALAR a1, vSCALAR a2);`
  (a0, a1, a2) ∈ ℝ × ℝ × ℝ
- `vSCALAR fnms(vSCALAR a0, vSCALAR a1, vSCALAR a2);`
  (a0, a1, a2) ∈ ℝ × ℝ × ℝ
- `vSCALAR ceil(vSCALAR a0);`
  a0 ∈ ℝ
- `vSCALAR floor(vSCALAR a0);`
  a0 ∈ ℝ
- `vSCALAR trunc(vSCALAR a0);`
  a0 ∈ ℝ
- `vSCALAR round_to_even(vSCALAR a0);`
  a0 ∈ ℝ
- `int all(vlSCALAR a0);`
  a0 ∈ 𝔹
- `int any(vlSCALAR a0);`
  a0 ∈ 𝔹
- `int nbtrue(vlSCALAR a0);`
  a0 ∈ 𝔹
- `vSCALAR reinterpret(vSCALAR a0);`
  a0 ∈ ℝ
- `vlSCALAR reinterpretl(vlSCALAR a0);`
  a0 ∈ 𝔹
- `vSCALAR cvt(vSCALAR a0);`
  a0 ∈ ℝ
- `vSCALARx2 upcvt(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for i8, u8, i16, u16, f16, i32, u32, f32
- `vSCALAR downcvt(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ
  Only available for i16, u16, f16, i32, u32, f32, i64, u64, f64
- `vSCALAR rec(vSCALAR a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `vSCALAR rec11(vSCALAR a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `vSCALAR sqrt(vSCALAR a0);`
  a0 ∈ [0, +∞)
  Only available for f64, f32, f16
- `vSCALAR rsqrt11(vSCALAR a0);`
  a0 ∈ [0, +∞)
  Only available for f64, f32, f16
- `vSCALAR ziplo(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ
- `vSCALAR ziphi(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ
- `vSCALARx2 zip(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ
- `vSCALAR unziplo(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ
- `vSCALAR unziphi(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ
- `vSCALARx2 unzip(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ
- `vSCALAR to_mask(vlSCALAR a0);`
  a0 ∈ ℝ
- `vlSCALAR to_logical(vSCALAR a0);`
  a0 ∈ ℝ
- `vSCALAR cos(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR cosd(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR cospi(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR acos(vSCALAR a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `vSCALAR acosd(vSCALAR a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `vSCALAR acospi(vSCALAR a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `vSCALAR cosh(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR acosh(vSCALAR a0);`
  a0 ∈ (1, +∞)
  Only available for f64, f32, f16
- `vSCALAR sin(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR sinpi(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR sind(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR asin(vSCALAR a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `vSCALAR asind(vSCALAR a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `vSCALAR asinpi(vSCALAR a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `vSCALAR sinh(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR asinh(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR tan(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅(z+0.5)}
  Only available for f64, f32, f16
- `vSCALAR tand(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅(z+0.5)}
  Only available for f64, f32, f16
- `vSCALAR tanpi(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅(z+0.5)}
  Only available for f64, f32, f16
- `vSCALAR atan(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR atand(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR atanpi(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR tanh(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR atanh(vSCALAR a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `vSCALAR sinc(vSCALAR a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `vSCALAR sincpi(vSCALAR a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `vSCALAR sinhc(vSCALAR a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `vSCALAR atan2(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
  Only available for f64, f32, f16
- `vSCALAR atan2d(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
  Only available for f64, f32, f16
- `vSCALAR atan2pi(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
  Only available for f64, f32, f16
- `vSCALAR cot(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `vSCALAR cotd(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `vSCALAR cotpi(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `vSCALAR acot(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR acotd(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR acotpi(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR coth(vSCALAR a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `vSCALAR acoth(vSCALAR a0);`
  a0 ∈ (-∞, -1) ⋃ (1, +∞)
  Only available for f64, f32, f16
- `vSCALAR csc(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `vSCALAR cscd(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `vSCALAR cscpi(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `vSCALAR acsc(vSCALAR a0);`
  a0 ∈ (-∞, -1) ⋃ (1, +∞)
  Only available for f64, f32, f16
- `vSCALAR acscd(vSCALAR a0);`
  a0 ∈ (-∞, -1) ⋃ (1, +∞)
  Only available for f64, f32, f16
- `vSCALAR acscpi(vSCALAR a0);`
  a0 ∈ (-∞, -1) ⋃ (1, +∞)
  Only available for f64, f32, f16
- `vSCALAR csch(vSCALAR a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `vSCALAR acsch(vSCALAR a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `vSCALAR sec(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅(z+0.5)}
  Only available for f64, f32, f16
- `vSCALAR secd(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR secpi(vSCALAR a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅(z+0.5)}
  Only available for f64, f32, f16
- `vSCALAR asec(vSCALAR a0);`
  a0 ∈ (-∞, -1] ⋃ [1, +∞)
  Only available for f64, f32, f16
- `vSCALAR asecd(vSCALAR a0);`
  a0 ∈ (-∞, -1] ⋃ [1, +∞)
  Only available for f64, f32, f16
- `vSCALAR asecpi(vSCALAR a0);`
  a0 ∈ (-∞, -1] ⋃ [1, +∞)
  Only available for f64, f32, f16
- `vSCALAR sech(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR asech(vSCALAR a0);`
  a0 ∈ (0, 1]
  Only available for f64, f32, f16
- `vSCALAR pow_abs(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
  Only available for f64, f32, f16
- `vSCALAR pow(vSCALAR a0, vSCALAR a1);`
  (a0, a1) ∈ ℝ × ℝ
  Only available for f64, f32, f16
- `vSCALAR exp(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR exp2(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR exp10(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR expm1(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR exprecnegc(vSCALAR a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `vSCALAR exprecneg(vSCALAR a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `vSCALAR expx2(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR log(vSCALAR a0);`
  a0 ∈ (0, +∞)
  Only available for f64, f32, f16
- `vSCALAR log2(vSCALAR a0);`
  a0 ∈ (0, +∞)
  Only available for f64, f32, f16
- `vSCALAR log10(vSCALAR a0);`
  a0 ∈ (0, +∞)
  Only available for f64, f32, f16
- `vSCALAR log1p(vSCALAR a0);`
  a0 ∈ (-1, +∞)
  Only available for f64, f32, f16
- `vSCALAR gamma(vSCALAR a0);`
  a0 ∈ ℝ∖{n∈ℕ:-n}
  Only available for f64, f32, f16
- `vSCALAR lgamma(vSCALAR a0);`
  a0 ∈ ℝ∖{n∈ℕ:-n}
  Only available for f64, f32, f16
- `vSCALAR stirling(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR erf(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR erfc(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `vSCALAR erfcx(vSCALAR a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16


## C++ advanced API

The C++ advanced API is called advanced not because it requires C++11 or above
but because it makes use of the particular implementation of ARM SVE by ARM
in their compiler. We do not know if GCC (and possibly MSVC in the distant
future) will use the same approach. Anyway the current implementation allows
us to put SVE SIMD vectors inside some kinf of structs that behave like
standard structs. If you want to be sure to write portable code do *not* use
this API. Two new types are available.

- `nsimd::pack<SCALAR, N, SIMDEXT>` represents `N` SIMD vectors containing
  SCALAR elements of SIMD extension SIMDEXT. You can specify only the first
  template argument. The second defaults to 1 while the third defaults to the
  detected SIMDEXT.
- `nsimd::packl<SCALAR, N, SIMDEXT>` represents `N` SIMD vectors of logical
  type containing SCALAR elements of SIMD extension SIMDEXT. You can specify
  only the first template argument. The second defaults to 1 while the third
  defaults to the detected SIMDEXT.

Use N > 1 when declaring packs to have an unroll of N. This is particularily
useful on ARM.

Functions that takes packs do not take any other argument unless specified
otherwise e.g. the load family of funtions. It is impossible to determine
the kind of pack (unroll and SIMDEXT) from the type of a pointer. Therefore
in this case, the last argument must be a pack and this same type will then
return. Also some functions are available as C++ operators.

Here is the list of functions that act on packs.

- `int len(pack<T, N, SimdExt> const&);`
- `pack<T, N, SimdExt> set1(T a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> loadu(T const* a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> load2u(T const* a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> load3u(T const* a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> load4u(T const* a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> loada(T const* a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> load2a(T const* a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> load3a(T const* a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> load4a(T const* a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `packl<T, N, SimdExt> loadlu(T const* a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `packl<T, N, SimdExt> loadla(T const* a0, pack<T, N, SimdExt> const&);`
  a0 ∈ ℝ
- `void storeu(T* a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
- `void store2u(T* a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2);`
  (a0, a1, a2) ∈ ℝ × ℝ
- `void store3u(T* a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2, pack<T, N, SimdExt> const& a3);`
  (a0, a1, a2, a3) ∈ ℝ × ℝ × ℝ
- `void store4u(T* a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2, pack<T, N, SimdExt> const& a3, pack<T, N, SimdExt> const& a4);`
  (a0, a1, a2, a3, a4) ∈ ℝ × ℝ × ℝ × ℝ
- `void storea(T* a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
- `void store2a(T* a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2);`
  (a0, a1, a2) ∈ ℝ × ℝ
- `void store3a(T* a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2, pack<T, N, SimdExt> const& a3);`
  (a0, a1, a2, a3) ∈ ℝ × ℝ × ℝ
- `void store4a(T* a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2, pack<T, N, SimdExt> const& a3, pack<T, N, SimdExt> const& a4);`
  (a0, a1, a2, a3, a4) ∈ ℝ × ℝ × ℝ × ℝ
- `void storelu(T* a0, packl<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
- `void storela(T* a0, packl<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
- `pack<T, N, SimdExt> orb(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator|
- `pack<T, N, SimdExt> andb(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator&
- `pack<T, N, SimdExt> andnotb(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
- `pack<T, N, SimdExt> notb(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Available as operator~
- `pack<T, N, SimdExt> xorb(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator^
- `packl<T, N, SimdExt> orl(packl<T, N, SimdExt> const& a0, packl<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ 𝔹 × 𝔹
  Available as operator||
- `packl<T, N, SimdExt> andl(packl<T, N, SimdExt> const& a0, packl<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ 𝔹 × 𝔹
  Available as operator&&
- `packl<T, N, SimdExt> andnotl(packl<T, N, SimdExt> const& a0, packl<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ 𝔹 × 𝔹
- `packl<T, N, SimdExt> xorl(packl<T, N, SimdExt> const& a0, packl<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ 𝔹 × 𝔹
- `packl<T, N, SimdExt> notl(packl<T, N, SimdExt> const& a0);`
  a0 ∈ 𝔹
  Available as operator!
- `pack<T, N, SimdExt> add(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator+
- `pack<T, N, SimdExt> sub(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator-
- `T addv(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> mul(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator*
- `pack<T, N, SimdExt> div(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ∖{0}
  Available as operator/
- `pack<T, N, SimdExt> neg(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Available as operator-
- `pack<T, N, SimdExt> min(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
- `pack<T, N, SimdExt> max(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
- `pack<T, N, SimdExt> shr(pack<T, N, SimdExt> const& a0, int a1);`
  (a0, a1) ∈ ℝ × ℕ
  Available as operator>>
  Only available for i64, i32, i16, i8, u64, u32, u16, u8
- `pack<T, N, SimdExt> shl(pack<T, N, SimdExt> const& a0, int a1);`
  (a0, a1) ∈ ℝ × ℕ
  Available as operator<<
  Only available for i64, i32, i16, i8, u64, u32, u16, u8
- `packl<T, N, SimdExt> eq(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator==
- `packl<T, N, SimdExt> ne(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator!=
- `packl<T, N, SimdExt> gt(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator>
- `packl<T, N, SimdExt> ge(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator>=
- `packl<T, N, SimdExt> lt(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator<
- `packl<T, N, SimdExt> le(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Available as operator<=
- `pack<T, N, SimdExt> if_else1(packl<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2);`
  (a0, a1, a2) ∈ 𝔹 × ℝ × ℝ
- `pack<T, N, SimdExt> abs(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> fma(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2);`
  (a0, a1, a2) ∈ ℝ × ℝ × ℝ
- `pack<T, N, SimdExt> fnma(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2);`
  (a0, a1, a2) ∈ ℝ × ℝ × ℝ
- `pack<T, N, SimdExt> fms(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2);`
  (a0, a1, a2) ∈ ℝ × ℝ × ℝ
- `pack<T, N, SimdExt> fnms(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1, pack<T, N, SimdExt> const& a2);`
  (a0, a1, a2) ∈ ℝ × ℝ × ℝ
- `pack<T, N, SimdExt> ceil(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> floor(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> trunc(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> round_to_even(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
- `int all(packl<T, N, SimdExt> const& a0);`
  a0 ∈ 𝔹
- `int any(packl<T, N, SimdExt> const& a0);`
  a0 ∈ 𝔹
- `int nbtrue(packl<T, N, SimdExt> const& a0);`
  a0 ∈ 𝔹
- `pack<T, N, SimdExt> reinterpret(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
- `packl<T, N, SimdExt> reinterpretl(packl<T, N, SimdExt> const& a0);`
  a0 ∈ 𝔹
- `pack<T, N, SimdExt> cvt(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> upcvt(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for i8, u8, i16, u16, f16, i32, u32, f32
- `pack<T, N, SimdExt> downcvt(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
  Only available for i16, u16, f16, i32, u32, f32, i64, u64, f64
- `pack<T, N, SimdExt> rec(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> rec11(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> sqrt(pack<T, N, SimdExt> const& a0);`
  a0 ∈ [0, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> rsqrt11(pack<T, N, SimdExt> const& a0);`
  a0 ∈ [0, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> ziplo(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
- `pack<T, N, SimdExt> ziphi(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
- `pack<T, N, SimdExt> zip(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
- `pack<T, N, SimdExt> unziplo(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
- `pack<T, N, SimdExt> unziphi(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
- `pack<T, N, SimdExt> unzip(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ
- `pack<T, N, SimdExt> to_mask(packl<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
- `packl<T, N, SimdExt> to_logical(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
- `pack<T, N, SimdExt> cos(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> cosd(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> cospi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acos(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acosd(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acospi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> cosh(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acosh(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (1, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> sin(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> sinpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> sind(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> asin(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> asind(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> asinpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> sinh(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> asinh(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> tan(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅(z+0.5)}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> tand(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅(z+0.5)}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> tanpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅(z+0.5)}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> atan(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> atand(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> atanpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> tanh(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> atanh(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-1, 1)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> sinc(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> sincpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> sinhc(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> atan2(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> atan2d(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> atan2pi(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> cot(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> cotd(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> cotpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acot(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acotd(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acotpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> coth(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acoth(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-∞, -1) ⋃ (1, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> csc(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> cscd(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> cscpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅z}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acsc(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-∞, -1) ⋃ (1, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acscd(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-∞, -1) ⋃ (1, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acscpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-∞, -1) ⋃ (1, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> csch(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> acsch(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> sec(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅(z+0.5)}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> secd(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> secpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{z∈ℤ:π⋅(z+0.5)}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> asec(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-∞, -1] ⋃ [1, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> asecd(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-∞, -1] ⋃ [1, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> asecpi(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-∞, -1] ⋃ [1, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> sech(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> asech(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (0, 1]
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> pow_abs(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> pow(pack<T, N, SimdExt> const& a0, pack<T, N, SimdExt> const& a1);`
  (a0, a1) ∈ ℝ × ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> exp(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> exp2(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> exp10(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> expm1(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> exprecnegc(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> exprecneg(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{0}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> expx2(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> log(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (0, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> log2(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (0, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> log10(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (0, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> log1p(pack<T, N, SimdExt> const& a0);`
  a0 ∈ (-1, +∞)
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> gamma(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{n∈ℕ:-n}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> lgamma(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ∖{n∈ℕ:-n}
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> stirling(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> erf(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> erfc(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
- `pack<T, N, SimdExt> erfcx(pack<T, N, SimdExt> const& a0);`
  a0 ∈ ℝ
  Only available for f64, f32, f16
