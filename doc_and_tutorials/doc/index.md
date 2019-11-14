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

# Introduction

Single instruction, multiple data ([SIMD](https://en.wikipedia.org/wiki/SIMD))
instructions or multimedia extensions have been available for many years.
They are designed to significantly accelerate code execution, however they
require expertise to be used correctly, depends on non-uniform compiler support,
the use of low-level intrinsics, or vendor-specific libraries.

[`nsimd`](https://github.com/agenium-scale/nsimd) is library which aims to
simplify the error-prone process of developing application exploiting the
potential of SIMD instructions sets. `nsimd` is designed to seamlessly integrate
into existing projects so that you can quickly and easily start developing high
performance, portable and future proof software.


## Why use `nsimd`?

`nsimd` standardizes and simplifies the use of SIMD instructions across hardware
by not relying on verbose, low-level SIMD instructions. Furthermore, the
portability of `nsimd` eliminates the need to re-write cumbersome code for each
revision of each target architecture, accounting for each architecture's vendor
provided API as well as architecture dependent implementation details. This
greatly reduces the design complexity and maintenance of SIMD code,
significantly decreasing the time required to develop, test and deploy software
as well as decreasing the scope for introducing bugs.

`nsimd` allows you to focus on the important part of your work: the development
of new features and functionality. We take care of all of the architecture and
compiler specific details and we provide updates when new architectures are
released by manufacturers. All you have to do is re-compile your code every time
you target a new architecture.


## Inside `nsimd`

`nsimd` is a vectorization library that abstracts SIMD programming. It was
designed to exploit the maximum power of processors at a low development cost.

To achieve maximum performance, `nsimd` mainly relies on the inline optimization
pass of the compiler. Therefore using any mainstream compiler such as GCC,
Clang, MSVC, XL C/C++, ICC and others with `nsimd` will give you a zero-cost
SIMD abstraction library.

To allow inlining, a lot of code is placed in header files. *Small* functions
such as addition, multiplication, square root, etc, are all present in header
files whereas big functions such as I/O are put in source files that are
compiled as a `.so`/`.dll` library.

`nsimd` provides C89, C++98, C++11 and C++14 APIs. All APIs allow writing
generic code. For the C API this is achieved through a thin layer of macros; for
the C++ APIs it is achieved using templates and function overloading. The C++
API is split in two. The first part is a C-like API with only function calls and
direct type definitions for SIMD types while the second one provides operator
overloading, higher level type definitions that allows unrolling. C++11, C++14
APIs add for instance templated type definitions and templated constants.

Binary compatibility is guaranteed by the fact that only a C ABI is exposed. The
C++ API only wraps the C calls.


## `nsimd` Philosophy

The library aims to provide a portable zero-cost abstraction over SIMD vendor
intrinsics disregarding the underlying SIMD vector length.

NSIMD was designed following as closely as possible the following guidelines:

- Do not aim for a fully [IEEE](https://en.wikipedia.org/wiki/IEEE_754)
  compliant library, rely on intrinsics, errors induced by non compliance are
  small and acceptable.
- Correctness primes over speed.
- Emulate with tricks and intrinsic integer arithmetic when not available.
- Use common names as found in common computation libraries.
- Do not hide SIMD registers, one variable (of a type such as `nsimd::pack`)
  matches one register.
- Keep the code simple to allow the compiler to perform as many optimizations as
  possible.

You may wrap intrinsics that require compile time knowledge of the underlying
vector length but this should be done with caution.

Wrapping intrinsics that do not exist for all types is difficult and may require
casting or emulation. For instance, 8 bit integer vector multiplication using
`SSE2` does not exist. We can either process each pair of integers individually or
we can cast the 8 bit vectors to 16 bit vectors, do the multiplication and cast
them back to 8 bit vectors. In the second case, chaining operations will
generate many unwanted casts.

To avoid hiding important details to the user, overloads of operators involving
scalars and SIMD vectors are not provided by default. Those can be included
explicitely to emphasize the fact that using expressions like `scalar + vector`
might incur an optimization penalty.

The use of `nsimd::pack` may not be portable to ARM `SVE` and therefore must be
included manually. ARM `SVE` registers can only be stored in sizeless structs
(`__sizeless_struct`). This feature (as of 2019/04/05) is only supported by the
ARM compiler. We do not know whether other compilers will use the same keyword
or paradigm to support SVE intrinsics.


## A Short Example Using `nsimd`

Let's take a simple case where we calculate the sum of two vectors of 32-bit
floats:
@[INCLUDE_CODE:L92:L94](../src/hello_world.cpp)

Each element of the results vector is independent of every other element -
therefore this function may easily be vectorized as there is latent data
parallelism which may be exploited. This simple loop may be vectorized for an
x86 processor using Intel intrinsic functions. For example, the following code
vectorizes this loop for a SSE enabled processor:
@[INCLUDE_CODE:L108:L114](../src/hello_world.cpp)

Looks difficult? How about we vectorize it for the following generation of Intel
processor equipped with AVX instructions:
@[INCLUDE_CODE:L130:L136](../src/hello_world.cpp)

Both of these processors are manufactured by Intel yet two different versions of
the code are required to get the best performance possible from each processor.
This is quickly getting complicated and annoying.

Now, look at how the code can become simpler with `nsimd`.

`nsimd` C++11 version without the advanced API:
@[INCLUDE_CODE:L152:L159](../src/hello_world.cpp)

`nsimd` C++11 version using the advanced API (not recommended for portability
with ARM `SVE`):
@[INCLUDE_CODE:L179:L186](../src/hello_world.cpp)

`nsimd` C++98 version without the advanced API:
@[INCLUDE_CODE:L206:L213](../src/hello_world.cpp)

`nsimd` C++98 version using the advanced API (not recommended for portability
with ARM `SVE`):
@[INCLUDE_CODE:L227:L233](../src/hello_world.cpp)

`nsimd` C (C89, C99, C11) version:
@[INCLUDE_CODE:L179:L186](../src/hello_world.c)

Download full source code:
- [hello_world.cpp](../src/hello_world.cpp)
- [hello_world.c](../src/hello_world.c)


## Supported Compilers and Hardware by `nsimd`

`nsimd` includes support for some Intel, ARM and IBM processors. The support of
IBM processors is ongoing and will be available soon.

**Architecture** | **Extensions**
---------------- | --------------
Intel            | `SSE2`, `SSE4.2`, `AVX`, `AVX2`, `AVX-512` (`KNL` and `SKYLAKE`)
ARM              | `Aarch64`, `NEON` (`ARMv7`), `SVE`
IBM              | `POWER7`, `POWER8`

@[INCLUDE](compilers_and_versions.md)


## Contributing

The wrapping of intrinsics, the writing of test and bench files are tedious and
repetitive tasks. Most of those are generated using Python scripts that can be
found in `egg`.

- Intrinsics that do not require to known the vector length can be wrapped and
  will be accepted with no problem.
- Intrinsics that do require the vector length at compile time can be wrapped
  but it is up to the maintainer to accept it.
- Use `clang-format` when writing C or C++ code.
- The `.cpp` files are written in C++14.
- The headers files must be compatible with C89 (when possible otherwise
  C99), C++98, C++11 and C++14.

Please see [contribute](contribute.md) for more details.


## License

Copyright (c) 2019 Agenium Scale

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
