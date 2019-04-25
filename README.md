NSIMD
=====

[![Build Status](https://travis-ci.org/agenium-scale/nsimd.svg?branch=master)](https://travis-ci.org/agenium-scale/nsimd)

NSIMD is a vectorization library that abstracts [SIMD
programming](<https://en.wikipedia.org/wiki/SIMD>). It was designed to exploit
the maximum power of processors at a low development cost.

To achieve maximum performance, NSIMD mainly relies on the inline optimization
pass of the compiler. Therefore using any mainstream compiler such as GCC,
Clang, MSVC, XL C/C++, ICC and others with NSIMD will give you a zero-cost SIMD
abstraction library.

To allow inlining, a lot of code is placed in header files. *Small* functions
such as addition, multiplication, square root, etc, are all present in header
files whereas big functions such as I/O are put in source files that are
compiled as a `.so`/`.dll` library.

NSIMD provides C89, C++98, C++11 and C++14 APIs. All APIs allow writing generic
code. For the C API this is achieved through a thin layer of macros; for the C++
APIs it is achieved using templates and function overloading. The C++ API is
split in two. The first part is a C-like API with only function calls and direct
type definitions for SIMD types while the second one provides operator
overloading, higher level type definitions that allows unrolling. C++11, C++14
APIs add for instance templated type definitions and templated constants.

Binary compatibility is guaranteed by the fact that only a C ABI is exposed. The
C++ API only wraps the C calls.

Supported SIMD instruction sets
-------------------------------

- Intel:
  + SSE2
  + SSE4.2
  + AVX
  + AVX2
  + AVX-512 as found on KNLs
  + AVX-512 as found on Xeon Skylake CPUs
- Arm
  + NEON 128 bits as found on ARMv7 CPUs
  + NEON 128 bits as foudn on Aarch64 CPUs
  + SVE

Supported compilers
-------------------

NSIMD is tested with GCC, Clang and MSVC. As a C89 and a C++98 API are provided,
other compilers should work fine. Old compiler versions should work as long as
they support the targeted SIMD extension. For instance, NSIMD can compile to
SSE4.2 on MSVC 2010.

Build the library
=================

The library should be built with
[CMake](<https://gitlab.kitware.com/cmake/cmake>) on Linux and Windows.

Dependencies
------------

Generating C/C++ files is done by the Python3 code contained in the `egg`.
Python should be installed by default on any Linux distro. On Windows it comes
with the latest versions of Visual Studio on Windows
(<https://visualstudio.microsoft.com/vs/community/>), you can also download and
install it directly from <https://www.python.org/>.

The Python code calls `clang-format` to properly format all generated C/C++
source. On Linux you can install it via your package manager. On Windows you can
use the official binary at <https://llvm.org/builds/>.

Testing the library requires the Google Test library that can be found at
<https://github.com/google/googletest> and the MPFR library that can be found at
<https://www.mpfr.org/>.

Benchmarking the library requires Google Benchmark that can be found at
<https://github.com/google/benchmark> plus all the other SIMD libraries used
for comparison:
- MIPP (<https://github.com/aff3ct/MIPP>)
- Sleef (<https://sleef.org/>)

Compiling the library requires a C++14 compiler. Any recent version of GCC,
Clang and MSVC will do. Note that the produced library and header files for the
end-user are C89, C++98, C++11 compatible.

```bash
mkdir build
cd build
cmake ..
make
```

You can set the target architecture using the `-DSIMD=<simd>` option for CMake:

```bash
# Enable AVX2 support for nsimd
cmake .. -DSIMD=AVX2
make
```

Some SIMD instructions are optional. FMA (Fused Multiply-Add) can be enabled
using the option `-DSIMD_OPTIONALS=<simd-optional>...`:

```bash
# Enable AVX2 with FMA support for nsimd
cmake .. -DSIMD=AVX2 -DSIMD_OPTIONALS=FMA
make
```

The generated sources might be big, using `ninja` over `make` is generally
better:

```bash
cmake .. -GNinja
ninja
```

Build on Windows
----------------

Make sure you are typing in a Visual Studio prompt. We give examples with
`ninja`. We also explicitely specify the MSVC compiler. Note that you can
use the latest versions of Visual Studio to build the library using its
`CMakeLists.txt`.

```bash
md build
cd build
cmake .. -GNinja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
ninja
```

You can also set the SIMD instruction using the `-DSIMD=<simd>` option when
generating with cmake like:

```bash
# Enable AVX2 support for nsimd
cmake .. -DSIMD=AVX2 -GNinja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
ninja
```

Some SIMD instructions are optional like for FMA (Fused Multiply-Add), you
can enable them using the option `-DSIMD_OPTIONALS=<simd-optional>...`:

```bash
# Enable AVX2 with FMA support for nsimd
cmake .. -DSIMD=AVX2 -DSIMD_OPTIONALS=FMA -GNinja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
ninja
```

Philosophy
==========

The library aims to provide a portable zero-cost abstraction over SIMD vendor
intrinsics disregarding the underlying SIMD vector length.

NSIMD was designed following as closely as possible the following guidelines:

- Do not aim for a fully IEEE compliant library, rely on intrinsics, errors
  induced by non compliance are small and acceptable.
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
SSE2 does not exist. We can either process each pair of integers individually or
we can cast the 8 bit vectors to 16 bit vectors, do the multiplication and cast
them back to 8 bit vectors. In the second case, chaining operations will
generate many unwanted casts.

To avoid hiding important details to the user, overloads of operators involving
scalars and SIMD vectors are not provided by default. Those can be included
explicitely to emphasize the fact that using expressions like `scalar + vector`
might incur an optimization penalty.

The use of `nsimd::pack` may not be portable to ARM SVE and is therefore must be
included manually. ARM SVE registers can only be stored in sizeless strucs
(`__sizeless_struct`). This feature (as of 2019/04/05) is only supported by the
ARM compiler. We do not know whether other compilers will use the same keyword
to support SVE.

Contributing
============

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

LICENSE
=======

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
