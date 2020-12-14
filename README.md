Documentation can be found [here](https://agenium-scale.github.io/nsimd/).

# What is NSIMD?

At its core, NSIMD is a vectorization library that abstracts [SIMD
programming](<https://en.wikipedia.org/wiki/SIMD>). It was designed to exploit
the maximum power of processors at a low development cost. NSIMD comes with
modules. As of now two of them adds support for GPUs to NSIMD. The
direction that NSIMD is taking is to provide several programming paradigms
to address different problems and to allow a wider support of architectures.
With two of its modules NSIMD provides three programming paradigms:

- Imperative programming provided by NSIMD core that supports a lots of
  CPU/SIMD extensions.
- Expressions templates provided by the TET1D module that supports all
  architectures from NSIMD core and adds support for NVIDIA and AMD GPUs.
- Single Program Multiple Data provided by the SPMD module that supports all
  architectures from NSIMD core and adds support for NVIDIA and AMD GPUs.

## Supported architectures

| Architecture                          | NSIMD core | TET1D module | SPMD module |
|:--------------------------------------|:----------:|:------------:|:-----------:|
| CPU (SIMD emulation)                  |     Y      |      Y       |      Y      |
| Intel SSE 2                           |     Y      |      Y       |      Y      |
| Intel SSE 4.2                         |     Y      |      Y       |      Y      |
| Intel AVX                             |     Y      |      Y       |      Y      |
| Intel AVX2                            |     Y      |      Y       |      Y      |
| Intel AVX-512 for KNLs                |     Y      |      Y       |      Y      |
| Intel AVX-512 for Skylake processors  |     Y      |      Y       |      Y      |
| Arm NEON 128 bits (ARMv7 and earlier) |     Y      |      Y       |      Y      |
| Arm NEON 128 bits (ARMv8 and later)   |     Y      |      Y       |      Y      |
| Arm SVE (original sizeless SVE)       |     Y      |      Y       |      Y      |
| Arm fixed sized SVE                   |     Y      |      Y       |      Y      |
| NVIDIA CUDA                           |     N      |      Y       |      Y      |
| AMD ROCm                              |     N      |      Y       |      Y      |

## How it works?

To achieve maximum performance, NSIMD mainly relies on the inline optimization
pass of the compiler. Therefore using any mainstream compiler such as GCC,
Clang, MSVC, XL C/C++, ICC and others with NSIMD will give you a zero-cost SIMD
abstraction library.

To allow inlining, a lot of code is placed in header files. *Small* functions
such as addition, multiplication, square root, etc, are all present in header
files whereas big functions such as I/O are put in source files that are
compiled as a `.so`/`.dll` library.

NSIMD provides C89, C++98, C++11, C++14 and C++20 APIs. All APIs allow writing
generic code. For the C API this is achieved through a thin layer of macros;
for the C++ APIs it is achieved using templates and function overloading. The
C++ APIs are split into two. The first part is a C-like API with only function
calls and direct type definitions for SIMD types while the second one provides
operator overloading, higher level type definitions that allows unrolling.
C++11, C++14 APIs add for instance templated type definitions and templated
constants while the C++20 API uses concepts for better error reporting.

Binary compatibility is guaranteed by the fact that only a C ABI is exposed.
The C++ API only wraps the C calls.

## Supported compilers

NSIMD is tested with GCC, Clang, MSVC, NVCC, HIPCC and ARMClang. As a C89 and a
C++98 API are provided, other compilers should work fine. Old compiler versions
should work as long as they support the targeted SIMD extension. For instance,
NSIMD can compile on SSE 4.2 code with MSVC 2010.

# Build the library

The support for CMake has been dropped as it has several flaws:
- too slow especially on Windows,
- inability to use several compilers at once,
- inability to have a portable build system,
- very poor support for portable compilation flags,
- ...

## Dependencies

Generating C/C++ files is done by the Python3 code contained in the `egg`.
Python should be installed by default on any Linux distro. On Windows it comes
with the latest versions of Visual Studio on Windows
(<https://visualstudio.microsoft.com/vs/community/>), you can also download and
install it directly from <https://www.python.org/>.

The Python code can call `clang-format` to properly format all generated C/C++
source. On Linux you can install it via your package manager. On Windows you
can use the official binary at <https://llvm.org/builds/>.

Testing the library requires the MPFR library that can be found at
<https://www.mpfr.org/>.

Benchmarking the library requires Google Benchmark version 1.3 that can be
found at <https://github.com/google/benchmark> plus all the other SIMD
libraries used for comparison:
- MIPP (<https://github.com/aff3ct/MIPP>)
- Sleef (<https://sleef.org/>)

Compiling the library requires a C++98 compiler. Any version of GCC, Clang or
MSVC will do. Note that the produced library and header files for the end-user
are C89, C++98, C++11 compatible. Note that C/C++ files are generated by a
bunch of Python scripts and they must be executed first before running building
the library.

## Build for Linux

```bash
bash scripts/build.sh for simd_ext1/.../simd_extN with comp1/.../compN
```

For each combination a directory `build-simd_ext-comp` will be created and
will contain the library. Supported SIMD extension are:

- sse2
- sse42
- avx
- avx2
- avx512\_knl
- avx512\_skylake
- neon128
- aarch64
- sve
- sve128
- sve256
- sve512
- sve1024
- sve2048
- cuda
- rocm

Supported compiler are:

- gcc
- clang
- icc
- armclang
- cl
- nvcc
- hipcc

Note that certain combination of SIMD extension/compilers are not supported
such as aarch64 with icc, or avx512\_skylake with nvcc.

## Build on Windows

Make sure you are typing in a Visual Studio prompt. The command is almost the
same as for Linux with the same constraints on the pairs SIMD
extension/compilers.

```batch
scripts\build.bat for simd_ext1/.../simd_extN with comp1/.../compN
```

## More details on building the library

The library uses a tool called nsconfig
(<https://github.com/agenium-scale/nstools>) which is basically a Makefile
translator. If you have just built NSIMD following what's described above
you should have a `nstools` directory which contains `bin/nsconfig`. If not
you can generate it using on Linux

```bash
bash scripts/setup.sh
```

and on Windows

```batch
scripts\setup.bat
```

Then you can use `nsconfig` directly it has a syntax similar to CMake at
command line. Here is a quick tutorial with Linux command line. We first
go to the NSIMD directory and generate both NSIMD and nsconfig.

```bash
$ cd nsimd
$ python3 egg/hatch.py -ltf
$ bash scripts/setup.sh
$ mkdir build
$ cd build
```

Help can be displayed using `--help`.

```bash
$ ../nstools/bin/nsconfig --help
usage: nsconfig [OPTIONS]... DIRECTORY
Configure project for compilation.

  -v                   verbose mode
  -nodev               Build system will never call nsconfig
  -DVAR=VALUE          Set value of variable VAR to VALUE
  -list-vars           List project specific variable
  -GBUILD_SYSTEM       Produce files for build system BUILD_SYSTEM
  -Ghelp               List supported build systems
  -oOUTPUT             Output to OUTPUT instead of default
  -comp=SUITE          Use compiler SUITE for both C and C++
  -ccomp=SUITE,PATH[,VERSION[,ARCHI]]
                       Use PATH as default C compiler from suite SUITE
                       If VERSION and/or ARCHI are given, nsconfig
                       try to determine those. This is useful for
                       cross compiling, ARCHI must be in { x86, x86_64,
                       arm, aarch64 }
  -ccomp=help          List supported C compilers
  -cppcomp=SUITE,PATH[,VERSION[,ARCHI]]
                       Use PATH as default C++ compiler from suite SUITE
                       If VERSION and/or ARCHI are given, nsconfig
                       try to determine those. This is useful for
                       cross compiling, ARCHI must be in { x86, x86_64,
                       arm, aarch64 }
  -cppcomp=help        List supported C++ compilers
  -prefix=PREFIX       Set prefix for installation to PREFIX
  -h, --help           Print the current text
```

Each project can defined its own set of variable controlling the generation of
the ninja file of Makefile.

```bash
$ ../nstools/bin/nsconfig .. -list-vars
Project variables list:
name           | description
---------------|---------------------------------------------------------
simd           | SIMD extension to use
optional_flags | Optional flags like FMA, FP16...
mpfr           | MPFR compilation flags (for tests only)
sleef          | Sleef compilation flags (for benchmarks only)
benchmark      | Google benchmark compilation flags (for benchmarks only)
```

Finally one can choose what to do and compile NSIMD and its tests.

```bash
$ ../nstools/bin/nsconfig .. -Dsimd=avx2
$ ninja
$ ninja tests
```

Note that MPFR (<https://www.mpfr.org/>) is needed to compile the tests. If you
do not have the MPFR header installed on your system or if you want to use a
custom version of MPFR you can tell nsconfig how where to find it.

```bash
$ ../nstools/bin/nsconfig .. -Dsimd=avx2 \
      -Dmpfr="-Iwhere/is/mpfr/include -Lwhere/is/mpfr/lib -lmpfr"
$ ninja
$ ninja tests
```

Nsconfig comes with nstest a small tool to execute tests.

```bash
$ ../nstools/bin/nstest -j20
```

## Cross compilation

It is useful to cross-compile for example when you are on a Intel workstation
and want to compile for a Raspberry Pi. Nsconfig generate some code, compile
and run it to obtain informations on the C or C++ compilers. When cross
compiling, unless you configured your Linux box with binfmt\_misc to
tranparently execute aarch64 binaries on a x86\_64 host you need to give
nsconfig all the informations about the compilers so that it does not need to
run aarch64 code on x86\_64 host.

```bash
$ ../nstools/bin/nsconfig .. -Dsimd=aarch64 \
      -ccomp=gcc,aarch64-linux-gnu-gcc,10.0,aarch64 \
      -cppcomp=gcc,aarch64-linux-gnu-g++,10.0,aarch64
```

## Defines that control NSIMD compilation and usage

Several defines control NSIMD.

- `FMA` or `NSIMD_FMA` indicate to NSIMD that fma intrinsics can be used
  when compiling code. This is useful on Intel SSE2, SSE42, AVX and AVX2.

- `FP16` or `NSIMD_FP16` indicate to NSIMD that the targeted architecture
  natively (and possibly partially) supports IEEE float16's. This is useful
  when compiling for Intel SSE2, SSE42, AVX and AVX2, Arm NEON128 and AARCH64.

# Philosophy of NSIMD

Originally the library aimed at providing a portable zero-cost abstraction over
SIMD vendor intrinsics disregarding the underlying SIMD vector length. NSIMD
will of course continue to wrap SIMD intrinsics from various vendors but
more efforts will be put into writing NSIMD modules and improving the existing
ones especially the SPMD module. 

## The SPMD paradigm

It is our belief that SPMD is a good paradigm for writing vectorized code. It
helps both the developer and the compiler writer. It forces the developers to
better arrange its data ion memory more suited for vectorization. On the
compiler side it is more simplier to write a "SPMD compiler" than a standard
C/C++/Fortran compiler that tries to autovectorize some weird loop with data
scattered all around the place. Our priority for our SPMD module are the
following:

- Add oneAPI/SYCL support.
- Provide a richer API.
- Provide cross-lane data transfer.
- Provide a way to abstract shared memory.

Our approach can be roughly compared to ISPC (<https://ispc.github.io/>)
but from a library point of view.

## Wrapping intrinsics in NSIMD core

NSIMD was designed following as closely as possible the following guidelines:

- Correctness primes over speed.
- Emulate with tricks and intrinsic integer arithmetic when not available.
- Use common names as found in common computation libraries.
- Do not hide SIMD registers, one variable (of a type such as `nsimd::pack`)
  matches one register.
- Make the life of the compiler as easy as possible: keep the code simple to
  allow the compiler to perform as many optimizations as possible.
- Favor the advanced C++ API.

You may wrap intrinsics that require compile time knowledge of the underlying
vector length but this should be done with caution.

Wrapping intrinsics that do not exist for all types is difficult and may
require casting or emulation. For instance, 8 bit integer vector multiplication
using SSE2 does not exist. We can either process each pair of integers
individually or we can cast the 8 bit vectors to 16 bit vectors, do the
multiplication and cast them back to 8 bit vectors. In the second case,
chaining operations will generate many unwanted casts.

To avoid hiding important details to the user, overloads of operators involving
scalars and SIMD vectors are not provided by default. Those can be included
explicitely to emphasize the fact that using expressions like `scalar + vector`
might incur an optimization penalty.

The use of `nsimd::pack` may not be portable to ARM SVE and therefore must be
included manually. ARM SVE registers can only be stored in sizeless strucs
(`__sizeless_struct`). This feature (as of 2019/04/05) is only supported by the
ARM compiler. We do not know whether other compilers will use the same keyword
or paradigm to support SVE intrinsics.

# Contributing to NSIMD

The wrapping of intrinsics, the writing of test and bench files are tedious and
repetitive tasks. Most of those are generated using Python scripts that can be
found in `egg`.

- Intrinsics that do not require to known the vector length can be wrapped and
  will be accepted with no problem.
- Intrinsics that do require the vector length at compile time can be wrapped
  but it is up to the maintainer to accept it.
- Use `clang-format` when writing C or C++ code.
- The `.cpp` files are written in C++98.
- The headers files must be compatible with C89 (when possible otherwise
  C99), C++98, C++11, C++14 up to and including C++20.

Please see <doc/markdown/CONTRIBUTE.md> for more details.

# LICENSE

Copyright (c) 2020 Agenium Scale

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
