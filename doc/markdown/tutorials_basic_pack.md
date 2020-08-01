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

# Basic Tutorial: The SIMD Pack

In this tutorial we will write and compile a very simple SIMD kernel to become
familiar with the basics of NSIMD.

## Getting Started

All `nsimd` library is available with this include:

@[INCLUDE_CODE:L25:L25](../src/pack.cpp)

## Basic Building Block

### `nsimd::pack<T>`

`nsimd::pack<T>` can be considered analogous to an SIMD register on your (or any
other) machine. Operations performed on packs - from elementary operations such
as addition to complicated functions such as `nsimd::sin(x)` - will be performed
using SIMD registers and operations if supported by your hardware. As shown in
the following example, data must be manually loaded into and stored from these
registers.

### Loading Data into a `nsimd::pack<T>`

One way to construct a `nsimd::pack<T>` is to simply declare (default-construct)
it. Such a pack may *not* zero-initialized and thus may *contain arbitrary
values*.

@[INCLUDE_CODE:L40:L42](../src/pack.cpp)

Another way to construct a `nsimd::pack<T>` is to fill it with a single value.
This so-called splatting constructor takes one scalar value and replicates it in
all elements of the pack.

@[INCLUDE_CODE:L49:L51](../src/pack.cpp)

Most common usage to construct a `nsimd::pack<T>` is by passing a pointer to a
block of contiguous, aligned memory.

@[INCLUDE_CODE:L58:L66](../src/pack.cpp)

If the memory is not aligned, not recommented, you can use `nsimd::loadu`:
@[INCLUDE_CODE:L73:L81](../src/pack.cpp)

NOTE:  
This vector uses a custom memory allocator to ensure that the memory used for
storage of the data is correctly aligned for the target architecture. Please see
[Memory Alignment](tutorials_basic_memory_alignment.md) for a detailed
explanation of this. 

When constructing a `nsimd::pack<T>` in this manner, you must ensure that there
is sufficient data in the block of memory to fill the `nsimd::pack<T>`. For
example, on an `AVX` enabled machine, a SIMD vector of `float` (32 bits)
contains `8` elements. Therefore, there must be at least `8` elements in the 
block of memory pointed to by this pointer. This same code compiled for the
`AVX-512`, would require that the block of memory contain `16` elements,
otherwise there would be undefined behaviour at runtime. When writing vectorized
code, care should be taken to write the code in as generic a manner as possible
to ensure portability across architectures.

NOTE:  
Other functions exist to explicitly load data from unaligned memory,
non-contiguous data and other more complex scenarios. These functions are
presented in later tutorials.

### Operations on `nsimd::pack<T>`

Once initialized, operations on `nsimd::pack<T>` instances are similar to scalar
operations as all operators and standard library math functions are provided.

@[INCLUDE_CODE:L87:L93](../src/pack.cpp)

### Storing the Result In Memory

The result may be saved to memory as follows:

@[INCLUDE_CODE:L100:L111](../src/pack.cpp)

If memory is not aligned which is not recommented, you can use `nsimd::loadu`:

@[INCLUDE_CODE:L118:L129](../src/pack.cpp)

Download full source code:
- [pack.cpp](nsimd/src/pack.cpp)

## Compiling the Code

The compilation of a program using `nsimd` should be like using any other
external library. With compilers like `GCC`, you must specify the include
directory with `-I /path/of/nsimd/nsimd-all.hpp` if this header file is not in
one directory of the default include directories. You also must specif
`-lsimd_ARCH` (for example, `-lnsimd_x86_64`) to link with `nsimd` library. If
the library is not in one directory of the default library directories, you must
specify it with `-L /path/of/libsimd_ARCH.so`. If you have to use `-L` option,
you will need set `LD_LIBRARY_PATH` before running the program. An alternative
is to add `-rpath /path/of/libsimd_ARCH.so` option during the compilation.

It is strongly recommended that you enable all of your compiler optimizations,
for example, `-O3` for compilers like `GCC`, to exploit the full performance
potential of `nsimd`. You should also pass the required compiler flag for your
target architecture to enable the SIMD extensions, especially if you are
cross-compiling. The exhaustive list of all compiler flags for all supported
compilers is provided in the [index page](index.html).

We can find an example of `Makefile` to compile programs using `nsimd` in the
sources of this documentation. Future tutorials will explain how to compile a
program using `nsimd` with `CMake` and with `nsconfig`, our alternative to
`CMake`.
