<!--

Copyright (c) 2020 Agenium Scale

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

# NSIMD tutorial

In this tutorial we will write and compile a simple SIMD kernel to become
familiar with the basics of NSIMD. We will also see different aspects of SIMD
programming:
- aligned vs. unaligned data access
- basic SIMD arithmetic
- SIMD loops
- SIMD branching
- architecture selection at runtime

## SIMD basics

SIMD programming means using the CPU SIMD registers to performs operations
on several data at once. A SIMD vector should be viewed as a set of bits
which are interpreted by the operators that operate on them. Taking a 128-bits
wide SIMD register, it can be interpreted as:
- 16 signed/unsigned chars
- 8 signed/unsigned shorts
- 4 signed/unsigned ints
- 4 floats
- 2 signed/unsigned longs
- 2 doubles
as shown in the picture below.

![Register layout](img/register.png)

## Computation kernel

We will explain the rewriting of the following kernel which uppercases ASCII
letters only.

@[INCLUDE_CODE:L7:L16](../../examples/tutorial.cpp)

Here is the corresponding SIMD version. Explanations to follow.

@[INCLUDE_CODE:L18:L39](../../examples/tutorial.cpp)

## Getting started with NSIMD

All APIs of NSIMD core is available with this include:

@[INCLUDE_CODE:L1:L1](../../examples/tutorial.cpp)

For ease of programming with use the NSIMD namespace inside the
`uppercase_simd` function.

@[INCLUDE_CODE:L20:L20](../../examples/tutorial.cpp)

## SIMD vectors

A `nsimd::pack<T>` can be considered analogous to a SIMD register (on your or
any other machine). Operations performed on packs - from elementary operations
such as addition to complicated functions such as `nsimd::rsqrt11(x)` - will be
performed using SIMD registers and operations if supported by your hardware. As
shown below, data must be manually loaded into and stored from these registers.
Again, for ease of programming we typedef a pack of T's.

@[INCLUDE_CODE:L21:L21](../../examples/tutorial.cpp)

NSIMD provides another type of pack called `nsimd::packl` which handles vectors
of booleans.

@[INCLUDE_CODE:L22:L22](../../examples/tutorial.cpp)

This distinction between pack's and packl's is necessary ffor two reasons:
- On recent hardware, SIMD vectors of booleans are handled by dedicated
  registers.
- Pack and Packl must have different semantics as arithmetic operators on
  booleans have no sense as well as logical operators on Pack's.

## Loading data from memory

One way to construct a `nsimd::pack<T>` is to simply declare
(default-construct) it. Such a pack may *not* be zero-initialized and thus may
*contain arbitrary values*.

Another way to construct a `nsimd::pack<T>` is to fill it with a single value.
This so-called splatting constructor takes one scalar value and replicates it
in all elements of the pack.

But most common usage to construct a `nsimd::pack<T>` is by using the copy
constructor from loading functions.

@[INCLUDE_CODE:L27:L27](../../examples/tutorial.cpp)

## Aligned vs. unaligned memory

Alignement of a given pointer `ptr` to memory to some value `A` means that
`ptr % A == 0`. On older hardware loading data from unaligned memory can
result in performance penalty. On recent hardware it is hard to exhibit a
difference. NSIMD provides two versions of "load":
- `loada` for loading data from aligned memory
- `loadu` for loading data from unaligned momery
Note that using `loada` on unaligned pointer may result in segfaults. As
recent hardware have good support for unaligned memory we use `loadu`.

@[INCLUDE_CODE:L27:L27](../../examples/tutorial.cpp)

To ensure that data allocated by `std::vector` is aligned, NSIMD provide
a C++ allocator.

```c++
std::vector<T, nsimd::allocator<T> > data;
```

When loading data from memory you must ensure that there is sufficient data in
the block of memory you load from to fill a `nsimd::pack<T>`. For example, on
an `AVX` capable machine, a SIMD vector of `float` (32 bits) contains 8
elements. Therefore, there must be at least 8 floats in the memory block you
load data from otherwise loading may result in segfaults. More on this below.

## Operations on pack's and packl's

Once initialized, `nsimd::pack<T>` instances can be used to perform arithmetic.
Usual operations are provided by NSIMD such:
- addition
- substraction
- multiplication
- division
- square root
- bitwise and/or/xor
- ...

@[INCLUDE_CODE:L28:L29](../../examples/tutorial.cpp)

C++ operators are also overloaded for pack's and packl's as well as between
pack's and scalars or packl's and booleans.

## SIMD branching

NSIMD provide the `if_else` operator which fill the output, lane by lane,
according to the lane value of its first argument:
- if it is true, the output lane will be filled with the second argument's lane
- if it is false, the output lane will be filled with the third argument's lane
Therefore the branching:

@[INCLUDE_CODE:L10:L14](../../examples/tutorial.cpp)

will be rewritten as

@[INCLUDE_CODE:L28:L30](../../examples/tutorial.cpp)

or as a one liner

@[INCLUDE_CODE:L36:L36](../../examples/tutorial.cpp)

## SIMD loops

A SIMD loop is similar to its scalar counterpart except that instead of
going through data one element at a time it goes 4 by 4 or 8 by 8 elements
at a time. More precisely SIMD loops generally goes from steps equal to
pack's length. Therefore the scalar loop

@[INCLUDE_CODE:L9:L9](../../examples/tutorial.cpp)

is rewritten as

@[INCLUDE_CODE:L23:L26](../../examples/tutorial.cpp)

Note that going step by step will only cover most of the data except maybe the
tail of data in case that the number of elements is not a multiple of the
Pack's length. Therefore to perform computations on the tail one has to
load data from only `n` elements where `n < len<p_t>()`. One can use
`maskz_loadu` which will load data only on lanes that are marked as true by
another argument to the function.

@[INCLUDE_CODE:L35:L35](../../examples/tutorial.cpp)

The mask can be computed manually but NSIMD provides a function for it.

@[INCLUDE_CODE:L34:L34](../../examples/tutorial.cpp)

Then the computation on the tail is exactly the same as within the loop. Put
together it gives for the tail:

@[INCLUDE_CODE:L34:L37](../../examples/tutorial.cpp)

Then the entire loop reads as follows.

@[INCLUDE_CODE:L25:L37](../../examples/tutorial.cpp)

## Compiling the Code

Here is the complete listing of the code.

@[INCLUDE_CODE](../../examples/tutorial.cpp)

The compilation of a program using `nsimd` is like any other library.

```bash
c++ -O3 -DAVX2 -mavx2 -L/path/to/lib -lnsimd_avx2 -I/path/to/include tutorial.cpp
```

When compiling with NSIMD, you have to decide at compile time the targeted
SIMD extensions, AVX2 in the example above. It is therefore necessary to
give `-mavx2` to the compiler for it to emit AVX2 instructions. To tell NSIMD
that AVX2 has to be used the `-DAVX2` has to be passed to the compiler. For
an exhaustive list of defines controlling compilation see <defines.md>. There
is a .so file for each SIMD extension, it is therefore necessary to link
against the proper .so file.

## Runtime selection of SIMD extensions

It is sometimes necessary to have several versions of a given algorithm for
different SIMD extensions. This is rather to do with NSIMD. Basically the
idea is to write the algorithm in a generic manner using pack's as shown above.
It is then sufficient to compile the same soure file for different SIMD
extensions and then link the resulting object files altogether. Suppose that
a file named `uppercase.cpp` contains the following code:

@[INCLUDE_CODE:L18:L38](../../examples/tutorial.cpp)

This would give the following in a Makefile.

```makefile
all: uppercase

uppercase_sse2.o: uppercase.cpp
	c++ -O3 -DSSE2 -msse2 -c $? -o $@

uppercase_sse42.o: uppercase.cpp
	c++ -O3 -DSSE42 -msse4.2 -c $? -o $@

uppercase_avx.o: uppercase.cpp
	c++ -O3 -DAVX -mavx -c $? -o $@

uppercase_avx2.o: uppercase.cpp
	c++ -O3 -DAVX2 -mavx2 -c $? -o $@

uppercase: uppercase_sse2.o \
           uppercase_sse42.o \
           uppercase_avx.o \
           uppercase_avx2.o
           main.cpp
	c++ $? -lnsimd_avx2 -o $@
```

Note that `libnsimd_avx2` contains all the functions for SSE 2, SSE 4.2, AVX
and AVX2. This is a consequence of the retrocompatiblity of Intel SIMD
extensions. The situation is the same on ARM where `libnsimd_sve.so` will
contain functions for AARCH64.

There is a small caveat. The symbol name corresponding to the `uppercase_simd`
function will be same for all the object files which will result in error
when linking together all objects. To avoid this situation one can use
function overloading as follows:

```c++
template <typename T>
void uppercase_simd(NSIMD_SIMD, T *dst, const T *src, int n) {
  // ...
}
```

The macro `NSIMD_SIMD` will be expanded to a type containing the information on
the SIMD extension currently requested by the user. This techniques is called
tag dispatching and does not require *any* modification of the algorithm
inside the function. Finally in `main` one has to do dispatching by using
either `cpuid` of by another mean.

```c++
int main() {
  // what follows is pseudo-code
  switch(cpuid()) {
  case cpuid_sse2:
    uppercase(nsimd::sse2, dst, src, n);
    break;
  case cpuid_sse42:
    uppercase(nsimd::sse42, dst, src, n);
    break;
  case cpuid_avx:
    uppercase(nsimd::avx, dst, src, n);
    break;
  case cpuid_avx2:
    uppercase(nsimd::avx2, dst, src, n);
    break;
  }
  return 0;
}
```
