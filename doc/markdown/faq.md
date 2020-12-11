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

# Frequently Asked Questions

## Is it good practice to use a `nsimd::pack` as a `std::vector`?

No, these are two very different objects. A `nsimd::pack` represent a SIMD
register whereas a `std::vector` represents a chunk of memory. You should
separate concerns and use `std::vector` to store data in your structs or
classes, `nsimd::pack` should only be used in computation kernels and nowhere
else especially not in structs or classes.

## Why is the speed-up of my code not as expected?

There are several reasons which can reduce the speed-up:

- Have you enabled compiler optimizations? You must enable all compiler
  optimizations (like `-O3`).

- Have you compiled in 64 bit mode? There is significant performance increase
  on architectures supporting 64 bit binaries.

- Is your code trivially vectorizable? Modern compilers can vectorize trivial
  code segments automatically. If you benchmark a trivial scalar code versus a
  vectorized code, the compiler may vectorize the scalar code, thereby giving
  similar performance to the vectorized version.

- Some architectures do not provides certains functionnalities. For example
  AVX2 chips do not provide a way to convert long to double. So using
  `nsimd::cvt<f64>` will produce an emulation for-loop in the resulting
  binary. To know which intrinsics are used by NSIMD you can consult
  <wrapped_intrinsics.md>.

## Why did my code segfaulted or crashed?

The most common cause of segfaults in SIMD codes is accessing non-aligned
memory. For best performance, all memory should be aligned. NSIMD includes an
aligned memory allocation function and an aligned memory allocator to help you
with this. Please refer to <tutorials.md> for details on how to
ensure that you memory is correctly aligned.

Another common cause is to read or write data beyond the allocated memory.
Do not forget that loading data into a SIMD vector will result in loading
16 bytes (or 4 floats) from memory. If this read occurs at the last 2 elements
of allocated memory then a segfault will be generated.

## My code compiled for AVX is not twice as fast as for SSE, why?

Not all SSE instructions have an equivalent AVX instruction. As a consequence
NSIMD uses two SSE operations to emulate the equivalent AVX operation.  Also,
the cycles required for certain instructions are not equal on both
architectures, for example, `sqrt` on `SSE` requires 13-14 cycles whereas
`sqrt` on `AVX` requires 21-28 cycles. Please refer
[here](https://www.agner.org/optimize/instruction_tables.pdf) for more
information.

Very few integer operations are supported on AVX, AVX2 is required for most
integer operations. If a NSIMD function is called on an integer AVX register,
this register will be split into two SSE registers and the equivalent
instruction called on both register. In the case, no speed-up will be observed
compared with SSE code. This is true also on POWER 7, where double is not
supported.

## I disassembled my code, and the generated code is less than optimal, why?

- Have you compiled in release mode, with full optimizations options?

- Have you used a 64 bit compiler?

- There are many SIMD related bugs across all compilers, and some compilers
  generate less than optimal code in some cases. Is it possible to update your
  compiler to a more modern compiler?

- We provide workarounds for several compiler bugs, however, we may have
  missed some. You may also have found a bug in `nsimd`. Please report this
  through issues on our github with a minimal code example. We responds quickly
  to bug reports and do our best to patch them as quickly as possible.

## How can I use a certain intrinsic?

If you require a certain intrinsic, you may search inside of NSIMD for it and
then call the relevant function or look at <wrapped_intrinsics.md>.

In rare cases, the intrinsic may not be included in NSIMD as we map the
intrinsic wherever it makes sense semantically. If a certain intrinsic does not
fit inside of this model, if may be excluded. In this case, you may call it
yourself, however, note this will not be portable. 

To use a particular intrinsic say `_mm_avg_epu8`, you can write the following.

```c++
nsimd::pack<u8> a, b, result;
result = nsimd::pack<u8>(_mm_avg_epu8(a.native_register(),
                                      b.native_register()));
```

## How do I convert integers/floats to/from logicals?

Use [`nsimd::to_mask`](api_to-mask.md) and
[`nsimd::to_logical`](api_to-logical.md).

## How about shuffles?

General shuffles are not provided by NSIMD. You can see
[issue 8 on github](https://github.com/agenium-scale/nsimd/issues/8). For now
we provide only some length agnostic shuffles such as zip and unzip, see
[the shuffle API](api.md) at the Shuffle section.

## Are there C++ STL like algorithms?

No. You are welcome to [contribute](contribute.md) to NSIMD and add them as
a NSIMD module. You should use
[expressions templates](module_tet1d_overview.md) instead. Strictly conforment
STL algorithms do not provide means to control for example the unroll factor
or the number of threads per block when compiling for GPUs.

## Are there masked operators in NSIMD?

Yes, we provide masked loads and stores, see [the api](api.md) at the
"Loads & stores" section. We also provide the
[`nsimd::mask_for_loop_tail`](api_mask-for-loop-tail.md) which computes the
mask for ending loops. But note that using these is not recommanded as on
most architectures there are no intrinsic. This will result in slow code. It
is recommanded to finish loops using a scalar implementation.

## Are there gathers and scatter in NSIMD?

Yes, we provide gathers and scatters, see [the api](api.md) at the
"Loads & stores" section. Note also that as most architectures do not provide
such intrinsics and so this could result in slow code.

## Why does not NSIMD recognize the target architecture automatically?

Autodetecting the SIMD extension is compiler/compiler version/cpu/system
dependant which means a lot of code for a (most likely buggy) feature which can
be an inconvenience sometimes. Plus some compilers do not permit this feature.
For example cf.
<https://www.boost.org/doc/libs/1_71_0/doc/html/predef/reference.html> and
<https://msdn.microsoft.com/en-us/library/b0084kay.aspx>. Thus a "manual"
system is always necessary.

## Why some operators have their names ending with an "1"?

This is because of C++ and our will not to use C++-useless-complicated stuff.
Taking the example with `if_else`, suppose that we have called it "if\_else"
without the "1". When working with packs, one wants to be able to use `if_else`
in this manner:

```c++
int main() {
  using namespace nsimd;
  
  typedef pack<int> pi;
  typedef pack<float> pf;

  int n;
  int *a, *b;      // suppose both points to n ints
  float *fa, *fb;  // suppose both points to n floats

  for (int i = 0; i < n; i += len()) {
    packl<int> cond = (loada<pi>(&a[i]) < loada<pi>(&b[i]));
    storea(&fb[i], if_else(cond, load<pf>(&fb[i]), set1<pf>(0.0f)));
  }

  return 0;
}
```

But this causes a compiler error, the overload of `if_else` is ambiguous.
Sure one can use many C++-ish techniques to tackle this problem but we chose
not to as the goal is to make the life of the compiler as easy as possible.
So as we want to favor the C++ advanced API as it is the most human readable,
users of the C and C++ base APIs will have to use `if_else1`.
