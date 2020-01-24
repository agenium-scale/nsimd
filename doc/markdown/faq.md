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

# Frequently Asked Questions

<!-- TODO -->

<!--## Is It Good Practice to Use a Pack as an Array?

The main element of `nsimd` is the `nsimd::pack` class. `nsimd::pack` is an
abstraction over a block of `N` elements of type `T`, quite similar to
`std::array`. The main semantic difference is that `boost::simd::pack` is
implemented as the best hardware specific type able to store this amount of data
which may be a simple scalar array, a single SIMD register or a tuple of SIMD
registers depending on `N` and `T`.

In general, the good practice is to store data in regular, dynamic allocated
data block and apply your algorithms over those data using pack and related
functions.-->


## Why Is the Speed-Up of My Code Not as Expected?

There are several factors which can reduce the speed-up obtained using `nsimd`.
- Have you enabled compiler optimizations? You must enable all compiler
  optimizations (like `-O3`).
- Have you compiled in 64 bit mode? There is significant performance increase on
  architectures supporting 64 bit binaries.
- Is you algorithm memory bound? Computational problems can often be classed as
  either compute-bound or memory-bound. Memory bound problems reach the limits
  of the system bandwidth transfering data between the memory and the processor,
  while compute-bound problems are limited by the processor's calculation
  ability.
  Processors have often several hierarchies of caches. The time taken to
  transfer data from the cache to a register varies depending on the cache.
  When the required data is not in the cache, it is referred to as a cache miss.
  Cache-misses are very expensive. Data is transferred from memory to the cache
  in cache-lines sized chunks. On modern x86 machines, a cache-line is 64 bytes
  or twice the size of an `AVX` register. It is therefore highly advantageous to
  use all data loaded into cache.
  The following loop is clearly memory bound as all of the time is spent loading
  and storing the data. The cost of the computation is negligible compared to
  that of the memory accesses.
  ```
  for (int i = 0; i < size; i += pack_t::static_size) {
    pack_t v0(&data[i]);
    v0 = v0 * 2;
    bs::aligned_store(v0, &output[i]);
  }
  ```
  The following loop is compute-bound. As most of the time is spent calculating
  exp, significant speed-up is observed when this code is vectorized.
  ```
  t0 = high_resolution_clock::now();
  for (int i = 0; i < size; i += pack_t::static_size) {
    pack_t v0(&data[i]);
    v0 = bs::exp(bs::exp(v0));
    bs::aligned_store(v0, &output[i]);
  }
  ```
- Is your code trivially vectorizable? Modern compilers can vectorize trivial
  code segments automatically. If you benchmark a trivial scalar code versus a
  vectorized code, the compiler may vectorize the scalar code, thereby giving
  similar performance to the vectorized version.
- Is your algorithm vectorizable?


## Why Did My Code Seg-Faulted or Crashed?

The most common cause of seg-faults in SIMD codes is accessing non-aligned
memory. For best performance, all memory should be aligned. `nsimd` includes an
aligned memory allocation function and an aligned memory allocator to help you
with this. Please refer to [tutorials](tutorials.md) for details on how to
ensure that you memory is correctly aligned.


<!--## I Tried to Use a Comparison Operator and My Code Failed to Compile

The most common reason for this is that the two packs being compared are not of
the same type. Another common reason is that the return type is incorrect. Using
auto is one way of preventing this error, however, it is best to be aware of the
types you are using. Comparison operators in `nsimd` are of two types, either
vectorized comparison, where the results is a vector of logical with the same
cardinal as the input vectors, or a reduction comparison, where the result is a
bool.-->


## How to Measure the Speed-Up of SIMD code?

There are several ways to measure the speed-up of your code. You may use the
[`Google Benchmark`](https://github.com/google/benchmark) to benchmark your code
segment. This allows you to measure the execution time of your code.

Otherwise, you may use standard timing routines. C++11 adds `std::chrono`. You
can use
[`std::chrono::high_resolution_clock`](https://en.cppreference.com/w/cpp/chrono/high_resolution_clock)
if `std::chrono::high_resolution_clock::is_steady` is true. If not, use
[`std::chrono::steady_clock`](https://en.cppreference.com/w/cpp/chrono/steady_clock) otherwise.
In C and C++98, use [`clock_gettime`](http://man7.org/linux/man-pages/man2/clock_gettime.2.html)
with `CLOCK_MONOTONIC` or equivalent for non POSIX.1-2001 system. In order to
accurately benchmark your code, there are several points to consider.
- Your input should be sufficiently large. This is to eliminate cache effects.
- Your code is compiled in release mode, with all optimizations enabled and
  debug information not included.
- You should measure several times and use the median (better than the average).

A typical case where a benchmark could give inaccurate results is where the
input is not large enough to fill the cache and a scalar and SIMD code segment
are individually benchmarked, one after the other. In this case, all of the data
will be loaded into the cache during the first bench, and will be available for
the second bench, therefore decreasing the execution time of the second bench.
Also, if you measure the code segment multiple times, the data from the first
execution will already be in the cache.

<!-- TODO Talk about / mention cache warming -->


## My Code Compiled for `AVX` Is Not Twice As Fast As for `SSE`

Not all `SSE` instructions have an equivalent `AVX` instruction. Also, the
cycles required for certain instructions are not equal on both architectures,
for example, `sqrt` on `SSE` requires 13-14 cycles whereas `sqrt` on `AVX`
requires 21-28 cycles. Please refer
[here](https://www.agner.org/optimize/instruction_tables.pdf) for more
information.

Very few integer operations are supported on `AVX`, `AVX2` is required for most
integer operations. If a `nsimd` function is called on an integer `AVX`
register, this register will be split into two `SSE` registers and the
equivalent instruction called on both register. In the case, no speed-up will be
observed compared with `SSE` code. This is true also on `POWER 7`, where
`double` is not supported.


## I Disassembled My Code, and the Generated Code Is Less Than Optimal

- Have you compiled in release mode, with full optimizations options?
- Have you used a 64 bit compiler?
- There are many SIMD related bugs across all compilers, and some compilers
  generate less than optimal code in some cases. Is it possible to update your
  compiler to a more modern compiler?
- We provide workarounds for several compiler bugs, however, we may have
  missed some. You may also have found a bug in `nsimd`. Please report this
  through issues on our github with a minimal code example. We responds quickly
  to bug reports and do our best to patch them as quickly as possible.


## How Can I Use a Certain Intrinsic?

If you require a certain intrinsic, you may search inside of `nsimd` for it and
then call the relevant function.

In rare cases, the intrinsic may not be included in `nsimd` as we map the
intrinsic wherever it makes sense semantically. If a certain intrinsic does not
fit inside of this model, if may be excluded. In this case, you may call it
yourself, however, this will not be portable. 
