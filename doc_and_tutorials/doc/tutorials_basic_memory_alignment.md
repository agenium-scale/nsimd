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

# Basic Tutorial: Memory Alignment

In this tutorial we will discuss memory alignment and demonstrate how to ensure
that your memory is always correctly aligned.


## What is Memory Alignment

In order to fully understand memory alignment and how to maximize performance of
your software by ensuring memory alignment, please read this excellent
[article](https://developer.ibm.com/articles/pa-dalign/) by IBM. Although this
article focuses on PowerPC processors, its conclusions are valid on many other
architectures. `nsimd` gives you portability across many architectures, in order
to have the best performance possible across a range of architectures, it is
worth considering the alignment requirements of all target architectures.


## How to Align Memory Manually?

In C11 and C++17, you can allocate aligned memory using
[std::aligned_alloc](https://en.cppreference.com/w/cpp/memory/c/aligned_alloc).

In C++11, you can get a pointer on aligned already allocated memory with
[`std::align`](https://en.cppreference.com/w/cpp/memory/align).
You can create aligned static storage using
[`std::aligned_storage`](https://en.cppreference.com/w/cpp/types/aligned_storage).

In C++ with `nsimd`, use `nsimd::allocator<T>` allocator with
[`std::vector`](https://en.cppreference.com/w/cpp/container/vector) to allocate
aligned memory.
@[INCLUDE_CODE:L56:L56](../src/hello_world.cpp)

In C with `nsimd`, use `nsimd_aligned_alloc` function to allocate aligned memory.
@[INCLUDE_CODE:L59:L59](../src/hello_world.c)
Do not forget to free the memory with `nsimd_aligned_free` function.
@[INCLUDE_CODE:L207:L208](../src/hello_world.c)

Download full source code:
- [sin_cos.cpp](../src/sin_cos.cpp)
