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

# Advanced Tutorial: Runtime Extension Selection

In this tutorial we will demonstrate the use of SIMD extension selection at
runtime.

## Introduction

The selection of SIMD extensions at runtime enables you to release one
executable which will be of the highest performance possible on many different
architectures. Without runtime extension selection, an executable released must
be compiled for the earliest generation of processor targeted, which is often
SSE2. This means that you are artificially restricting the performance of your
software for your clients which have more modern processors. NSIMD provides
everything you need to be able to target multiple architectures with one
binary.

The goal is to write the function doing computations once and compile it for
several architectures. In order to dispatch at runtime the name of the function
must be different for each architecture.  In C this is achived by using macros
and in C++ one can also use function overloading.

## How to select the correct extension at runtime in C

Let's consider this function

```C
void NSIMD_PP_CAT_2(funcname, NSIMD_SIMD)(float *a. float *b, int n) {
  int i, len;
  len = vlen(f32);
  for (i = 0; i + len <= n; i++) {
  }
}
```

In this section, we will demonstrate how to compile your program for several
different generations of x86 processors and select automatically the correct
version at runtime. In order to do this, you must add an extra argument to the
function you wish to call so that the correct function will be found at runtime.

```C++
void compute(float* a, float* b, float* res, int size, NSIMD_SIMD const& arch)
{
  // ...
}
```

The macro `BOOST_SIMD_DEFAULT_SITE` is set according to the architecture that 
the currect file is being compiled for. You must also declare a prototype for
each architecture that you wish to target:

```C++
void compute(float *a, float *b, float *res, int size, boost::simd::avx2_ const&);
void compute(float *a, float *b, float *res, int size, boost::simd::avx_ const&);
void compute(float *a, float *b, float *res, int size, boost::simd::sse4_2_ const&);
void compute(float *a, float *b, float *res, int size, boost::simd::sse4_1_ const&);
void compute(float *a, float *b, float *res, int size, boost::simd::sse3_ const&);
void compute(float *a, float *b, float *res, int size, boost::simd::sse2_ const&);
void compute(float *a, float *b, float *res, int size, boost::dispatch::cpu_ const&);
```

In this case, we are compiling a version of our function for each iteration of
x86 from sse2 to avx2. You must then call the correct version of your code,
depending on the architecture the binary is run on. The order of the conditions
in the if-block is very important as each generation of a processor in
**Boost.SIMD@**, inherits from the previous generation. This means that sse2 is
supported on an sse4.2 equipped processor, for example.
```C++
namespace bs = boost::simd;
if (bs::avx2.is_supported()) {
  compute(a.data(), b.data(), res.data(), size, bs::avx2_{});
} else if (bs::avx.is_supported()) {
  compute(a.data(), b.data(), res.data(), size, bs::avx_{});
} else if (bs::sse4_2.is_supported()) {
  compute(a.data(), b.data(), res.data(), size, bs::sse4_2_{});
} else if (bs::sse4_1.is_supported()) {
  compute(a.data(), b.data(), res.data(), size, bs::sse4_1_{});
} else if (bs::sse2.is_supported()) {
  compute(a.data(), b.data(), res.data(), size, bs::sse3_{});
}
```

Once you have compiled your file for each required architecture, you must link
your executable with each version of your code. On Linux, using cmake, this is
done as follows:
```CMake
cmake_minimum_required(VERSION 2.8)
set(CWD ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
set(CMAKE_BUILD_TYPE Release)
include_directories($ENV{BOOST_ROOT})
include_directories($ENV{SIMD_ROOT})
add_library(test_sse SHARED compute.cpp)
add_library(test_sse2 SHARED compute.cpp)
add_library(test_sse3 SHARED compute.cpp)
add_library(test_sse4.1 SHARED compute.cpp)
add_library(test_sse4.2 SHARED compute.cpp)
add_library(test_avx SHARED compute.cpp)
add_library(test_avx2 SHARED compute.cpp)
set_target_properties(test_sse2 PROPERTIES COMPILE_FLAGS "-msse2")
set_target_properties(test_sse3 PROPERTIES COMPILE_FLAGS "-msse3")
set_target_properties(test_sse4.1 PROPERTIES COMPILE_FLAGS "-msse4.1")
set_target_properties(test_sse4.2 PROPERTIES COMPILE_FLAGS "-msse4.2")
set_target_properties(test_avx PROPERTIES COMPILE_FLAGS "-mavx")
set_target_properties(test_avx2 PROPERTIES COMPILE_FLAGS "-mavx2")
add_executable(runtime_extension runtime_extension.cpp)
target_link_libraries(runtime_extension test_sse2 test_sse3 test_sse4.1 test_sse4.2 test_avx test_avx2)
```

Download full source code:
- [runtime_simd.cpp](../src/runtime_simd.cpp)
