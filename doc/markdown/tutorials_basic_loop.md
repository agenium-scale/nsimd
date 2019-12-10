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

# Basic Tutorial: A SIMD Loop

In this tutorial we will demonstrate how to construct a SIMD loop to subtract a
constant from a vector of input data.


## Transforming a Scalar Loop Into a SIMD Loop

In this tutorial we will demonstrate how to transform the following scalar loop
into a SIMD loop.

The scalar loop is:
@[INCLUDE_CODE:L53:L55](../src/loop.cpp)

Before constructing the `nsimd` loop, ensure the input and the output data are
aligned.
@[INCLUDE_CODE:L39:L41](../src/loop.cpp)

We now construct our loop. Note how `i` is incremented by `len`. `len` is the
lenght of the `nsimd::pack<T>`. For each iteration of the loop, we process
several elements of the input data. You can also note the condition, some
elements may not be computed.
@[INCLUDE_CODE:L66:L66](../src/loop.cpp)

In order to process the input data, we must oad the data from memory using
`nsimd::loada` (`nsimd::loadu` if the data are not aligned) and then store it
afterwards back in memory using using `nsimd::storea` (`nsimd::storeu` if the
data are not aligned).
@[INCLUDE_CODE:L64:L70](../src/loop.cpp)

If the input data is not a multiple of the lenght of the `nsimd::pack<T>`,
we have to handle this case by adding a scalar loop after the `nsimd` loop.
@[INCLUDE_CODE:L64:L73](../src/loop.cpp)

Download full source code:
- [loop.cpp](../src/loop.cpp)
