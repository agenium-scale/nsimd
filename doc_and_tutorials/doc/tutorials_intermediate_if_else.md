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

# Intermediate Tutorial: SIMD Branching

In this tutorial we will examine how to handle branches in SIMD programs.


## Branching

One of the fundamental principles of SIMD programming is that the same operation
must be performed on each element of the SIMD vector. At first glance, this
means that vectorized programs do not support conditional statements, however
this is not true. Let's take a common image processing operation known as
thresholding as a counter example. Thresholding is an operation to separate a
digital image into its background and foreground pixels. Any pixel whose value
is less than the chosen threshold is considered to be a background pixel and
conversely, any pixel whose value is greater or equal to the threshold is
considered to be a foreground pixel. The scalar version of this algorithm is
simply:
@[INCLUDE_CODE:L111:L117](../src/threshold.cpp)
Here we have a very clear branch in the code. How can we vectorize this loop?


## Logical SIMD Type

In this example, we need a new type, `nsimd::packl<T>`, which is an abstraction
for the equivalent of a `bool` on a particular architecture. Therefore, a
`nsimd::packl<T>` a pack of `bool`. This abstraction is necessary to ensure that
any code written using `nsimd` is portable due to the differences between how
various processors handle operations requiring logical values.

The return type of a comparison operation in C++ is a `bool`, therefore the
return type of a comparison operation with `nsimd::pack<T>` is a
`nsimd::packl<T>`. This `nsimd::packl<T>` is then used to generate a SIMD vector
of `0` and/or `255` using the function `nsimd::if_else1`. All that's left to do
now is to store this vector in its correct location in memory.
@[INCLUDE_CODE:L123:L146](../src/threshold.cpp)

Download full source code:
- [threshold.cpp](../src/threshold.cpp)

If you use
[this image](https://en.wikipedia.org/wiki/File:Pavlovsk_Railing_of_bridge_Yellow_palace_Winter.jpg)
as input data:  
![640px-Pavlovsk_Railing_of_bridge_Yellow_palace_Winter](img/640px-Pavlovsk_Railing_of_bridge_Yellow_palace_Winter.jpg)

You should obtain:  
![threshold_nsimd](img/threshold_nsimd.jpg)
