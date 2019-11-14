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

# Basic Tutorial: Mathematical Functions

In this tutorial we will demonstrate the use of SIMD mathematical functions
such as `nsimd::sin` and `nsimd::cos`.


## SIMD Mathematical Functions

Mathematical functions can be very expensive, so using a vectorized version of
these functions is of great benefit, especially if you have a large input array.
`nsimd` includes vectorized versions of all common mathematical functions. The
developers of `nsimd` take numerical precisions very seriously, so all of these
functions are extremely accurate despite being highly optimized.

The standard way to calculate the sine or cosine of a vector of data is to loop
over the data and calculate the sine and cosine of each element:
@[INCLUDE_CODE:L57:L60](../src/sin_cos.cpp)

This type of calculation is the perfect candidate for vectorization!
@[INCLUDE_CODE:L72:L79](../src/sin_cos.cpp)

In this example we are calculating the sine and the cosine of the same input vector X. The calculations of a sine and cosine contain many common steps, so it is possible to calculate them both simultaneously and thus save precious cpu cycles!
@[INCLUDE_CODE:L92:L98](../src/sin_cos.cpp)

Download full source code:
- [sin_cos.cpp](../src/sin_cos.cpp)
