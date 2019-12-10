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

# Intermediate Tutorial: N-Body Problem

In this tutorial we will show you how to optimize an n-body simulation using
`nsimd`.


## N-Body Problem

This problem is concerned with the calculation of the individual motion of
celestial objects caused by gravity. This is an example of a problem of
complexity. The formula to calculate the gravitational force between two objects
is:
<!-- \( F = { Gm_{i}m_{i}(q_{i} - q_{j}) \over ||q_{i} - q_{j}||^{3} } \) -->

Where:
- \(q_{i}\) is the position vector of object *i*
- \(m_{i}\) is the mass of object *i*
- \(G\) is the gravitational constant


## Calculation

The scalar calculation of this problem is done as follows:
@[INCLUDE_CODE:L55:L112](../src/nbody.cpp)

The `particules_t` structure is:
@[INCLUDE_CODE:L36:L53](../src/nbody.cpp)

Note the use of a small constant, \(\epsilon\). This is referred to as
*softening*, a numerical trick to prevent division by zero if two particles are
too close together.

This calculation is trivially vectorizable as follows:
@[INCLUDE_CODE:L114:L185](../src/nbody.cpp)

NOTE:  
All constants are loaded into SIMD vectors outside of the main loop, so that
these vectors are not generated each iteration of the calculation.

It is not possible to use aligned stores in the interior loop as these stores
will not be aligned for each store.

Download full source code:
- [nbody.cpp](../src/nbody.cpp)

<!-- Add bench and explanations like in the original bsimd tutorial  -->
