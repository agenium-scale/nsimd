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

# Advanced Tutorial: Mandelbrot Set

In this tutorial we will calculate a Mandelbrot set using `nsimd`. This example
is interesting because the number of iterations required for each pixel of the
image is different. This means that we need a method of exiting the calculation
loop when all of the pixels currently being worked on have converged.


## Mandelbrot Set

The calculation of the Mandelbrot set is an example of a problem which is
completely compute bound.

Here is the scalar version:
@[INCLUDE_CODE:L71:L127](../src/mandelbrot_set.cpp)

This code is vectorized as follows:
@[INCLUDE_CODE:L129:L185](../src/mandelbrot_set.cpp)

With the default paramaters, we should obtain this picture:  
[![mandelbrot_nsimd](img/mandelbrot_nsimd_640.jpg)](img/mandelbrot_nsimd.jpg)


<!--The function of interest here is bs::if_inc, which increments each element of the iter vector which has not yet converged. This allows us to continue our calculation on the relevant elements.

We have also used the function bs::any returns a boolean value if any of its parameter element is non zero, We have also used the bs::sqr function which squares its argument and the bs::fma function (fused multiply add) which can accelerate and increase accuracy "a*b+c" computations on some architectures.-->

Download full source code:
- [mandelbrot_set.cpp](../src/mandelbrot_set.cpp)

<!-- Add bench and explanations like in the original bsimd tutorial  -->
