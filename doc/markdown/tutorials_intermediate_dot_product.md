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

# Intermediate Tutorial: Writing a Dot Product the SIMD Way

In this tutorial we will show how data can be processed using `nsimd` by writing
a naive dot product using `nsimd`.


## Transforming a Scalar Reduction Into a SIMD Reduction

This tutorial will present how data can be processed using `nsimd` by writing a
naive dot product using `nsimd`.

A simple sequential, scalar dot product could be coded like this:
@[INCLUDE_CODE:L35:L41](../src/dot_product.cpp)
`dot` simply iterates over data pointed by `first0` and `first1`, computes the
product of this data and then sums them.


## Transition From Scalar to SIMD Code

In this case the algorithm is clearly vectorizable, let's unroll the loop
arbitrarily to show the inherent data parallelism:
@[INCLUDE_CODE:L43:L62](../src/dot_product.cpp)

The algorithm is split into two parts:
- First, we loop over each element inside both datasets and multiply them.
- Then, we sum the intermediate values into the final result.

By unrolling this pattern arbitrarily, we expose the fact that the
multiplication between the two dataset is purely "vertical" and so, is
vectorizable. The sum of the partial results itself is an "horizontal"
operation, i.e. a vectorizable computation operating across the elements of a
single vector.


## Building a SIMD loop

We are now going to use boost::simd::pack to vectorize this loop. The main idea
is to compute partial sums inside an instance of `nsimd::pack<T>` and then
perform a final summation. To do so, we will use `nsimd::load` to load data from
`first0` and `first1`, process these `nsimd::pack<T>` instances using the proper
operators and then advance the pointers by the length of `nsimd::pack<T>`.
@[INCLUDE_CODE:L64:L81](../src/dot_product.cpp)

That's it! Look at how similar the computation code is to the scalar version, we
simply jump over data using a larger step size and at the end we account for the
data which does not fit inside a SIMD vector.

NOTE:  
The code line `tmp += v0 * v1;` may replaced by `tmp = nsimd::fma(x1, x2, tmp);`
which may generate even more efficient code as many processors have special
instructions for performing this operation. If the target processor is not
equipped such an instruction, high quality vectorized code will nevertheless be
generated.

Download full source code:
- [dot_product.cpp](../src/dot_product.cpp)
