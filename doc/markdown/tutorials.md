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

# Tutorials

The general principles of `nsimd` are introduced in the following tutorials.
- using the pack abstraction of a native SIMD vector.
- compiling a program written using `nsimd`.
- using SIMD specific idioms such as reduction, branching and shuffling
- SIMD runtime dispatching


## Basic Tutorials

1. [The SIMD Pack](tutorials_basic_pack.md)
2. [A Basic SIMD Loop](tutorials_basic_loop.md)
3. [Memory Alignment](tutorials_basic_memory_alignment.md)
<!-- 4. [Using Mathematical Functions](tutorials_basic_mathematical_functions.md) -->


## Intermediate Tutorials

1. [Writing a Dot Product the SIMD Way](tutorials_intermediate_dot_product.md)
2. [SIMD Branching](tutorials_intermediate_if_else.md)
4. [Evaluation of a Neural Network](tutorials_intermediate_neural_network.md)
5. [Evaluation of the N-Body Problem](tutorials_intermediate_nbody.md)


## Advanced Tutorials

1. [Runtime Extension Selection](tutorials_advanced_runtime_simd.md)
<!-- 2. [AoS (Array of Structures) and SoA (Structure of Arrays)](tutorials_advanced_aos_soa.md) -->
4. [Vectorizing the Mandelbrot Set Calculation](tutorials_advanced_mandelbrot_set.md)
<!-- 4. [Vectorizing the Julia Set Calculation](tutorials_advanced_julia_set.md) -->
