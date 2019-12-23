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


# NSIMD fixed-point module

## Description

This module implements a fixed-point numbers support for the NSIMD library.
Fixed-point numbers are integer types used to represent decimal numbers. A number `lf` 
of bits are used to encode its integer part, and `rt` bits are used to encode its 
fractional part.

The fixed_point module uses the templated type `nsimd::fixed_point::fp_t<lf, rt>` to 
represent a fixed_point number. All the basic floating-point arithmetic operaors have 
been defined, therefore fp\_t elements can be manipulated as normal numbers.
The fixed\_point module will use a `int8_t`, `int16_t`, or `int32_t` integer type for 
storage, depending on the value of `lf + 2 * rt`. 

All the functions of the module are under the namespace `nsimd::fixed_point`, 
and match the same interface than NSIMD.

The `fp_t` struct type is defined in `fixed.hpp`, and the associated simd `fpsimd_t` 
struct type is defined in `simd.hpp`.

The modules redefines the NSIMD pack type for fixed-point numbers, templated with `lf` 
and `rt` :

```C++
template <uint8_t lf, uint8_t rt>
struct pack;
```

Then, the pack can be manipulated as an NSIMD pack like other scalar types. Here is 
a minimal example :

```C++
// Assumes that NSIMD is in your include path
#include <iostream>
#include <nsimd/modules/fixed_point.hpp>

int main() {
  typedef nsimd::fixed_point::fp_t<8, 8> fp_t;
  typedef nsimd::fixed_point::pack<fp_t> vec_t;
  
  fp_t *input0;
  fp_t *input1;
  fp_t *res;
  
  // Input and output initializations 
  // We assume that a function float rand_float(); has been 
  // previously defined
  for(size_t i = 0; i < nsimd::fixed_point::len(fp_t()); i++) {
    input0 = fp_t(rand_float());
    input1 = fp_t(rand_float());
  }
  
  vec_t v0 = nsimd::fixed_point::loadu<vec_t>(input0);
  vec_t v1 = nsimd::fixed_point::loadu<vec_t>(input1);
  vec_t vres = nsimd::fixed_point::add(input0, input1);
  nsimd::fixed_point::storeu(res, vres);
  
  for(size_t i = 0; i < nsimd::fixed_point::len(fp_t()); i++) {
    std::cout << float(res[i]) << " ";
  }
  std::cout << std::endl;
  
  return EXIT_SUCCESS;
}

```
        