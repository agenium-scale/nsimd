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


# NSIMD fixed point module

## Description

This module implements a fixed-point numbers support for the `nsimd` library.
Fixed-point numbers are integer types used to represent decimal numbers. A number `lf` 
of bits are used to encode its integer part, and `rt` bits are used to encode its 
fractional part.

The fixed_point module uses the templated type `nsimd::fixed_point::fp_t<lf, rt>` to 
represent a fixed_point number. All the basic floating-point arithmetic operaors have 
been defined, therefore fp_t elements can be manipulated as normal numbers.
The fixed_point module will use a `int8_t`, `int16_t`, or `int32_t` integer type for 
storage, depending on the value of `lf + 2 * rt`. 

All the functions of the module are under the namespace `nsimd::fixed_point`, 
and match the same interface than `nsimd`.

The `fp_t` struct type is defined in `fixed.hpp`, and the associated simd `fpsimd_t` 
struct type is defined in `simd.hpp`.

The modules redefines the `nsimd` pack type for fixed-point numbers, templated with `lf` 
and `rt` :

```C++
namespace nsimd {
namespace fixed_point {
template <uint8_t lf, uint8_t rt>
struct pack;
} // namespace fixed_point
} // namespace nsimd
```

Then, the pack can be manipulated as an `nsimd` pack like other scalar types. 

## Compatibility

The fixed point module is a C++ only API, compatible with the C++98 standard.
It has the same compilers and hardware support than the main `nsimd` API 
(see the [API index](../../index.md)).

## Example

Here is a minimal example(main.cpp) :

```C++
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <nsimd/modules/fixed_point.hpp>

float rand_float() {
  return 4.0f * ((float) rand() / (float) RAND_MAX) - 2.0f;        
}

int main() {
  // We use fixed point numbers with 8 bits of integer part and 8 bits of 
  // decimal part. It will use a 32 bits integer for internal storage.
  typedef nsimd::fixed_point::fp_t<8, 8> fp_t;
  typedef nsimd::fixed_point::pack<fp_t> fp_pack_t;
  
  const size_t v_size = nsimd::fixed_point::len(fp_t());

  fp_t *input0 = (fp_t*)malloc(v_size * sizeof(fp_t));
  fp_t *input1 = (fp_t *)malloc(v_size * sizeof(fp_t));
  fp_t *res = (fp_t *)malloc(v_size * sizeof(fp_t));
  
  // Input and output initializations 
  for(size_t i = 0; i < nsimd::fixed_point::len(fp_t()); i++) {
    input0[i] = fp_t(rand_float());
    input1[i] = fp_t(rand_float());
  }
  
  fp_pack_t v0 = nsimd::fixed_point::loadu<fp_pack_t>(input0);
  fp_pack_t v1 = nsimd::fixed_point::loadu<fp_pack_t>(input1);
  fp_pack_t vres = nsimd::fixed_point::add(v0, v1);
  nsimd::fixed_point::storeu(res, vres);
  
  for(size_t i = 0; i < nsimd::fixed_point::len(fp_t()); i++) {
    std::cout << float(input0[i]) << " | "
      << float(input1[i]) << " | "
      << float(res[i]) << "\n";
  }
  std::cout << std::endl;
  
  return EXIT_SUCCESS;
}

```

To test with avx2 run : 
```bash
export NSIMD_ROOT=<path/to/simd>
g++ -o main -I$NSIMD_ROOT/include -mavx2 -DNSIMD_AVX2 main.cpp
./main
```

The console output will look like this : 
```console
$>./main 
1.35938 | -0.421875 | 0.9375
1.13281 | 1.19531 | 2.32812
1.64844 | -1.21094 | 0.4375
-0.660156 | 1.07422 | 0.414062
-0.890625 | 0.214844 | -0.675781
-0.0898438 | 0.515625 | 0.425781
-0.539062 | 0.0546875 | -0.484375
1.80859 | 1.66406 | 3.47266
```
        