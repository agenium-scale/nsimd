// Copyright (c) 2019 Agenium Scale
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#include <ctime>
#include <cstdlib>
#include <iostream>
#include <nsimd/modules/fixed_point.hpp>

float rand_float() {
  return 4.0f * ((float) rand() / (float) RAND_MAX) - 2.0f;        
}

int main() {
  // We use fixed point numbers with 8 bits of integer part and 8 bits of 
  // decimal part. It will use 32 bits integers for internal storage.
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
