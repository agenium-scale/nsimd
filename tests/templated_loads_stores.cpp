/*

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

*/

#include <nsimd/nsimd-all.hpp>
#include <iostream>
#include <cstdlib>

// ----------------------------------------------------------------------------

float *getmem(nsimd::aligned, int sz) {
  float *ret = (float *)nsimd::aligned_alloc(sz);
  if (ret == NULL) {
    std::cerr << "ERROR: cannot malloc aligned memory" << std::endl;
  }
  return ret;
}

float *getmem(nsimd::unaligned, int sz) {
  return getmem(nsimd::aligned(), 2 * sz) + 1;
}

// ----------------------------------------------------------------------------

template <typename Alignment> int test() {
  using namespace nsimd;

  f32 *buf = getmem(Alignment(), NSIMD_MAX_LEN(f32));
  memset((void *)buf, 0, NSIMD_MAX_LEN(f32));

  pack<f32> v =
      masko_load<Alignment>(packl<f32>(false), buf, set1<pack<f32> >(1.0f));

  if (any(v != 1.0f)) {
    std::cerr << "[1]: v != [ 1.0f ... 1.0f ]" << std::endl;
    return -1;
  }

  v = load<pack<f32>, Alignment>(buf);
  if (any(v != 0.0f)) {
    std::cerr << "[2]: v != [ 0.0f ... 0.0f ]" << std::endl;
    return -1;
  }

  v = set1<pack<f32> >(1.0f);
  store<Alignment>(buf, v);
  for (int i = 0; i < len(pack<f32>()); i++) {
    if (buf[i] != 1.0f) {
      std::cerr << "[3]: buf != [ 1.0f ... 1.0f ]" << std::endl;
      return -1;
    }
  }

  v = set1<pack<f32> >(2.0f);
  mask_store<Alignment>(packl<f32>(false), buf, v);
  for (int i = 0; i < len(pack<f32>()); i++) {
    if (buf[i] != 1.0f) {
      std::cerr << "[4]: buf != [ 1.0f ... 1.0f ]" << std::endl;
      return -1;
    }
  }

  v = maskz_load<Alignment>(packl<f32>(false), buf);
  if (any(v != 0.0f)) {
    std::cerr << "[5]: v != [ 0.0f ... 0.0f ]" << std::endl;
    return -1;
  }

  return 0;
}

// ----------------------------------------------------------------------------

int main() { return test<nsimd::aligned>() || test<nsimd::unaligned>(); }
