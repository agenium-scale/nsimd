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

#include <iomanip>
#include <iostream>
#include <vector>

#include <nsimd/nsimd-all.hpp>

template <typename A> void print(std::vector<f32, A> const &v) {
  std::cout << "{ ";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cout << (i == 0 ? "" : ", ") << std::setw(3) << v[i];
  }
  std::cout << " }";
}

int main() {

  // Default-construct pack
  std::cout << "Default-construct pack" << std::endl;
  {
    nsimd::pack<float> pack;
    std::cout << pack << std::endl;
    // Possible output: { 5.88449e-39, 0, 5.88353e-39, 0 }
  }
  std::cout << std::endl;

  // Pack with same value
  std::cout << "Pack with same value" << std::endl;
  {
    nsimd::pack<float> pack(42);
    std::cout << pack << std::endl;
    // Output for SSE2: { 42, 42, 42, 42 }
  }
  std::cout << std::endl;

  // Pack from aligned memory
  std::cout << "Pack from aligned memory" << std::endl;
  {
    size_t N = size_t(nsimd::len(nsimd::pack<float>()));
    std::vector<float, nsimd::allocator<float> > data(N);
    for (size_t i = 0; i < N; ++i) {
      data[i] = float(i);
    }

    nsimd::pack<float> pack = nsimd::loada<nsimd::pack<float> >(&data[0]);
    std::cout << pack << std::endl;
    // Output for SSE2: { 0, 1, 2, 3 }
  }
  std::cout << std::endl;

  // Pack from unaligned memory (not recommended)
  std::cout << "Pack from unaligned memory (not recommended)" << std::endl;
  {
    size_t N = size_t(nsimd::len(nsimd::pack<float>()));
    std::vector<float> data(N);
    for (size_t i = 0; i < N; ++i) {
      data[i] = float(i);
    }

    nsimd::pack<float> pack = nsimd::loadu<nsimd::pack<float> >(&data[0]);
    std::cout << pack << std::endl;
    // Output for SSE2: { 0, 1, 2, 3 }
  }
  std::cout << std::endl;

  // + *
  {
    nsimd::pack<float> pack0(7);
    nsimd::pack<float> pack1(21);
    nsimd::pack<float> r = (pack0 + pack1) * 2.0f;
    std::cout << "(" << pack0 << " + " << pack1 << ") * 2.0f = " << r
              << std::endl;
    // Output for SSE2:
    // ({ 7, 7, 7, 7 } + { 21, 21, 21, 21 }) * 2.0f = { 56, 56, 56, 56 }
  }
  std::cout << std::endl;

  // Store pack in aligned memory
  std::cout << "Store pack in aligned memory" << std::endl;
  {
    size_t N = size_t(nsimd::len(nsimd::pack<float>()));
    std::vector<float, nsimd::allocator<float> > data(N);

    nsimd::pack<float> pack(42);
    nsimd::storea(&data[0], pack);

    std::cout << "{";
    for (size_t i = 0; i < N; ++i) {
      std::cout << (i == 0 ? " " : ", ") << data[i];
    }
    std::cout << " }" << std::endl;
    // Output for SSE2: { 42, 42, 42, 42 }
  }
  std::cout << std::endl;

  // Store pack in unaligned memory (not recommended)
  std::cout << "Store pack in unaligned memory (not recommended)" << std::endl;
  {
    size_t N = size_t(nsimd::len(nsimd::pack<float>()));
    std::vector<float> data(N);

    nsimd::pack<float> pack(42);
    nsimd::storeu(&data[0], pack);

    std::cout << "{";
    for (size_t i = 0; i < N; ++i) {
      std::cout << (i == 0 ? " " : ", ") << data[i];
    }
    std::cout << " }" << std::endl;
    // Output for SSE2: { 42, 42, 42, 42 }
  }
  std::cout << std::endl;

  return 0;
}
