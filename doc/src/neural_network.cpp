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

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <nsimd/nsimd-all.hpp>

template <typename T, typename A> void print(std::vector<T, A> const &v) {
  std::cout << "{ ";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cout << (i == 0 ? "" : ", ") << v[i];
  }
  std::cout << " }";
}

template <typename T> void fct(T *first0, T *last0, T *first1) {
  for (; first0 < last0; ++first0, ++first1) {
    *first1 = T(1) / (T(1) + std::abs(*first0));
  }
}

template <typename T> void fct_nsimd(T *first0, T *last0, T *first1) {
  nsimd::pack<float> pack_1(T(1));

  size_t len = size_t(nsimd::len(nsimd::pack<T>()));
  for (; first0 + len < last0; first0 += len, first1 += len) {
    // Load current values
    auto pack = nsimd::loada<nsimd::pack<T> >(first0);
    // Computation
    pack = pack_1 / (pack_1 + nsimd::abs(pack));
    // Store
    nsimd::storea(first1, pack);
  }

  for (; first0 < last0; ++first0, ++first1) {
    *first1 = T(1) / (T(1) + std::abs(*first0));
  }
}

int main() {

  size_t N = 23;

  std::vector<float, nsimd::allocator<float> > in(N);
  std::vector<float, nsimd::allocator<float> > out_seq(N);
  std::vector<float, nsimd::allocator<float> > out_nsimd(N);

  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = i;
  }

  std::cout << "in = " << std::endl;
  print(in);
  std::cout << std::endl << std::endl;

  // Sequential
  fct(in.data(), in.data() + N, out_seq.data());
  std::cout << "out (sequential) = " << std::endl;
  print(out_seq);
  std::cout << std::endl << std::endl;

  // nsimd
  fct_nsimd(in.data(), in.data() + N, out_nsimd.data());
  std::cout << "out (nsimd) = " << std::endl;
  print(out_nsimd);
  std::cout << std::endl << std::endl;

  return 0;
}
