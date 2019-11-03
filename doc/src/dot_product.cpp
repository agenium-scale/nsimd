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

#include <cmath>
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

template <typename T> T dot(T *first0, T *last0, T *first1) {
  T v(0);
  for (; first0 < last0; ++first0, ++first1) {
    v += (*first0) * (*first1);
  }
  return v;
}

template <typename T> T dot_unroll2(T *first0, T *last0, T *first1) {
  T v0(0);
  T v1(0);

  for (; first0 + 1 < last0;) {
    v0 += (*first0) * (*first1);
    ++first0;
    ++first1;

    v1 += (*first0) * (*first1);
    ++first0;
    ++first1;
  }

  T r = v0 + v1;

  r += dot(first0, last0, first1);

  return r;
}

template <typename T> T dot_nsimd(T *first0, T *last0, T *first1) {
  nsimd::pack<T> v(0);

  size_t len = size_t(nsimd::len(nsimd::pack<T>()));
  for (; first0 + len < last0; first0 += len, first1 += len) {
    // Load current values
    auto v0 = nsimd::loada<nsimd::pack<T> >(first0);
    auto v1 = nsimd::loada<nsimd::pack<T> >(first1);
    // Computation
    v = v + (v0 * v1);
  }

  T r = nsimd::addv(v); // horizontal SIMD vector summation

  r += dot(first0, last0, first1);

  return r;
}

int main() {

  size_t N = 23;

  std::vector<float, nsimd::allocator<float> > in0(N);
  std::vector<float, nsimd::allocator<float> > in1(N);

  for (size_t i = 0; i < in0.size(); ++i) {
    in0[i] = i;
    in1[i] = N - i;
  }

  std::cout << "in0 = " << std::endl;
  print(in0);
  std::cout << std::endl;
  std::cout << "in1 = " << std::endl;
  print(in1);
  std::cout << std::endl << std::endl;

  // Sequential
  float out_dot = dot(in0.data(), in0.data() + N, in1.data());
  std::cout << "out (sequential) = " << out_dot << std::endl << std::endl;

  // Unroll 2
  float out_dot_unroll2 = dot_unroll2(in0.data(), in0.data() + N, in1.data());
  std::cout << "out (unroll2) = " << out_dot_unroll2 << std::endl << std::endl;

  // nsimd
  float out_dot_nsimd = dot_nsimd(in0.data(), in0.data() + N, in1.data());
  std::cout << "out (nsimd) = " << out_dot_nsimd << std::endl << std::endl;

  return 0;
}
