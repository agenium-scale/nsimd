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

int main() {

  double const pi = std::atan(1.0) * 4.0;

  size_t N = 16;

  std::vector<float, nsimd::allocator<float> > in(N);
  std::vector<float> sin_seq(N);
  std::vector<float> cos_seq(N);
  std::vector<float, nsimd::allocator<float> > sin_nsimd(N);
  std::vector<float, nsimd::allocator<float> > cos_nsimd(N);

  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = pi / double(i + 1);
  }

  std::cout << "in = " << std::endl;
  print(in);
  std::cout << std::endl << std::endl;

  // Sequential

  for (size_t i = 0; i < in.size(); ++i) {
    sin_seq[i] = std::sin(in[i]);
    cos_seq[i] = std::cos(in[i]);
  }

  std::cout << "sin (sequential) = " << std::endl;
  print(sin_seq);
  std::cout << std::endl;
  std::cout << "cos (sequential) = " << std::endl;
  print(cos_seq);
  std::cout << std::endl << std::endl;

  // nsimd

  {
    size_t len = size_t(nsimd::len(nsimd::pack<float>()));
    for (size_t i = 0; i + len < in.size(); i += len) {
      auto v = nsimd::loada<nsimd::pack<float> >(&in[0]);
      auto v_sin = nsimd::sin(v);
      auto v_cos = nsimd::cos(v);
      nsimd::storea(&sin_nsimd[i], v_sin);
      nsimd::storea(&cos_nsimd[i], v_cos);
    }
  }

  std::cout << "sin (nsimd) = " << std::endl;
  print(sin_nsimd);
  std::cout << std::endl;
  std::cout << "cos (nsimd) = " << std::endl;
  print(cos_nsimd);
  std::cout << std::endl << std::endl;

  // nsimd: sincos

  {
    size_t len = size_t(nsimd::len(nsimd::pack<float>()));
    for (size_t i = 0; i + len < in.size(); i += len) {
      auto v = nsimd::loada<nsimd::pack<float> >(&in[0]);
      auto v_sincos = nsimd::sincos(v);
      nsimd::storea(&sin_nsimd[i], v_sincos.first);
      nsimd::storea(&cos_nsimd[i], v_sincos.second);
    }
  }

  std::cout << "sin (nsimd: sincos) = " << std::endl;
  print(sin_nsimd);
  std::cout << std::endl;
  std::cout << "cos (nsimd: sincos) = " << std::endl;
  print(cos_nsimd);
  std::cout << std::endl << std::endl;

  return 0;
}
