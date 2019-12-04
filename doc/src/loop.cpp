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

template <typename T, typename A> void print(std::vector<T, A> const &v) {
  std::cout << "{ ";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cout << (i == 0 ? "" : ", ") << std::setw(2) << v[i];
  }
  std::cout << " }";
}

int main() {

  size_t N = 23;

  std::vector<float, nsimd::allocator<float> > in(N);
  std::vector<float> out_seq(N);
  std::vector<float, nsimd::allocator<float> > out_nsimd(N);

  for (size_t i = 0; i < in.size(); ++i) {
    in[i] = 10 + i;
  }

  std::cout << "in = " << std::endl;
  print(in);
  std::cout << std::endl << std::endl;

  // Sequential

  for (size_t i = 0; i < in.size(); ++i) {
    out_seq[i] = in[i] * 2.0f;
  }

  std::cout << "out (sequential) = " << std::endl;
  print(out_seq);
  std::cout << std::endl << std::endl;

  // nsimd

  {
    size_t i;
    size_t len = size_t(nsimd::len(nsimd::pack<float>()));
    for (i = 0; i + len < in.size(); i += len) {
      auto v = nsimd::loada<nsimd::pack<float> >(&in[0]);
      v = v * 2.0f;
      nsimd::storea(&out_nsimd[i], v);
    }
    for (; i < in.size(); ++i) {
      out_nsimd[i] = in[i] * 2.0f;
    }
  }

  std::cout << "out (nsimd) = " << std::endl;
  print(out_nsimd);
  std::cout << std::endl << std::endl;

  return 0;
}
