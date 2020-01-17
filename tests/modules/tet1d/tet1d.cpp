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

#include <nsimd/modules/tet1d.hpp>

#include <nsimd/nsimd.h>

#include <vector>
#include <iostream>
#include <iomanip>

int main()
{
  size_t N = 16;

  std::vector<float/*, nsimd::allocator<float> */> a(N);
  std::vector<float/*, nsimd::allocator<float> */> b(N);
  std::vector<float/*, nsimd::allocator<float> */> c(N);
  std::vector<float/*, nsimd::allocator<float> */> d(N);
  std::vector<float/*, nsimd::allocator<float> */> e(N);


  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = i + 0;
    b[i] = i + 1;
    c[i] = i + 2;
    d[i] = 10;
    e[i] = 10;
  }

  std::vector<float> r;
  tet1d::out(r) =
    tet1d::fma( tet1d::in(d), -tet1d::in(e)
              , -(tet1d::in(a) + tet1d::in(b) + tet1d::in(c))
            ) * tet1d::in(2.f);

  for (size_t i = 0; i < r.size(); ++i) {
    std::cout << "-("
              << d[i] << " x -" << e[i] << " + ("
              << std::setw(2) << a[i] << " + "
              << std::setw(2) << b[i] << " + "
              << std::setw(2) << c[i] << ") * 2 = "
              << r[i] << std::endl;
  }

  return 0;
}
