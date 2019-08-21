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

#include <iostream>
#include <nsimd/modules/fixed_point.hpp>

namespace fp = nsimd::fixed_point;

int main() {
  using fp_t = nsimd::fixed_point::fp_t<8, 8>;
  using vec_t = nsimd::fixed_point::pack<8, 8>;
  using raw_t = nsimd::fixed_point::pack<8, 8>::value_type;

  const size_t v_size = nsimd::fixed_point::len(fp_t());

  fp_t tab0[v_size];
  fp_t tab1[v_size];
  fp_t res[v_size];

  for (size_t i = 0; i < v_size; i++) {
    tab0[i] = (fp_t)i;
    tab1[i] = (fp_t)i;
  }

  vec_t v0 = nsimd::fixed_point::loadu<vec_t>(tab0);
  vec_t v1 = nsimd::fixed_point::loadu<vec_t>(tab1);
  vec_t sum = nsimd::fixed_point::add(v0, v1);
  nsimd::fixed_point::storeu(res, sum);

  std::cout << "Output vector : [";
  for (size_t i = 0; i < v_size; i++) {
    std::cout << " " << res[i];
  }
  std::cout << "]" << std::endl;
  return 0;
}
