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

#include <cstdlib>
#include <vector>

int main() {

  std::vector<float, nsimd::cuda_allocator<float> > v;

  v.clear();
  v.resize(100);

  v.clear();
  v.resize(100);
  v.resize(10000);

  v.clear();
  v.reserve(30);

  for (int i = 0; i < 1000; i++) {
    v.push_back(float(i));
  }
  if (v.size() != 1000) {
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < 500; i++) {
    v.pop_back();
  }
  if (v.size() != 500) {
    exit(EXIT_FAILURE);
  }

  return 0;
}
