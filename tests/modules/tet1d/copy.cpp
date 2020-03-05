/*

Copyright (c) 2020 Agenium Scale

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
#include "common.hpp"

// ----------------------------------------------------------------------------

template <typename T>
bool test_copy(unsigned int n) {
  T *src = get123<T>(n);
  if (src == NULL) {
    return false;
  }
  T *dst = get000<T>(n);
  if (dst == NULL) {
    del(src);
    return false;
  }
  tet1d::out(dst) = tet1d::in(src, n);
  bool ret = cmp(src, dst, n);
  del(dst);
  del(src);
  return ret;
}

int main() {
  for (unsigned int i = 100; i < 10000000; i *= 2) {
    if (test_copy<signed char>(i) == false ||
        test_copy<unsigned char>(i) == false || test_copy<short>(i) == false ||
        test_copy<unsigned short>(i) == false || test_copy<int>(i) == false ||
        test_copy<unsigned int>(i) == false || test_copy<long>(i) == false ||
        test_copy<unsigned long>(i) == false || test_copy<float>(i) == false ||
        test_copy<double>(i) == false) {
      return -1;
    }
  }
  return 0;
}
