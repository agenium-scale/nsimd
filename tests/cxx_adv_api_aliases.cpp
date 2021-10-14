/*

Copyright (c) 2021 Agenium Scale

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

#include <nsimd/cxx_adv_api_aliases.hpp>

/* ------------------------------------------------------------------------- */
/* Random number */

template <typename T> T get_rand() {
  return (T)((rand() % 100) - 50);
}

template <> f16 get_rand() {
  return nsimd_f32_to_f16(get_rand<f32>());
}

/* ------------------------------------------------------------------------- */

template <typename T> int test_aliases(size_t n) {
  std::vector<T> a(n), b(n);

  for (size_t i = 0; i < n; i++) {
    a[i] = get_rand<T>();
    b[i] = get_rand<T>();
  }

  using namespace nsimd;
  typedef pack<T> pack;
  size_t step = size_t(len(pack()));
  for (size_t i = 0; i + step <= n; i += step) {
    pack tmp1 = loadu<pack>(&a[i]);
    pack tmp2 = loadu<pack>(&b[i]);
    if (any(fabs(tmp1) != abs(tmp1))) {
      return -1;
    }
    if (any(fmin(tmp1, tmp2) != min(tmp1, tmp2))) {
      return -1;
    }
    if (any(fmax(tmp1, tmp2) != max(tmp1, tmp2))) {
      return -1;
    }
  }

  return 0;
}

/* ------------------------------------------------------------------------- */

int main() {
  return test_aliases<i8>(2048) || test_aliases<u8>(2048) ||
         test_aliases<i16>(2048) || test_aliases<u16>(2048) ||
         test_aliases<f16>(2048) || test_aliases<i32>(2048) ||
         test_aliases<u32>(2048) || test_aliases<f32>(2048) ||
         test_aliases<i64>(2048) || test_aliases<u64>(2048) ||
         test_aliases<f64>(2048);
}
