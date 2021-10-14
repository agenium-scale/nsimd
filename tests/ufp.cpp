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

#include <nsimd/nsimd.h>
#include <cstdlib>

// ----------------------------------------------------------------------------

template <typename U> U randbits() {
  U ret = 0;
  U mask = ((U)1 << CHAR_BIT) - 1;
  for (int i = 0; i < (int)sizeof(U); i++) {
    ret = (U)(ret | (U)((((U)rand()) & mask) << (CHAR_BIT * i)));
  }
  return ret;
}

// ----------------------------------------------------------------------------

template <typename U> int log_std_ulp(U a, U b) {
  U d = (U)(a < b ? b - a : a - b);
  int i = 0;
  for (; i < 63 && d >= (U)1 << i; i++)
    ;
  return i;
}

// ----------------------------------------------------------------------------

template <typename T> struct mantissa{};
template <> struct mantissa<f64> { static const int size = 53; };
template <> struct mantissa<f32> { static const int size = 24; };
template <> struct mantissa<f16> { static const int size = 11; };

// ----------------------------------------------------------------------------

template <typename T, typename U>
int test_ufp(int n) {
  T a = nsimd::scalar_cvt(T(), (U)1);
  U ua = nsimd::scalar_reinterpret(U(), a);
  T ap1 = nsimd::scalar_reinterpret(T(), (U)(ua + 1));
  if (nsimd::ufp(a, ap1) != mantissa<T>::size - 1) {
    return -1;
  }

  T am1 = nsimd::scalar_reinterpret(T(), (U)(ua - 1));
  if (nsimd::ufp(a, am1) != mantissa<T>::size - 1) {
    return -1;
  }

  if (nsimd::ufp(a, a) != mantissa<T>::size) {
    return -1;
  }
  if (nsimd::ufp(a, a) != mantissa<T>::size) {
    return -1;
  }
  if (nsimd::ufp(a, a) != mantissa<T>::size) {
    return -1;
  }

  T ax4 = nsimd::scalar_cvt(T(), (U)4);
  if (nsimd::ufp(a, ax4) != 0) {
    return -1;
  }

  U mask = (U)1 << (mantissa<T>::size - 1);
  U exponent = (U)((~mask) & ua);
  for (int i = 0; i < n; i++) {
    U ub = exponent | (randbits<U>() & mask);
    T b = nsimd::scalar_reinterpret(T(), ub);
    U uc = exponent | (randbits<U>() & mask);
    T c = nsimd::scalar_reinterpret(T(), uc);
    if (nsimd::ufp(b, c) != mantissa<T>::size - log_std_ulp(ub, uc)) {
      return -1;
    }
  }

  return 0;
}

// ----------------------------------------------------------------------------

int main(void) {
  int n = 10000;
  return test_ufp<f64, u64>(n) || test_ufp<f32, u32>(n) ||
         test_ufp<f16, u16>(n);
}
