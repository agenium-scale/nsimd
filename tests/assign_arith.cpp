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

#include <nsimd/nsimd-all.hpp>
#include <iostream>

/* ------------------------------------------------------------------------- */
/* Random number */

template <typename T> T get_rand() {
  return (T)((rand() % 10) + 1);
}

template <> f16 get_rand() {
  return nsimd_f32_to_f16(get_rand<f32>());
}

/* ------------------------------------------------------------------------- */
/* Arithmetic operators */

#define HELPER(op1, op2, name)                                                \
  template <typename T> int test_##name##_T(size_t n) {                       \
    std::vector<T> a(n), b(n);                                                \
    for (size_t i = 0; i < n; i++) {                                          \
      a[i] = get_rand<T>();                                                   \
      b[i] = get_rand<T>();                                                   \
    }                                                                         \
                                                                              \
    using namespace nsimd;                                                    \
    typedef pack<T> pack;                                                     \
    for (size_t i = 0; i < n; i += size_t(len(pack<T>()))) {                  \
      pack tmp1 = loadu<pack>(&a[i]);                                         \
      tmp1 op1 loadu<pack>(&b[i]);                                            \
      pack tmp2 = loadu<pack>(&a[i]) op2 loadu<pack>(&b[i]);                  \
      if (any(tmp1 != tmp2)) {                                                \
        return -1;                                                            \
      }                                                                       \
    }                                                                         \
    return 0;                                                                 \
  }                                                                           \
                                                                              \
  int test_##name(size_t n) {                                                 \
    return test_##name##_T<i8>(n) || test_##name##_T<u8>(n) ||                \
           test_##name##_T<i16>(n) || test_##name##_T<u16>(n) ||              \
           test_##name##_T<f16>(n) || test_##name##_T<i32>(n) ||              \
           test_##name##_T<u32>(n) || test_##name##_T<f32>(n) ||              \
           test_##name##_T<i64>(n) || test_##name##_T<u64>(n) ||              \
           test_##name##_T<f64>(n);                                           \
  }                                                                           \
                                                                              \
  int test_##name##_int_only(size_t n) {                                      \
    return test_##name##_T<i8>(n) || test_##name##_T<u8>(n) ||                \
           test_##name##_T<i16>(n) || test_##name##_T<u16>(n) ||              \
           test_##name##_T<i32>(n) || test_##name##_T<u32>(n) ||              \
           test_##name##_T<i64>(n) || test_##name##_T<u64>(n);                \
  }

HELPER(+=, +, add)
HELPER(-=, -, sub)
HELPER(*=, *, mul)
HELPER(/=, /, div)
HELPER(|=, |, orb)
HELPER(&=, &, andb)
HELPER(^=, ^, xorb)

#undef HELPER

/* ------------------------------------------------------------------------- */
/* Shift operators */

#define HELPER(op1, op2, name)                                                \
  template <typename T> int test_##name##_T(size_t n) {                       \
    std::vector<T> a(n);                                                      \
    for (size_t i = 0; i < n; i++) {                                          \
      a[i] = get_rand<T>();                                                   \
    }                                                                         \
                                                                              \
    using namespace nsimd;                                                    \
    typedef pack<T> pack;                                                     \
    for (int s = 0; s <= 3; s++) {                                            \
      for (size_t i = 0; i < n; i += size_t(len(pack<T>()))) {                \
        pack tmp = loadu<pack>(&a[i]);                                        \
        tmp op1 s;                                                            \
        if (any(tmp != (loadu<pack>(&a[i]) op2 s))) {                         \
          return -1;                                                          \
        }                                                                     \
      }                                                                       \
    }                                                                         \
    return 0;                                                                 \
  }                                                                           \
                                                                              \
  int test_##name(size_t n) {                                                 \
    return test_##name##_T<i8>(n) || test_##name##_T<u8>(n) ||                \
           test_##name##_T<i16>(n) || test_##name##_T<u16>(n) ||              \
           test_##name##_T<i32>(n) || test_##name##_T<u32>(n) ||              \
           test_##name##_T<i64>(n) || test_##name##_T<u64>(n);                \
  }

HELPER(<<=, <<, shl)
HELPER(>>=, >>, shr)

#undef HELPER

/* ------------------------------------------------------------------------- */

int main() {
  const size_t n = 2048;
  return test_add(n) || test_sub(n) || test_mul(n) || test_div(n) ||
         test_orb_int_only(n) || test_andb_int_only(n) ||
         test_xorb_int_only(n) || test_shl(n) || test_shr(n);
}

