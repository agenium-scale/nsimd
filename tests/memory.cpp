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

#include <vector>
#include <cstdlib>

int test_aligned_alloc() {
  void *ptr = nsimd_aligned_alloc(17);
  if (ptr == NULL || ((size_t)ptr % NSIMD_MAX_ALIGNMENT) != 0) {
    return EXIT_FAILURE;
  }
  nsimd_aligned_free(ptr);
  return EXIT_SUCCESS;
}

template <typename T>
int test_aligned_alloc_for() {
  void *ptr = nsimd::aligned_alloc(17);
  if (ptr == NULL || ((size_t)ptr % NSIMD_MAX_ALIGNMENT) != 0) {
    return EXIT_FAILURE;
  }
  nsimd::aligned_free(ptr);
  return EXIT_SUCCESS;
}

template <typename T>
int test_allocator_for() {
  std::vector< T, nsimd::allocator<T> > v(17);
  if (v.size() != 17 || ((size_t)v.data() % NSIMD_MAX_ALIGNMENT) != 0) {
    return EXIT_FAILURE;
  }

  v.resize(17017);
  if (v.size() != 17017 || ((size_t)v.data() % NSIMD_MAX_ALIGNMENT) != 0) {
    return EXIT_FAILURE;
  }

  v.clear();
  if (v.size() != 0) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

int main() {
  return test_aligned_alloc() ||
         test_aligned_alloc_for<i8>() ||
         test_aligned_alloc_for<u8>() ||
         test_aligned_alloc_for<i16>() ||
         test_aligned_alloc_for<u16>() ||
         test_aligned_alloc_for<i32>() ||
         test_aligned_alloc_for<u32>() ||
         test_aligned_alloc_for<i64>() ||
         test_aligned_alloc_for<u64>() ||
         test_aligned_alloc_for<f32>() ||
         test_aligned_alloc_for<f64>() ||
         test_allocator_for<i8>() ||
         test_allocator_for<u8>() ||
         test_allocator_for<i16>() ||
         test_allocator_for<u16>() ||
         test_allocator_for<i32>() ||
         test_allocator_for<u32>() ||
         test_allocator_for<i64>() ||
         test_allocator_for<u64>() ||
         test_allocator_for<f32>() ||
         test_allocator_for<f64>();
}
