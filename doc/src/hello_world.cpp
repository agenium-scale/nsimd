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

// make build/hello_world_cpp98_sse2
// make build/hello_world_cpp11_sse2

// make build/hello_world_cpp98_avx2
// make build/hello_world_cpp11_avx2

#include <iomanip>
#include <iostream>
#include <vector>

#if defined(SSE2)
#include <xmmintrin.h> // __m128
#endif

#if defined(AVX)
#include <immintrin.h> // __m256
#endif

#include <nsimd/nsimd-all.hpp>

template <typename A> void print(std::vector<f32, A> const &v) {
  std::cout << "{ ";
  for (size_t i = 0; i < v.size(); ++i) {
    std::cout << (i == 0 ? "" : ", ") << std::setw(3) << v[i];
  }
  std::cout << " }";
}

int main() {

  size_t N = 16;

  // f32 is float
  std::vector<f32, nsimd::allocator<f32> > in0(N);
  std::vector<f32, nsimd::allocator<f32> > in1(N);
  std::vector<f32, nsimd::allocator<f32> > out(N);

  in0[0] = 72;
  in0[1] = 101;
  in0[2] = 108;
  in0[3] = 108;
  in0[4] = 111;
  in0[5] = 32;
  in0[6] = 119;
  in0[7] = 111;
  in0[8] = 114;
  in0[9] = 108;
  in0[10] = 100;
  in0[11] = 33;
  in0[12] = 33;
  in0[13] = 33;
  in0[14] = 33;
  in0[15] = 32;
  for (size_t i = 0; i < N; ++i) {
    in0[i] = in0[i] + 10.0f + i;
    in1[i] = -10.0f - i;
  }

  std::cout << "Input 0:" << std::endl;
  print(in0);
  std::cout << std::endl;
  std::cout << "Input 1:" << std::endl;
  print(in1);
  std::cout << std::endl << std::endl;

  // Sequential

  for (size_t i = 0; i < N; ++i) {
    out[i] = 0;
  }

  for (size_t i = 0; i < N; ++i) {
    out[i] = in0[i] + in1[i];
  }

  std::cout << "Sequential" << std::endl;
  print(out);
  std::cout << std::endl << std::endl;

#if defined(SSE2)
  // SSE

  for (size_t i = 0; i < N; ++i) {
    out[i] = 0;
  }

  {
    size_t len_sse = 4;
    for (size_t i = 0; i < N; i += len_sse) {
      __m128 v0_sse = _mm_load_ps(&in0[i]);
      __m128 v1_sse = _mm_load_ps(&in1[i]);
      __m128 r_sse = _mm_add_ps(v0_sse, v1_sse);
      _mm_store_ps(&out[i], r_sse);
    }
  }

  std::cout << "SSE2" << std::endl;
  print(out);
  std::cout << std::endl << std::endl;
#endif

#if defined(AVX)
  // AVX

  for (size_t i = 0; i < N; ++i) {
    out[i] = 0;
  }

  {
    std::size_t len_avx = 8;
    for (size_t i = 0; i < N; i += len_avx) {
      __m256 v0_avx = _mm256_load_ps(&in0[i]);
      __m256 v1_avx = _mm256_load_ps(&in1[i]);
      __m256 r_avx = _mm256_add_ps(v0_avx, v1_avx);
      _mm256_store_ps(&out[i], r_avx);
    }
  }

  std::cout << "AVX" << std::endl;
  print(out);
  std::cout << std::endl << std::endl;
#endif

#if __cplusplus >= 201103L
  // nsimd C++11

  for (size_t i = 0; i < N; ++i) {
    out[i] = 0;
  }

  {
    size_t len = size_t(nsimd::len(f32()));
    for (size_t i = 0; i < N; i += len) {
      // auto is nsimd::simd_vector<f32>
      auto v0 = nsimd::loada(&in0[i], f32());
      auto v1 = nsimd::loada(&in1[i], f32());
      auto r = nsimd::add(v0, v1, f32());
      nsimd::storea(&out[i], r, f32());
    }
  }

  std::cout << "nsimd C++11" << std::endl;
  print(out);
  std::cout << std::endl << std::endl;

  static_assert(std::is_same<decltype(nsimd::loada(&in0[0], f32())),
                             nsimd::simd_vector<f32> >::value,
                "auto is not nsimd::simd_vector<f32>");
#endif

#if __cplusplus >= 201103L
  // nsimd C++11 advanced API

  for (size_t i = 0; i < N; ++i) {
    out[i] = 0;
  }

  {
    size_t len = size_t(nsimd::len(f32()));
    for (size_t i = 0; i < N; i += len) {
      // auto is nsimd::pack<f32>
      auto v0 = nsimd::loada<nsimd::pack<f32> >(&in0[i]);
      auto v1 = nsimd::loada<nsimd::pack<f32> >(&in1[i]);
      auto r = v0 + v1;
      nsimd::storea(&out[i], r);
    }
  }

  std::cout << "nsimd C++11 (Advanced API)" << std::endl;
  print(out);
  std::cout << std::endl << std::endl;

  static_assert(
      std::is_same<decltype(nsimd::loada(nsimd::pack<f32>(), &in0[0])),
                   nsimd::pack<f32> >::value,
      "auto is not nsimd::pack<f32>");
#endif

  // nsimd C++98

  for (size_t i = 0; i < N; ++i) {
    out[i] = 0;
  }

  {
    size_t len = size_t(nsimd::len(f32()));
    typedef nsimd::simd_traits<f32, nsimd::NSIMD_SIMD>::simd_vector vec_t;
    for (size_t i = 0; i < N; i += len) {
      vec_t v0 = nsimd::loada(&in0[i], f32());
      vec_t v1 = nsimd::loada(&in1[i], f32());
      vec_t r = nsimd::add(v0, v1, f32());
      nsimd::storea(&out[i], r, f32());
    }
  }

  std::cout << "nsimd C++98" << std::endl;
  print(out);
  std::cout << std::endl << std::endl;

  // nsimd C++98 advanced API

  for (size_t i = 0; i < N; ++i) {
    out[i] = 0;
  }

  {
    size_t len = size_t(nsimd::len(f32()));
    for (size_t i = 0; i < N; i += len) {
      nsimd::pack<f32> v0 = nsimd::loada<nsimd::pack<f32> >(&in0[i]);
      nsimd::pack<f32> v1 = nsimd::loada<nsimd::pack<f32> >(&in1[i]);
      nsimd::pack<f32> r = v0 + v1;
      nsimd::storea(&out[i], r);
    }
  }

  std::cout << "nsimd C++98 (Advanced API)" << std::endl;
  print(out);
  std::cout << std::endl << std::endl;

  // Print string
  for (size_t i = 0; i < out.size(); ++i) {
    std::cout << char(out[i]);
  }
  std::cout << std::endl << std::endl;

  return 0;
}
