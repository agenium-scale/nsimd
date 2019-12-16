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

/* make build/hello_world_c89_sse2 */
/* make build/hello_world_c99_sse2 */
/* make build/hello_world_c11_sse2 */

/* make build/hello_world_c89_avx2 */
/* make build/hello_world_c99_avx2 */
/* make build/hello_world_c11_avx2 */

#include <stdio.h>

#if defined(SSE2)
#include <xmmintrin.h> /* __m128 */
#endif

#if defined(AVX)
#include <immintrin.h> /* __m256 */
#endif

#include <nsimd/nsimd.h>

void print(f32 *p, size_t const N) {
  size_t i;
  printf("{ ");
  for (i = 0; i < N; ++i) {
    printf("%s%3.0f", i == 0 ? "" : ", ", p[i]);
  }
  printf(" }");
}

int main() {

  size_t N = 16;

  /* f32 is float */
  f32 *in0 = (f32 *)nsimd_aligned_alloc((nat)(N * sizeof(f32)));
  f32 *in1 = (f32 *)nsimd_aligned_alloc((nat)(N * sizeof(f32)));
  f32 *out = (f32 *)nsimd_aligned_alloc((nat)(N * sizeof(f32)));

  if (in0 == NULL || in1 == NULL || out == NULL) {
    fprintf(stderr, "ERROR: nsimd_aligned_alloc fails\n");
    return 1;
  }

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
  {
    size_t i;
    for (i = 0; i < N; ++i) {
      in0[i] = in0[i] + 10.0f + i;
      in1[i] = -10.0f - i;
    }
  }

  printf("Input 0:\n");
  print(in0, N);
  printf("\n");
  printf("Input 1:\n");
  print(in1, N);
  printf("\n\n");

  /* Sequential */

  {
    size_t i;
    for (i = 0; i < N; ++i) {
      out[i] = 0;
    }
  }

  {
    size_t i;
    for (i = 0; i < N; ++i) {
      out[i] = in0[i] + in1[i];
    }
  }

  printf("Sequential\n");
  print(out, N);
  printf("\n\n");

#if defined(SSE2)
  /* SSE */

  {
    size_t i;
    for (i = 0; i < N; ++i) {
      out[i] = 0;
    }
  }

  {
    size_t len_sse = 4;
    size_t i;
    for (i = 0; i < N; i += len_sse) {
      __m128 v0_sse = _mm_load_ps(&in0[i]);
      __m128 v1_sse = _mm_load_ps(&in1[i]);
      __m128 r_sse = _mm_add_ps(v0_sse, v1_sse);
      _mm_store_ps(&out[i], r_sse);
    }
  }

  printf("SSE2\n");
  print(out, N);
  printf("\n\n");
#endif

#if defined(AVX)
  /* AVX */

  {
    size_t i;
    for (i = 0; i < N; ++i) {
      out[i] = 0;
    }
  }

  {
    size_t len_avx = 8;
    size_t i;
    for (i = 0; i < N; i += len_avx) {
      __m256 v0_avx = _mm256_load_ps(&in0[i]);
      __m256 v1_avx = _mm256_load_ps(&in1[i]);
      __m256 r_avx = _mm256_add_ps(v0_avx, v1_avx);
      _mm256_store_ps(&out[i], r_avx);
    }
  }

  printf("AVX\n");
  print(out, N);
  printf("\n\n");
#endif

  /* nsimd */

  {
    size_t i;
    for (i = 0; i < N; ++i) {
      out[i] = 0;
    }
  }

  {
    size_t len = (size_t)vlen(f32);
    size_t i;
    for (i = 0; i < N; i += len) {
      vec(f32) v0 = vloada(&in0[i], f32);
      vec(f32) v1 = vloada(&in1[i], f32);
      vec(f32) r = vadd(v0, v1, f32);
      vstorea(&out[i], r, f32);
    }
  }

  printf("nsimd C (C89, C99, C11)\n");
  print(out, N);
  printf("\n\n");

  /* Print string */
  {
    size_t i;
    for (i = 0; i < N; ++i) {
      printf("%c", (char)out[i]);
    }
    printf("\n\n");
  }

  /* Save the world */
  nsimd_aligned_free(in0);
  in0 = NULL;
  nsimd_aligned_free(in1);
  in1 = NULL;
  nsimd_aligned_free(out);
  out = NULL;

  return 0;
}
