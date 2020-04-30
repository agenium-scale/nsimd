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

#include <iostream>
#include <nsimd/modules/spmd.hpp>

#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) || \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)

// For now, we do not test GPUs
// TODO: Fix it later

int main() { return 0; }

#else

#define N 10000

// ----------------------------------------------------------------------------

spmd_kernel(binpow, float *dst, float *a_ptr, int *p_ptr)

  k_float a, ret;
  k_set(a, k_load(a_ptr));
  k_set(ret, 1.0f);
  k_int p;
  k_set(p, k_load(p_ptr));

  k_while(p > 0)

    k_if ((p & 1) != 0)

      k_set(ret, ret * a);

    k_endif

    k_set(a, a * a);
    k_set(p, p >> 1);

  k_endwhile

  k_store(dst, ret);

spmd_kernel_end

int test_binpow() {
  float dst[N], a[N];
  int p[N];
  for (int i = 0; i < N; i++) {
    a[i] = (float)(((unsigned int)i * 34872948 + 916794) % 20);
    p[i] = (int)(((unsigned int)i * 69428380 + 784295) % 5);
    dst[i] = -1.0f;
  }
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  spmd_launch_kernel_1d(binpow, 32, 1, N, dst, a, p);
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  for (int i = 0; i < N; i++) {
    //fprintf(stderr, "[%d] %f vs %f\n", i, (double)a[i], (double)dst[i]);
    if (a[i] <= 5.0f && dst[i] != 0.0f) {
      return -1;
    }
    if (a[i] == 6.0f && dst[i] != 1.0f) {
      return -1;
    }
    if (a[i] > 6.0f && dst[i] != a[i] - 4.0f) {
      return -1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------------
// mapping with if's
// 0 1 2 3 4 5 --> 0
// 6 --> 1
// 7 8 9 .. 20 --> x - 4

spmd_kernel(nested_ifs, float *dst, float *a)
  k_float tmp;
  k_set(tmp, k_load(a));

  k_if (tmp > 5.0f)

    k_if (tmp == 6.0f)

      k_store(dst, 1.0f);
      k_return;

    k_endif

    k_store(dst, tmp - 4.0f);
    k_return;

  k_endif

  k_store(dst, 0.0f);

spmd_kernel_end

int test_nested_ifs() {
  float dst[N], a[N];
  for (int i = 0; i < N; i++) {
    a[i] = (float)(((unsigned int)i * 34872948 + 916794) % 20);
    dst[i] = 20.0f;
  }
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  spmd_launch_kernel_1d(nested_ifs, 32, 1, N, dst, a);
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  for (int i = 0; i < N; i++) {
    //fprintf(stderr, "[%d] %f vs %f\n", i, (double)a[i], (double)dst[i]);
    if (a[i] <= 5.0f && dst[i] != 0.0f) {
      return -1;
    }
    if (a[i] == 6.0f && dst[i] != 1.0f) {
      return -1;
    }
    if (a[i] > 6.0f && dst[i] != a[i] - 4.0f) {
      return -1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------------

spmd_kernel(threshold, float *dst, float *a)
  k_if(k_load(a) > 10.0f)
    k_store(dst, 10.0f);
  k_endif
spmd_kernel_end

int test_threshold() {
  float dst[N], a[N];
  for (int i = 0; i < N; i++) {
    a[i] = (float)(((unsigned int)i * 34872948 + 916794) % 20);
    dst[i] = 0.0f;
  }
  spmd_launch_kernel_1d(threshold, 32, 1, N, dst, a);
  for (int i = 0; i < N; i++) {
    if (dst[i] > 10.0f) {
      return -1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------------

spmd_kernel(add, float *dst, float *a, float *b)
  k_store(dst, k_load(a) + k_load(b));
spmd_kernel_end

int test_add() {
  float dst[N], a[N], b[N];
  for (int i = 0; i < N; i++) {
    a[i] = (float)i;
    b[i] = 2.0f * (float)i + 1.0f;
  }
  spmd_launch_kernel_1d(add, 32, 1, N, dst, a, b);
  for (int i = 0; i < N; i++) {
    if (dst[i] != 3.0f * (float)i + 1.0f) {
      return -1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------------

int main() {
  return test_add() || test_threshold() || test_nested_ifs() || test_binpow();
}

// ----------------------------------------------------------------------------

#endif
