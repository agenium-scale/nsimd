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
#include <nsimd/modules/memory-management.hpp>
#include <nsimd/modules/spmd.hpp>

#define N 10000

// ----------------------------------------------------------------------------

nsimd_fill_dev_mem_func(prng20, (((unsigned int)i * 34872948 + 916794) % 20))

nsimd_fill_dev_mem_func(prng5, (((unsigned int)i * 69428380 + 784295) % 5))

// ----------------------------------------------------------------------------

struct mul_t {
  static float impl(float a, float b) { return a * b; }

  spmd_dev_func(static k_float dev_impl, k_float a, k_float b)
    return a * b;
  spmd_dev_func_end
};

struct add_t {
  static float impl(float a, float b) { return a + b; }

  spmd_dev_func(static k_float dev_impl, k_float a, k_float b)
    return a + b;
  spmd_dev_func_end
};

spmd_tmpl_dev_func(k_float trampoline, Op, k_float a, k_float b)
  return Op::template spmd_call_dev_func(dev_impl, a, b);
spmd_dev_func_end

spmd_tmpl_kernel_1d(tmpl_kernel, Op, float *dst, float *a, float *b)
  k_store(dst, spmd_call_tmpl_dev_func(trampoline, Op, k_load(a), k_load(b)));
spmd_kernel_end

template <typename Op>
int test_tmpl_func() {
  nsimd::paired_pointers_t<float> dst =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> a =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> b =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  prng20(a.device_ptr, N);
  prng20(b.device_ptr, N);

  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");

  spmd_launch_tmpl_kernel_1d(tmpl_kernel, Op, 32, 1, N, dst.device_ptr,
                             a.device_ptr, b.device_ptr);

  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");

  nsimd::copy_to_host(a);
  nsimd::copy_to_host(b);
  nsimd::copy_to_host(dst);
  for (int i = 0; i < N; i++) {
    if (dst.host_ptr[i] != Op::impl(a.host_ptr[i], b.host_ptr[i])) {
      return -1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------------

spmd_dev_func(k_float device_add, k_float a, k_float b)
  return a + b;
spmd_dev_func_end

spmd_kernel_1d(add_with_func, float *dst, float *a, float *b)
  k_store(dst, spmd_call_dev_func(device_add, k_load(a), k_load(b)));
spmd_kernel_end

int test_add_with_func() {
  nsimd::paired_pointers_t<float> dst =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> a =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> b =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  prng20(a.device_ptr, N);
  prng20(b.device_ptr, N);
  spmd_launch_kernel_1d(add_with_func, 32, 1, N, dst.device_ptr,
                        a.device_ptr, b.device_ptr);
  nsimd::copy_to_host(a);
  nsimd::copy_to_host(b);
  nsimd::copy_to_host(dst);
  for (int i = 0; i < N; i++) {
    if (dst.host_ptr[i] != a.host_ptr[i] + b.host_ptr[i]) {
      return -1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------------

spmd_kernel_1d(unmasked_stuff, float *dst, float *a_ptr)

  k_float a, b;
  k_unmasked_set(a, k_unmasked_load(a_ptr));
  k_unmasked_set(b, a + 1.0f);
  k_unmasked_store(dst, b);

spmd_kernel_end

int test_unmasked_stuff() {
  nsimd::paired_pointers_t<float> dst =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> a =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  prng20(a.device_ptr, N);
  spmd_launch_kernel_1d(unmasked_stuff, 32, 1, N, dst.device_ptr,
                        a.device_ptr);
  nsimd::copy_to_host(a);
  nsimd::copy_to_host(dst);
  for (int i = 0; i < N; i++) {
    //fprintf(stderr, "[%d] %f vs %f\n", i, (double)dst[i], (double)res);
    if (dst.host_ptr[i] != a.host_ptr[i] + 1.0f) {
      return -1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------------

spmd_kernel_1d(if_elseif_else, float *dst, float *a_ptr)

  k_float a, ret;
  k_set(a, k_load(a_ptr));

  k_if (a > 15.0f)

    k_set(ret, 15.0f);

  k_elseif ( a > 10.0f)

    k_set(ret, 10.0f);

  k_elseif ( a > 5.0f)

    k_set(ret, 5.0f);

  k_else

    k_set(ret, 0.0f);

  k_endif

  k_store(dst, ret);

spmd_kernel_end

int test_if_elseif_else() {
  nsimd::paired_pointers_t<float> dst =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> a =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  prng20(a.device_ptr, N);
  spmd_launch_kernel_1d(if_elseif_else, 32, 1, N, dst.device_ptr,
                        a.device_ptr);
  for (int i = 0; i < N; i++) {
    float res = 0.0f;
    if (a.host_ptr[i] > 15.0f) {
      res = 15.0f;
    } else if (a.host_ptr[i] > 10.0f) {
      res = 10.0f;
    } else if (a.host_ptr[i] > 5.0f) {
      res = 5.0f;
    } else {
      res = 0.0f;
    }
    //fprintf(stderr, "[%d] %f vs %f\n", i, (double)dst[i], (double)res);
    if (dst.host_ptr[i] != res) {
      return -1;
    }
  }
  return 0;
}


// ----------------------------------------------------------------------------

spmd_kernel_1d(binpow, float *dst, float *a_ptr, int *p_ptr)

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
  nsimd::paired_pointers_t<float> dst =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> a =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<int> p =
      nsimd::pair_malloc_or_exit<int>(N * sizeof(float));
  prng20(a.device_ptr, N);
  prng5(p.device_ptr, N);

  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");

  spmd_launch_kernel_1d(binpow, 32, 1, N, dst.device_ptr, a.device_ptr,
                        p.device_ptr);

  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");
  __asm__ __volatile__ ("nop");

  nsimd::copy_to_host(a);
  nsimd::copy_to_host(p);
  nsimd::copy_to_host(dst);
  for (int i = 0; i < N; i++) {
    float res = 1.0f;
    for (int j = 0; j < p.host_ptr[i]; j++) {
      res *= a.host_ptr[i];
    }
    //fprintf(stderr, "[%d] %f vs %f\n", i, (double)dst[i], (double)res);
    if (dst.host_ptr[i] != res) {
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

spmd_kernel_1d(nested_ifs, float *dst, float *a)
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
  nsimd::paired_pointers_t<float> dst =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> a =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  prng20(a.device_ptr, N);
  spmd_launch_kernel_1d(nested_ifs, 32, 1, N, dst.device_ptr, a.device_ptr);
  nsimd::copy_to_host(a);
  nsimd::copy_to_host(dst);
  for (int i = 0; i < N; i++) {
    //fprintf(stderr, "[%d] %f vs %f\n", i, (double)a[i], (double)dst[i]);
    if (a.host_ptr[i] <= 5.0f && dst.host_ptr[i] != 0.0f) {
      return -1;
    }
    if (a.host_ptr[i] == 6.0f && dst.host_ptr[i] != 1.0f) {
      return -1;
    }
    if (a.host_ptr[i] > 6.0f && dst.host_ptr[i] != a.host_ptr[i] - 4.0f) {
      return -1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------------

spmd_kernel_1d(threshold, float *dst, float *a)
  k_if(k_load(a) > 10.0f)
    k_store(dst, 10.0f);
  k_endif
spmd_kernel_end

int test_threshold() {
  nsimd::paired_pointers_t<float> dst =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> a =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  prng20(a.device_ptr, N);
  spmd_launch_kernel_1d(threshold, 32, 1, N, dst.device_ptr, a.device_ptr);
  nsimd::copy_to_host(dst);
  for (int i = 0; i < N; i++) {
    if (dst.host_ptr[i] > 10.0f) {
      return -1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------------

spmd_kernel_1d(add, float *dst, float *a, float *b)
  k_store(dst, k_load(a) + k_load(b));
spmd_kernel_end

int test_add() {
  nsimd::paired_pointers_t<float> dst =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> a =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  nsimd::paired_pointers_t<float> b =
      nsimd::pair_malloc_or_exit<float>(N * sizeof(float));
  prng20(a.device_ptr, N);
  prng20(b.device_ptr, N);
  spmd_launch_kernel_1d(add, 32, 1, N, dst.device_ptr, a.device_ptr,
                        b.device_ptr);
  nsimd::copy_to_host(a);
  nsimd::copy_to_host(b);
  nsimd::copy_to_host(dst);
  for (int i = 0; i < N; i++) {
    if (dst.host_ptr[i] != a.host_ptr[i] + b.host_ptr[i]) {
      return -1;
    }
  }
  return 0;
}

// ----------------------------------------------------------------------------

int main() {
  return test_add() || test_threshold() || test_nested_ifs() ||
         test_binpow() || test_if_elseif_else() || test_unmasked_stuff() ||
         test_add_with_func() || test_tmpl_func<add_t>() ||
         test_tmpl_func<mul_t>();
}
