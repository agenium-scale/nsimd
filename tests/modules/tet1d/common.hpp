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

#ifndef NSIMD_MODULES_TET1D_COMMON_HPP
#define NSIMD_MODULES_TET1D_COMMON_HPP

#include <nsimd/nsimd.h>
#include <nsimd/scalar_utilities.h>
#include <iostream>
#include <cstring>
#include <cstdlib>

// ----------------------------------------------------------------------------
// CUDA

#if defined(NSIMD_CUDA)

template <typename T>
__global__ void device_fill(T *dst, int n, int variant) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    switch(variant) {
      case 123:
        dst[i] = T(i % 10);
        break;
      case 321:
        dst[i] = T((2147483647 * i) % 10);
        break;
      case 101:
        dst[i] = T(i % 2);
        break;
      case 110:
        dst[i] = T(((2147483647 * i) % 13) % 2);
        break;
    }
  }
}

// perform reduction on blocks first, note that this could be optimized
// but to check correctness we don't need it now
template <typename T>
__global__ void device_cmp_blocks(T *src1, T *src2, int n) {
  extern __shared__ char buf_[]; // size of a block
  T *buf = (T*)buf_;
  int tid = threadIdx.x;
  int i = tid + blockIdx.x * blockDim.x;
  if (i < n) {
    printf("DEBUG: %d: %f vs. %f\n", i, (double)src1[i], (double)src2[i]);
    buf[tid] = T(nsimd::gpu_eq(src1[i], src2[i]) ? 1 : 0);
  }
  __syncthreads();
  for (int s = blockDim.x / 2; s != 0; s /= 2) {
    if (tid < s && i < n) {
      buf[tid] = nsimd::gpu_mul(buf[tid], buf[tid + s]);
      __syncthreads();
    }
  }
  if (tid == 0) {
    src1[i] = buf[0];
  }
}

template <typename T>
__global__ void device_cmp_array(int *dst, T *src1, int n) {
  // reduction on the whole vector
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  T buf = T(1);
  for (int i = 0; i < n; i += blockDim.x) {
    buf = nsimd::gpu_mul(buf, src1[i]);
  }
  if (i == 0) {
    dst[0] = int(buf);
  }
}

int cuda_do(cudaError_t code) {
  if (code != cudaSuccess) {
    std::cerr << "ERROR: " << cudaGetErrorString(code) << std::endl;
    return -1;
  }
  return 0;
}

template <typename T> T *get000(unsigned int n) {
  T *ret;
  if (cuda_do(cudaMalloc((void **)&ret, n * sizeof(T)))) {
    return NULL;
  }
  if (cuda_do(cudaMemset((void *)ret, 0, n * sizeof(T)))) {
    cudaFree((void *)ret);
    return NULL;
  }
  return ret;
}

template <typename T> T *getXXX(unsigned int n, int variant) {
  T *ret;
  if (cuda_do(cudaMalloc((void **)&ret, n * sizeof(T)))) {
    return NULL;
  }
  device_fill<<<(n + 127) / 128, 128>>>(ret, int(n), variant);
  return ret;
}

template <typename T> bool cmp(T *src1, T *src2, unsigned int n) {
  int host_ret;
  int *device_ret;
  if (cuda_do(cudaMalloc((void **)&device_ret, sizeof(int)))) {
    return false;
  }
  device_cmp_blocks<<<(n + 127) / 128, 128, 128 * sizeof(T)>>>(src1, src2,
                                                               int(n));
  device_cmp_array<<<(n + 127) / 128, 128>>>(device_ret, src1, int(n));
  if (cuda_do(cudaMemcpy((void *)&host_ret, (void *)device_ret, sizeof(int),
                 cudaMemcpyDeviceToHost))) {
    host_ret = 0;
  }
  cudaFree((void *)device_ret);
  return bool(host_ret);
}

template <typename T> void del(T *ptr) { cudaFree(ptr); }

#elif defined(NSIMD_HIP)

// ----------------------------------------------------------------------------
// HIP

// TODO

#else

// ----------------------------------------------------------------------------
// SIMD

template <typename T> T *get000(unsigned int n) {
  return (T *)calloc(n, sizeof(T));
}

template <typename T> struct helper_cast {
  static T cast(unsigned int a) { return T(a); }
};

template <> struct helper_cast<f16> {
  static f16 cast(unsigned int a) { return nsimd_f32_to_f16(f32(a)); }
};

template <typename T> T *getXXX(unsigned int n, int variant) {
  T *ret = (T *)malloc(n * sizeof(T));
  if (ret == NULL) {
    return NULL;
  }
  // we don't really care about performances here and laziness commands to
  // put the switch inside the for-loop
  for (unsigned int i = 0; i < n; i++) {
    switch(variant) {
      case 123:
        ret[i] = helper_cast<T>::cast(i % 10);
        break;
      case 321:
        ret[i] = helper_cast<T>::cast((2147483647 * i) % 10);
        break;
      case 101:
        ret[i] = helper_cast<T>::cast(i % 2);
        break;
      case 110:
        ret[i] = helper_cast<T>::cast(((2147483647 * i) % 13) % 2);
        break;
    }
  }
  return NULL; // should never be reached
}

template <typename T> bool cmp(T *src1, T *src2, unsigned int n) {
  return memcmp(src1, src2, n * sizeof(T)) == 0;
}

template <typename T> void del(T *ptr) { free(ptr); }

#endif

// ----------------------------------------------------------------------------

template <typename T> T *get123(unsigned int n) { return getXXX<T>(n, 123); }
template <typename T> T *get321(unsigned int n) { return getXXX<T>(n, 321); }
template <typename T> T *get101(unsigned int n) { return getXXX<T>(n, 101); }
template <typename T> T *get110(unsigned int n) { return getXXX<T>(n, 110); }

// ----------------------------------------------------------------------------

#endif
