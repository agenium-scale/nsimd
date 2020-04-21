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
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>

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

template <typename T>
__device__ T device_cmp_numbers(const T &src1, const T &src2) {
  return T(nsimd::gpu_eq(src1, src2) ? 1 : 0);
}

template <>
__device__ f32 device_cmp_numbers(const f32 &src1, const f32 &src2) {
  if (nsimd::gpu_is_nan(src1) || nsimd::gpu_is_nan(src2)) {
    return nsimd::gpu_is_nan(src1) && nsimd::gpu_is_nan(src2);
  }
  return f32(nsimd::gpu_eq(src1, src2) ? 1 : 0);
}

template <>
__device__ f64 device_cmp_numbers(const f64 &src1, const f64 &src2) {
  if (nsimd::gpu_is_nan(src1) || nsimd::gpu_is_nan(src2)) {
    return nsimd::gpu_is_nan(src1) && nsimd::gpu_is_nan(src2);
  }
  return f64(nsimd::gpu_eq(src1, src2) ? 1 : 0);
}

template <>
__device__ f16 device_cmp_numbers(const f16 &src1, const f16 &src2) {
  if (nsimd::gpu_is_nan(src1) || nsimd::gpu_is_nan(src2)) {
    return nsimd::gpu_is_nan(src1) && nsimd::gpu_is_nan(src2);
  }
  return f16(nsimd::gpu_eq(src1, src2) ? 1 : 0);
}

// perform reduction on blocks first, note that this could be optimized
// but to check correctness we don't need it now
template <typename T>
__global__ void device_cmp_blocks(T *src1, T *src2, int n) {
  extern __shared__ char buf_[];
  T *buf = (T*)buf_;
  int tid = threadIdx.x;
  int i = tid + blockIdx.x * blockDim.x;
  if (i < n) {
    buf[tid] = device_cmp_numbers(src1[i], src2[i]);
  }

  const int block_start = blockIdx.x * blockDim.x;
  const int block_end = block_start + blockDim.x;

  int size;
  if (block_end < n) {
    size = blockDim.x;
  } else {
    size = n - block_start;
  }

  __syncthreads();
  for (int s = size / 2; s != 0; s /= 2) {
    if (tid < s && i < n) {
      buf[tid] = nsimd::gpu_mul(buf[tid], buf[tid + s]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    src1[i] = buf[0];
  }
}

template <typename T>
__global__ void device_cmp_array(int *dst, T *src1, int n, int step) {
  // reduction on the whole vector
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  T buf = T(1);
  for (int i = 0; i < n; i += step) {
    buf = nsimd::gpu_mul(buf, src1[i]);
  }
  if (tid == 0) {
    *dst = int(buf);
  }
}

#define cuda_check_error(ans) gpuCheck((ans), __FILE__, __LINE__)
inline int gpuCheck(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    return -1;
  }
  return 0;
}

#define cuda_assert(ans)                                                      \
  if (gpuCheck((ans), __FILE__, __LINE__)) {                                  \
    exit((ans));                                                              \
  }

template <typename T> T *get000(unsigned int n) {
  T *ret;
  if (cuda_check_error(cudaMalloc((void **)&ret, n * sizeof(T)))) {
    return NULL;
  }
  if (cuda_check_error(cudaMemset((void *)ret, 0, n * sizeof(T)))) {
    cudaFree((void *)ret);
    return NULL;
  }
  return ret;
}

template <typename T> T *getXXX(unsigned int n, int variant) {
  T *ret;
  if (cuda_check_error(cudaMalloc((void **)&ret, n * sizeof(T)))) {
    return NULL;
  }
  // clang-format off
  device_fill<<<(n + 127) / 128, 128>>>(ret, int(n), variant);
  // clang-format on
  return ret;
}

template <typename T> bool cmp(T *src1, T *src2, unsigned int n) {
  int host_ret;
  int *device_ret;
  cuda_assert(cudaMalloc((void **)&device_ret, sizeof(int)));

  // clang-format off
  const int block_size1 = 512;
  device_cmp_blocks
    <<<(n + block_size1 - 1) / block_size1, block_size1, block_size1 * sizeof(T)>>>
    (src1, src2, int(n));
  // clang-format on
  cuda_assert(cudaDeviceSynchronize());

  // clang-format off
  const int block_size2 = 32;
  device_cmp_array<<<1, block_size2>>>(device_ret, src1, int(n), block_size1);
  // clang-format on
  cuda_assert(cudaDeviceSynchronize());

  cuda_assert(cudaMemcpy((void *)&host_ret, (void *)device_ret, sizeof(int),
                         cudaMemcpyDeviceToHost));
  cuda_assert(cudaFree((void *)device_ret));
  return bool(host_ret);
}

template <typename T> void del(T *ptr) { cudaFree(ptr); }

#elif defined(NSIMD_ROCM)

// ----------------------------------------------------------------------------
// HIP

template <typename T>
__global__ void device_fill(T *dst, unsigned int n, int variant) {
  unsigned int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
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
__global__ void device_cmp_blocks(T *src1, T *src2, unsigned int n) {
  extern __shared__ char buf_[]; // size of a block
  T *buf = (T*)buf_;
  unsigned int tid = hipThreadIdx_x;
  unsigned int i = tid + hipBlockIdx_x * hipBlockDim_x;
  if (i < n) {
    buf[tid] = T(nsimd::gpu_eq(src1[i], src2[i]) ? 1 : 0);
  }
  __syncthreads();
  for (unsigned int s = hipBlockDim_x / 2; s != 0; s /= 2) {
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
__global__ void device_cmp_array(int *dst, T *src1, unsigned int n) {
  // reduction on the whole vector
  unsigned int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  T buf = T(1);
  for (unsigned int i = 0; i < n; i += blockDim.x) {
    buf = nsimd::gpu_mul(buf, src1[i]);
  }
  if (i == 0) {
    dst[0] = int(buf);
  }
}

int hip_do(hipError_t code) {
  if (code != hipSuccess) {
    std::cerr << "ERROR: " << hipGetErrorString(code) << std::endl;
    return -1;
  }
  return 0;
}

template <typename T> T *get000(unsigned int n) {
  T *ret;
  if (hip_do(hipMalloc((void **)&ret, n * sizeof(T)))) {
    return NULL;
  }
  if (hip_do(hipMemset((void *)ret, 0, n * sizeof(T)))) {
    hipFree((void *)ret);
    return NULL;
  }
  return ret;
}

template <typename T> T *getXXX(unsigned int n, int variant) {
  T *ret;
  if (hip_do(hipMalloc((void **)&ret, n * sizeof(T)))) {
    return NULL;
  }
  hipLaunchKernelGGL(device_fill, (n + 127) / 128, 128, 0, 0, ret, n, variant);
  return ret;
}

template <typename T> bool cmp(T *src1, T *src2, unsigned int n) {
  int host_ret;
  int *device_ret;
  if (hip_do(hipMalloc((void **)&device_ret, sizeof(int)))) {
    return false;
  }
  hipLaunchKernelGGL(device_cmp_blocks, (n + 127) / 128, 128, 128 * sizeof(T),
                     0, src1, src2, n);
  hipLaunchKernelGGL(device_cmp_array, (n + 127) / 128, 128, 0, 0, device_ret,
                     src1, n);
  if (hip_do(hipMemcpy((void *)&host_ret, (void *)device_ret, sizeof(int),
                 hipMemcpyDeviceToHost))) {
    host_ret = 0;
  }
  hipFree((void *)device_ret);
  return bool(host_ret);
}

template <typename T> void del(T *ptr) { hipFree(ptr); }

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
  errno = 0;
  T *ret = (T *)malloc(n * sizeof(T));
  if (ret == NULL) {
    std::cerr << "ERROR: " << strerror(errno) << std::endl;
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
  return ret;
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
