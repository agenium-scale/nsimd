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

#ifndef NSIMD_MODULES_SPMD_COMMON_HPP
#define NSIMD_MODULES_SPMD_COMMON_HPP

#include <nsimd/nsimd.h>
#include <nsimd/scalar_utilities.h>
#include <iostream>
#include <cstring>
#include <cerrno>
#include <cstdlib>

// ----------------------------------------------------------------------------
// CUDA

#if defined(NSIMD_CUDA)

template <typename T> __device__ bool cmp_Ts(T a, T b) { return a == b; }

__device__ bool cmp_Ts(__half a, __half b) {
  return __half_as_short(a) == __half_as_short(b);
}

__device__ bool cmp_Ts(float a, float b) {
  return __float_as_int(a) == __float_as_uint(b);
}

__device__ bool cmp_Ts(double a, double b) {
  return __double_as_longlong(a) == __double_as_longlong(b);
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
    buf[tid] = T(cmp_Ts(src1[i], src2[i]) ? 1 : 0);
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

template <typename T> bool cmp(T *src1, T *src2, unsigned int n) {
  int host_ret;
  int *device_ret;
  if (cudaMalloc((void **)&device_ret, sizeof(int)) != cudaSuccess) {
    std::cerr << "ERROR: cannot cudaMalloc " << sizeof(int) << " bytes\n";
    exit(EXIT_FAILURE);
  }
  device_cmp_blocks<<<(n + 127) / 128, 128, 128 * sizeof(T)>>>(src1, src2,
                                                               int(n));
  device_cmp_array<<<(n + 127) / 128, 128>>>(device_ret, src1, int(n));
  cudaMemcpy((void *)&host_ret, (void *)device_ret, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaFree((void *)device_ret);
  return bool(host_ret);
}

template <typename T> void del(T *ptr) { cudaFree(ptr); }

#elif defined(NSIMD_ROCM)

// ----------------------------------------------------------------------------
// HIP

// perform reduction on blocks first, note that this could be optimized
// but to check correctness we don't need it now
template <typename T>
__global__ void device_cmp_blocks(T *src1, T *src2, unsigned int n) {
  extern __shared__ char buf_[]; // size of a block
  T *buf = (T*)buf_;
  unsigned int tid = hipThreadIdx_x;
  unsigned int i = tid + hipBlockIdx_x * hipBlockDim_x;
  if (i < n) {
    //printf("DEBUG: %d: %f vs. %f\n", i, (double)src1[i], (double)src2[i]);
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

template <typename T> bool cmp(T *src1, T *src2, unsigned int n) {
  return memcmp(src1, src2, n * sizeof(T)) == 0;
}

#endif

// ----------------------------------------------------------------------------

#endif
