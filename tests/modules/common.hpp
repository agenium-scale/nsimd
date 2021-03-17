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

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <iostream>
#include <nsimd/nsimd.h>
#include <nsimd/scalar_utilities.h>

// ----------------------------------------------------------------------------
// Num threads per block on device

#if defined(NSIMD_CUDA) || defined(NSIMD_ROCM) || defined(NSIMD_ONEAPI)
#define THREADS_PER_BLOCK 128
#else
#define THREADS_PER_BLOCK 1
#endif

// ----------------------------------------------------------------------------
// Common code for devices

#if defined(NSIMD_CUDA) || defined(NSIMD_ROCM)

template <typename T> __device__ bool cmp_Ts(T a, T b) { return a == b; }

__device__ bool cmp_Ts(__half a, __half b) {
  return __half_as_short(a) == __half_as_short(b);
}

__device__ bool cmp_Ts(float a, float b) {
  return __float_as_int(a) == __float_as_int(b);
}

__device__ bool cmp_Ts(double a, double b) {
  return __double_as_longlong(a) == __double_as_longlong(b);
}

#elif defined(NSIMD_ONEAPI)

template <typename T> bool cmp_Ts(const T a, const T b) { return a == b; }

bool cmp_Ts(const sycl::half a, const sycl::half b) {
  i16 a_i16;
  memcpy((void *)&a_i16, (void *)&a, sizeof(a_i16));
  i16 b_i16;
  memcpy((void *)&b_i16, (void *)&b, sizeof(b_i16));
  return a_i16 == b_i16;
}

bool cmp_Ts(float a, float b) {
  return nsimd_scalar_reinterpret_i32_f32(a) ==
         nsimd_scalar_reinterpret_i32_f32(b);
}

bool cmp_Ts(double a, double b) {
  return nsimd_scalar_reinterpret_i64_f64(a) ==
         nsimd_scalar_reinterpret_i64_f64(b);
}

#endif

// ----------------------------------------------------------------------------
// CUDA

#if defined(NSIMD_CUDA)

// perform reduction on blocks first, note that this could be optimized
// but to check correctness we don't need it now
template <typename T>
__global__ void device_cmp_blocks(T *src1, T *src2, int n) {
  extern __shared__ char buf_[]; // size of a block
  T *buf = (T *)buf_;
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
    }
    __syncthreads();
  }
  if (tid == 0) {
    src1[i] = buf[0];
  }
}

template <typename T>
__global__ void device_cmp_array(int *dst, T *src1, int n) {
  // reduction on the whole vector
  T buf = T(1);
  for (int i = 0; i < n; i += blockDim.x) {
    buf = nsimd::gpu_mul(buf, src1[i]);
  }
  int i = threadIdx.x + blockIdx.x * blockDim.x;
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

template <typename T> bool cmp(T *src1, T *src2, unsigned int n, double) {
  return cmp(src1, src2, n);
}

template <typename T> void del(T *ptr) { cudaFree(ptr); }

#elif defined(NSIMD_ROCM)

// ----------------------------------------------------------------------------
// ROCm

// perform reduction on blocks first, note that this could be optimized
// but to check correctness we don't need it now
template <typename T>
__global__ void device_cmp_blocks(T *src1, T *src2, size_t n) {
  extern __shared__ char buf_[]; // size of a block
  T *buf = (T *)buf_;
  size_t tid = hipThreadIdx_x;
  size_t i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (i < n) {
    buf[tid] = T(cmp_Ts(src1[i], src2[i]) ? 1 : 0);
  }

  const size_t block_start = hipBlockIdx_x * hipBlockDim_x;
  const size_t block_end = block_start + hipBlockDim_x;

  size_t size;
  if (block_end < n) {
    size = hipBlockDim_x;
  } else {
    size = n - block_start;
  }

  __syncthreads();
  for (size_t s = size / 2; s != 0; s /= 2) {
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
__global__ void device_cmp_array(int *dst, T *src1, size_t n) {
  // reduction on the whole vector
  T buf = T(1);
  for (size_t i = 0; i < n; i += blockDim.x) {
    buf = nsimd::gpu_mul(buf, src1[i]);
  }
  size_t i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
  if (i == 0) {
    dst[0] = int(buf);
  }
}

template <typename T> bool cmp(T *src1, T *src2, size_t n) {
  int host_ret;
  int *device_ret;
  if (hipMalloc((void **)&device_ret, sizeof(int)) != hipSuccess) {
    return false;
  }
  hipLaunchKernelGGL(device_cmp_blocks, (n + 127) / 128, 128, 128 * sizeof(T),
                     0, src1, src2, n);
  hipLaunchKernelGGL(device_cmp_array, (n + 127) / 128, 128, 0, 0, device_ret,
                     src1, n);
  hipMemcpy((void *)&host_ret, (void *)device_ret, sizeof(int),
            hipMemcpyDeviceToHost);
  hipFree((void *)device_ret);
  return bool(host_ret);
}

template <typename T> bool cmp(T *src1, T *src2, size_t n, double) {
  return cmp(src1, src2, n);
}

template <typename T> void del(T *ptr) { hipFree(ptr); }

// ----------------------------------------------------------------------------
// oneAPI

#elif defined(NSIMD_ONEAPI)

// perform reduction on blocks first, note that this could be optimized
// but to check correctness we don't need it now
template <typename T>
void device_cmp_blocks(T *src1, T *src2, const size_t n,
                       sycl::nd_item<1> item) {
  char buf_[THREADS_PER_BLOCK]; // size of a block
  T *buf = (T *)buf_;
  size_t tid = item.get_local_id().get(0);
  size_t i = item.get_global_id().get(0);
  if(i >= n){ return; }
  buf[tid] = T(cmp_Ts(src1[i], src2[i]) ? 1 : 0);

  sycl::nd_range<1> nd_range = item.get_nd_range();
  sycl::range<1> range = nd_range.get_local_range();
  const size_t block_start = nd_range.get_offset();
  assert(block_start < n);
  const size_t block_end = block_start + range.size();

  size_t size;
  if (block_end < n) {
    size = range.size();
  } else {
    size = n - block_start;
  }
  item.barrier(sycl::access::fence_space::local_space);

  // reduction mul on the block (sub-group)

  /*
  TODO: test the following usage correctness
  sycl::ONEAPI::sub_group sg = item.get_sub_group();
  if(tid == 0){
    src1[i] = sycl::ONEAPI::reduce(sg, buf[0], sycl::ONEAPI::multiplies<>());
  }
  */

    for (size_t s = size / 2; s != 0; s /= 2) {
      if(tid < s){
        buf[tid] = nsimd::oneapi_mul(buf[tid], buf[tid + s]);
      }
      // book p 217
      // it is critically important that either all work-items
      // in the group execute the barrier or no work-items in
      // the group execute the barrier.
      item.barrier(sycl::access::fence_space::local_space);
    }
  if (tid == 0) {
    src1[i] = buf[0];
  }
}

template <typename T>
void device_cmp_array(int *dst, T *src1, const size_t n,
                      sycl::nd_item<1> item) {
  // reduction mul on the whole vector
  T buf = T(1);
  sycl::nd_range<1> nd_range = item.get_nd_range();
  sycl::range<1> range = nd_range.get_local_range();
  for (size_t i = 0; i < n; i += range.size()) {
    buf = nsimd::oneapi_mul(buf, src1[i]);
  }
  size_t i = item.get_global_id().get(0);
  if (i == 0) {
    dst[0] = int(buf);
  }
}

template <typename T> bool cmp(T *src1, T *src2, unsigned int n) {
  sycl::queue q = spmd::_get_global_queue();
  int *device_ret = sycl::malloc_device<int>(n, q);
  if (device_ret == NULL) {
    std::cerr << "ERROR: cannot sycl::malloc_device " << sizeof(int)
              << " bytes\n";
    exit(EXIT_FAILURE);
  }
  sycl::event e1 = q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(n), sycl::range<1>(THREADS_PER_BLOCK)),
      [=](sycl::nd_item<1> item_) {
        device_cmp_blocks(src1, src2, size_t(n), item_);
      });
  e1.wait();
  sycl::event e2 = q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(n), sycl::range<1>(THREADS_PER_BLOCK)),
      [=](sycl::nd_item<1> item_) {
        device_cmp_array(device_ret, src1, size_t(n), item_);
      });
  e2.wait();
  int host_ret;
  q.memcpy((void *)&host_ret, (void *)device_ret, n * sizeof(T)).wait();
  sycl::free(device_ret, q);
  return bool(host_ret);
}

template <typename T> bool cmp(T *src1, T *src2, unsigned int n, double) {
  return cmp(src1, src2, n);
}

template <typename T> void del(T *ptr) {
  sycl::queue q = spmd::_get_global_queue();
  sycl::free(ptr, q);
}

#else

// ----------------------------------------------------------------------------
// SIMD

template <typename T> bool cmp(T *src1, T *src2, unsigned int n) {
  return memcmp(src1, src2, n * sizeof(T)) == 0;
}

inline double to_double(f64 a) { return a; }
inline double to_double(f32 a) { return (double)a; }
inline double to_double(f16 a) { return (double)nsimd_f16_to_f32(a); }

template <typename T>
bool cmp(T *src1, T *src2, unsigned int n, double epsilon) {
  for (unsigned int i = 0; i < n; i++) {
    double a = to_double(src1[i]);
    double b = to_double(src2[i]);
    double ma, mi;

    if (std::isnan(a) && std::isnan(b)) {
      continue;
    }

    if (std::isnan(a) || std::isnan(b)) {
      return false;
    }

    if (std::isinf(a) && std::isinf(b) &&
        ((a > 0 && b > 0) || (a < 0 && b < 0))) {
      continue;
    }

    if (std::isinf(a) || std::isinf(b)) {
      return false;
    }

    a = (a > 0.0 ? a : -a);
    b = (b > 0.0 ? b : -b);
    ma = (a > b ? a : b);
    mi = (a < b ? a : b);

    if (ma == 0.0) {
      continue;
    }

    if ((ma - mi) / ma > epsilon) {
      return false;
    }
  }

  return true;
}

#endif

// ----------------------------------------------------------------------------

#endif
