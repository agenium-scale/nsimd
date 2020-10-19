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

#ifndef NSIMD_MODULES_MEMORY_MANAGEMENT_HPP
#define NSIMD_MODULES_MEMORY_MANAGEMENT_HPP

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <nsimd/nsimd.h>

#if defined(NSIMD_SYCL_COMPILING_FOR_DEVICE)
#include <CL/sycl.hpp>
#endif

namespace nsimd {

// ----------------------------------------------------------------------------
// CUDA

#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE)

template <typename T> T *device_malloc(size_t sz) {
  void *ret;
  if (cudaMalloc(&ret, sz * sizeof(T)) != cudaSuccess) {
    return NULL;
  }
  return (T *)ret;
}

template <typename T> T *device_calloc(size_t sz) {
  void *ret;
  if (cudaMalloc(&ret, sz * sizeof(T)) != cudaSuccess) {
    return NULL;
  }
  if (cudaMemset((void *)ret, 0, sz * sizeof(T)) != cudaSuccess) {
    cudaFree(ret);
    return NULL;
  }
  return (T *)ret;
}

template <typename T> void device_free(T *ptr) { cudaFree((void *)ptr); }

template <typename T>
void copy_to_device(T *device_ptr, T *host_ptr, size_t sz) {
  cudaMemcpy((void *)device_ptr, (void *)host_ptr, sz * sizeof(T),
             cudaMemcpyHostToDevice);
}

template <typename T>
void copy_to_host(T *host_ptr, T *device_ptr, size_t sz) {
  cudaMemcpy((void *)host_ptr, (void *)device_ptr, sz * sizeof(T),
             cudaMemcpyDeviceToHost);
}

#define nsimd_fill_dev_mem_func(func_name, expr)                              \
  template <typename T>                                                       \
  __global__ void kernel_##func_name##_(T *ptr, int n) {                      \
    int i = threadIdx.x + blockIdx.x * blockDim.x;                            \
    if (i < n) {                                                              \
      ptr[i] = (T)(expr);                                                     \
    }                                                                         \
  }                                                                           \
                                                                              \
  template <typename T> void func_name(T *ptr, size_t sz) {                   \
    kernel_##func_name##_<<<(unsigned int)((sz + 127) / 128), 128>>>(         \
        ptr, int(sz));                                                        \
  }

// ----------------------------------------------------------------------------
// ROCm

#elif defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)

template <typename T> T *device_malloc(size_t sz) {
  void *ret;
  if (hipMalloc(&ret, sz * sizeof(T)) != hipSuccess) {
    return NULL;
  }
  return (T *)ret;
}

template <typename T> T *device_calloc(size_t sz) {
  void *ret;
  if (hipMalloc(&ret, sz * sizeof(T)) != hipSuccess) {
    return NULL;
  }
  if (hipMemset((void *)ret, 0, sz * sizeof(T)) != hipSuccess) {
    hipFree(ret);
    return NULL;
  }
  return (T *)ret;
}

template <typename T> void device_free(T *ptr) { hipFree((void *)ptr); }

template <typename T>
void copy_to_device(T *device_ptr, T *host_ptr, size_t sz) {
  hipMemcpy((void *)device_ptr, (void *)host_ptr, sz * sizeof(T),
            hipMemcpyHostToDevice);
}

template <typename T>
void copy_to_host(T *host_ptr, T *device_ptr, size_t sz) {
  hipMemcpy((void *)host_ptr, (void *)device_ptr, sz * sizeof(T),
            hipMemcpyDeviceToHost);
}

#define nsimd_fill_dev_mem_func(func_name, expr)                              \
  template <typename T>                                                       \
  __global__ void kernel_##func_name##_(T *ptr, unsigned int n) {             \
    unsigned int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;          \
    if (i < n) {                                                              \
      ptr[i] = (T)(expr);                                                     \
    }                                                                         \
  }                                                                           \
                                                                              \
  template <typename T> void func_name(T *ptr, size_t sz) {                   \
    hipLaunchKernelGGL((kernel_##func_name##_<T>),                            \
                       (unsigned int)((sz + 127) / 128), 128, 0, NULL, ptr,   \
                       (unsigned int)sz);                                     \
  }

// -----------------------------------------------------------------------------
// SYCL

#elif defined(NSIMD_SYCLL_COMPILING_FOR_DEVICE)

template <typename T> T *device_malloc(size_t sz) {
  return sycl::malloc_device<T>(sz, sycl::queue(sycl::default_selector()));
}

template <typename T> T *device_calloc(size_t sz) {
  T *ret = sycl::malloc_device<T>(sz, sycl::queue(sycl::default_selector()));
  if (ret == NULL) {
    return NULL;
  }
  sycl::queue(sycl::default_selector()) q;
  q.memset((void *)ret, sz * sizeof(T));
  return ret;
}

template <typename T> void device_free(T *ptr) {
  sycl::queue(sycl::default_selector()) q;
  sycl::free((void *)ptr, q);
}

template <typename T>
void copy_to_device(T *device_ptr, T *host_ptr, size_t sz) {
  sycl::queue(sycl::default_selector()) q;
  q.memcpy((void *)device_ptr, (void *)host_ptr, sz * sizeof(T));
}

template <typename T>
void copy_to_host(T *host_ptr, T *device_ptr, size_t sz) {
  sycl::queue(sycl::default_selector()) q;
  q.memcpy((void *)host_ptr, (void *)device_ptr, sz * sizeof(T));
}

#define nsimd_fill_dev_mem_func(func_name, expr)                              \
  template <typename T>                                                       \
  inline void kernel_##func_name##_(T *ptr, int n, sycl::id<2> item) {        \
    int i = item.get_id(1) * item.get_range()[0] + item.get_id(0);            \
    if (i < n) {                                                              \
      ptr[i] = (T)(expr);                                                     \
    }                                                                         \
  }                                                                           \
                                                                              \
  template <typename T> void func_name(T *ptr, size_t sz) {                   \
    sycl::queue(sycl::default_selector())                                     \
        .parallel_for(sycl::range<2>((sz + 127) / 128, 128),                  \
                      [=](sycl::item<2> item) {                               \
                        kernel_##func_name##_<T>(ptr, (int)sz, item);         \
                      });                                                     \
  }

// ----------------------------------------------------------------------------
// CPU

#else

template <typename T> T *device_malloc(size_t sz) {
  return (T *)malloc(sz * sizeof(T));
}

template <typename T> T *device_calloc(size_t sz) {
  return (T *)calloc(sz * sizeof(T), 1);
}

template <typename T> void device_free(T *ptr) { free((void *)ptr); }

template <typename T>
void copy_to_device(T *device_ptr, T *host_ptr, size_t sz) {
  memcpy((void *)device_ptr, (void *)host_ptr, sz * sizeof(T));
}

template <typename T>
void copy_to_host(T *host_ptr, T *device_ptr, size_t sz) {
  memcpy((void *)host_ptr, (void *)device_ptr, sz * sizeof(T));
}

#define nsimd_fill_dev_mem_func(func_name, expr)                              \
  template <typename T> void func_name(T *ptr, size_t sz) {                   \
    for (size_t i = 0; i < sz; i++) {                                         \
      ptr[i] = nsimd::to<T>(expr);                                            \
    }                                                                         \
  }

#endif

// ----------------------------------------------------------------------------
// Pair of pointers

template <typename T> struct paired_pointers_t {
  T *device_ptr, *host_ptr;
  size_t sz;
};

template <typename T> paired_pointers_t<T> pair_malloc(size_t sz) {
  paired_pointers_t<T> ret;
  ret.sz = 0;
#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) ||                               \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
  ret.device_ptr = device_malloc<T>(sz);
  if (ret.device_ptr == NULL) {
    ret.host_ptr = NULL;
    return ret;
  }
  ret.host_ptr = (T *)malloc(sz);
  if (ret.host_ptr == NULL) {
    device_free(ret.device_ptr);
    ret.device_ptr = NULL;
    return ret;
  }
#else
  // Works with OneAPI unified memory too
  ret.device_ptr = device_malloc<T>(sz);
  ret.host_ptr = ret.device_ptr;
#endif
  ret.sz = sz;
  return ret;
}

template <typename T> paired_pointers_t<T> pair_malloc_or_exit(size_t sz) {
  paired_pointers_t<T> ret = pair_malloc<T>(sz);
  if (ret.device_ptr == NULL) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": error cannot malloc " << sz
              << " bytes" << std::endl;
    exit(EXIT_FAILURE);
  }
  return ret;
}

template <typename T> paired_pointers_t<T> pair_calloc(size_t sz) {
  paired_pointers_t<T> ret;
  ret.sz = 0;
#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) ||                               \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
  ret.device_ptr = device_calloc<T>(sz);
  if (ret.device_ptr == NULL) {
    ret.host_ptr = NULL;
    return ret;
  }
  ret.host_ptr = calloc(sz, 1);
  if (ret.host_ptr == NULL) {
    device_free(ret.device_ptr);
    ret.device_ptr = NULL;
    return ret;
  }
#else
  // Works with OneAPi unified momory too
  ret.device_ptr = device_calloc<T>(sz);
  ret.host_ptr = ret.device_ptr;
#endif
  ret.sz = sz;
  return ret;
}

template <typename T> paired_pointers_t<T> pair_calloc_or_exit(size_t sz) {
  paired_pointers_t<T> ret = pair_calloc<T>(sz);
  if (ret.device_ptr == NULL) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": error cannot calloc " << sz
              << " bytes" << std::endl;
    exit(EXIT_FAILURE);
  }
  return ret;
}

template <typename T> void pair_free(paired_pointers_t<T> p) {
#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) ||                               \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
  device_free(p.device_free);
  free((void *)p.host_ptr);
#else
  // Works for unified memory OneAPI too
  free((void *)p.host_ptr);
#endif
}

template <typename T> void copy_to_device(paired_pointers_t<T> p) {
#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) ||                               \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
  copy_to_device(p.device_ptr, p.host_ptr, p.sz);
#else
  // Works for unified memory OneAPI too
  (void)p;
#endif
}

template <typename T> void copy_to_host(paired_pointers_t<T> p) {
#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) ||                               \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
  copy_to_host(p.host_ptr, p.device_ptr, p.sz);
#else
  // Works for unified memory OneAPI too
  (void)p;
#endif
}

} // namespace nsimd

#endif
