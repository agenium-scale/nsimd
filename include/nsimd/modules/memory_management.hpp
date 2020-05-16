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

namespace nsimd {

// ----------------------------------------------------------------------------
// CUDA

#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE)

template <typename T> T *device_malloc(size_t sz) {
  void *ptr;
  if (cudaMalloc(&ptr, sz * sizeof(T)) != cudaSuccess) {
    return NULL;
  }
  return (T *)ret;
}

template <typename T> T *device_calloc(size_t sz) {
  void *ptr;
  if (cudaMalloc(&ptr, sz * sizeof(T)) != cudaSuccess) {
    return NULL;
  }
  if (cudaMemset((void *)ret, 0, n * sizeof(T)) != cudaSuccess) {
    cudaFree(ptr);
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

#define nsimd_fill_dev_mem_func(func_name, expr)                    \
  template <typename T> __global__ kernel_##func_name##_(T *ptr, int n) {     \
    if (i < n) {                                                              \
      ptr[i] = (T)(expr);                                                     \
    }                                                                         \
  }                                                                           \
                                                                              \
  template <typename T> void func_name(T *ptr, size_t sz) {                   \
    kernel_##func_name##_<<<(unsigned int)((sz + 127) / 128), 128> > >(       \
        ptr, int(sz));                                                        \
  }

// ----------------------------------------------------------------------------
// ROCm

#elif defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)

// ----------------------------------------------------------------------------
// CPU

#else

template <typename T> T *device_malloc(size_t sz) {
  return (T *)malloc(sz);
}

template <typename T> T *device_calloc(size_t sz) {
  return (T *)calloc(sz, 1);
}

template <typename T> void device_free(T *ptr) { free((void *)ptr); }

template <typename T>
void copy_to_device(T *device_ptr, T *host_ptr, size_t sz) {
  memcpy((void *)device_ptr, (void *)host_ptr, sz);
}

template <typename T>
void copy_to_host(T *host_ptr, T *device_ptr, size_t sz) {
  memcpy((void *)host_ptr, (void *)device_ptr, sz);
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

template <typename T>
struct paired_pointers_t {
  T *device_ptr, *host_ptr;
  size_t sz;
};

template <typename T> paired_pointers_t<T> pair_malloc(size_t sz) {
  paired_pointers_t<T> ret;
  ret.sz = 0;
#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) || \
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
#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) || \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
  ret.device_ptr = device_calloc<T>(sz);
  if (ret.device_ptr == NULL) {
    ret.host_ptr = NULL;
    return ret;
  }
  ret.host_ptr = calloc(sz);
  if (ret.host_ptr == NULL) {
    device_free(ret.device_ptr);
    ret.device_ptr = NULL;
    return ret;
  }
#else
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
#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) || \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
  device_free(p.device_free);
  free((void *)p.host_ptr);
#else
  free((void *)p.host_ptr);
#endif
}

template <typename T> void copy_to_device(paired_pointers_t<T> p) {
#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) || \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
  copy_to_device(p.device_ptr, p.host_ptr, p.sz);
#else
  (void)p;
#endif
}

template <typename T> void copy_to_host(paired_pointers_t<T> p) {
#if defined(NSIMD_CUDA_COMPILING_FOR_DEVICE) || \
    defined(NSIMD_ROCM_COMPILING_FOR_DEVICE)
  copy_to_host(p.host_ptr, p.device_ptr, p.sz);
#else
  (void)p;
#endif
}

} // namespace nsimd

#endif
