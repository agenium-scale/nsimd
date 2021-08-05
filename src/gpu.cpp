/*

Copyright (c) 2021 Agenium Scale

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

#define NSIMD_INSIDE
#include <nsimd/nsimd.h>

#if defined(NSIMD_ONEAPI) && NSIMD_CXX > 0

// ----------------------------------------------------------------------------
// oneAPI

// NSIMD error handler
namespace nsimd {
namespace oneapi {
template <typename Exception = sycl::exception>
struct sycl_async_error_handler {
  void operator()(const sycl::exception_list &elist) {
    for (const auto &exc : elist) {
      try {
        std::rethrow_exception(exc);
      } catch (const Exception &exc) {
        fprintf(stderr, "NSIMD Internal error:\n\tError: %s %s %d\n",
                exc.what(), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }
    }
  }
};
} // namespace oneapi
} // namespace nsimd

extern "C" {

// Singleton to get default oneAPI queue
NSIMD_DLLSPEC void *nsimd_oneapi_default_queue() {
  static sycl::queue ret(sycl::default_selector{},
                         sycl_async_error_handler<>{});
  return (void *)&ret;
}

NSIMD_DLLSPEC nsimd_nat nsimd_kernel_param(nsimd_nat nb_items,
                                           nsimd_nat block_size) {
  return block_size * ((nb_items + block_size - 1) / block_size);
}

} // extern "C"

#elif defined(NSIMD_CUDA) || defined(NSIMD_ROCM)

// ----------------------------------------------------------------------------
// CUDA/ROCm

NSIMD_DLLSPEC nsimd_nat nsimd_kernel_param(nsimd_nat nb_items,
                                           nsimd_nat block_size) {
  return (nb_items + block_size - 1) / block_size;
}

#else

// ----------------------------------------------------------------------------
// CPU/SIMD

NSIMD_DLLSPEC nsimd_nat nsimd_kernel_param(nsimd_nat nb_items,
                                           nsimd_nat block_size) {
  return nb_items / block_size;
}

// ----------------------------------------------------------------------------

#endif
