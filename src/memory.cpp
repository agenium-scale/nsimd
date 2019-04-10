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

#ifdef NSIMD_IS_MSVC
  #include <malloc.h>
#else
  #ifndef _POSIX_C_SOURCE
    #define _POSIX_C_SOURCE 200112L
  #endif
  #include <stdlib.h>
#endif

// ----------------------------------------------------------------------------

#define NSIMD_INSIDE
#include <nsimd/nsimd.h>

// ----------------------------------------------------------------------------

extern "C" {

NSIMD_DLLEXPORT void *nsimd_aligned_alloc(nat n) {
#ifdef NSIMD_IS_MSVC
  return _aligned_malloc(n, NSIMD_MAX_ALIGNMENT);
#else
  void *ptr;
  if (posix_memalign(&ptr, NSIMD_MAX_ALIGNMENT, (size_t)n)) {
    return NULL;
  } else {
    return ptr;
  }
#endif
}

NSIMD_DLLEXPORT void nsimd_aligned_free(void *ptr) {
#ifdef NSIMD_IS_MSVC
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

} // extern "C"

// ----------------------------------------------------------------------------

namespace nsimd {

NSIMD_DLLEXPORT void *aligned_alloc(nat n) {
  return nsimd_aligned_alloc(n);
}

NSIMD_DLLEXPORT void aligned_free(void *ptr) {
  nsimd_aligned_free(ptr);
}

} // namespace nsimd
