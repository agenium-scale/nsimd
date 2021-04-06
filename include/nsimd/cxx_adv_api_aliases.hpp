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

#ifndef NSIMD_CXX_ADV_API_ALIASES_HPP
#define NSIMD_CXX_ADV_API_ALIASES_HPP

#include <nsimd/cxx_adv_api.hpp>

namespace nsimd {

/* ------------------------------------------------------------------------- */

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, N, SimdExt> fabs(pack<T, N, SimdExt> const &a0) {
  return abs(a0);
}

/* ------------------------------------------------------------------------- */

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, N, SimdExt> fmin(pack<T, N, SimdExt> const &a0,
                         pack<T, N, SimdExt> const &a1) {
  return min(a0, a1);
}

/* ------------------------------------------------------------------------- */

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, N, SimdExt> fmax(pack<T, N, SimdExt> const &a0,
                         pack<T, N, SimdExt> const &a1) {
  return max(a0, a1);
}

/* ------------------------------------------------------------------------- */

} // namespace nsimd

#endif
