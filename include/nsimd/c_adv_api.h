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

#ifndef NSIMD_C_ADV_API_H
#define NSIMD_C_ADV_API_H

#include <nsimd/nsimd.h>

#if NSIMD_C >= 2011

NSIMD_INLINE void nsimd_c11_type_unsupported(void) {}

/* ------------------------------------------------------------------------- */

#include <nsimd/c_adv_api_functions.h>

/* ------------------------------------------------------------------------- */
/* We add by hand parametrized loads/stores. */

/* loads */

#define nsimd_load_aligned(type, ptr) nsimd_loada(type, ptr)
#define nsimd_load_unaligned(type, ptr) nsimd_loadu(type, ptr)

#define nsimd_load(alignment, type, ptr)                                      \
  NSIMD_PP_CAT_2(nsimd_load_, alignment)(type, ptr)

/* stores */

#define nsimd_store_aligned(ptr, vec) nsimd_storea(ptr, vec)
#define nsimd_store_unaligned(ptr, vec) nsimd_storeu(ptr, vec)

#define nsimd_store(alignment, ptr, vec)                                      \
  NSIMD_PP_CAT_2(nsimd_store_, alignment)(ptr, vec)


#endif /* NSIMD_C >= 2011 */

#endif /* NSIMD_C_ADV_API_HPP */
