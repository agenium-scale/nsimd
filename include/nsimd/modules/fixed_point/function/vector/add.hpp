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

#ifndef NSIMD_MODULES_FUNCTION_VECTOR_ADD_HPP
#define NSIMD_MODULES_FUNCTION_VECTOR_ADD_HPP

#include "fixed_point/fixed_vector.hpp"
#include "fixed_point/simd.hpp"

namespace nsimd
{
namespace fixed_point
{
template <unsigned char _lf, unsigned char _rt>
inline void add(fpv_t<_lf, _rt> &c, const fpv_t<_lf, _rt> &a, const fpv_t<_lf, _rt> &b)
{
  if(c.size() < b.size())
  {
    c.resize(b.size());
  }
  using p_type = typename fp_t<_lf, _rt>::value_type;
  p_type *aa = (p_type *) a.data();
  p_type *bb = (p_type *) b.data();
  p_type *cc = (p_type *) c.data();

  int card = fpsimd_n<_lf, _rt>();
  int i = 0;
  int end = int(a.size()) + 1 - card; // In case of small input
  for(; i < end; i += card)
  {
    // fpsimd_t<_lf,_rt> aaa( &(aa[i]) );
    // fpsimd_t<_lf,_rt> bbb( &(bb[i]) );
    // simd_add<_lf,_rt>( aaa , aaa , bbb );
    // simd_store<_lf,_rt>( aaa , &(cc[i]) );
  }
  for(; i < a.size(); ++i)
  {
    c[i] = a[i] + b[i];
  }
}

// Compatibility with base types
template <unsigned char _lf, unsigned char _rt, typename T>
inline void add(fpv_t<_lf, _rt> &c, const fpv_t<_lf, _rt> &a, const T &b)
{
  if(c.size() < b.size())
  {
    c.resize(b.size());
  }
  using p_type = typename fp_t<_lf, _rt>::value_type;
  p_type *aa = (p_type *) a.data();
  p_type *cc = (p_type *) c.data();

  // To be sure that the conversion only happens once
  fp_t<_lf, _rt> b_fp(b);
  fpsimd_t<_lf, _rt> bbb(b_fp._raw);

  int card = fpsimd_n<_lf, _rt>();
  int i = 0;
  int end = int(a.size()) + 1 - card; // In case of small input
  for(; i < end; i += card)
  {
    // fpsimd_t<_lf,_rt> aaa( &(aa[i]) );
    // simd_add<_lf,_rt>( aaa , aaa , bbb );
    // simd_store<_lf,_rt>( aaa , &(cc[i]) );
  }
  for(; i < a.size(); ++i)
  {
    c[i] = a[i] + b[i];
  }
}

template <unsigned char _lf, unsigned char _rt, typename T>
inline void add(fpv_t<_lf, _rt> &c, const T &b, const fpv_t<_lf, _rt> &a)
{
  add(c, a, b);
}

// Operator overload with base type compatibility
template <unsigned char _lf, unsigned char _rt>
inline fpv_t<_lf, _rt> operator+(const fpv_t<_lf, _rt> &a, const fpv_t<_lf, _rt> &b)
{
  fpv_t<_lf, _rt> c(a.size());
  add(c, a, b);
  return c;
}

template <unsigned char _lf, unsigned char _rt, typename T>
inline fpv_t<_lf, _rt> operator+(const fpv_t<_lf, _rt> &a, const T &b)
{
  fpv_t<_lf, _rt> c(a.size());
  add(c, a, b);
  return c;
}

template <unsigned char _lf, unsigned char _rt, typename T>
inline fpv_t<_lf, _rt> operator+(const T &b, const fpv_t<_lf, _rt> &a)
{
  return (a + b);
}

} // namespace fixed_point
} // namespace nsimd

#endif
