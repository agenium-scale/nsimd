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

#ifndef NSIMD_MODULES_CONSTANTS_HPP
#define NSIMD_MODULES_CONSTANTS_HPP

#include "fixed_point/fixed.hpp"

namespace nsimd
{
namespace fixed_point
{
namespace constants
{
#define DEFINE_CONSTANT(name, value)                                                     \
  template <unsigned char lf, unsigned char rt>                                          \
  constexpr inline fp_t<lf, rt> name()                                                   \
  {                                                                                      \
    return fp_t<lf, rt>(value);                                                          \
  }

DEFINE_CONSTANT(zero, 0);
DEFINE_CONSTANT(one, 1);
DEFINE_CONSTANT(two, 2);
DEFINE_CONSTANT(neg, -1);
DEFINE_CONSTANT(e, 2.718281828469045235360287471352662497757247093);
DEFINE_CONSTANT(log2_e, 1.44269504089);
DEFINE_CONSTANT(log2_10, 3.32192809489);
DEFINE_CONSTANT(pi, 3.14159265359);
DEFINE_CONSTANT(twopi, 2 * 3.14159265359);
DEFINE_CONSTANT(halfpi, 3.14159265359 / 2);
DEFINE_CONSTANT(pi_cvt, 0.01745329251); // pi / 180

} // namespace constants
} // namespace fixed_point
} // namespace nsimd

#endif
