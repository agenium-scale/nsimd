# Copyright (c) 2019 Agenium Scale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import argparse

"""
This script can be used to (re)generate the fixed_point.hpp header file.
"""

## -----------------------------------------------------------------------------

arithmetic_op_template = """\
template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> {op_name}(pack<lf, rt> a0, pack<lf, rt> a1)
{{
  pack<lf, rt> res;
  res.val = simd_{op_name}<lf, rt>(a0.val, a1.val);
  return res;
}}
"""

comparison_op_template = """\
template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> {op_name}(pack<lf, rt> a0, pack<lf, rt> a1)
{{
  packl<lf, rt> res;
  res.val = simd_{op_name}(a0.val, a1.val);
  return res;
}}
"""

bitwise_op_template = """\
template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> {op_name}b(pack<lf, rt> a0, pack<lf, rt> a1)
{{
  pack<lf, rt> res;
  res.val = simd_{op_name}b(a0.val, a1.val);
  return res;
}}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> {op_name}l(packl<lf, rt> a0, packl<lf, rt> a1)
{{
  packl<lf, rt> res;
  res.val = simd_{op_name}l(a0.val, a1.val);
  return res;
}}
"""

math_func_template = """\
template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> {op_name}(pack<lf, rt> a0)
{{
  pack<lf, rt> res;
  res.val = simd_{op_name}(a0.val);
  return res;
}}
"""

file_template = """\
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

#ifndef NSIMD_MODULES_FIXED_POINT_HPP
#define NSIMD_MODULES_FIXED_POINT_HPP

#include "fixed_point/fixed.hpp"
#include "fixed_point/simd.hpp"
#include "fixed_point/simd_math.hpp"

#include <nsimd/nsimd.h>

namespace nsimd
{{
namespace fixed_point
{{
// -----------------------------------------------------------------------------
// ------------------------ Types definitions and len --------------------------
// -----------------------------------------------------------------------------

template <uint8_t lf, uint8_t rt>
struct pack
{{
  using scalar_type =fp_t<lf, rt>; 
  using base_type = fp_t<lf, rt>;
  fpsimd_t<lf, rt> val;
}};

template <uint8_t lf, uint8_t rt>
struct packl
{{
  using scalar_type = fp_t<lf, rt>;
  using base_type = typename fp_t<lf, rt>::logical_type;
  fpsimdl_t<lf, rt> val;
}};

template <uint8_t lf, uint8_t rt>
constexpr size_t len(const fp_t<lf, rt> &)
{{
  return fpsimd_n(fp_t<lf, rt>());
}}

template <uint8_t lf, uint8_t rt>
constexpr size_t len(const pack<lf, rt> &)
{{
  return fpsimd_n(fpsimd_t<lf, rt>());
}}

// -----------------------------------------------------------------------------
// ------------------- Basic arithmetic operators ------------------------------
// -----------------------------------------------------------------------------

{arithmetic_ops}
// -----------------------------------------------------------------------------
// ------------------- Comparison operators ------------------------------------
// -----------------------------------------------------------------------------

{comparison_ops}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> if_else1(packl<lf, rt> a0, pack<lf, rt> a1,
  pack<lf, rt> a2)
{{
  pack<lf, rt> res;
  res.cal = simd_if_else1(a0.val, a1.val, a2.val);
  return res;
}}

// -----------------------------------------------------------------------------
// ------------------- Bitwise operators  --------------------------------------
// -----------------------------------------------------------------------------

{bitwise_ops}
// -----------------------------------------------------------------------------
// ------------------- Math functions ------------------------------------------
// -----------------------------------------------------------------------------


{math_functions}
// -----------------------------------------------------------------------------
// -------------------- Load functions -----------------------------------------
// -----------------------------------------------------------------------------

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE pack<lf, rt> loadu(typename pack<lf, rt>::base_type *p)
{{
  pack<lf, rt> res;
  res.val = simd_loadu<lf, rt>(p);
  return res;
}}

// template <typename vec_t>
// NSIMD_INLINE vec_t loadu(typename vec_t::base_type *p)
// {{
//   return loadu(p);
// }}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE packl<lf, rt> loadlu(typename packl<lf, rt>::base_type *p)
{{
  packl<lf, rt> res;
  res.val = simd_loadlu<lf, rt>(p);
  return res;
}}

// template <typename vecl_t>
// NSIMD_INLINE vecl_t loadlu(typename vecl_t::base_type *p)
// {{
//   return loadlu(p);
// }}

// -----------------------------------------------------------------------------
// -------------------- Store functions ----------------------------------------
// -----------------------------------------------------------------------------

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE void storeu(typename pack<lf, rt>::base_type *p, pack<lf, rt> &v)
{{
  simd_storeu<lf, rt>(p, v.val);
}}

template <uint8_t lf, uint8_t rt>
NSIMD_INLINE void storelu(typename packl<lf, rt>::base_type *p, packl<lf, rt> v)
{{
  simd_storelu<lf, rt>(p, v.val);
}}

}} // namespace fixed_point
}} // namespace nsimd

#endif
"""

## -----------------------------------------------------------------------------

arithmetic_ops = ["add", "sub", "mul", "div"]
comparison_ops = ["eq", "ne", "le", "lt", "ge", "gt"]
bitwise_ops = ["and", "andnot", "not", "or", "xor"]
math_funcs = ["rec"]

def get_function_decl(template, ops):
    res = ""
    for op in ops:
        print("Generating wrapper for : " + op)
        res += template.format(op_name=op)
        res += "\n"
    return res

if __name__ == "__main__":
    src = file_template.format(
        arithmetic_ops=get_function_decl(arithmetic_op_template, arithmetic_ops),
        comparison_ops=get_function_decl(comparison_op_template, comparison_ops),
        bitwise_ops=get_function_decl(bitwise_op_template, bitwise_ops),
        math_functions=get_function_decl(math_func_template, math_funcs)
    )
    with open("fixed_point.hpp", "w") as fp:
        fp.write(src)
    os.system('clang-format -style=file -i fixed_point.hpp')
        
