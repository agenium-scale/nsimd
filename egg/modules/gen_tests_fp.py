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
sys.path.append("..")
import common

# -------------------------------------------------------------------------------

def get_filename(opts, op, lf, rt):
    tests_dir = os.path.join(opts.tests_dir, "modules/fixed_point")
    common.mkdir_p(tests_dir) 
    filename = os.path.join(tests_dir, '{}.fp_{}_{}.cpp'.format(op, lf, rt))
    if os.path.exists(filename):
        os.remove(filename)
    if common.can_create_filename(opts, filename):
        return filename
    else:
        return None

includes = """\
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <ctime>

#include <nsimd/nsimd.h>
#include <nsimd/modules/fixed_point.hpp>
"""

arithmetic_aliases = """\
using fp_t = nsimd::fixed_point::fp_t<{lf}, {rt}>;
using vec_t = nsimd::fixed_point::pack<{lf}, {rt}>;
using vecl_t = nsimd::fixed_point::packl<{lf}, {rt}>;
using raw_t = nsimd::fixed_point::pack<{lf}, {rt}>::base_type;
using log_t = nsimd::fixed_point::packl<{lf}, {rt}>::base_type;
const size_t v_size = nsimd::fixed_point::len(fp_t());
"""

# ------------------------------------------------------------------------------
# Utility functions

check = """\
#define CHECK(a) {{ \\
  if (!(a)) {{ \\
    fprintf(stderr, "ERROR: " #a ":%s: %d\\n", \\
            __FILE__, __LINE__); \\
    fflush(stderr); \\
    exit(EXIT_FAILURE); \\
  }} \\
}}

"""

limits = """\
template <uint8_t lf, uint8_t rt>
static constexpr float __get_numeric_precision() {
  return ldexpf(1.0f, -(int)rt);
}

"""

comparison_fp = """\
template <uint8_t lf, uint8_t rt>
bool __compare_values(nsimd::fixed_point::fp_t<lf, rt> val, float ref)
{
  return abs(nsimd::fixed_point::fixed2float<lf, rt>(val) - ref) 
    <= __get_numeric_precision<lf, rt>();
}

"""

comparison_log = """\
template <typename T, uint8_t lf, uint8_t rt>
bool __check_logical_val(T val, nsimd::fixed_point::fp_t<lf, rt> v0, 
    nsimd::fixed_point::fp_t<lf, rt> v1)
{{
  return (((v0._raw {op_val} v1._raw) && (val != 0)) 
      || (!(v0._raw {op_val} v1._raw) && (val == 0)));
}}

"""

gen_random_val = """\
template <uint8_t lf, uint8_t rt>
nsimd::fixed_point::fp_t<lf, rt> __gen_random_val() {
  constexpr float max_val = ldexp(1.0f, (lf - 1) / 2) - 1;
  constexpr float min_val = -ldexp(1.0f, (lf - 1) / 2);
  constexpr int n_vals = (int)ldexp(1.0f, rt);

  const float integral =
      roundf(((float)rand() / (float)RAND_MAX) * (max_val - min_val) + min_val);
  const float decimal =
      (float)(rand() % n_vals) * __get_numeric_precision<lf, rt>();

  return nsimd::fixed_point::fp_t<lf, rt>(integral + decimal);
}

"""

# ------------------------------------------------------------------------------
# Template for arithmetic binary operators

arithmetic_test_template = """\
{includes}
// -----------------------------------------------------------------------------

{decls}
// -----------------------------------------------------------------------------

int main() {{
  {aliases}

  srand(time(NULL));

  // FP vectors
  fp_t tab0_fp[v_size];
  fp_t tab1_fp[v_size];
  fp_t res_fp[v_size];

  // Floating point equivalent
  float tab0_f[v_size];
  float tab1_f[v_size];
  float res_f[v_size];

  for (size_t i = 0; i < v_size; i++) {{
    tab0_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab1_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab0_f[i] = nsimd::fixed_point::fixed2float(tab0_fp[i]);
    tab1_f[i] = nsimd::fixed_point::fixed2float(tab1_fp[i]);
  }}

  vec_t v0_fp = nsimd::fixed_point::loadu<{lf}, {rt}>(tab0_fp);
  vec_t v1_fp = nsimd::fixed_point::loadu<{lf}, {rt}>(tab1_fp);
  vec_t vres_fp = nsimd::fixed_point::{op_name}(v0_fp, v1_fp);
  nsimd::fixed_point::storeu<{lf}, {rt}>(res_fp, vres_fp);

  for (size_t i = 0; i < v_size; i++) {{
    res_f[i] = tab0_f[i] {op_val} tab1_f[i];
  }}

  for(size_t i = 0; i < v_size; i++) {{
    CHECK(__compare_values(res_fp[i], res_f[i]));
  }}
 
  fprintf(stdout, \"Test of {op_name} for fp_t<{lf},{rt}>... OK\\n\");
  return EXIT_SUCCESS;
}}
"""

# ------------------------------------------------------------------------------
# Template for math operators

math_test_template = """\
{includes}
// -----------------------------------------------------------------------------

{decls}

// Rec operator on floating points (avoid to write a particular test for rec)
static inline float rec(const float x){{return 1.0f / x;}}
// -----------------------------------------------------------------------------

int main() {{
  {aliases}

  srand(time(NULL));

  // FP vectors
  fp_t tab0_fp[v_size];
  fp_t res_fp[v_size];

  // Floating point equivalent
  float tab0_f[v_size];
  float res_f[v_size];

  for (size_t i = 0; i < v_size; i++) {{
    tab0_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab0_f[i] = nsimd::fixed_point::fixed2float(tab0_fp[i]);
  }}

  vec_t v0_fp = nsimd::fixed_point::loadu<{lf}, {rt}>(tab0_fp);
  vec_t vres_fp = nsimd::fixed_point::{op_name}(v0_fp);
  nsimd::fixed_point::storeu<{lf}, {rt}>(res_fp, vres_fp);

  for (size_t i = 0; i < v_size; i++) {{
    res_f[i] = {op_name}(tab0_f[i]);
  }}

  for(size_t i = 0; i < v_size; i++) {{
    CHECK(__compare_values(res_fp[i], res_f[i]));
  }}
 
  fprintf(stdout, \"Test of {op_name} for fp_t<{lf},{rt}>... OK\\n\");
  return EXIT_SUCCESS;
}}
"""

# ------------------------------------------------------------------------------
# Comparison operators

comparison_test_template = """\
{includes}
// -----------------------------------------------------------------------------

{decls}
// -----------------------------------------------------------------------------

int main(){{
  {aliases}

  srand(time(NULL));

  // FP vectors
  fp_t tab0_fp[v_size];
  fp_t tab1_fp[v_size];
  log_t resl_fp[v_size];

  for(size_t i = 0; i < v_size; i++) {{
    tab0_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab1_fp[i] = __gen_random_val<{lf}, {rt}>();
  }}
  // Be sure there is at least one equality to test all the cases.
  tab0_fp[0] = tab1_fp[0];
  
  vec_t v0_fp = nsimd::fixed_point::loadu<{lf}, {rt}>(tab0_fp);
  vec_t v1_fp = nsimd::fixed_point::loadu<{lf}, {rt}>(tab1_fp);
  vecl_t vres_fp = nsimd::fixed_point::{op_name}(v0_fp, v1_fp);
  nsimd::fixed_point::storelu<{lf}, {rt}>(resl_fp, vres_fp);

  for(size_t i = 0; i < v_size; i++) {{
    CHECK((__check_logical_val<log_t, {lf}, {rt}>(
        resl_fp[i], tab0_fp[i], tab1_fp[i])));
  }}

  fprintf(stdout, \"Test of {op_name} for fp_t<{lf},{rt}>... OK\\n\");
  return EXIT_SUCCESS;
}}
"""

# ------------------------------------------------------------------------------
# Bitwise binary operators

bitwise_binary_test_template = """\
{includes}
#include <limits>

// -----------------------------------------------------------------------------

{decls}
// -----------------------------------------------------------------------------

int main() {{
  {aliases}
  
  srand(time(NULL));
  
  {typ} tab0[v_size];
  {typ} tab1[v_size];
  {typ} res[v_size];

  for(size_t i = 0; i < v_size; i++)
  {{
    tab0[i] = {rand_statement}
    tab1[i] = {rand_statement}
  }}
  // Be sure there is at least one equality to test all the cases.
  tab0[0] = tab1[0];

  vec{l}_t v0 = nsimd::fixed_point::load{l}u<{lf}, {rt}>(tab0);
  vec{l}_t v1 = nsimd::fixed_point::load{l}u<{lf}, {rt}>(tab1);
  vec{l}_t v_res = nsimd::fixed_point::{op_name}{term}(v0, v1);
  nsimd::fixed_point::store{l}u<{lf}, {rt}>(res, v_res);

  for(size_t i = 0; i < v_size; i++)
  {{
    {typ} a = tab0[i];
    {typ} b = tab1[i];
    {typ} c = res[i];
    CHECK({test_statement});
  }}  
 
  fprintf(stdout, \"Test of {op_name}{term} for fp_t<{lf},{rt}>... OK\\n\");
  return EXIT_SUCCESS;
}}
"""

# ------------------------------------------------------------------------------
# Bitwise unary operators
# Provide one wrapper
# Make it parametric with logical or not X

# ------------------------------------------------------------------------------

load_ops = ["loadu", "loadlu", "loada", "loadla"]
store_ops = ["storeu", "storelu", "storea", "storela"]
arithmetic_ops = [("add", "+"), ("sub", "-"), ("mul", "*"), ("div","/")]
math_ops = ["rec"]
comparison_ops = [("eq","=="), ("ne","!="), ("le","<="), ("lt","<"),
                  ("ge",">="), ("gt",">")]
bitwise_binary_ops = [("and", "c._raw == (a._raw & b._raw)", "c == (a & b)"),
                      ("andnot", "c._raw == (a._raw & ~b._raw)", "c == (a & ~b)"),
                      ("or", "c._raw == (a._raw | b._raw)", "c == (a | b)"),
                      ("xor","c._raw == (~a._raw & b._raw) | (a._raw & ~b._raw)",
                       "c == ((~a & b) | (a & ~b))")]

lf_vals = ["4", "5", "6", "7", "8", "9", "12", "16"]
rt_vals = ["1", "2", "3", "4", "6", "8", "12", "16"]

# -------------------------------------------------------------------------------
# Entry point

def doit(opts):
    print ('-- Generating tests for module fixed_point')
    ## Arithmetic operators
    for op_name, op_val in arithmetic_ops:
        decls = check + limits + comparison_fp + gen_random_val
        for lf in lf_vals:
            for rt in rt_vals:
                content_src = arithmetic_test_template.format(
                    op_name=op_name, op_val=op_val, lf=lf, rt=rt,
                    includes=includes, decls=decls,
                    aliases=arithmetic_aliases.format(lf=lf, rt=rt))
                filename = get_filename(opts, op_name, lf, rt)
                with common.open_utf8(filename) as fp:
                    fp.write(content_src)
                common.clang_format(opts, filename)

    ## Math functions
    for op_name in math_ops:
        decls = check + limits + comparison_fp + gen_random_val
        for lf in lf_vals:
            for rt in rt_vals:
                content_src = math_test_template.format(
                    op_name=op_name, lf=lf, rt=rt,
                    includes=includes, decls=decls,
                    aliases=arithmetic_aliases.format(lf=lf, rt=rt))
                filename = get_filename(opts, op_name, lf, rt)
                with common.open_utf8(filename) as fp:
                    fp.write(content_src)
                common.clang_format(opts, filename)

    ## Comparison operators
    for op_name, op_val in comparison_ops:
        decls = check + limits + comparison_log.format(op_val=op_val) + gen_random_val
        for lf in lf_vals:
            for rt in rt_vals:
                content_src = comparison_test_template.format(
                    op_name=op_name, op_val=op_val, lf=lf, rt=rt,
                    includes=includes, decls=decls,
                    aliases=arithmetic_aliases.format(lf=lf, rt=rt))
                filename = get_filename(opts, op_name, lf, rt)
                with common.open_utf8(filename) as fp:
                    fp.write(content_src)
                common.clang_format(opts, filename)

    ## Boolean operators
    for op_name, s0, s1 in bitwise_binary_ops:
        decls = check + limits + gen_random_val
        for lf in lf_vals:
            for rt in rt_vals:
                # {op}b
                content_src = bitwise_binary_test_template.format(
                    op_name=op_name, lf=lf, rt=rt,
                    includes=includes, decls=decls,
                    rand_statement="__gen_random_val<{lf}, {rt}>();".format(lf=lf, rt=rt),
                    typ="fp_t", test_statement=s0, l="", term="b",
                    aliases=arithmetic_aliases.format(lf=lf, rt=rt))
                filename = get_filename(opts, op_name + "b", lf, rt)
                with common.open_utf8(filename) as fp:
                    fp.write(content_src)
                common.clang_format(opts, filename)

                # {op}l
                content_src = bitwise_binary_test_template.format(
                    op_name=op_name, lf=lf, rt=rt,
                    includes=includes, decls=decls,
                    rand_statement="(log_t)(rand() % 2);".format(lf=lf, rt=rt),
                    typ="log_t", test_statement=s1, l="l", term="l",
                    aliases=arithmetic_aliases.format(lf=lf, rt=rt))
                filename = get_filename(opts, op_name + "l", lf, rt)
                with common.open_utf8(filename) as fp:
                    fp.write(content_src)
                common.clang_format(opts, filename)
