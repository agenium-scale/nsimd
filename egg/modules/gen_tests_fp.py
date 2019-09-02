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
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <nsimd/nsimd.h>
#include <nsimd/modules/fixed_point.hpp>
"""

arithmetic_aliases = """\
typedef nsimd::fixed_point::fp_t<{lf}, {rt}> fp_t;
typedef nsimd::fixed_point::pack<fp_t> vec_t;
typedef nsimd::fixed_point::packl<fp_t> vecl_t;
typedef nsimd::fixed_point::pack<fp_t>::value_type raw_t;
typedef nsimd::fixed_point::packl<fp_t>::value_type log_t;
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
static double __get_numeric_precision() {
  return ldexpf(1.0, -(int)rt);
}

"""

comparison_fp = """\
template <uint8_t lf, uint8_t rt>
bool __compare_values(nsimd::fixed_point::fp_t<lf, rt> val, double ref)
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
nsimd::fixed_point::fp_t<lf, rt> __gen_random_val() {{
  const double max_val = ldexp(1.0, (lf - 1) / 2) - 1;
  const double min_val = -ldexp(1.0, (lf - 1) / 2);
  const int n_vals = (int)ldexp(1.0, (rt - 1) / 2);
  
  nsimd::fixed_point::fp_t<lf, rt> res = 0;
  const double integral =
      roundf(((double)rand() / (double)RAND_MAX) * (max_val - min_val) + min_val);
  const double decimal =
      ((double)(rand() % n_vals) + ldexp(1.0, rt / 2))
      * __get_numeric_precision<lf, rt>();
  // Ensure abs(val) > 1
  double val = integral + decimal;
  if(abs(val) < 1.0)
  {{
    val += val > 0.0 ? 1.0 : -1.0;
  }}
  res = nsimd::fixed_point::fp_t<lf, rt>(val);
  return res;
}}

"""

# ------------------------------------------------------------------------------
# Template for arithmetic binary operators

arithmetic_test_template = """\
{includes}
// -----------------------------------------------------------------------------

{decls}
// -----------------------------------------------------------------------------

int main() {{
  typedef nsimd::fixed_point::fp_t<{lf}, {rt}> fp_t;
  typedef nsimd::fixed_point::pack<fp_t> vec_t;
  const size_t v_size = nsimd::fixed_point::len(fp_t());

  srand(time(NULL));

  // FP vectors
  fp_t *tab0_fp = (fp_t *) malloc(v_size * sizeof(fp_t));
  fp_t *tab1_fp = (fp_t *) malloc(v_size * sizeof(fp_t));
  fp_t *res_fp  = (fp_t *) malloc(v_size * sizeof(fp_t));

  // Floating point equivalent
  double *tab0_f = (double *) malloc(v_size * sizeof(double));
  double *tab1_f = (double *) malloc(v_size * sizeof(double));
  double *res_f  = (double *) malloc(v_size * sizeof(double));

  for (size_t i = 0; i < v_size; i++) {{
    tab0_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab1_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab0_f[i] = nsimd::fixed_point::fixed2float(tab0_fp[i]);
    tab1_f[i] = nsimd::fixed_point::fixed2float(tab1_fp[i]);
  }}

  vec_t v0_fp = nsimd::fixed_point::loadu<vec_t>(tab0_fp);
  vec_t v1_fp = nsimd::fixed_point::loadu<vec_t>(tab1_fp);
  vec_t vres_fp = nsimd::fixed_point::{op_name}(v0_fp, v1_fp);
  nsimd::fixed_point::storeu(res_fp, vres_fp);

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

arithmetic_ops = [("add", "+"), ("sub", "-"), ("mul", "*"), ("div","/")]
def gen_arithmetic_ops_tests(lf, rt, opts):
    for op_name, op_val in arithmetic_ops:
        decls = check + limits + comparison_fp + gen_random_val
        content_src = arithmetic_test_template.format(
            op_name=op_name, op_val=op_val, lf=lf, rt=rt,
            includes=includes, decls=decls)
        filename = get_filename(opts, op_name, lf, rt)
        with common.open_utf8(filename) as fp:
            fp.write(content_src)
        common.clang_format(opts, filename)

# ------------------------------------------------------------------------------
# Ternary ops (FMA and co)

ternary_ops_template = """\
{includes}
// -----------------------------------------------------------------------------

{decls}
// -----------------------------------------------------------------------------

int main() {{
  typedef nsimd::fixed_point::fp_t<{lf}, {rt}> fp_t;
  typedef nsimd::fixed_point::pack<fp_t> vec_t;
  const size_t v_size = nsimd::fixed_point::len(fp_t());
 
  srand(time(NULL));

  // FP vectors
  fp_t *tab0_fp = (fp_t *) malloc(v_size * sizeof(fp_t));
  fp_t *tab1_fp = (fp_t *) malloc(v_size * sizeof(fp_t));
  fp_t *tab2_fp = (fp_t *) malloc(v_size * sizeof(fp_t));
  fp_t *res_fp  = (fp_t *) malloc(v_size * sizeof(fp_t));

  // Floating point equivalent
  double *tab0_f = (double *) malloc(v_size * sizeof(double));;
  double *tab1_f = (double *) malloc(v_size * sizeof(double));;
  double *tab2_f = (double *) malloc(v_size * sizeof(double));;
  double *res_f  = (double *) malloc(v_size * sizeof(double));;

  for (size_t i = 0; i < v_size; i++) {{
    tab0_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab1_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab2_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab0_f[i] = nsimd::fixed_point::fixed2float(tab0_fp[i]);
    tab1_f[i] = nsimd::fixed_point::fixed2float(tab1_fp[i]);
    tab2_f[i] = nsimd::fixed_point::fixed2float(tab2_fp[i]);
  }}

  vec_t v0_fp = nsimd::fixed_point::loadu<vec_t>(tab0_fp);
  vec_t v1_fp = nsimd::fixed_point::loadu<vec_t>(tab1_fp);
  vec_t v2_fp = nsimd::fixed_point::loadu<vec_t>(tab2_fp);
  vec_t vres_fp = nsimd::fixed_point::{op_name}(v0_fp, v1_fp, v2_fp);
  nsimd::fixed_point::storeu(res_fp, vres_fp);

  for(size_t i = 0; i < v_size; i++) {{
    const double a = tab0_f[i];
    const double b = tab1_f[i];
    const double c = tab2_f[i];     
    
    {check_statement}
  }}

  for(size_t i = 0; i < v_size; i++) {{
    CHECK(__compare_values(res_fp[i], res_f[i]));
  }}
 
  fprintf(stdout, \"Test of {op_name} for fp_t<{lf},{rt}>... OK\\n\");
  return EXIT_SUCCESS;
}}
"""

ternary_ops = [("fma", "res_f[i] = a + b * c;")]
def gen_ternary_ops_tests(lf, rt, opts):
    for op_name, statement in ternary_ops:
        decls = check + limits + comparison_fp + gen_random_val
        content_src = ternary_ops_template.format(
            op_name=op_name, check_statement=statement.format(lf=lf, rt=rt),
            lf=lf, rt=rt,includes=includes, decls=decls)
        filename = get_filename(opts, op_name, lf, rt)
        with common.open_utf8(filename) as fp:
            fp.write(content_src)
        common.clang_format(opts, filename)

# ------------------------------------------------------------------------------
# Template for math operators

math_test_template = """\
{includes}
// -----------------------------------------------------------------------------

{decls}

// Rec operator on floating points (avoid to write a particular test for rec)
static inline double rec(const double x){{return 1.0 / x;}}
// -----------------------------------------------------------------------------

int main() {{
  typedef nsimd::fixed_point::fp_t<{lf}, {rt}> fp_t;
  typedef nsimd::fixed_point::pack<fp_t> vec_t;
  const size_t v_size = nsimd::fixed_point::len(fp_t());

  srand(time(NULL));

  // FP vectors
  fp_t *tab0_fp= (fp_t *) malloc(v_size * sizeof(fp_t));
  fp_t *res_fp = (fp_t *) malloc(v_size * sizeof(fp_t));

  // Floating point equivalent
  double *tab0_f = (double *) malloc(v_size * sizeof(double));
  double *res_f  = (double *) malloc(v_size * sizeof(double));

  for (size_t i = 0; i < v_size; i++) {{
    tab0_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab0_f[i] = nsimd::fixed_point::fixed2float(tab0_fp[i]);
  }}

  vec_t v0_fp = nsimd::fixed_point::loadu<vec_t>(tab0_fp);
  vec_t vres_fp = nsimd::fixed_point::{op_name}(v0_fp);
  nsimd::fixed_point::storeu(res_fp, vres_fp);

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

math_ops = ["rec"]
def gen_math_functions_tests(lf, rt, opts):
    for op_name in math_ops:
        decls = check + limits + comparison_fp + gen_random_val
        content_src = math_test_template.format(
            op_name=op_name, lf=lf, rt=rt,
            includes=includes, decls=decls)
        filename = get_filename(opts, op_name, lf, rt)
        with common.open_utf8(filename) as fp:
            fp.write(content_src)
        common.clang_format(opts, filename)

# ------------------------------------------------------------------------------
# Comparison operators

comparison_test_template = """\
{includes}
// -----------------------------------------------------------------------------

{decls}
// -----------------------------------------------------------------------------

int main(){{
  typedef nsimd::fixed_point::fp_t<{lf}, {rt}> fp_t;
  typedef nsimd::fixed_point::pack<fp_t> vec_t;
  typedef nsimd::fixed_point::packl<fp_t> vecl_t;
  typedef nsimd::fixed_point::packl<fp_t>::value_type log_t;
  const size_t v_size = nsimd::fixed_point::len(fp_t());
  
  srand(time(NULL));

  // FP vectors
  fp_t *tab0_fp = (fp_t *) malloc(v_size * sizeof(fp_t));
  fp_t *tab1_fp = (fp_t *) malloc(v_size * sizeof(fp_t));
  log_t *resl_fp = (log_t *) malloc(v_size * sizeof(log_t));

  for(size_t i = 0; i < v_size; i++) {{
    tab0_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab1_fp[i] = __gen_random_val<{lf}, {rt}>();
  }}
  // Be sure there is at least one equality to test all the cases.
  tab0_fp[0] = tab1_fp[0];
  
  vec_t v0_fp = nsimd::fixed_point::loadu<vec_t>(tab0_fp);
  vec_t v1_fp = nsimd::fixed_point::loadu<vec_t>(tab1_fp);
  vecl_t vres_fp = nsimd::fixed_point::{op_name}(v0_fp, v1_fp);
  nsimd::fixed_point::storelu(resl_fp, vres_fp);

  for(size_t i = 0; i < v_size; i++) {{
    CHECK((__check_logical_val<log_t, {lf}, {rt}>(
        resl_fp[i], tab0_fp[i], tab1_fp[i])));
  }}

  fprintf(stdout, \"Test of {op_name} for fp_t<{lf},{rt}>... OK\\n\");
  return EXIT_SUCCESS;
}}
"""

comparison_ops = [("eq","=="), ("ne","!="), ("le","<="), ("lt","<"),
                  ("ge",">="), ("gt",">")]

def gen_comparison_tests(lf, rt, opts):
    for op_name, op_val in comparison_ops:
        decls = check + limits + comparison_log.format(op_val=op_val) + gen_random_val  
        content_src = comparison_test_template.format(
            op_name=op_name, op_val=op_val, lf=lf, rt=rt,
            includes=includes, decls=decls)
        filename = get_filename(opts, op_name, lf, rt)
        with common.open_utf8(filename) as fp:
            fp.write(content_src)
        common.clang_format(opts, filename)

# ------------------------------------------------------------------------------
# Bitwise binary operators

bitwise_binary_test_template = """\
{includes}
#include <limits>

// -----------------------------------------------------------------------------

{decls}
// -----------------------------------------------------------------------------

int main() {{
  typedef nsimd::fixed_point::fp_t<{lf}, {rt}> fp_t;
  typedef nsimd::fixed_point::pack{l}<fp_t> vec{l}_t;
  typedef nsimd::fixed_point::pack{l}<fp_t>::value_type raw_t;
  const size_t v_size = nsimd::fixed_point::len(fp_t());

  srand(time(NULL));
  
  raw_t *tab0 = (raw_t *) malloc(v_size * sizeof(raw_t));
  raw_t *tab1 = (raw_t *) malloc(v_size * sizeof(raw_t));
  raw_t *res  = (raw_t *) malloc(v_size * sizeof(raw_t));
 
  for(size_t i = 0; i < v_size; i++)
  {{
    tab0[i] = {rand_statement}
    tab1[i] = {rand_statement}
  }}
  // Be sure there is at least one equality to test all the cases.
  tab0[0] = tab1[0];

  vec{l}_t v0 = nsimd::fixed_point::load{l}u<vec{l}_t>(tab0);
  vec{l}_t v1 = nsimd::fixed_point::load{l}u<vec{l}_t>(tab1);
  vec{l}_t v_res = nsimd::fixed_point::{op_name}{term}(v0, v1);
  nsimd::fixed_point::store{l}u(res, v_res);

  for(size_t i = 0; i < v_size; i++)
  {{
    raw_t a = tab0[i];
    raw_t b = tab1[i];
    raw_t c = res[i];
    CHECK({test_statement});
  }}  
 
  fprintf(stdout, \"Test of {op_name}{term} for fp_t<{lf},{rt}>... OK\\n\");
  return EXIT_SUCCESS;
}}
"""

bitwise_binary_ops = [("and", "c._raw == (a._raw & b._raw)", "c == (a & b)"),
                      ("andnot", "c._raw == (a._raw & ~b._raw)", "c == (a & ~b)"),
                      ("or", "c._raw == (a._raw | b._raw)", "c == (a | b)"),
                      ("xor","c._raw == ((~a._raw & b._raw) | (a._raw & ~b._raw))",
                       "c == ((~a & b) | (a & ~b))")]
def gen_bitwise_ops_tests(lf, rt, opts):
    for op_name, s0, s1 in bitwise_binary_ops:
        # {op}b
        decls = check + limits + gen_random_val
        content_src = bitwise_binary_test_template.format(
            op_name=op_name, lf=lf, rt=rt,
            includes=includes, decls=decls,
            rand_statement="__gen_random_val<{lf}, {rt}>();".format(lf=lf, rt=rt),
            test_statement=s0, l="", term="b")
        filename = get_filename(opts, op_name + "b", lf, rt)
        with common.open_utf8(filename) as fp:
            fp.write(content_src)
        common.clang_format(opts, filename)
        
        # {op}l
        content_src = bitwise_binary_test_template.format(
            op_name=op_name, lf=lf, rt=rt,
            includes=includes, decls=decls,
            rand_statement="(raw_t)(rand() % 2);".format(lf=lf, rt=rt),
            test_statement=s1, l="l", term="l")
        filename = get_filename(opts, op_name + "l", lf, rt)
        with common.open_utf8(filename) as fp:
            fp.write(content_src)
        common.clang_format(opts, filename)

# ------------------------------------------------------------------------------
# Bitwise unary operators

bitwise_unary_test_template = """\
{includes}

// -----------------------------------------------------------------------------

{decls}
// -----------------------------------------------------------------------------

int main() {{
  typedef nsimd::fixed_point::fp_t<{lf}, {rt}> fp_t;
  typedef nsimd::fixed_point::pack{l}<fp_t> vec{l}_t;
  typedef nsimd::fixed_point::pack{l}<fp_t>::value_type raw_t;
  const size_t v_size = nsimd::fixed_point::len(fp_t());
  
  srand(time(NULL));
  
  raw_t *tab0 = (raw_t *) malloc(v_size * sizeof(raw_t));;
  raw_t *res  = (raw_t *) malloc(v_size * sizeof(raw_t));;

  for(size_t i = 0; i < v_size; i++)
  {{
    tab0[i] = {rand_statement}
  }}

  vec{l}_t v0 = nsimd::fixed_point::load{l}u<vec{l}_t>(tab0);
  vec{l}_t v_res = nsimd::fixed_point::{op_name}{term}(v0);
  nsimd::fixed_point::store{l}u(res, v_res);

  for(size_t i = 0; i < v_size; i++)
  {{
    raw_t a = tab0[i];
    raw_t b = res[i];
    CHECK({test_statement});
  }}  
 
  fprintf(stdout, \"Test of {op_name}{term} for fp_t<{lf},{rt}>... OK\\n\");
  return EXIT_SUCCESS;
}}
"""

bitwise_unary_ops = [("not", "b._raw == ~a._raw",
                      "((b == 0) && (a == 1)) | ((b == 1) && (a == 0))")]
def gen_unary_ops_tests(lf, rt, opts):
    for op_name, s0, s1 in bitwise_unary_ops:
        decls = check + limits + gen_random_val  
        # {op}b
        content_src = bitwise_unary_test_template.format(
            op_name=op_name, lf=lf, rt=rt,
            includes=includes, decls=decls,
            rand_statement="__gen_random_val<{lf}, {rt}>();".format(lf=lf, rt=rt),
            test_statement=s0, l="", term="b")
        filename = get_filename(opts, op_name + "b", lf, rt)
        with common.open_utf8(filename) as fp:
            fp.write(content_src)
        common.clang_format(opts, filename)

        # {op}l
        content_src = bitwise_unary_test_template.format(
            op_name=op_name, lf=lf, rt=rt,
            includes=includes, decls=decls,
            rand_statement="(raw_t)(rand() % 2);".format(lf=lf, rt=rt),
            test_statement=s1, l="l", term="l")
        filename = get_filename(opts, op_name + "l", lf, rt)
        with common.open_utf8(filename) as fp:
            fp.write(content_src)
        common.clang_format(opts, filename)
        
# -----------------------------------------------------------------------------
# if_else

if_else_test_template = """\
{includes}

// -----------------------------------------------------------------------------

{decls}
// -----------------------------------------------------------------------------

int main() {{
  typedef nsimd::fixed_point::fp_t<{lf}, {rt}> fp_t;
  typedef nsimd::fixed_point::pack<fp_t> vec_t;
  typedef nsimd::fixed_point::packl<fp_t> vecl_t;
  typedef nsimd::fixed_point::packl<fp_t>::value_type log_t;
  const size_t v_size = nsimd::fixed_point::len(fp_t());
  
  srand(time(NULL));
  
  fp_t *tab0_fp = (fp_t *) malloc(v_size * sizeof(fp_t));
  fp_t *tab1_fp = (fp_t *) malloc(v_size * sizeof(fp_t));
  fp_t *res_fp  = (fp_t *) malloc(v_size * sizeof(fp_t));
  log_t *mask = (log_t *) malloc(v_size * sizeof(log_t));

  for(size_t i = 0; i < v_size; i++) {{
    tab0_fp[i] = __gen_random_val<{lf}, {rt}>();
    tab1_fp[i] = __gen_random_val<{lf}, {rt}>();
    mask[i] = (log_t) (rand() % 2);
  }}

  vec_t v0 = nsimd::fixed_point::loadu<vec_t>(tab0_fp);
  vec_t v1 = nsimd::fixed_point::loadu<vec_t>(tab1_fp);
  vecl_t vl = nsimd::fixed_point::loadlu<vecl_t>(mask);
  vec_t v_res = nsimd::fixed_point::if_else1(vl, v0, v1);
  nsimd::fixed_point::storeu(res_fp, v_res);

  for(size_t i = 0; i < v_size; i++)
  {{
    fp_t ref = mask[i] ? tab0_fp[i] : tab1_fp[i];
    CHECK(ref._raw == res_fp[i]._raw);
  }}
 
  fprintf(stdout, \"Test of if_else1 for fp_t<{lf},{rt}>... OK\\n\");
  return EXIT_SUCCESS;
}}
"""

def gen_if_else_tests(lf, rt, opts):
    decls = check + limits + comparison_fp + gen_random_val
    content_src = if_else_test_template.format(
        lf=lf, rt=rt, includes=includes, decls=decls)
    filename = get_filename(opts, "if_else", lf, rt)
    with common.open_utf8(filename) as fp:
        fp.write(content_src)
    common.clang_format(opts, filename)
# -------------------------------------------------------------------------------

load_ops = ["loadu", "loadlu", "loada", "loadla"]
store_ops = ["storeu", "storelu", "storea", "storela"]

# -------------------------------------------------------------------------------
# Entry point

lf_vals = ["4", "8", "16"]
rt_vals = ["1", "2", "3", "4", "5", "6", "7", "8"]
def doit(opts):
    print ('-- Generating tests for module fixed_point')
    for lf in lf_vals:
        for rt in rt_vals:
            ## Arithmetic operators
            gen_arithmetic_ops_tests(lf, rt, opts)

            ## Ternary_operators
            gen_ternary_ops_tests(lf, rt, opts)
    
            ## Math functions
            gen_math_functions_tests(lf, rt, opts) 
            
            ## Comparison operators
            gen_comparison_tests(lf, rt, opts)

            ## Bitwise binary operators
            gen_bitwise_ops_tests(lf, rt, opts)
            
            ## Bitwise unary operators
            gen_unary_ops_tests(lf, rt, opts)

            ## If_else
            gen_if_else_tests(lf, rt, opts)
