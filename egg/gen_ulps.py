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

# -----------------------------------------------------------------------------
# Import section

import gen_tests
import common
import operators
import os

# -----------------------------------------------------------------------------
# Includes

includes = \
'''
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <limits.h>
#include <mpfr.h>
#include <nsimd/nsimd.h>
#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/cxx_adv_api_functions.hpp>
#include <stdint.h>
'''

# -----------------------------------------------------------------------------
# Random numbers generators

random_f16_generator = \
'''
uint16_t acc = 0;
for (i = 0; i < SIZE; i++) {
  memcpy(&vin1[i], &acc, sizeof(uint16_t));
  ++acc;
}
'''

random_f32_generator = \
'''
uint32_t acc = 0;
for (i = 0; i < SIZE; i++) {
  memcpy(&vin1[i], &acc, sizeof(uint32_t));
  acc+=(uint32_t)((rand()%(16*16))+1);
}
'''

random_f64_generator = \
'''
for (i = 0; i < SIZE/2; ++i) {
  double num = 0.;

  uint64_t *inum = reinterpret_cast<uint64_t*>(&num);
  for (uint64_t j=0; j<64; ++j) {
      uint64_t tmp = ((uint64_t)(rand())%2u) << j;
      *inum = *inum | tmp;
  }

  vin1[i] = num;
}

for (; i<SIZE; ++i) {
  double num = (double)(2 * (rand() % 2) - 1) * 8.0 * (double)rand() /
                        (double)RAND_MAX;
  vin1[i] = num;
}

/* Ensure that 0 is tested */
vin1[0] = 0.;
'''

# -----------------------------------------------------------------------------
# Main template

code = \
'''
int main(void) {{
  int i, step;
  {typ} *vout0, *vout1;
  {typ} *vin1;

  step = vlen({typ});

  nat SIZE = {SIZE};

  vin1 = ({typ} *)nsimd_aligned_alloc(SIZE * (int)sizeof({typ}));
  if (!vin1) {{
    fprintf(stderr, "vin - malloc failed\\n");
    exit(1);
  }}

  vout0 = ({typ} *)nsimd_aligned_alloc(SIZE * (int)sizeof({typ}));
  if (!vout0) {{
    fprintf(stderr, "vout0 - malloc failed\\n");
    exit(1);
  }}
  vout1 = ({typ} *)nsimd_aligned_alloc(SIZE * (int)sizeof({typ}));
  if (!vout1) {{
    fprintf(stderr, "vout1 - malloc failed\\n");
    exit(1);
  }}

  /* Fill input vector(s) with random numbers */
  {random_generator}

  /* Fill vector 0 with mpfr values */
  for (i = 0; i < SIZE; i++) {{
    mpfr_t c, a1;
    mpfr_init2(c, 64);
    mpfr_init2(a1, 64);
    mpfr_set_{mpfr_suffix}(a1, {convert_from_type}(vin1[i]), MPFR_RNDN);
    {mpfr_func}(c, a1 {mpfr_rnd});
    vout0[i] = {convert_to_type}(mpfr_get_{mpfr_suffix}(c, MPFR_RNDN));
    mpfr_clear(c);
    mpfr_clear(a1);
  }}

  /* Fill vector 1 with nsimd values */
  for (i = 0; i < SIZE; i += step) {{
    nsimd::pack<{typ}> va1, vc;
    va1 = nsimd::loadu<nsimd::pack<{typ}> >(&vin1[i]);
    vc = nsimd::{nsimd_func}(va1);
    nsimd::storeu(&vout1[i], vc);
  }}

  int ulp = -{mantisse};
  double worst_rel = 0.;
  int worst_value_index = 0;

  int ulp_dnz = -{mantisse};
  double worst_rel_dnz = 0.;
  int64_t worst_value_dnz_index = 0;

  int inf_error = false;
  int64_t inf_error_index = 0;

  int nan_error = false;
  int64_t nan_error_index = 0;

  /* Compare results */
  for (i = 0; i < SIZE; ++i) {{
      double rel = relative_distance((double){convert_from_type}(vout0[i]),
                                     (double){convert_from_type}(vout1[i]));

      uint64_t hex_in = 0;
      memcpy(&hex_in, &vin1[i], sizeof(uint32_t));

      {typ} mpfr_out = vout0[i];
      {typ} nsimd_out = vout1[i];

      if (std::fpclassify({convert_from_type}(mpfr_out)) == FP_SUBNORMAL) {{
        // Result should be a subnormal float
        if (std::fpclassify({convert_from_type}(nsimd_out)) == FP_SUBNORMAL) {{
          if (rel > worst_rel_dnz) {{
            worst_rel_dnz = rel;
            worst_value_dnz_index = i;
            ulp_dnz = (int) log2(rel);
          }}
        }} else if (std::fpclassify({convert_from_type}(nsimd_out)) == FP_ZERO) {{
            worst_rel_dnz = DBL_MAX;
            worst_value_dnz_index = i;
            ulp_dnz = 1;
        }}
      }}
      else if (rel < 0) {{
        #ifdef DEBUG
        printf("IN: %e 0x%lx\\t", (double){convert_from_type}(vin1[i]), hex_in);
        printf("OUT: %e %e\\n", (double){convert_from_type}(vout0[i]),
                                (double){convert_from_type}(vout1[i]));
        #endif

        if (std::fpclassify({convert_from_type}(mpfr_out)) == FP_NAN) {{
            nan_error = true;
            nan_error_index = i;
        }} else {{
            inf_error = true;
            inf_error_index = i;
        }}

        worst_rel = DBL_MAX;
      }} else if (rel > worst_rel) {{
      #ifdef DEBUG
        printf("IN: %e 0x%lx\\t", (double){convert_from_type}(vin1[i]), hex_in);
        printf("OUT: %e %e\\n", (double){convert_from_type}(vout0[i]),
                                (double){convert_from_type}(vout1[i]));
      #endif
        ulp = (int) log2(rel);
        worst_rel = rel;
        worst_value_index = i;
      }}
  }}

  ulp = std::min(-ulp, {mantisse});
  ulp_dnz = std::min(-ulp_dnz, {mantisse});

  uint64_t worst_value = 0, nan_value, inf_value, worst_value_dnz;
  memcpy(&worst_value, &vin1[worst_value_index], sizeof({typ}));
  memcpy(&nan_value, &vin1[nan_error_index], sizeof({typ}));
  memcpy(&inf_value, &vin1[inf_error_index], sizeof({typ}));
  memcpy(&worst_value_dnz, &vin1[worst_value_dnz_index], sizeof({typ}));

  fprintf(stdout, "{{\\n\\t"
                  "\\"func\\":\\"{nsimd_func}\\", "
                  "\\"type\\":\\"{typ}\\",\\n\\t"
                  "\\"ulps\\" : \\"%d\\", "
                  "\\"Worst value\\": \\"0x%lx\\",\\n\\t"
                  "\\"ulps for denormalized output\\" : \\"%d\\", "
                  "\\"Worst value for dnz output\\" : \\"0x%lx\\",\\n\\t"
                  "\\"NaN Error\\":\\"%s\\", "
                  "\\"Value causing NaN\\":\\"0x%lx\\",\\n\\t"
                  "\\"Inf Error\\":\\"%s\\", "
                  "\\"Value causing Inf error\\":\\"0x%lx\\"\\n"
                  "}}",
                  ulp,
                  worst_value,
                  ulp_dnz,
                  worst_value_dnz,
                  nan_error?"true":"false",
                  nan_value,
                  inf_error?"true":"false",
                  inf_value);
  fflush(stdout);

  free(vin1);
  free(vout0);
  free(vout1);

  return 0;
}}
'''

# -----------------------------------------------------------------------------
# Entry point

# TODO: redo a second pass after swaping numbers around
# (to avoid vector filled with similar numbers)

def doit(opts):
    print ('-- Generating ulps')
    common.mkdir_p(opts.ulps_dir)
    for op_name, operator in operators.operators.items():
        if not operator.tests_mpfr:
            continue
        if op_name in ['gammaln', 'lgamma', 'pow']:
            continue

        mpfr_func = operator.tests_mpfr_name()
        mpfr_rnd = ", MPFR_RNDN"

        for typ in common.ftypes:
            if typ == 'f16':
                random_generator = random_f16_generator
                convert_to_type = "nsimd_f32_to_f16"
                convert_from_type = "nsimd_f16_to_f32"
                mantisse=10
                size = 0xffff
                mpfr_suffix = "flt"
            elif typ == 'f32':
                convert_to_type = "(f32)"
                convert_from_type = ""
                random_generator = random_f32_generator
                mantisse=23
                #size = 0xffffffff
                size = 0x00ffffff
                mpfr_suffix = "flt"
            elif typ == 'f64':
                convert_to_type = "(f64)"
                convert_from_type = ""
                random_generator = random_f64_generator
                mantisse = 52
                size = 0x00ffffff
                mpfr_suffix = "d"
            else:
                raise Exception('Unsupported type "{}"'.format(typ))

            filename = os.path.join(opts.ulps_dir, '{}_{}_{}.cpp'. \
                       format(op_name, "ulp", typ));

            if not common.can_create_filename(opts, filename):
                continue

            with common.open_utf8(filename) as out:
                out.write(includes)
                out.write(gen_tests.relative_distance_cpp)
                out.write(code.format(
                    typ = typ,
                    nsimd_func = op_name,
                    mpfr_func = mpfr_func,
                    mpfr_rnd = mpfr_rnd,
                    random_generator = random_generator,
                    convert_from_type = convert_from_type,
                    convert_to_type = convert_to_type,
                    mantisse = mantisse,
                    SIZE=size,
                    mpfr_suffix=mpfr_suffix))

            common.clang_format(opts, filename)
