# Use utf-8 encoding
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Agenium Scale
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

from src_templates import *

## -----------------------------------------------------------------------------

c_src_template = '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <nsimd/nsimd.h>
#include <nsimd/modules/convolution/nsimd_convolution.h>

/* -------------------------------------------------------------------------- */

/* Scalar reference implementation */
static void nsimd_conv_compute_ref_{typ}(
    {typ} *input_img, {typ} *output_img, {typ} *kernel, const size_t c_in,
    const size_t h_in, const size_t w_in, const size_t c_out,
    const size_t h_out, const size_t w_out, const size_t k_h, const size_t k_w,
    const size_t stride_h, const size_t stride_w)
{{
  memset(output_img, 0, c_out * h_out * w_out * sizeof({typ}));
 
  for(size_t oc = 0; oc < c_out; oc++)
  {{
    {typ} *out_ptr = output_img + oc * h_out * w_out;
    for(size_t ic = 0; ic < c_in; ic++)
    {{
      {typ} *in_ptr = input_img + ic * h_in * w_in;
      {typ} *kernel_ptr = kernel + oc * c_in * k_h * k_w + ic * k_h * k_w;
      for(size_t i = 0; i < h_out; i++)
      {{
        for(size_t j = 0; j < w_out; j++)
        {{
          {typ} val = ({typ})0;
          for(size_t s = 0; s < k_h; s++)
          {{
            for(size_t t = 0; t < k_w; t++)
            {{
              val += kernel_ptr[s * k_w + t] 
                * in_ptr[(stride_h * i + s) * w_in + stride_w * j + t];
            }}
          }}
          out_ptr[i * w_out +j] += val;
        }}
      }}
    }}
  }}
}}

{relative_distance}

/* -------------------------------------------------------------------------- */

int main()
{{
  size_t c_in =  {c_in};
  size_t h_in = {h_in};
  size_t w_in = {w_in};
  size_t c_out = {c_out};
  size_t h_out = {h_out};
  size_t w_out = {w_out};
  size_t k_h = {k_h};
  size_t k_w = {k_w};
  size_t s_h = {s_h};
  size_t s_w = {s_w};

  size_t p_h = h_in + ((k_h >> 1) << 1);
  size_t p_w = w_in + ((k_w >> 1) << 1);

  fprintf(stdout, \"Running tests for {name} with {typ}...\\n\");

  {typ} *kernel = ({typ} *)malloc(c_in * c_out * k_h * k_w * sizeof({typ}));

  {typ} *input_ref = ({typ} *)malloc(c_in * p_h * p_w * sizeof({typ}));
  {typ} *output_ref = ({typ} *)malloc(c_out * h_out * w_out * sizeof({typ}));

  {typ} *input_comp = ({typ} *)malloc(c_in * p_h * p_w * sizeof({typ}));
  {typ} *output_comp = ({typ} *)malloc(c_out * h_out * w_out * sizeof({typ}));

  /* Kernel initialization */
  for(size_t i = 0; i < c_in * c_out * k_h * k_w; i++)
  {{
    kernel[i] = {rand};
  }}

  /* Images initialization. The image is supposed to have a zero border, but
  for the tests, we don't care. */
  for(size_t i = 0; i < c_in * p_h * p_w; i++)
  {{
    {typ} val = {rand};
    input_ref[i] = val;
    input_comp[i] = val;
  }}

  /* Run tests */
  nsimd_conv_compute_ref_{typ}(
    input_ref, output_ref, kernel, c_in, p_h, p_w, c_out,
    h_out, w_out, k_h, k_w, s_h, s_w);

  nsimd_conv_compute_{typ}(
    input_comp, output_comp, kernel, c_in, p_h, p_w, c_out,
    h_out, w_out, k_h, k_w, s_h, s_w);

  /* Compare results */
  double cmp;
  for(size_t i = 0; i < c_out * h_out * w_out; i++)
  {{
    {test}
  }}

  fprintf(stdout, \"Tests with {name} : OK\\n\");

  return EXIT_SUCCESS;
}}
'''

get_2th_power = '''\
/* One could use ldexp, but it is not available in C89. Plus we really
   don't care about performences here. */
double get_2th_power(int a) {
  double ret = 1.0;
  if (a == 0) {
    return ret;
  }
  if (a < 0) {
    int i;
    for (i = 0; i < (-a); i++) {
      ret /= 2.0;
    }
    return ret;
  }
  /* a > 0 */ {
    int i;
    for (i = 0; i < a; i++) {
      ret *= 2.0;
    }
    return ret;
  }
}

'''

relative_distance_c = '''\
double relative_distance(double a, double b) {
  double ma, mi;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
  if (isnan(a) && isnan(b)) {
    return 0.0;
  }

  if (isnan(a) || isnan(b)) {
    return -1.;
  }

  if (isinf(a) * isinf(b) > 0) {
    return 0.0;
  }

  if (isinf(a) || isinf(b)) {
    return -1.;
  }
#pragma GCC diagnostic pop

  a = (a > 0.0 ? a : -a);
  b = (b > 0.0 ? b : -b);
  ma = (a > b ? a : b);
  mi = (a < b ? a : b);

  if (ma == 0.0) {
    return 0.0;
  }

  return (ma - mi) / ma;
}

/* -------------------------------------------------------------------------- */
''' + get_2th_power

## -----------------------------------------------------------------------------
def gen_filename(params, typ):
    return 'tests/test_{k_h}x{k_w}_{c_in}_{h_in}_{w_in}_{s}.{typ}.c'.\
        format(k_h=params['k_h'], k_w=params['k_w'], c_in=params['c_in'],
               h_in=params['h_in'], w_in=params['w_in'], s=params['s_h'],
               typ=typ)

types = ['u8', 'i8', 'u16', 'i16', 'u32', 'i32', 'u64', 'i64', 'f32', 'f64']
def get_src(typ, params):
    name = gen_filename(params, typ)
    if typ in ['f32', 'f64']:
        rand = '''\
        ({typ})(2 * (rand() % 2) - 1) * ({typ})(1 << (rand() % 4)) /
        ({typ})(1 << (rand() % 4))'''.format(typ=typ)
        if typ == 'f32':
            test = '''\
            cmp = relative_distance((double)output_ref[i],
                     (double)output_comp[i]);
            if (cmp > get_2th_power(-32)) {{
            fprintf(stdout, \"Tests failed for {name} : %f\\n\", cmp);
            return EXIT_FAILURE;}}'''.format(name=name)
        else:
            test = '''\
            cmp = relative_distance((double)output_ref[i],
                     (double)output_comp[i]);            
            if (cmp > get_2th_power(-64)) {{
            fprintf(stdout, \"Tests failed for {name} : %f\\n\", cmp);
            return EXIT_FAILURE;}}'''.format(name=name)

    elif typ == 'f16':
        rand = '''\
        nsimd_f32_to_f16((f32)(2 * (rand() % 2) - 1) *
        (f32)(1 << (rand() % 4)) / (f32)(1 << (rand() % 4)))'''
        test = '''\
        cmp = relative_distance((double) nsimd_f16_to_f32(output_ref[i]), 
                (double) nsimd_f16_to_f32(output_comp[i]));
        if (cmp > get_2th_power(-{nbits})){{
        fprintf(stdout, \"Tests failed for {name} : %f\\n\", cmp);
        return EXIT_FAILURE;
        }}'''.format(name=name)
    else:
        rand = '({})(rand() % 4)'.format(typ)
        test = '''\
        cmp = (double)(output_ref[i] - output_comp[i]);
        if(cmp != (double)0){{
        fprintf(stdout, \"Tests failed for {name} : %f\\n\", cmp);
            return EXIT_FAILURE;}}'''.format(name=name)
    return c_src_template.format(
        typ=typ, relative_distance=relative_distance_c,
        test=test, rand=rand, name=name,
        c_in=params['c_in'], h_in=params['h_in'], w_in=params['w_in'],
        c_out=params['c_out'], h_out=params['h_out'], w_out=params['w_out'],
        k_h=params['k_h'], k_w=params['k_w'],
        s_h=params['s_h'], s_w=params['s_w'])
        
if __name__ == '__main__':
    os.system('mkdir -p tests')
    for typ in types:
        for k in range(1, 5):
            for stride in range(1, 3):
                params = {
                    'c_in':139, 'c_out':257,
                    'h_in':59, 'h_out':59 // stride,
                    'w_in':61, 'w_out':61 // stride,
                    'k_h':k, 'k_w':k, 's_h':stride, 's_w':stride}
                filename = gen_filename(params, typ)
                content = get_src(typ, params)
                write_src(filename, content)

# if __name__ =='__main__':
#     print('Generating tests...')
#     os.system('mkdir -p tests')
#     typ = 'f32'
#     stride = 1
#     k = 3
#     params = {
#         'c_in':128, 'c_out':128,
#         'h_in':128, 'h_out':128 // stride,
#         'w_in':24, 'w_out':24 // stride,
#         'k_h':k, 'k_w':k, 's_h':stride, 's_w':stride}
#     filename = gen_filename(params, typ)
#     content = get_src(typ, params)
#     write_src(filename, content)
    
