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

## -----------------------------------------------------------------------------

licence = '''/*

Copyright (c) 2020 Agenium Scale

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
'''

def write_src(filename, content):
    with open(filename, 'w') as fp:
        fp.write(licence + '\n' + content)
    os.system('clang-format -style=file -i {}'.format(filename))

# -----------------------------------------------------------------------------
# Directories locations

# Use it to generate files
conv_include_path = 'include/nsimd/modules/convolution'
conv_src_path = 'src/modules/convolution'

# Use it to include files
conv_include_head = 'nsimd/modules/convolution'

def get_head_filename(filename):
    return os.path.join(conv_include_head, filename)

def get_head_include(filename):
    return '#include \"{}\"\n'.format(get_head_filename(filename))

def get_include_filename(filename):
    return os.path.join(conv_include_path, filename)

def get_src_filename(filename):
    return os.path.join(conv_src_path, filename)

# -----------------------------------------------------------------------------

# Scalar default implementation : use it by default and for the tests.
scalar_conv_decl = '''\
static void nsimd_conv_compute_scalar_{typ}(
    {typ} *input_img, {typ} *output_img, {typ} *kernel, const size_t c_in,
    const size_t h_in, const size_t w_in, const size_t c_out,
    const size_t h_out, const size_t w_out, const size_t k_h, const size_t k_w,
    const size_t stride_h, const size_t stride_w)'''

scalar_conv_src = '''\
static void nsimd_conv_compute_scalar_{typ}(
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
'''

# Cpp templated scalar convolution
scalar_generic_conv_src = '''\
template <typename T>
void convolution_compute(
    T *input_img, T *output_img, T *kernel, const size_t c_in,
    const size_t h_in, const size_t w_in, const size_t c_out,
    const size_t h_out, const size_t w_out, const size_t k_h, const size_t k_w,
    const size_t stride_h, const size_t stride_w)
{{
  memset(output_img, 0, c_out * h_out * w_out * sizeof(T));
 
  for(size_t oc = 0; oc < c_out; oc++)
  {{
    T *out_ptr = output_img + oc * h_out * w_out;
    for(size_t ic = 0; ic < c_in; ic++)
    {{
      T *in_ptr = input_img + ic * h_in * w_in;
      T *kernel_ptr = kernel + oc * c_in * k_h * k_w + ic * k_h * k_w;
      for(size_t i = 0; i < h_out; i++)
      {{
        for(size_t j = 0; j < w_out; j++)
        {{
          T val = T(0);
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
'''

# -----------------------------------------------------------------------------
# Conv kernels

kernel_signature = '''\
static void _nsimd_conv_add_dot_{typ}_{k_h}x{k_w}_{regs_w}x{regs_ch}_{stride}(
    {typ} *input, {typ} *output, {typ} *kernel, const size_t h_in,
    const size_t w_in, const size_t h_out, const size_t w_out,
    const size_t c_in)'''

kernel_generic_signature = '''\
static void _nsimd_conv_add_dot_{typ}_nxn_{regs_w}x{regs_ch}_{stride}(
    {typ} *input, {typ} *output, {typ} *kernel, const size_t h_in,
    const size_t w_in, const size_t h_out, const size_t w_out,
    const size_t c_in, const size_t k_h, const size_t k_w)'''

odd_kernel_signature = '''\
static void _nsimd_conv_add_dot_{typ}_{k_h}x{k_w}_oddx{regs_ch}_{stride}(
    {typ} *input, {typ} *output, {typ} *kernel, const size_t h_in,
    const size_t w_in, const size_t h_out, const size_t w_out,
    const size_t c_in, const size_t n)'''

odd_kernel_generic_signature = '''\
static void _nsimd_conv_add_dot_{typ}_nxn_oddx{regs_ch}_{stride}(
    {typ} *input, {typ} *output, {typ} *kernel, const size_t h_in,
    const size_t w_in, const size_t h_out, const size_t w_out,
    const size_t c_in, const size_t n, const size_t k_h, const size_t k_w)'''

kernel_def = '''\
{signature}
{{
{acc_decl}

{input_decl}
{v_typ} v_coeff;

{typ} *__restrict kernel_ptr = kernel;

{acc_load}

for(size_t n_ic = 0; n_ic < c_in; n_ic++)
{{
  {typ} *__restrict input_ptr = input + n_ic * w_in * h_in;
  {kernel_begin_loop}
  {update_block}
  {kernel_end_loop}
}} /* End of c_in loop */

{acc_store}
}}

'''

# Functions for loading odd values in SIMD vectors    
# Maybe there are more efficient ways to do this with masked load/stores.

odd_loads_s1 = '''\
NSIMD_INLINE {v_type} _vld_n_{typ}({typ} *ptr, const size_t n)
{{
  {typ} ret[vlen({typ})];
  for(size_t i = 0; i < n; i++)
  {{
    ret[i] = ptr[i];
  }}
  return vloadu(ret, {typ});
}}

NSIMD_INLINE void _vst_n_{typ}({typ} *ptr, {v_type} v0, const size_t n)
{{
  {typ} tmp[vlen({typ})];
  vstoreu(tmp, v0, {typ});
  for(size_t i = 0; i < n; i++)
  {{
    ptr[i] = tmp[i];
  }}
}}

'''

odd_loads_s2 = '''\
NSIMD_INLINE {v_type} _vld_n_{typ}({typ} *ptr, const size_t n)
{{
  {typ} ret[vlen({typ})];
  for(size_t i = 0; i < n; i++)
  {{
    ret[i] = ptr[i];
  }}
  return vloadu(ret, {typ});
}}

NSIMD_INLINE {v_type} _vld_n_{typ}_2({typ} *ptr, const size_t n)
{{
  {typ} ret[2 * vlen({typ})];
  for(size_t  i = 0; i < n; i++)
  {{
    ret[2 * i] = ptr[2 * i];
  }}
  return vload2u(ret, {typ}).v0;
}}

NSIMD_INLINE void _vst_n_{typ}({typ} *ptr, {v_type} v0, const size_t n)
{{
  {typ} tmp[vlen({typ})];
  vstoreu(tmp, v0, {typ});
  for(size_t i = 0; i < n; i++)
  {{
    ptr[i] = tmp[i];
  }}
}}
'''

# -----------------------------------------------------------------------------
# Conv generation

conv_specific_signature = '''
void nsimd_conv_{typ}_{k_h}x{k_w}_{stride}(
    {typ} *input_img, {typ} *output_img, {typ} *kernel, const size_t c_in,
    const size_t h_in, const size_t w_in, const size_t c_out,
    const size_t h_out, const size_t w_out)'''

conv_generic_signature = '''
void nsimd_conv_{typ}_nxn_{stride}(
    {typ} *input_img, {typ} *output_img, {typ} *kernel, const size_t c_in,
    const size_t h_in, const size_t w_in, const size_t c_out,
    const size_t h_out, const size_t w_out, const size_t k_h, 
    const size_t k_w)'''

conv_specific_src = '''
{signature}
{{
  {typ} *packed_kernel = ({typ} *) malloc(128 * c_out * {size} * {size} * sizeof({typ}));

  for(size_t n_ic = 0; n_ic < c_in; n_ic += 128)
  {{
    const size_t ci = NS_MIN(128, c_in - n_ic);

    /* Kernel packing */
    for(size_t n_oc = 0; n_oc < (c_out / {regs_ch}) * {regs_ch}; n_oc += {regs_ch})
    {{
      nsimd_conv_pack_kernel_{typ}_{regs_ch}(
          kernel + (n_oc * c_in + n_ic) * {size} * {size},
          packed_kernel + (n_oc / {regs_ch}) * ci * {regs_ch} * {size} * {size}, c_in, c_out, {size}, {size}, ci);
    }}

    const size_t n_oc = (c_out / {regs_ch}) * {regs_ch};
    switch(c_out - n_oc)
    {{
      {packing_case}
    }}

    for(size_t i = 0; i < h_out; i++)
    {{
      size_t j = 0;
      for(; j < (w_out / ({regs_w} * vlen({typ}))) * {regs_w} * vlen({typ}); j += {regs_w} * vlen({typ}))
      {{
        for(size_t n_oc = 0; n_oc < (c_out / {regs_ch}) * {regs_ch}; n_oc += {regs_ch})
        {{
          {typ} *__restrict in_ptr =
              input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
          {typ} *__restrict out_ptr =
              output_img + n_oc * h_out * w_out + i * w_out + j;
          {typ} *__restrict kernel_ptr =
              packed_kernel + (n_oc / {regs_ch}) * ci * {regs_ch} * {size} * {size};
          _nsimd_conv_add_dot_{typ}_{size}x{size}_{regs_w}x{regs_ch}_{stride}(
              in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci);
        }}

        const size_t n_oc = (c_out / {regs_ch}) * {regs_ch};
        {typ} *__restrict in_ptr =
            input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
        {typ} *__restrict out_ptr =
            output_img + n_oc * h_out * w_out + i * w_out + j;
        {typ} *__restrict kernel_ptr = packed_kernel
                                       + (c_out / {regs_ch}) * ci * {regs_ch} * {size} * {size}
                                       + (n_oc % {regs_ch}) * ci * {size} * {size};
        switch(c_out - n_oc)
        {{
          {case_statement0}
        }}        
      }}

      for(; j < (w_out / vlen({typ})) * vlen({typ}); j += vlen({typ}))
      {{
        for(size_t n_oc = 0; n_oc < (c_out / {regs_ch}) * {regs_ch}; n_oc += {regs_ch})
        {{
          {typ} *__restrict in_ptr =
              input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
          {typ} *__restrict out_ptr =
              output_img + n_oc * h_out * w_out + i * w_out + j;
          {typ} *__restrict kernel_ptr =
              packed_kernel + (n_oc / {regs_ch}) * ci * {regs_ch} * {size} * {size};
          _nsimd_conv_add_dot_{typ}_{size}x{size}_1x{regs_ch}_{stride}(
              in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci);
        }}

        const size_t n_oc = (c_out / {regs_ch}) * {regs_ch};
        {typ} *__restrict in_ptr =
            input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
        {typ} *__restrict out_ptr =
            output_img + n_oc * h_out * w_out + i * w_out + j;
        {typ} *__restrict kernel_ptr = packed_kernel
                                       + (c_out / {regs_ch}) * ci * {regs_ch} * {size} * {size}
                                       + (n_oc % {regs_ch}) * ci * {size} * {size};
        switch(c_out - n_oc)
        {{
          {case_statement1}
        }}
      }}

      j = (w_out / vlen({typ})) * vlen({typ});
      if(j < w_out)
      {{
        for(size_t n_oc = 0; n_oc < (c_out / {regs_ch}) * {regs_ch}; n_oc += {regs_ch})
        {{
          {typ} *__restrict in_ptr =
              input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
          {typ} *__restrict out_ptr =
              output_img + n_oc * h_out * w_out + i * w_out + j;
          {typ} *__restrict kernel_ptr =
              packed_kernel + (n_oc / {regs_ch}) * ci * {regs_ch} * {size} * {size};
          _nsimd_conv_add_dot_{typ}_{size}x{size}_oddx{regs_ch}_{stride}(
              in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci,
              w_out - j);
        }}

        {typ} *__restrict in_ptr =
            input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
        {typ} *__restrict out_ptr =
            output_img + n_oc * h_out * w_out + i * w_out + j;
        {typ} *__restrict kernel_ptr = packed_kernel
                                       + (c_out / {regs_ch}) * ci * {regs_ch} * {size} * {size}
                                       + (n_oc % {regs_ch}) * ci * {size} * {size};
        switch(c_out - n_oc)
        {{
          {case_statement2}
        }}
      }}
    }}
  }}
  free(packed_kernel);
}}
'''

conv_generic_src = '''
{signature}
{{
  {typ} *packed_kernel = ({typ} *) malloc(128 * c_out * k_h * k_w * sizeof({typ}));

  for(size_t n_ic = 0; n_ic < c_in; n_ic += 128)
  {{
    const size_t ci = NS_MIN(128, c_in - n_ic);

    /* Kernel packing */
    for(size_t n_oc = 0; n_oc < (c_out / {regs_ch}) * {regs_ch}; n_oc += {regs_ch})
    {{
      nsimd_conv_pack_kernel_{typ}_{regs_ch}(
          kernel + (n_oc * c_in + n_ic) * k_h * k_w,
          packed_kernel + (n_oc / {regs_ch}) * ci * {regs_ch} * k_h * k_w, c_in, c_out, k_h, k_w, ci);
    }}

    const size_t n_oc = (c_out / {regs_ch}) * {regs_ch};
    switch(c_out - n_oc)
    {{
      {packing_case}
    }}

    for(size_t i = 0; i < h_out; i++)
    {{
      size_t j = 0;
      for(; j < (w_out / ({regs_w} * vlen({typ}))) * {regs_w} * vlen({typ}); j += {regs_w} * vlen({typ}))
      {{
        for(size_t n_oc = 0; n_oc < (c_out / {regs_ch}) * {regs_ch}; n_oc += {regs_ch})
        {{
          {typ} *__restrict in_ptr =
              input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
          {typ} *__restrict out_ptr =
              output_img + n_oc * h_out * w_out + i * w_out + j;
          {typ} *__restrict kernel_ptr =
              packed_kernel + (n_oc / {regs_ch}) * ci * {regs_ch} * k_h * k_w;
          _nsimd_conv_add_dot_{typ}_nxn_{regs_w}x{regs_ch}_{stride}(
              in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci, k_h, k_w);
        }}

        const size_t n_oc = (c_out / {regs_ch}) * {regs_ch};
        {typ} *__restrict in_ptr =
            input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
        {typ} *__restrict out_ptr =
            output_img + n_oc * h_out * w_out + i * w_out + j;
        {typ} *__restrict kernel_ptr = packed_kernel
                                       + (c_out / {regs_ch}) * ci * {regs_ch} * k_h * k_w
                                       + (n_oc % {regs_ch}) * ci * k_h * k_w;
        switch(c_out - n_oc)
        {{
          {case_statement0}
        }}        
      }}

      for(; j < (w_out / vlen({typ})) * vlen({typ}); j += vlen({typ}))
      {{
        for(size_t n_oc = 0; n_oc < (c_out / {regs_ch}) * {regs_ch}; n_oc += {regs_ch})
        {{
          {typ} *__restrict in_ptr =
              input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
          {typ} *__restrict out_ptr =
              output_img + n_oc * h_out * w_out + i * w_out + j;
          {typ} *__restrict kernel_ptr =
              packed_kernel + (n_oc / {regs_ch}) * ci * {regs_ch} * k_h * k_w;
          _nsimd_conv_add_dot_{typ}_nxn_1x{regs_ch}_{stride}(
              in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci, k_h, k_w);
        }}

        const size_t n_oc = (c_out / {regs_ch}) * {regs_ch};
        {typ} *__restrict in_ptr =
            input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
        {typ} *__restrict out_ptr =
            output_img + n_oc * h_out * w_out + i * w_out + j;
        {typ} *__restrict kernel_ptr = packed_kernel
                                       + (c_out / {regs_ch}) * ci * {regs_ch} * k_h * k_w
                                       + (n_oc % {regs_ch}) * ci * k_h * k_w;
        switch(c_out - n_oc)
        {{
          {case_statement1}
        }}
      }}

      j = (w_out / vlen({typ})) * vlen({typ});
      if(j < w_out)
      {{
        for(size_t n_oc = 0; n_oc < (c_out / {regs_ch}) * {regs_ch}; n_oc += {regs_ch})
        {{
          {typ} *__restrict in_ptr =
              input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
          {typ} *__restrict out_ptr =
              output_img + n_oc * h_out * w_out + i * w_out + j;
          {typ} *__restrict kernel_ptr =
              packed_kernel + (n_oc / {regs_ch}) * ci * {regs_ch} * k_h * k_w;
          _nsimd_conv_add_dot_{typ}_nxn_oddx{regs_ch}_{stride}(
              in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci,
              w_out - j, k_h, k_w);
        }}

        {typ} *__restrict in_ptr =
            input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;
        {typ} *__restrict out_ptr =
            output_img + n_oc * h_out * w_out + i * w_out + j;
        {typ} *__restrict kernel_ptr = packed_kernel
                                       + (c_out / {regs_ch}) * ci * {regs_ch} * k_h * k_w
                                       + (n_oc % {regs_ch}) * ci * k_h * k_w;
        switch(c_out - n_oc)
        {{
          {case_statement2}
        }}
      }}
    }}
  }}
  free(packed_kernel);
}}
'''

# Case statement for kernel_packing
packing_case = '''\
case {i}:
nsimd_conv_pack_kernel_{typ}_{i}(
  kernel + (n_oc * c_in + n_ic) * {k_h} * {k_w},
  packed_kernel + (c_out / {regs_ch}) * ci * {regs_ch} * {k_h} * {k_w}
  + (n_oc % {regs_ch}) * ci * {k_h} * {k_w}, c_in, c_out, {k_h}, {k_w}, ci);
break;
'''

# Fixed size conv kernel
case_statement0 = '''\
case {i}:
_nsimd_conv_add_dot_{typ}_{size}x{size}_{regs_w}x{i}_{stride}(
  in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci);
break;'''

case_statement1 = '''\
case {i}:
 _nsimd_conv_add_dot_{typ}_{size}x{size}_1x{i}_{stride}(
  in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci);
break;'''

case_statement2 = '''\
case {i}:
_nsimd_conv_add_dot_{typ}_{size}x{size}_oddx{i}_{stride}(
  in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci,
  w_out - j);
break;'''

# Generic size conv kernel
generic_case_statement0 = '''\
case {i}:
_nsimd_conv_add_dot_{typ}_nxn_{regs_w}x{i}_{stride}(
  in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci, k_h, k_w);
break;'''

generic_case_statement1 = '''\
case {i}:
 _nsimd_conv_add_dot_{typ}_nxn_1x{i}_{stride}(
  in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci, k_h, k_w);
break;'''

generic_case_statement2 = '''\
case {i}:
_nsimd_conv_add_dot_{typ}_nxn_oddx{i}_{stride}(
  in_ptr, out_ptr, kernel_ptr, h_in, w_in, h_out, w_out, ci,
  w_out - j, k_h, k_w);
break;'''

# -----------------------------------------------------------------------------
# Kernel packing functions

kernel_packing_signature = '''\
void nsimd_conv_pack_kernel_{typ}_{regs_ch}(
    {typ} *kernel, {typ} *packed_kernel, const size_t c_in, const size_t c_out,
    const size_t k_h, const size_t k_w, const size_t n_ch_in)'''

kernel_packing_src = '''\
{signature}
{{
  {typ} *__restrict packed_kernel_ptr = packed_kernel;
  {ptr_list}

  {init_list}

  for(size_t cci = 0; cci < n_ch_in; cci++)
  {{
    for(size_t s = 0; s < k_h; s++)
    {{
      for(size_t t = 0; t < k_w; t++)
      {{
        {update_list}
      }}
    }}
  }}
}}
'''

# -----------------------------------------------------------------------------

# Convolution API
conv_signature = '''\
void nsimd_conv_compute_{typ}(
    {typ} *input_img, {typ} *output_img, {typ} *kernel, const size_t c_in,
    const size_t h_in, const size_t w_in, const size_t c_out,
    const size_t h_out, const size_t w_out, const size_t k_h, const size_t k_w,
    const size_t stride_h, const size_t stride_w)'''

base_header_template = '''\
#ifndef {guard}
#define {guard}

#include <stdlib.h>
#include <nsimd/nsimd.h>

{includes_list}

#if NSIMD_CXX > 0
extern \"C\" {{
#endif

{decl_list}

# if NSIMD_CXX > 0
}} /* extern C */
#endif

#endif /* {guard} */
'''

c_api_header = '''\
#ifndef __NSIMD_MODULES_CONVOLUTION_API_H__
#define __NSIMD_MODULES_CONVOLUTION_API_H__

#include <stdlib.h>
#include <nsimd/nsimd.h>

{includes_list}

#define nsimd_conv_compute(                                                    \\
    input_img, output_img, kernel, c_in, h_in, w_in, cout, h_out, w_out, k_h,  \\
    k_w, stride_h, stride_w, T)                                                \\
  nsimd_conv_compute_##T(                                                      \\
      input_img, output_img, kernel, c_in, h_in, w_in, cout, h_out, w_out,     \\
      k_h, k_w, stride_h, stride_w)

#endif // __NSIMD_MODULES_CONVOLUTION_API_H__
'''

conv_src_template = '''\
{signature}
{{
if(stride_h == 1 && stride_w == 1)
{{
  if(k_h == 1 && k_w == 1)
  {{
    nsimd_conv_{typ}_1x1_1(input_img, output_img, kernel, 
                           c_in, h_in, w_in,  c_out, h_out, w_out);
  }}
  else if(k_h == 3 && k_w == 3)
  {{
    nsimd_conv_{typ}_3x3_1(input_img, output_img, kernel, 
                           c_in, h_in, w_in,  c_out, h_out, w_out);
  }}
  else
  {{
    nsimd_conv_{typ}_nxn_1(input_img, output_img, kernel, 
                           c_in, h_in, w_in,  c_out, h_out, w_out, 
                           k_h, k_w);
  }}
}}
else if (stride_h == 2 && stride_w == 2)
{{
  if(k_h == 1 && k_w == 1)
  {{
    nsimd_conv_{typ}_1x1_2(input_img, output_img, kernel, 
                           c_in, h_in, w_in,  c_out, h_out, w_out);
  }}
  else if(k_h == 3 && k_w == 3)
  {{
    nsimd_conv_{typ}_3x3_2(input_img, output_img, kernel, 
                           c_in, h_in, w_in,  c_out, h_out, w_out);
  }}
  else
  {{
    nsimd_conv_{typ}_nxn_2(input_img, output_img, kernel, 
                           c_in, h_in, w_in,  c_out, h_out, w_out, 
                           k_h, k_w);
  }}
}}
else
{{
nsimd_conv_compute_scalar_{typ}(
    input_img, output_img, kernel, 
    c_in, h_in, w_in,  c_out, h_out, w_out, 
    k_h, k_w, stride_h, stride_w);
}}
}}
'''

nsimd_conv_header = '''\
#ifndef __NSIMD_MODULES_CONVOLUTION_{TYP}_H__
#define __NSIMD_MODULES_CONVOLUTION_{TYP}_H__

#include <stdlib.h>
#include <string.h>
#include <nsimd/nsimd.h>

{includes_list}

#if NSIMD_CXX > 0
extern \"C\" {{
#endif

{signature}

# if NSIMD_CXX > 0
}} /* extern C */
#endif

#endif /* __NSIMD_MODULES_CONVOLUTION_{TYP}_H__ */
'''

nsimd_conv_src = '''\
#include \"nsimd/modules/convolution/nsimd_convolution.h\"

{scalar_conv_decl}

/* -------------------------------------------------------------------------- */

{api_func_def}

/* -------------------------------------------------------------------------- */

{scalar_conv_def}
'''

nsimd_conv_macros_header = '''\
#ifndef __NSIMD_MODULES_CONVOLUTION_MACROS_H__
#define __NSIMD_MODULES_CONVOLUTION_MACROS_H__

#include <stdlib.h>
#include <nsimd/nsimd.h>

#define NS_MIN(A, B) ((A) < (B) ? (A) : (B))

{macro_defs} 

#endif // __NSIMD_MODULES_CONVOLUTION_MACROS_H__
'''
