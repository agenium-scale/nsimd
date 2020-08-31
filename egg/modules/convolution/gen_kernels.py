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

# -----------------------------------------------------------------------------

def v_typ(typ):
    return 'v' + typ

# -----------------------------------------------------------------------------
# Scalar kernel

scalar_kernel_signature = '''\
void _nsimd_conv_add_dot_{typ}_{regs_w}x{regs_ch}(
    {typ} *input, {typ} *output, {typ} *kernel, const size_t h_out,
    const size_t w_out, const size_t k_h, const size_t k_w, const size_t tile_h,
    const size_t n_ch_in)'''

scalar_kernel_src = '''\
{{
{acc_decl}

{typ} input0;
{typ} coeff;

{typ} *__restrict input_ptr = input;
for(size_t i = 0; i < tile_h; i++)
{{
  {typ} *__restrict kernel_ptr = kernel;
  for(size_t j = 0; j < tile_w; j++)
  {{
    {acc_init}
  }}

  for(size_t n_ic = 0; n_ic < n_ch_in; n_ic++)
  {{
    for(size_t s = 0; s < k_h; s++)
    {{
      for(size_t t = 0; t < k_w; t++)
      {{
        for(size_t j = 0; j < tile_w; j++)
        {{
          input0 = input_ptr[j];
          {acc_update}
        }}
        kernel_ptr += {regs_ch};
        input_ptr += tile_w;
      }}
    }}
  }}

  for(size_t j = 0; j < tile_w; j++)
  {{
    {acc_store}
  }}
}}
}}
'''

def gen_scalar_kernel(typ, regs_ch, regs_w):
    acc_decl = ''.\
        join('{typ} acc{i}[vlen({typ}) * {regs_w}];\n'.\
             format(typ=typ, i=i, regs_w=regs_w)\
             for i in range(0, regs_ch))
    acc_init = ''.\
        join('acc{i}[j] = output[{i} * h_out * w_out + i * w_out + j];\n'.\
        format(i=i) for i in range(0, regs_ch))
    acc_update = ''.\
        join('acc{i}[j] += kernel_ptr[{i}] * input0;\n'.format(i=i)\
             for i in range(0, regs_ch))
    acc_store = ''.\
        join('output[{i} + h_out * w_out + i * w_out + j] = acc{i}[j];\n'.\
             format(i=i) for i in range(0, regs_ch))

    ret = scalar_kernel_signature.\
        format(typ=typ, regs_ch=regs_ch, regs_w=regs_w)
    ret += scalar_kernel_src.\
        format(typ=typ, regs_ch=regs_ch, regs_w=regs_w,
               acc_decl=acc_decl, acc_init=acc_init, acc_update=acc_update,
               acc_store=acc_store)
    return ret;

## -----------------------------------------------------------------------------
# Odd kernel with masks

## -----------------------------------------------------------------------------
# Odd kernel without masks

## -----------------------------------------------------------------------------
# Regular kernel

regular_kernel_signature = '''\
void conv_add_dot_{typ}_{regs_w}x{regs_ch}(
    {typ} *input, {typ} *output, {typ} *kernel, const size_t h_out,
    const size_t w_out, const size_t k_h, const size_t k_w, const size_t tile_h,
    const size_t n_ch_in)'''

regular_kernel_src = '''\
{{
  const size_t out_stride = h_out * w_out;

  {acc_decl}

  {input_decl}
  v{typ} v_coeff;

  {typ} *__restrict input_ptr = input;
  for(size_t i = 0; i < tile_h; i++)
  {{ 
    {typ} *__restrict kernel_ptr = kernel;
    
    {acc_load}

    for(size_t n_ic = 0; n_ic < n_ch_in; n_ic++)
    {{
      for(size_t s = 0; s < k_h; s++)
      {{
        for(size_t t = 0; t < k_w; t++)
        {{
          {input_load}

          {acc_update}
        }}
      }}
      input_ptr += k_h * k_w * {regs_w} * {regs_ch} * vlen({typ});
    }}
    {acc_store}
  }}
}}
'''

def gen_regular_kernel(typ, regs_ch, regs_w):
    acc_decl = ''
    for i in range(0, regs_ch):
        acc_decl += 'v{} '.format(typ) \
            + ','.join('v_acc{}{}'.format(i, j) \
                       for j in range(0, regs_w)) + ';\n'
        acc_decl += '\n'
    input_decl = 'v{}'.format(typ) \
        + ','.join('v_input{}'.format(i) for i in range(0, regs_w)) + '\n'
    acc_load = ''
    for i in range(0, regs_ch):
        for j in range(0, regs_w):
            acc_load += '''\
            v_acc{i}{j} = vloadu(
            output + {i} * out_stride + i * w_out + vlen({typ}) * {j}, {typ});\n'''.\
                format(i=i, j=j, typ=typ)
        acc_load += '\n'
    input_load = ''
    for i in range(0, regs_w):
        input_load += '''\
        v_input0 = vloadu(
        input_ptr + {regs_w} * vlen({typ}) * (s * k_w + t) 
        + {i} * vlen({typ}));\n'''.format(i=i, typ=typ, regs_w=regs_w)
    acc_update = ''
    for i in range(0, regs_ch):
        for j in range(0, regs_w):
            acc_update += 'v_coeff = vset1(*kernel_ptr++, {typ});\n'.format(typ=typ)
            acc_update += '''\
            v_acc{i}{j} = vfma(v_input{j}, v_coeff, v_acc{i}{j}, {typ});\n'''.\
            format(i=i, j=j, typ=typ)
            acc_update += '\n'
    acc_store = ''
    for i in range(0, regs_ch):
        for j in range(0, regs_w):
            acc_store += '''\
            vstoreu(output + {i} * out_stride 
            + i * w_out + vlen({typ}) * {j}, v_acc{i}{j});\n'''. \
            format(i=i, j=j, typ=typ)
        acc_store += '\n'
    ret = regular_kernel_signature.\
        format(typ=typ, regs_ch=regs_ch, regs_w=regs_w)
    ret += regular_kernel_src.format(typ=typ, regs_w=regs_w, regs_ch=regs_ch,
        acc_decl=acc_decl, input_decl=input_decl, acc_load=acc_load,
        input_load=input_load, acc_update=acc_update, acc_store=acc_store)
    return ret

## -----------------------------------------------------------------------------
# Stride = 1 specific kernel for a fixed-size kernel length

stride_1_kernel_signature = '''\
void _nsimd_conv_add_dot_{typ}_{k}x{k}_{regs_h}x{regs_w}_1(
    f32 *input, f32 *output, f32 *kernel, const size_t h_in, const size_t w_in,
    const size_t h_out, const size_t w_out, const size_t c_in)'''

stride_1_kernel_src = '''\
{{
{acc_decl}

{input_decl}
{v_typ} v_coeff;

{begin_h_loop}
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
{end_h_loop}
}}

'''

def gen_stride_1_kernel(typ, regs_ch, regs_w, k, unroll=False):
    if unroll:
        pass
    else:
        pass    

## -----------------------------------------------------------------------------
# Stride = 2 specific kernel

## -----------------------------------------------------------------------------

if __name__ == '__main__':
    print(gen_scalar_kernel('f32', 4, 3))
    print(gen_regular_kernel('f32', 4, 3))
