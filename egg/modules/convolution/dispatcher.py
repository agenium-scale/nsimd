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

class Conv:
    special_cases  = []
    default_case = 'scalar'

arm_conv = Conv()
arm_conv.special_cases = [
    ({'k_h' : 1, 'k_w' : 1, 'stride_h' : 1, 'stride_w' : 1},
     '''nsimd_conv_{}_1x1_1(input_img, output_img, kernel, 
     c_in, h_in, w_in, c_out, h_out, w_out)'''),
    ({'k_h' : 3, 'k_w' : 3, 'stride_h' : 1, 'stride_w' : 1},
     '''nsimd_conv_{}_3x3_1(input_img, output_img, kernel, 
     c_in, h_in, w_in, c_out, h_out, w_out)'''),
    ({'k_h' : 1, 'k_w' : 1, 'stride_h' : 2, 'stride_w' : 2},
     '''nsimd_conv_{}_1x1_2(input_img, output_img, kernel, 
     c_in, h_in, w_in, c_out, h_out, w_out)'''),
    ({'k_h' : 3, 'k_w' : 3, 'stride_h' : 2, 'stride_w' : 2},
     '''nsimd_conv_{}_3x3_2(input_img, output_img, kernel, 
     c_in, h_in, w_in, c_out, h_out, w_out)''')]
arm_conv.default_case = ''' _
nsimd_conv_{}_vect(input_img, output_img, kernel, c_in, h_in, w_in, c_out, 
h_out, w_out, kèh, k_w, stride_h, stride_w)'''

## -----------------------------------------------------------------------------

def gen_src(opts):
    if opts.simd_ext in neon:
        pass
    elif opts.simd_ext in intel:
        pass
    else:
        pass # SVE do it later

def gen_case_statement(s, i):
    ret = '''\
    case 0;
    {f};
    break;
    '''.format(f=s.format(n=i))
    ret += '\n'.join('''\
    case {n} :
    {f};
    break;'''.format(n=n, f=s.format(n=n)) for n in range(1, i))
    return ret

def gen_kernel_pack(regs_ch, typ, k):
    ret = '''\
    for(size_t n_oc = 0; n_oc < c_out; n_oc += {regs_ch}) {{
    switch((c_out - n_oc) % {regs_ch}) {{
    {case_list}
    }}
    }}
    '''
    f = 'nsimd_conv_pack_kernel_{}_'.format(typ, regs_ch) \
        + '{n}(' \
        + '''kernel + (n_oc * c_in + n_ic) * {k} * {k},
        packed_kernel + n_oc * ci * {k} * {k}, c_in, 
        c_out, {k}, {k}, ci);'''.format(k=k )
    case_list = gen_case_statement(f, regs_ch)
    return ret.format(regs_ch=regs_ch, case_list=case_list)

def gen_compute_block(regs_w, regs_ch, typ, k, stride):
    in_ptr = '''\
    {typ} *__restrict in_ptr =
    input_img + n_ic * h_in * w_in + {stride} * i * w_in + {stride} * j;'''.\
        format(typ=typ, stride=stride)
    out_ptr = '''\
    {typ} *__restrict out_ptr =
    output_img + n_oc * h_out * w_out + i * w_out + j;'''.\
        format(typ=typ)
    k_ptr = '''\
    {typ} *__restrict kernel_ptr = packed_kernel + n_oc * ci * {k} * {k};'''.\
    format(typ=typ, k=k)
        
def get_func_call(strat):
    return ''

def gen_conv_global(disp, typ):
    ret = ''
    for case, call in disp.special_cases:
        cond = ' && '.join('{} == {}'.format(key, val) \
                           for key, val in case.items())
        ret += 'if({}) {{\n {};\n return;\n}}'.format(cond, call.format(typ))
    ret += disp.default_case.format(typ)
    return ret

## -----------------------------------------------------------------------------

if __name__ == '__main__':
    src = gen_kernel_pack(4, 'f32', 3)
    src1 = gen_conv_global(arm_conv, 'f32')
    print(src)
    print(src1)
