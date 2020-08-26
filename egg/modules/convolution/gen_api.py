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
# -80K lines generated with these macros
load_acc_src = '''\
v_acc##I##{j} = vloadu(output + I * h_out * w_out + {j} * vlen(T), T)'''

load_odd_acc_src = '''\
v_acc##I##{j} = _vld_n_##T(output + I * h_out * w_out + {j} * vlen(T), n)'''

update_acc_src = '''\
v_acc##I##{j} = vfma(v_input##{j}, v_coeff, v_acc##I##{j}, T)'''

store_acc_src = '''\
vstoreu(output + I * h_out * w_out + {j} * vlen(T), v_acc##I##{j}, T)'''

store_odd_acc_src= '''\
_vst_n_##T(output + I * h_out * w_out + {j} * vlen(T), v_acc##I##{j}, n)'''

## -----------------------------------------------------------------------------

def gen_load_acc_macro(regs_w):
    content = '''\
    #define LOAD_ACC_LINE_1(I, T) \\
    {}\n
    '''.format(load_acc_src.format(j=0))    
    content += ''' \
    #define LOAD_ACC_LINE_{}(I, T) \\
    {}\n
    '''.format(regs_w, ';\\\n'.join(load_acc_src.format(j=j) \
                                    for j in range(0, regs_w)))
    return content

def gen_load_odd_acc_macro(regs_w):
    content = '''\
    #define LOAD_ODD_ACC_LINE_1(I, T) \\
    {}\n
    '''.format(load_odd_acc_src.format(j=0))    
    content += ''' \
    #define LOAD_ODD_ACC_LINE_{}(I, T) \\
    {}\n
    '''.format(regs_w, ';\\\n'.join(load_odd_acc_src.format(j=j) \
                                    for j in range(0, regs_w)))
    return content

def gen_store_acc_macro(regs_w):
    content = '''\
    #define STORE_ACC_LINE_1(I, T) \\
    {}\n
    '''.format(store_acc_src.format(j=0))
    content += '''\
    #define STORE_ACC_LINE_{}(I, T) \\
    {}\n
    '''.format(regs_w, ';\\\n'.join(store_acc_src.format(j=j)\
                                    for j in range(0, regs_w)))
    return content

def gen_store_odd_acc_macro(regs_w):
    content = '''\
    #define STORE_ODD_ACC_LINE_1(I, T) \\
    {}\n
    '''.format(store_odd_acc_src.format(j=0))
    content += '''\
    #define STORE_ODD_ACC_LINE_{}(I, T) \\
    {}\n
    '''.format(regs_w, ';\\\n'.join(store_odd_acc_src.format(j=j)\
                                    for j in range(0, regs_w)))
    return content

def gen_update_acc_macro(regs_w):
    content = '''\
     #define UPDATE_ACC_LINE_1(I, T) \\
    v_coeff = vset1(*kernel_ptr++, T); \\
    {}\n
    '''.format(update_acc_src.format(j=0))
    content += '''\
    #define UPDATE_ACC_LINE_{}(I, T) \\
    v_coeff = vset1(*kernel_ptr++, T); \\
    {}\n
    '''.format(regs_w, ';\\\n'.join(update_acc_src.format(j=j) \
                            for j in range(0, regs_w)))
    return content

def gen_load_acc_block(regs_ch, regs_w):
    return '''\
    #define LOAD_ACC_BLOCK_{}x{}(T) \\
    {}\n
    '''.format(regs_w, regs_ch,
               ';\\\n'.join('LOAD_ACC_LINE_{}({}, T)'.format(regs_w, i)\
                            for i in range(0, regs_ch - 1)))

def gen_load_odd_acc_block(regs_ch, regs_w):
    return '''\
    #define LOAD_ODD_ACC_BLOCK_{}x{}(T) \\
    {}\n
    '''.format(regs_w, regs_ch,
               ';\\\n'.join('LOAD_ODD_ACC_LINE_{}({}, T)'.format(regs_w, i)\
                            for i in range(0, regs_ch - 1)))

def gen_update_acc_block(regs_ch, regs_w):
    return '''\
    #define UPDATE_ACC_BLOCK_{}x{}(T) \\
    {}\n
    '''.format(regs_w, regs_ch,
               ';\\\n'.join('UPDATE_ACC_LINE_{}({}, T)'.format(regs_w, i)\
                            for i in range(0, regs_ch)))

def gen_store_acc_block(regs_ch, regs_w):
    return '''\
    #define STORE_ACC_BLOCK_{}x{}(T) \\
    {}\n
    '''.format(regs_w, regs_ch,
               ';\\\n'.join('STORE_ACC_LINE_{}({}, T)'.format(regs_w, i)\
                            for i in range(0, regs_ch - 1)))

def gen_store_odd_acc_block(regs_ch, regs_w):
    return '''\
    #define STORE_ODD_ACC_BLOCK_{}x{}(T) \\
    {}\n
    '''.format(regs_w, regs_ch,
               ';\\\n'.join('STORE_ODD_ACC_LINE_{}({}, T)'.format(regs_w, i)\
                            for i in range(0, regs_ch - 1)))

def gen_macros_header(regs_ch, regs_w):
    macro_defs = ''
    macro_defs += gen_load_acc_macro(regs_w)
    macro_defs += gen_load_odd_acc_macro(regs_w)
    macro_defs += gen_update_acc_macro(regs_w)
    macro_defs += gen_store_acc_macro(regs_w)
    macro_defs += gen_store_odd_acc_macro(regs_w)
    for i in range(1, regs_ch + 1):
        macro_defs += gen_load_acc_block(i, regs_w)
        macro_defs += gen_load_acc_block(i, 1)
        macro_defs += gen_load_odd_acc_block(i, regs_w)
        macro_defs += gen_load_odd_acc_block(i, 1)
        macro_defs += gen_update_acc_block(i, regs_w)
        macro_defs += gen_update_acc_block(i, 1)
        macro_defs += gen_store_acc_block(i, regs_w)
        macro_defs += gen_store_acc_block(i, 1)
        macro_defs += gen_store_odd_acc_block(i, regs_w)
        macro_defs += gen_store_odd_acc_block(i, 1)
    content = nsimd_conv_macros_header.format(macro_defs=macro_defs)
    filename = get_include_filename('macros.h')
    write_src(filename, content)
    
def gen_packing_header(typ, regs_ch):
    filename = get_include_filename('packing/kernel_packing.{}.h'.format(typ))
    guard = '__NSIMD_CONVOLUTION_KERNEL_PACKING_{}_H__'.\
        format(typ.upper())
    includes_list = '#include <nsimd/nsimd.h>\n'
    decl_list = '\n'.join(
        'NSIMD_DLLSPEC ' \
        + kernel_packing_signature.format(typ=typ, regs_ch=i) \
        + ';\n' \
        for i in range(1, regs_ch + 1))
    content = base_header_template.format(guard=guard,
                                          includes_list=includes_list,
                                          decl_list=decl_list)
    write_src(filename, content)

def gen_kernel_header(typ, size, stride):
    includes_list = '#include <nsimd/nsimd.h>\n' \
        + get_head_include('macros.h') \
        + get_head_include('packing/kernel_packing.{}.h'.format(typ))
    
    # Generate filename
    if size == 0:
        filename = get_include_filename('kernels/conv_nxn_{}.{}.h'.\
                                format(stride, typ))
        guard = '__NSIMD_MODULES_CONVOLUTION_CONV_NXN_{}_{}_H__'.\
        format(stride, typ.upper())
        decl_list = 'NSIMD_DLLSPEC '\
            + conv_generic_signature.format(typ=typ, stride=stride) \
            + ';\n'
    else:
        filename = get_include_filename('kernels/conv_{0}x{0}_{1}.{2}.h'.\
                                format(size, stride, typ))
        guard = '__NSIMD_MODULES_CONVOLUTION_CONV_{0}X{0}_{1}_{2}_H__'.\
            format(size, stride, typ.upper())
        decl_list = 'NSIMD_DLLSPEC ' \
            + conv_specific_signature.format(typ=typ, k_h=size,
                                             k_w=size, stride=stride) \
            +';\n'

    content = base_header_template.format(guard=guard,
                                          includes_list=includes_list,
                                          decl_list=decl_list)
    write_src(filename, content)

def gen_nsimd_convolution_header(typ, files):
    base_dir = 'include/nsimd/modules/convolution'
    filename = os.path.join(base_dir, 'nsimd_convolution.{}.h'.format(typ))
    content = nsimd_conv_header.format(
        TYP=typ.upper(), typ=typ,
        signature='NSIMD_DLLSPEC ' + conv_signature.format(typ=typ) + ';\n',
        includes_list=''.join(get_head_include(f) for f in files))
    write_src(filename, content)
    
# -----------------------------------------------------------------------------

regs_ch = 4
regs_w = 3
kernel_sizes = [1, 3]
types = ['u8', 'i8', 'u16', 'i16', 'u32', 'i32', 'u64', 'i64', 'f32', 'f64']
# types = ['f32']

if __name__ == '__main__':
    os.system('mkdir -p include/nsimd/modules/convolution')
    os.system('mkdir -p include/nsimd/modules/convolution/kernels')
    os.system('mkdir -p include/nsimd/modules/convolution/packing')

    for typ in types:
        files = []
        gen_packing_header(typ, regs_ch)
        for size in kernel_sizes:
            files.append('kernels/conv_{0}x{0}_1.{1}.h'.format(size, typ))
            files.append('kernels/conv_{0}x{0}_2.{1}.h'.format(size, typ))
            gen_kernel_header(typ, size, 1)
            gen_kernel_header(typ, size, 2)
        files.append('kernels/conv_nxn_1.{}.h'.format(typ))
        files.append('kernels/conv_nxn_2.{}.h'.format(typ))  
        gen_kernel_header(typ, 0, 1)
        gen_kernel_header(typ, 0, 2)
        gen_nsimd_convolution_header(typ, files)

    # Gen C API file
    filename = get_include_filename('nsimd_convolution.h')
    includes_list = ''.join(
        get_head_include('nsimd_convolution.{}.h'.format(typ)) \
        for typ in types)
    content = c_api_header.format(includes_list=includes_list)
    write_src(filename, content)

    gen_macros_header(regs_ch, regs_w)
    
