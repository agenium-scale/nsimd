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

# -----------------------------------------------------------------------------
# Helpers

def v_typ(typ):
    ''' Returns the nsimd vector type for a given scalar type.
    eg : v_typ('f32') will return:
    'vf32'. 
    '''
    return 'v' + typ
    
def decl_reg_line(name, typ, regs_w):
    ''' Generates a code that declares n registers, with a given base name
    and indexed until regs_w.
    For example decl_reg_line('v_acc', f32, 3) will output:
      'vf32 v_acc0, v_acc1, v_acc2;\n'.
    This function is used to declare input vectors for ths convolution kernels.
    '''
    return '{} {};\n'.format(v_typ(typ),
                             ','.join('{}{}'.format(name, i) \
                             for i in range(0, regs_w)))

def decl_reg_array(name, typ, regs_h, regs_w):
    ''' Generates the code that declares a regs_h x regs_w wide 
    nsimd registers block. Each vectors are named by {base_bame}{i}{j},
    where i is in [0, regs_h -1] and j is in [0, regs_w - 1].
    For example, decl_reg_line('v_acc', f32, 4, 3) will output :
      'vf32 v_acc00, v_acc01, v_acc02;\n'
      'vf32 v_acc10, v_acc11, v_acc12;\n'
      'vf32 v_acc20, v_acc21, v_acc22;\n'
      'vf32 v_acc30, v_acc31, v_acc32;\n'
    This function is used to declare register blocks in the convolution kernels.
    '''
    ret = ''
    for i in range(0, regs_h):
        ret += decl_reg_line('{}{}'.format(name, i), typ, regs_w)
    return ret

def get_img_offset(img, i, j, typ):
    return '{img} + {i} * h_out * w_out + {j} * vlen({typ})'.\
        format(img=img, i=i, j=j, typ=typ)

def load_acc_block(regs_h, regs_w, reg_name, op_name, img, typ):
    ret = ''
    for i in range(0, regs_h):
        ret += ''.join(
            '{reg_name}{i}{j} = {op_name}({index}, {typ});\n'.\
            format(reg_name=reg_name, i=i, j=j, op_name=op_name,
                   index=get_img_offset(img, i, j, typ), typ=typ) \
            for j in range(0, regs_w))
        ret += '\n'
    return ret

def load_odd_acc_block(regs_h, regs_w, reg_name, op_name, img, typ):
    ret = ''
    for i in range(0, regs_h):
        ret += ''.join(
            '{reg_name}{i}{j} = {op_name}_{typ}({index}, n);\n'.\
            format(reg_name=reg_name, i=i, j=j, op_name=op_name,
                   index=get_img_offset(img, i, j, typ), typ=typ) \
            for j in range(0, regs_w))
        ret += '\n'
    return ret

def load_input_block(regs_w, reg_name, op_name, img, typ, stride,
                     use_t=True, t_value='t'):
    if stride == 1:
        if use_t:
            return '\n'.join(
                '{reg_name}{i} = {op_name}({img} + {t} + {i} * vlen({typ}), {typ});'.\
                format(reg_name=reg_name, t=t_value, i=i, op_name=op_name, img=img,
                       typ=typ) for i in range(0, regs_w)) + '\n'
        else:
            return '\n'.join(
                '{reg_name}{i} = {op_name}({img} + {i} * vlen({typ}), {typ});'.\
                format(reg_name=reg_name, i=i, op_name=op_name, img=img,
                       typ=typ) for i in range(0, regs_w)) + '\n'
    else:
        if use_t:
            return '\n'.join(
                '{reg_name}{i} = {op_name}({img} + {t} + {i} * {stride} * vlen({typ}), {typ}).v0;'.\
                format(reg_name=reg_name, t=t_value, i=i, op_name=op_name, img=img,
                       stride=stride, typ=typ) for i in range(0, regs_w)) + '\n'
        else:
            return '\n'.join(
                '{reg_name}{i} = {op_name}({img} + {i} * {stride} * vlen({typ}), {typ}).v0;'.\
                format(reg_name=reg_name, i=i, op_name=op_name, img=img, 
                       stride=stride, typ=typ) for i in range(0, regs_w)) + '\n' 

def load_odd_input_block(regs_w, reg_name, op_name, img, typ, stride,
                     use_t=True, t_value='t'):
    if stride == 1:
        if use_t:
            return '\n'.join(
                '{reg_name}{i} = {op_name}({img} + {t} + {i} * vlen({typ}), n);'.\
                format(reg_name=reg_name, t=t_value, i=i, op_name=op_name, img=img,
                       typ=typ) for i in range(0, regs_w)) + '\n'
        else:
            return '\n'.join(
                '{reg_name}{i} = {op_name}({img} + {i} * vlen({typ}), n);'.\
                format(reg_name=reg_name, i=i, op_name=op_name, img=img,
                       typ=typ) for i in range(0, regs_w)) + '\n'
    else:
        if use_t:
            return '\n'.join(
                '{reg_name}{i} = {op_name}({img} + {t} + {i} * vlen({typ}), n);'.\
                format(reg_name=reg_name, t=t_value, i=i, op_name=op_name, img=img,
                       typ=typ) for i in range(0, regs_w)) + '\n'
        else:
            return '\n'.join(
                '{reg_name}{i} = {op_name}({img} + {i} * vlen({typ}), n);'.\
                format(reg_name=reg_name, i=i, op_name=op_name, img=img,
                       typ=typ) for i in range(0, regs_w)) + '\n' 
        
def gen_update_block(regs_ch, regs_w, set1_op, fma_op, typ):
    ret = ''
    for i in range(0, regs_ch):
        ret += 'v_coeff = {set1_op}(*kernel_ptr++, {typ});\n'.\
            format(set1_op=set1_op, typ=typ)
        ret += ''.join(
            'v_acc{i}{j} = {fma_op}(v_input{j}, v_coeff, v_acc{i}{j}, {typ});\n'.\
            format(i=i, j=j, fma_op=fma_op, typ=typ) for j in range(0, regs_w))
        ret += '\n'
    return ret

def store_acc_block(regs_ch, regs_w, reg_name, op_name, img, typ):
    ret = ''
    for i in range(0, regs_ch):
        ret += ''.join(
            '{op_name}({index}, {reg_name}{i}{j}, {typ});\n'.\
            format(reg_name=reg_name, i=i, j=j, op_name=op_name,
                   index=get_img_offset(img, i, j, typ), typ=typ) \
            for j in range(0, regs_w))
        ret += '\n'
    return ret

def store_odd_acc_block(regs_ch, regs_w, reg_name, op_name, img, typ):
    ret = ''
    for i in range(0, regs_ch):
        ret += ''.join(
            '{op_name}_{typ}({index}, {reg_name}{i}{j}, n);\n'.\
            format(reg_name=reg_name, i=i, j=j, op_name=op_name,
                   index=get_img_offset(img, i, j, typ), typ=typ) \
            for j in range(0, regs_w))
        ret += '\n'
    return ret

# -----------------------------------------------------------------------------
# Source code generation

sep = '''
/* -------------------------------------------------------------------------- */

'''

def odd_loads(typ, stride):
    if stride == 1:
        return odd_loads_s1.format(typ=typ, v_type=v_typ(typ))
    else :
        return odd_loads_s2.format(typ=typ, v_type=v_typ(typ))
                                
def get_kernel_signature(regs_ch, regs_w, kernel_size, typ, stride):
    if kernel_size > 0:
        return kernel_signature.\
            format(typ=typ, k_h=kernel_size, k_w=kernel_size, stride=stride,
                   regs_w=regs_w, regs_ch=regs_ch)\
                   +';\n'
    else:
        return kernel_generic_signature.\
            format(typ=typ, stride=stride, regs_w=regs_w, regs_ch=regs_ch) +';\n'

def get_odd_kernel_signature(regs_ch, kernel_size, typ, stride):
    if kernel_size > 0:
        return odd_kernel_signature.\
            format(typ=typ, k_h=kernel_size, k_w=kernel_size, stride=stride,
                   regs_ch=regs_ch) +';\n'
    else:
        return odd_kernel_generic_signature.\
            format(typ=typ, stride=stride,
                   regs_ch=regs_ch) +';\n'
 
    
def get_kernel_def(typ, regs_ch, regs_w, kernel_size, stride):
    # Variables
    acc_load_op = 'vloadu'
    in_load_op = 'vloadu' if stride ==1 else 'vload2u'
    set1_op = 'vset1'
    fma_op = 'vfma'
    store_op = 'vstoreu'

    # Function signature
    if kernel_size > 0:
        signature = kernel_signature.\
            format(typ=typ, k_h=kernel_size, k_w=kernel_size, stride=stride,
                   regs_w=regs_w, regs_ch=regs_ch)
    else :
        signature = kernel_generic_signature.\
            format(typ=typ, stride=stride, regs_w=regs_w, regs_ch=regs_ch)
    
    # Statements
    acc_decl = decl_reg_array('v_acc', typ, regs_ch, regs_w)
    input_decl = decl_reg_line('v_input', typ, regs_w)
    acc_load = load_acc_block(regs_ch, regs_w, 'v_acc', acc_load_op, 'output',
                              typ)
    # acc_load = 'LOAD_ACC_BLOCK_{}x{}({});\n'.format(regs_w, regs_ch, typ)
    # update_func = 'UPDATE_ACC_BLOCK_{}x{}({});\n\n'.format(regs_w, regs_ch, typ)
    
    # Loop
    if kernel_size == 1:
        kernel_begin_loop = ''
        input_load = load_input_block(regs_w, 'v_input', in_load_op, 'input_ptr',
                                      typ, stride, False)
        update_block = input_load \
            + '\n' + gen_update_block(regs_ch, regs_w, set1_op, fma_op, typ)
        # update_block += update_func
        kernel_end_loop = ''
    elif kernel_size == 3:
        kernel_begin_loop = ''
        update_block = ''
        for s in range(0, 3):
            update_block += '/* s = {} */\n'.format(s)
            for t in range(0, 3):
                update_block += '/* t = {} */\n'.format(t)
                update_block += \
                    load_input_block(regs_w, 'v_input', in_load_op, 'input_ptr',
                                      typ, stride, True, t)
                update_block += '\n'
                update_block += gen_update_block(regs_ch, regs_w, set1_op,
                                                 fma_op, typ)
                # update_block += update_func
            update_block += 'input_ptr += w_in;\n\n'
        kernel_end_loop = ''
    else:
        kernel_begin_loop = 'for(size_t s = 0; s < k_h; s++){\n'
        kernel_begin_loop += 'for(size_t t = 0; t < k_h; t++){\n'
        input_load = load_input_block(regs_w, 'v_input', in_load_op, 'input_ptr',
                                      typ, stride, True)
        update_block = input_load \
            + '\n' + gen_update_block(regs_ch, regs_w, set1_op, fma_op, typ)
        # update_block += update_func
        kernel_end_loop = '}\n input_ptr += w_in;\n }\n'

    # Store
    acc_store = store_acc_block(regs_ch, regs_w, 'v_acc', store_op, 'output',
                                typ)
    # acc_store = 'STORE_ACC_BLOCK_{}x{}({});\n'.format(regs_w, regs_ch, typ)

    return kernel_def.\
        format(signature=signature, acc_decl=acc_decl, input_decl=input_decl,
               acc_load=acc_load, kernel_begin_loop=kernel_begin_loop,
               update_block=update_block,
               kernel_end_loop=kernel_end_loop,
               acc_store=acc_store, v_typ=v_typ(typ), typ=typ)

def get_odd_kernel_def(typ, regs_ch, kernel_size, stride):
    # Variables
    acc_load_op = '_vld_n'
    in_load_op = '_vld_n_{}'.format(typ) if stride == 1 \
        else '_vld_n_{}_2'.format(typ)
    set1_op = 'vset1'
    fma_op = 'vfma'
    store_op = '_vst_n'.format(typ)

    # Function signature
    if kernel_size > 0:
        signature = odd_kernel_signature.\
            format(typ=typ, k_h=kernel_size, k_w=kernel_size, stride=stride,
                   regs_ch=regs_ch)
    else :
        signature = odd_kernel_generic_signature.\
            format(typ=typ, stride=stride, regs_ch=regs_ch)
    
    # Statements
    acc_decl = decl_reg_array('v_acc', typ, regs_ch, 1)
    input_decl = decl_reg_line('v_input', typ, 1)
    acc_load = load_odd_acc_block(regs_ch, 1, 'v_acc', acc_load_op, 'output',
                              typ)
    # acc_load = 'LOAD_ODD_ACC_BLOCK_1x{}({});\n'.format(regs_ch, typ)
    # update_func = 'UPDATE_ACC_BLOCK_1x{}({});\n\n'.format(regs_ch, typ)

    # Loop
    if kernel_size == 1:
        kernel_begin_loop = ''
        input_load = load_odd_input_block(1, 'v_input', in_load_op, 'input_ptr',
                                          typ, stride, False)
        update_block = input_load \
            + '\n' + gen_update_block(regs_ch, 1, set1_op, fma_op, typ)
        # update_block += update_func
        kernel_end_loop = ''
    elif kernel_size == 3:
        kernel_begin_loop = ''
        update_block = ''
        for s in range(0, 3):
            update_block += '/* s = {} */\n'.format(s)
            for t in range(0, 3):
                update_block += '/* t = {} */\n'.format(t)
                update_block += \
                    load_odd_input_block(1, 'v_input', in_load_op, 'input_ptr',
                                         typ, stride, True, t)
                update_block += '\n'
                update_block += gen_update_block(regs_ch, 1, set1_op,
                                                 fma_op, typ)
                # update_block += update_func
            update_block += 'input_ptr += w_in;\n\n'
        kernel_end_loop = ''
    else:
        kernel_begin_loop = 'for(size_t s = 0; s < k_h; s++){\n'
        kernel_begin_loop += 'for(size_t t = 0; t < k_h; t++){\n'
        input_load = load_odd_input_block(1, 'v_input', in_load_op, 'input_ptr',
                                          typ, stride, True)
        update_block = input_load \
            + '\n' + gen_update_block(regs_ch, 1, set1_op, fma_op, typ)
        # update_block += update_func
        kernel_end_loop = '}\n input_ptr += w_in;\n }\n'

    # Store
    acc_store = store_odd_acc_block(regs_ch, 1, 'v_acc', store_op, 'output',
                                typ)
    # acc_store = 'STORE_ODD_ACC_BLOCK_1x{}({})\n;'.format(regs_ch, typ)
    return kernel_def.\
        format(signature=signature, acc_decl=acc_decl, input_decl=input_decl,
               acc_load=acc_load, kernel_begin_loop=kernel_begin_loop,
               update_block=update_block,
               kernel_end_loop=kernel_end_loop,
               acc_store=acc_store, v_typ=v_typ(typ), typ=typ)

# -----------------------------------------------------------------------------

regs_ch = 4
regs_w = 3
regs_ch_vals = [i for i in range(1, regs_ch + 1)]
regs_w_vals = [regs_w, 1]
kernel_sizes = [1, 3]
types = ['u8', 'i8', 'u16', 'i16', 'u32', 'i32', 'u64', 'i64', 'f32', 'f64']
# types = ['f32']

def gen_src_file(typ, size, stride):
    base_dir = 'src/modules/convolution'

    # Generate filename
    if size == 0:
        filename = get_src_filename('kernels/conv_nxn_{}.{}.c'.format(stride, typ))
        content = get_head_include('kernels/conv_nxn_{}.{}.h'.format(stride, typ)) + '\n'
    else:
        filename = get_src_filename('kernels/conv_{0}x{0}_{1}.{2}.c'.format(size, stride, typ))
        content = get_head_include('kernels/conv_{0}x{0}_{1}.{2}.h'.format(size, stride, typ)) + '\n'

    # Generate src
    for regs_ch in regs_ch_vals:
        for regs_w in regs_w_vals:
            content += get_kernel_signature(regs_ch, regs_w,
                                            size, typ, stride) + '\n'
    for regs_ch in regs_ch_vals:
        content += get_odd_kernel_signature(regs_ch, size, typ, stride) + '\n'
    content += odd_loads(typ, stride)
    content += sep
    if size == 0:
        content += conv_generic_src.format(
            signature=conv_generic_signature.format(typ=typ, stride=stride),
            typ=typ, stride=stride,
            regs_ch=max(regs_ch_vals), regs_w=max(regs_w_vals),
            case_statement0=''.\
            join(generic_case_statement0.\
                 format(i=i, typ=typ, regs_w=max(regs_w_vals), stride=stride) \
                 for i in range(1, max(regs_ch_vals))),
            case_statement1=''.\
            join(generic_case_statement1.\
                 format(i=i, typ=typ, regs_w=max(regs_w_vals), stride=stride) \
                 for i in range(1, max(regs_ch_vals))),
            case_statement2=''\
            .join(generic_case_statement2.\
                  format(i=i, typ=typ, regs_w=max(regs_w_vals), stride=stride) \
                  for i in range(1, max(regs_ch_vals))),
            packing_case=''.\
            join(packing_case.\
                 format(i=i, typ=typ, k_h='k_h', k_w='k_w', regs_ch=max(regs_ch_vals)) \
                        for i in range(1, max(regs_ch_vals))))      
    else:
        content += conv_specific_src.format(
            signature=conv_specific_signature.format(typ=typ, k_h=size,
                                                     k_w=size, stride=stride),
            typ=typ, size=size, stride=stride,
            regs_ch=max(regs_ch_vals), regs_w=max(regs_w_vals),
            case_statement0=''\
            .join(case_statement0.\
                  format(i=i, typ=typ, regs_w=max(regs_w_vals),
                         size=size, stride=stride) \
                  for i in range(1, max(regs_ch_vals))),
            case_statement1=''.\
            join(case_statement1.\
                 format(i=i, typ=typ, regs_w=max(regs_w_vals),
                        size=size, stride=stride) \
                 for i in range(1, max(regs_ch_vals))),
            case_statement2=''.\
            join(case_statement2.\
                 format(i=i, typ=typ, regs_w=max(regs_w_vals),
                        size=size, stride=stride) \
                 for i in range(1, max(regs_ch_vals))),
            packing_case=''.\
            join(packing_case.\
                 format(i=i, typ=typ, k_h=size, k_w=size, regs_ch=max(regs_ch_vals)) \
                        for i in range(1, max(regs_ch_vals))))   
    content += sep
    for regs_ch in regs_ch_vals:
        for regs_w in regs_w_vals:  
            content += get_kernel_def(typ, regs_ch, regs_w, size, stride)
    for regs_ch in regs_ch_vals:
        content += get_odd_kernel_def(typ, regs_ch, size, stride)

    # Output file
    write_src(filename, content)

def gen_pack_kernel_src_file(typ, regs_ch_vals):
    filename = get_src_filename('packing/kernel_packing.{}.c'.format(typ))
    content = get_head_include('packing/kernel_packing.{}.h'.format(typ)) + '\n'
    
    for regs_ch in regs_ch_vals:
        content += kernel_packing_src.\
            format(
                signature=kernel_packing_signature.\
                format(typ=typ, regs_ch=regs_ch),
                typ=typ,
                ptr_list=''.join('{typ} *__restrict kernel_ptr{i};\n'.\
                                 format(typ=typ, i=i) for i in range(0, regs_ch)),
                init_list=''.join('kernel_ptr{i} = kernel + {i} * c_in * k_h * k_w;\n'.\
                                  format(i=i) for i in range(0, regs_ch)),
                update_list=''.join('*packed_kernel_ptr++ = *kernel_ptr{i}++;\n'.\
                                    format(i=i) for i in range(0, regs_ch)))
        content += '\n'
        write_src(filename, content)

def gen_nsimd_convolution(typ):
    base_dir = 'src/modules/convolution'
    filename = get_src_filename('nsimd_convolution.{}.c'.format(typ))
    content = nsimd_conv_src.format(
        scalar_conv_decl=scalar_conv_decl.format(typ=typ) + ';\n',
        api_func_def=conv_src_template.\
        format(typ=typ, signature=conv_signature.format(typ=typ)),
        scalar_conv_def=scalar_conv_src.format(typ=typ))
    write_src(filename, content)
    
if __name__ == '__main__':
    os.system('mkdir -p src/modules/convolution/')
    os.system('mkdir -p src/modules/convolution/packing/')
    os.system('mkdir -p src/modules/convolution/kernels/')
        
    # Generate src files
    for typ in types:
        gen_nsimd_convolution(typ)
        gen_pack_kernel_src_file(typ, regs_ch_vals)
        gen_src_file(typ, 0, 1)
        gen_src_file(typ, 0, 2)
        
        for size in kernel_sizes:
            gen_src_file(typ, size, 1)
            gen_src_file(typ, size, 2)
    
