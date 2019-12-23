# Use utf-8 encoding
# -*- coding: utf-8 -*-

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
import platform
import io
import sys
import subprocess
import collections
import re
import string

import common
from modules.fixed_point.operators import *

operators = fp_operators

# ------------------------------------------------------------------------------
# Get output of command

def get_command_output(args):
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    lines = p.communicate()[0].split('\n')[0:-1]
    return '\n'.join(['    {}'.format(l) for l in lines])

def gen_overview(opts):
    dirname = os.path.join(opts.script_dir, '..', 'doc', 'markdown',
                            'modules', 'fixed_point')
    os.system('mkdir -p {}'.format(dirname))
    filename = os.path.join(opts.script_dir, '..', 'doc', 'markdown',
                            'modules', 'fixed_point', 'overview.md')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as fout:
        fout.write('''
# NSIMD fixed-point module

## Description

This module implements a fixed-point numbers support for the NSIMD library.
Fixed-point numbers are integer types used to represent decimal numbers. A number `lf` 
of bits are used to encode its integer part, and `rt` bits are used to encode its 
fractional part.

The fixed_point module uses the templated type `nsimd::fixed_point::fp_t<lf, rt>` to 
represent a fixed_point number. All the basic floating-point arithmetic operaors have 
been defined, therefore fp\_t elements can be manipulated as normal numbers.
The fixed\_point module will use a `int8_t`, `int16_t`, or `int32_t` integer type for 
storage, depending on the value of `lf + 2 * rt`. 

All the functions of the module are under the namespace `nsimd::fixed_point`, 
and match the same interface than NSIMD.

The `fp_t` struct type is defined in `fixed.hpp`, and the associated simd `fpsimd_t` 
struct type is defined in `simd.hpp`.

The modules redefines the NSIMD pack type for fixed-point numbers, templated with `lf` 
and `rt` :

```C++
template <uint8_t lf, uint8_t rt>
struct pack;
```

Then, the pack can be manipulated as an NSIMD pack like other scalar types. Here is 
a minimal example :

```C++
// Assumes that NSIMD is in your include path
#include <iostream>
#include <nsimd/modules/fixed_point.hpp>

int main() {
  typedef nsimd::fixed_point::fp_t<8, 8> fp_t;
  typedef nsimd::fixed_point::pack<fp_t> vec_t;
  
  fp_t *input0;
  fp_t *input1;
  fp_t *res;
  
  // Input and output initializations 
  // We assume that a function float rand_float(); has been 
  // previously defined
  for(size_t i = 0; i < nsimd::fixed_point::len(fp_t()); i++) {
    input0 = fp_t(rand_float());
    input1 = fp_t(rand_float());
  }
  
  vec_t v0 = nsimd::fixed_point::loadu<vec_t>(input0);
  vec_t v1 = nsimd::fixed_point::loadu<vec_t>(input1);
  vec_t vres = nsimd::fixed_point::add(input0, input1);
  nsimd::fixed_point::storeu(res, vres);
  
  for(size_t i = 0; i < nsimd::fixed_point::len(fp_t()); i++) {
    std::cout << float(res[i]) << \" \";
  }
  std::cout << std::endl;
  
  return EXIT_SUCCESS;
}

```
        ''')

api_template = '''\
# {full_name}

{desc}
**TODO : Precise which type can be given to T (fixed point or scalar).**

## C++ APIs

```c++
{decl}
```
'''

decl_template = '''\
template <typename T> 
{ret}{op}({args});\n\n'''

def get_type(param):
    if param == 'V':
        return 'void '
    elif param == 'T':
        return 'T '
    elif param == 's': # Scalar parameter
        return 'typename T::value_type '
    elif param == 'cs': # Const scalar parameter
        return 'const typename T::value_type '
    elif param == 'cT&':
        return 'const T &'
    elif param == 's*': # Pointer to a scalar
        return 'typename T::value_type *'
    elif param == 'v': # Vector type
        return 'pack<T> '
    elif param == 'v&': # Vector type ref
        return 'pack<T> &'
    elif param == 'cv': # Const vector type
        return 'const pack<T> '
    elif param == 'cv&': # Const vector type reference
        return 'const pack<T> &'
    elif param == 'vl': # Vector of logical type
        return 'packl<T> '
    elif param == 'vl&': # Vector of logical type reference
        return 'packl<T> &'
    elif param == 'cvl': # Const vector of logical type
        return 'const packl<T> '
    elif param == 'cvl&': # Const vector of logical type reference
        return 'const packl<T> &'
    elif param == 'p':
        return 'int '
    else:
        return '<unknown>'
 
def gen_decl(op):
    ret = ''
    op_sign = op.cxx_operator
    for signature in op.signatures:
        signature = signature.split(' ')
        params = signature[2:]
        args = ', '.join('{}{}'.format(get_type(params[i]),'a{}'.format(i)) \
                         for i in range(0, len(params)))
        decl_base = decl_template.format(ret=get_type(signature[0]),
                                         op=signature[1], args=args)
        decl_op = ''
        if op_sign != '':
            decl_op = decl_template.format(ret=get_type(signature[0]),
                                                  op='operator{}'.format(op_sign), args=args)
        ret += decl_base + decl_op
        
    ret = 'namespace nsimd {\n' \
        +'namespace fixed_point {\n' \
        + ret \
        + '} // namespace fixed_point\n' \
        + '} // namespace nsimd'
    return ret

def gen_api(opts):
    filename = os.path.join(opts.script_dir, '..', 'doc', 'markdown',
                            'modules', 'fixed_point',
                            'api.md')
    with common.open_utf8(opts, filename) as fout:
        fout.write('''# NSIMD fixed-point API\n''')
        for cat in fp_categories:
            ops = [op for op in fp_operators if cat in op.categories]
            if(len(ops) == 0):
                continue
            
            fout.write('\n## {}\n\n'.format(cat))

            for op in ops:
                fout.write('- [{full_name} ({op_name})](api_{op_name}.md)\n'\
                           .format(full_name=op.full_name, op_name=op.name))
    
def gen_doc(opts):
    for op in operators:
        filename = os.path.join(opts.script_dir, '..', 'doc', 'markdown',
                                            'modules', 'fixed_point',
                                            'api_{}.md'.format(op.name))
        with common.open_utf8(opts, filename) as fout:
            fout.write(api_template.format(full_name=op.full_name,
                                           desc=op.desc, decl=gen_decl(op)))
            
def doit(opts):
    print('-- Generating doc for module fixed_point')
    gen_overview(opts)
    gen_api(opts)
    gen_doc(opts)
    
