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
import operators

## List of the NSIMD operators currently suppported by the module
# op_list = [
#     'len', 'set1', 'loadu', 'loada', 'loadlu', 'loadla', 'storeu', 'storea',
#     'add', 'sub', 'mul', 'div', 'fma', 'min', 'max', 'eq', 'ne', 'le', 'lt',
#     'ge', 'gt', 'if_else1', 'andb', 'andnotb', 'notb', 'orb', 'xorb', 'andl',
#     'andnotl', 'orl']

from modules.fixed_point.operators import *

operators = fp_operators

# ------------------------------------------------------------------------------

def gen_overview(opts):
    filename = common.get_markdown_file(opts, 'overview', 'fixed_point')
    with common.open_utf8(opts, filename) as fout:
        fout.write('''
# NSIMD fixed point module

## Description

This module implements a fixed-point numbers support for the `nsimd` library.
Fixed-point numbers are integer types used to represent decimal numbers. A number `lf` 
of bits are used to encode its integer part, and `rt` bits are used to encode its 
fractional part.

The fixed_point module uses the templated type `nsimd::fixed_point::fp_t<lf, rt>` to 
represent a fixed_point number. All the basic floating-point arithmetic operaors have 
been defined, therefore fp_t elements can be manipulated as normal numbers.
The fixed_point module will use a `int8_t`, `int16_t`, or `int32_t` integer type for 
storage, depending on the value of `lf + 2 * rt`. 

All the functions of the module are under the namespace `nsimd::fixed_point`, 
and match the same interface than `nsimd` C++ .

The `fp_t` struct type is defined in `fixed.hpp`, and the associated simd `fpsimd_t` 
struct type are defined in `simd.hpp`.

The modules redefines the `nsimd` pack type for fixed-point numbers, templated with `lf` 
and `rt` :

```C++
namespace nsimd {
namespace fixed_point {
template <uint8_t lf, uint8_t rt>
struct pack;
} // namespace fixed_point
} // namespace nsimd
```

Then, the pack can be manipulated as an `nsimd` pack like other scalar types. 

## Compatibility

The fixed point module is a C++ only API, compatible with the C++98 standard.
It has the same compilers and hardware support than the main `nsimd` API 
(see the [API index](index.md)).

## Example

Here is a minimal example([main.cpp](../src/module_fixed_point_example.cpp)) :
@[INCLUDE_CODE:L21:L61](../src/module_fixed_point_example.cpp)

To test with avx2 run : 
```bash
export NSIMD_ROOT=<path/to/simd>
g++ -o main -I$NSIMD_ROOT/include -mavx2 -DNSIMD_AVX2 main.cpp
./main
```

The console output will look like this : 
```console
$>./main 
1.35938 | -0.421875 | 0.9375
1.13281 | 1.19531 | 2.32812
1.64844 | -1.21094 | 0.4375
-0.660156 | 1.07422 | 0.414062
-0.890625 | 0.214844 | -0.675781
-0.0898438 | 0.515625 | 0.425781
-0.539062 | 0.0546875 | -0.484375
1.80859 | 1.66406 | 3.47266
```
        ''')

api_template = '''\
# {full_name}

{desc}

## Template parameter type for T: 

When using the following typedef :
```c++
typedef nsimd::fixed_point::fp_t<lf, rt> fp_t
```

The T template parameter is one of the following types depending on the operator:

- `set1`, `loadu` and `loada`: 
```c++
nsimd::fixed_point::pack<fp_t>
```
- `loadlu`, `loadla`: 
```c++
nsimd::fixed_point::packl<fp_t>
```
- Other operators: 
```c++ 
nsimd::fixed_point::fp_t<lf, rt>
```

## C++ API

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
    elif param == 'cs&': # Const scalar parameter
        return 'const typename T::value_type &'
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
    filename = common.get_markdown_file(opts, 'api', 'fixed_point')
    with common.open_utf8(opts, filename) as fout:
        fout.write('''# NSIMD fixed point API\n''')
        for cat in fp_categories:
            ops = [op for op in fp_operators if cat in op.categories]
            if(len(ops) == 0):
                continue
            
            fout.write('\n## {}\n\n'.format(cat))

            for op in ops:
                fout.write(
                    '- [{full_name} ({op_name})](module_fixed_point_api_{op_name}.md)\n'\
                           .format(full_name=op.full_name, op_name=op.name))
    
def gen_doc(opts):
    for op in operators:
        filename = common.get_markdown_api_file(opts, op.name, 'fixed_point')
        with common.open_utf8(opts, filename) as fout:
            fout.write(api_template.format(full_name=op.full_name,
                                           desc=op.desc, decl=gen_decl(op)))

def gen_html(opts):
    links = {
        'overview': 'Overview',
        'api': 'Reference'}
    common.gen_doc_html(opts, 'Fixed point support', 'fixed_point', links)
 
def doit(opts):
    print('-- Generating doc for module fixed_point')
    gen_overview(opts)
    gen_api(opts)
    gen_doc(opts)
    gen_html(opts)
    
