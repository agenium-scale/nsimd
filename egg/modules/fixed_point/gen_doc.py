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

# ------------------------------------------------------------------------------

def gen_overview(opts):
    filename = common.get_markdown_file(opts, 'overview', 'fixed_point')
    with common.open_utf8(opts, filename) as fout:
        fout.write('''
# NSIMD fixed point module

## Description

This module implements a fixed-point numbers support for the `nsimd` library.
Fixed-point numbers are integer types used to represent decimal numbers. A
number `lf` of bits are used to encode its integer part, and `rt` bits are used
to encode its fractional part.

The fixed_point module uses the templated type `nsimd::fixed_point::fp_t<lf,
rt>` to represent a fixed_point number. All the basic floating-point arithmetic
operaors have been defined, therefore fp_t elements can be manipulated as
normal numbers.  The fixed_point module will use a `i8`, `i16`, or
`i32` integer type for storage, depending on the value of `lf + 2 * rt`.

All the functions of the module are under the namespace `nsimd::fixed_point`,
and match the same interface than `nsimd` C++ .

The `fp_t` struct type is defined in `fixed.hpp`, and the associated simd
`fpsimd_t` struct type are defined in `simd.hpp`.

The modules redefines the `nsimd` pack type for fixed-point numbers, templated
with `lf` and `rt` :

```C++
namespace nsimd {
namespace fixed_point {
template <u8 lf, u8 rt>
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

Here is a minimal example([main.cpp](../../examples/module_fixed_point.cpp)):

@[INCLUDE_CODE:L21:L61](../../examples/module_fixed_point.cpp)

To test with avx2 run :
```bash
export NSIMD_ROOT=<path/to/nsimd>
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

# -----------------------------------------------------------------------------

def get_type(param, return_typ=False):
    if param == '_':
        return 'void'
    elif param == '*':
        return 'typename T::value_type *'
    elif param == 'c*':
        return 'const typename T::value_type *'
    elif param == 's':
        return 'typename T::value_type'
    elif param in 'v':
        return 'pack<T>' if return_typ else 'const pack<T> &'
    elif param == 'l':
        return 'packl<T>' if return_typ else 'const packl<T> &'
    elif param == 'p':
        return 'int '
    else:
        return None

# -----------------------------------------------------------------------------

def gen_decl(op):
    sig = '{}{} {{}}({});'.format(
            'template <typename T> ' \
                if 'v' not in op.params[1:] and \
                   'l' not in op.params[1:] else '',
            get_type(op.params[0], True),
            ', '.join(['{} {}'.format(
                               get_type(op.params[i + 1]),
                                        common.get_arg(i)) \
                                        for i in range(len(op.params[1:]))])
          )
    ret = 'namespace nsimd {\n' \
          'namespace fixed_point {\n\n' + sig.format(op.name) + '\n\n'
    if op.cxx_operator != None:
        ret += sig.format('operator' + op.cxx_operator) + '\n\n'
    ret += '} // namespace fixed_point\n' \
           '} // namespace nsimd'
    return ret

# -----------------------------------------------------------------------------

def gen_api(opts, op_list):
    api = dict()
    for _, operator in operators.operators.items():
        if operator.name not in op_list:
            continue
        for c in operator.categories:
            if c not in api:
                api[c] = [operator]
            else:
                api[c].append(operator)

    filename = common.get_markdown_file(opts, 'api', 'fixed_point')
    with common.open_utf8(opts, filename) as fout:
        fout.write('''# NSIMD fixed point API\n''')
        for c, ops in api.items():
            if len(ops) == 0:
                continue
            fout.write('\n## {}\n\n'.format(c.title))
            for op in ops:
                fout.write('- [{} ({})](module_fixed_point_api_{}.md)\n'. \
                           format(op.full_name, op.name,
                                  common.to_filename(op.name)))

# -----------------------------------------------------------------------------

def gen_doc(opts, op_list):
    for _, op in operators.operators.items():
        if op.name not in op_list:
            continue
        filename = common.get_markdown_api_file(opts, op.name, 'fixed_point')
        with common.open_utf8(opts, filename) as fout:
            fout.write(api_template.format(full_name=op.full_name,
                                           desc=op.desc, decl=gen_decl(op)))

# -----------------------------------------------------------------------------

def doit(opts, op_list):
    common.myprint(opts, 'Generating doc for module fixed_point')
    gen_overview(opts)
    gen_api(opts, op_list)
    gen_doc(opts, op_list)

