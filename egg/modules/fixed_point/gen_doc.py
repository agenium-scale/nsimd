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
and match the same interface than `nsimd`.

The `fp_t` struct type is defined in `fixed.hpp`, and the associated simd `fpsimd_t` 
struct type is defined in `simd.hpp`.

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
(see the [API index](../../index.md)).

## Example

Here is a minimal example(main.cpp) :

```C++
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <nsimd/modules/fixed_point.hpp>

float rand_float() {
  return 4.0f * ((float) rand() / (float) RAND_MAX) - 2.0f;        
}

int main() {
  // We use fixed point numbers with 8 bits of integer part and 8 bits of 
  // decimal part. It will use a 32 bits integer for internal storage.
  typedef nsimd::fixed_point::fp_t<8, 8> fp_t;
  typedef nsimd::fixed_point::pack<fp_t> fp_pack_t;
  
  const size_t v_size = nsimd::fixed_point::len(fp_t());

  fp_t *input0 = (fp_t*)malloc(v_size * sizeof(fp_t));
  fp_t *input1 = (fp_t *)malloc(v_size * sizeof(fp_t));
  fp_t *res = (fp_t *)malloc(v_size * sizeof(fp_t));
  
  // Input and output initializations 
  for(size_t i = 0; i < nsimd::fixed_point::len(fp_t()); i++) {
    input0[i] = fp_t(rand_float());
    input1[i] = fp_t(rand_float());
  }
  
  fp_pack_t v0 = nsimd::fixed_point::loadu<fp_pack_t>(input0);
  fp_pack_t v1 = nsimd::fixed_point::loadu<fp_pack_t>(input1);
  fp_pack_t vres = nsimd::fixed_point::add(v0, v1);
  nsimd::fixed_point::storeu(res, vres);
  
  for(size_t i = 0; i < nsimd::fixed_point::len(fp_t()); i++) {
    std::cout << float(input0[i]) << \" | \"
      << float(input1[i]) << \" | \"
      << float(res[i]) << \"\\n\";
  }
  std::cout << std::endl;
  
  return EXIT_SUCCESS;
}

```

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

The T template parameter is of the following type depending on the operator:

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
    filename = os.path.join(opts.script_dir, '..', 'doc', 'markdown',
                            'modules', 'fixed_point',
                            'api.md')
    with common.open_utf8(opts, filename) as fout:
        fout.write('''# NSIMD fixed point API\n''')
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

header_src = '''\
<!DOCTYPE html>

<html>
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>{doc_title}</title>
    <style type=\"text/css\">
      body {{
        /*margin:40px auto;*/
        margin:10px auto;
        /*max-width:650px;*/
        max-width:800px;
        /*line-height:1.6;*/
        line-height:1.4;
        /*font-size:18px;*/
        color:#444;
        padding:0 10px
      }}
      h1,h2,h3 {{
        line-height:1.2
      }}
      table,th, td {{
        border: 1px solid gray;
        border-collapse : collapse;
        padding: 1px 3px;
      }}
    </style>
    <!-- https://www.mathjax.org/#gettingstarted -->
    <script src=\"{assets_dir}/polyfill.min.js\"></script>
    <script id=\"MathJax-script\" async src=\"{assets_dir}/tex-mml-chtml.js\"></script>
    <!-- Highlight.js -->
    <link rel=\"stylesheet\" href= \"{assets_dir}/highlight.js.default.min.css\">
    <script src=\"{assets_dir}/highlight.min.js\"></script>
    <script src=\"{assets_dir}/cpp.min.js\"></script>
    <script>hljs.initHighlightingOnLoad();</script>
  </head>
<body>

<center>
  <img src=\"{img_dir}/logo.svg\"><br>
  <br>
  <a href=\"../../index.html\">Index</a> |
  <a href=\"../../quick_start.html\">Quick Start</a> |
  <a href=\"../../tutorials.html\">Tutorials</a> |
  <a href=\"../../faq.html\">FAQ</a> |
  <a href=\"../../contribute.html\">Contribute</a> |
  <a href=\"../../overview.html\">API overview</a> |
  <a href=\"../../api.html\">API reference</a> |
  <a href=\"../../modules.html\">Modules</a>
  <br><hr>
  <b>Fixed point support : </b>
  <a href=\"overview.html\">Overview</a> |
  <a href=\"api.html\">Reference</a>
</center>
'''

footer_src = '''\
  </body>
</html>
'''

def gen_html(opts):
    # check if md2html exists
    md2html = 'md2html.exe' if platform.system() == 'Windows' else 'md2html'
    doc_dir = os.path.join(opts.script_dir, '..', 'doc')
    full_path_md2html = os.path.join(doc_dir, md2html)
    if not os.path.isfile(full_path_md2html):
        msg = '-- Cannot generate HTML: {} not found. '.format(md2html)
        if platform.system() == 'Windows':
            msg += 'Run "nmake /F Makefile.win" in {}'.format(doc_dir)
        else:
            msg += 'Run "make -f Makefile.nix" in {}'.format(doc_dir)
        print(msg)
        return

    # get all markdown files
    md_dir = os.path.join(doc_dir, 'markdown/modules/fixed_point')
    html_dir = os.path.join(doc_dir, 'html/modules/fixed_point')
    common.mkdir_p(html_dir)
    dirs = [md_dir]
    md_files = []
    while len(dirs) > 0:
        curr_dir = dirs.pop()
        entries = os.listdir(curr_dir)
        for entry in entries:
            full_path_entry = os.path.join(curr_dir, entry)
            if full_path_entry == '..' or full_path_entry == '.':
                continue
            elif os.path.isdir(full_path_entry):
                continue
            elif entry.endswith('.md'):
                md_files.append(full_path_entry)

    # header and footer
    doc_title = '`nsimd` fixed point module documentation'
    root_dir = '../..'
    assets_dir = '../../assets'
    img_dir = '../../img'
    header = header_src.format(doc_title=doc_title,
                               root_dir=root_dir,
                               img_dir=img_dir,
                               assets_dir=assets_dir)
    footer = footer_src
    tmp_file = os.path.join(doc_dir, 'tmp.html')
    for filename in md_files:
        i = filename.rfind('markdown')
        if i == -1:
            continue
        output = filename[0:i] + 'html' + filename[i + 8:-2] + 'html'
        common.mkdir_p(os.path.dirname(output))
        os.system('{} "{}" "{}"'.format(full_path_md2html, filename, tmp_file))
        with common.open_utf8(opts, output) as fout:
            fout.write(header)
            with io.open(tmp_file, mode='r', encoding='utf-8') as fin:
                fout.write(fin.read())
            fout.write(footer)
    
def doit(opts):
    print('-- Generating doc for module fixed_point')
    gen_overview(opts)
    gen_api(opts)
    gen_doc(opts)
    gen_html(opts)
    
