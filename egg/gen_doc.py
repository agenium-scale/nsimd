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
import common
import collections
import operators
import re
import string

categories = operators.categories
operators = operators.operators

# -----------------------------------------------------------------------------
# Get output of command

def get_command_output(args):
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    lines = p.communicate()[0].split('\n')[0:-1]
    return '\n'.join(['    {}'.format(l) for l in lines])

# -----------------------------------------------------------------------------

def gen_overview(opts):
    filename = common.get_markdown_file(opts, 'overview')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as fout:
        fout.write('''# Overview

## NSIMD scalar types

Their names follows the following pattern: `Sxx` where

- `S` is `i` for signed integers, `u` for unsigned integer and `f` for
  floatting point number.
- `xx` is the number of bits taken to represent the number.

Full list of scalar types:

''')
        for t in common.types:
            fout.write('- `{}`\n'.format(t))
        fout.write('''

## NSIMD SIMD vector types

Their names follows the following pattern: `vSCALAR` where `SCALAR` is a
one of scalar type listed above. For example `vi8` means a SIMD vector
containing `i8`'s.

Full list of SIMD vector types:

''')
        for t in common.types:
            fout.write('- `v{}`\n'.format(t))
        fout.write('''

## C/C++ base APIs

These come automatically when you include `nsimd/nsimd.h`. You do *not* need
to include a header file for having a function. In NSIMD, we call a platform
an architecture e.g. Intel, ARM, POWERPC. We call SIMD extension a set of
low-level functions and types provided to access a given SIDM extension.
Examples include SSE2, SSE42, AVX, ...

Here is a list of supported platforms and their corresponding SIMD extensions.

''')
        platforms = common.get_platforms(opts)
        for p in platforms:
            fout.write('- Platform `{}`\n'.format(p))
            for s in platforms[p].get_simd_exts():
                fout.write('  - `{}`\n'.format(s))
        fout.write('''
Each simd extension has its own set of SIMD types and functions. Types follow
the following pattern: `nsimd_SIMDEXT_vSCALAR` where

- `SIMDEXT` is the SIMD extensions.
- `SCALAR` is one of scalar types listed above.

There are also logical types associated to each SIMD vector type. These types
are used to represent the result of a comparison of SIMD vectors. They are
usually bit masks. Their name follow the following pattern:
`nsimd_SIMDEXT_vlSCALAR` where

- `SIMDEXT` is the SIMD extensions.
- `SCALAR` is one of scalar types listed above.

Note 1: Platform `cpu` is scalar fallback when no SIMD extension has been
specified.

Note 2: as all SIMD extensions of all platforms are different there is no
need to put the name of the platform in each identifier.

Function names follow the following pattern: `nsimd_SIMDEXT_FUNCNAME_SCALAR`
where

- `SIMDEXT` is the SIMD extensions.
- `FUNCNAME` is the name of a function e.g. `add` or `sub`.
- `SCALAR` is one of scalar types listed above.

### Generic identifier

In C, genericity is achieved using macros.

- `vec(SCALAR)` represents the SIMD vector type containing SCALAR elements.
  SCALAR must be one of scalar types listed above.
- `vecl(SCALAR)` represents the SIMD vector of logicals type containing SCALAR
  elements. SCALAR must be one of scalar types listed above.
- `vec_e(SCALAR)` represents the SIMD vector type containing SCALAR elements.
  SCALAR must be one of scalar types listed above.
- `vecl_e(SCALAR)` represents the SIMD vector of logicals type containing
  SCALAR elements. SCALAR must be one of scalar types listed above.
- `vFUNCNAME` is the macro name to access the function FUNCNAME e.g. `vadd`,
  `vsub`.
- `vFUNCNAME_e` is the macro name to access the function FUNCNAME e.g.
  `vadd_e`, `vsub_e`.

In C++98 and C++03, type traits are available.

- `nsimd::simd_traits<SCALAR, SIMDEXT>::vector` is the SIMD vector type for
  platform SIMDEXT containing SCALAR elements. SIMDEXT is one of SIMD
  extension listed above, SCALAR is one of scalar type listed above.
- `nsimd::simd_traits<SCALAR, SIMDEXT>::vectorl` is the SIMD vector of logicals
  type for platform SIMDEXT containing SCALAR elements. SIMDEXT is one of
  SIMD extensions listed above, SCALAR is one of scalar type listed above.

In C++11 and beyond, type traits are still available but typedefs are also
provided.

- `nsimd::vector<SCALAR, SIMDEXT>` is a typedef to
  `nsimd::simd_traits<SCALAR, SIMDEXT>::vector`.
- `nsimd::vectorl<SCALAR, SIMDEXT>` is a typedef to
  `nsimd::simd_traits<SCALAR, SIMDEXT>::vectorl`.

Note that all macro and functions available in plain C are still available in
C++.

### List of functions available for manipulation of SIMD vectors

For each FUNCNAME a C function (also available in C++)
named `nsimd_SIMDEXT_FUNCNAME_SCALAR` is available for each SCALAR type unless
specified otherwise.

For each FUNCNAME, a C macro (also available in C++) named `vFUNCNAME` is
available and takes as its last argument a SCALAR type.

For each FUNCNAME, a C macro (also available in C++) named `vFUNCNAME_a` is
available and takes as its two last argument a SCALAR type and a SIMDEXT.

For each FUNCNAME, a C++ function in namespace `nsimd` named `FUNCNAME` is
available. It takes as its last argument the SCALAR type and can optionnally
take the SIMDEXT as its last last argument.

For example, for the addition of two SIMD vectors `a` and `b` here are the
possibilities:

    c = nsimd_add_avx_f32(a, b); // use AVX
    c = nsimd::add(a, b, f32()); // use detected SIMDEXT
    c = nsimd::add(a, b, f32(), avx()); // force AVX even if detected SIMDEXT is not AVX
    c = vadd(a, b, f32); // use detected SIMDEXT
    c = vadd_e(a, b, f32, avx); // force AVX even if detected SIMDEXT is not AVX

Here is a list of available FUNCNAME.

''')
        for op_name, operator in operators.items():
            return_typ = common.get_one_type_generic(operator.params[0],
                                                     'SCALAR')
            func = operator.name
            args = ', '.join([common.get_one_type_generic(p, 'SCALAR') + \
                              ' a' + str(count) for count, p in \
                              enumerate(operator.params[1:])])
            fout.write('- `{} {}({});`\n'.format(return_typ, func, args))

            if operator.domain and len(operator.params[1:]) > 0:
                params = operator.params[1:]

                if len(params) == 1:
                    fout.write('  a0 ∈ {}\n'.format(operator.domain))
                else:
                    param = ', '.join(['a' + str(count) for count in \
                                       range(len(params))])
                    fout.write('  ({}) ∈ {}\n'.format(param, operator.domain))

            if len(operator.types) < len(common.types):
                typs = ', '.join(['{}'.format(t) for t in operator.types])
                fout.write('  Only available for {}\n'.format(typs))
        fout.write('''

## C++ advanced API

The C++ advanced API is called advanced not because it requires C++11 or above
but because it makes use of the particular implementation of ARM SVE by ARM
in their compiler. We do not know if GCC (and possibly MSVC in the distant
future) will use the same approach. Anyway the current implementation allows
us to put SVE SIMD vectors inside some kind of structs that behave like
standard structs. If you want to be sure to write portable code do *not* use
this API. Two new types are available.

- `nsimd::pack<SCALAR, N, SIMDEXT>` represents `N` SIMD vectors containing
  SCALAR elements of SIMD extension SIMDEXT. You can specify only the first
  template argument. The second defaults to 1 while the third defaults to the
  detected SIMDEXT.
- `nsimd::packl<SCALAR, N, SIMDEXT>` represents `N` SIMD vectors of logical
  type containing SCALAR elements of SIMD extension SIMDEXT. You can specify
  only the first template argument. The second defaults to 1 while the third
  defaults to the detected SIMDEXT.

Use N > 1 when declaring packs to have an unroll of N. This is particularily
useful on ARM.

Functions that takes packs do not take any other argument unless specified
otherwise e.g. the load family of funtions. It is impossible to determine
the kind of pack (unroll and SIMDEXT) from the type of a pointer. Therefore
in this case, the last argument must be a pack and this same type will then
return. Also some functions are available as C++ operators.

Here is the list of functions that act on packs.

''')
        for op_name, operator in operators.items():
            return_typ = common.get_one_type_pack(operator.params[0], 1, 'N')
            func = operator.name
            args = ', '.join([common.get_one_type_pack(p, 0, 'N') + ' a' + \
                              str(count) for count, p in \
                              enumerate(operator.params[1:])])
            if 'v' not in operator.params[1:] and 'l' not in operator.params[1:]:
                args = args + ', pack<T, N, SimdExt> const&' if args != '' \
                              else 'pack<T, N, SimdExt> const&'
            fout.write('- `{} {}({});`\n'.format(return_typ, func, args))

            if operator.domain and len(operator.params[1:]) > 0:
                params = operator.params[1:]
                if len(params) == 1:
                    fout.write('  a0 ∈ {}\n'.format(operator.domain))
                else:
                    param = ', '.join(['a'+str(count) for count in \
                                       range(len(params))])
                    fout.write('  ({}) ∈ {}\n'.format(param, operator.domain))

            if operator.cxx_operator:
                fout.write('  Available as `{}`\n'. \
                    format('operator'+operator.cxx_operator))

            if len(operator.types) < len(common.types):
                typs = ', '.join(['{}'.format(t) for t in operator.types])
                fout.write('  Only available for {}\n'.format(typs))

# -----------------------------------------------------------------------------

def gen_doc(opts):
    common.myprint(opts, 'Generating doc for each function')

    # Build tree for api.md
    api = dict()
    for _, operator in operators.items():
        for c in operator.categories:
            if c not in api:
                api[c] = [operator]
            else:
                api[c].append(operator)

    # api.md
    # filename = os.path.join(opts.script_dir, '..','doc', 'markdown', 'api.md')
    filename = common.get_markdown_file(opts, 'api')
    if common.can_create_filename(opts, filename):
        with common.open_utf8(opts, filename) as fout:
            fout.write('# API\n')
            for c, ops in api.items():
                if len(ops) == 0:
                    continue
                fout.write('\n## {}\n\n'.format(c.title))
                for op in ops:
                    Full_name = op.full_name[0].upper() + op.full_name[1:]
                    fout.write('- [{} ({})](api_{}.md)\n'.format(
                        Full_name, op.name, common.to_filename(op.name)))

    # helper to get list of function signatures
    def to_string(var):
        sigs = [var] if type(var) == str or not hasattr(var, '__iter__') \
                     else list(var)
        for i in range(0, len(sigs)):
            sigs[i] = re.sub('[ \n\t\r]+', ' ', sigs[i])
        return '\n'.join(sigs)

    # Operators (one file per operator)
    # dirname = os.path.join(opts.script_dir, '..','doc', 'markdown')
    dirname = common.get_markdown_dir(opts)
    common.mkdir_p(dirname)
    for op_name, operator in operators.items():
        # Skip non-matching doc
        if opts.match and not opts.match.match(op_name):
            continue
        # filename = os.path.join(dirname, 'api_{}.md'.format(common.to_filename(
        #                operator.name)))
        filename = common.get_markdown_api_file(opts, operator.name)
        if not common.can_create_filename(opts, filename):
            continue
        Full_name = operator.full_name[0].upper() + operator.full_name[1:]
        with common.open_utf8(opts, filename) as fout:
            fout.write('# {}\n\n'.format(Full_name))
            fout.write('## Description\n\n')
            fout.write(operator.desc)
            fout.write('\n\n## C base API (generic)\n\n')
            fout.write('```c\n')
            fout.write(to_string(operator.get_generic_signature('c_base')))
            fout.write('\n```\n\n')
            fout.write('## C++ base API (generic)\n\n')
            fout.write('```c++\n')
            fout.write(to_string(operator.get_generic_signature('cxx_base')))
            fout.write('\n```\n\n')
            fout.write('## C++ advanced API\n\n')
            fout.write('```c++\n')
            fout.write(to_string(operator.get_generic_signature('cxx_adv'). \
                                 values()))
            fout.write('\n```\n\n')
            fout.write('## C base API (architecture specifics)')
            for simd_ext in opts.simd:
                fout.write('\n\n### {}\n\n'.format(simd_ext.upper()))
                fout.write('```c\n')
                for typ in operator.types:
                    fout.write(operator.get_signature(typ, 'c_base', simd_ext))
                    fout.write(';\n')
                fout.write('```')
            fout.write('\n\n## C++ base API (architecture specifics)')
            for simd_ext in opts.simd:
                fout.write('\n\n### {}\n\n'.format(simd_ext.upper()))
                fout.write('```c\n')
                for typ in operator.types:
                    fout.write(operator.get_signature(typ, 'cxx_base',
                                                      simd_ext))
                    fout.write(';\n')
                fout.write('```')

# -----------------------------------------------------------------------------

def gen_modules_md(opts):
    common.myprint(opts, 'Generating modules.md')
    mods = common.get_modules(opts)
    ndms = []
    for mod in mods:
        name = eval('mods[mod].{}.hatch.name()'.format(mod))
        desc = eval('mods[mod].{}.hatch.desc()'.format(mod))
        ndms.append([name, desc, mod])
    filename = common.get_markdown_file(opts, 'modules')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as fout:
        fout.write('''# Modules
NSIMD comes with several additional modules. A module provides a set of
functionnalities that are usually not at the same level as SIMD intrinsics
and/or that do not provide all C and C++ APIs. These functionnalities are
given with the library because they make heavy use of NSIMD core which
abstract SIMD intrinsics. Below is the exhaustive list of modules.

''')
        for ndm in ndms:
            fout.write('- [{}](module_{}_overview.md)  \n'.format(ndm[0],
                                                                  ndm[2]))
            fout.write('\n'.join(['  {}'.format(line.strip()) \
                                  for line in ndm[1].split('\n')]))
            fout.write('\n\n')

# -----------------------------------------------------------------------------

def gen_html(opts):
    common.myprint(opts, 'Generating HTML documentation')
    common.gen_doc_html(opts, 'NSIMD documentation')

# -----------------------------------------------------------------------------

def doit(opts):
    gen_overview(opts)
    gen_doc(opts)
    gen_modules_md(opts)
    gen_html(opts) # This must be last
