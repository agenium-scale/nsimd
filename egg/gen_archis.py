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

import operators
import common
import os
from datetime import date
import sys

# -----------------------------------------------------------------------------
# Generate code for output

def get_simd_implementation(operator, mod, simd_ext):
    typ_pairs = []
    for t in operator.types:
        return_typs = common.get_output_types(t, operator.output_to)
        for tt in return_typs:
            typ_pairs.append([t, tt])

    if not operator.closed:
        tmp = [p for p in typ_pairs if p[0] in common.ftypes and \
                                       p[1] in common.ftypes]
        tmp += [p for p in typ_pairs if p[0] in common.itypes and \
                                        p[1] in common.itypes]
        tmp += [p for p in typ_pairs if p[0] in common.utypes and \
                                        p[1] in common.utypes]
        tmp += [p for p in typ_pairs \
                if (p[0] in common.utypes and p[1] in common.itypes) or \
                   (p[0] in common.itypes and p[1] in common.utypes)]
        tmp += [p for p in typ_pairs \
                if (p[0] in common.iutypes and p[1] in common.ftypes) or \
                   (p[0] in common.ftypes and p[1] in common.iutypes)]
        typ_pairs = tmp

    ret = ''
    for pair in typ_pairs:
        from_typ = pair[0]
        to_typ = pair[1]
        fmtspec = operator.get_fmtspec(from_typ, to_typ, simd_ext)
        if operator.src:
            ret += \
            '''{hbar}

               #if NSIMD_CXX > 0
               extern "C" {{
               #endif

               NSIMD_DLLSPEC
               {return_typ} nsimd_{name}_{simd_ext}_{suf}({c_args});

               #if NSIMD_CXX > 0
               }} // extern "C"
               #endif

               #if NSIMD_CXX > 0
               namespace nsimd {{
                 NSIMD_INLINE {return_typ} {name}({cxx_args}) {{
                   {returns}nsimd_{name}_{simd_ext}_{suf}({vas});
                 }}
               }} // namespace nsimd
               #endif

               '''.format(**fmtspec)
        else:
            ret += \
            '''{hbar}

               NSIMD_INLINE
               {return_typ} nsimd_{name}_{simd_ext}_{suf}({c_args}) {{
                 {content}
               }}

               #if NSIMD_CXX > 0
               namespace nsimd {{
                 NSIMD_INLINE {return_typ} {name}({cxx_args}) {{
                   {returns}nsimd_{name}_{simd_ext}_{suf}({vas});
                 }}
               }} // namespace nsimd
               #endif

               '''.format(content=mod.get_impl(operator.name,
                          simd_ext, from_typ, to_typ), **fmtspec)
    return ret[0:-2]


# -----------------------------------------------------------------------------
# Generate code for output

def gen_archis_write_put(opts, platform, simd_ext, simd_dir):
    filename = os.path.join(simd_dir, 'put.h')
    if not common.can_create_filename(opts, filename):
        return
    op = None
    with common.open_utf8(opts, filename) as out:
        out.write( \
        '''#ifndef NSIMD_{PLATFORM}_{SIMD_EXT}_PUT_H
           #define NSIMD_{PLATFORM}_{SIMD_EXT}_PUT_H

           {include_cpu_put}#include <nsimd/{platform}/{simd_ext}/types.h>
           #include <stdio.h>

           {hbar}

           '''.format(year=date.today().year, hbar=common.hbar,
                      simd_ext=simd_ext, platform=platform,
                      PLATFORM=platform.upper(), SIMD_EXT=simd_ext.upper(),
                      include_cpu_put='#include <nsimd/cpu/cpu/put.h>\n' \
                      if simd_ext != 'cpu' else ''))
        for typ in common.types:
            out.write( \
            '''#if NSIMD_CXX > 0
               extern "C" {{
               #endif

               NSIMD_DLLSPEC
               int nsimd_put_{simd_ext}_{typ}(FILE *, const char *,
                                              nsimd_{simd_ext}_v{typ});

               #if NSIMD_CXX > 0
               }} // extern "C"
               #endif

               #if NSIMD_CXX > 0
               namespace nsimd {{
               NSIMD_INLINE int put(FILE *out, const char *fmt,
                                    nsimd_{simd_ext}_v{typ} a0, {typ},
                                    {simd_ext}) {{
                 return nsimd_put_{simd_ext}_{typ}(out, fmt, a0);
               }}
               }} // namespace nsimd
               #endif

               {hbar}

               #if NSIMD_CXX > 0
               extern "C" {{
               #endif

               NSIMD_DLLSPEC
               int nsimd_put_{simd_ext}_l{typ}(FILE *, const char *,
                                              nsimd_{simd_ext}_vl{typ});

               #if NSIMD_CXX > 0
               }} // extern "C"
               #endif

               #if NSIMD_CXX > 0
               namespace nsimd {{
               NSIMD_INLINE int putl(FILE *out, const char *fmt,
                                    nsimd_{simd_ext}_vl{typ} a0, {typ},
                                    {simd_ext}) {{
                 return nsimd_put_{simd_ext}_l{typ}(out, fmt, a0);
               }}
               }} // namespace nsimd
               #endif

               {hbar}
               '''.format(simd_ext=simd_ext, hbar=common.hbar, typ=typ))
        out.write('#endif')
    common.clang_format(opts, filename)


# -----------------------------------------------------------------------------
# Generate code for architectures

def gen_archis_write_file(opts, op, platform, simd_ext, simd_dir):
    filename = os.path.join(simd_dir, '{}.h'.format(op.name))
    if not common.can_create_filename(opts, filename):
        return
    mod = opts.platforms[platform]
    with common.open_utf8(opts, filename) as out:
        out.write('''#ifndef {guard}
                     #define {guard}

                     #include <nsimd/{platform}/{simd_ext}/types.h>
                     {additional_include}

                     {code}

                     {hbar}

                     #endif
                     '''.format(additional_include=mod.get_additional_include(
                                                     op.name, platform,
                                                     simd_ext),
                                year=date.today().year,
                                guard=op.get_header_guard(platform, simd_ext),
                                platform=platform,
                                simd_ext=simd_ext,
                                func=op.name, hbar=common.hbar,
                                code=get_simd_implementation(op, mod, simd_ext)))
    common.clang_format(opts, filename)

def gen_archis_simd(opts, platform, simd_ext, simd_dir):
    for op_name, operator in operators.operators.items():
        gen_archis_write_file(opts, operator, platform, simd_ext, simd_dir)
    gen_archis_write_put(opts, platform, simd_ext, simd_dir)

def gen_archis_types(opts, simd_dir, platform, simd_ext):
    filename = os.path.join(simd_dir, 'types.h')
    if not common.can_create_filename(opts, filename):
        return
    mod = opts.platforms[platform]
    c_code = '\n'.join(['typedef {} nsimd_{}_v{};'.format(mod.get_type(
                       simd_ext, t), simd_ext, t) for t in common.types])
    c_code += '\n\n'
    c_code += '\n'.join(['typedef {} nsimd_{}_vl{};'.format(
              mod.get_logical_type(simd_ext, t), simd_ext, t)
              for t in common.types])
    if mod.has_compatible_SoA_types(simd_ext):
        for deg in range(2, 5):
            c_code += '\n'.join(['typedef {} nsimd_{}_v{}x{};'. \
                                 format(mod.get_SoA_type(simd_ext, typ, deg),
                                 simd_ext, typ, deg) for typ in common.types])
    else:
        c_code += '\n'.join(['''
                             typedef struct nsimd_{simd_ext}_v{typ}x2 {{
                               nsimd_{simd_ext}_v{typ} v0;
                               nsimd_{simd_ext}_v{typ} v1;
                             }} nsimd_{simd_ext}_v{typ}x2;
                             '''.format(simd_ext=simd_ext, typ=typ) \
                                        for typ in common.types])
        c_code += '\n'.join(['''
                             typedef struct nsimd_{simd_ext}_v{typ}x3 {{
                               nsimd_{simd_ext}_v{typ} v0;
                               nsimd_{simd_ext}_v{typ} v1;
                               nsimd_{simd_ext}_v{typ} v2;
                             }} nsimd_{simd_ext}_v{typ}x3;
                             '''.format(simd_ext=simd_ext, typ=typ) \
                                        for typ in common.types])
        c_code += '\n'.join(['''
                             typedef struct nsimd_{simd_ext}_v{typ}x4 {{
                               nsimd_{simd_ext}_v{typ} v0;
                               nsimd_{simd_ext}_v{typ} v1;
                               nsimd_{simd_ext}_v{typ} v2;
                               nsimd_{simd_ext}_v{typ} v3;
                             }} nsimd_{simd_ext}_v{typ}x4;
                             '''.format(simd_ext=simd_ext, typ=typ) \
                                        for typ in common.types])
        c_code += '\n\n'
    cxx_code = '\n\n'.join(['''template <>
                               struct simd_traits<{typ}, {simd_ext}> {{
                                 typedef nsimd_{simd_ext}_v{typ} simd_vector;
                                 typedef nsimd_{simd_ext}_v{typ}x2 simd_vectorx2;
                                 typedef nsimd_{simd_ext}_v{typ}x3 simd_vectorx3;
                                 typedef nsimd_{simd_ext}_v{typ}x4 simd_vectorx4;
                                 typedef nsimd_{simd_ext}_vl{typ} simd_vectorl;
                               }};'''.format(typ=t, simd_ext=simd_ext)
                               for t in common.types])
    with common.open_utf8(opts, filename) as out:
        out.write('''#ifndef NSIMD_{platform}_{SIMD_EXT}_TYPES_H
                     #define NSIMD_{platform}_{SIMD_EXT}_TYPES_H

                     {c_code}

                     #define NSIMD_{simd_ext}_NB_REGISTERS  {nb_registers}

                     #if NSIMD_CXX > 0
                     namespace nsimd {{

                     struct {simd_ext} {{}};

                     {cxx_code}

                     }} // namespace nsimd
                     #endif

                     #endif
                     '''.\
                     format(year=date.today().year,
                            platform=platform.upper(),
                            SIMD_EXT=simd_ext.upper(),
                            c_code=c_code, cxx_code=cxx_code,
                            simd_ext=simd_ext,
                            nb_registers=mod.get_nb_registers(simd_ext)))
    common.clang_format(opts, filename)

def gen_archis_platform(opts, platform):
    include_dir = os.path.join(opts.include_dir, platform);
    for s in opts.platforms[platform].get_simd_exts():
        print ('-- Found new SIMD extension: {}'.format(s))
        if s in opts.simd:
            simd_dir = os.path.join(include_dir, s)
            common.mkdir_p(simd_dir)
            gen_archis_types(opts, simd_dir, platform, s)
            gen_archis_simd(opts, platform, s, simd_dir)
        else:
            print ('--   Extension excluded by command line')

def doit(opts):
    print ('-- Generating SIMD implementations')
    opts.platforms = common.get_platforms(opts)
    for p in opts.platforms:
        common.mkdir_p(os.path.join(opts.include_dir, p))
        gen_archis_platform(opts, p)
