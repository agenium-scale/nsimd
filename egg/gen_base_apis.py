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
# C base generic implem

def get_c_base_generic(operator):
    vas = common.get_args(len(operator.params) - 1)
    sig = operator.get_generic_signature('c_base')
    if not operator.closed:
        return \
        '''{sig} NSIMD_PP_CAT_6(nsimd_{name}_, NSIMD_SIMD, _, \\
                                to_type, _, from_type)({vas})

           {sig_e} NSIMD_PP_CAT_6(nsimd_{name}_, simd_ext, _, \\
                                  to_type, _, from_type)({vas})'''. \
           format(sig=sig[0], sig_e=sig[1], name=operator.name, vas=vas)
    else:
        return \
        '''{sig} NSIMD_PP_CAT_4(nsimd_{name}_, NSIMD_SIMD, _, type)({vas})

           {sig_e} NSIMD_PP_CAT_4(nsimd_{name}_, simd_ext, _, type)({vas})'''. \
           format(sig=sig[0], sig_e=sig[1], name=operator.name, vas=vas)

# -----------------------------------------------------------------------------
# C++ base generic implem

def get_cxx_base_generic(operator):
    returns = '' if operator.params[0] == '_' else 'return'
    temp = common.get_args(len(operator.params) - 1)
    temp += ', ' if temp != '' else ''
    args = temp + 'F(), T()' if not operator.closed else temp + 'T()'
    return \
    '''#if NSIMD_CXX > 0
       namespace nsimd {{
       {sig} {{
         {returns} {name}({args}, NSIMD_SIMD());
       }}
       }} // namespace nsimd
       #endif'''.format(name=operator.name, args=args, returns=returns,
                        sig=operator.get_generic_signature('cxx_base')[:-1])

# -----------------------------------------------------------------------------
# Declarations for output

def get_put_decl():
    return \
    '''#include NSIMD_AUTO_INCLUDE(put.h)

       #define vput(out, fmt, a0, type) \
           NSIMD_PP_CAT_4(nsimd_put_, NSIMD_SIMD, _, type)(out, fmt, a0)

       #define vput_e(out, fmt, a0, type, simd_ext) \
           NSIMD_PP_CAT_4(nsimd_put_, simd_ext, _, type)(out, fmt, a0)

       #if NSIMD_CXX > 0
       namespace nsimd {
       template <typename A0, typename T>
       int put(FILE *out, const char *fmt, A0 a0, T) {
         return put(out, fmt, a0, T(), NSIMD_SIMD());
       }
       } // namespace nsimd
       #endif
       '''

# -----------------------------------------------------------------------------
# Generate base APIs

def doit(opts):
    print ('-- Generating base APIs')
    common.mkdir_p(opts.include_dir)
    filename = os.path.join(opts.include_dir, 'functions.h')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as out:
        out.write('''#ifndef NSIMD_FUNCTIONS_H
                     #define NSIMD_FUNCTIONS_H

                     '''.format(year=date.today().year))

        for op_name, operator in operators.operators.items():
            out.write('''{}

                         #include NSIMD_AUTO_INCLUDE({}.h)

                         {}

                         {}

                         '''.format(common.hbar, operator.name,
                                    get_c_base_generic(operator),
                                    get_cxx_base_generic(operator)))

        out.write('''{hbar}

                     {put_decl}

                     {hbar}

                     #endif'''. \
                     format(hbar=common.hbar, put_decl=get_put_decl()))
    common.clang_format(opts, filename)
