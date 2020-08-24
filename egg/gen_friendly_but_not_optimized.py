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

import common
import operators
import os
from datetime import date
import sys

# -----------------------------------------------------------------------------
# Generate advanced C++ API

def get_impl(operator):
    if operator.params == ['v', 'v', 'v'] or \
       operator.params == ['l', 'v', 'v']:
        return \
        '''template <typename T, int N, typename SimdExt, typename S>
        pack{l}<T, N, SimdExt>
        operator{cxx_op}(pack<T, N, SimdExt> const &v, S s) {{
          return {op_name}(v, pack<T, N, SimdExt>(T(s)));
        }}

        template <typename S, typename T, int N, typename SimdExt>
        pack{l}<T, N, SimdExt>
        operator{cxx_op}(S s, pack<T, N, SimdExt> const &v) {{
          return {op_name}(pack<T, N, SimdExt>(T(s)), v);
        }}'''.format(l='l' if operator.params[0] == 'l' else '',
                     cxx_op=operator.cxx_operator, op_name=operator.name)
    if operator.params == ['l', 'l', 'l']:
        return \
        '''template <typename T, int N, typename SimdExt, typename S>
        packl<T, N, SimdExt>
        operator{cxx_op}(packl<T, N, SimdExt> const &v, S s) {{
          return {op_name}(v, packl<T, N, SimdExt>(bool(s)));
        }}

        template <typename S, typename T, int N, typename SimdExt>
        packl<T, N, SimdExt>
        operator{cxx_op}(S s, packl<T, N, SimdExt> const &v) {{
          return {op_name}(pack<T, N, SimdExt>(bool(s)), v);
        }}

        template <typename T, typename S, int N, typename SimdExt>
        packl<T, N, SimdExt> operator{cxx_op}(packl<T, N, SimdExt> const &v,
                                      packl<S, N, SimdExt> const &w) {{
          return {op_name}(v, reinterpretl<packl<T, N, SimdExt> >(w));
        }}'''.format(cxx_op=operator.cxx_operator, op_name=operator.name)

# -----------------------------------------------------------------------------
# Generate advanced C++ API

def doit(opts):
    common.myprint(opts, 'Generating friendly but not optimized advanced '
                         'C++ API')
    filename = os.path.join(opts.include_dir, 'friendly_but_not_optimized.hpp')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as out:
        out.write('''#ifndef NSIMD_FRIENDLY_BUT_NOT_OPTIMIZED_HPP
                     #define NSIMD_FRIENDLY_BUT_NOT_OPTIMIZED_HPP

                     #include <nsimd/nsimd.h>
                     #include <nsimd/cxx_adv_api.hpp>

                     namespace nsimd {{

                     '''.format(year=date.today().year))
        for op_name, operator in operators.operators.items():
            if operator.cxx_operator == None or len(operator.params) != 3 or \
               operator.name in ['shl', 'shr']:
                continue
            out.write('''{hbar}

                         {code}

                         '''.format(hbar=common.hbar, code=get_impl(operator)))
        out.write('''{hbar}

                     }} // namespace nsimd

                     #endif'''.format(hbar=common.hbar))
    common.clang_format(opts, filename)
