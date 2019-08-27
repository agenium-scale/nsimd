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
    args_list = common.enum(operator.params[1:])
    tmpl_args = []
    if not operator.closed:
        tmpl_args += ['typename ToType']
    tmpl_args += ['typename T']
    tmpl_args += ['typename A{}'.format(i[0]) for i in args_list \
                  if i[1] not in ['v', 'l']]
    tmpl_args = ', '.join(tmpl_args)
    def arg_type(arg):
        if arg[1] in ['v', 'l']:
            return 'T a{}'.format(arg[0])
        else:
            return 'A{i} a{i}'.format(i=arg[0])
    args = [arg_type(i) for i in args_list]
    varis = ['a{}'.format(a[0]) for a in args_list]
    varis += ['ToType()', 'T()'] if not operator.closed else ['T()']
    varis = ', '.join(varis)
    returns = '' if operator.params[0] == '_' else 'return '
    return_typ = 'ToType' if not operator.closed else 'T'
    template = '''template <{tmpl_args}>
                  {return_typ} {op_name}({{args}}) {{{{
                    {returns}{op_name}({varis}, cpu());
                  }}}}'''. \
                  format(tmpl_args=tmpl_args, return_typ=return_typ,
                         op_name=operator.name, args=', '.join(args),
                         returns=returns, varis=varis)
    ret = template.format(args=', '.join(args))
    if not operator.closed:
        args = ['ToType'] + args
        ret += '\n\n' + template.format(args=', '.join(args))
    if operator.cxx_operator != '' and operator.cxx_operator != None and \
       (operator.params == ['v', 'v', 'v'] or \
        operator.params == ['l', 'v', 'v']):
        ret += \
        '''

        template <typename T, int N, typename SimdExt, typename S>
        pack{l}<T, N, SimdExt> {cxx_op}(pack<T, N, SimdExt> const &v, S s) {{
          return {op_name}(v, pack<T, N, SimdExt>(T(s)));
        }}

        template <typename S, typename T, int N, typename SimdExt>
        pack{l}<T, N, SimdExt> {cxx_op}(S s, pack<T, N, SimdExt> const &v) {{
          return {op_name}(pack<T, N, SimdExt>(T(s)), v);
        }}'''.format(l='l' if operator.params[0] == 'l' else '',
                     cxx_op=operator.cxx_operator, op_name=operator.name)

    return ret


# -----------------------------------------------------------------------------
# Generate advanced C++ API

def doit(opts):
    print ('-- Generating friendly but not optimized advanced C++ API')
    filename = os.path.join(opts.include_dir, 'friendly_but_not_optimized.hpp')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(filename) as out:
        out.write('''#ifndef NSIMD_FRIENDLY_BUT_NOT_OPTIMIZED_HPP
                     #define NSIMD_FRIENDLY_BUT_NOT_OPTIMIZED_HPP

                     #include <nsimd/nsimd.h>
                     #include <nsimd/cxx_adv_api.hpp>

                     namespace nsimd {{

                     '''.format(year=date.today().year))
        for op_name, operator in operators.operators.items():
            if operator.load_store or op_name == 'set1':
                continue
            out.write('''{hbar}

                         {code}

                         '''.format(hbar=common.hbar, code=get_impl(operator)))
        out.write('''{hbar}

                     }} // namespace nsimd

                     #endif'''.format(hbar=common.hbar))
    common.clang_format(opts, filename)
