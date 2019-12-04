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
# Actual implementation

def get_cxx_advanced_generic(operator):
    def get_pack(param):
        return 'pack{}'.format(param[1:]) if param[0] == 'v' else 'packl'
    args_list = common.enum(operator.params[1:])
    inter = [i for i in ['v', 'l', 'vx2', 'vx3', 'vx4'] \
             if i in operator.params[1:]]
    need_tmpl_pack = get_pack(operator.params[0]) if inter == [] else None

    # Compute parameters passed to the base C++ API functions
    def var(arg, N):
        member = 'car' if N == '1' else 'cdr'
        if arg[1] in ['v', 'l']:
            return 'a{}.{}'.format(arg[0], member)
        elif (arg[1] in ['*', 'c*']) and N != '1':
            return 'a{} + len_'.format(arg[0])
        else:
            return 'a{}'.format(arg[0])
    vars1 = [var(i, '1') for i in args_list] + ['T()'] + \
            (['typename ToPackType::value_type()'] if not operator.closed \
             else []) + ['SimdExt()']
    varsN = [var(i, 'N') for i in args_list]
    other_varsN = ', '.join(['a{}'.format(i[0]) for i in args_list])
    if not operator.closed:
        varsN = ['typename ToPackType::value_type()'] + varsN
    if need_tmpl_pack != None:
        varsN = ['{}<T, N - 1, SimdExt>()'.format(need_tmpl_pack)] + varsN
    vars1 = ', '.join(vars1)
    varsN = ', '.join(varsN)

    # Compute return type
    ret1 = 'ToPackType' if not operator.closed \
           else common.get_one_type_generic_adv_cxx(operator.params[0],
                                                    'T', '1')
    retN = 'ToPackType' if not operator.closed \
           else common.get_one_type_generic_adv_cxx(operator.params[0],
                                                            'T', 'N')

    # Dump C++
    if operator.params[0] in ['v', 'l']:
        return_ret = 'return ret;'
        ret_car = 'ret.car = '
        ret_cdr = 'ret.cdr = '
        post_car = ''
        post_cdr = ''
        pack1_ret = '{} ret;'.format(ret1)
        packN_ret = '{} ret;'.format(retN)
    elif operator.params[0] in ['vx2', 'vx3', 'vx4']:
        num = operator.params[0][-1:]
        return_ret = 'return ret;'
        if operator.closed:
            ret_car = \
                'typename simd_traits<T, SimdExt>::simd_vectorx{} car = '. \
                format(num)
        else:
            ret_car = \
                '''typename simd_traits<typename ToPackType::value_type,
                       SimdExt>::simd_vectorx{} car = '''.format(num)
        ret_cdr = 'packx{}<T, N - 1, SimdExt> cdr = '.format(num)
        post_car = '; ret.set_car({})'.format(', '.join( \
            ['car.v{}'.format(i) for i in range(0, int(num))]))
        post_cdr = '; ret.set_cdr({})'.format(', '.join( \
            ['cdr.v{}'.format(i) for i in range(0, int(num))]))
        pack1_ret = '{} ret;'.format(ret1)
        packN_ret = '{} ret;'.format(retN)
    else:
        return_ret = ''
        ret_car = ''
        ret_cdr = ''
        post_car = ''
        post_cdr = ''
        pack1_ret = ''
        packN_ret = ''
    if '*' in operator.params[1:] or 'c*' in operator.params[1:]:
        # store*[au] does not contain any packx* argument, therefore the offset
        # cannot be correctly computed
        if operator.name in ['store2u', 'store2a']:
            multiplier = '2 * '
        elif operator.name in ['store3u', 'store3a']:
            multiplier = '3 * '
        elif operator.name in ['store4u', 'store4a']:
            multiplier = '4 * '
        else:
            multiplier = ''
        int_len = 'int len_ = {}len({}<T, 1, SimdExt>());'. \
                  format(multiplier, get_pack(inter[0]) if inter != [] \
                                     else need_tmpl_pack)
    else:
        int_len = ''

    sig = operator.get_generic_signature('cxx_adv')
    for k in sig:
        sig[k] = sig[k][:-1] # remove trailing ';'

    tmpl = '''{{sig1}} {{{{{pack1_ret}
                {ret_car}{name}({vars1}){post_car};
              {return_ret}}}}}

              {{sigN}} {{{{{packN_ret}{int_len}
                {ret_car}{name}({vars1}){post_car};
                {ret_cdr}{{cxx_name}}({varsN}){post_cdr};
              {return_ret}}}}}'''. \
	      format(pack1_ret=pack1_ret, ret_car=ret_car, name=operator.name,
	             vars1=vars1, return_ret=return_ret, retN=retN,
	             packN_ret=packN_ret, int_len=int_len, ret_cdr=ret_cdr,
	             varsN=varsN, post_car=post_car, post_cdr=post_cdr)

    ret = ''
    if operator.cxx_operator:
        ret += tmpl.format(cxx_name=operator.cxx_operator,
                           sig1=sig['op1'], sigN=sig['opN']) + '\n\n'
    ret += tmpl.format(cxx_name=operator.name,
                       sig1=sig['1'], sigN=sig['N']) + '\n\n'

    if not operator.closed:
        return_ins = 'return ' if operator.params[0] != '_' else ''
        ret += '\n\n'
        ret += '''{sig} {{
                    {return_ins} {cxx_name}(ToPackType(), {other_varsN});
                  }}'''. \
                  format(cxx_name=operator.name, sig=sig['dispatch'],
                         other_varsN=other_varsN, return_ins=return_ins)
    if need_tmpl_pack != None:
        ret += '\n\n'
        ret += '''{sig} {{
                    return {cxx_name}(SimdVector(), {other_varsN});
                  }}'''. \
                  format(sig=sig['dispatch'], cxx_name=operator.name,
                         other_varsN=other_varsN)
    return ret

# -----------------------------------------------------------------------------
# Generate advanced C++ API

def doit(opts):
    print ('-- Generating advanced C++ API')
    filename = os.path.join(opts.include_dir, 'cxx_adv_api_functions.hpp')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as out:
        out.write('''#ifndef NSIMD_CXX_ADV_API_FUNCTIONS_HPP
                     #define NSIMD_CXX_ADV_API_FUNCTIONS_HPP

                     namespace nsimd {{

                     '''.format(year=date.today().year))

        for op_name, operator in operators.operators.items():
            if not operator.autogen_cxx_adv:
                continue

            out.write('''{hbar}

                         {code}

                         '''.format(hbar=common.hbar,
                                    code=get_cxx_advanced_generic(operator)))


        out.write('''{hbar}

                     }} // namespace nsimd

                     #endif'''.format(hbar=common.hbar))
    common.clang_format(opts, filename)
