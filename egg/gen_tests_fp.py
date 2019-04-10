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
import sys
import common
import operators

# -----------------------------------------------------------------------------
# Entry point

def doit(opts):
    ulps = common.load_ulps_informations(opts)

    print ('-- Generating tests')
    for op_name, operator in operators.operators.items():
        ## Skip non-matching tests
        if opts.match and not opts.match.match(op_name):
            continue
        if op_name  in ['if_else1', 'loadu', 'loada', 'storeu', 'storea',
                        'len', 'loadlu', 'loadla', 'storelu', 'storela',
                        'set1', 'store2a', 'store2u', 'store3a', 'store3u',
                        'store4a', 'store4u']:
            continue
        for typ in operator.types:
            if operator.name in ['notb', 'andb', 'xorb', 'orb'] and \
               typ == 'f16':
                continue
            elif operator.name == 'nbtrue':
                gen_nbtrue(opts, operator, typ, 'c_base')
                gen_nbtrue(opts, operator, typ, 'cxx_base')
                gen_nbtrue(opts, operator, typ, 'cxx_adv')
            elif operator.name in ['all', 'any']:
                gen_all_any(opts, operator, typ, 'c_base')
                gen_all_any(opts, operator, typ, 'cxx_base')
                gen_all_any(opts, operator, typ, 'cxx_adv')
            elif operator.name in ['reinterpret', 'reinterpretl', 'cvt']:
                for to_typ in common.get_same_size_types(typ):
                    gen_reinterpret_convert(opts, operator, typ, to_typ,
                                            'c_base')
                    gen_reinterpret_convert(opts, operator, typ, to_typ,
                                            'cxx_base')
                    gen_reinterpret_convert(opts, operator, typ, to_typ,
                                            'cxx_adv')
            elif operator.name in ['load2a', 'load2u', 'load3a', 'load3u', 'load4a',
                              'load4u']:
                gen_load_store(opts, operator, typ, 'c_base')
                gen_load_store(opts, operator, typ, 'cxx_base')
                gen_load_store(opts, operator, typ, 'cxx_adv')
            elif operator.name == 'reverse':
                gen_reverse(opts, operator, typ, 'c_base');
                gen_reverse(opts, operator, typ, 'cxx_base');
                gen_reverse(opts, operator, typ, 'cxx_adv');
            else:
                gen_test(opts, operator, typ, 'c_base', ulps)
                gen_test(opts, operator, typ, 'cxx_base', ulps)
                gen_test(opts, operator, typ, 'cxx_adv', ulps)


