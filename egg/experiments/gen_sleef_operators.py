# Copyright (c) 2021 Agenium Scale
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

script_dir = os.path.dirname(os.path.realpath(__file__))
sleef_dir = os.path.join(script_dir, '..', '..', '_deps-sleef')
sleef_version = '3.5.1'

funcproto = os.path.join(sleef_dir, 'sleef-{}'.format(sleef_version),
                         'src', 'libm', 'funcproto.h')

ulp_suffix = {
    '0' : '',
    '1' : '_u1',
    '2' : '_u05',
    '3' : '_u35',
    '4' : '_u15',
    '5' : '_u3500'
}

func_type = {
    '0' : 'v {} v',
    '1' : 'v {} v v',
    '2' : 'vx2 {} v',
    '3' : 'v {} v p',
    '4' : 'v {} v',
    '5' : 'v {} v v v',
    '6' : 'vx2 {} v',
    '7' : 'p {} p',
    '8' : '* {} p'
}

props = {
    'cos' : ['cosine', 'DocTrigo', 'R'],
    'sin' : ['sine', 'DocTrigo', 'R'],
    'fastcos' : ['cosine', 'DocTrigo', 'R'],
    'fastsin' : ['sine', 'DocTrigo', 'R'],
    'cospi' : ['cosine of multiple of pi argument', 'DocTrigo', 'R'],
    'sinpi' : ['sine of multiple of pi argument', 'DocTrigo', 'R'],
    'tan' : ['tangent', 'DocTrigo', 'R\{(z+0.5)*pi}'],
    'acos' : ['arc cosine', 'DocTrigo', '(-1,1)'],
    'asin' : ['arc sine', 'DocTrigo', '(-1,1)'],
    'atan' : ['arc tangent', 'DocTrigo', 'R'],
    'atan2' : ['arc tangent', 'DocTrigo', 'RxR'],

    'log' : ['natural logarithmic', 'DocExpLog', '(0,Inf)'],
    'log2' : ['base-2 logarithmic', 'DocExpLog', '(0,Inf)'],
    'log10' : ['base-10 logarithmic', 'DocExpLog', '(0,Inf)'],
    'log1p' : ['logarithm of one plus argument', 'DocExpLog', '(-1,Inf)'],
    'exp' : ['exponential', 'DocExpLog', 'R'],
    'exp2' : ['base-2 exponential', 'DocExpLog', 'R'],
    'exp10' : ['base-10 exponential', 'DocExpLog', 'R'],
    'expm1' : ['exponential minus 1', 'DocExpLog', 'R'],
    'pow' : ['power', 'DocExpLog', 'RxR'],
    'fastpow' : ['power', 'DocExpLog', 'RxR'],

    'cbrt' : ['cubic root', 'DocBasicArithmetic', 'R'],
    'hypot' : ['hypotenuse', 'DocBasicArithmetic', 'RxR'],

    'sinh': ['hyperbolic sine', 'DocHyper', 'R'],
    'cosh': ['hyperbolic cosine', 'DocHyper', 'R'],
    'tanh': ['hyperbolic tangent', 'DocHyper', 'R'],
    'asinh': ['hyperbolic arc sine', 'DocHyper', 'R'],
    'acosh': ['hyperbolic arc cosine', 'DocHyper', '(1,Inf)'],
    'atanh': ['hyperbolic arc tangent', 'DocHyper', '(-1,1)'],

    'lgamma' : ['log gamma', 'DocMisc', 'R\{-n}'],
    'tgamma' : ['gamma', 'DocMisc', 'R\{-n}'],
    'erf' : ['error function', 'DocMisc', 'R'],
    'erfc' : ['complementary error function', 'DocMisc', 'R']
}

with open(funcproto, 'r') as fin:
    for line in fin:
        if not (line.find('{') != -1 and line.find('}') != -1):
            continue
        items = [item.strip() for item in line.strip(' \n\r{},').split(',')]
        items[0] = items[0].strip('"')
        if items[0] == 'NULL':
            break
        if items[0] not in props:
            continue
        name = items[0] + '_u' + items[1]
        symbol = 'nsimd_sleef_{}'.format(name)
        prop = props[items[0]]
        print('Class {}{}(SrcOperator):'. \
              format(name[0].upper(), name[1:]))
        print('  full_name = \'{}\''.format(prop[0]))
        print('  signature = \'{}\''.format(func_type[items[3]]) \
                                    .format(name))
        print('  sleef_symbol_prefix = \'{}\''.format(symbol))
        print('  domain = Domain(\'{}\')'.format(prop[2]))
        print('  categories = [{}]'.format(prop[1]))
        print('  desc = \'Compute the {} of its argument{} with ' \
                 'a precision of {} ulps. For more informations visit ' \
                 '<https://sleef.org/purec.xhtml>.\''.format(prop[0],
                 's' if items[3] in ['1', '3', '5'] else '',
                 float(items[1]) / 10.0))
        print('')
