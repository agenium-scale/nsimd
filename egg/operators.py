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

if __name__ == 'operators':
    import common
    from common import Domain
else:
    from . import common
    from .common import Domain
import collections

# -----------------------------------------------------------------------------
# Metaclass and class to gather all operator categories

categories = collections.OrderedDict()

class MAddToCategories(type):
    def __new__(cls, name, bases, dct):
        if name != 'DocCategory':
            if 'title' not in dct:
                raise Exception('No member title provided for class {}'. \
                                format(name))
            dct['name'] = name
            dct['id'] = '/categories/{}'.format(name)
        ret = type.__new__(cls, name, bases, dct)
        if name != 'DocCategory':
            categories[name] = ret()
        return ret

class DocCategory(object, metaclass=MAddToCategories):
    pass

# -----------------------------------------------------------------------------
# Operators categories

class DocShuffle(DocCategory):
    title = 'Shuffle functions'

class DocTrigo(DocCategory):
    title = 'Trigonometric functions'

class DocHyper(DocCategory):
    title = 'Hyperbolic functions'

class DocExpLog(DocCategory):
    title = 'Exponential and logarithmic functions'

class DocBasicArithmetic(DocCategory):
    title = 'Basic arithmetic operators'

class DocBitsOperators(DocCategory):
    title = 'Bits manipulation operators'

class DocLogicalOperators(DocCategory):
    title = 'Logicals operators'

class DocMisc(DocCategory):
    title = 'Miscellaneous'

class DocLoadStore(DocCategory):
    title = 'Loads & stores'

class DocComparison(DocCategory):
    title = 'Comparison operators'

class DocRounding(DocCategory):
    title = 'Rounding functions'

class DocConversion(DocCategory):
    title = 'Conversion operators'

# -----------------------------------------------------------------------------
# Metaclass and class to gather all operators

operators = collections.OrderedDict()

class MAddToOperators(type):
    def __new__(cls, name, bases, dct):
        # We don't care about the parent class
        if name == 'Operator' or name == 'SrcOperator':
            return type.__new__(cls, name, bases, dct)

        # Mandatory members
        mm = ['categories', 'signature']
        for m in mm:
            if m not in dct:
                raise Exception('Mandatory member "{}" not given in "{}"'. \
                                format(m, name))

        # Check that all items in categories exists
        for c in dct['categories']:
            if type(c) == str:
                raise Exception( \
                      'Category "{}" must not be a string for operator "{}"'. \
                      format(c, name))
            if not hasattr(c, 'name'):
                raise Exception( \
                      'Category "{}" does not exist for operator "{}"'. \
                      format(c.__class__.__name__, name))
            if c.name not in categories:
                raise Exception( \
                      'Category "{}" does not exist for operator "{}"'. \
                      format(c.__class__.__name__, name))

        # Some defaults, that are fixed by the implementation
        (dct['name'], dct['params']) = common.parse_signature(dct['signature'])
        if 'output_to' in dct:
            if dct['output_to'] == common.OUTPUT_TO_SAME_TYPE:
                dct['closed'] = True
            else:
                dct['closed'] = False
        else:
            dct['closed'] = True
            dct['output_to'] = common.OUTPUT_TO_SAME_TYPE

        # If the operator takes as inputs vectors and returns a scalar, then
        # by default we cannot autogenerate the C++ advanced API because we
        # cannot guess how to combine pieces of a unrolled pack
        if 'autogen_cxx_adv' not in dct:
            if dct['params'][0] in ['p', 's']:
                dct['autogen_cxx_adv'] = False
            else:
                dct['autogen_cxx_adv'] = True

        # Fill domain, default is R
        if 'domain' not in dct:
            dct['domain'] = Domain('R')

        # Check that params is not empty
        if len(dct['params']) == 0:
            raise Exception('"params" is empty for operator "{}"'. \
                            format(name))

        # Fill full_name, default is same as name
        if 'full_name' not in dct:
            dct['full_name'] = name

        # Fill desc, default is a basic sentence using full_name
        if 'desc' not in dct:
            arg = 'arguments' if len(dct['params']) > 2 else 'argument'
            if dct['params'][0] == '_':
                dct['desc'] = \
                '{} the {}. Defined over {}.'. \
                format(dct['full_name'].capitalize(), arg, dct['domain'])
            else:
                dct['desc'] = \
                'Returns the {} of the {}. Defined over {}.'.\
                format(dct['full_name'], arg, dct['domain'])

        ret = type.__new__(cls, name, bases, dct)
        operators[dct['name']] = ret()
        return ret

class Operator(object, metaclass=MAddToOperators):

    # Default values (for general purpose)
    domain = Domain('R')
    cxx_operator = None
    autogen_cxx_adv = True
    output_to = common.OUTPUT_TO_SAME_TYPE
    src = False
    load_store = False
    types = common.types
    params = []
    signature = ''

    # Enable bench by default
    do_bench = True

    # Default values (for documentation)
    desc = ''

    # Defaults values (for benches)
    returns_any_type = False
    bench_auto_against_cpu = True
    bench_auto_against_mipp = False
    bench_auto_against_sleef = False
    bench_auto_against_std = False
    use_for_parsing = True

    # Defaults values (for tests)
    tests_mpfr = False
    tests_ulps = None

    @property
    def returns(self):
        return self.params[0]

    @property
    def args(self):
        return self.params[1:]

    @property
    def cxx_operator_symbol(self):
        assert self.cxx_operator is not None
        return self.cxx_operator.replace('operator', '')

    def __init__(self):
        (self.name, self.params) = common.parse_signature(self.signature)
        super(Operator, self).__init__()

    def get_return(self):
        return self.params[0]

    def tests_mpfr_name(self):
        return 'mpfr_' + self.name

    def bench_mipp_name(self, typ):
        return 'mipp::{}<{}>'.format(self.name, typ)

    def bench_mipp_types(self):
        return common.ftypes_no_f16

    def bench_sleef_name(self, simd, typ):
        return common.sleef_name(self.name, simd, typ)

    def bench_sleef_types(self):
        return common.ftypes_no_f16

    def bench_std_name(self, simd, typ):
        return 'std::{}'.format(self.name)

    def bench_std_types(self):
        return self.types

    # TODO: move to gen_archis.py
    def get_header_guard(self, platform, simd_ext):
        return 'NSIMD_{}_{}_{}_H'.format(platform.upper(),
            simd_ext.upper(), self.name.upper())

    def get_fmtspec(self, t, tt, simd_ext):
        ret = {}
        return_typ = common.get_one_type_specific(self.params[0], simd_ext, tt)
        ret['return_typ'] = return_typ
        ret['returns'] = '' if return_typ == 'void' else 'return '
        args_list = common.enum([common.get_one_type_specific(p, simd_ext, t)
                                 for p in self.params[1:]])
        if len(args_list) > 0:
            ret['c_args'] = ', '.join(['{} a{}'.format(i[1], i[0])
                                       for i in args_list])
            ret['cxx_args'] = ret['c_args'] + ', '
        else:
            ret['c_args'] = 'void'
            ret['cxx_args'] = ''
        if self.closed:
            ret['cxx_args'] += '{}, {}'.format(t, simd_ext)
        else:
            ret['cxx_args'] += '{}, {}, {}'.format(t, tt, simd_ext)
        ret['vas'] = ', '.join(['a{}'.format(i[0]) for i in args_list])
        ret['suf'] = tt if self.closed else '{}_{}'.format(tt, t)
        ret['name'] = self.name
        ret['hbar'] = common.hbar
        ret['simd_ext'] = simd_ext
        return ret

    def get_generic_signature(self, lang):
        if lang == 'c_base':
            vas = common.get_args(len(self.params) - 1)
            args = vas + (', ' if vas != '' else '')
            args += 'from_type, to_type' if not self.closed else 'type'
            return ['#define v{name}({args})'.format(name=self.name,
                    args=args),
                    '#define v{name}_e({args}, simd_ext)'. \
                    format(name=self.name, args=args)]
        elif lang == 'cxx_base':
            return_typ = common.get_one_type_generic(self.params[0], 'T')
            if return_typ.startswith('vT'):
                return_typ = \
                'typename simd_traits<T, NSIMD_SIMD>::simd_vector{}'. \
                format(return_typ[2:])
            elif return_typ == 'vlT':
                return_typ = \
                'typename simd_traits<T, NSIMD_SIMD>::simd_vectorl'
            args_list = common.enum(self.params[1:])

            temp = ', '.join(['typename A{}'.format(a[0]) for a in args_list])
            temp += ', ' if temp != '' else ''
            if not self.closed :
                tmpl_args = temp + 'typename F, typename T'
            else:
                tmpl_args = temp + 'typename T'

            temp = ', '.join(['A{i} a{i}'.format(i=a[0]) for a in args_list])
            temp += ', ' if temp != '' else ''
            if not self.closed :
                func_args = temp + 'F, T'
            else:
                func_args = temp + 'T'

            return \
            'template <{tmpl_args}> {return_typ} {name}({func_args});'. \
            format(return_typ=return_typ, tmpl_args=tmpl_args,
                   func_args=func_args, name=self.name)
        elif lang == 'cxx_adv':
            def get_pack(param):
                return 'pack{}'.format(param[1:]) if param[0] == 'v' \
                                                  else 'packl'
            args_list = common.enum(self.params[1:])
            inter = [i for i in ['v', 'l', 'vx2', 'vx3', 'vx4'] \
                     if i in self.params[1:]]
            # Do we need tag dispatching on pack<>? e.g. len, set1 and load*
            need_tmpl_pack = get_pack(self.params[0]) if inter == [] else None

            # Compute template arguments
            tmpl_args = []
            if not self.closed:
                tmpl_args += ['typename ToPackType']
            tmpl_args1 = tmpl_args + ['typename T', 'typename SimdExt']
            tmpl_argsN = tmpl_args + ['typename T', 'int N', 'typename SimdExt']
            other_tmpl_args = ['typename A{}'.format(i[0]) for i in args_list \
                               if i[1] not in ['v', 'l']]
            tmpl_args1 += other_tmpl_args
            tmpl_argsN += other_tmpl_args
            tmpl_args1 = ', '.join(tmpl_args1)
            tmpl_argsN = ', '.join(tmpl_argsN)

            # Compute function arguments
            def arg_type(arg, N):
                if arg[1] in ['v', 'l']:
                    pack_typ = 'pack' if arg[1] == 'v' else 'packl'
                    return '{}<T, {}, SimdExt> const&'.format(pack_typ, N)
                else:
                    return 'A{}'.format(arg[0])
            args1 = ['{} a{}'.format(arg_type(i, '1'), i[0]) for i in args_list]
            argsN = ['{} a{}'.format(arg_type(i, 'N'), i[0]) for i in args_list]
            # Arguments without tag dispatching on pack
            other_argsN = ', '.join(argsN)
            if not self.closed:
                args1 = ['ToPackType'] + args1
                argsN = ['ToPackType'] + argsN
            if need_tmpl_pack != None:
                args1 = ['{}<T, 1, SimdExt> const&'.format(need_tmpl_pack)] + \
                        args1
                argsN = ['{}<T, N, SimdExt> const&'.format(need_tmpl_pack)] + \
                        argsN
            args1 = ', '.join(args1)
            argsN = ', '.join(argsN)

            # Compute return type
            ret1 = 'ToPackType' if not self.closed \
                   else common.get_one_type_generic_adv_cxx(self.params[0],
                                                            'T', '1')
            retN = 'ToPackType' if not self.closed \
                   else common.get_one_type_generic_adv_cxx(self.params[0],
                                                            'T', 'N')

            ret = { \
                '1': 'template <{tmpl_args1}> {ret1} {cxx_name}({args1});'. \
                     format(tmpl_args1=tmpl_args1, ret1=ret1, args1=args1,
                            cxx_name=self.name),
                'N': 'template <{tmpl_argsN}> {retN} {cxx_name}({argsN});'. \
                     format(tmpl_argsN=tmpl_argsN, retN=retN, argsN=argsN,
                            cxx_name=self.name)
            }
            if self.cxx_operator:
                ret.update({ \
                    'op1':
                    'template <{tmpl_args1}> {ret1} {cxx_name}({args1});'. \
                    format(tmpl_args1=tmpl_args1, ret1=ret1, args1=args1,
                           cxx_name=self.cxx_operator),
                    'opN':
                    'template <{tmpl_argsN}> {retN} {cxx_name}({argsN});'. \
                    format(tmpl_argsN=tmpl_argsN, retN=retN, argsN=argsN,
                           cxx_name=self.cxx_operator)
                })
            if not self.closed:
                ret['dispatch'] = \
                'template <{tmpl_argsN}> {retN} {cxx_name}({other_argsN});'. \
                format(tmpl_argsN=tmpl_argsN, other_argsN=other_argsN,
                       retN=retN, cxx_name=self.name)
            elif need_tmpl_pack != None:
                other_tmpl_args = ', '.join(['typename SimdVector'] + \
                                            other_tmpl_args)
                ret['dispatch'] = \
                '''template <{other_tmpl_args}>
                   SimdVector {cxx_name}({other_argsN});'''. \
                   format(other_tmpl_args=other_tmpl_args,
                          other_argsN=other_argsN, cxx_name=self.name)
            return ret
        else:
            raise Exception('Lang must be one of c_base, cxx_base, cxx_adv')

    def get_signature(self, typename, lang, simd_ext):
        # Check that the type is available for this function
        if typename not in self.types:
            raise Exception('Type {} not supported for function {}'. \
                            format(typename, self.name))

        fmtspec = self.get_fmtspec(typename, typename, simd_ext)

        if lang == 'c_base':
            sig = '{return_typ} nsimd_{name}_{simd_ext}_{suf}({c_args})'. \
                  format(**fmtspec)
        elif lang == 'cxx_base':
            sig = '{return_typ} {name}({cxx_args})'.format(**fmtspec)
        elif lang == 'cxx_adv':
            sig = ''
            raise Exception('TODO cxx_adv for {}'.format(lang))
        else:
            raise Exception('Unknown langage {}'.format(lang))

        return sig

class SrcOperator(Operator):
    src = True
    types = common.ftypes

# -----------------------------------------------------------------------------
# List of functions/operators

class Len(Operator):
    signature = 'p len'
    domain = Domain('')
    desc = 'Returns the length of the nsimd vector.'
    categories = [DocMisc]

class Set1(Operator):
    signature = 'v set1 s'
    categories = [DocMisc]
    desc = 'Set all elements in the vector to the given value.'

class Loadu(Operator):
    full_name = 'loadu'
    signature = 'v loadu c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from unaligned memory.'

class Load2u(Operator):
    full_name = 'load2u'
    signature = 'vx2 load2u c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 2 members from unaligned memory.'

class Load3u(Operator):
    full_name = 'load3u'
    signature = 'vx3 load3u c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 3 members from unaligned memory.'

class Load4u(Operator):
    full_name = 'load4u'
    signature = 'vx4 load4u c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 4 members from unaligned memory.'

class Loada(Operator):
    full_name = 'loada'
    signature = 'v loada c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from aligned memory.'

class Load2a(Operator):
    full_name = 'load2a'
    signature = 'vx2 load2a c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 2 members from aligned memory.'

class Load3a(Operator):
    full_name = 'load3a'
    signature = 'vx3 load3a c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 3 members from aligned memory.'

class Load4a(Operator):
    full_name = 'load4a'
    signature = 'vx4 load4a c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 4 members from aligned memory.'

class Loadlu(Operator):
    full_name = 'loadlu'
    signature = 'l loadlu c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from unaligned memory and interpret it as booleans. ' + \
           'Zero is interpreted as False and nonzero as True.'

class Loadla(Operator):
    full_name = 'loadla'
    signature = 'l loadla c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from aligned memory and interpret it as booleans. ' + \
           'Zero is interpreted as False and nonzero as True.'

class Storeu(Operator):
    full_name = 'storeu'
    signature = '_ storeu * v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store SIMD vector into unaligned memory.'

class Store2u(Operator):
    full_name = 'store2u'
    signature = '_ store2u * v v'
    load_store = True
    domain = Domain('RxR')
    categories = [DocLoadStore]
    desc = 'Store 2 SIMD vectors as array of structures of 2 members into ' + \
           'unaligned memory.'

class Store3u(Operator):
    full_name = 'store3u'
    signature = '_ store3u * v v v'
    load_store = True
    domain = Domain('RxRxR')
    categories = [DocLoadStore]
    desc = 'Store 3 SIMD vectors as array of structures of 3 members into ' + \
           'unaligned memory.'

class Store4u(Operator):
    full_name = 'store4u'
    signature = '_ store4u * v v v v'
    load_store = True
    domain = Domain('RxRxRxR')
    categories = [DocLoadStore]
    desc = 'Store 4 SIMD vectors as array of structures of 4 members into ' + \
           'unaligned memory.'

class Storea(Operator):
    full_name = 'storea'
    signature = '_ storea * v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store SIMD vector into aligned memory.'

class Store2a(Operator):
    full_name = 'store2a'
    signature = '_ store2a * v v'
    load_store = True
    domain = Domain('RxR')
    categories = [DocLoadStore]
    desc = 'Store 2 SIMD vectors as array of structures of 2 members into ' + \
           'aligned memory.'

class Store3a(Operator):
    full_name = 'store3a'
    signature = '_ store3a * v v v'
    load_store = True
    domain = Domain('RxRxR')
    categories = [DocLoadStore]
    desc = 'Store 3 SIMD vectors as array of structures of 3 members into ' + \
           'aligned memory.'

class Store4a(Operator):
    full_name = 'store4a'
    signature = '_ store4a * v v v v'
    load_store = True
    domain = Domain('RxRxRxR')
    categories = [DocLoadStore]
    desc = 'Store 4 SIMD vectors as array of structures of 4 members into ' + \
           'aligned memory.'

class Storelu(Operator):
    full_name = 'storelu'
    signature = '_ storelu * l'
    load_store = True
    categories = [DocLoadStore]
    domain = Domain('R')
    desc = 'Store SIMD vector of booleans into unaligned memory. True is ' + \
           'stored as 1 and False as 0.'

class Storela(Operator):
    full_name = 'storela'
    signature = '_ storela * l'
    load_store = True
    categories = [DocLoadStore]
    domain = Domain('R')
    desc = 'Store SIMD vector of booleans into aligned memory. True is ' + \
           'stored as 1 and False as 0.'

class Orb(Operator):
    full_name = 'bitwise or'
    signature = 'v orb v v'
    cxx_operator = 'operator|'
    domain = Domain('RxR')
    categories = [DocBitsOperators]
    #bench_auto_against_std = True ## TODO: Add check to floating-types
    bench_auto_against_mipp = True

class Andb(Operator):
    full_name = 'bitwise and'
    signature = 'v andb v v'
    cxx_operator = 'operator&'
    domain = Domain('RxR')
    categories = [DocBitsOperators]
    #bench_auto_against_std = True ## TODO: Add check to floating-types
    bench_auto_against_mipp = True

class Andnotb(Operator):
    full_name = 'bitwise and not'
    signature = 'v andnotb v v'
    domain = Domain('RxR')
    categories = [DocBitsOperators]
    bench_auto_against_mipp = True

    def bench_mipp_name(self, typ):
        return 'mipp::andnb<{}>'.format(typ)

class Notb(Operator):
    full_name = 'bitwise not'
    signature = 'v notb v'
    cxx_operator = 'operator~'
    domain = Domain('R')
    categories = [DocBitsOperators]
    #bench_auto_against_std = True ## TODO: Add check to floating-types
    bench_auto_against_mipp = True

class Xorb(Operator):
    full_name = 'bitwise xor'
    signature = 'v xorb v v'
    cxx_operator = 'operator^'
    domain = Domain('RxR')
    categories = [DocBitsOperators]
    #bench_auto_against_std = True ## TODO: Add check to floating-types
    bench_auto_against_mipp = True

class Orl(Operator):
    full_name = 'logical or'
    signature = 'l orl l l'
    cxx_operator = 'operator||'
    domain = Domain('BxB')
    categories = [DocLogicalOperators]
    bench_auto_against_std = True

class Andl(Operator):
    full_name = 'logical and'
    signature = 'l andl l l'
    cxx_operator = 'operator&&'
    domain = Domain('BxB')
    categories = [DocLogicalOperators]
    bench_auto_against_std = True

class Andnotl(Operator):
    full_name = 'logical and not'
    signature = 'l andnotl l l'
    domain = Domain('BxB')
    categories = [DocLogicalOperators]

class Xorl(Operator):
    full_name = 'logical xor'
    signature = 'l xorl l l'
    domain = Domain('BxB')
    categories = [DocLogicalOperators]

class Notl(Operator):
    full_name = 'logical not'
    signature = 'l notl l'
    cxx_operator = 'operator!'
    domain = Domain('B')
    categories = [DocLogicalOperators]
    bench_auto_against_std = True

class Add(Operator):
    full_name = 'addition'
    signature = 'v add v v'
    cxx_operator = 'operator+'
    domain = Domain('RxR')
    categories = [DocBasicArithmetic]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

class Sub(Operator):
    full_name = 'subtraction'
    signature = 'v sub v v'
    cxx_operator = 'operator-'
    domain = Domain('RxR')
    categories = [DocBasicArithmetic]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

class Addv(Operator):
    full_name = 'horizontal sum'
    signature = 's addv v'
    types = common.ftypes
    domain = Domain('R')
    categories = [DocMisc]
    desc = 'Returns the sum of all the elements contained in v'
    do_bench = False

class Mul(Operator):
    full_name = 'product'
    signature = 'v mul v v'
    cxx_operator = 'operator*'
    domain = Domain('RxR')
    categories = [DocBasicArithmetic]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

class Div(Operator):
    full_name = 'division'
    signature = 'v div v v'
    cxx_operator = 'operator/'
    domain = Domain('RxR\{0}')
    categories = [DocBasicArithmetic]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

class Neg(Operator):
    full_name = 'negation'
    signature = 'v neg v'
    cxx_operator = 'operator-'
    domain = Domain('R')
    categories = [DocBasicArithmetic]
    bench_auto_against_std = True

class Min(Operator):
    full_name = 'min'
    signature = 'v min v v'
    domain = Domain('RxR')
    categories = [DocBasicArithmetic]

class Max(Operator):
    full_name = 'max'
    signature = 'v max v v'
    domain = Domain('RxR')
    categories = [DocBasicArithmetic]
    bench_auto_against_mipp = True

class Shr(Operator):
    full_name = 'right shift'
    signature = 'v shr v p'
    types = common.iutypes
    cxx_operator = 'operator>>'
    domain = Domain('RxN')
    categories = [DocBitsOperators]
    bench_auto_against_mipp = True

    def bench_mipp_name(self, typ):
        return 'mipp::rshift<{}>'.format(typ)

class Shl(Operator):
    full_name = 'left shift'
    signature = 'v shl v p'
    types = common.iutypes
    cxx_operator = 'operator<<'
    domain = Domain('RxN')
    categories = [DocBitsOperators]
    bench_auto_against_mipp = True

    def bench_mipp_name(self, typ):
        return 'mipp::lshift<{}>'.format(typ)

class Eq(Operator):
    signature = 'l eq v v'
    cxx_operator = 'operator=='
    domain = Domain('RxR')
    categories = [DocComparison]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

    def bench_mipp_name(self, typ):
        return 'mipp::cmpeq<{}>'.format(typ)

class Ne(Operator):
    signature = 'l ne v v'
    cxx_operator = 'operator!='
    domain = Domain('RxR')
    categories = [DocComparison]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

    def bench_mipp_name(self, typ):
        return 'mipp::cmpneq<{}>'.format(typ)

class Gt(Operator):
    signature = 'l gt v v'
    cxx_operator = 'operator>'
    domain = Domain('RxR')
    categories = [DocComparison]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

    def bench_mipp_name(self, typ):
        return 'mipp::cmpgt<{}>'.format(typ)

class Ge(Operator):
    signature = 'l ge v v'
    cxx_operator = 'operator>='
    domain = Domain('RxR')
    categories = [DocComparison]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

    def bench_mipp_name(self, typ):
        return 'mipp::cmpge<{}>'.format(typ)

class Lt(Operator):
    signature = 'l lt v v'
    cxx_operator = 'operator<'
    domain = Domain('RxR')
    categories = [DocComparison]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

    def bench_mipp_name(self, typ):
        return 'mipp::cmplt<{}>'.format(typ)

class Le(Operator):
    signature = 'l le v v'
    cxx_operator = 'operator<='
    domain = Domain('RxR')
    categories = [DocComparison]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

    def bench_mipp_name(self, typ):
        return 'mipp::cmple<{}>'.format(typ)

class If_else1(Operator):
    signature = 'v if_else1 l v v'
    domain = Domain('BxRxR')
    categories = [DocMisc]

class Abs(Operator):
    signature = 'v abs v'
    domain = Domain('R')
    categories = [DocBasicArithmetic]
    bench_auto_against_mipp = True
    bench_auto_against_sleef = True
    #bench_auto_against_std = True

    def bench_sleef_name(self, simd, typ):
        return common.sleef_name('fabs', simd, typ)

class Fma(Operator):
    signature = 'v fma v v v'
    domain = Domain('RxRxR')
    categories = [DocBasicArithmetic]

class Fnma(Operator):
    signature = 'v fnma v v v'
    domain = Domain('RxRxR')
    categories = [DocBasicArithmetic]

class Fms(Operator):
    signature = 'v fms v v v'
    domain = Domain('RxRxR')
    categories = [DocBasicArithmetic]

class Fnms(Operator):
    signature = 'v fnms v v v'
    domain = Domain('RxRxR')
    categories = [DocBasicArithmetic]

class Ceil(Operator):
    signature = 'v ceil v'
    domain = Domain('R')
    categories = [DocRounding]
    bench_auto_against_sleef = True
    bench_auto_against_std = True

class Floor(Operator):
    signature = 'v floor v'
    domain = Domain('R')
    categories = [DocRounding]
    bench_auto_against_sleef = True
    bench_auto_against_std = True

class Trunc(Operator):
    signature = 'v trunc v'
    domain = Domain('R')
    categories = [DocRounding]
    bench_auto_against_sleef = True
    bench_auto_against_std = True

class Round_to_even(Operator):
    signature = 'v round_to_even v'
    domain = Domain('R')
    categories = [DocRounding]

class All(Operator):
    signature = 'p all l'
    domain = Domain('B')
    categories = [DocMisc]

class Any(Operator):
    signature = 'p any l'
    domain = Domain('B')
    categories = [DocMisc]

class Nbtrue(Operator):
    signature = 'p nbtrue l'
    domain = Domain('B')
    categories = [DocMisc]

class Reinterpret(Operator):
    signature = 'v reinterpret v'
    output_to = common.OUTPUT_TO_SAME_SIZE_TYPES
    domain = Domain('R')
    categories = [DocConversion]
    ## Disable bench
    do_bench = False

class Reinterpretl(Operator):
    signature = 'l reinterpretl l'
    domain = Domain('B')
    categories = [DocConversion]
    output_to = common.OUTPUT_TO_SAME_SIZE_TYPES
    ## Disable bench
    do_bench = False

class Cvt(Operator):
    signature = 'v cvt v'
    output_to = common.OUTPUT_TO_SAME_SIZE_TYPES
    domain = Domain('R')
    categories = [DocConversion]
    ## Disable bench
    do_bench = False

class Upcvt(Operator):
    signature = 'vx2 upcvt v'
    output_to = common.OUTPUT_TO_UP_TYPES
    domain = Domain('R')
    types = ['i8', 'u8', 'i16', 'u16', 'f16', 'i32', 'u32', 'f32']
    categories = [DocConversion]
    ## Disable bench
    do_bench = False

class Downcvt(Operator):
    signature = 'v downcvt v v'
    output_to = common.OUTPUT_TO_DOWN_TYPES
    domain = Domain('R')
    types = ['i16', 'u16', 'f16', 'i32', 'u32', 'f32', 'i64', 'u64', 'f64']
    categories = [DocConversion]
    ## Disable bench
    do_bench = False

class Rec(Operator):
    full_name = 'reciprocal'
    signature = 'v rec v'
    types = common.ftypes
    domain = Domain('R\{0}')
    categories = [DocBasicArithmetic]

class Rec11(Operator):
    full_name = 'reciprocal with relative error at most 2^{-11}'
    signature = 'v rec11 v'
    types = common.ftypes
    categories = [DocBasicArithmetic]
    domain = Domain('R\{0}')
    tests_ulps = 11

class Rec8(Operator):
    full_name = 'reciprocal with relative error at most 2^{-8}'
    signature = 'v rec8 v'
    types = common.ftypes
    categories = [DocBasicArithmetic]
    domain = Domain('R\{0}')
    tests_ulps = 8

class Sqrt(Operator):
    full_name = 'square root'
    signature = 'v sqrt v'
    types = common.ftypes
    domain = Domain('[0,Inf)')
    categories = [DocBasicArithmetic]
    bench_auto_against_mipp = True
    bench_auto_against_sleef = True
    bench_auto_against_std = True
    tests_mpfr = True

class Rsqrt11(Operator):
    full_name = 'square root'
    signature = 'v rsqrt11 v'
    types = common.ftypes
    domain = Domain('[0,Inf)')
    categories = [DocBasicArithmetic]
    tests_ulps = 11

class Rsqrt8(Operator):
    full_name = 'square root'
    signature = 'v rsqrt8 v'
    types = common.ftypes
    domain = Domain('[0,Inf)')
    categories = [DocBasicArithmetic]
    tests_ulps = 8

class Ziplo(Operator):
    full_name = 'ziplo'
    signature = 'v ziplo v v'
    types = common.types
    domain = Domain('R')
    categories = [DocMisc]
    do_bench = False

class Ziphi(Operator):
    full_name = 'ziphi'
    signature = 'v ziphi v v'
    types = common.types
    domain = Domain('R')
    categories = [DocMisc]
    do_bench = False

class Unziplo(Operator):
    full_name = 'unziplo'
    signature = 'v unziplo v v'
    types = common.types
    domain = Domain('R')
    categories = [DocMisc]
    do_bench = False

class Unziphi(Operator):
    full_name = 'unziphi'
    signature = 'v unziphi v v'
    types = common.types
    domain = Domain('R')
    categories = [DocMisc]
    do_bench = False

class ToMask(Operator):
    full_name = 'square root'
    signature = 'v to_mask l'
    categories = [DocLogicalOperators]
    do_bench = False

class ToLogical(Operator):
    full_name = 'square root'
    signature = 'l to_logical v'
    categories = [DocLogicalOperators]
    do_bench = False

# -----------------------------------------------------------------------------
# Import other operators if present: this is not Pythonic and an issue was
# opened for this

import os
import sys
import io

sep = ';' if sys.platform == 'win32' else ':'
search_dirs = os.getenv('NSIMD_OPERATORS_PATH')
if search_dirs != None:
    dirs = search_dirs.split(sep)
    for d in dirs:
        operators_file = os.path.join(d, 'operators.py')
        with io.open(operators_file, mode='r', encoding='utf-8') as fin:
            exec(fin.read())
