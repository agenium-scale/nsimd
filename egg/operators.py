# Use utf-8 encoding
# -*- coding: utf-8 -*-

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

if __name__ == 'operators':
    import common
else:
    from . import common
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

        def member_is_defined(member):
            if member in dct:
                return True
            for bc in range(len(bases)):
                if member in bases[bc].__dict__:
                    return True
            return False

        def get_member_value(member):
            if member in dct:
                return dct[member]
            for bc in range(len(bases)):
                if member in bases[bc].__dict__:
                    return bases[bc].__dict__[member]
            raise Exception('Member does not exists in class {}'.format(name))

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

        # By default tests are done on random numbers depending on the type
        # but sometimes one needs to produce only integers even if the
        # type is a floating point type.
        if 'tests_on_integers_only' not in dct:
            dct['tests_on_integers_only'] = False;

        # Fill domain, default is [-20 ; +20]
        if 'domain' not in dct:
            dct['domain'] = [[-20, 20], [-20, 20], [-20, 20]]

        # Number of UFP (cf. documentation) for testing
        if 'ufp' not in dct:
            dct['ufp'] = {'f16': 8, 'f32': 18, 'f64': 45}

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
                dct['desc'] = '{} the {}.'. \
                              format(dct['full_name'].capitalize(), arg)
            else:
                dct['desc'] = 'Returns the {} of the {}.'.\
                              format(dct['full_name'], arg)

        # Fill src, default is operator is in header not in source
        if not member_is_defined('src'):
            dct['src'] = False

        # Fill load_store, default is operator is not for loading/storing
        if 'load_store' not in dct:
            dct['load_store'] = False

        # Fill has_scalar_impl, default is based on several properties
        if 'has_scalar_impl' not in dct:
            if DocShuffle in dct['categories'] or \
               DocMisc in dct['categories'] or \
               'vx2' in dct['params'] or \
               'vx3' in dct['params'] or \
               'vx4' in dct['params'] or \
               dct['output_to'] in [common.OUTPUT_TO_UP_TYPES,
                                    common.OUTPUT_TO_DOWN_TYPES] or \
               dct['load_store']:
                dct['has_scalar_impl'] = False
            else:
                dct['has_scalar_impl'] = True

        ret = type.__new__(cls, name, bases, dct)
        operators[dct['name']] = ret()
        return ret

class Operator(object, metaclass=MAddToOperators):

    # Default values (for general purpose)
    cxx_operator = None
    autogen_cxx_adv = True
    output_to = common.OUTPUT_TO_SAME_TYPE
    types = common.types
    params = []
    aliases = []
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

    @property
    def returns(self):
        return self.params[0]

    @property
    def args(self):
        return self.params[1:]

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
        if self.src and 'sleef_symbol_prefix' in self.__class__.__dict__:
            ret['sleef_symbol_prefix'] = self.sleef_symbol_prefix
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
        elif lang == 'c_adv':
            args = ['a{}'.format(i - 1) for i in range(1, len(self.params))]
            if not self.closed:
                args = ['to_type'] + args
            args = ', '.join(args)
            return '#define nsimd_{}({})'.format(self.name, args)
        elif lang == 'cxx_base':
            def get_type(param, typename):
                if param == '_':
                    return 'void'
                elif param == 'p':
                    return 'int'
                elif param == 's':
                    return typename
                elif param == '*':
                    return '{}*'.format(typename)
                elif param == 'c*':
                    return '{} const*'.format(typename)
                elif param == 'vi':
                    return 'typename simd_traits<typename traits<{}>::itype,' \
                           ' NSIMD_SIMD>::simd_vector'.format(typename)
                elif param == 'l':
                    return \
                    'typename simd_traits<{}, NSIMD_SIMD>::simd_vectorl'. \
                    format(typename)
                elif param.startswith('v'):
                    return \
                    'typename simd_traits<{}, NSIMD_SIMD>::simd_vector{}'. \
                    format(typename, param[1:])
                else:
                    raise ValueError("Unknown param '{}'".format(param))
            return_typ = get_type(self.params[0], 'T')
            args_list = common.enum(self.params[1:])

            if not self.closed :
                tmpl_args = 'NSIMD_CONCEPT_VALUE_TYPE F, ' \
                            'NSIMD_CONCEPT_VALUE_TYPE T'
                typename = 'F'
            else:
                tmpl_args = 'NSIMD_CONCEPT_VALUE_TYPE T'
                typename = 'T'

            temp = ', '.join(['{} a{}'.format(get_type(a[1], typename),
                              a[0]) for a in args_list])
            temp += ', ' if temp != '' else ''
            if not self.closed:
                func_args = temp + 'F, T'
                if self.output_to == common.OUTPUT_TO_SAME_SIZE_TYPES:
                    cxx20_require = \
                        'NSIMD_REQUIRES(sizeof_v<F> == sizeof_v<T>) '
                elif self.output_to == common.OUTPUT_TO_UP_TYPES:
                    cxx20_require = \
                        'NSIMD_REQUIRES(2 * sizeof_v<F> == sizeof_v<T>) '
                else:
                    cxx20_require = \
                        'NSIMD_REQUIRES(sizeof_v<F> == 2 * sizeof_v<T>) '
            else:
                func_args = temp + 'T'
                cxx20_require = ''

            return 'template <{tmpl_args}> {cxx20_require}{return_typ} ' \
                   'NSIMD_VECTORCALL {name}({func_args});'. \
                   format(return_typ=return_typ, tmpl_args=tmpl_args,
                          func_args=func_args, name=self.name,
                          cxx20_require=cxx20_require)
        elif lang == 'cxx_adv':
            def get_type(param, typename, N):
                if param == '_':
                    return 'void'
                elif param == 'p':
                    return 'int'
                elif param == 's':
                    return typename
                elif param == '*':
                    return '{}*'.format(typename)
                elif param == 'c*':
                    return '{} const*'.format(typename)
                elif param == 'vi':
                    return 'pack<typename traits<{}>::itype, {}, SimdExt>'. \
                           format(typename, N)
                elif param == 'l':
                    return 'packl<{}, {}, SimdExt>'.format(typename, N)
                elif param.startswith('v'):
                    return 'pack{}<{}, {}, SimdExt>'. \
                    format(param[1:], typename, N)
                else:
                    raise ValueError("Unknown param '{}'".format(param))
            args_list = common.enum(self.params[1:])
            # Do we need tag dispatching on pack<>? e.g. len, set1 and load*
            inter = [i for i in ['v', 'l', 'vi', 'vx2', 'vx3', 'vx4'] \
                     if i in self.params[1:]]
            tag_dispatching = (inter == [])

            # Compute template arguments
            tmpl_args1 = ['NSIMD_CONCEPT_VALUE_TYPE T',
                          'NSIMD_CONCEPT_SIMD_EXT SimdExt']
            tmpl_argsN = ['NSIMD_CONCEPT_VALUE_TYPE T', 'int N',
                          'NSIMD_CONCEPT_SIMD_EXT SimdExt']
            def get_PACK(arg):
                if arg == 'l':
                    return 'PACKL'
                elif arg == 'v':
                    return 'PACK'
                else:
                    return 'PACK{}'.format(arg[1:].upper())
            if not self.closed:
                tmpl = 'NSIMD_CONCEPT_{} ToPackType'. \
                       format(get_PACK(self.params[0]))
                tmpl_args1 = [tmpl] + tmpl_args1
                tmpl_argsN = [tmpl] + tmpl_argsN
            tmpl_args1 = ', '.join(tmpl_args1)
            tmpl_argsN = ', '.join(tmpl_argsN)

            # Compute function arguments
            def arg_type(arg, typename, N):
                if arg in ['v', 'vi', 'vx2', 'vx3', 'vx4', 'l']:
                    return '{} const&'.format(get_type(arg, typename, N))
                else:
                    return get_type(arg, typename, N)
            args1 = ['{} a{}'.format(arg_type(i[1], 'T', '1'), i[0]) \
                     for i in args_list]
            argsN = ['{} a{}'.format(arg_type(i[1], 'T', 'N'), i[0]) \
                     for i in args_list]

            # Arguments without tag dispatching on pack
            other_argsN = ', '.join(argsN)

            # If we need tag dispatching, then the first argument type
            # is the output type:
            #   1. If not closed, then the output type is ToPackType
            #   2. If closed, then the output type is pack<T, N, SimdExt>
            if not self.closed:
                args1 = ['ToPackType const&'] + args1
                argsN = ['ToPackType const&'] + argsN
            elif tag_dispatching:
                args1 = [arg_type(self.params[0], 'T', '1')] + args1
                argsN = [arg_type(self.params[0], 'T', 'N')] + argsN
            args1 = ', '.join(args1)
            argsN = ', '.join(argsN)

            # Compute return type
            if not self.closed:
                ret1 = 'ToPackType'
                retN = 'ToPackType'
            else:
                ret1 = get_type(self.params[0], 'T', '1')
                retN = get_type(self.params[0], 'T', 'N')

            # For non closed operators that need tag dispatching we have a
            # require clause
            cxx20_require = ''
            if not self.closed:
                tmpl = 'NSIMD_REQUIRES((' \
                    '{}sizeof_v<typename ToPackType::value_type> == ' \
                        '{}sizeof_v<T> && ' \
                    'ToPackType::unroll == {{}} && '\
                    'std::is_same_v<typename ToPackType::simd_ext, SimdExt>))'
                if self.output_to == common.OUTPUT_TO_SAME_SIZE_TYPES:
                    cxx20_require = tmpl.format('', '')
                elif self.output_to == common.OUTPUT_TO_UP_TYPES:
                    cxx20_require = tmpl.format('', '2 * ')
                else:
                    cxx20_require = tmpl.format('2 * ', '')

            ret = { \
                '1': 'template <{tmpl_args1}> {cxx20_require}{ret1} ' \
                     '{cxx_name}({args1});'. \
                     format(tmpl_args1=tmpl_args1,
                            cxx20_require=cxx20_require.format('1'),
                            ret1=ret1, args1=args1, cxx_name=self.name),
                'N': 'template <{tmpl_argsN}> {cxx20_require}{retN} ' \
                     '{cxx_name}({argsN});'. \
                     format(tmpl_argsN=tmpl_argsN,
                            cxx20_require=cxx20_require.format('N'),
                            retN=retN, argsN=argsN, cxx_name=self.name)
            }
            if self.cxx_operator:
                ret.update({ \
                    'op1':
                    '''template <{tmpl_args1}>
                    {ret1} operator{cxx_name}({args1});'''. \
                    format(tmpl_args1=tmpl_args1, ret1=ret1, args1=args1,
                           cxx_name=self.cxx_operator),
                    'opN':
                    '''template <{tmpl_argsN}>
                    {retN} operator{cxx_name}({argsN});'''. \
                    format(tmpl_argsN=tmpl_argsN, retN=retN, argsN=argsN,
                           cxx_name=self.cxx_operator)
                })
            if not self.closed:
                ret['dispatch'] = \
                'template <{tmpl_argsN}> {cxx20_require}{retN} ' \
                '{cxx_name}({other_argsN});'. \
                format(tmpl_argsN=tmpl_argsN,
                       cxx20_require=cxx20_require.format('N'),
                       other_argsN=other_argsN, retN=retN, cxx_name=self.name)
            elif tag_dispatching:
                if [i for i in ['s', '*', 'c*'] if i in self.params[1:]] == []:
                    tmpl_T = ''
                    requires = ''
                else:
                    tmpl_T = ', NSIMD_CONCEPT_VALUE_TYPE T'
                    requires = 'NSIMD_REQUIRES((' \
                        'std::is_same_v<typename SimdVector::value_type, T>))'
                ret['dispatch'] = \
                '''template <NSIMD_CONCEPT_{PACK} SimdVector{tmpl_T}>{requires}
                   SimdVector {cxx_name}({other_argsN});'''.format(
                   PACK=get_PACK(self.params[0]), requires=requires,
                   other_argsN=other_argsN, cxx_name=self.name, tmpl_T=tmpl_T)
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
            sig = '{return_typ} NSIMD_VECTORCALL ' \
                  'nsimd_{name}_{simd_ext}_{suf}({c_args})'.format(**fmtspec)
        elif lang == 'cxx_base':
            sig = '{return_typ} NSIMD_VECTORCALL ' \
                  '{name}({cxx_args})'.format(**fmtspec)
        elif lang == 'cxx_adv':
            sig = ''
            raise Exception('TODO cxx_adv for {}'.format(lang))
        else:
            raise Exception('Unknown langage {}'.format(lang))

        return sig

    def get_scalar_signature(self, cpu_gpu, t, tt, lang):
        sig = '__device__ ' if cpu_gpu == 'gpu' else ''
        sig += common.get_one_type_scalar(self.params[0], tt) + ' '
        func_name = 'nsimd_' if lang == 'c' else ''
        func_name += 'gpu_' if cpu_gpu in ['gpu', 'oneapi'] else 'scalar_'
        func_name += self.name
        operator_on_logicals = (self.params == ['l'] * len(self.params))
        if lang == 'c' and not operator_on_logicals:
            func_name += '_{}_{}'.format(tt, t) if not self.closed \
                                                else '_{}'.format(t)
        sig += func_name
        args_list = common.enum([common.get_one_type_scalar(p, t)
                                 for p in self.params[1:]])
        args = ['{} a{}'.format(i[1], i[0]) for i in args_list]
        if lang == 'cxx' and (not self.closed or \
           ('v' not in self.params[1:] and not operator_on_logicals)):
            args = [tt] + args
        sig += '(' + ', '.join(args) + ')'
        return sig

class SrcOperator(Operator):
    src = True
    types = common.ftypes

# -----------------------------------------------------------------------------
# List of functions/operators

class Len(Operator):
    full_name = 'vector length'
    signature = 'p len'
    categories = [DocMisc]

class Set1(Operator):
    full_name = 'value broadcast'
    signature = 'v set1 s'
    categories = [DocMisc]
    desc = 'Returns a vector whose all elements are set to the given value.'

class Set1l(Operator):
    full_name = 'logical value broadcast'
    signature = 'l set1l p'
    categories = [DocMisc]
    desc = 'Returns a vector whose all elements are set to the given ' \
           'boolean value: zero means false and nonzero means true.'

class Loadu(Operator):
    signature = 'v loadu c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from unaligned memory.'

class MaskoLoadu1(Operator):
    signature = 'v masko_loadu1 l c* v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from unaligned memory corresponding to True elements.'

class MaskzLoadu1(Operator):
    signature = 'v maskz_loadu1 l c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from unaligned memory corresponding to True elements.'

class Load2u(Operator):
    full_name = 'load array of structure'
    signature = 'vx2 load2u c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 2 members from unaligned memory.'

class Load3u(Operator):
    full_name = 'load array of structure'
    signature = 'vx3 load3u c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 3 members from unaligned memory.'

class Load4u(Operator):
    full_name = 'load array of structure'
    signature = 'vx4 load4u c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 4 members from unaligned memory.'

class Loada(Operator):
    signature = 'v loada c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from aligned memory.'

class MaskoLoada(Operator):
    signature = 'v masko_loada1 l c* v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from aligned memory.'

class MaskzLoada(Operator):
    signature = 'v maskz_loada1 l c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from aligned memory corresponding to True elements.'

class Load2a(Operator):
    full_name = 'load array of structure'
    signature = 'vx2 load2a c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 2 members from aligned memory.'

class Load3a(Operator):
    full_name = 'load array of structure'
    signature = 'vx3 load3a c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 3 members from aligned memory.'

class Load4a(Operator):
    full_name = 'load array of structure'
    signature = 'vx4 load4a c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load array of structures of 4 members from aligned memory.'

class Loadlu(Operator):
    full_name = 'load vector of logicals'
    signature = 'l loadlu c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from unaligned memory and interpret it as booleans. ' + \
           'Zero is interpreted as False and nonzero as True.'

class Loadla(Operator):
    full_name = 'load vector of logicals'
    signature = 'l loadla c*'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Load data from aligned memory and interpret it as booleans. ' + \
           'Zero is interpreted as False and nonzero as True.'

class Storeu(Operator):
    signature = '_ storeu * v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store SIMD vector into unaligned memory.'

class MaskStoreu1(Operator):
    signature = '_ mask_storeu1 l * v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store active SIMD vector elements into unaligned memory.'

class Store2u(Operator):
    signature = '_ store2u * v v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store 2 SIMD vectors as array of structures of 2 members into ' + \
           'unaligned memory.'

class Store3u(Operator):
    full_name = 'store into array of structures'
    signature = '_ store3u * v v v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store 3 SIMD vectors as array of structures of 3 members into ' + \
           'unaligned memory.'

class Store4u(Operator):
    full_name = 'store into array of structures'
    signature = '_ store4u * v v v v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store 4 SIMD vectors as array of structures of 4 members into ' + \
           'unaligned memory.'

class Storea(Operator):
    signature = '_ storea * v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store SIMD vector into aligned memory.'

class MaskStorea1(Operator):
    signature = '_ mask_storea1 l * v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store active SIMD vector elements into aligned memory.'

class Store2a(Operator):
    full_name = 'store into array of structures'
    signature = '_ store2a * v v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store 2 SIMD vectors as array of structures of 2 members into ' + \
           'aligned memory.'

class Store3a(Operator):
    full_name = 'store into array of structures'
    signature = '_ store3a * v v v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store 3 SIMD vectors as array of structures of 3 members into ' + \
           'aligned memory.'

class Store4a(Operator):
    full_name = 'store into array of structures'
    signature = '_ store4a * v v v v'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store 4 SIMD vectors as array of structures of 4 members into ' + \
           'aligned memory.'

class Gather(Operator):
    full_name = 'gather elements from memory into a SIMD vector'
    signature = 'v gather c* vi'
    load_store = True
    categories = [DocLoadStore]
    types = common.ftypes + ['i16', 'u16', 'u32', 'i32', 'i64', 'u64']
    desc = 'Gather elements from memory with base address given as first ' \
           'argument and offsets given as second argument.'

class GatherLinear(Operator):
    full_name = 'gather elements from memory into a SIMD vector'
    signature = 'v gather_linear c* p'
    load_store = True
    categories = [DocLoadStore]
    types = common.types
    desc = 'Gather elements from memory with base address given as first ' \
           'argument and steps given as second argument. This operator ' \
           'using a SIMD register.'

#class MaskzGather(Operator):
#    full_name = 'gather active elements from SIMD vector to memory and put ' \
#                'zeros in inactive elements.'
#    signature = 'v maskz_gather l * vi'
#    load_store = True
#    categories = [DocLoadStore]
#    types = common.ftypes + ['i16', 'u16', 'u32', 'i32', 'i64', 'u64']
#    desc = 'Gather elements from memory with base address given as second ' \
#           'argument and offsets given as third argument. Inactive elements ' \
#           '(first argument) are set to zero.'

#class MaskoGather(Operator):
#    full_name = 'gather active elements from SIMD vector to memory and put ' \
#                'zeros in inactive elements.'
#    signature = 'v masko_gather l * vi v'
#    load_store = True
#    categories = [DocLoadStore]
#    types = common.ftypes + ['i16', 'u16', 'u32', 'i32', 'i64', 'u64']
#    desc = 'Gather elements from memory with base address given as second ' \
#           'argument and offsets given as third argument. Inactive elements ' \
#           '(first argument) are set to corresponding elements from fourth ' \
#           'argument.'

class Scatter(Operator):
    full_name = 'scatter elements from SIMD vector to memory'
    signature = '_ scatter * vi v'
    load_store = True
    categories = [DocLoadStore]
    types = common.ftypes + ['i16', 'u16', 'u32', 'i32', 'i64', 'u64']
    desc = 'Scatter elements from third argument to memory with base ' \
           'address given as first argument and offsets given as second ' \
           'argument.'

class ScatterLinear(Operator):
    full_name = 'scatter elements from SIMD vector to memory'
    signature = '_ scatter_linear * p v'
    load_store = True
    categories = [DocLoadStore]
    types = common.types
    desc = 'Scatter elements from third argument to memory with base ' \
           'address given as first argument and steps given as second ' \
           'argument. This operator avoids using a SIMD register.'

#class MaskScatter(Operator):
#    full_name = 'scatter active elements from SIMD vector to memory'
#    signature = '_ mask_scatter l * vi v'
#    load_store = True
#    categories = [DocLoadStore]
#    types = common.ftypes + ['i16', 'u16', 'u32', 'i32', 'i64', 'u64']
#    desc = 'Scatter active (first argument) elements from fourth argument ' \
#           'to memory with base address given as second argument and ' \
#           'offsets given as third argument.'

class Storelu(Operator):
    full_name = 'store vector of logicals'
    signature = '_ storelu * l'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store SIMD vector of booleans into unaligned memory. True is ' + \
           'stored as 1 and False as 0.'

class Storela(Operator):
    full_name = 'store vector of logicals'
    signature = '_ storela * l'
    load_store = True
    categories = [DocLoadStore]
    desc = 'Store SIMD vector of booleans into aligned memory. True is ' + \
           'stored as 1 and False as 0.'

class Orb(Operator):
    full_name = 'bitwise or'
    signature = 'v orb v v'
    cxx_operator = '|'
    categories = [DocBitsOperators]

class Andb(Operator):
    full_name = 'bitwise and'
    signature = 'v andb v v'
    cxx_operator = '&'
    categories = [DocBitsOperators]

class Andnotb(Operator):
    full_name = 'bitwise andnot'
    signature = 'v andnotb v v'
    categories = [DocBitsOperators]
    desc = 'Returns the bitwise andnot of its arguments, more precisely ' \
           '"arg1 and (not arg2)"'

class Notb(Operator):
    full_name = 'bitwise not'
    signature = 'v notb v'
    cxx_operator = '~'
    categories = [DocBitsOperators]

class Xorb(Operator):
    full_name = 'bitwise xor'
    signature = 'v xorb v v'
    cxx_operator = '^'
    categories = [DocBitsOperators]

class Orl(Operator):
    full_name = 'logical or'
    signature = 'l orl l l'
    cxx_operator = '||'
    categories = [DocLogicalOperators]

class Andl(Operator):
    full_name = 'logical and'
    signature = 'l andl l l'
    cxx_operator = '&&'
    categories = [DocLogicalOperators]

class Andnotl(Operator):
    full_name = 'logical andnot'
    signature = 'l andnotl l l'
    categories = [DocLogicalOperators]
    desc = 'Returns the logical andnot of its arguments, more precisely ' \
           '"arg1 and (not arg2)"'

class Xorl(Operator):
    full_name = 'logical xor'
    signature = 'l xorl l l'
    categories = [DocLogicalOperators]

class Notl(Operator):
    full_name = 'logical not'
    signature = 'l notl l'
    cxx_operator = '!'
    categories = [DocLogicalOperators]
    bench_auto_against_std = True

class Add(Operator):
    full_name = 'addition'
    signature = 'v add v v'
    cxx_operator = '+'
    categories = [DocBasicArithmetic]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

class Sub(Operator):
    full_name = 'subtraction'
    signature = 'v sub v v'
    cxx_operator = '-'
    categories = [DocBasicArithmetic]
    bench_auto_against_std = True
    bench_auto_against_mipp = True

class Addv(Operator):
    full_name = 'horizontal sum'
    signature = 's addv v'
    categories = [DocMisc]
    desc = 'Returns the sum of all the elements contained in v'
    do_bench = False
    types = common.ftypes

class Mul(Operator):
    full_name = 'multiplication'
    signature = 'v mul v v'
    cxx_operator = '*'
    categories = [DocBasicArithmetic]

class Div(Operator):
    full_name = 'division'
    signature = 'v div v v'
    cxx_operator = '/'
    domain = [[-20, 20], [0.5, 20]]
    categories = [DocBasicArithmetic]

class Neg(Operator):
    full_name = 'opposite'
    signature = 'v neg v'
    cxx_operator = '-'
    categories = [DocBasicArithmetic]

class Min(Operator):
    full_name = 'minimum'
    signature = 'v min v v'
    categories = [DocBasicArithmetic]

class Max(Operator):
    full_name = 'maximum'
    signature = 'v max v v'
    categories = [DocBasicArithmetic]

class Shr(Operator):
    full_name = 'right shift in zeros'
    signature = 'v shr v p'
    types = common.iutypes
    cxx_operator = '>>'
    categories = [DocBitsOperators]

class Shl(Operator):
    full_name = 'left shift'
    signature = 'v shl v p'
    types = common.iutypes
    cxx_operator = '<<'
    categories = [DocBitsOperators]

class Shra(Operator):
    full_name = 'arithmetic right shift'
    signature = 'v shra v p'
    types = common.iutypes
    categories = [DocBitsOperators]
    desc = 'Performs a right shift operation with sign extension.'

class Eq(Operator):
    full_name = 'compare for equality'
    signature = 'l eq v v'
    cxx_operator = '=='
    categories = [DocComparison]

class Ne(Operator):
    full_name = 'compare for inequality'
    signature = 'l ne v v'
    cxx_operator = '!='
    categories = [DocComparison]
    desc = 'Compare the inputs for inequality.'

class Gt(Operator):
    full_name = 'compare for greater-than'
    signature = 'l gt v v'
    cxx_operator = '>'
    categories = [DocComparison]
    desc = 'Compare the inputs for greater-than.'

class Ge(Operator):
    full_name = 'compare for greater-or-equal-than'
    signature = 'l ge v v'
    cxx_operator = '>='
    categories = [DocComparison]
    desc = 'Compare the inputs for greater-or-equal-than.'

class Lt(Operator):
    full_name = 'compare for lesser-than'
    signature = 'l lt v v'
    cxx_operator = '<'
    categories = [DocComparison]
    desc = 'Compare the inputs for lesser-than.'

class Le(Operator):
    full_name = 'compare for lesser-or-equal-than'
    signature = 'l le v v'
    cxx_operator = '<='
    categories = [DocComparison]
    desc = 'Compare the inputs for lesser-or-equal-than.'

class If_else1(Operator):
    full_name = 'blend'
    signature = 'v if_else1 l v v'
    categories = [DocMisc]
    desc = 'Blend the inputs using the vector of logical as a first ' + \
           'argument. Elements of the second input is taken when the ' + \
           'corresponding elements from the vector of logicals is true, ' + \
           'otherwise elements of the second input are taken.'

class Abs(Operator):
    full_name = 'absolute value'
    signature = 'v abs v'
    categories = [DocBasicArithmetic]

class Fma(Operator):
    full_name = 'fused multiply-add'
    signature = 'v fma v v v'
    categories = [DocBasicArithmetic]
    desc = 'Multiply the first and second inputs and then adds the third ' + \
           'input.'
    tests_on_integers_only = True

class Fnma(Operator):
    full_name = 'fused negate-multiply-add'
    signature = 'v fnma v v v'
    categories = [DocBasicArithmetic]
    desc = 'Multiply the first and second inputs, negate the intermediate ' + \
           'result and then adds the third input.'
    tests_on_integers_only = True

class Fms(Operator):
    full_name = 'fused multiply-substract'
    signature = 'v fms v v v'
    categories = [DocBasicArithmetic]
    desc = 'Substracts the third input to multiplication the first and ' + \
           'second inputs.'
    tests_on_integers_only = True

class Fnms(Operator):
    full_name = 'fused negate-multiply-substract'
    signature = 'v fnms v v v'
    categories = [DocBasicArithmetic]
    desc = 'Multiply the first and second inputs, negate the intermediate ' + \
           'result and then substracts the third input to the ' + \
           'intermediate result.'
    tests_on_integers_only = True

class Ceil(Operator):
    full_name = 'rounding up to integer value'
    signature = 'v ceil v'
    categories = [DocRounding]

class Floor(Operator):
    full_name = 'rounding down to integer value'
    signature = 'v floor v'
    categories = [DocRounding]

class Trunc(Operator):
    full_name = 'rounding towards zero to integer value'
    signature = 'v trunc v'
    categories = [DocRounding]

class Round_to_even(Operator):
    full_name = 'rounding to nearest integer value, tie to even'
    signature = 'v round_to_even v'
    categories = [DocRounding]

class All(Operator):
    full_name = 'check all elements'
    signature = 'p all l'
    categories = [DocMisc]
    desc = 'Return true if and only if all elements of the inputs are true.'

class Any(Operator):
    full_name = 'check for one true elements'
    signature = 'p any l'
    categories = [DocMisc]
    desc = 'Return true if and only if at least one element of the inputs ' + \
           'is true.'

class Nbtrue(Operator):
    full_name = 'count true elements'
    signature = 'p nbtrue l'
    categories = [DocMisc]
    desc = 'Return the number of true elements in the input.'

class Reinterpret(Operator):
    full_name = 'reinterpret vector'
    signature = 'v reinterpret v'
    output_to = common.OUTPUT_TO_SAME_SIZE_TYPES
    categories = [DocConversion]
    desc = 'Reinterpret input vector into a different vector type ' + \
           'preserving all bits.'

class Reinterpretl(Operator):
    full_name = 'reinterpret vector of logicals'
    signature = 'l reinterpretl l'
    categories = [DocConversion]
    output_to = common.OUTPUT_TO_SAME_SIZE_TYPES
    has_scalar_impl = False
    desc = 'Reinterpret input vector of logicals into a different vector ' + \
           'type of logicals preserving all elements values. The output ' + \
           'type must have same length as input type.'

class Cvt(Operator):
    full_name = 'convert vector'
    signature = 'v cvt v'
    output_to = common.OUTPUT_TO_SAME_SIZE_TYPES
    categories = [DocConversion]
    desc = 'Convert input vector into a different vector type. The output ' + \
           'type must have same length as input type.'

class Upcvt(Operator):
    full_name = 'convert vector to larger type'
    signature = 'vx2 upcvt v'
    output_to = common.OUTPUT_TO_UP_TYPES
    types = ['i8', 'u8', 'i16', 'u16', 'f16', 'i32', 'u32', 'f32']
    categories = [DocConversion]
    desc = 'Convert input vector into a different larger vector type. The ' + \
           'output type must be twice as large as the input type.'

class Downcvt(Operator):
    full_name = 'convert vector to narrow type'
    signature = 'v downcvt v v'
    output_to = common.OUTPUT_TO_DOWN_TYPES
    types = ['i16', 'u16', 'f16', 'i32', 'u32', 'f32', 'i64', 'u64', 'f64']
    categories = [DocConversion]
    desc = 'Convert input vector into a different narrow vector type. The ' + \
           'output type must be twice as less as the input type.'

class Rec(Operator):
    full_name = 'reciprocal'
    signature = 'v rec v'
    types = common.ftypes
    domain = [[-20, -0.5, 0.5, 20]]
    categories = [DocBasicArithmetic]

class Rec11(Operator):
    full_name = 'reciprocal with relative error at most $2^{-11}$'
    signature = 'v rec11 v'
    types = common.ftypes
    categories = [DocBasicArithmetic]
    domain = [[-20, -0.5, 0.5, 20]]
    ufp = { 'f16': 10, 'f32': 10, 'f64': 10 }

class Rec8(Operator):
    full_name = 'reciprocal with relative error at most $2^{-8}$'
    signature = 'v rec8 v'
    types = common.ftypes
    categories = [DocBasicArithmetic]
    domain = [[-20, -0.5, 0.5, 20]]
    ufp = { 'f16': 7, 'f32': 7, 'f64': 7 }

class Sqrt(Operator):
    full_name = 'square root'
    signature = 'v sqrt v'
    types = common.ftypes
    domain = [[0, 20]]
    categories = [DocBasicArithmetic]

class Rsqrt11(Operator):
    full_name = 'square root with relative error at most $2^{-11}$'
    signature = 'v rsqrt11 v'
    types = common.ftypes
    domain = [[0.5, 20]]
    ufp = { 'f16': 10, 'f32': 10, 'f64': 10 }
    categories = [DocBasicArithmetic]

class Rsqrt8(Operator):
    full_name = 'square root with relative error at most $2^{-8}$'
    signature = 'v rsqrt8 v'
    types = common.ftypes
    domain = [[0.5, 20]]
    ufp = { 'f16': 7, 'f32': 7, 'f64': 7 }
    categories = [DocBasicArithmetic]

class Ziplo(Operator):
    full_name = 'zip low halves'
    signature = 'v ziplo v v'
    types = common.types
    categories = [DocShuffle]
    desc = 'Construct a vector where elements of the first low half input ' + \
           'are followed by the corresponding element of the second low ' + \
           'half input.'

class Ziphi(Operator):
    full_name = 'zip high halves'
    signature = 'v ziphi v v'
    types = common.types
    categories = [DocShuffle]
    desc = 'Construct a vector where elements of the first high half ' + \
           'input are followed by the corresponding element of the second ' + \
           'high half input.'

class Unziplo(Operator):
    full_name = 'unziplo'
    signature = 'v unziplo v v'
    types = common.types
    categories = [DocShuffle]

class Unziphi(Operator):
    full_name = 'unziphi'
    signature = 'v unziphi v v'
    types = common.types
    categories = [DocShuffle]

class Zip(Operator):
    full_name = 'zip'
    signature = 'vx2 zip v v'
    types = common.types
    categories = [DocShuffle]

class Unzip(Operator):
    full_name = 'unzip'
    signature = 'vx2 unzip v v'
    types = common.types
    categories = [DocShuffle]

class ToMask(Operator):
    full_name = 'build mask from logicals'
    signature = 'v to_mask l'
    categories = [DocLogicalOperators]
    desc = 'Returns a mask consisting of all ones for true elements and ' + \
           'all zeros for false elements.'

class ToLogical(Operator):
    full_name = 'build logicals from data'
    signature = 'l to_logical v'
    categories = [DocLogicalOperators]
    desc = 'Returns a vector of logicals. Set true when the corresponding ' + \
           'elements are non zero (at least one bit to 1) and false ' + \
           'otherwise.'

class Iota(Operator):
    full_name = 'fill vector with increasing values'
    signature = 'v iota'
    categories = [DocMisc]
    desc = 'Returns a vectors whose first element is zero, the second is ' \
           'one and so on.'

class MaskForLoopTail(Operator):
    full_name = 'build mask for ending loops'
    signature = 'l mask_for_loop_tail p p'
    categories = [DocMisc]
    desc = 'Returns a mask for loading/storing data at loop tails by ' \
           'setting the first elements to True and the last to False. ' \
           'The first argument is index in a loop whose number of elements ' \
           'is given by the second argument.'

class Adds(Operator):
    full_name = 'addition using saturation'
    signature = 'v adds v v'
    categories = [DocBasicArithmetic]
    desc = 'Returns the saturated sum of the two vectors given as arguments'

class Subs(Operator):
    full_name = 'subtraction using saturation'
    signature = 'v subs v v'
    categories = [DocBasicArithmetic]
    desc = 'Returns the saturated subtraction of the two vectors given as ' \
           'arguments'

class Sin_u35(SrcOperator):
    full_name = 'sine'
    signature = 'v sin_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_sin_u35'
    categories = [DocTrigo]
    desc = 'Compute the sine of its argument with a precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Cos_u35(SrcOperator):
    full_name = 'cosine'
    signature = 'v cos_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_cos_u35'
    categories = [DocTrigo]
    desc = 'Compute the cosine of its argument with a precision of ' \
           '3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Tan_u35(SrcOperator):
    full_name = 'tangent'
    signature = 'v tan_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_tan_u35'
    domain = [[-4.7, -1.6, -1.5, 1.5, 1.6, 4.7]]
    categories = [DocTrigo]
    desc = 'Compute the tangent of its argument with a precision of ' \
           '3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Asin_u35(SrcOperator):
    full_name = 'arcsine'
    signature = 'v asin_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_asin_u35'
    domain = [[-0.9, 0.9]]
    categories = [DocTrigo]
    desc = 'Compute the arcsine of its argument with a precision of ' \
           '3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Acos_u35(SrcOperator):
    full_name = 'arccosine'
    signature = 'v acos_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_acos_u35'
    domain = [[-0.9, 0.9]]
    categories = [DocTrigo]
    desc = 'Compute the arccosine of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Atan_u35(SrcOperator):
    full_name = 'arctangent'
    signature = 'v atan_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_atan_u35'
    categories = [DocTrigo]
    desc = 'Compute the arctangent of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Atan2_u35(SrcOperator):
    full_name = 'arctangent'
    signature = 'v atan2_u35 v v'
    sleef_symbol_prefix = 'nsimd_sleef_atan2_u35'
    domain = [[-20, 20], [-20, -0.5, 0.5, 20]]
    categories = [DocTrigo]
    desc = 'Compute the arctangent of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Log_u35(SrcOperator):
    full_name = 'natural logarithm'
    signature = 'v log_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_log_u35'
    domain = [[0.5, 20]]
    categories = [DocExpLog]
    desc = 'Compute the natural logarithm of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Cbrt_u35(SrcOperator):
    full_name = 'cube root'
    signature = 'v cbrt_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_cbrt_u35'
    categories = [DocBasicArithmetic]
    desc = 'Compute the cube root of its argument with a precision of ' \
           '3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Sin_u10(SrcOperator):
    full_name = 'sine'
    signature = 'v sin_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_sin_u10'
    categories = [DocTrigo]
    desc = 'Compute the sine of its argument with a precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Cos_u10(SrcOperator):
    full_name = 'cosine'
    signature = 'v cos_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_cos_u10'
    categories = [DocTrigo]
    desc = 'Compute the cosine of its argument with a precision of ' \
           '1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Tan_u10(SrcOperator):
    full_name = 'tangent'
    signature = 'v tan_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_tan_u10'
    domain = [[-4.7, -1.6, -1.5, 1.5, 1.6, 4.7]]
    categories = [DocTrigo]
    desc = 'Compute the tangent of its argument with a precision of ' \
           '1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Asin_u10(SrcOperator):
    full_name = 'arcsine'
    signature = 'v asin_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_asin_u10'
    domain = [[-0.9, 0.9]]
    categories = [DocTrigo]
    desc = 'Compute the arcsine of its argument with a precision of ' \
           '1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Acos_u10(SrcOperator):
    full_name = 'arccosine'
    signature = 'v acos_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_acos_u10'
    domain = [[-0.9, 0.9]]
    categories = [DocTrigo]
    desc = 'Compute the arccosine of its argument with a precision of ' \
           '1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Atan_u10(SrcOperator):
    full_name = 'arctangent'
    signature = 'v atan_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_atan_u10'
    categories = [DocTrigo]
    desc = 'Compute the arctangent of its argument with a precision of ' \
           '1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Atan2_u10(SrcOperator):
    full_name = 'arctangent'
    signature = 'v atan2_u10 v v'
    sleef_symbol_prefix = 'nsimd_sleef_atan2_u10'
    domain = [[-20, 20], [-20, -0.5, 0.5, 20]]
    categories = [DocTrigo]
    desc = 'Compute the arctangent of its argument with a precision of ' \
           '1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Log_u10(SrcOperator):
    full_name = 'natural logarithm'
    signature = 'v log_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_log_u10'
    domain = [[0.5, 20]]
    categories = [DocExpLog]
    desc = 'Compute the natural logarithm of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Cbrt_u10(SrcOperator):
    full_name = 'cube root'
    signature = 'v cbrt_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_cbrt_u10'
    categories = [DocBasicArithmetic]
    desc = 'Compute the cube root of its argument with a precision of ' \
           '1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Exp_u10(SrcOperator):
    full_name = 'base-e exponential'
    signature = 'v exp_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_exp_u10'
    domain = [[-20, 5]]
    categories = [DocExpLog]
    desc = 'Compute the base-e exponential of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Pow_u10(SrcOperator):
    full_name = 'power'
    signature = 'v pow_u10 v v'
    sleef_symbol_prefix = 'nsimd_sleef_pow_u10'
    domain = [[0, 5], [-5, 5]]
    categories = [DocExpLog]
    desc = 'Compute the power of its argument with a precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Sinh_u10(SrcOperator):
    full_name = 'hyperbolic sine'
    signature = 'v sinh_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_sinh_u10'
    categories = [DocHyper]
    desc = 'Compute the hyperbolic sine of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Cosh_u10(SrcOperator):
    full_name = 'hyperbolic cosine'
    signature = 'v cosh_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_cosh_u10'
    categories = [DocHyper]
    desc = 'Compute the hyperbolic cosine of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Tanh_u10(SrcOperator):
    full_name = 'hyperbolic tangent'
    signature = 'v tanh_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_tanh_u10'
    categories = [DocHyper]
    desc = 'Compute the hyperbolic tangent of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Sinh_u35(SrcOperator):
    full_name = 'hyperbolic sine'
    signature = 'v sinh_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_sinh_u35'
    categories = [DocHyper]
    desc = 'Compute the hyperbolic sine of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Cosh_u35(SrcOperator):
    full_name = 'hyperbolic cosine'
    signature = 'v cosh_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_cosh_u35'
    categories = [DocHyper]
    desc = 'Compute the hyperbolic cosine of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Tanh_u35(SrcOperator):
    full_name = 'hyperbolic tangent'
    signature = 'v tanh_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_tanh_u35'
    categories = [DocHyper]
    desc = 'Compute the hyperbolic tangent of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Asinh_u10(SrcOperator):
    full_name = 'inverse hyperbolic sine'
    signature = 'v asinh_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_asinh_u10'
    categories = [DocHyper]
    desc = 'Compute the inverse hyperbolic sine of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Acosh_u10(SrcOperator):
    full_name = 'inverse hyperbolic cosine'
    signature = 'v acosh_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_acosh_u10'
    categories = [DocHyper]
    domain = [[1, 20]]
    desc = 'Compute the inverse hyperbolic cosine of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Atanh_u10(SrcOperator):
    full_name = 'inverse hyperbolic tangent'
    signature = 'v atanh_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_atanh_u10'
    domain = [[-0.9, 0.9]]
    categories = [DocHyper]
    desc = 'Compute the inverse hyperbolic tangent of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Exp2_u10(SrcOperator):
    full_name = 'base-2 exponential'
    signature = 'v exp2_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_exp2_u10'
    domain = [[-20, 5]]
    categories = [DocExpLog]
    desc = 'Compute the base-2 exponential of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Exp2_u35(SrcOperator):
    full_name = 'base-2 exponential'
    signature = 'v exp2_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_exp2_u35'
    domain = [[-20, 5]]
    categories = [DocExpLog]
    desc = 'Compute the base-2 exponential of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Exp10_u10(SrcOperator):
    full_name = 'base-10 exponential'
    signature = 'v exp10_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_exp10_u10'
    domain = [[-5, 3]]
    categories = [DocExpLog]
    desc = 'Compute the base-10 exponential of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Exp10_u35(SrcOperator):
    full_name = 'base-10 exponential'
    signature = 'v exp10_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_exp10_u35'
    domain = [[-5, 3]]
    categories = [DocExpLog]
    desc = 'Compute the base-10 exponential of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Expm1_u10(SrcOperator):
    full_name = 'exponential minus 1'
    signature = 'v expm1_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_expm1_u10'
    domain = [[-5, 3]]
    categories = [DocExpLog]
    desc = 'Compute the exponential minus 1 of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Log10_u10(SrcOperator):
    full_name = 'base-10 logarithm'
    signature = 'v log10_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_log10_u10'
    domain = [[0.5, 20]]
    categories = [DocExpLog]
    desc = 'Compute the base-10 logarithm of its argument with a precision ' \
           'of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Log2_u10(SrcOperator):
    full_name = 'base-2 logarithm'
    signature = 'v log2_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_log2_u10'
    domain = [[0.5, 20]]
    categories = [DocExpLog]
    desc = 'Compute the base-2 logarithm of its argument with a precision ' \
           'of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Log2_u35(SrcOperator):
    full_name = 'base-2 logarithm'
    signature = 'v log2_u35 v'
    sleef_symbol_prefix = 'nsimd_sleef_log2_u35'
    domain = [[0.5, 20]]
    categories = [DocExpLog]
    desc = 'Compute the base-2 logarithm of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Log1p_u10(SrcOperator):
    full_name = 'logarithm of 1 plus argument'
    signature = 'v log1p_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_log1p_u10'
    domain = [[-0.5, 19]]
    categories = [DocExpLog]
    desc = 'Compute the logarithm of 1 plus argument of its argument with ' \
           'a precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Sinpi_u05(SrcOperator):
    full_name = 'sine of pi times argument'
    signature = 'v sinpi_u05 v'
    sleef_symbol_prefix = 'nsimd_sleef_sinpi_u05'
    categories = [DocTrigo]
    desc = 'Compute the sine of pi times argument of its argument with a ' \
           'precision of 0.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Cospi_u05(SrcOperator):
    full_name = 'cosine of pi times argument'
    signature = 'v cospi_u05 v'
    sleef_symbol_prefix = 'nsimd_sleef_cospi_u05'
    categories = [DocTrigo]
    desc = 'Compute the cosine of pi times argument of its argument with ' \
           'a precision of 0.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Hypot_u05(SrcOperator):
    full_name = 'Euclidean distance'
    signature = 'v hypot_u05 v v'
    sleef_symbol_prefix = 'nsimd_sleef_hypot_u05'
    categories = [DocBasicArithmetic]
    desc = 'Compute the Euclidean distance of its argument with a ' \
           'precision of 0.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Hypot_u35(SrcOperator):
    full_name = 'Euclidean distance'
    signature = 'v hypot_u35 v v'
    sleef_symbol_prefix = 'nsimd_sleef_hypot_u35'
    categories = [DocBasicArithmetic]
    desc = 'Compute the Euclidean distance of its argument with a ' \
           'precision of 3.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Remainder(SrcOperator):
    full_name = 'floating-point remainder'
    signature = 'v remainder v v'
    sleef_symbol_prefix = 'nsimd_sleef_remainder'
    domain = [[1, 20], [1, 20]]
    categories = [DocBasicArithmetic]
    desc = 'Compute the floating-point remainder of its arguments. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Fmod(SrcOperator):
    full_name = 'floating-point remainder'
    signature = 'v fmod v v'
    sleef_symbol_prefix = 'nsimd_sleef_fmod'
    domain = [[1, 20], [1, 20]]
    categories = [DocBasicArithmetic]
    desc = 'Compute the floating-point remainder of its argument. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Lgamma_u10(SrcOperator):
    full_name = 'log gamma'
    signature = 'v lgamma_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_lgamma_u10'
    domain = [[0.5, 20]]
    categories = [DocExpLog]
    desc = 'Compute the log gamma of its argument with a precision of ' \
           '1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Tgamma_u10(SrcOperator):
    full_name = 'true gamma'
    signature = 'v tgamma_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_tgamma_u10'
    domain = [[0.5, 5]]
    categories = [DocExpLog]
    desc = 'Compute the true gamma of its argument with a precision of ' \
           '1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Erf_u10(SrcOperator):
    full_name = 'complementary error'
    signature = 'v erf_u10 v'
    sleef_symbol_prefix = 'nsimd_sleef_erf_u10'
    categories = [DocExpLog]
    desc = 'Compute the complementary error of its argument with a ' \
           'precision of 1.0 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

class Erfc_u15(SrcOperator):
    full_name = 'complementary error'
    signature = 'v erfc_u15 v'
    sleef_symbol_prefix = 'nsimd_sleef_erfc_u15'
    categories = [DocExpLog]
    desc = 'Compute the complementary error of its argument with a ' \
           'precision of 1.5 ulps. ' \
           'For more informations visit <https://sleef.org/purec.xhtml>.'

