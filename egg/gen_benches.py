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
from datetime import date
from collections import OrderedDict

# -----------------------------------------------------------------------------
# Sig

def sig_replace_name(sig, name):
    sig = sig.split(' ')
    sig[1] = name
    return ' '.join(sig)

def sig_translate(sig, translates, name=None):
    sig = sig.split(' ')
    ## Translates a given type to another
    sig[0] = translates.get(sig[0], sig[0])
    ## Do not use sig[1] (the function name)
    for i, p in enumerate(sig[2:]):
        sig[2 + i] = translates.get(p, p)
    sig = ' '.join(sig)
    ## Redefine name if available
    if name:
        sig = sig_replace_name(sig, name)
    return sig

# -----------------------------------------------------------------------------
# Errors

class BenchError(RuntimeError):
    pass

# -----------------------------------------------------------------------------
# Markers

def asm_marker(simd, bench_name):
    if simd in common.x86_simds:
        return '__asm__ __volatile__("callq __asm_marker__{bench_name}");'. \
               format(bench_name=bench_name)
    elif simd in common.arm_simds:
        return '__asm__ __volatile__("bl __asm_marker__{bench_name}");'. \
               format(bench_name=bench_name)
    else:
        raise BenchError('Unable to write marker for SIMD: {}'.format(simd))

# -----------------------------------------------------------------------------
# Metaclass

# Provides __static_init__ hook
class StaticInitMetaClass(type):
    def __new__(cls, name, bases, dct):
        x = type.__new__(cls, name, bases, dct)
        x.__static_init__(x)
        return x

# -----------------------------------------------------------------------------
# Basic nsimd types

## Will be automatically populated thanks to the metaclass
types = {}

# -----------------------------------------------------------------------------

class TypeBase(object, metaclass=StaticInitMetaClass):

    @staticmethod
    def __static_init__(c):
        ## Skip base class
        if c.__name__.endswith('Base'):
            return
        types[c.name] = c()

    def is_simd(self):
        return False

    def is_volatile(self):
        return False

class TypeVectorBase(TypeBase):
    def is_simd(self):
        return True

# -----------------------------------------------------------------------------

class TypeVoid(TypeBase):
    name = '_'

    def as_type(self, typ):
        return 'void'

# -----------------------------------------------------------------------------

class TypeScalar(TypeBase):
    name = 's'

    def as_type(self, typ):
        return typ

    def code_load(self, simd, typ, ptr):
        return '*({})'.format(ptr)

    def code_store(self, simd, typ, lhs, rhs):
        return '*({}) = {}'.format(lhs, rhs)

# -----------------------------------------------------------------------------

class TypeVolatileScalar(TypeScalar):
    name = 'volatile-s'

    def is_volatile(self):
        return True

# -----------------------------------------------------------------------------

class TypeLogicalScalar(TypeBase):
    name = 'ls'

    def as_type(self, typ):
        return {
            'i8': 'u8',
            'i16': 'u16',
            'i32': 'u32',
            'i64': 'u64',
            'f32': 'u32',
            'f64': 'u64',
            }.get(typ, typ)

    def code_load(self, simd, typ, ptr):
        return '({})(*({}))'.format(self.as_type(typ), ptr)

    def code_store(self, simd, typ, lhs, rhs):
        return '*({}) = ({})({})'.format(lhs, typ, rhs)

# -----------------------------------------------------------------------------

class TypeVolatileLogicalScalar(TypeLogicalScalar):
    name = 'volatile-ls'

    def is_volatile(self):
        return True

# -----------------------------------------------------------------------------

class TypeInt(TypeScalar):
    name = 'p'

    def as_type(self, typ):
        return 'int'

# -----------------------------------------------------------------------------

class TypePtr(TypeBase):
    name = '*'

    def as_type(self, typ):
        return typ + '*'

# -----------------------------------------------------------------------------

class TypeConstPtr(TypeBase):
    name = 'c*'

    def as_type(self, typ):
        return 'const ' + typ + '*'

# -----------------------------------------------------------------------------

class TypeVector(TypeVectorBase):
    name = 'v'

    def as_type(self, typ):
        return 'v' + typ

    def code_load(self, simd, typ, ptr):
        return 'nsimd::loada({}, {}())'.format(ptr, typ)

    def code_store(self, simd, typ, ptr, expr):
        return 'nsimd::storea({}, {}, {}())'.format(ptr, expr, typ)

# -----------------------------------------------------------------------------

class TypeCPUVector(TypeVector):
    name = 'vcpu'

    def code_load(self, simd, typ, ptr):
        return 'nsimd::loada({}, {}(), nsimd::cpu())'.format(ptr, typ)

    def code_store(self, simd, typ, ptr, expr):
        return 'nsimd::storea({}, {}, {}(), nsimd::cpu())'.format(ptr, expr, typ)

# -----------------------------------------------------------------------------

class TypeUnrolledVectorBase(TypeVectorBase):
    def as_type(self, typ):
        raise NotImplemented()

    def code_load(self, simd, typ, ptr):
        return 'nsimd::loada<nsimd::pack<{}, {}>>({})'. \
               format(typ, self.unroll, ptr)

    def code_store(self, simd, typ, ptr, expr):
        return 'nsimd::storea({}, {})'.format(ptr, expr)

# -----------------------------------------------------------------------------

class TypeUnrolledVector1(TypeUnrolledVectorBase):
    name = 'vu1'
    unroll = 1

class TypeUnrolledVector2(TypeUnrolledVectorBase):
    name = 'vu2'
    unroll = 2

class TypeUnrolledVector3(TypeUnrolledVectorBase):
    name = 'vu3'
    unroll = 3

class TypeUnrolledVector4(TypeUnrolledVectorBase):
    name = 'vu4'
    unroll = 4

class TypeUnrolledVector5(TypeUnrolledVectorBase):
    name = 'vu5'
    unroll = 5

class TypeUnrolledVector6(TypeUnrolledVectorBase):
    name = 'vu6'
    unroll = 6

class TypeUnrolledVector7(TypeUnrolledVectorBase):
    name = 'vu7'
    unroll = 7

class TypeUnrolledVector8(TypeUnrolledVectorBase):
    name = 'vu8'
    unroll = 8

class TypeUnrolledVector9(TypeUnrolledVectorBase):
    name = 'vu9'
    unroll = 9

# -----------------------------------------------------------------------------

class TypeVectorX2(TypeVectorBase):
    name = 'vx2'

    def as_type(self, typ):
        return 'v' + typ + 'x2'

# -----------------------------------------------------------------------------

class TypeVectorX3(TypeVectorBase):
    name = 'vx3'

    def as_type(self, typ):
        return 'v' + typ + 'x3'

# -----------------------------------------------------------------------------

class TypeVectorX4(TypeVectorBase):
    name = 'vx4'

    def as_type(self, typ):
        return 'v' + typ + 'x4'

# -----------------------------------------------------------------------------

class TypeLogical(TypeVectorBase):
    name = 'l'

    def as_type(self, typ):
        return 'vl' + typ

    def code_load(self, simd, typ, ptr):
        return 'nsimd::loadla({}, {}())'.format(ptr, typ)

    def code_store(self, simd, typ, ptr, expr):
        return 'nsimd::storela({}, {}, {}())'.format(ptr, expr, typ)

# -----------------------------------------------------------------------------

class TypeCPULogical(TypeLogical):
    name = 'lcpu'

    def code_load(self, simd, typ, ptr):
        return 'nsimd::loadla({}, {}(), nsimd::cpu())'.format(ptr, typ)

    def code_store(self, simd, typ, ptr, expr):
        return 'nsimd::storela({}, {}, {}(), nsimd::cpu())'.format(ptr, expr, typ)

# -----------------------------------------------------------------------------

class TypeUnrolledLogicalBase(TypeVectorBase):
    def as_type(self, typ):
        raise NotImplemented()

    def code_load(self, simd, typ, ptr):
        return 'nsimd::loadla<nsimd::packl<{}, {}>>({})'. \
               format(typ, self.unroll, ptr)

    def code_store(self, simd, typ, ptr, expr):
        return 'nsimd::storela({}, {})'.format(ptr, expr)

# -----------------------------------------------------------------------------

class TypeUnrolledLogical1(TypeUnrolledLogicalBase):
    name = 'lu1'
    unroll = 1

class TypeUnrolledLogical2(TypeUnrolledLogicalBase):
    name = 'lu2'
    unroll = 2

class TypeUnrolledLogical3(TypeUnrolledLogicalBase):
    name = 'lu3'
    unroll = 3

class TypeUnrolledLogical4(TypeUnrolledLogicalBase):
    name = 'lu4'
    unroll = 4

class TypeUnrolledLogical5(TypeUnrolledLogicalBase):
    name = 'lu5'
    unroll = 5

class TypeUnrolledLogical6(TypeUnrolledLogicalBase):
    name = 'lu6'
    unroll = 6

class TypeUnrolledLogical7(TypeUnrolledLogicalBase):
    name = 'lu7'
    unroll = 7

class TypeUnrolledLogical8(TypeUnrolledLogicalBase):
    name = 'lu8'
    unroll = 8

class TypeUnrolledLogical9(TypeUnrolledLogicalBase):
    name = 'lu9'
    unroll = 9

# -----------------------------------------------------------------------------

class TypeBoostSimdVector(TypeVectorBase):
    name = 'boost::simd::pack'

    def as_type(self, typ):
        return 'boost::simd::pack<{}>'.format(typ)

    def code_load(self, simd, typ, ptr):
        return '{}({})'.format(self.as_type(typ), ptr)

    def code_store(self, simd, typ, ptr, expr):
        return 'nsimd::storea({}, {}, {}())'.format(ptr, expr, typ)

# -----------------------------------------------------------------------------

class TypeBoostSimdLogicalVector(TypeVectorBase):
    name = 'boost::simd::lpack'

    def as_type(self, typ):
        return 'boost::simd::pack<boost::simd::logical<{}>>'.format(typ)

    def code_load(self, simd, typ, ptr):
        return '{}({})'.format(self.as_type(typ), ptr)

    def code_store(self, simd, typ, ptr, expr):
        return 'nsimd::storea({}, {}, {}())'.format(ptr, expr, typ)

# -----------------------------------------------------------------------------

class TypeMIPPReg(TypeVectorBase):
    name = 'mipp::reg'

    def as_type(self, typ):
        return 'mipp::Reg<{}>'.format(typ)

    def code_load(self, simd, typ, ptr):
        return 'mipp::load<{}>({})'.format(typ, ptr)

    def code_store(self, simd, typ, ptr, expr):
        return 'mipp::store({}, {})'.format(ptr, expr)

# -----------------------------------------------------------------------------

class TypeMIPPMsk(TypeVectorBase):
    name = 'mipp::msk'

    def as_type(self, typ):
        return 'mipp::Msk<{}>'.format(typ)

    def code_load(self, simd, typ, ptr):
        if simd in ['avx512_knl', 'avx512_skylake']:
            return '*({})'.format(ptr)
        else:
            return 'mipp::load<{}>({})'.format(typ, ptr)

    def code_store(self, simd, typ, ptr, expr):
        if simd in ['avx512_knl', 'avx512_skylake']:
            return '*({}) = {}'.format(ptr, expr)
        else:
            return 'mipp::store({}, {})'.format(ptr, expr)

# -----------------------------------------------------------------------------

def type_of(param):
    if param in types:
        return types[param]
    else:
        raise BenchError("Unable to find corresponding type for: " + param)

def as_type(param, typ):
    return type_of(param).as_type(typ)

# -----------------------------------------------------------------------------
# Operator class needs to be reinforced for benches

class BenchOperator(object, metaclass=type):
    def __init__(self):
        self.typed_params_ = []
        for p in self.params:
            self.typed_params_.append(type_of(p))

    @property
    def function_name(self):
        return self.name.split('::')[-1].split('<')[0]

    ## Generates list of includes to be included
    def gen_includes(self, lang):
        includes = []
        includes.append('<nsimd/nsimd.h>')
        if lang == 'cxx_adv':
            includes.append('<nsimd/cxx_adv_api.hpp>')
        if lang == 'c_base':
            includes += ['<stdlib.h>', '<stdio.h>', '<errno.h>', '<string.h>']
        else:
            includes += ['<cstdlib>', '<cstdio>', '<cerrno>', '<cstring>']
        return includes

    def match_sig(self, signature):
        (name, params) = common.parse_signature(signature)
        if len(params) != len(self.params):
            return False
        for p1, p2 in zip(params, self.params):
            if p1 != p2:
                return False
        return True

    def bench_code_before(self, typ):
        return ''

    def bench_against_init(self):
        bench = {}
        for simd in ['*'] + common.simds:
            bench[simd] = OrderedDict()
            for typ in ['*'] + common.types:
                bench[simd][typ] = OrderedDict()
        return bench

    def bench_against_cpu(self):
        bench = self.bench_against_init()
        ## Enable bench against nsimd (cpu architecture)
        if self.bench_auto_against_cpu:
            bench['*']['*'][common.nsimd_category('cpu')] = \
                    cpu_fun_from_sig(sig_translate(self.signature, {
                                     's': 'volatile-s',
                                     'v': 'vcpu',
                                     'l': 'lcpu',
                                     }))
        return bench

    def bench_against_libs(self):
        bench = self.bench_against_init()
        ## Enable bench against all other libraries
        if self.bench_auto_against_mipp:
            for typ in self.bench_mipp_types():
                ## MIPP always requires template
                mipp_name = self.bench_mipp_name(typ)
                signature = sig_translate(self.signature, {
                    'v': 'mipp::reg',
                    'l': 'mipp::msk',
                    }, name=mipp_name)
                if signature:
                    bench['*'][typ]['MIPP'] = signature
        if self.bench_auto_against_sleef:
            for simd in common.simds:
                for typ in self.bench_sleef_types():
                    if not common.sleef_support_type(simd, typ):
                        continue
                    sleef_name = self.bench_sleef_name(simd, typ)
                    if sleef_name is None:
                        continue
                    ## IMPORTANT:
                    ## If simd is cpu, then make the signature using scalar
                    if simd == 'cpu':
                        signature = sig_translate(self.signature, {
                            's': 'volatile-s',
                            'v': 'volatile-s',
                            'l': 'volatile-s',
                            }, sleef_name)
                    else:
                        signature = sig_translate(self.signature, {},
                                                        sleef_name)
                    if signature:
                        bench[simd][typ]['Sleef'] = signature
        if self.bench_auto_against_std:
            for simd in common.simds:
                for typ in self.bench_std_types():
                    std_name = self.bench_std_name(simd, typ)
                    signature = sig_translate(self.signature, {
                        's': 'volatile-s',
                        'v': 'volatile-s',
                        'l': 'volatile-s',
                        }, std_name)
                    if signature:
                        if self.cxx_operator:
                            bench[simd][typ]['std'] = std_operator_from_sig(signature,
                                    self.cxx_operator_symbol)
                        else:
                            bench[simd][typ]['std'] = std_fun_from_sig(signature)
        return bench

    def code_call(self, typ, args):
        return 'nsimd::{}({}, {}())'.format(self.name,
                                            common.pprint_commas(args), typ)

    def code_ptr_step(self, typ):
        if any(p.is_simd() for p in self.typed_params_):
            return 'vlen({})'.format(typ)
        else:
            return '1'

class BenchOperatorWithNoMakers(BenchOperator):
    use_for_parsing = False

    # Classes that inherit from me do not have their name member
    # which is mandatory so I fill it for them here.
    def __init__(self):
        BenchOperator.__init__(self)
        (self.name, void) = common.parse_signature(self.signature)

# -----------------------------------------------------------------------------
# Make the list of all operators, they will inherit from the corresponding
# operators.Operator and then from BenchOperator

functions = {}

class dummy(operators.MAddToOperators):
    def __new__(cls, name, bases, dct):
        return type.__new__(cls, name, bases, dct)

for op_name, operator in operators.operators.items():
    if operator.load_store: # We do not bench loads/stores
        continue
    op_class = dummy(operator.__class__.__name__,
                     (operator.__class__, BenchOperator), {})
    functions[op_name] = op_class()

# -----------------------------------------------------------------------------
# Function helpers

def nsimd_unrolled_fun_from_sig(from_sig, unroll):
    sig = sig_translate(from_sig, {
        'v': 'vu' + str(unroll),
        'l': 'lu' + str(unroll),
        })
    class InlineNSIMDUnrolledFun(operators.Operator, BenchOperatorWithNoMakers,
                                 metaclass=dummy):
        signature = sig
        def code_call(self, typ, args):
            return 'nsimd::{}({})'.format(self.name,
                                          common.pprint_commas(args))
        def code_ptr_step(self, typ):
            return 'nsimd::len(nsimd::pack<{}, {}>())'.format(typ, unroll)
    return InlineNSIMDUnrolledFun()

def fun_from_sig(from_sig):
    class InlineFun(operators.Operator, BenchOperatorWithNoMakers,
                    metaclass=dummy):
        signature = from_sig
        def code_call(self, typ, args):
            return '{}({})'.format(self.name, common.pprint_commas(args))
    return InlineFun()

def std_fun_from_sig(from_sig):
    return fun_from_sig(from_sig)

def std_operator_from_sig(from_sig, op):
    class InlineStdOperatorFun(operators.Operator, BenchOperatorWithNoMakers,
                               metaclass=dummy):
        __metaclass__ = dummy
        signature = from_sig
        operator = op
        def code_call(self, typ, args):
            if len(args) == 1:
                return '{}({})'.format(self.operator, args[0])
            elif len(args) == 2:
                return '{} {} {}'.format(args[0], self.operator, args[1])
            else:
                raise BenchError('std:: operators requires 1 or 2 arguments!')
    return InlineStdOperatorFun()

def cpu_fun_from_sig(from_sig):
    class InlineCPUFun(operators.Operator, BenchOperatorWithNoMakers,
                       metaclass=dummy):
        signature = from_sig
        def code_call(self, typ, args):
            return 'nsimd::{}({}, {}(), nsimd::cpu())'. \
                   format(self.name, common.pprint_commas(args), typ)
    return InlineCPUFun()

def sanitize_fun_name(name):
    return ''.join(map(lambda c: c if c.isalnum() else '_', name))

# -----------------------------------------------------------------------------
# Code

def code_cast(typ, expr):
    return '({})({})'.format(typ, expr)

def code_cast_ptr(typ, expr):
    return code_cast(typ + '*', expr)

# -----------------------------------------------------------------------------
# Globals

_opts = None
_lang = 'cxx_adv'

# -----------------------------------------------------------------------------
# Generates

def TODO(f):
    if _opts.verbose:
        print('-- @@ TODO: ' + f.name)

def gen_filename(f, simd, typ):
    ## Retrieve directory from global options
    benches_dir = common.mkdir_p(os.path.join(_opts.benches_dir, _lang))
    ## Generate path (composed from: function name + type + extension)
    return os.path.join(benches_dir, '{}.{}.{}.{}'.format(
        f.name, simd, typ, common.ext_from_lang(_lang)))

def gen_bench_name(category, name, unroll=None):
    bench_name = '{}_{}'.format(category, name)
    if unroll:
        bench_name += '_unroll{}'.format(unroll)
    return bench_name

def gen_bench_from_code(f, typ, code):
    header = ''
    header += common.pprint_includes(f.gen_includes(_lang))
    header += \
    '''

    // Required for random generation
    #include "../benches.hpp"

    // Google benchmark
    #include <benchmark/benchmark.h>

    // std
    #include <cmath>

    // Sleef
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wignored-qualifiers"
    #include <sleef.h>
    #pragma GCC diagnostic pop

    // MIPP
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wconversion"
    #pragma GCC diagnostic ignored "-Wsign-conversion"
    #pragma GCC diagnostic ignored "-Wdouble-promotion"
    #pragma GCC diagnostic ignored "-Wunused-parameter"
    #if defined(__clang__)
    #pragma GCC diagnostic ignored "-Wzero-length-array"
    #endif
    #include <mipp.h>
    #pragma GCC diagnostic pop
    '''
    return \
    '''{header}

    // -------------------------------------------------------------------------

    static const int sz = 1024;

    template <typename Random>
    static {type}* make_data(int sz, Random r) {{
      {type}* data = ({type}*)nsimd_aligned_alloc(sz * {sizeof});

      for (int i = 0; i < sz; ++i) {{
        data[i] = r();
      }}
      return data;
    }}

    static {type}* make_data(int sz) {{
      {type}* data = ({type}*)nsimd_aligned_alloc(sz * {sizeof});

      for (int i = 0; i < sz; ++i) {{
        data[i] = {type}(0);
      }}
      return data;
    }}

    {random_code}

    // -------------------------------------------------------------------------

    {code}

    BENCHMARK_MAIN();

    '''.format(
            name=f.name,
            type=typ,
            year=date.today().year,
            code=code,
            random_code=f.domain.code('rand_param', typ),
            sizeof=common.sizeof(typ),
            header=header,
    )

def gen_bench_info_from(f, simd, typ):
    bench_args_init = []
    bench_args_decl = []
    bench_args_call = []
    ## Generate code for parameters
    for i, arg in enumerate(f.args):
        p = type_of(arg)
        qualifiers = ''
        if p.is_volatile():
            qualifiers += 'volatile '
        bench_args_init.append('make_data(sz, &rand_param{n})'.format(n=i))
        bench_args_decl.append('{} {}* _{}'.format(qualifiers, typ, i))
        bench_args_call.append(p.code_load(simd, typ, '_{} + i'.format(i)))
    ## Generate code for bench (using function return type)
    r = type_of(f.get_return())
    bench_call = r.code_store(simd, typ, '_r + i',
                              f.code_call(typ, bench_args_call))
    return bench_args_init, bench_args_decl, bench_args_call, bench_call

def gen_bench_asm_function(f, simd, typ, category):
    bench_args_init, bench_args_decl, \
    bench_args_call, bench_call = gen_bench_info_from(f, simd, typ)
    ## Add function that can easily be parsed to get assembly and plain code
    return \
    '''
    void {bench_name}__asm__({type}* _r, {bench_args_decl}, int sz) {{
      __asm__ __volatile__("nop");
      __asm__ __volatile__("nop");
      __asm__ __volatile__("nop");
      __asm__ __volatile__("nop");
      __asm__ __volatile__("nop");
      __asm__ __volatile__("nop");
      // code:{{
      int n = {step};
      #pragma clang loop unroll(disable)
      for (int i = 0; i < sz; i += n) {{
        {bench_call};
      }}
      // code:}}
      __asm__ __volatile__("nop");
      __asm__ __volatile__("nop");
      __asm__ __volatile__("nop");
      __asm__ __volatile__("nop");
      __asm__ __volatile__("nop");
      __asm__ __volatile__("nop");
    }}
    '''.format(
        bench_name=gen_bench_name(category, f.function_name),
        type=typ,
        step=f.code_ptr_step(typ),
        bench_call=bench_call,
        bench_args_decl=common.pprint_commas(bench_args_decl)
        )

def gen_bench_from_basic_fun(f, simd, typ, category, unroll=None):
    bench_args_init, bench_args_decl, bench_args_call, bench_call = \
            gen_bench_info_from(f, simd, typ)
    bench_name = gen_bench_name(category, f.function_name, unroll)
    return \
    '''
    {code_before}
    extern "C" {{ void __asm_marker__{bench_name}() {{}} }}

    void {bench_name}(benchmark::State& state, {type}* _r, {bench_args_decl}, int sz) {{
      // Normalize size depending on the step so that we're not going out of boundaies
      // (Required when the size is'nt a multiple of `n`, like for unrolling benches)
      sz = (sz / {step}) * {step};
      try {{
        for (auto _ : state) {{
          {asm_marker}
          // code: {bench_name}
          int n = {step};

          #pragma clang loop unroll(disable)
          for (int i = 0; i < sz; i += n) {{
            {bench_call};
          }}
          // code: {bench_name}
          {asm_marker}
        }}
      }} catch (std::exception const& e) {{
        std::string message("ERROR: ");
        message += e.what();
        state.SkipWithError(message.c_str());
      }}
    }}
    BENCHMARK_CAPTURE({bench_name}, {type}, make_data(sz), {bench_args_init}, sz);
    '''.format(
            bench_name=bench_name,
            type=typ,
            step=f.code_ptr_step(typ),
            bench_call=bench_call,
            bench_args_init=common.pprint_commas(bench_args_init),
            bench_args_decl=common.pprint_commas(bench_args_decl),
            bench_args_call=common.pprint_commas(bench_args_call),
            code_before=f.bench_code_before(typ),
            asm_marker=asm_marker(simd, bench_name)
            )

def gen_code(f, simd, typ, category):
    code = None
    if f.returns_any_type:
        return TODO(f)
    ## TODO: We have to refactor this, it's annoying to add every possible signatures...
    if f.match_sig('v * v v') or f.match_sig('v * v v v') \
        or f.match_sig('l * v v') or f.match_sig('l * l l') \
        or f.match_sig('l * l') or f.match_sig('v * v') \
        or f.match_sig('s * s') \
        or f.match_sig('s * s s') \
        or f.match_sig('s * s s s') \
        or f.match_sig('vcpu * vcpu') \
        or f.match_sig('vcpu * vcpu vcpu') \
        or f.match_sig('vcpu * vcpu vcpu vcpu') \
        or f.match_sig('lcpu * lcpu') \
        or f.match_sig('lcpu * lcpu lcpu') \
        or f.match_sig('lcpu * vcpu vcpu') \
        or f.match_sig('vcpu * lcpu vcpu vcpu') \
        or f.match_sig('volatile-s * volatile-s') \
        or f.match_sig('volatile-s * volatile-s volatile-s') \
        or f.match_sig('volatile-s * volatile-s volatile-s volatile-s') \
        or f.match_sig('volatile-ls * volatile-s') \
        or f.match_sig('volatile-ls * volatile-s volatile-s') \
        or f.match_sig('volatile-ls * volatile-ls') \
        or f.match_sig('volatile-ls * volatile-ls volatile-ls') \
        or f.match_sig('volatile-s * volatile-ls volatile-s volatile-s') \
        or f.match_sig('boost::simd::pack * boost::simd::pack') \
        or f.match_sig('boost::simd::pack * boost::simd::pack boost::simd::pack') \
        or f.match_sig('boost::simd::pack * boost::simd::pack boost::simd::pack boost::simd::pack') \
        or f.match_sig('boost::simd::lpack * boost::simd::pack') \
        or f.match_sig('boost::simd::lpack * boost::simd::pack boost::simd::pack') \
        or f.match_sig('mipp::reg * mipp::reg') \
        or f.match_sig('mipp::reg * mipp::reg mipp::reg') \
        or f.match_sig('mipp::msk * mipp::reg') \
        or f.match_sig('mipp::msk * mipp::reg mipp::reg') \
        or f.match_sig('v * l v v'):
        code = gen_bench_from_basic_fun(f, simd, typ, category=category)
    if f.match_sig('p * l'):
        return TODO(f)
    if f.match_sig('v * s'):
        return TODO(f)
    if f.match_sig('p *'):
        return TODO(f)
    if f.match_sig('v * v p'):
        return TODO(f)
    if code is None:
        raise BenchError('Unable to generate bench for signature: ' + \
                         f.signature)
    return code

def gen_bench_unrolls(f, simd, typ, category):
    code = ''
    sig = f.signature
    for unroll in [2, 3, 4]:
        f = nsimd_unrolled_fun_from_sig(sig, unroll)
        code += gen_bench_from_basic_fun(f, simd, typ, category=category,
                                         unroll=unroll)
    return code

def gen_bench_against(f, simd, typ, against):
    code = ''
    # "against" dict looks like: { simd: { type: { name: sig } } }
    for s in [simd, '*']:
        if not s in against:
            continue
        for t in [typ, '*']:
            if not t in against[s]:
                continue
            for category, f in against[s][t].items():
                # Allow function to be simple str (you use this most of the
                # time)
                if isinstance(f, str):
                    f = fun_from_sig(f)
                # Now that we have a `Fun` type, we can generate code
                code += gen_code(f, simd, typ, category=category)
    return code

def gen_bench(f, simd, typ):
    ## TODO
    path = gen_filename(f, simd, typ)
    ## Check if we need to create the file
    if not common.can_create_filename(_opts, path):
        return
    ## Generate specific code for the bench
    category = common.nsimd_category(simd)
    code = gen_code(f, simd, typ, category=category)
    if code is None:
        return
    ## Now aggregate every parts
    bench = ''
    #bench += gen_bench_asm_function(f, typ, category)
    bench += gen_bench_against(f, simd, typ, f.bench_against_cpu())
    bench += code
    bench += gen_bench_unrolls(f, simd, typ, category)
    bench += gen_bench_against(f, simd, typ, f.bench_against_libs())
    ## Finalize code
    code = gen_bench_from_code(f, typ, bench)
    ## Write file
    with common.open_utf8(path) as f:
        f.write(code)
    ## Clang-format it!
    common.clang_format(_opts, path)

# -----------------------------------------------------------------------------
# Entry point

def doit(opts):
    global _opts
    _opts = opts
    print ('-- Generating benches')
    for f in functions.values():
        if not f.do_bench:
            if opts.verbose:
                print('-- Skipping bench: {}'.format(f.name))
            continue
        # WE MUST GENERATE CODE FOR EACH SIMD EXTENSION AS OTHER LIBRARY
        # USUALLY DO NOT PROPOSE A GENERIC INTERFACE
        for simd in _opts.simd:
            ## FIXME
            if simd in ['neon128', 'cpu']:
                continue
            for typ in f.types:
                ## FIXME
                if typ == 'f16':
                    continue
                ## Skip non-matching benches
                if opts.match and not opts.match.match(f.name):
                    continue
                ## FIXME
                if f.name in ['gamma', 'lgamma']:
                    continue
                gen_bench(f, simd, typ)
