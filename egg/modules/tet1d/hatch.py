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
import operators
import common
import cuda
import gen_scalar_utilities
#import hip

# -----------------------------------------------------------------------------
# CUDA: default number of threads per block

tpb = 128
gpu_params = '(n + {}) / {}, {}'.format(tpb, tpb - 1, tpb)

def is_not_closed(operator):
    return (operator.output_to == common.OUTPUT_TO_SAME_SIZE_TYPES \
            or ('v' not in operator.params[1:] and 'l' not in
            operator.params[1:]))

# -----------------------------------------------------------------------------

def gen_doc_overview(opts):
    filename = common.get_markdown_file(opts, 'overview', 'tet1d')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as fout:
        fout.write('''# Overview

## What are expression templates?

Expression templates are a C++ template metaprogramming technique that
essentially allows high level programming for loop fusion. Take the following
exemple.

```c++
std::vector<float> operator+(std::vector<float> const &a,
                             std::vector<float> const &b) {{
  std::vector<float> ret(a.size());
  for (size_t i = 0; i < a.size(); i++) {{
    ret[i] = a[i] + b[i];
  }}
  return ret;
}}

int main() {{
  std::vector<float> a, b, c, d, sum;

  ...

  sum = a + b + c + d;

  ...

  return 0;
}}
```

The expression `a + b + c + d` involves three calls to `operator+` and at least
nine memory passes are necessary. This can be optimized as follows.

```c++
int main() {{
  std::vector<float> a, b, c, d, sum;

  ...

  for (size_t i = 0; i < a.size(); i++) {{
    ret[i] = a[i] + b[i] + c[i] + d[i];
  }}

  ...

  return 0;
}}
```

The rewriting above requires only four memory passes which is of course better
but as humans we prefer the writing `a + b + c + d`. Expression templates
solves exactly this problem and allows the programmer to write `a + b + c + d`
and the compiler to see the loop written above.

## Expressions templates with NSIMD

This module provides expression templates on top of NSIMD core. As a
consequence the loops seen by the compiler deduced from the high-level
expressions are optimized using SIMD instructions. Note also that NVIDIA and
AMD GPUs are supported through CUDA and ROCm/HIP. The API for expression
templates in NSIMD is C++98 compatible and is able to work with any container
as its only requirement for data is that it must be contiguous.

All inputs to an expression must be declared using `tet1d::in` while the
output must be declared using `tet1d::out`.

```c++
int main() {{
  std::vector<float> a, b, c;

  ...

  tet1d::out(a) = tet1d::in(&a[0], a.size()) + tet1d::in(&b[0], b.size());

  ...

  return 0;
}}
```

- `template <typename T, typename I> inline node in(const T *data, I sz);`{nl}
  Construct an input for expression templates starting at address `data` and
  containing `sz` elements. The return type of this functin `node` can be used
  with the help of the `TET1D_IN(T)` macro where `T` if the underlying type of
  data (ints, floats, doubles...).

- `template <typename T> node out(T *data);`{nl}
  Construct an output for expression templates starting at address `data`. Note
  that memory must be allocated by the user before passing it to the expression
  template engine. The output type can be used with the `TET1D_OUT(T)` where
  `T` is the underlying type (ints, floats, doubles...).

Note that it is possible to pass parameters to the expression template engine
to specify the number of threads per block for GPUs or the SIMD extension to
use...

- `template <typename T, typename Pack> node out(T *data, int
  threads_per_block, void *stream);`{nl}
  Construct an output for expression templates starting at address `data`. Note
  that memory must be allocated by the user before passing it to the expression
  template engine. The `Pack` parameter is useful when compiling for CPUs. The
  type is `nsimd::pack<...>` allowing the developper to specify all details
  about the NSIMD packs that will be used by the expression template engine.
  The `threads_per_block` and `stream` arguments are used only when compiling
  for GPUs. Their meaning is contained in their names. The output type can be
  used with the `TET1D_OUT_EX(T, N, SimdExt)` where `T` is the underlying type
  (ints, floats, doubles...), `N` is the unroll factor and `SimdExt` the SIMD
  extension.

Moreover a MATLAB-like syntax is provided. One can select a subrange of given
input. Indexes are understood as for Python: -1 represents the last element.
The contant `tet1d::end = -1` allows one to write portable code.

```c++
int main() {{
  std::vector<float> a, b, c;

  ...

  TET1D_IN(float) va = tet1d::in(&a[0], a.size());
  TET1D_IN(float) vb = tet1d::in(&b[0], b.size());
  tet1d::out(c) = va(10, tet1d::end - 10) + vb;

  ...

  return 0;
}}
```

One can also specify which elements of the output must be rewritten with
the following syntax.

```c++
int main() {{
  std::vector<float> a, b, c;

  ...

  TET1D_IN(float) va = tet1d::in(&a[0], a.size());
  TET1D_IN(float) vb = tet1d::in(&b[0], b.size());
  TET1D_OUT(float) vc = tet1d::out(&c[0]);
  vc(va >= 10 && va < 20) = vb;

  ...

  return 0;
}}
```

In the exemple above, element `i` in `vc` is written only if `va[i] >= 10` and
`va[i] < 20`. The expression appearing in the parenthesis can contain
arbitrary expression templates as soon as the underlying type is `bool`.

## Warning using `auto`

Using auto can lead to surprising results. We advice you never to use auto
when dealing with expression templates. Indeed using `auto` will make the
variable an obscure type representing the computation tree of the expression
template. This implies that you won't be able to get data from this variable
i.e. get the `.data` member for exemple. Again this variable or its type cannot
be used in template arguments where you need it.
'''.format(nl='  '))

# -----------------------------------------------------------------------------

def gen_doc_api(opts):
    filename = common.get_markdown_file(opts, 'api', 'tet1d')
    if not common.can_create_filename(opts, filename):
        return

    # Build tree for api.md
    api = dict()
    for _, operator in operators.operators.items():
        if not operator.has_scalar_impl:
            continue
        for c in operator.categories:
            if c not in api:
                api[c] = [operator]
            else:
                api[c].append(operator)

    def get_signature(op):
        def get_type(typ):
            if typ == 'p':
                return 'int'
            elif typ == 'v':
                return 'ExprNumber'
            elif typ == 'l':
                return 'ExprBool'
        ret = get_type(op.params[0]) + ' ' + op.name + '('
        if is_not_closed(op):
            ret += 'ToType' + (', ' if len(op.params[1:]) > 0 else '')
        ret += ', '.join(['{{t}} {{in{i}}}'.format(i=i). \
                          format(t=get_type(op.params[i + 1]), in0=common.in0,
                          in1=common.in1, in2=common.in2, in3=common.in3) \
                          for i in range(len(op.params[1:]))])
        ret += ');'
        return ret

    with common.open_utf8(opts, filename) as fout:
        fout.write(
'''# NSIMD TET1D API reference

This page contains the exhaustive API of the TET1D module. Note that most
operators names follow their NSIMD counterparts and have the same
semantics. This page is light, you may use CTRL+F to find the operator you
are looking for.

Note that all operators accept literals and scalars. For example you may
write `tet1d::add(a, 1)`. This also applies when using infix operators. Note
that literals or scalars of different types can be used with expression
involving other types.

In all signature below the following pseudo types are used for simplification:
- `ExprNumber` to designate an existing expression template on signed, unsigned
  integers of floatting point types or a scalar of signed, unsigned integers or
  floatting point types.
- `ExprBool` to designate an existing expression template over booleans or
  a boolean.
- `ToType` to designate a base type (signed, unsigned integers or floatting
  point types) and is used when a change in type is requested for example
  when converting data.

''')

        for c, ops in api.items():
            if len(ops) == 0:
                continue
            fout.write('\n## {}\n\n'.format(c.title))
            for op in ops:
                fout.write('- `{}`  \n'.format(get_signature(op)))
                if op.cxx_operator != None:
                    fout.write('  Infix operator: `{}`  \n'. \
                               format(op.cxx_operator[8:]))
                fout.write('  {}\n\n'.format(op.desc))

# -----------------------------------------------------------------------------

def gen_tests_for_shifts(opts, t, operator):
    op_name = operator.name
    dirname = os.path.join(opts.tests_dir, 'modules', 'tet1d')
    common.mkdir_p(dirname)
    filename = os.path.join(dirname, '{}.{}.cpp'.format(op_name, t))
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as out:
        out.write(
        '''#include <nsimd/modules/tet1d.hpp>
        #include <nsimd/modules/memory_management.hpp>
        #include "../common.hpp"

        #if defined(NSIMD_CUDA)

        __global__ void kernel({t} *dst, {t} *tab0, int n, int s) {{
          int i = threadIdx.x + blockIdx.x * blockDim.x;
          if (i < n) {{
            dst[i] = nsimd::gpu_{op_name}(tab0[i], s);
          }}
        }}

        void compute_result({t} *dst, {t} *tab0, unsigned int n, int s) {{
          kernel<<<{gpu_params}>>>(dst, tab0, int(n), s);
        }}

        #elif defined(NSIMD_ROCM)

        __global__ void kernel({t} *dst, {t} *tab0, size_t n, int s) {{
          size_t i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
          if (i < n) {{
            dst[i] = nsimd::gpu_{op_name}(tab0[i], s);
          }}
        }}

        void compute_result({t} *dst, {t} *tab0, size_t n, int s) {{
          hipLaunchKernelGGL(kernel, {gpu_params}, 0, 0, dst, tab0, n, s);
        }}

        #else

        void compute_result({t} *dst, {t} *tab0, unsigned int n, int s) {{
          for (unsigned int i = 0; i < n; i++) {{
            dst[i] = nsimd_scalar_{op_name}_{t}(tab0[i], s);
          }}
        }}

        #endif

        nsimd_fill_dev_mem_func(prng5,
            1 + (((unsigned int)i * 69342380 + 414585) % 5))

        int main() {{
          unsigned int n_[3] = {{ 10, 1001, 10001 }};
          for (int i = 0; i < (int)(sizeof(n_) / sizeof(int)); i++) {{
            unsigned int n = n_[i];
            for (int s = 0; s < {typnbits}; s++) {{
              int ret = 0;
              {t} *tab0 = nsimd::device_calloc<{t}>(n);
              prng5(tab0, n);
              {t} *ref = nsimd::device_calloc<{t}>(n);
              {t} *out = nsimd::device_calloc<{t}>(n);
              compute_result(ref, tab0, n, s);
              tet1d::out(out) = tet1d::{op_name}(tet1d::in(tab0, n), s);
              if (!cmp(ref, out, n)) {{
                ret = -1;
              }}
              nsimd::device_free(ref);
              nsimd::device_free(out);
              nsimd::device_free(tab0);
              if (ret != 0) {{
                return ret;
              }}
            }}
          }}
          return 0;
        }}
        '''.format(gpu_params=gpu_params, op_name=op_name, t=t,
                   typnbits=t[1:]))
    common.clang_format(opts, filename, cuda=True)

def gen_tests_for(opts, t, tt, operator):
    op_name = operator.name
    dirname = os.path.join(opts.tests_dir, 'modules', 'tet1d')
    common.mkdir_p(dirname)
    filename = os.path.join(dirname, '{}.{}.cpp'.format(op_name,
               t if t == tt else '{}_{}'.format(t, tt)))
    if not common.can_create_filename(opts, filename):
        return

    arity = len(operator.params[1:])
    args_tabs = ', '.join(['{typ} *tab{i}'.format(typ=t, i=i) \
                           for i in range(arity)])
    args_tabs_call = ', '.join(['tab{i}'.format(i=i) \
                                for i in range(arity)])
    args_tabs_i_call = ', '.join(['tab{i}[i]'.format(i=i) \
                                  for i in range(arity)])
    args_in_tabs_call = ', '.join(['tet1d::in(tab{i}, n)'. \
                                   format(i=i) \
                                   for i in range(arity)])

    fill_tabs = '\n'.join(['{typ} *tab{i} = nsimd::device_calloc<{typ}>(n);\n' \
                           'prng{ip5}(tab{i}, n);'. \
                           format(typ=t, i=i, ip5=i + 5) \
                           for i in range(arity)])

    free_tabs = '\n'.join(['nsimd::device_free(tab{i});'. \
                           format(typ=t, i=i) for i in range(arity)])

    zero = '{}(0)'.format(t) if t != 'f16' else '{f32_to_f16}(0.0f)'
    one = '{}(1)'.format(t) if t != 'f16' else '{f32_to_f16}(1.0f)'
    comp_tab0_to_1 = 'tab0[i] == {}(1)'.format(t) if t != 'f16' else \
                     '{f16_to_f32}(tab0[i]) == 1.0f'
    comp_tab1_to_1 = 'tab1[i] == {}(1)'.format(t) if t != 'f16' else \
                     '{f16_to_f32}(tab1[i]) == 1.0f'

    if op_name == 'cvt':
        tet1d_code = \
            '''tet1d::out(out) = tet1d::cvt<{t}>(tet1d::cvt<{tt}>(
                                     tet1d::in(tab0, n)));'''. \
                                     format(t=t, tt=tt)
        compute_result_kernel = \
            '''dst[i] = nsimd::{{p}}_cvt({t}(), nsimd::{{p}}_cvt(
                            {tt}(), tab0[i]));'''.format(t=t, tt=tt)
    elif op_name == 'reinterpret':
        tet1d_code = \
            '''tet1d::out(out) = tet1d::reinterpret<{t}>(
                                     tet1d::reinterpret<{tt}>(tet1d::in(
                                         tab0, n)));'''.format(t=t, tt=tt)
        compute_result_kernel = \
            '''dst[i] = nsimd::{{p}}_reinterpret({t}(),
                            nsimd::{{p}}_reinterpret({tt}(),
                                tab0[i]));'''.format(t=t, tt=tt)
    elif op_name in ['to_mask', 'to_logical']:
        tet1d_code = \
            '''tet1d::out(out) = tet1d::to_mask(tet1d::to_logical(tet1d::in(
                                     tab0, n)));'''
        compute_result_kernel = \
            '''dst[i] = nsimd::{{p}}_to_mask({t}(),
                            nsimd::{{p}}_to_logical(tab0[i]));'''. \
                            format(t=t)
    elif operator.params == ['v'] * len(operator.params):
        compute_result_kernel = \
            'dst[i] = nsimd::{{p}}_{op_name}({args_tabs_i_call});'. \
            format(op_name=op_name, args_tabs_i_call=args_tabs_i_call)
        if operator.cxx_operator != None:
            if len(operator.params[1:]) == 1:
                tet1d_code = 'tet1d::out(out) = {cxx_op}tet1d::in(tab0, n);'. \
                             format(cxx_op=operator.cxx_operator)
            else:
                tet1d_code = 'tet1d::out(out) = tet1d::in(tab0, n) {cxx_op} ' \
                             'tet1d::in(tab1, n);'. \
                             format(cxx_op=operator.cxx_operator)
        else:
            tet1d_code = \
                'tet1d::out(out) = tet1d::{op_name}({args_in_tabs_call});'. \
                format(op_name=op_name, args_in_tabs_call=args_in_tabs_call)
    elif operator.params == ['l', 'v', 'v']:
        if operator.cxx_operator != None:
            cond = 'A {} B'.format(operator.cxx_operator)
        else:
            cond = 'tet1d::{}(A, B)'.format(op_name)
        tet1d_code = \
            '''TET1D_OUT({typ}) Z = tet1d::out(out);
               TET1D_IN({typ}) A = tet1d::in(tab0, n);
               TET1D_IN({typ}) B = tet1d::in(tab1, n);
               Z({cond}) = 1;'''.format(cond=cond, typ=t)
        compute_result_kernel = \
            '''if (nsimd::{{p}}_{op_name}(tab0[i], tab1[i])) {{{{
                 dst[i] = {one};
               }}}} else {{{{
                 dst[i] = {zero};
               }}}}'''.format(op_name=op_name, typ=t, one=one, zero=zero)
    elif operator.params == ['l'] * len(operator.params):
        if len(operator.params[1:]) == 1:
            if operator.cxx_operator != None:
                cond = '{}(A == 1)'.format(operator.cxx_operator)
            else:
                cond = 'tet1d::{}(A == 1)'.format(op_name)
            tet1d_code = \
                '''TET1D_OUT({typ}) Z = tet1d::out(out);
                   TET1D_IN({typ}) A = tet1d::in(tab0, n);
                   Z({cond}) = 1;'''.format(cond=cond, typ=t)
            compute_result_kernel = \
                '''if (nsimd::{{p}}_{op_name}({comp_tab0_to_1})) {{{{
                     dst[i] = {one};
                   }}}} else {{{{
                     dst[i] = {zero};
                   }}}}'''.format(op_name=op_name, typ=t, one=one, zero=zero,
                                  comp_tab0_to_1=comp_tab0_to_1)
        if len(operator.params[1:]) == 2:
            if operator.cxx_operator != None:
                cond = '(A == 1) {} (B == 1)'.format(operator.cxx_operator)
            else:
                cond = 'tet1d::{}(A == 1, B == 1)'.format(op_name)
            tet1d_code = \
                '''TET1D_OUT({typ}) Z = tet1d::out(out);
                   TET1D_IN({typ}) A = tet1d::in(tab0, n);
                   TET1D_IN({typ}) B = tet1d::in(tab1, n);
                   Z({cond}) = 1;'''.format(cond=cond, typ=t)
            compute_result_kernel = \
                '''if (nsimd::{{p}}_{op_name}({comp_tab0_to_1},
                                              {comp_tab1_to_1})) {{{{
                     dst[i] = {one};
                   }}}} else {{{{
                     dst[i] = {zero};
                   }}}}'''.format(op_name=op_name, typ=t, one=one, zero=zero,
                                  comp_tab0_to_1=comp_tab0_to_1,
                                  comp_tab1_to_1=comp_tab1_to_1)
    else:
        raise Exception('Unsupported operator: "{}"'.format(op_name))

    cpu_kernel = compute_result_kernel.format(p='scalar',
                                              f32_to_f16='nsimd_f32_to_f16',
                                              f16_to_f32='nsimd_f16_to_f32')
    gpu_kernel = compute_result_kernel.format(p='gpu',
                                              f32_to_f16='__float2half',
                                              f16_to_f32='__half2float')

    if op_name in ['rec11', 'rsqrt11']:
        comp = '!cmp(ref, out, n, .0009765625 /* = 2^-10 */)'
    elif op_name in ['rec8', 'rsqrt8']:
        comp = '!cmp(ref, out, n, .0078125 /* = 2^-7 */)'
    else:
        comp = '!cmp(ref, out, n)'

    with common.open_utf8(opts, filename) as out:
        out.write(
        '''#include <nsimd/modules/tet1d.hpp>
        #include <nsimd/modules/memory_management.hpp>
        #include "../common.hpp"

        #if defined(NSIMD_CUDA)

        __global__ void kernel({typ} *dst, {args_tabs}, int n) {{
          int i = threadIdx.x + blockIdx.x * blockDim.x;
          if (i < n) {{
            {gpu_kernel}
          }}
        }}

        void compute_result({typ} *dst, {args_tabs}, unsigned int n) {{
          kernel<<<{gpu_params}>>>(dst, {args_tabs_call}, int(n));
        }}

        #elif defined(NSIMD_ROCM)

        __global__ void kernel({typ} *dst, {args_tabs}, size_t n) {{
          size_t i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
          if (i < n) {{
            {gpu_kernel}
          }}
        }}

        void compute_result({typ} *dst, {args_tabs}, size_t n) {{
          hipLaunchKernelGGL(kernel, {gpu_params}, 0, 0, dst, {args_tabs_call},
                             n);
        }}

        #else

        void compute_result({typ} *dst, {args_tabs},
                            unsigned int n) {{
          for (unsigned int i = 0; i < n; i++) {{
            {cpu_kernel}
          }}
        }}

        #endif

        nsimd_fill_dev_mem_func(prng5,
            1 + (((unsigned int)i * 69342380 + 414585) % 5))
        nsimd_fill_dev_mem_func(prng6,
            1 + (((unsigned int)i * 12528380 + 784535) % 6))
        nsimd_fill_dev_mem_func(prng7,
            1 + (((unsigned int)i * 22328380 + 644295) % 7))

        int main() {{
          unsigned int n_[3] = {{ 10, 1001, 10001 }};
          for (int i = 0; i < (int)(sizeof(n_) / sizeof(int)); i++) {{
            unsigned int n = n_[i];
            int ret = 0;
            {fill_tabs}
            {typ} *ref = nsimd::device_calloc<{typ}>(n);
            {typ} *out = nsimd::device_calloc<{typ}>(n);
            compute_result(ref, {args_tabs_call}, n);
            {tet1d_code}
            if ({comp}) {{
              ret = -1;
            }}
            nsimd::device_free(ref);
            nsimd::device_free(out);
            {free_tabs}
            if (ret != 0) {{
              return ret;
            }}
          }}
          return 0;
        }}
        '''.format(typ=t, args_tabs=args_tabs, fill_tabs=fill_tabs,
                   args_tabs_call=args_tabs_call, gpu_params=gpu_params,
                   free_tabs=free_tabs, tet1d_code=tet1d_code, comp=comp,
                   cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel))

    common.clang_format(opts, filename, cuda=True)

def gen_tests(opts):
    for op_name, operator in operators.operators.items():
        if not operator.has_scalar_impl:
            continue

        for t in operator.types:

            tts = common.get_output_types(t, operator.output_to)

            for tt in tts:
                if t == 'f16' and op_name in ['notb', 'andnotb', 'orb',
                                              'xorb', 'andb']:
                    continue
                if operator.name in ['shl', 'shr', 'shra']:
                    gen_tests_for_shifts(opts, t, operator)
                else:
                    gen_tests_for(opts, tt, t, operator)

# -----------------------------------------------------------------------------

def gen_functions(opts):
    functions = ''

    for op_name, operator in operators.operators.items():
        if not operator.has_scalar_impl:
            continue

        not_closed = is_not_closed(operator)
        not_closed_tmpl_args = 'typename ToType, ' if not_closed else ''
        not_closed_tmpl_params = 'ToType' if not_closed else 'none_t'

        if op_name in ['shl', 'shr', 'shra']:
            tmpl_args = 'typename Left'
            tmpl_params = 'Left, none_t, none_t'
            size = 'return left.size();'
            args = 'Left const &left, int s'
            members = 'Left left; int s;'
            members_assignment = 'ret.left = to_node(left); ret.s = s;'
            to_node_type = 'typename to_node_t<Left>::type, none_t, none_t'
        elif len(operator.params) == 2:
            tmpl_args = not_closed_tmpl_args + 'typename Left'
            tmpl_params = 'Left, none_t, ' + not_closed_tmpl_params
            size = 'return left.size();'
            args = 'Left const &left'
            members = 'Left left;'
            members_assignment = 'ret.left = to_node(left);'
            to_node_type = 'typename to_node_t<Left>::type, none_t, none_t'
        elif len(operator.params) == 3:
            tmpl_args = 'typename Left, typename Right'
            tmpl_params = 'Left, Right, none_t'
            size = 'return compute_size(left.size(), right.size());'
            args = 'Left const &left, Right const &right'
            members = 'Left left;\nRight right;'
            members_assignment = '''ret.left = to_node(left);
                                    ret.right = to_node(right);'''
            to_node_type = 'typename to_node_t<Left>::type, ' \
                           'typename to_node_t<Right>::type, none_t'
        elif len(operator.params) == 4:
            tmpl_args = 'typename Left, typename Right, typename Extra'
            tmpl_params = 'Left, Right, Extra'
            size = \
            'return compute_size(left.size(), right.size(), extra.size());'
            args = 'Left const &left, Right const &right, Extra const &extra'
            members = 'Left left;\nRight right;\nExtra extra;'
            members_assignment = '''ret.left = to_node(left);
                                    ret.right = to_node(right);
                                    ret.extra = to_node(extra);'''
            to_node_type = 'typename to_node_t<Left>::type, ' \
                           'typename to_node_t<Right>::type, ' \
                           'typename to_node_t<Extra>::type'

        if operator.returns == 'v':
            to_pack = 'to_pack_t'
            return_type = 'out_type'
        else:
            to_pack = 'to_packl_t'
            return_type = 'bool'

        if not_closed:
            to_typ_arg = 'out_type(), '
            to_typ_tmpl_arg = '<typename {to_pack}<out_type, Pack>::type>'. \
                              format(to_pack=to_pack)
            in_out_typedefs = '''typedef typename Left::out_type in_type;
                                 typedef ToType out_type;'''
            to_node_type = 'typename to_node_t<Left>::type, none_t, ToType'
        else:
            to_typ_arg = '' if op_name != 'to_mask' else 'out_type(), '
            to_typ_tmpl_arg = ''
            in_out_typedefs = '''typedef typename Left::out_type in_type;
                                 typedef typename Left::out_type out_type;'''

        impl_args = 'left.{cpu_gpu}_get{tmpl}(i)'
        if (len(operator.params[1:]) >= 2):
            if operator.params[2] == 'p':
                impl_args += ', s'
            else:
                impl_args += ', right.{cpu_gpu}_get{tmpl}(i)'
        if (len(operator.params[1:]) >= 3):
            impl_args += ', extra.{cpu_gpu}_get{tmpl}(i)'

        impl_scalar = 'return nsimd::scalar_{}({}{});'. \
                      format(op_name, to_typ_arg,
                             impl_args.format(cpu_gpu='scalar', tmpl=''))

        impl_gpu = 'return nsimd::gpu_{}({}{});'. \
                   format(op_name, to_typ_arg,
                          impl_args.format(cpu_gpu='gpu', tmpl=''))

        impl_simd = 'return nsimd::{}{}({});'. \
                      format(op_name, to_typ_tmpl_arg,
                             impl_args.format(cpu_gpu='template simd',
                                              tmpl='<Pack>'))

        functions += \
        '''struct {op_name}_t {{}};

        template <{tmpl_args}>
        struct node<{op_name}_t, {tmpl_params}> {{
          {in_out_typedefs}

          {members}

          nsimd::nat size() const {{
            {size}
          }}

        #if defined(NSIMD_CUDA) || defined(NSIMD_ROCM)
          __device__ {return_type} gpu_get(nsimd::nat i) const {{
            {impl_gpu}
          }}
        #else
          {return_type} scalar_get(nsimd::nat i) const {{
            {impl_scalar}
          }}
          template <typename Pack> typename {to_pack}<out_type, Pack>::type
          simd_get(nsimd::nat i) const {{
            {impl_simd}
          }}
        #endif
        }};

        template<{tmpl_args}>
        node<{op_name}_t, {to_node_type}> {op_name}({args}) {{
          node<{op_name}_t, {to_node_type}> ret;
          {members_assignment}
          return ret;
        }}'''.format(op_name=op_name, tmpl_args=tmpl_args, size=size,
                     tmpl_params=tmpl_params, return_type=return_type,
                     args=args, to_pack=to_pack, to_node_type=to_node_type,
                     members=members, members_assignment=members_assignment,
                     in_out_typedefs=in_out_typedefs,
                     impl_gpu=impl_gpu,
                     impl_scalar=impl_scalar,
                     impl_simd=impl_simd)

        if operator.cxx_operator != None and len(operator.params) == 2:
            functions += \
            '''
            template <typename Op, typename Left, typename Right,
                      typename Extra>
            node<{op_name}_t, node<Op, Left, Right, Extra>, none_t, none_t>
            operator{cxx_operator}(node<Op, Left, Right, Extra> const &node) {{
              return tet1d::{op_name}(node);
            }}'''.format(op_name=op_name,
                         cxx_operator=operator.cxx_operator);
        if operator.cxx_operator != None and len(operator.params) == 3:
            functions += '''

            template <typename Op, typename Left, typename Right,
                      typename Extra, typename T>
            node<{op_name}_t, node<Op, Left, Right, Extra>,
                 node<scalar_t, none_t, none_t,
                      typename node<Op, Left, Right, Extra>::in_type>, none_t>
            operator{cxx_operator}(node<Op, Left, Right, Extra> const &node, T a) {{
              typedef typename tet1d::node<Op, Left, Right, Extra>::in_type S;
              return tet1d::{op_name}(node, literal_to<S>::impl(a));
            }}

            template <typename T, typename Op, typename Left, typename Right,
                      typename Extra>
            node<{op_name}_t, node<scalar_t, none_t, none_t,
                              typename node<Op, Left, Right, Extra>::in_type>,
                 node<Op, Left, Right, Extra>, none_t>
            operator{cxx_operator}(T a, node<Op, Left, Right, Extra> const &node) {{
              typedef typename tet1d::node<Op, Left, Right, Extra>::in_type S;
              return tet1d::{op_name}(literal_to<S>::impl(a), node);
            }}

            template <typename LeftOp, typename LeftLeft, typename LeftRight,
                      typename LeftExtra, typename RightOp, typename RightLeft,
                      typename RightRight, typename RightExtra>
            node<{op_name}_t, node<LeftOp, LeftLeft, LeftRight, LeftExtra>,
                              node<RightOp, RightLeft, RightRight, RightExtra>,
                 none_t>
            operator{cxx_operator}(node<LeftOp, LeftLeft, LeftRight,
                                LeftExtra> const &left,
                           node<RightOp, RightLeft, RightRight,
                                RightExtra> const &right) {{
              return tet1d::{op_name}(left, right);
            }}'''.format(op_name=op_name,
                         cxx_operator=operator.cxx_operator);

        functions += '\n\n{}\n\n'.format(common.hbar)

    # Write the code to file
    dirname = os.path.join(opts.include_dir, 'modules', 'tet1d')
    common.mkdir_p(dirname)
    filename = os.path.join(dirname, 'functions.hpp')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as out:
        out.write('#ifndef NSIMD_MODULES_TET1D_FUNCTIONS_HPP\n')
        out.write('#define NSIMD_MODULES_TET1D_FUNCTIONS_HPP\n\n')
        out.write('namespace tet1d {\n\n')
        out.write('{}\n\n'.format(common.hbar))
        out.write(functions)
        out.write('} // namespace tet1d\n\n')
        out.write('#endif\n')
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------

def name():
    return 'Tiny expression templates 1D'

def desc():
    return '''This module provide a thin layer of expression templates above
NSIMD core. It also allows the programmer to target NVIDIA and AMD GPUs.
Expression template are a C++ technique that allows the programmer to write
code "à la MATLAB" where variables usually represents vectors and operators
are itemwise.'''

def doc_menu():
    return {'Overview': 'overview', 'API reference': 'api'}

# -----------------------------------------------------------------------------

def doit(opts):
    common.myprint(opts, 'Generating module tet1d')
    if opts.library:
        gen_functions(opts)
    if opts.tests:
        gen_tests(opts)
    if opts.doc:
        gen_doc_api(opts)
        gen_doc_overview(opts)
