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

# -----------------------------------------------------------------------------

def gen_doc(opts):
    #for op_name, operator in operators.operators.items():
    #    if not operator.has_scalar_impl:
    #        continue
    #    f.write(
    #    '''template<{tmpl_args}>
    #       node<{op_name}_t, {to_node_type}> {op_name}({args})''')
    #    f.write(operator.desc)
    #    if opeartor.cxx_operator != None:
    #        f.wirte('operator+')
    pass

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

        __global__ void kernel({t} *dst, {t} *tab0, unsigned int n, int s) {{
          unsigned int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
          if (i < n) {{
            dst[i] = nsimd::gpu_{op_name}(tab0[i], s);
          }}
        }}

        void compute_result({t} *dst, {t} *tab0, unsigned int n, int s) {{
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
          for (unsigned int n = 64; n < 1000000; n *= 2) {{
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
                             format(cxx_op=operator.cxx_operator[8:])
            else:
                tet1d_code = 'tet1d::out(out) = tet1d::in(tab0, n) {cxx_op} ' \
                             'tet1d::in(tab1, n);'. \
                             format(cxx_op=operator.cxx_operator[8:])
        else:
            tet1d_code = \
                'tet1d::out(out) = tet1d::{op_name}({args_in_tabs_call});'. \
                format(op_name=op_name, args_in_tabs_call=args_in_tabs_call)
    elif operator.params == ['l', 'v', 'v']:
        if operator.cxx_operator != None:
            cond = 'A {} B'.format(operator.cxx_operator[8:])
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
                cond = '{}(A == 1)'.format(operator.cxx_operator[8:])
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
                cond = '(A == 1) {} (B == 1)'.format(operator.cxx_operator[8:])
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

        __global__ void kernel({typ} *dst, {args_tabs}, unsigned int n) {{
          unsigned int i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
          if (i < n) {{
            {gpu_kernel}
          }}
        }}

        void compute_result({typ} *dst, {args_tabs}, unsigned int n) {{
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
          for (unsigned int n = 64; n < 1000000; n *= 2) {{
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

        not_closed = (operator.output_to == common.OUTPUT_TO_SAME_SIZE_TYPES \
                      or ('v' not in operator.params[1:] and 'l' not in
                      operator.params[1:]))

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

        not_closed = (operator.output_to == common.OUTPUT_TO_SAME_SIZE_TYPES \
                      or ('v' not in operator.params[1:] and 'l' not in
                      operator.params[1:]))
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
          __device__ {return_type} gpu_get(nsimd:: nat i) const {{
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
            {cxx_operator}(node<Op, Left, Right, Extra> const &node) {{
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
            {cxx_operator}(node<Op, Left, Right, Extra> const &node, T a) {{
              typedef typename tet1d::node<Op, Left, Right, Extra>::in_type S;
              return tet1d::{op_name}(node, literal_to<S>::impl(a));
            }}

            template <typename T, typename Op, typename Left, typename Right,
                      typename Extra>
            node<{op_name}_t, node<scalar_t, none_t, none_t,
                              typename node<Op, Left, Right, Extra>::in_type>,
                 node<Op, Left, Right, Extra>, none_t>
            {cxx_operator}(T a, node<Op, Left, Right, Extra> const &node) {{
              typedef typename tet1d::node<Op, Left, Right, Extra>::in_type S;
              return tet1d::{op_name}(literal_to<S>::impl(a), node);
            }}

            template <typename LeftOp, typename LeftLeft, typename LeftRight,
                      typename LeftExtra, typename RightOp, typename RightLeft,
                      typename RightRight, typename RightExtra>
            node<{op_name}_t, node<LeftOp, LeftLeft, LeftRight, LeftExtra>,
                              node<RightOp, RightLeft, RightRight, RightExtra>,
                 none_t>
            {cxx_operator}(node<LeftOp, LeftLeft, LeftRight,
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
    print('-- Generating module tet1d')
    if opts.library:
        gen_functions(opts)
    if opts.tests:
        gen_tests(opts)
    if opts.doc:
        gen_doc(opts)
