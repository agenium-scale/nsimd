# Copyright (c) 2020 Agenium Scale
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
# oneAPI: default number of threads per block

tpb = 128
one_api_gpu_params = ''

# -----------------------------------------------------------------------------
# helpers

def append(s1, s2):
    if s1 == '':
        return s2
    if s2 == '':
        return s1
    return s1 + ', ' + s2

k_typ = {'i': 'k_int', 'u': 'k_uint', 'f': 'k_float'}

def get_signature(op):
    args = ', '.join(['a{}'.format(i - 1) for i in range(1, len(op.params))])
    if op.output_to == common.OUTPUT_TO_SAME_SIZE_TYPES or \
       op.name == 'to_mask':
        args = append('to_type', args)
    return '#define k_{}({})'.format(op.name, args)

# -----------------------------------------------------------------------------

def gen_doc_overview(opts):
    filename = common.get_markdown_file(opts, 'overview', 'spmd')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as fout:
        fout.write('''# Overview

## What is SPMD?

SPMD stands for *Single Program Multiple Data*. It is a programming paradigm.
It is used by NVIDIA CUDA. Its strengh lies in writing computation kernels.
Basically you concentrate your attention on the kernel itself and not on
how to run it. An example is worth more than a long speech, let's take vector
addition of `float`'s.

```c++
spmd_kernel_1d(add, float *dst, float *a, float *b)
  k_store(dst, k_load(a) + k_load(b));
spmd_kernel_end
```

It would be written as follows for CUDA (assuming that the vector lenghts are
multiples of block's sizes).

```c++
__global__ add(float *dst, float *a, float *b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  dst[i] = a[i] + b[i];
}
```

NSIMD's SPMD is a small DSL in standard C++98 that can be used to write
computation kernels for GPUs (NVIDIA's and AMD's) and any SIMD units supported
by NSIMD. On a more technical side, the DSL keywords are macros that:
- translates to C-ish keywords for GPUs and
- use masks for CPUs as Intel ISPC (<https://ispc.github.io/>).

The difference between NSIMD's SPMD is that a single code can be compiled
to target GPUs and CPUs whereas:
- NVIDIA CUDA only targets NVIDIA GPUs
- AMD HIP only targets NVIDIA and AMD GPUs
- INTEL ICP only targets Intel SIMD units and ARM NEON

## Writing kernels and device functions

As for CUDA kernels you can write templated and non-templated CUDA kernels.
Declaring a kernel function and launching it is straight forward:

```c++
spmd_kernel_1d(kernel_name, arguments)
  // kernel code
spmd_kernel_end

int main() {

  spmd_launch_kernel_1d(kernel_name, bit_width, param,
                        vector_size, arguments);

  return 0;
}
```

The `bit_width` argument indicates the types width in bits that will be
available inside kernels. The `param` argument indicates the unroll factor for
CPUs and the number of threads per block for GPUs. The `vector_size` argument
indicates the vectors length passed as arguments.

Device functions can also been implemented. They are functions that will
only run on the device. As for kernels, they have the same restrictions.

```c++
spmd_dev_func(k_float device_func, k_float a, k_float b)
  // Device function code
spmd_dev_func_end

spmd_kernel_1d(kernel, arguments)

  // ...

  spmd_call_dev_func(device_func, a, b);

  // ...

spmd_kernel_end
```

The caveat with `spmd_dev_func` is that its first argument must be the return
type followed by the device function name.

It is also possible to write templated kernels. Due to C++ `__VA_ARGS__`
limitations the number of template argument is limited to one of kind
`typename`. If more types or integers are to be passed to device kernels or
functions they have to be boxed inside a struct.

```c++
struct mul_t {
  spmd_dev_func(static k_float dev_impl, k_float a, k_float b)
    return a * b;
  spmd_dev_func_end
};

struct add_t {
  spmd_dev_func(static k_float dev_impl, k_float a, k_float b)
    return a + b;
  spmd_dev_func_end
};

// Op is the template argument (typename Op in C++ code)
spmd_tmpl_dev_func(k_float trampoline, Op, k_float a, k_float b)
  return Op::template spmd_call_dev_func(dev_impl, a, b);
spmd_dev_func_end

// Op is the template argument (typename Op in C++ code)
spmd_tmpl_kernel_1d(tmpl_kernel, Op, arguments)

  // ...

  spmd_call_tmpl_dev_func(trampoline, Op, a, b);

  // ...

spmd_kernel_end

int main() {

  // Kernel call for addition
  spmd_launch_tmpl_kernel_1d(tmpl_kernel, add_t, 32, 1, N, arguments);

  // Kernel call for multiplication
  spmd_launch_tmpl_kernel_1d(tmpl_kernel, mul_t, 32, 1, N, arguments);

  return 0;
}
```

## The NSIMD SPMD C++ DSL

The DSL is of course constraint by C++ syntax and constructs. This implies
some strange syntax and the impossibility to use infix operator `=`.
For now (2020/05/16) the NSIMD SPMD DSL does only supports `if`'s, while-loops
and `returns`. It seems that for-loops and do-while-loops cannot be nicely
proposed, i.e. with a nice syntax, the switch-case keywords cannot be
implemented with a good conformence to the semantic of their C++ counterparts.
Goto's also cannot be implemented properly.

### Variables types available in kernels and device functions

The following self-explanatory variable types are available inside kernels
and devices functions:

- `k_int` for signed integers
- `k_uint` for unsigned integers
- `k_float` for floatting point numbers
- `k_bool` for booleans

As explained above the bit-width of the above types are determined by the
launch kernel function. Note that `k_float` does not exists for 8-bits types.

### Load/store from/to memory

Given a pointer, the proper way to load data is to use `k_load(ptr)`. For
storing a value to memory `k_store` is to be used.

```c++
k_store(ptr, value);
k_store(ptr, expression);
```

As explained above, there is no need to compute the offset to apply to
pointers. This is hidden from the programmer.

### Assignment operator (`operator=`)

Due to C++ ADL (<https://en.cppreference.com/w/cpp/language/adl>) and the
need for keeping things simple for the compiler (which does not always mean
simple for the programmer) the use of infix operator `=` will not produce
a copmilation error but will give incorrect result. You should use `k_set`.

```c++
k_set(var, value);
k_set(var, expression);
```

As written above, `k_set` assign value or the result of an expression to a
variable.

### if, then, else

You should not use plan C++ `if`'s or `else`'s. This will not cause compilation
error but will produce incorrect results at runtime. You should use `k_if`,
`k_else`, `k_elseif` and `k_endif` instead. they have the same semantic as
their C++ counterparts.

```c++
spmd_kernel_1d(if_elseif_else, float *dst, float *a_ptr)

  k_float a, ret;
  k_set(a, k_load(a_ptr));

  k_if (a > 15.0f)

    k_set(ret, 15.0f);

  k_elseif ( a > 10.0f)

    k_set(ret, 10.0f);

  k_elseif ( a > 5.0f)

    k_set(ret, 5.0f);

  k_else

    k_set(ret, 0.0f);

  k_endif

  k_store(dst, ret);

spmd_kernel_end
```

### while loops

You should not use plan C++ `while`'s, `break`'s and `continue`'s. This will
not cause compilation error but will produce incorrect results at runtime.
You should use `k_while`, `k_break`, `k_continue` and `k_endif` instead. They
have the same semantic as their C++ counterparts.

```c++
spmd_kernel_1d(binpow, float *dst, float *a_ptr, int *p_ptr)

  k_float a, ret;
  k_set(a, k_load(a_ptr));
  k_set(ret, 1.0f);
  k_int p;
  k_set(p, k_load(p_ptr));

  k_while(p > 0)

    k_if ((p & 1) != 0)

      k_set(ret, ret * a);

    k_endif

    k_set(a, a * a);
    k_set(p, p >> 1);

  k_endwhile

  k_store(dst, ret);

spmd_kernel_end
```

### Returns

Returns cannot be implemented as macros overloading is not possible in a
standard way with an overload taking zero arguments. So returning has to be
done correctly. The `k_return` keyword has the same semantic as the C++
`return` keyword without arguments and can be used at will for kernels (as
kernels return type is always `void`) and for device functions returning
`void`.

For device functions returning a value it is recommanded to proceed this way:

1. Declare a variable, say `ret`, to store the return value.
2. Whereever you need to return, set the variable appropriately with `k_set`
   and return with `k_return`.
3. At the end of the function use `return ret;`.

```c++
spmd_dev_func(k_int func, k_int a)

  k_float ret;

  k_if (a == 0)
    k_set(ret, 0);
    k_return;
  k_endif

  k_if (a == 1)
    k_set(ret, -1);
    k_return;
  k_endif

  k_set(ret, a);

  return ret;

spmd_dev_func_end
```

## Advanced techniques and functions

This paragraph applies mainly when targeting CPUs. Using techniques described
below won't affect GPUs.

If you are familiar with the SIMD technique of masking to emulate loops and
if's you may know that `k_set` and `k_store` are implemented using respectively
`nsimd::if_else` and `nsimd::maskz_storeu` which may incur performance
penalties. When you know that a simple assignment or store is sufficient
you may use the unmasked variants:

- `k_unmasked_set` translates into a C++ assignment.
- `k_unmasked_store` translates into a C++ SIMD store.

Their arguments are exactly the same as `k_set` and `k_store`. Unmasked
operations can usually be used at the beginning of device functions and also
inside loops, on temporary variables, knowing that the result of the latter
won't be needed later.

You may also use C++ standard keywords and constructs. But be aware that doing
so will apply all the same treatment too all SIMD lanes. This can be useful
when the operations involved are independant of the processed data as in the
example below.

```c++
spmd_dev_func(k_float newton_raphson_sqrt, k_float a, k_float x0)
  k_float ret;
  for (int i = 0; i < 6; i++) {
    k_unmasked_set(ret, (ret + ret * a) / 2.0f);
  }
  return ret;
spmd_dev_func_end
```
''')

# -----------------------------------------------------------------------------

def gen_doc_api(opts):
    filename = common.get_markdown_file(opts, 'api', 'spmd')
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

    with common.open_utf8(opts, filename) as fout:
        fout.write(
'''# NSIMD SPMD API reference

This page contains the exhaustive API of the SPMD module. Note that most
operators names follow the simple naming `k_[NSIMD name]` and have the same
semantics. This page is light, you may use CTRL+F to find the operator you
are looking for.

For genericity on the base type you should use operator names instead of
infix operators, e.g. `k_add` instead of `+`. Indeed for `f16`'s NVIDIA CUDA
and NSIMD do not provide overloads and therefore code using `+` will fail to
compile.

Note that all operators accept literals and scalars. For example you may
write `k_add(a, 1)` or `float s; k_add(a, s);`. This also applies when
using infix operators. But note that literals or scalars must have the
same type as the other operands.

''')

        for c, ops in api.items():
            if len(ops) == 0:
                continue
            fout.write('\n## {}\n\n'.format(c.title))
            for op in ops:
                fout.write('- `{}`  \n'.format(get_signature(op)))
                if op.cxx_operator != None:
                    fout.write('  Infix operator: `{}` ' \
                               '(*for certain types only*)  \n'.\
                               format(op.cxx_operator))
                fout.write('  {}\n\n'.format(op.desc))

# -----------------------------------------------------------------------------

def gen_tests_for_shifts(opts, t, operator):
    op_name = operator.name
    dirname = os.path.join(opts.tests_dir, 'modules', 'spmd')
    common.mkdir_p(dirname)
    filename = os.path.join(dirname, '{}.{}.cpp'.format(op_name, t))
    if not common.can_create_filename(opts, filename):
        return

    if op_name in ['rec11', 'rsqrt11']:
        comp = '!cmp(ref, out, n, .0009765625 /* = 2^-10 */)'
    elif op_name in ['rec8', 'rsqrt8']:
        comp = '!cmp(ref, out, n, .0078125 /* = 2^-7 */)'
    else:
        comp = '!cmp(ref, out, n)'

    with common.open_utf8(opts, filename) as out:
        out.write(
        '''#include <nsimd/modules/spmd.hpp>
        #include <nsimd/modules/memory_management.hpp>
        #include <nsimd/scalar_utilities.h>
        #include "../common.hpp"

        #if defined(NSIMD_CUDA)

        __global__ void kernel({typ} *dst, {typ} *a0, int n, int s) {{
          int i = threadIdx.x + blockIdx.x * blockDim.x;
          if (i < n) {{
            dst[i] = nsimd::gpu_{op_name}(a0[i], s);
          }}
        }}

        void compute_result({typ} *dst, {typ} *a0, unsigned int n, int s) {{
          kernel<<<{gpu_params}>>>(dst, a0, int(n), s);
        }}

        #elif defined(NSIMD_ROCM)

        __global__ void kernel({typ} *dst, {typ} *a0, size_t n, int s) {{
          size_t i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
          if (i < n) {{
            dst[i] = nsimd::gpu_{op_name}(a0[i], s);
          }}
        }}

        void compute_result({typ} *dst, {typ} *a0, size_t n, int s) {{
          hipLaunchKernelGGL(kernel, {gpu_params}, 0, 0, dst, a0, n, s);
        }}

        #elif defined(NSIMD_ONEAPI)

        inline void kernel({typ} *dst, {typ} *a0, const size_t n,
                           const int s, sycl::nd_item<1> item) {{
          auto idx = item.get_global_id();
          if(idx < n){{
            dst[idx] = nsimd::scalar_{op_name}(a0[idx], s);
          }}
        }}

        void compute_result({typ} *dst, {typ} *a0, const size_t n, const int s) {{
	  sycl::queue q_ = spmd::_get_global_queue();
	  q_.submit([&](handler& h){
	      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(n),
	                                       sycl::range<1>({threads_per_block})),
	                                       [=](sycl::nd_item<1> item){
	      kernel(dst, a0, n, s, item);
	    }).wait();
	  });
        }}

        #else

        void compute_result({typ} *dst, {typ} *a0, unsigned int n, int s) {{
          for (unsigned int i = 0; i < n; i++) {{
            dst[i] = nsimd::scalar_{op_name}(a0[i], s);
          }}
        }}

        #endif

        // clang-format off

        nsimd_fill_dev_mem_func(prng7,
            1 + (((unsigned int)i * 22328380 + 644295) % 7))

        spmd_kernel_1d(kernel, {typ} *dst, {typ} *a0, int s)
          k_store(dst, k_{op_name}(k_load(a0), s));
        spmd_kernel_end

        // clang-format on

        int main() {{
          unsigned int n_[3] = {{ 10, 1001, 10001 }};
          for (int i = 0; i < (int)(sizeof(n_) / sizeof(int)); i++) {{
            unsigned int n = n_[i];
            for (int s = 0; s < {typnbits}; s++) {{
              int ret = 0;
              {typ} *a0 = nsimd::device_calloc<{typ}>(n);
              prng7(a0, n);
              {typ} *ref = nsimd::device_calloc<{typ}>(n);
              {typ} *out = nsimd::device_calloc<{typ}>(n);
              spmd_launch_kernel_1d(kernel, {typnbits}, 1, n, out, a0, s);
              compute_result(ref, a0, n, s);
              if ({comp}) {{
                ret = -1;
              }}
              nsimd::device_free(a0);
              nsimd::device_free(ref);
              nsimd::device_free(out);
              if (ret != 0) {{
                return ret;
              }}
            }}
          }}
          return 0;
        }}
        '''.format(typ=t, op_name=op_name, typnbits=t[1:], comp=comp,
                   gpu_params=gpu_params))

    common.clang_format(opts, filename, cuda=True)

# -----------------------------------------------------------------------------

def gen_tests_for_cvt_reinterpret(opts, t, tt, operator):
    op_name = operator.name
    dirname = os.path.join(opts.tests_dir, 'modules', 'spmd')
    common.mkdir_p(dirname)
    filename = os.path.join(dirname, '{}.{}_{}.cpp'.format(op_name, t, tt))
    if not common.can_create_filename(opts, filename):
        return

    with common.open_utf8(opts, filename) as out:
        out.write(
        '''#include <nsimd/modules/spmd.hpp>
        #include <nsimd/modules/memory_management.hpp>
        #include <nsimd/scalar_utilities.h>
        #include "../common.hpp"

        #if defined(NSIMD_CUDA)

        __global__ void kernel({typ} *dst, {typ} *a0, int n) {{
          int i = threadIdx.x + blockIdx.x * blockDim.x;
          if (i < n) {{
            dst[i] = nsimd::gpu_{op_name}({typ}(), nsimd::gpu_{op_name}(
                         {totyp}(), a0[i]));
          }}
        }}

        void compute_result({typ} *dst, {typ} *a0, unsigned int n) {{
          kernel<<<{gpu_params}>>>(dst, a0, int(n));
        }}

        #elif defined(NSIMD_ROCM)

        __global__ void kernel({typ} *dst, {typ} *a0, size_t n) {{
          size_t i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
          if (i < n) {{
            dst[i] = nsimd::gpu_{op_name}({typ}(), nsimd::gpu_{op_name}(
                         {totyp}(), a0[i]));
          }}
        }}

        void compute_result({typ} *dst, {typ} *a0, size_t n) {{
          hipLaunchKernelGGL(kernel, {gpu_params}, 0, 0, dst, a0, n);
        }}

        #elif defined(NSIMD_ONEAPI)

        inline void kernel({typ} *dst, {typ} *a0, const size_t n, sycl::id<1> id) {{
          const size_t ii = id.get(0);
          if(ii < n){{
            dst[ii] = nsimd::scalar_{op_name}({typ}(), nsimd::scalar_{op_name}(
                                     {totyp}(), a0[ii]));
          }}
        }}

        void compute_result({typ} *dst, {typ} *a0, const size_t n) {{
          sycl::queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> id){{
            kernel(dst, a0, n, id);
          }};).wait();
        }}

        #else

        void compute_result({typ} *dst, {typ} *a0, unsigned int n) {{
          for (unsigned int i = 0; i < n; i++) {{
            dst[i] = nsimd::scalar_{op_name}({typ}(), nsimd::scalar_{op_name}(
                         {totyp}(), a0[i]));
          }}
        }}

        #endif

        // clang-format off

        nsimd_fill_dev_mem_func(prng7,
            1 + (((unsigned int)i * 22328380 + 644295) % 7))

        spmd_kernel_1d(kernel, {typ} *dst, {typ} *a0)
          k_store(dst, k_{op_name}({k_typ}, k_{op_name}({k_totyp},
                  k_load(a0))));
        spmd_kernel_end

        // clang-format on

        int main() {{
          unsigned int n_[3] = {{ 10, 1001, 10001 }};
          for (int i = 0; i < (int)(sizeof(n_) / sizeof(int)); i++) {{
            unsigned int n = n_[i];
            int ret = 0;
            {typ} *a0 = nsimd::device_calloc<{typ}>(n);
            prng7(a0, n);
            {typ} *ref = nsimd::device_calloc<{typ}>(n);
            {typ} *out = nsimd::device_calloc<{typ}>(n);
            spmd_launch_kernel_1d(kernel, {typnbits}, 1, n, out, a0);
            compute_result(ref, a0, n);
            if (!cmp(ref, out, n)) {{
              ret = -1;
            }}
            nsimd::device_free(a0);
            nsimd::device_free(ref);
            nsimd::device_free(out);
            if (ret != 0) {{
              return ret;
            }}
          }}
          return 0;
        }}
        '''.format(typ=t, totyp=tt, op_name=op_name, typnbits=t[1:],
                   gpu_params=gpu_params, k_typ=k_typ[t[0]],
                   k_totyp=k_typ[tt[0]]))

    common.clang_format(opts, filename, cuda=True)

# -----------------------------------------------------------------------------

def gen_tests_for(opts, t, operator):
    op_name = operator.name
    dirname = os.path.join(opts.tests_dir, 'modules', 'spmd')
    common.mkdir_p(dirname)
    filename = os.path.join(dirname, '{}.{}.cpp'.format(op_name, t))
    if not common.can_create_filename(opts, filename):
        return

    arity = len(operator.params[1:])
    k_args = ', '.join(['{} *a{}'.format(t, i) for i in range(arity)])
    k_call_args = ', '.join(['a{}'.format(i) for i in range(arity)])

    fill_tabs = '\n'.join(['{typ} *a{i} = nsimd::device_calloc<{typ}>(n);\n' \
                           'prng{ip5}(a{i}, n);'. \
                           format(typ=t, i=i, ip5=i + 5) \
                           for i in range(arity)])

    free_tabs = '\n'.join(['nsimd::device_free(a{i});'. \
                           format(typ=t, i=i) for i in range(arity)])

    # spmd
    def get_cte_spmd(typ, cte):
        if typ == 'f16':
            return 'k_f32_to_f16((f32){})'.format(cte)
        else:
            return '({}){}'.format(typ, cte)

    def spmd_load_code(param, typ, i):
        if param == 'l':
            return 'k_lt(k_load(a{}), {})'.format(i, get_cte_spmd(typ, 4))
        if param == 'v':
            return 'k_load(a{})'.format(i)

    args = ', '.join([spmd_load_code(operator.params[i + 1], t, i) \
                      for i in range(arity)])
    if op_name == 'to_mask':
        args = k_typ[t[0]] + ', ' + args
    if operator.params[0] == 'v':
        k_code = 'k_store(dst, k_{}({}));'.format(op_name, args)
    else:
        k_code = '''k_if (k_{}({}))
                      k_store(dst, 1);
                    k_else
                      k_store(dst, 0);
                    k_endif'''.format(op_name, args)

    # gpu
    def get_cte_gpu(typ, cte):
        if typ == 'f16':
            return '__float2half((f32){})'.format(cte)
        else:
            return '({}){}'.format(typ, cte)

    def gpu_load_code(param, typ, i):
        if param == 'l':
            return 'nsimd::gpu_lt(a{}[i], {})'.format(i, get_cte_gpu(typ, 4))
        if param == 'v':
            return 'a{}[i]'.format(i)

    args = ', '.join([gpu_load_code(operator.params[i + 1], t, i) \
                      for i in range(arity)])
    if op_name == 'to_mask':
        args = t + '(), ' + args
    if operator.params[0] == 'v':
        gpu_kernel = 'dst[i] = nsimd::gpu_{}({});'.format(op_name, args)
    else:
        gpu_kernel = '''if (nsimd::gpu_{op_name}({args})) {{
                          dst[i] = {one};
                        }} else {{
                          dst[i] = {zero};
                        }}'''.format(op_name=op_name, args=args,
                                     one=get_cte_gpu(t, 1),
                                     zero=get_cte_gpu(t, 0))

    # cpu
    def get_cte_cpu(typ, cte):
        if typ == 'f16':
            return 'nsimd_f32_to_f16((f32){})'.format(cte)
        else:
            return '({}){}'.format(typ, cte)

    def gpu_load_code(param, typ, i):
        if param == 'l':
            return 'nsimd::scalar_lt(a{}[i], {})'. \
                   format(i, get_cte_cpu(typ, 4))
        if param == 'v':
            return 'a{}[i]'.format(i)

    args = ', '.join([gpu_load_code(operator.params[i + 1], t, i) \
                      for i in range(arity)])
    if op_name == 'to_mask':
        args = t + '(), ' + args
    if operator.params[0] == 'v':
        cpu_kernel = 'dst[i] = nsimd::scalar_{}({});'.format(op_name, args)
    else:
        cpu_kernel = '''if (nsimd::scalar_{op_name}({args})) {{
                          dst[i] = {one};
                        }} else {{
                          dst[i] = {zero};
                        }}'''.format(op_name=op_name, args=args,
                                     one=get_cte_cpu(t, 1),
                                     zero=get_cte_cpu(t, 0))

    if op_name in ['rec11', 'rsqrt11']:
        comp = '!cmp(ref, out, n, .0009765625 /* = 2^-10 */)'
    elif op_name in ['rec8', 'rsqrt8']:
        comp = '!cmp(ref, out, n, .0078125 /* = 2^-7 */)'
    else:
        comp = '!cmp(ref, out, n)'

    with common.open_utf8(opts, filename) as out:
        out.write(
        '''#include <nsimd/modules/spmd.hpp>
        #include <nsimd/modules/memory_management.hpp>
        #include <nsimd/scalar_utilities.h>
        #include "../common.hpp"

        #if defined(NSIMD_CUDA)

        __global__ void kernel({typ} *dst, {k_args}, int n) {{
          int i = threadIdx.x + blockIdx.x * blockDim.x;
          if (i < n) {{
            {gpu_kernel}
          }}
        }}

        void compute_result({typ} *dst, {k_args}, unsigned int n) {{
          kernel<<<{gpu_params}>>>(dst, {k_call_args}, int(n));
        }}

        #elif defined(NSIMD_ROCM)

        __global__ void kernel({typ} *dst, {k_args}, size_t n) {{
          size_t i = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
          if (i < n) {{
            {gpu_kernel}
          }}
        }}

        void compute_result({typ} *dst, {k_args}, size_t n) {{
          hipLaunchKernelGGL(kernel, {gpu_params}, 0, 0, dst, {k_call_args},
                             n);
        }}

        #elif defined(NSIMD_ONEAPI)

        inline void kernel({typ} *dst, {k_args}, const size_t n, sycl::id<1> id) {{
          if(id.get(0) < n){{
            {cpu_kernel}
          }}
        }}

        void compute_result({typ} *dst, {k_args}, const size_t n) {{
          sycl::queue().parallel_for(sycl::range<1>(n), [=](sycl::id<1> id){{
            kernel(dst, {k_call_args}, n, id);
          }};).wait();
        }}

        #else

        void compute_result({typ} *dst, {k_args}, unsigned int n) {{
          for (unsigned int i = 0; i < n; i++) {{
            {cpu_kernel}
          }}
        }}

        #endif

        // clang-format off

        nsimd_fill_dev_mem_func(prng5,
            1 + (((unsigned int)i * 69342380 + 414585) % 5))
        nsimd_fill_dev_mem_func(prng6,
            1 + (((unsigned int)i * 12528380 + 784535) % 6))
        nsimd_fill_dev_mem_func(prng7,
            1 + (((unsigned int)i * 22328380 + 644295) % 7))

        spmd_kernel_1d(kernel, {typ} *dst, {k_args})
          {k_code}
        spmd_kernel_end

        // clang-format on

        #if defined(NSIMD_CUDA) || defined(NSIMD_ROCM)
        #define THREADS_PER_BLOCK 128
        #else
        #define THREADS_PER_BLOCK 1
        #endif

        int main() {{
          unsigned int n_[3] = {{ 10, 1001, 10001 }};
          for (int i = 0; i < (int)(sizeof(n_) / sizeof(int)); i++) {{
            unsigned int n = n_[i];
            int ret = 0;
            {fill_tabs}
            {typ} *ref = nsimd::device_calloc<{typ}>(n);
            {typ} *out = nsimd::device_calloc<{typ}>(n);
            spmd_launch_kernel_1d(kernel, {typnbits}, THREADS_PER_BLOCK, n,
                                  out, {k_call_args});
            compute_result(ref, {k_call_args}, n);
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
        '''.format(typ=t, free_tabs=free_tabs, fill_tabs=fill_tabs,
                   k_code=k_code, k_call_args=k_call_args, k_args=k_args,
                   cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel, comp=comp,
                   gpu_params=gpu_params, typnbits=t[1:]))

    common.clang_format(opts, filename, cuda=True)

def gen_tests(opts):
    for op_name, operator in operators.operators.items():
        if not operator.has_scalar_impl:
            continue

        not_closed = (operator.output_to == common.OUTPUT_TO_SAME_SIZE_TYPES \
                      or ('v' not in operator.params[1:] and 'l' not in
                      operator.params[1:]))

        for t in operator.types:

            if operator.name in ['notb', 'andb', 'xorb', 'orb',
                                 'andnotb'] and t == 'f16':
                continue

            tts = common.get_output_types(t, operator.output_to)

            for tt in tts:
                if t == 'f16' and op_name in ['notb', 'andnotb', 'orb',
                                              'xorb', 'andb']:
                    continue
                if operator.name in ['shl', 'shr', 'shra']:
                    gen_tests_for_shifts(opts, t, operator)
                elif operator.name in ['cvt', 'reinterpret', 'reinterpretl']:
                    gen_tests_for_cvt_reinterpret(opts, tt, t, operator)
                else:
                    gen_tests_for(opts, t, operator)

# -----------------------------------------------------------------------------

def gen_functions(opts):
    functions = ''

    for op_name, operator in operators.operators.items():
        if not operator.has_scalar_impl:
            continue

        if operator.params[0] == 'l':
            s_ret_typ = 'bool'
            v_ret_typ = \
                'nsimd::packl<typename base_type<A0>::type, N>'
        else:
            s_ret_typ = 'T'
            v_ret_typ = 'nsimd::pack<typename base_type<A0>::type, N>'

        def s_typ(typ):
            if typ == 'p':
                return 'int'
            if typ == 'v':
                return 'T'
            if typ == 'l':
                return 'bool'

        s_args = ', '.join(['{} a{}'.format(s_typ(operator.params[i]), i - 1) \
                            for i in range(1, len(operator.params))])
        s_call_args = ', '.join(['a{}'.format(i - 1) \
                                 for i in range(1, len(operator.params))])
        s_tmpl = 'typename T' if 'v' in operator.params[1:] else ''

        def v_typ(typ, i):
            if typ == 'p':
                return 'int'
            if typ in ['v', 'l']:
                return 'A{}'.format(i)
        v_args = ', '.join(['{} a{}'. \
                            format(v_typ(operator.params[i], i - 1), i - 1) \
                            for i in range(1, len(operator.params))])

        def v_call_arg(typ, i):
            if typ == 'p':
                return '(int)a{}'.format(i)
            if typ == 'v':
                return 'spmd::to_pack<T, N>(a{})'.format(i)
            if typ == 'l':
                return 'spmd::to_packl<T, N>(a{})'.format(i)

        v_call_args = ', '.join([v_call_arg(operator.params[i], i - 1) \
                                 for i in range(1, len(operator.params))])

        v_tmpl = ', '.join(['typename A{}'.format(i - 1) \
                            for i in range(1, len(operator.params)) \
                            if operator.params[i] != 'p'])

        m_call_args_cpu = s_call_args
        m_call_args_gpu = s_call_args
        m_call_args_sycl = s_call_args
        to_type = ''
        ToType = ''
        v_op_name = op_name
        s_op_name = op_name
        template = ''

        # Override for non closed operators
        if operator.output_to == common.OUTPUT_TO_SAME_SIZE_TYPES or \
           op_name == 'to_mask':
            s_ret_typ = 'ToType'
            s_tmpl = append('typename ToType', s_tmpl)
            m_call_args_gpu = append('to_type()', s_call_args)
            m_call_args_sycl = append('to_type()', s_call_args)
            s_call_args = append('ToType()', s_call_args)
            v_tmpl = append('typename ToType', v_tmpl)
            to_type = '<to_type>'
            template = 'template '
            v_ret_typ = 'ToType'
            ToType = '<ToType>'

        # special case for to_mask
        if op_name == 'to_mask':
            v_op_name = 'reinterpret'
            v_call_args = 'to_mask({})'.format(v_call_args)

        if v_tmpl != '':
            v_tmpl = 'template <{}>'.format(v_tmpl)
        if s_tmpl != '':
            s_tmpl = 'template <{}>'.format(s_tmpl)

        functions += \
        '''#if defined(NSIMD_CUDA) || defined(NSIMD_ROCM)

           {signature} nsimd::gpu_{s_op_name}({m_call_args_gpu})

           #elif defined(NSIMD_ONEAPI)

           {signature} nsimd::scalar_{s_op_name}({m_call_args_sycl})

           #else

           template <typename KernelType, int N> struct {op_name}_helper {{}};

           template <int N> struct {op_name}_helper<spmd::KernelScalar, N> {{
             {s_tmpl} static {s_ret_typ} impl({s_args}) {{
               return nsimd::scalar_{s_op_name}({s_call_args});
             }}
           }};

           template <int N> struct {op_name}_helper<spmd::KernelSIMD, N> {{
             {v_tmpl} static {v_ret_typ} impl({v_args}) {{
               typedef typename spmd::base_type<A0>::type T;
               return nsimd::{v_op_name}{ToType}({v_call_args});
             }}
           }};

           {signature} \\
               spmd::{op_name}_helper<spmd_KernelType_, \\
                                      spmd_N_>::{template}impl{to_type}( \\
                                        {m_call_args_cpu})

           #endif

           {hbar}

           '''.format(hbar=common.hbar, s_op_name=s_op_name, s_tmpl=s_tmpl,
                      s_ret_typ=s_ret_typ, s_args=s_args, v_args=v_args,
                      v_call_args=v_call_args, s_call_args=s_call_args,
                      v_tmpl=v_tmpl, v_ret_typ=v_ret_typ, ToType=ToType,
                      m_call_args_cpu=m_call_args_cpu, to_type=to_type,
                      v_op_name=v_op_name, op_name=op_name, template=template,
                      m_call_args_gpu=m_call_args_gpu,
                      m_call_args_sycl=m_call_args_sycl,
                      signature=get_signature(operator))

    # Write the code to file
    dirname = os.path.join(opts.include_dir, 'modules', 'spmd')
    common.mkdir_p(dirname)
    filename = os.path.join(dirname, 'functions.hpp')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as out:
        out.write('#ifndef NSIMD_MODULES_SPMD_FUNCTIONS_HPP\n')
        out.write('#define NSIMD_MODULES_SPMD_FUNCTIONS_HPP\n\n')
        out.write('namespace spmd {\n\n')
        out.write('{}\n\n'.format(common.hbar))
        out.write(functions)
        out.write('} // namespace spmd\n\n')
        out.write('#endif\n')
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------

def name():
    return 'SPMD programming'

def desc():
    return '''SPMD programming allows the programmer to focus on kernels and
the compiler to vectorize kernel code more effectively. Basically this
module provides a "à la CUDA" programming C++ DSL to targets CPU SIMD as well
as NVIDIA and AMD GPUs.'''

def doc_menu():
    return {'Overview': 'overview', 'API reference': 'api'}

# -----------------------------------------------------------------------------

def doit(opts):
    common.myprint(opts, 'Generating module spmd')
    if opts.library:
        gen_functions(opts)
    if opts.tests:
        gen_tests(opts)
    if opts.doc:
        gen_doc_api(opts)
        gen_doc_overview(opts)
