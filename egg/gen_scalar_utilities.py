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
import common
import operators
import scalar
import cuda
import rocm

# -----------------------------------------------------------------------------

def get_gpu_impl(gpu_sig, cuda_impl, rocm_impl):
    if cuda_impl == rocm_impl:
        return '''#if (defined(NSIMD_CUDA) && defined(NSIMD_IS_NVCC)) || \\
                      defined(NSIMD_ROCM)

                  inline {gpu_sig} {{
                    {cuda_impl}
                  }}

                  #endif'''.format(gpu_sig=gpu_sig, cuda_impl=cuda_impl)
    else:
        return '''#if defined(NSIMD_CUDA) && defined(NSIMD_IS_NVCC)

                  inline {gpu_sig} {{
                    {cuda_impl}
                  }}

                  #endif

                  #ifdef NSIMD_ROCM

                  inline {gpu_sig} {{
                    {rocm_impl}
                  }}

                  #endif'''.format(gpu_sig=gpu_sig, cuda_impl=cuda_impl,
                                   rocm_impl=rocm_impl)

# -----------------------------------------------------------------------------

def doit(opts):
    common.myprint(opts, 'Generating scalar implementation for CPU and GPU')
    filename = os.path.join(opts.include_dir, 'scalar_utilities.h')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as out:
        # we declare reinterprets now as we need them
        scalar_tmp = []
        gpu_tmp = []
        for t in operators.Reinterpret.types:
            for tt in common.get_output_types(
                          t, operators.Reinterpret.output_to):
                scalar_tmp += [operators.Reinterpret(). \
                               get_scalar_signature('cpu', t, tt, 'c')]
                gpu_tmp += [operators.Reinterpret(). \
                            get_scalar_signature('gpu', t, tt, 'cxx')]
        scalar_reinterpret_decls = '\n'.join(['NSIMD_INLINE ' + sig + ';' \
                                              for sig in scalar_tmp])
        gpu_reinterpret_decls = '\n'.join(['inline ' + sig + ';' \
                                           for sig in gpu_tmp])
        out.write(
        '''#ifndef NSIMD_SCALAR_UTILITIES_H
           #define NSIMD_SCALAR_UTILITIES_H

           #if NSIMD_CXX > 0
           #include <cmath>
           #include <cstring>
           #else
           #include <math.h>
           #include <string.h>
           #endif

           #ifdef NSIMD_NATIVE_FP16
             #if defined(NSIMD_IS_GCC)
               #pragma GCC diagnostic push
               #pragma GCC diagnostic ignored "-Wdouble-promotion"
             #elif defined(NSIMD_IS_CLANG)
               #pragma clang diagnostic push
               #pragma clang diagnostic ignored "-Wdouble-promotion"
             #endif
           #endif

           {scalar_reinterpret_decls}

           #if (defined(NSIMD_CUDA) && defined(NSIMD_IS_NVCC)) || \\
               defined(NSIMD_ROCM)

           namespace nsimd {{

           {gpu_reinterpret_decls}

           }} // namespace nsimd

           #endif
           '''. \
           format(scalar_reinterpret_decls=scalar_reinterpret_decls,
                  gpu_reinterpret_decls=gpu_reinterpret_decls))
        for op_name, operator in operators.operators.items():
            if not operator.has_scalar_impl:
                continue
            if operator.params == ['l'] * len(operator.params):
                out.write('\n\n' + common.hbar + '\n\n')
                out.write(\
                '''NSIMD_INLINE {c_sig} {{
                  {scalar_impl}
                }}

                #if NSIMD_CXX > 0

                namespace nsimd {{

                NSIMD_INLINE {cxx_sig} {{
                  return nsimd_scalar_{op_name}({c_args});
                }}

                {gpu_impl}

                }} // namespace nsimd

                #endif'''.format(
                c_sig=operator.get_scalar_signature('cpu', '', '', 'c'),
                cxx_sig=operator.get_scalar_signature('cpu', '', '', 'cxx'),
                op_name=op_name,
                c_args=', '.join(['a{}'.format(i - 1) \
                               for i in range(1, len(operator.params))]),
                scalar_impl=scalar.get_impl(operator, tt, t),
                gpu_impl=get_gpu_impl(
                    operator.get_scalar_signature('gpu', t, tt, 'cxx'),
                    cuda.get_impl(operator, tt, t),
                    rocm_impl=rocm.get_impl(operator, tt, t))))
                continue
            for t in operator.types:
                tts = common.get_output_types(t, operator.output_to)
                for tt in tts:
                    out.write('\n\n' + common.hbar + '\n\n')
                    out.write(\
                    '''NSIMD_INLINE {c_sig} {{
                      {scalar_impl}
                    }}

                    #if NSIMD_CXX > 0

                    namespace nsimd {{

                    NSIMD_INLINE {cxx_sig} {{
                      return nsimd_scalar_{op_name}_{suffix}({c_args});
                    }}

                    {gpu_impl}

                    }} // namespace nsimd

                    #endif'''.format(
                    c_sig=operator.get_scalar_signature('cpu', t, tt, 'c'),
                    cxx_sig=operator.get_scalar_signature('cpu', t, tt, 'cxx'),
                    op_name=op_name,
                    suffix=t if operator.closed else '{}_{}'.format(tt, t),
                    c_args=', '.join(['a{}'.format(i - 1) \
                                   for i in range(1, len(operator.params))]),
                    scalar_impl=scalar.get_impl(operator, tt, t),
                    gpu_impl=get_gpu_impl(
                        operator.get_scalar_signature('gpu', t, tt, 'cxx'),
                        cuda.get_impl(operator, tt, t),
                        rocm_impl=rocm.get_impl(operator, tt, t))))

        out.write('''

                  {hbar}

                  #ifdef NSIMD_NATIVE_FP16
                    #if defined(NSIMD_IS_GCC)
                      #pragma GCC diagnostic pop
                    #elif defined(NSIMD_IS_CLANG)
                      #pragma clang diagnostic pop
                    #endif
                  #endif

                  #endif'''.format(hbar=common.hbar))
    common.clang_format(opts, filename)

