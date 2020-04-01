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

import common
import operators
import os
from datetime import date
import sys

# -----------------------------------------------------------------------------
# Implementations for output

def get_put_impl(simd_ext):
    args = {
      'i8' :      ['"%d"', '(int)buf[i]'],
      'u8' :      ['"%d"', '(int)buf[i]'],
      'i16':      ['"%d"', '(int)buf[i]'],
      'u16':      ['"%d"', '(int)buf[i]'],
      'i32':      ['"%d"', 'buf[i]'],
      'u32':      ['"%u"', 'buf[i]'],
      'i64':      ['"%ld"', 'buf[i]'],
      'u64':      ['"%lu"', 'buf[i]'],
      'i64_msvc': ['"%lld"', 'buf[i]'],
      'u64_msvc': ['"%llu"', 'buf[i]'],
      'f16':      ['"%e"', '(double)nsimd_f16_to_f32(buf[i])'],
      'f32':      ['"%e"', '(double)buf[i]'],
      'f64':      ['"%e"', 'buf[i]'],
    }
    ret = '''extern "C" {

             #include <cstdio>

             '''
    for typ in common.types:

        if typ in ['i64', 'u64']:
            fprintf = '''#if defined(NSIMD_IS_MSVC ) || (NSIMD_WORD_SIZE == 32)
                           code = fprintf(out, {fmt_msvc}, {val});
                         #else
                           code = fprintf(out, {fmt}, {val});
                         #endif'''.format(fmt_msvc=args[typ + '_msvc'][0],
                                          fmt=args[typ][0], val=args[typ][1])
        else:
            fprintf = 'code = fprintf(out, {fmt}, {val});'. \
                      format(fmt=args[typ][0], val=args[typ][1])

        fmt = '''NSIMD_DLLEXPORT
           int nsimd_put_{simd_ext}_{l}{typ}(FILE *out, const char *fmt,
                                          nsimd_{simd_ext}_v{l}{typ} v) {{
             using namespace nsimd;
             {typ} buf[max_len<{typ}>];

             int n = len({typ}(), {simd_ext}());
             store{l}u(buf, v, {typ}(), {simd_ext}());
             if (fputs("{{ ", out) == EOF) {{
               return -1;
             }}
             int ret = 2;
             for (int i = 0; i < n; i++) {{
               int code;
               if (fmt != NULL) {{
                 code = fprintf(out, fmt, {val});
               }} else {{
                 {fprintf}
               }}
               if (code < 0) {{
                 return -1;
               }}
               ret += code;
               if (i < n - 1) {{
                 if (fputs(", ", out) == EOF) {{
                   return -1;
                 }}
                 ret += 2;
               }}
             }}
             if (fputs(" }}", out) == EOF) {{
               return -1;
             }}
             return ret + 2;
           }}
           {hbar}
           '''

        ret += fmt\
        .format(typ=typ, l='', simd_ext=simd_ext, hbar=common.hbar,
                      fprintf=fprintf, val=args[typ][1])

        ret += fmt\
        .format(typ=typ, l='l', simd_ext=simd_ext, hbar=common.hbar,
                      fprintf=fprintf, val=args[typ][1])
    ret += \
    '''} // extern "C"
       '''
    return ret

# -----------------------------------------------------------------------------
# Implementations for all other functions

def get_impl(operator, emulate_fp16, simd_ext):
    ret = ''
    for t in operator.types:
        if not operator.closed:
            # For now we do not support generation of non closed operators
            # for the binary
            raise Exception('Non closed operators not supported')
        fmtspec = operator.get_fmtspec(t, t, simd_ext)
        args_list = common.enum(operator.params[1:])
        args = []
        args1 = []
        args2 = []
        for a in args_list:
            if a[1] == 'v':
                if emulate_fp16 and t == 'f16':
                    # cpu is the only exception
                    if simd_ext == 'cpu':
                        n = common.CPU_NBITS // 16 // 2
                        args1 += ['nsimd::pack<f32>(nsimd_cpu_vf32{' + \
                                  ','.join('a{}.v{}'.format(a[0], i) \
                                  for i in range(0, n)) + '})']
                        args2 += ['nsimd::pack<f32>(nsimd_cpu_vf32{' + \
                                  ','.join('a{}.v{}'.format(a[0], i + n) \
                                  for i in range(0, n)) + '})']
                    else:
                        args += ['nsimd::pack<f32>(a{}.v{{lohi}})'. \
                                 format(a[0])]
                else:
                    args += ['nsimd::pack<{}>(a{})'.format(t, a[0])]
            elif a[1] == 'l':
                if emulate_fp16 and t == 'f16':
                    if simd_ext == 'cpu':
                        n = common.CPU_NBITS // 16 // 2
                        args1 += ['nsimd::packl<f32>(nsimd_cpu_vlf32{' + \
                                  ','.join('a{}.v{}'.format(a[0], i) \
                                  for i in range(0, n)) + '})']
                        args2 += ['nsimd::packl<f32>(nsimd_cpu_vlf32{' + \
                                  ','.join('a{}.v{}'.format(a[0], i + n) \
                                  for i in range(0, n)) + '})']
                    else:
                        args += ['nsimd::packl<f32>(a{}.v{{lohi}})'. \
                                 format(a[0])]
                else:
                    args += ['nsimd::packl<{}>(a{})'.format(t, a[0])]
            else:
                args += ['a{}'.format(a[0])]
        args = ', '.join(args)
        args1 = ', '.join(args1)
        args2 = ', '.join(args2)
        if emulate_fp16 and t == 'f16':
            if simd_ext == 'cpu':
                n = common.CPU_NBITS // 16
                lo = '\n'.join(['ret.v{} = tmp.car.v{};'.format(i, i) \
                                for i in range(0, n // 2)])
                hi = '\n'.join(['ret.v{} = tmp.car.v{};'. \
                                format(i + n // 2, i) \
                                for i in range(0, n // 2)])
                ret += \
                '''{hbar}

                   extern "C" {{

                   NSIMD_DLLEXPORT
                   {return_typ} nsimd_{name}_cpu_{suf}({c_args}) {{
                     nsimd_cpu_v{logical}f16 ret;
                     nsimd::pack{logical}<f32> tmp;
                     tmp = nsimd::impl::{name}({args1});
                     {lo}
                     tmp = nsimd::impl::{name}({args2});
                     {hi}
                     return ret;
                   }}

                   }} // extern "C"

                   '''.format(args1=args1, args2=args2, lo=lo, hi=hi,
                              logical='l' if operator.params[0] == 'l' else '',
                              member='.f' if operator.params[0] == 'v' \
                              else '.u', **fmtspec)
            else:
                ret += \
                '''{hbar}

                   extern "C" {{

                   NSIMD_DLLEXPORT
                   {return_typ} nsimd_{name}_{simd_ext}_{suf}({c_args}) {{
                     nsimd_{simd_ext}_v{logical}f16 ret;
                     auto buf = nsimd::impl::{name}({args1});
                     ret.v0 = buf.car;
                     buf = nsimd::impl::{name}({args2});
                     ret.v1 = buf.car;
                     return ret;
                   }}

                   }} // extern "C"

                   '''.format(args1=args.format(lohi='0'),
                              args2=args.format(lohi='1'),
                              logical='l' if operator.params[0] == 'l' else '',
                              **fmtspec)
        else:
            if t == 'f16':
                inputs = \
                '\n'.join(['''f16 buf{i}_f16[max_len<f16>];
                              f32 buf{i}_f32[max_len<f16>];
                              storeu(buf{i}_f16, a{i}, f16(), {simd_ext}());
                              for (int i = 0; i < len_f16; i++) {{
                                buf{i}_f32[i] = (f32)buf{i}_f16[i];
                              }}
                              '''.format(i=i, **fmtspec) for i in \
                                         range(0, len(args_list))])
                f32_args_lo = \
                ', '.join(['loadu<pack<f32>>(buf{}_f32)'. \
                           format(i) for i in range(0, len(args_list))])
                f32_args_hi = \
                ', '.join(['loadu<pack<f32>>(buf{}_f32 + len_f32)'. \
                           format(i) for i in range(0, len(args_list))])
                ret += \
                '''{hbar}

                   extern "C" {{

                   NSIMD_DLLEXPORT
                   {return_typ} nsimd_{name}_{simd_ext}_{suf}({c_args}) {{
                     using namespace nsimd;
                     int len_f16 = len(pack<f16>());
                     int len_f32 = len(pack<f32>());
                     {inputs}
                     auto temp = nsimd::impl::{name}({f32_args_lo});
                     storeu(buf0_f32, temp.car, f32(), {simd_ext}());
                     temp = nsimd::impl::{name}({f32_args_hi});
                     storeu(buf0_f32 + len_f32, temp.car, f32(), {simd_ext}());
                     for (int i = 0; i < len_f16; i++) {{
                       buf0_f16[i] = (f16)buf0_f32[i];
                     }}
                     return loadu(buf0_f16, f16());
                   }}

                   }} // extern "C"

                   '''.format(inputs=inputs, f32_args_lo=f32_args_lo,
                              f32_args_hi=f32_args_hi, **fmtspec)
            else:
                ret += \
                '''{hbar}

                   extern "C" {{

                   NSIMD_DLLEXPORT
                   {return_typ} nsimd_{name}_{simd_ext}_{suf}({c_args}) {{
                     auto buf = nsimd::impl::{name}({args});
                     return buf.car;
                   }}

                   }} // extern "C"

                   '''.format(args=args, **fmtspec)
    return ret


# -----------------------------------------------------------------------------
# Generate base APIs

def write_cpp(opts, simd_ext, emulate_fp16):
    filename = os.path.join(opts.src_dir, 'api_{}.cpp'.format(simd_ext))
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as out:
        out.write('''#define NSIMD_INSIDE
                     #include <nsimd/nsimd.h>
                     #include <nsimd/cxx_adv_api.hpp>

                     '''.format(year=date.today().year))
        for op_name, operator in operators.operators.items():
            if operator.src:
                out.write('''{hbar}

                             #include <nsimd/src/{name}.hpp>

                             '''.format(name=operator.name, hbar=common.hbar))
                out.write(get_impl(operator, emulate_fp16, simd_ext))
        out.write(get_put_impl(simd_ext))

    common.clang_format(opts, filename)

def doit(opts):
    common.mkdir_p(opts.src_dir)
    common.myprint(opts, 'Generating source for binary')
    opts.platforms = common.get_platforms(opts)
    for platform in opts.platforms:
        mod = opts.platforms[platform]
        for simd_ext in mod.get_simd_exts():
            write_cpp(opts, simd_ext,
                      mod.emulate_fp16(simd_ext))
