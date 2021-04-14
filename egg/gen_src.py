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

import common
import operators
import os
from datetime import date
import sys

# -----------------------------------------------------------------------------
# Implementations for output

def get_put_impl(simd_ext):
    args = {
      'i8' : ['"%d"', '(int)buf[i]'],
      'u8' : ['"%d"', '(int)buf[i]'],
      'i16': ['"%d"', '(int)buf[i]'],
      'u16': ['"%d"', '(int)buf[i]'],
      'i32': ['"%d"', 'buf[i]'],
      'u32': ['"%u"', 'buf[i]'],
      'i64': ['"%lld"', '(nsimd_longlong)buf[i]'],
      'u64': ['"%llu"', '(nsimd_ulonglong)buf[i]'],
      'f16': ['"%e"', '(double)nsimd_f16_to_f32(buf[i])'],
      'f32': ['"%e"', '(double)buf[i]'],
      'f64': ['"%e"', 'buf[i]'],
    }
    ret = '''#ifdef NSIMD_LONGLONG_IS_EXTENSION
               #if defined(NSIMD_IS_GCC) || defined(NSIMD_IS_CLANG)
                 #pragma GCC diagnostic ignored "-Wformat"
               #endif
             #endif

             #include <cstdio>

             extern "C" {

             '''
    for typ in common.types:

        fmt = \
        '''NSIMD_DLLEXPORT int NSIMD_VECTORCALL
           nsimd_put_{simd_ext}_{l}{typ}(FILE *out, const char *fmt,
                                         nsimd_{simd_ext}_v{l}{typ} v) {{
             using namespace nsimd;
             {typ} buf[NSIMD_MAX_LEN({typ})];

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
                 code = fprintf(out, {fmt}, {val});
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

        ret += fmt.format(typ=typ, l='', simd_ext=simd_ext, hbar=common.hbar,
                          fmt=args[typ][0], val=args[typ][1])
        ret += fmt.format(typ=typ, l='l', simd_ext=simd_ext, hbar=common.hbar,
                          fmt=args[typ][0], val=args[typ][1])
    ret += '} // extern "C"\n'
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
