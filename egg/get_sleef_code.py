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
import shutil
import requests
import zipfile
import os

# -----------------------------------------------------------------------------

def doit(opts):
    common.myprint(opts, 'Copy native Sleef version {}'. \
                         format(opts.sleef_version))

    # First download Sleef
    sleef_dir = os.path.join(opts.script_dir, '..', '_deps-sleef')
    common.mkdir_p(sleef_dir)
    url = 'https://github.com/shibatch/sleef/archive/refs/tags/{}.zip'. \
          format(opts.sleef_version)
    r = requests.get(url, allow_redirects=True)
    sleef_zip = os.path.join(sleef_dir, 'sleef.zip')
    with open(sleef_zip, 'wb') as fout:
        fout.write(r.content)

    # Unzip sleef
    with zipfile.ZipFile(sleef_zip, 'r') as fin:
        fin.extractall(path=sleef_dir)

    # Copy helper function
    def copy(filename):
        dst_filename = os.path.basename(filename)
        shutil.copyfile(os.path.join(sleef_dir,
                                     'sleef-{}'.format(opts.sleef_version),
                                     filename), os.path.join(opts.src_dir,
                                                             dst_filename))

    # Copy files
    copy('src/libm/sleefsimddp.c')
    copy('src/libm/sleefsimdsp.c')
    copy('src/libm/sleefdp.c')
    copy('src/libm/sleefsp.c')
    copy('src/common/misc.h')
    copy('src/libm/estrin.h')
    copy('src/libm/dd.h')
    copy('src/libm/df.h')
    copy('src/libm/rempitab.c')
    copy('src/arch/helpersse2.h')
    copy('src/arch/helperavx.h')
    copy('src/arch/helperavx2.h')
    copy('src/arch/helperavx512f.h')
    copy('src/arch/helperneon32.h')
    copy('src/arch/helperadvsimd.h')

    # Sleef uses aliases but we don't need those so we comment them
    def comment_DALIAS_lines(filename):
        src = os.path.join(opts.src_dir, filename)
        dst = os.path.join(opts.src_dir, 'tmp.c')
        with open(src, 'r') as fin, open(dst, 'w') as fout:
            for line in fin:
                if line.startswith('DALIAS_'):
                    fout.write('/* {} */\n'.format(line.strip()))
                else:
                    fout.write(line)
        shutil.copyfile(dst, src)
        os.remove(dst)
    comment_DALIAS_lines('sleefsimdsp.c')
    comment_DALIAS_lines('sleefsimddp.c')

    # Sleef provides runtime SIMD detection via cpuid but we don't need it
    def replace_x86_cpuid(filename):
        src = os.path.join(opts.src_dir, filename)
        dst = os.path.join(opts.src_dir, 'tmp.c')
        with open(src, 'r') as fin, open(dst, 'w') as fout:
            for line in fin:
                if line.startswith('void Sleef_x86CpuID'):
                    fout.write(
                    '''static inline
                       void Sleef_x86CpuID(int32_t out[4], uint32_t eax,
                                           uint32_t ecx) {
                         /* We don't care for cpuid detection */
                         out[0] = 0xFFFFFFFF;
                         out[1] = 0xFFFFFFFF;
                         out[2] = 0xFFFFFFFF;
                         out[3] = 0xFFFFFFFF;
                       }
                       ''')
                else:
                    fout.write(line)
        shutil.copyfile(dst, src)
        os.remove(dst)
    replace_x86_cpuid('helpersse2.h')
    replace_x86_cpuid('helperavx.h')
    replace_x86_cpuid('helperavx2.h')
    replace_x86_cpuid('helperavx512f.h')

    # Sleef uses force inline through its INLINE macro defined in misc.h
    # We modify it to avoid warnings and because force inline has been a pain
    # in the past. We also rename some exported symbols.
    with open(os.path.join(opts.src_dir, 'misc.h'), 'a') as fout:
        fout.write(
        '''/* NSIMD specific */
           #ifndef NSIMD_SLEEF_MISC_H
           #define NSIMD_SLEEF_MISC_H

           #ifdef INLINE
           #undef INLINE
           #endif
           #define INLINE inline

           #define Sleef_rempitabdp nsimd_sleef_rempitab_f64
           #define Sleef_rempitabsp nsimd_sleef_rempitab_f32

           #endif''')

    # Sleef functions must be renamed properly for each SIMD extensions.
    # Moreover their name must contain their precision (in ULPs). This
    # precision is not the same for all functions and some functions can have
    # differents flavours (or precisions). The "database" is contained within
    # src/libm/funcproto.h. So we parse it and produce names
    # in headers "rename[SIMD ext].h" to avoid modifying Sleef C files.
    funcproto = os.path.join(sleef_dir, 'sleef-{}'.format(opts.sleef_version),
                             'src', 'libm', 'funcproto.h')
    defines = []
    ulp_suffix = {
        '0' : '',
        '1' : '_u1',
        '2' : '_u05',
        '3' : '_u35',
        '4' : '_u15',
        '5' : '_u3500'
    }
    with open(funcproto, 'r') as fin:
        for line in fin:
            if (line.find('{') != -1 and line.find('}') != -1):
                items = [item.strip() \
                         for item in line.strip(' \n\r{},').split(',')]
                items[0] = items[0].strip('"')
                if items[0] == 'NULL':
                    break
                sleef_name_f64 = items[0] + ulp_suffix[items[2]]
                sleef_name_f32 = items[0] + 'f' + ulp_suffix[items[2]]
                items[1] = items[1] if items[1] != '5' else '05'
                if items[1] == '-1':
                    nsimd_name_f64 = 'nsimd_sleef_{}_{{nsimd_ext}}_f64'. \
                                     format(items[0])
                    nsimd_name_f32 = 'nsimd_sleef_{}_{{nsimd_ext}}_f32'. \
                                     format(items[0])
                else:
                    nsimd_name_f64 = \
                    'nsimd_sleef_{}_u{}{{det}}_{{nsimd_ext}}_f64'. \
                    format(items[0], items[1])
                    nsimd_name_f32 = \
                    'nsimd_sleef_{}_u{}{{det}}_{{nsimd_ext}}_f32'. \
                    format(items[0], items[1])
                defines.append('#define x{} {}'.format(sleef_name_f64,
                                                       nsimd_name_f64))
                defines.append('#define x{} {}'.format(sleef_name_f32,
                                                       nsimd_name_f32))
    defines = '\n'.join(defines)

    sleef_to_nsimd = {
        '':        ['scalar'],
        'sse2':    ['sse2'],
        'sse4':    ['sse42'],
        'avx':     ['avx'],
        'avx2':    ['avx2'],
        'avx512f': ['avx512_knl', 'avx512_skylake'],
        'neon32':  ['neon128'],
        'advsimd': ['aarch64'],
        'sve':     ['sve128', 'sve256', 'sve512', 'sve1024', 'sve2048']
    }

    for simd_ext in ['', 'sse2', 'sse4', 'avx', 'avx2', 'avx512f', 'neon32',
                     'advsimd', 'sve']:
        renameheader = os.path.join(opts.src_dir,
                                    'rename{}.h'.format(simd_ext))
        se = simd_ext if simd_ext != '' else 'scalar'
        with open(renameheader, 'w') as fout:
            fout.write(
            '''#ifndef RENAME{SIMD_EXT}_H
               #define RENAME{SIMD_EXT}_H

               '''.format(SIMD_EXT=se.upper()))
            for nse in sleef_to_nsimd[simd_ext]:
                ifdef = '' if simd_ext == '' \
                           else '#ifdef NSIMD_{}'.format(nse.upper())
                endif = '' if simd_ext == '' else '#endif'
                fout.write(
                '''{hbar}
                   /* Naming of functions {nsimd_ext} */

                   {ifdef}

                   #ifdef DETERMINISTIC

                   {defines_det_f32}

                   #else

                   {defines_nondet_f32}

                   #endif

                   #define rempi nsimd_sleef_rempi_{nsimd_ext}
                   #define rempif nsimd_sleef_rempif_{nsimd_ext}
                   #define rempisub nsimd_sleef_rempisub_{nsimd_ext}
                   #define rempisubf nsimd_sleef_rempisubf_{nsimd_ext}
                   #define gammak nsimd_gammak_{nsimd_ext}
                   #define gammafk nsimd_gammafk_{nsimd_ext}

                   #endif

                   {endif}

                   '''.format(NSIMD_EXT=nse.upper(), nsimd_ext=nse,
                   hbar=common.hbar, ifdef=ifdef, endif=endif,
                   defines_det_f32=defines.format(det='d', nsimd_ext=nse),
                   defines_nondet_f32=defines.format(det='', nsimd_ext=nse),
                   defines_det_f64=defines.format(det='d', nsimd_ext=nse),
                   defines_nondet_f64=defines.format(det='', nsimd_ext=nse)))

            common.clang_format(opts, renameheader)
