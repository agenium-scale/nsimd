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

# This file gives the implementation of platform x86, i.e. Intel/AMD SIMD.
# Reading this file is NOT straightforward. X86 SIMD extensions is a mess.
# This script nonetheless tries to be as readable as possible. It implements
# SSE2, SSE42, AVX, AVX2, AVX512 as found on KNLs and AVX512 as found on Xeon
# Skylakes.

import common
import x86_load_store_deg234 as ldst234

# -----------------------------------------------------------------------------
# Helpers

sse = ['sse2', 'sse42']
avx = ['avx', 'avx2']
avx512 = ['avx512_knl', 'avx512_skylake']

# -----------------------------------------------------------------------------
# Implementation of mandatory functions for this module

def get_simd_exts():
    return ['sse2', 'sse42', 'avx', 'avx2', 'avx512_knl', 'avx512_skylake']

def emulate_fp16(simd_ext):
    if not simd_ext in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    return True

def get_type(simd_ext, typ):
    # Number of bits
    if simd_ext in sse:
        bits = '128'
    elif simd_ext in avx:
        bits = '256'
    elif simd_ext in avx512:
        bits = '512'
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    # Suffix
    if typ == 'f16':
        return 'struct {{__m{} v0; __m{} v1; }}'.format(bits, bits)
    elif typ == 'f32':
        return '__m{}'.format(bits)
    elif typ == 'f64':
        return '__m{}d'.format(bits)
    elif typ in common.iutypes:
        return '__m{}i'.format(bits)
    else:
        raise ValueError('Unknown type "{}"'.format(typ))

def get_logical_type(simd_ext, typ):
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    if simd_ext in sse + avx:
        return get_type(simd_ext, typ)
    elif simd_ext in avx512:
        if typ == 'f16':
            return 'struct { __mmask16 v0; __mmask16 v1; }'
        return '__mmask{}'.format(512 // common.bitsize(typ))
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_nb_registers(simd_ext):
    if simd_ext in sse + avx:
        return '16'
    elif simd_ext in avx512:
        return '32'
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def has_compatible_SoA_types(simd_ext):
    if simd_ext not in sse + avx + avx512:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    else:
        return False

def get_additional_include(func, platform, simd_ext):
    ret = ''
    if simd_ext == 'sse2':
        ret += '''#include <nsimd/cpu/cpu/{}.h>
                  '''.format(func)
    elif simd_ext == 'sse42':
        ret += '''#include <nsimd/x86/sse2/{}.h>
                  '''.format(func)
    elif simd_ext == 'avx':
        ret += '''#include <nsimd/x86/sse42/{}.h>
                  '''.format(func)
    elif simd_ext == 'avx2':
        ret += '''#include <nsimd/x86/avx/{}.h>
                  '''.format(func)
    elif simd_ext == 'avx512_knl':
        ret += '''#include <nsimd/x86/avx2/{}.h>
                  '''.format(func)
    elif simd_ext == 'avx512_skylake':
        ret += '''#include <nsimd/x86/avx2/{}.h>
                  '''.format(func)
    if func in ['loadla', 'loadlu', 'storela', 'storelu']:
        ret += '''#include <nsimd/x86/{simd_ext}/set1.h>
                  #include <nsimd/x86/{simd_ext}/eq.h>
                  #include <nsimd/x86/{simd_ext}/notl.h>
                  #include <nsimd/x86/{simd_ext}/if_else1.h>
                  '''.format(simd_ext=simd_ext)
    if func in ['notb']:
        ret += '''#include <nsimd/x86/{simd_ext}/andnotb.h>
                  '''.format(simd_ext=simd_ext)
    if func in ['notl']:
        ret += '''#include <nsimd/x86/{simd_ext}/andnotb.h>
                  #include <nsimd/x86/{simd_ext}/andnotl.h>
                  '''.format(simd_ext=simd_ext)
    if func in ['min', 'max']:
        ret += '''#include <nsimd/x86/{simd_ext}/gt.h>
                  '''.format(simd_ext=simd_ext)
    if func in ['lt']:
        ret += '''#include <nsimd/x86/{simd_ext}/gt.h>
                  '''.format(simd_ext=simd_ext)
    if func in ['ge']:
        ret += '''#include <nsimd/x86/{simd_ext}/lt.h>
                  '''.format(simd_ext=simd_ext)
    if func in ['if_else1']:
        ret += '''#include <nsimd/x86/{simd_ext}/notb.h>
                  #include <nsimd/x86/{simd_ext}/orb.h>
                  #include <nsimd/x86/{simd_ext}/andnotb.h>
                  #include <nsimd/x86/{simd_ext}/andb.h>
                  '''.format(simd_ext=simd_ext)
    if func in ['abs']:
        ret += '''#include <nsimd/x86/{simd_ext}/if_else1.h>
                  #include <nsimd/x86/{simd_ext}/set1.h>
                  '''.format(simd_ext=simd_ext)
    if func == 'reinterpretl' and simd_ext in ['sse', 'avx']:
        ret += '''#include <nsimd/x86/{simd_ext}/storeu.h>
                  #include <nsimd/x86/{simd_ext}/loadu.h>
                  '''.format(simd_ext=simd_ext)
    if func == 'upcvt':
        ret += '''#include <nsimd/x86/{simd_ext}/cvt.h>
                  '''.format(simd_ext=simd_ext)
    if func == 'ziplo' and simd_ext in ['avx512_knl', 'avx512_skylake']:
        ret += '''#include <nsimd/x86/avx2/ziphi.h>
                  '''.format(simd_ext=simd_ext)
    if func == 'ziphi' and simd_ext in ['avx512_knl', 'avx512_skylake']:
        ret += '''#include <nsimd/x86/avx2/ziplo.h>
                  '''.format(simd_ext=simd_ext)
    if func == 'zip':
        ret += '''#include <nsimd/x86/{simd_ext}/ziplo.h>
                  #include <nsimd/x86/{simd_ext}/ziphi.h>
                  '''.format(simd_ext=simd_ext)
    if simd_ext in avx512 and func in ['loadlu', 'loadla']:
        ret += '''
                  #if NSIMD_CXX > 0
                  extern "C" {{
                  #endif

                  NSIMD_INLINE
                  nsimd_{simd_ext}_vlu16
                  nsimd_{func}_{simd_ext}_u16(const u16*);

                  #if NSIMD_CXX > 0
                  }} // extern "C"
                  #endif
                  '''.format(func=func, **fmtspec)
    if func in ['load2u', 'load3u', 'load4u', 'load2a', 'load3a', 'load4a']:
        ret += '''
                  #include <nsimd/x86/{simd_ext}/loadu.h>
                  #include <nsimd/x86/{simd_ext}/storeu.h>

                  #if NSIMD_CXX > 0
                  extern "C" {{
                  #endif

                  NSIMD_INLINE nsimd_{simd_ext}_vu16x{deg}
                  nsimd_{func}_{simd_ext}_u16(const u16*);

                  #if NSIMD_CXX > 0
                  }} // extern "C"
                  #endif
                  '''.format(func=func, deg=func[4], **fmtspec)
    if func in ['store2u', 'store3u', 'store4u', 'store2a', 'store3a',
                'store4a']:
        deg = func[5]
        args = ','.join(['nsimd_{simd_ext}_vu16'.format(**fmtspec) \
                         for i in range(1, int(deg) + 1)])
        ret += '''
                  #include <nsimd/x86/{simd_ext}/loadu.h>
                  #include <nsimd/x86/{simd_ext}/storeu.h>

                  #if NSIMD_CXX > 0
                  extern "C" {{
                  #endif

                  NSIMD_INLINE void nsimd_{func}_{simd_ext}_u16(u16*, {args});

                  #if NSIMD_CXX > 0
                  }} // extern "C"
                  #endif
                  '''.format(func=func, deg=deg, args=args, **fmtspec)
    if func == 'to_logical':
        ret += '''#include <nsimd/x86/{simd_ext}/ne.h>
                  #include <nsimd/x86/{simd_ext}/reinterpretl.h>
                  '''.format(simd_ext=simd_ext)

    return ret

# -----------------------------------------------------------------------------
# Function prefixes and suffixes

def pre(simd_ext):
    # Number of bits
    if simd_ext in sse:
        bits = ''
    elif simd_ext in avx:
        bits = '256'
    elif simd_ext in avx512:
        bits = '512'
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    return '_mm{}_'.format(bits)

def suf_ep(typ):
    if typ == 'f16':
        return '_ph'
    elif typ == 'f32':
        return '_ps'
    elif typ == 'f64':
        return '_pd'
    elif typ in common.iutypes:
        return '_epi{}'.format(typ[1:])
    else:
        raise ValueError('Unknown type "{}"'.format(typ))

def nbits(simd_ext):
    if simd_ext in sse:
        return '128'
    elif simd_ext in avx:
        return '256'
    else:
        return '512'

def suf_si(simd_ext, typ):
    if typ == 'f16':
        return '_ph'
    elif typ == 'f32':
        return '_ps'
    elif typ == 'f64':
        return '_pd'
    elif typ in common.iutypes:
        return '_si{}'.format(nbits(simd_ext))
    else:
        raise ValueError('Unknown type "{}"'.format(typ))

# -----------------------------------------------------------------------------
# Other helper functions

fmtspec = {}

LO = 0
HI = 1

def extract(simd_ext, typ, lohi, var):
    if simd_ext in avx:
        lohi_arg = '0' if lohi == LO else '1'
        if typ == 'f32':
            if lohi == LO:
                return '_mm256_castps256_ps128({})'.format(var)
            else:
                return '_mm256_extractf128_ps({}, 1)'.format(var)
        elif typ == 'f64':
            if lohi == LO:
                return '_mm256_castpd256_pd128({})'.format(var)
            else:
                return '_mm256_extractf128_pd({}, 1)'.format(var)
        else:
            if lohi == LO:
                return '_mm256_castsi256_si128({})'.format(var)
            else:
                return '_mm256_extractf128_si256({}, 1)'.format(var)
    elif simd_ext in avx512:
        lohi_arg = '0' if lohi == LO else '1'
        if typ == 'f32':
            if lohi == LO:
                return '_mm512_castps512_ps256({})'.format(var)
            else:
                return '''_mm256_castsi256_ps(_mm512_extracti64x4_epi64(
                              _mm512_castps_si512({}), 1))'''.format(var)
        elif typ == 'f64':
            if lohi == LO:
                return '_mm512_castpd512_pd256({})'.format(var)
            else:
                return '_mm512_extractf64x4_pd({}, 1)'.format(var)
        else:
            if lohi == LO:
                return '_mm512_castsi512_si256({})'.format(var)
            else:
                return '_mm512_extracti64x4_epi64({}, 1)'.format(var)

def setr(simd_ext, typ, var1, var2):
    if simd_ext in avx:
        if typ == 'f32':
            return '''_mm256_insertf128_ps(_mm256_castps128_ps256(
                        {}), {}, 1)'''.format(var1, var2)
        elif typ == 'f64':
            return '''_mm256_insertf128_pd(_mm256_castpd128_pd256(
                        {}), {}, 1)'''.format(var1, var2)
        else:
            return '''_mm256_insertf128_si256(_mm256_castsi128_si256(
                        {}), {}, 1)'''.format(var1, var2)
    elif simd_ext in avx512:
        if typ == 'f32':
            return '''_mm512_castpd_ps(_mm512_insertf64x4(_mm512_castps_pd(
                        _mm512_castps256_ps512({})), _mm256_castps_pd(
                          {}), 1))'''. \
                      format(var1, var2)
        elif typ == 'f64':
            return '''_mm512_insertf64x4(_mm512_castpd256_pd512(
                        {}), {}, 1)'''.format(var1, var2)
        else:
            return '''_mm512_inserti64x4(_mm512_castsi256_si512(
                        {}), {}, 1)'''.format(var1, var2)

def how_it_should_be_op2(func, simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = {pre}{func}_ps({in0}.v0, {in1}.v0);
                  ret.v1 = {pre}{func}_ps({in0}.v1, {in1}.v1);
                  return ret;'''.format(func=func, **fmtspec)
    else:
        return 'return {pre}{func}{suf}({in0}, {in1});'. \
               format(func=func, **fmtspec)

def split_opn(func, simd_ext, typ, n):
    simd_ext2 = 'sse42' if simd_ext in avx else 'avx2'
    inp = [common.in0, common.in1, common.in2]
    defi = ''
    for i in range(0, n):
        defi += \
        '''nsimd_{simd_ext2}_v{typ} v{i}0 = {extract_loi};
           nsimd_{simd_ext2}_v{typ} v{i}1 = {extract_hii};'''. \
           format(simd_ext2=simd_ext2, typ=typ, i=i,
                  extract_loi=extract(simd_ext, typ, LO, inp[i]),
                  extract_hii=extract(simd_ext, typ, HI, inp[i]))
    vlo = ', '.join(['v{}0'.format(i) for i in range(0, n)])
    vhi = ', '.join(['v{}1'.format(i) for i in range(0, n)])
    return '''{defi}
              v00 = nsimd_{func}_{simd_ext2}_{typ}({vlo});
              v01 = nsimd_{func}_{simd_ext2}_{typ}({vhi});
              return {merge};'''. \
              format(defi=defi, vlo=vlo, vhi=vhi,
                     func=func, simd_ext2=simd_ext2, typ=typ,
                     merge=setr(simd_ext, typ, 'v00', 'v01'))

def split_op2(func, simd_ext, typ):
    return split_opn(func, simd_ext, typ, 2)

def emulate_op2(op, simd_ext, typ):
    return '''int i;
              {typ} buf0[{le}], buf1[{le}];
              {pre}storeu{sufsi}(({intrin_typ}*)buf0, {in0});
              {pre}storeu{sufsi}(({intrin_typ}*)buf1, {in1});
              for (i = 0; i < {le}; i++) {{
                buf0[i] = ({typ})(buf0[i] {op} buf1[i]);
              }}
              return {pre}loadu{sufsi}(({intrin_typ}*)buf0);'''. \
              format(intrin_typ=get_type(simd_ext, typ), op=op, **fmtspec)

def emulate_op1(func, simd_ext, typ):
    if typ in common.iutypes:
        cast = '({}*)'.format(get_type(simd_ext, typ))
    else:
        cast = ''
    return '''int i;
              {typ} buf0[{le}];
              {pre}storeu{sufsi}({cast}buf0, {in0});
              for (i = 0; i < {le}; i += nsimd_len_cpu_{typ}()) {{
                nsimd_storeu_cpu_{typ}(&buf0[i], nsimd_{func}_cpu_{typ}(
                  nsimd_loadu_cpu_{typ}(&buf0[i])));
              }}
              return {pre}loadu{sufsi}({cast}buf0);'''. \
              format(cast=cast, func=func, **fmtspec)

def split_cmp2(func, simd_ext, typ):
    simd_ext2 = 'sse42' if simd_ext in avx else 'avx2'
    leo2 = int(fmtspec['le']) // 2
    if simd_ext in avx512:
        if typ in ['i8', 'u8', 'f32', 'f64']:
            merge = \
            '''return (__mmask{le})(u32)_mm256_movemask{suf}(
                        v00) | ((__mmask{le})(u32)_mm256_movemask{suf}(
                          v01) << {leo2});'''. \
                       format(leo2=leo2, **fmtspec)
        elif typ in ['i32', 'u32', 'i64', 'u64']:
            ftyp = 'f{typnbits}'.format(**fmtspec)
            merge = \
            '''return (__mmask{le})(u32)_mm256_movemask{fsuf}(
                        _mm256_castsi256{suf}(v00)) |
                          (((__mmask{le})(u32)_mm256_movemask{fsuf}(
                            _mm256_castsi256{suf}(v01))) << {leo2});'''. \
                            format(fsuf=suf_ep(ftyp), leo2=leo2, **fmtspec)
        else:
            merge = \
            '''v00 = _mm256_permute4x64_epi64(v00, 216); /* exchange middle qwords */
               nsimd_avx2_vi16 lo1 = _mm256_unpacklo_epi16(v00, v00);
               nsimd_avx2_vi16 hi1 = _mm256_unpackhi_epi16(v00, v00);
               v01 = _mm256_permute4x64_epi64(v01, 216); /* exchange middle qwords */
               nsimd_avx2_vi16 lo2 = _mm256_unpacklo_epi16(v01, v01);
               nsimd_avx2_vi16 hi2 = _mm256_unpackhi_epi16(v01, v01);
               return (__mmask32)(u32)_mm256_movemask_ps(
                                   _mm256_castsi256_ps(lo1)) |
                      (__mmask32)((u32)_mm256_movemask_ps(
                                   _mm256_castsi256_ps(hi1)) << 8) |
                      (__mmask32)((u32)_mm256_movemask_ps(
                                   _mm256_castsi256_ps(lo2)) << 16) |
                      (__mmask32)((u32)_mm256_movemask_ps(
                                   _mm256_castsi256_ps(hi2)) << 24);'''. \
                                   format(**fmtspec)
    else:
        merge = 'return {};'.format(setr(simd_ext, typ, 'v00', 'v01'))
    return '''nsimd_{simd_ext2}_v{typ} v00 = {extract_lo0};
              nsimd_{simd_ext2}_v{typ} v01 = {extract_hi0};
              nsimd_{simd_ext2}_v{typ} v10 = {extract_lo1};
              nsimd_{simd_ext2}_v{typ} v11 = {extract_hi1};
              v00 = nsimd_{func}_{simd_ext2}_{typ}(v00, v10);
              v01 = nsimd_{func}_{simd_ext2}_{typ}(v01, v11);
              {merge}'''. \
              format(simd_ext2=simd_ext2,
                     extract_lo0=extract(simd_ext, typ, LO, common.in0),
                     extract_hi0=extract(simd_ext, typ, HI, common.in0),
                     extract_lo1=extract(simd_ext, typ, LO, common.in1),
                     extract_hi1=extract(simd_ext, typ, HI, common.in1),
                     func=func, merge=merge, **fmtspec)

def f16_cmp2(func, simd_ext):
    return '''nsimd_{simd_ext}_vlf16 ret;
              ret.v0 = nsimd_{func}_{simd_ext}_f32({in0}.v0, {in1}.v0);
              ret.v1 = nsimd_{func}_{simd_ext}_f32({in0}.v1, {in1}.v1);
              return ret;'''.format(func=func, **fmtspec)

def cmp2_with_add(func, simd_ext, typ):
    cte = { 'u8': '0x80', 'u16': '0x8000', 'u32': '0x80000000',
            'u64': '0x8000000000000000' }
    return \
    '''nsimd_{simd_ext}_v{typ} cte = nsimd_set1_{simd_ext}_{typ}({cte});
       return nsimd_{func}_{simd_ext}_{ityp}(
                {pre}add{suf}({in0}, cte),
                {pre}add{suf}({in1}, cte));'''. \
                format(func=func, cte=cte[typ],
                       ityp='i{}'.format(typ[1:]), **fmtspec)

# -----------------------------------------------------------------------------
# Returns C code for func

## Load

def load(simd_ext, typ, aligned):
    align = '' if aligned else 'u'
    cast = '(__m{}i*)'.format(nbits(simd_ext)) if typ in common.iutypes else ''
    if typ == 'f16':
        if simd_ext in sse:
            return \
            '''#ifdef NSIMD_FP16
                 nsimd_{simd_ext}_vf16 ret;
                 __m128i v = _mm_load{align}_si128((__m128i*){in0});
                 ret.v0 = _mm_cvtph_ps(v);
                 v = _mm_shuffle_epi32(v, 14); /* = (3 << 2) | (2 << 0) */
                 ret.v1 = _mm_cvtph_ps(v);
                 return ret;
               #else
                 /* Note that we can do much better but is it useful? */
                 nsimd_{simd_ext}_vf16 ret;
                 f32 buf[4];
                 buf[0] = nsimd_u16_to_f32(*(u16*){in0});
                 buf[1] = nsimd_u16_to_f32(*((u16*){in0} + 1));
                 buf[2] = nsimd_u16_to_f32(*((u16*){in0} + 2));
                 buf[3] = nsimd_u16_to_f32(*((u16*){in0} + 3));
                 ret.v0 = _mm_loadu_ps(buf);
                 buf[0] = nsimd_u16_to_f32(*((u16*){in0} + 4));
                 buf[1] = nsimd_u16_to_f32(*((u16*){in0} + 5));
                 buf[2] = nsimd_u16_to_f32(*((u16*){in0} + 6));
                 buf[3] = nsimd_u16_to_f32(*((u16*){in0} + 7));
                 ret.v1 = _mm_loadu_ps(buf);
                 return ret;
               #endif'''.format(align=align, **fmtspec)
        elif simd_ext in avx:
            return '''#ifdef NSIMD_FP16
                        nsimd_{simd_ext}_vf16 ret;
                        ret.v0 = _mm256_cvtph_ps(_mm_load{align}_si128(
                                   (__m128i*){in0}));
                        ret.v1 = _mm256_cvtph_ps(_mm_load{align}_si128(
                                   (__m128i*){in0} + 1));
                        return ret;
                      #else
                        /* Note that we can do much better but is it useful? */
                        nsimd_{simd_ext}_vf16 ret;
                        f32 buf[8];
                        int i;
                        for (i = 0; i < 8; i++) {{
                          buf[i] = nsimd_u16_to_f32(*((u16*){in0} + i));
                        }}
                        ret.v0 = _mm256_loadu_ps(buf);
                        for (i = 0; i < 8; i++) {{
                          buf[i] = nsimd_u16_to_f32(*((u16*){in0} + (8 + i)));
                        }}
                        ret.v1 = _mm256_loadu_ps(buf);
                        return ret;
                      #endif'''.format(align=align, **fmtspec)
        elif simd_ext in avx512:
            return '''nsimd_{simd_ext}_vf16 ret;
                      ret.v0 = _mm512_cvtph_ps(
                                 _mm256_load{align}_si256((__m256i*){in0})
                               );
                      ret.v1 = _mm512_cvtph_ps(
                                 _mm256_load{align}_si256((__m256i*){in0} + 1)
                               );
                      return ret;
                      '''.format(align=align, **fmtspec)
    else:
        return 'return {pre}load{align}{sufsi}({cast}{in0});'. \
               format(align=align, cast=cast, **fmtspec)

# -----------------------------------------------------------------------------
## Loads of degree 2, 3 and 4

def load_deg234(simd_ext, typ, align, deg):
    if typ == 'f16':
        a = 'a' if align else 'u'
        code = '\n'.join([ \
               '''nsimd_storeu_{simd_ext}_u16(buf, tmp.v{i});
                  ret.v{i} = nsimd_loadu_{simd_ext}_f16((f16 *)buf);'''. \
                  format(i=i, **fmtspec) for i in range(0, deg)])
        return \
        '''nsimd_{simd_ext}_v{typ}x{deg} ret;
           u16 buf[{le}];
           nsimd_{simd_ext}_vu16x{deg} tmp =
               nsimd_load{deg}{a}_{simd_ext}_u16((u16*)a0);
           {code}
           return ret;'''.format(code=code, a=a, deg=deg, **fmtspec)
    if simd_ext in sse:
        if deg == 2:
            return ldst234.load2_sse(simd_ext, typ, align, fmtspec)
        if deg == 3:
            return ldst234.load3_sse(simd_ext, typ, align, fmtspec)
        if deg == 4:
            return ldst234.load4_sse(simd_ext, typ, align, fmtspec)
    if simd_ext in avx:
        if deg == 2:
            return ldst234.load2_avx(simd_ext, typ, align, fmtspec)
        if deg == 3:
            return ldst234.load3_avx(simd_ext, typ, align, fmtspec)
        if deg == 4:
            return ldst234.load4_avx(simd_ext, typ, align, fmtspec)
    if simd_ext in avx512:
        if deg == 2:
            return ldst234.load2_avx512(simd_ext, typ, align, fmtspec)
        if deg == 3:
            return ldst234.load3_avx512(simd_ext, typ, align, fmtspec)
        if deg == 4:
            return ldst234.load4_avx512(simd_ext, typ, align, fmtspec)
    return common.NOT_IMPLEMENTED

# -----------------------------------------------------------------------------
## Stores of degree 2, 3 and 4

def store_deg234(simd_ext, typ, align, deg):
    if typ == 'f16':
        a = 'a' if align else 'u'
        variables = ', '.join(['v{}'.format(i) for i in range(0, deg)])
        code = '\n'.join([ \
               '''nsimd_storeu_{{simd_ext}}_f16((f16 *)buf, {{in{ip1}}});
                  v{i} = nsimd_loadu_{{simd_ext}}_u16((u16 *)buf);'''. \
                  format(i=i, ip1=i + 1).format(**fmtspec) \
                  for i in range(0, deg)])
        return \
        '''nsimd_{simd_ext}_vu16 {variables};
           u16 buf[{le}];
           {code}
           nsimd_store{deg}{a}_{simd_ext}_u16((u16 *){in0}, {variables});'''. \
           format(variables=variables, code=code, a=a, deg=deg, **fmtspec)
    if simd_ext in sse:
        if deg == 2:
            return ldst234.store2(simd_ext, typ, align, fmtspec)
        if deg == 3:
            return ldst234.store3_sse(simd_ext, typ, align, fmtspec)
        if deg == 4:
            return ldst234.store4_sse(typ, align, fmtspec)
    if simd_ext in avx:
        if deg == 2:
            return ldst234.store2(simd_ext, typ, align, fmtspec)
        if deg == 3:
            return ldst234.store3_avx(simd_ext, typ, align, fmtspec)
        if deg == 4:
            return ldst234.store4_avx(simd_ext, typ, align, fmtspec)
    if simd_ext in avx512:
        if deg == 2:
            return ldst234.store2(simd_ext, typ, align, fmtspec)
        if deg == 3:
            return ldst234.store3_avx512(simd_ext, typ, align, fmtspec)
        if deg == 4:
            return ldst234.store4_avx512(simd_ext, typ, align, fmtspec)
    return common.NOT_IMPLEMENTED

# -----------------------------------------------------------------------------
## Store

def store(simd_ext, typ, aligned):
    align = '' if aligned else 'u'
    cast = '(__m{}i*)'.format(nbits(simd_ext)) if typ in common.iutypes else ''
    if typ == 'f16':
        if simd_ext in sse:
            return \
            '''#ifdef NSIMD_FP16
                 __m128i v0 = _mm_cvtps_ph(
                   {in1}.v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                 __m128i v1 = _mm_cvtps_ph(
                   {in1}.v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                 __m128d v = _mm_shuffle_pd(_mm_castsi128_pd(v0),
                               _mm_castsi128_pd(v1),
                                 0 /* = (0 << 1) | (0 << 0) */);
                 _mm_store{align}_pd((f64*){in0}, v);
               #else
                 /* Note that we can do much better but is it useful? */
                 f32 buf[4];
                 _mm_storeu_ps(buf, {in1}.v0);
                 *((u16*){in0}    ) = nsimd_f32_to_u16(buf[0]);
                 *((u16*){in0} + 1) = nsimd_f32_to_u16(buf[1]);
                 *((u16*){in0} + 2) = nsimd_f32_to_u16(buf[2]);
                 *((u16*){in0} + 3) = nsimd_f32_to_u16(buf[3]);
                 _mm_storeu_ps(buf, {in1}.v1);
                 *((u16*){in0} + 4) = nsimd_f32_to_u16(buf[0]);
                 *((u16*){in0} + 5) = nsimd_f32_to_u16(buf[1]);
                 *((u16*){in0} + 6) = nsimd_f32_to_u16(buf[2]);
                 *((u16*){in0} + 7) = nsimd_f32_to_u16(buf[3]);
               #endif'''.format(align=align, **fmtspec)
        elif simd_ext in avx:
            return \
            '''#ifdef NSIMD_FP16
                 _mm_store{align}_si128((__m128i*){in0},
                   _mm256_cvtps_ph({in1}.v0,
                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
                 _mm_store{align}_si128((__m128i*){in0} + 1,
                   _mm256_cvtps_ph({in1}.v1,
                     _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
               #else
                 /* Note that we can do much better but is it useful? */
                 int i;
                 f32 buf[8];
                 _mm256_storeu_ps(buf, {in1}.v0);
                 for (i = 0; i < 8; i++) {{
                   *((u16*){in0} + i) = nsimd_f32_to_u16(buf[i]);
                 }}
                 _mm256_storeu_ps(buf, {in1}.v1);
                 for (i = 0; i < 8; i++) {{
                   *((u16*){in0} + (8 + i)) = nsimd_f32_to_u16(buf[i]);
                 }}
               #endif'''.format(align=align, **fmtspec)
        elif simd_ext in avx512:
            return \
            '''_mm256_store{align}_si256((__m256i*){in0},
                   _mm512_cvtps_ph({in1}.v0,
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
               _mm256_store{align}_si256((__m256i*){in0} + 1,
                   _mm512_cvtps_ph({in1}.v1,
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));'''. \
                        format(align=align, **fmtspec)
    else:
        return '{pre}store{align}{sufsi}({cast}{in0}, {in1});'. \
               format(align=align, cast=cast, **fmtspec)

# -----------------------------------------------------------------------------
## Code for binary operators: and, or, xor

def binop2(func, simd_ext, typ, logical=False):
    logical = 'l' if logical else ''
    func = func[0:-1]
    if typ == 'f16':
        return \
        '''nsimd_{simd_ext}_v{logi}f16 ret;
           ret.v0 = nsimd_{func}{logi2}_{simd_ext}_f32({in0}.v0, {in1}.v0);
           ret.v1 = nsimd_{func}{logi2}_{simd_ext}_f32({in0}.v1, {in1}.v1);
           return ret;'''.format(logi='l' if logical else '', func=func,
                                 logi2='l' if logical else 'b', **fmtspec)
    normal = 'return {pre}{func}{sufsi}({in0}, {in1});'. \
             format(func=func, **fmtspec)
    if simd_ext in sse:
        return normal
    if simd_ext in avx:
        if simd_ext == 'avx2' or typ in ['f32', 'f64']:
            return normal
        else:
            return '''return _mm256_castpd_si256(_mm256_{func}_pd(
                               _mm256_castsi256_pd({in0}),
                                 _mm256_castsi256_pd({in1})));'''. \
                                 format(func=func, **fmtspec)
    if simd_ext in avx512:
        if simd_ext == 'avx512_skylake' or typ in common.iutypes:
            return normal
        else:
            return \
            '''return _mm512_castsi512{suf}(_mm512_{func}_si512(
                        _mm512_cast{typ2}_si512({in0}),
                          _mm512_cast{typ2}_si512({in1})));'''. \
                          format(func=func, typ2=suf_ep(typ)[1:], **fmtspec)

# -----------------------------------------------------------------------------
## Code for logical binary operators: andl, orl, xorl

def binlop2(func, simd_ext, typ):
    op = { 'orl': '|', 'xorl': '^', 'andl': '&' }
    op_fct = { 'orl': 'kor', 'xorl': 'kxor', 'andl': 'kand' }
    if simd_ext not in avx512:
        if typ == 'f16':
            return binop2(func, simd_ext, typ, True)
        else:
            return binop2(func, simd_ext, typ)
    else: # avx512
        if typ == 'f16':
            return '''nsimd_{simd_ext}_vlf16 ret;
                      ret.v0 = (__mmask16)({in0}.v0 {op} {in1}.v0);
                      ret.v1 = (__mmask16)({in0}.v1 {op} {in1}.v1);
                      return ret;'''. \
                      format(op=op[func], op_fct=op_fct[func], **fmtspec)
        else:
            # gcc:  error: inlining failed in call to always_inline
            #              ‘__mmask8 _kor_mask8(__mmask8, __mmask8)’:
            #              target specific option mismatch
            # icc: tests fail with _{op_fct}_mask8 for f64, u64 and i64
            r = ''
            if fmtspec['le'] == 8:
                r += '#if defined(NSIMD_IS_CLANG) || defined(NSIMD_IS_GCC) || defined(NSIMD_IS_ICC)'
                r += '\n'
            else:
                r += '#if defined(NSIMD_IS_CLANG)' + '\n'
            r += '  return (__mmask{le})({in0} {op} {in1});' + '\n'
            r += '#else' + '\n'
            r += '  return _{op_fct}_mask{le}({in0}, {in1});' + '\n'
            r += '#endif' + '\n'
            return r.format(op=op[func], op_fct=op_fct[func], **fmtspec)

# -----------------------------------------------------------------------------
## andnot

def andnot2(simd_ext, typ, logical=False):
    if typ == 'f16':
        return \
        '''nsimd_{simd_ext}_v{logi}f16 ret;
           ret.v0 = nsimd_andnot{logi2}_{simd_ext}_f32({in0}.v0, {in1}.v0);
           ret.v1 = nsimd_andnot{logi2}_{simd_ext}_f32({in0}.v1, {in1}.v1);
           return ret;'''.format(logi='l' if logical else '',
                                 logi2='l' if logical else 'b', **fmtspec)
    if simd_ext in sse:
        return 'return _mm_andnot{sufsi}({in1}, {in0});'.format(**fmtspec)
    if simd_ext in avx:
        if simd_ext == 'avx2' or typ in ['f32', 'f64']:
            return 'return _mm256_andnot{sufsi}({in1}, {in0});'. \
                   format(**fmtspec)
        else:
            return '''return _mm256_castpd_si256(_mm256_andnot_pd(
                               _mm256_castsi256_pd({in1}),
                               _mm256_castsi256_pd({in0})));'''. \
                               format(**fmtspec)
    if simd_ext in avx512:
        if simd_ext == 'avx512_skylake' or typ in common.iutypes:
            return 'return _mm512_andnot{sufsi}({in1}, {in0});'. \
                   format(**fmtspec)
        else:
            return '''return _mm512_castsi512{suf}(_mm512_andnot_si512(
                               _mm512_cast{suf2}_si512({in1}),
                               _mm512_cast{suf2}_si512({in0})));'''. \
                               format(suf2=fmtspec['suf'][1:], **fmtspec)

# -----------------------------------------------------------------------------
## logical andnot

def landnot2(simd_ext, typ):
    if simd_ext in avx512:
        if typ == 'f16':
            return '''nsimd_{simd_ext}_vlf16 ret;
                      ret.v0 = (__mmask16)({in0}.v0 & (~{in1}.v0));
                      ret.v1 = (__mmask16)({in0}.v1 & (~{in1}.v1));
                      return ret;'''.format(**fmtspec)
        else:
            return 'return (__mmask{le})({in0} & (~{in1}));'.format(**fmtspec)
    return andnot2(simd_ext, typ, True)

# -----------------------------------------------------------------------------
## Code for unary not

def not1(simd_ext, typ, logical=False):
    if typ == 'f16':
        return \
        '''nsimd_{simd_ext}_v{logi}f16 ret;
           nsimd_{simd_ext}_vf32 cte = {pre}castsi{nbits}_ps(
                                         {pre}set1_epi8(-1));
           ret.v0 = nsimd_andnot{logi2}_{simd_ext}_f32(cte, {in0}.v0);
           ret.v1 = nsimd_andnot{logi2}_{simd_ext}_f32(cte, {in0}.v1);
           return ret;'''.format(logi='l' if logical else '',
                                 logi2='l' if logical else 'b', **fmtspec)
    elif typ in ['f32', 'f64']:
        return '''return nsimd_andnotb_{simd_ext}_{typ}(
                           {pre}castsi{nbits}{suf}(
                             {pre}set1_epi8(-1)), {in0});'''.format(**fmtspec)
    else:
        return '''return nsimd_andnotb_{simd_ext}_{typ}(
                           {pre}set1_epi8(-1), {in0});'''.format(**fmtspec)

# -----------------------------------------------------------------------------
## Code for unary logical lnot

def lnot1(simd_ext, typ):
    if simd_ext in avx512:
        if typ == 'f16':
            return '''nsimd_{simd_ext}_vlf16 ret;
                      ret.v0 = (__mmask16)(~{in0}.v0);
                      ret.v1 = (__mmask16)(~{in0}.v1);
                      return ret;'''.format(**fmtspec)
        else:
            return 'return (__mmask{le})(~{in0});'.format(**fmtspec)
    return not1(simd_ext, typ, True)

# -----------------------------------------------------------------------------
## Addition and substraction

def addsub(func, simd_ext, typ):
    if typ in common.ftypes or simd_ext in sse or \
       (simd_ext in avx512 and typ in ['u32', 'i32', 'u64', 'i64']):
        return how_it_should_be_op2(func, simd_ext, typ)
    else:
        if simd_ext in ['avx2', 'avx512_skylake']:
            return how_it_should_be_op2(func, simd_ext, typ)
        else:
            return split_op2(func, simd_ext, typ)

# -----------------------------------------------------------------------------
## Len

def len1(simd_ext, typ):
    return 'return {le};'.format(**fmtspec)

# -----------------------------------------------------------------------------
## Division

def div2(simd_ext, typ):
    if typ in common.ftypes:
        return how_it_should_be_op2('div', simd_ext, typ)
    return emulate_op2('/', simd_ext, typ)

# -----------------------------------------------------------------------------
## Multiplication

def mul2(simd_ext, typ):
    emulate = emulate_op2('*', simd_ext, typ)
    split = split_op2('mul', simd_ext, typ)
    # Floats
    if typ in common.ftypes:
        return how_it_should_be_op2('mul', simd_ext, typ)
    # Integers 16, 32 on SSE
    if simd_ext in sse and typ in ['i16', 'u16']:
        return 'return _mm_mullo_epi16({in0}, {in1});'.format(**fmtspec)
    if simd_ext in sse and typ in ['i32', 'u32']:
        if simd_ext == 'sse42':
            return 'return _mm_mullo_epi32({in0}, {in1});'.format(**fmtspec)
        else:
            return emulate
    # Integers 16, 32 on AVX
    if simd_ext in avx and typ in ['i16', 'u16', 'i32', 'u32']:
        if simd_ext == 'avx2':
            return 'return _mm256_mullo{suf}({in0}, {in1});'.format(**fmtspec)
        else:
            return split
    # Integers 64 on SSE on AVX
    if simd_ext in sse + avx and typ in ['i64', 'u64']:
        return emulate_op2('*', simd_ext, typ)
    # Integers 16 on AVX512
    if simd_ext in avx512 and typ in ['i16', 'u16']:
        if simd_ext == 'avx512_skylake':
            return 'return _mm512_mullo_epi16({in0}, {in1});'.format(**fmtspec)
        else:
            return split
    # Integers 32 on AVX512
    if simd_ext in avx512 and typ in ['i32', 'u32']:
        return 'return _mm512_mullo_epi32({in1}, {in0});'.format(**fmtspec)
    # Integers 64 on AVX512
    if simd_ext in avx512 and typ in ['i64', 'u64']:
        if simd_ext == 'avx512_skylake':
            return 'return _mm512_mullo_epi64({in0}, {in1});'.format(**fmtspec)
        else:
            return emulate
    # Integers 8 on SSE
    with_epi16 = '''nsimd_{simd_ext}_v{typ} lo =
                        {pre}mullo_epi16({in0}, {in1});
                    nsimd_{simd_ext}_v{typ} hi = {pre}slli_epi16(
                        {pre}mullo_epi16({pre}srli_epi16({in0}, 8),
                          {pre}srli_epi16({in1}, 8)), 8);
                    return {pre}or{sufsi}({pre}and{sufsi}(
                              lo, {pre}set1_epi16(255)),hi);'''. \
                    format(**fmtspec)
    split_epi16 = split_op2('mul', simd_ext, typ)
    if simd_ext in sse and typ in ['i8', 'u8']:
        return with_epi16
    if simd_ext in avx + avx512 and typ in ['i8', 'u8']:
        if simd_ext in ['avx2', 'avx512_skylake']:
            return with_epi16
        else:
            return split_epi16

# -----------------------------------------------------------------------------
## Shift left and right

def shl_shr(func, simd_ext, typ):
    if typ in ['f16', 'f32', 'f64']:
        return ''
    intrinsic = 'srl' if func == 'shr' else 'sll'
    simd_ext2 = 'sse42' if simd_ext in avx else 'avx2'
    split = '''nsimd_{simd_ext2}_v{typ} v0 = {extract_lo};
               nsimd_{simd_ext2}_v{typ} v1 = {extract_hi};
               v0 = nsimd_{func}_{simd_ext2}_{typ}(v0, {in1});
               v1 = nsimd_{func}_{simd_ext2}_{typ}(v1, {in1});
               return {merge};'''. \
               format(simd_ext2=simd_ext2, func=func,
                      extract_lo=extract(simd_ext, typ, LO, common.in0),
                      extract_hi=extract(simd_ext, typ, HI, common.in0),
                      merge=setr(simd_ext, typ, 'v0', 'v1'), **fmtspec)
    normal_16_32_64 = '''return {pre}{intrinsic}{suf}(
                           {in0}, _mm_set1_epi64x({in1}));'''. \
                      format(intrinsic=intrinsic, **fmtspec)
    FFs = '0x' + ('F' * int((int(typ[1:]) // 4)))
    FFOOs = FFs  + ('0' * int((int(typ[1:]) // 4)))
    with_2n_for_n = '''nsimd_{simd_ext}_v{typ} lo = {pre}and{sufsi}(
                         {pre}{intrinsic}_epi{typ2nbits}(
                           {in0}, _mm_set1_epi64x({in1})),
                             nsimd_set1_{simd_ext}_u{typ2nbits}({masklo}));
                       nsimd_{simd_ext}_v{typ} hi =
                         {pre}{intrinsic}_epi{typ2nbits}({pre}and{sufsi}({in0},
                           nsimd_set1_{simd_ext}_u{typ2nbits}({maskhi})),
                             _mm_set1_epi64x({in1}));
                       return {pre}or{sufsi}(hi, lo);'''. \
                       format(intrinsic=intrinsic, typ2nbits=2 * int(typ[1:]),
                              masklo=FFs if func == 'shl' else FFOOs,
                              maskhi=FFOOs if func == 'shl' else FFs, **fmtspec)
    with_32_for_8 = '''nsimd_{simd_ext}_v{typ} masklo =
                         nsimd_set1_{simd_ext}_u32(0xFF00FF);
                       nsimd_{simd_ext}_v{typ} lo =
                         {pre}and{sufsi}({pre}{intrinsic}_epi32(
                           {pre}and{sufsi}({in0}, masklo),
                             _mm_set1_epi64x({in1})), masklo);
                       nsimd_{simd_ext}_v{typ} maskhi =
                         nsimd_set1_{simd_ext}_u32(0xFF00FF00);
                       nsimd_{simd_ext}_v{typ} hi =
                           {pre}and{sufsi}({pre}{intrinsic}_epi32(
                             {pre}and{sufsi}({in0}, maskhi),
                               _mm_set1_epi64x({in1})), maskhi);
                       return {pre}or{sufsi}(hi, lo);'''. \
                       format(intrinsic=intrinsic, **fmtspec)
    if simd_ext in sse:
        if typ in ['i8', 'u8']:
            return with_2n_for_n
        if typ in ['i16', 'u16', 'i32', 'u32', 'i64', 'u64']:
            return normal_16_32_64
    if simd_ext in avx:
        if typ in ['i8', 'u8']:
            return with_2n_for_n if simd_ext == 'avx2' else split
        if typ in ['i16', 'u16', 'i32', 'u32', 'i64', 'u64']:
            return normal_16_32_64 if simd_ext == 'avx2' else split
    if simd_ext in avx512:
        if typ in ['i8', 'u8']:
            return with_2n_for_n if simd_ext == 'avx512_skylake' \
                                 else with_32_for_8
        if typ in ['i16', 'u16']:
            return normal_16_32_64 if simd_ext == 'avx512_skylake' \
                                   else with_2n_for_n
        if typ in ['i32', 'u32', 'i64', 'u64']:
            return normal_16_32_64

# -----------------------------------------------------------------------------
## set1 or splat function

def set1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  f32 f = nsimd_f16_to_f32({in0});
                  ret.v0 = {pre}set1_ps(f);
                  ret.v1 = {pre}set1_ps(f);
                  return ret;'''.format(**fmtspec)
    if simd_ext in sse + avx:
        if typ == 'i64':
            return 'return {pre}set1_epi64x({in0});'.format(**fmtspec)
        if typ == 'u64':
            return '''union {{ u64 u; i64 i; }} buf;
                      buf.u = {in0};
                      return {pre}set1_epi64x(buf.i);'''.format(**fmtspec)
    if typ in ['u8', 'u16', 'u32', 'u64']:
        return '''union {{ {typ} u; i{typnbits} i; }} buf;
                  buf.u = {in0};
                  return {pre}set1{suf}(buf.i);'''.format(**fmtspec)
    return 'return {pre}set1{suf}({in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
## Equality

def eq2(simd_ext, typ):
    if typ == 'f16':
        return f16_cmp2('eq', simd_ext)
    if simd_ext in sse:
        if typ in ['i64', 'u64']:
            if simd_ext == 'sse42':
                return how_it_should_be_op2('cmpeq', simd_ext, typ)
            else:
                return \
                '''__m128i t = _mm_cmpeq_epi32({in0}, {in1});
                   return _mm_and_si128(t,
                            _mm_shuffle_epi32(t, 177) /* = 2|3|0|1 */);'''. \
                            format(**fmtspec)
        else:
            return how_it_should_be_op2('cmpeq', simd_ext, typ)
    if simd_ext in avx:
        if typ in ['f32', 'f64']:
            return 'return _mm256_cmp{suf}({in0}, {in1}, _CMP_EQ_OQ);'. \
                   format(**fmtspec)
        else:
            if simd_ext == 'avx2':
                return how_it_should_be_op2('cmpeq', simd_ext, typ)
            else:
                return split_cmp2('eq', simd_ext, typ)
    if simd_ext in avx512:
        if typ in ['f32', 'f64']:
            return 'return _mm512_cmp{suf}_mask({in0}, {in1}, _CMP_EQ_OQ);'. \
                   format(**fmtspec)
        elif typ in ['i32', 'u32', 'i64', 'u64']:
            return \
            'return _mm512_cmp{suf}_mask({in0}, {in1}, _MM_CMPINT_EQ);'. \
            format(**fmtspec)
        else:
            if simd_ext == 'avx512_skylake':
                return \
                'return _mm512_cmp{suf}_mask({in0}, {in1}, _MM_CMPINT_EQ);'. \
                format(**fmtspec)
            else:
                return split_cmp2('eq', simd_ext, typ)

# -----------------------------------------------------------------------------
## not equal

def neq2(simd_ext, typ):
    if typ == 'f16':
        return f16_cmp2('ne', simd_ext)
    if simd_ext in sse and typ in ['f32', 'f64']:
        return how_it_should_be_op2('cmpneq', simd_ext, typ)
    if simd_ext in avx and typ in ['f32', 'f64']:
        return 'return _mm256_cmp{suf}({in0}, {in1}, _CMP_NEQ_OQ);'. \
               format(**fmtspec)
    if simd_ext in avx512 and typ in ['f32', 'f64']:
        return 'return _mm512_cmp{suf}_mask({in0}, {in1}, _CMP_NEQ_OQ);'. \
               format(**fmtspec)
    noteq = '''return nsimd_notl_{simd_ext}_{typ}(
                        nsimd_eq_{simd_ext}_{typ}({in0}, {in1}));'''. \
                        format(**fmtspec)
    if simd_ext in avx512:
        intrinsic = \
            'return _mm512_cmp{suf}_mask({in0}, {in1}, _MM_CMPINT_NE);'. \
            format(**fmtspec)
        if typ in ['i32', 'u32', 'i64', 'u64']:
            return intrinsic
        else:
            return intrinsic if  simd_ext == 'avx512_skylake' else noteq
    return noteq

# -----------------------------------------------------------------------------
## Greater than

def gt2(simd_ext, typ):
    if typ == 'f16':
        return f16_cmp2('gt', simd_ext)
    if simd_ext in sse:
        if typ in ['f32', 'f64', 'i8', 'i16', 'i32']:
            return how_it_should_be_op2('cmpgt', simd_ext, typ)
        if typ == 'i64':
            if simd_ext == 'sse42':
                return how_it_should_be_op2('cmpgt', simd_ext, typ)
            return '''return _mm_sub_epi64(_mm_setzero_si128(), _mm_srli_epi64(
                               _mm_sub_epi64({in1}, {in0}), 63));'''. \
                               format(**fmtspec)
        return cmp2_with_add('gt', simd_ext, typ)
    if simd_ext in avx:
        if typ in ['f32', 'f64']:
            return 'return _mm256_cmp{suf}({in0}, {in1}, _CMP_GT_OQ);'. \
                   format(**fmtspec)
        if typ in ['i8', 'i16', 'i32', 'i64']:
            if simd_ext == 'avx2':
                return how_it_should_be_op2('cmpgt', simd_ext, typ)
            else:
                return split_cmp2('gt', simd_ext, typ)
        if simd_ext == 'avx2':
            return cmp2_with_add('gt', simd_ext, typ)
        else:
            return split_cmp2('gt', simd_ext, typ)
    # AVX512
    if typ in ['f32', 'f64', 'i32', 'i64']:
        return \
        'return _mm512_cmp{suf}_mask({in0}, {in1}, {cte});'. \
        format(cte='_CMP_GT_OQ' if typ in ['f32', 'f64'] else '_MM_CMPINT_NLE',
               **fmtspec)
    if typ in ['u32', 'u64']:
        return \
        'return _mm512_cmp_epu{typ2}_mask({in0}, {in1}, _MM_CMPINT_NLE);'. \
        format(typ2=typ[1:], **fmtspec)
    if simd_ext == 'avx512_skylake':
        return \
        'return _mm512_cmp_ep{typ}_mask({in0}, {in1}, _MM_CMPINT_NLE);'. \
        format(**fmtspec)
    else:
        return split_cmp2('gt', simd_ext, typ)

# -----------------------------------------------------------------------------
## lesser than

def lt2(simd_ext, typ):
    return 'return nsimd_gt_{simd_ext}_{typ}({in1}, {in0});'. \
           format(**fmtspec)

# -----------------------------------------------------------------------------
## greater or equal

def geq2(simd_ext, typ):
    if typ == 'f16':
        return f16_cmp2('ge', simd_ext)
    notlt = '''return nsimd_notl_{simd_ext}_{typ}(
                        nsimd_lt_{simd_ext}_{typ}({in0}, {in1}));'''. \
            format(**fmtspec)
    if simd_ext in sse:
        if typ in ['f32', 'f64']:
            return how_it_should_be_op2('cmpge', simd_ext, typ)
    if simd_ext in avx:
        if typ in ['f32', 'f64']:
            return 'return _mm256_cmp{suf}({in0}, {in1}, _CMP_GE_OQ);'. \
                   format(**fmtspec)
    if simd_ext in avx512:
        if typ in ['i32', 'i64', 'u32', 'u64']:
            return \
              'return _mm512_cmp_ep{typ}_mask({in0}, {in1}, _MM_CMPINT_NLT);'. \
              format(**fmtspec)
        if typ in ['f32', 'f64']:
            return 'return _mm512_cmp{suf}_mask({in0}, {in1}, _CMP_GE_OQ);'. \
                   format(**fmtspec)
        if simd_ext == 'avx512_skylake':
            return \
            'return _mm512_cmp_ep{typ}_mask({in0}, {in1}, _MM_CMPINT_NLT);'. \
            format(**fmtspec)
        else:
            return notlt
    return notlt

# -----------------------------------------------------------------------------
## lesser or equal

def leq2(simd_ext, typ):
    if typ == 'f16':
        return f16_cmp2('le', simd_ext)
    notgt = '''return nsimd_notl_{simd_ext}_{typ}(
                        nsimd_gt_{simd_ext}_{typ}({in0}, {in1}));'''. \
                        format(**fmtspec)
    if simd_ext in sse and typ in ['f32', 'f64']:
        return 'return _mm_cmpngt{suf}({in0}, {in1});'.format(**fmtspec)
    if simd_ext in avx and typ in ['f32', 'f64']:
            return 'return _mm256_cmp{suf}({in0}, {in1}, _CMP_LE_OQ);'. \
                   format(**fmtspec)
    if simd_ext in avx512:
        if typ in ['i32', 'i64', 'u32', 'u64']:
            return \
              'return _mm512_cmp_ep{typ}_mask({in0}, {in1}, _MM_CMPINT_LE);'. \
              format(**fmtspec)
        if typ in ['f32', 'f64']:
            return 'return _mm512_cmp{suf}_mask({in0}, {in1}, _CMP_LE_OQ);'. \
                   format(**fmtspec)
        if simd_ext == 'avx512_skylake':
            return \
            'return _mm512_cmp_ep{typ}_mask({in0}, {in1}, _MM_CMPINT_LE);'. \
            format(**fmtspec)
        else:
            return notgt
    return notgt

# -----------------------------------------------------------------------------
## if_else1 function

def if_else1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_if_else1_{simd_ext}_f32(
                             {in0}.v0, {in1}.v0, {in2}.v0);
                  ret.v1 = nsimd_if_else1_{simd_ext}_f32(
                             {in0}.v1, {in1}.v1, {in2}.v1);
                  return ret;'''.format(**fmtspec)
    manual = '''return nsimd_orb_{simd_ext}_{typ}(
                         nsimd_andb_{simd_ext}_{typ}({in1}, {in0}),
                         nsimd_andnotb_{simd_ext}_{typ}({in2}, {in0}));'''. \
                         format(**fmtspec)
    if simd_ext in sse:
        if simd_ext == 'sse42':
            return 'return _mm_blendv{fsuf}({in2}, {in1}, {in0});'. \
                   format(fsuf=suf_ep(typ) if typ in ['f32', 'f64']
                          else '_epi8', **fmtspec)
        else:
            return manual
    if simd_ext in avx:
        if typ in ['f32', 'f64']:
            return 'return _mm256_blendv{suf}({in2}, {in1}, {in0});'. \
                   format(**fmtspec)
        else:
            if simd_ext == 'avx2':
                return 'return _mm256_blendv_epi8({in2}, {in1}, {in0});'. \
                       format(**fmtspec)
            else:
                return manual
    if simd_ext in avx512:
        if typ in ['f32', 'f64', 'i32', 'u32', 'i64', 'u64']:
            return 'return _mm512_mask_blend{suf}({in0}, {in2}, {in1});'. \
                   format(**fmtspec)
        else:
            if simd_ext == 'avx512_skylake':
                return 'return _mm512_mask_blend{suf}({in0}, {in2}, {in1});'. \
                       format(**fmtspec)
            else:
                return '''int i;
                          {typ} buf0[{le}], buf1[{le}];
                          _mm512_storeu_si512(buf0, {in1});
                          _mm512_storeu_si512(buf1, {in2});
                          for (i = 0; i < {le}; i++) {{
                            if ((({in0} >> i) & 1) == 0) {{
                              buf0[i] = buf1[i];
                            }}
                          }}
                          return _mm512_loadu_si512(buf0);'''.format(**fmtspec)

# -----------------------------------------------------------------------------
## min and max functions

def minmax(func, simd_ext, typ):
    if typ in ['f16', 'f32', 'f64']:
        return how_it_should_be_op2(func, simd_ext, typ)
    with_if_else = '''return nsimd_if_else1_{simd_ext}_{typ}(
                               nsimd_gt_{simd_ext}_{typ}(
                                 {args}), {in0}, {in1});'''. \
                   format(args = '{in0}, {in1}'.format(**fmtspec)
                            if func == 'max'
                            else '{in1}, {in0}'.format(**fmtspec), **fmtspec)
    if simd_ext in sse:
        if typ in ['u8', 'i16']:
            return 'return _mm_{func}_ep{typ}({in0}, {in1});'. \
                   format(func=func, **fmtspec)
        if typ in ['i8', 'u16', 'i32', 'u32']:
            if simd_ext == 'sse42':
                return 'return _mm_{func}_ep{typ}({in0}, {in1});'. \
                       format(func=func, **fmtspec)
            else:
                return with_if_else
    if simd_ext in avx and typ in ['i8', 'u8', 'i16', 'u16', 'i32', 'u32']:
        if simd_ext == 'avx2':
            return 'return _mm256_{func}_ep{typ}({in0}, {in1});'. \
                   format(func=func, **fmtspec)
        else:
            return split_op2(func, simd_ext, typ)
    if simd_ext in avx512:
        if typ in ['i32', 'u32', 'i64', 'u64']:
            return 'return _mm512_{func}_ep{typ}({in0}, {in1});'. \
                   format(func=func, **fmtspec)
        else:
            if simd_ext == 'avx512_skylake':
                return 'return _mm512_{func}_ep{typ}({in0}, {in1});'. \
                       format(func=func, **fmtspec)
            else:
                return split_op2(func, simd_ext, typ)
    return with_if_else

# -----------------------------------------------------------------------------
## sqrt

def sqrt1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = {pre}sqrt_ps({in0}.v0);
                  ret.v1 = {pre}sqrt_ps({in0}.v1);
                  return ret;'''.format(**fmtspec)
    return 'return {pre}sqrt{suf}({in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
## Load logical

def loadl(simd_ext, typ, aligned):
    if simd_ext in avx512:
        if typ == 'f16':
            return '''/* This can surely be improved but it is not our
                         priority. Note that we take advantage of the fact that
                         floating zero is represented as integer zero to
                         simplify code. */
                      nsimd_{simd_ext}_vlf16 ret;
                      __mmask32 tmp = nsimd_loadlu_{simd_ext}_u16((u16*){in0});
                      ret.v0 = (__mmask16)(tmp & 0xFFFF);
                      ret.v1 = (__mmask16)((tmp >> 16) & 0xFFFF);
                      return ret;'''.format(**fmtspec)
        return '''/* This can surely be improved but it is not our priority. */
                  int i;
                  __mmask{le} ret = 0;
                  for (i = 0; i < {le}; i++) {{
                    if ({in0}[i] != ({typ})0) {{
                      ret |= (__mmask{le})((__mmask{le})1 << i);
                    }}
                  }}
                  return ret;'''.format(**fmtspec)
    return \
    '''/* This can surely be improved but it is not our priority. */
       return nsimd_notl_{simd_ext}_{typ}(nsimd_eq_{simd_ext}_{typ}(
                nsimd_load{align}_{simd_ext}_{typ}(
                  {in0}), nsimd_set1_{simd_ext}_{typ}({zero})));'''. \
       format(align='a' if aligned else 'u',
              zero = 'nsimd_f32_to_f16(0.0f)' if typ == 'f16'
              else '({})0'.format(typ), **fmtspec)

# -----------------------------------------------------------------------------
## Store logical

def storel(simd_ext, typ, aligned):
    if simd_ext in avx512:
        if typ == 'f16':
            return '''/* This can surely be improved but it is not our
                         priority. Note that we take advantage of the fact that
                         floating zero is represented as integer zero to
                         simplify code. */
                      int i;
                      u16 one = 0x3C00; /* FP16 IEEE754 representation of 1 */
                      for (i = 0; i < 16; i++) {{
                        ((u16*){in0})[i] = (({in1}.v0 >> i) & 1) ? one
                                                                 : (u16)0;
                      }}
                      for (i = 0; i < 16; i++) {{
                        ((u16*){in0})[i + 16] = (({in1}.v1 >> i) & 1) ? one
                                                                      : (u16)0;
                      }}'''.format(**fmtspec)
        return '''/* This can surely be improved but it is not our priority. */
                  int i;
                  for (i = 0; i < {le}; i++) {{
                    {in0}[i] = (({in1} >> i) & 1) ? ({typ})1 : ({typ})0;
                  }}'''.format(**fmtspec)
    return \
    '''/* This can surely be improved but it is not our priority. */
       nsimd_store{align}_{simd_ext}_{typ}({in0},
         nsimd_if_else1_{simd_ext}_{typ}({in1},
           nsimd_set1_{simd_ext}_{typ}({one}),
           nsimd_set1_{simd_ext}_{typ}({zero})));'''. \
       format(align = 'a' if aligned else 'u',
              one = 'nsimd_f32_to_f16(1.0f)' if typ == 'f16'
              else '({})1'.format(typ),
              zero = 'nsimd_f32_to_f16(0.0f)' if typ == 'f16'
              else '({})0'.format(typ), **fmtspec)

# -----------------------------------------------------------------------------
## Absolute value

def abs1(simd_ext, typ):
    def mask(typ):
        return '0x7F' + ('F' * int(((int(typ[1:]) - 8) // 4)))
    if typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vf16 ret;
           nsimd_{simd_ext}_vf32 mask = {pre}castsi{nbits}_ps(
                                          nsimd_set1_{simd_ext}_u32({mask}));
           ret.v0 = nsimd_andb_{simd_ext}_f32({in0}.v0, mask);
           ret.v1 = nsimd_andb_{simd_ext}_f32({in0}.v1, mask);
           return ret;'''.format(mask=mask('f32'), **fmtspec)
    if typ in ['u8', 'u16', 'u32', 'u64']:
        return 'return {in0};'.format(**fmtspec)
    if typ in ['f32', 'f64']:
        return \
        '''nsimd_{simd_ext}_v{typ} mask = {pre}castsi{nbits}{suf}(
               nsimd_set1_{simd_ext}_u{typnbits}({mask}));
           return nsimd_andb_{simd_ext}_{typ}({in0}, mask);'''. \
           format(mask=mask(typ), **fmtspec)
    bit_twiddling_arith_shift = \
    '''nsimd_{simd_ext}_v{typ} mask = {pre}srai{suf}({in0}, {typnbitsm1});
       return {pre}xor{sufsi}({pre}add{suf}({in0}, mask), mask);'''. \
       format(typnbitsm1=int(typ[1:]) - 1, **fmtspec)
    bit_twiddling_no_arith_shift = \
    '''nsimd_{simd_ext}_v{typ} mask = {pre}sub{suf}({pre}setzero{sufsi}(),
                                        nsimd_shr_{simd_ext}_{typ}(
                                          {in0}, {typnbitsm1}));
       return {pre}xor{sufsi}({pre}add{suf}({in0}, mask), mask);'''. \
       format(typnbitsm1=int(typ[1:]) - 1, **fmtspec)
    with_blendv = \
    '''return nsimd_if_else1_{simd_ext}_{typ}({in0}, {pre}sub{suf}(
                {pre}setzero{sufsi}(), {in0}), {in0});'''.format(**fmtspec)
    if simd_ext in sse:
        if typ in ['i16', 'i32']:
            if simd_ext == 'sse42':
                return 'return _mm_abs{suf}({in0});'.format(**fmtspec)
            else:
                return bit_twiddling_arith_shift
        if typ == 'i8':
            if simd_ext == 'sse42':
                return 'return _mm_abs{suf}({in0});'.format(**fmtspec)
            else:
                return bit_twiddling_no_arith_shift
        if typ == 'i64':
            return bit_twiddling_no_arith_shift
    if simd_ext in avx:
        if typ in ['i8', 'i16', 'i32']:
            if simd_ext == 'avx2':
                return 'return _mm256_abs{suf}({in0});'.format(**fmtspec)
            else:
                return split_opn('abs', simd_ext, typ, 1)
        else:
            if simd_ext == 'avx2':
                return with_blendv
            else:
                return split_opn('abs', simd_ext, typ, 1)
    if simd_ext in avx512:
        if typ in ['i32', 'i64']:
            return 'return _mm512_abs{suf}({in0});'.format(**fmtspec)
        else:
            if simd_ext == 'avx512_skylake':
                return 'return _mm512_abs{suf}({in0});'.format(**fmtspec)
            else:
                return split_opn('abs', simd_ext, typ, 1)

# -----------------------------------------------------------------------------
## FMA and FMS

def fma_fms(func, simd_ext, typ):
    op = 'add' if func in ['fma', 'fnma'] else 'sub'
    neg = 'n' if func in ['fnma', 'fnms'] else ''
    if typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vf16 ret;
           ret.v0 = nsimd_{func}_{simd_ext}_f32({in0}.v0, {in1}.v0, {in2}.v0);
           ret.v1 = nsimd_{func}_{simd_ext}_f32({in0}.v1, {in1}.v1, {in2}.v1);
           return ret;'''.format(neg=neg, func=func, **fmtspec)
    if neg == '':
        emulate = '''return nsimd_{op}_{simd_ext}_{typ}(
                              nsimd_mul_{simd_ext}_{typ}({in0}, {in1}),
                                {in2});'''.format(op=op, **fmtspec)
    else:
        emulate = '''return nsimd_{op}_{simd_ext}_{typ}(
                              nsimd_sub_{simd_ext}_{typ}(
                                {pre}setzero{sufsi}(),
                                  nsimd_mul_{simd_ext}_{typ}({in0}, {in1})),
                                    {in2});'''.format(op=op, **fmtspec)
    # One could use only emulate and no split. But to avoid splitting and
    # merging SIMD register for each operation: sub, mul and add, we use
    # emulation only for SIMD extensions that have natively add, sub and mul
    # intrinsics.
    split = split_opn(func, simd_ext, typ, 3)
    if typ in ['f32', 'f64']:
        if simd_ext in sse + avx:
            return '''#ifdef NSIMD_FMA
                        return {pre}f{neg}m{op}{suf}({in0}, {in1}, {in2});
                      #else
                        {emulate}
                      #endif'''.format(op=op, neg=neg, emulate=emulate,
                                       **fmtspec)
        else:
            return 'return {pre}f{neg}m{op}{suf}({in0}, {in1}, {in2});'. \
                   format(op=op, neg=neg, **fmtspec)
    if simd_ext in avx:
        return emulate if simd_ext == 'avx2' else split
    if simd_ext in avx512:
        return emulate if simd_ext == 'avx512_skylake' else split
    return emulate

# -----------------------------------------------------------------------------
## Ceil and floor

def round1(func, simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_{func}_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_{func}_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(func=func, **fmtspec)
    if typ in ['f32', 'f64']:
        normal = 'return {pre}{func}{suf}({in0});'.format(func=func, **fmtspec)
        if simd_ext not in sse:
            return normal
        if simd_ext == 'sse42':
            return normal
        else:
            return emulate_op1(func, simd_ext, typ)
    return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Trunc

def trunc1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_trunc_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_trunc_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(**fmtspec)
    if typ in ['f32', 'f64']:
        normal = '''return {pre}round{suf}({in0}, _MM_FROUND_TO_ZERO |
                               _MM_FROUND_NO_EXC);'''.format(**fmtspec)
        if simd_ext == 'sse2':
            return emulate_op1('trunc', simd_ext, typ)
        if simd_ext == 'sse42':
            return normal
        if simd_ext in avx:
            return normal
        if simd_ext in avx512:
            return \
            '''__mmask{le} cond = nsimd_gt_{simd_ext}_{typ}(
                                    {in0}, _mm512_setzero{sufsi}());
               return nsimd_if_else1_{simd_ext}_{typ}(cond,
                        nsimd_floor_{simd_ext}_{typ}({in0}),
                          nsimd_ceil_{simd_ext}_{typ}({in0}));'''. \
                          format(**fmtspec)
    return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Round to even

def round_to_even1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_round_to_even_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_round_to_even_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(**fmtspec)
    if typ in ['f32', 'f64']:
        normal = '''return {pre}round{suf}({in0}, _MM_FROUND_TO_NEAREST_INT |
                               _MM_FROUND_NO_EXC);'''.format(**fmtspec)
        if simd_ext == 'sse2':
            return emulate_op1('round_to_even', simd_ext, typ)
        if simd_ext == 'sse42':
            return normal
        if simd_ext in avx:
            return normal
        if simd_ext in avx512:
            return 'return _mm512_roundscale{suf}({in0}, 0);'.format(**fmtspec)
    return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------
## All and any functions

def all_any(func, simd_ext, typ):
    if typ == 'f16':
        return \
        '''return nsimd_{func}_{simd_ext}_f32({in0}.v0) {and_or}
                  nsimd_{func}_{simd_ext}_f32({in0}.v1);'''. \
                  format(func=func, and_or='&&' if func == 'all' else '||',
                         **fmtspec)
    if simd_ext in sse:
        if typ in common.iutypes:
            return 'return (u32)_mm_movemask_epi8({in0}) {test};'. \
                   format(test='== 0xFFFF' if func == 'all' else '!= 0u',
                          **fmtspec)
        else:
            mask = '0xF' if typ == 'f32' else '0x3'
            return 'return (u32)_mm_movemask{suf}({in0}) {test};'. \
                   format(test='== ' + mask if func == 'all' else '!= 0u',
                          **fmtspec)
    if simd_ext in avx:
        if typ in common.iutypes:
            if simd_ext == 'sse42':
                return 'return _mm256_movemask_epi8({in0}) {test};'. \
                       format(test='== -1' if func == 'all' else '!= 0',
                              **fmtspec)
            else:
                return \
                '''nsimd_sse42_v{typ} lo = {extract_lo};
                   nsimd_sse42_v{typ} hi = {extract_hi};
                   return nsimd_{func}_sse42_{typ}(lo) {and_or}
                          nsimd_{func}_sse42_{typ}(hi);'''. \
                   format(extract_lo=extract(simd_ext, typ, LO, common.in0),
                          extract_hi=extract(simd_ext, typ, HI, common.in0),
                          func=func, and_or='&&' if func == 'all' else '||',
                          **fmtspec)
        else:
            mask = '0xFF' if typ == 'f32' else '0xF'
            return 'return _mm256_movemask{suf}({in0}) {test};'. \
                   format(test='== ' + mask if func == 'all' else '!= 0',
                          **fmtspec)
    if simd_ext in avx512:
        all_test = '== 0x' + ('F' * int((512 // int(typ[1:]) // 4)))
        return 'return {in0} {test};'. \
               format(test=all_test if func == 'all' else '!= 0', **fmtspec)

# -----------------------------------------------------------------------------
# Reinterpret (bitwise_cast)

def reinterpret1(simd_ext, from_typ, to_typ):
    if from_typ == to_typ:
        return 'return {in0};'.format(**fmtspec)
    if to_typ == 'f16':
        emulate = '''{from_typ} buf[{le}];
                     nsimd_storeu_{simd_ext}_{from_typ}(buf, {in0});
                     return nsimd_loadu_{simd_ext}_f16((f16*)buf);'''. \
                     format(**fmtspec)
        native = '''nsimd_{simd_ext}_vf16 ret;
                    ret.v0 = {pre}cvtph_ps({extract_lo});
                    ret.v1 = {pre}cvtph_ps({extract_hi});
                    return ret;'''.format(
                    extract_lo=extract(simd_ext, 'u16', LO, common.in0),
                    extract_hi=extract(simd_ext, 'u16', HI, common.in0),
                    **fmtspec)
        if simd_ext in sse:
            return \
            '''#ifdef NSIMD_FP16
                 nsimd_{simd_ext}_vf16 ret;
                 ret.v0 = _mm_cvtph_ps({in0});
                 {in0} = _mm_shuffle_epi32({in0}, 14); /* = (3 << 2) | (2 << 0) */
                 ret.v1 = _mm_cvtph_ps({in0});
                 return ret;
               #else
                 {emulate}
               #endif'''.format(emulate=emulate, **fmtspec)
        if simd_ext in avx:
            return \
            '''#ifdef NSIMD_FP16
                 {}
               #else
                 {}
               #endif'''.format(native, emulate)
        if simd_ext in avx512:
            return native
    if from_typ == 'f16':
        emulate = \
        '''u16 buf[{le}];
           nsimd_storeu_{simd_ext}_f16((f16*)buf, {in0});
           return nsimd_loadu_{simd_ext}_{to_typ}(({to_typ}*)buf);'''. \
           format(**fmtspec)
        native = 'return {};'.format(setr(simd_ext, 'u16',
                 '''{pre}cvtps_ph({in0}.v0, _MM_FROUND_TO_NEAREST_INT |
                                            _MM_FROUND_NO_EXC)'''. \
                                            format(**fmtspec),
                 '''{pre}cvtps_ph({in0}.v1, _MM_FROUND_TO_NEAREST_INT |
                                            _MM_FROUND_NO_EXC)'''. \
                                            format(**fmtspec)))
        if simd_ext in sse:
            return \
            '''#ifdef NSIMD_FP16
                 __m128i lo = _mm_cvtps_ph({in0}.v0,
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                 __m128i hi = _mm_cvtps_ph({in0}.v1,
                                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                 return _mm_castpd_si128(_mm_shuffle_pd(
                          _mm_castsi128_pd(lo), _mm_castsi128_pd(hi), 0));
               #else
                 {emulate}
               #endif'''.format(emulate=emulate, **fmtspec)
        if simd_ext in avx:
            return \
            '''#ifdef NSIMD_FP16
                 {}
               #else
                 {}
               #endif'''.format(native, emulate)
        if simd_ext in avx512:
            return native
    if from_typ in common.iutypes and to_typ in common.iutypes:
        return 'return {in0};'.format(**fmtspec)
    if to_typ in ['f32', 'f64']:
        return 'return {pre}castsi{nbits}{to_suf}({in0});'. \
               format(to_suf=suf_ep(to_typ), **fmtspec)
    if from_typ in ['f32', 'f64']:
        return 'return {pre}cast{from_suf}_si{nbits}({in0});'. \
               format(from_suf=suf_ep(from_typ)[1:], **fmtspec)

# -----------------------------------------------------------------------------
# Reinterpretl, i.e. reinterpret on logicals

def reinterpretl1(simd_ext, from_typ, to_typ):
    if from_typ == to_typ:
        return 'return {in0};'.format(**fmtspec)
    if to_typ == 'f16':
        if simd_ext in sse:
            return \
            '''nsimd_{simd_ext}_vlf16 ret;
               ret.v0 = _mm_castsi128_ps(_mm_unpacklo_epi16({in0}, {in0}));
               ret.v1 = _mm_castsi128_ps(_mm_unpackhi_epi16({in0}, {in0}));
               return ret;'''.format(**fmtspec)
        if simd_ext == 'avx':
            return \
            '''nsimd_{simd_ext}_vlf16 ret;
               nsimd_sse42_vlf16 tmp1 =
                   nsimd_reinterpretl_sse42_f16_{from_typ}(
                     _mm256_castsi256_si128({in0}));
               nsimd_sse42_vlf16 tmp2 =
                   nsimd_reinterpretl_sse42_f16_{from_typ}(
                      _mm256_extractf128_si256({in0}, 1));
               ret.v0 = {setr_tmp1};
               ret.v1 = {setr_tmp2};
               return ret;'''. \
               format(setr_tmp1=setr('avx', 'f32', 'tmp1.v0', 'tmp1.v1'),
                      setr_tmp2=setr('avx', 'f32', 'tmp2.v0', 'tmp2.v1'),
                      **fmtspec)
        if simd_ext == 'avx2':
            return \
            '''nsimd_{simd_ext}_vlf16 ret;
               ret.v0 = _mm256_castsi256_ps(_mm256_cvtepi16_epi32(
                          _mm256_castsi256_si128({in0})));
               ret.v1 = _mm256_castsi256_ps(_mm256_cvtepi16_epi32(
                          _mm256_extractf128_si256({in0}, 1)));
               return ret;'''.format(**fmtspec)
        if simd_ext in avx512:
            return '''nsimd_{simd_ext}_vlf16 ret;
                      ret.v0 = (__mmask16)({in0} & 0xFFFF);
                      ret.v1 = (__mmask16)(({in0} >> 16) & 0xFFFF);
                      return ret;'''.format(**fmtspec)
    if from_typ == 'f16':
        if simd_ext in sse + avx:
            return '''f32 in[{le}];
                      {to_typ} out[{le}];
                      int i;
                      nsimd_storeu_{simd_ext}_f32(in, {in0}.v0);
                      nsimd_storeu_{simd_ext}_f32(in + {leo2}, {in0}.v1);
                      for (i = 0; i < {le}; i++) {{
                        out[i] = ({to_typ})(in[i] != 0.0f ? -1 : 0);
                      }}
                      return nsimd_loadu_{simd_ext}_{to_typ}(out);'''. \
                      format(leo2=int(fmtspec['le']) // 2, **fmtspec)
        if simd_ext in avx512:
            return \
            'return (__mmask32){in0}.v0 | ((__mmask32){in0}.v1 << 16);'. \
            format(**fmtspec)
    if simd_ext in sse + avx:
        return reinterpret1(simd_ext, from_typ, to_typ)
    else:
        return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Convert

def convert1(simd_ext, from_typ, to_typ):
    if to_typ == from_typ or \
       to_typ in common.iutypes and from_typ in common.iutypes:
        return 'return {in0};'.format(**fmtspec)
    if to_typ == 'f16':
        if simd_ext in sse:
            getlo = '{in0}'.format(**fmtspec)
            gethi = '_mm_unpackhi_epi64({in0}, {in0})'.format(**fmtspec)
        if simd_ext in avx:
            getlo = '_mm256_castsi256_si128({in0})'.format(**fmtspec)
            gethi = '_mm256_extractf128_si256({in0}, 1)'.format(**fmtspec)
        if simd_ext in avx512:
            getlo = '_mm512_castsi512_si256({in0})'.format(**fmtspec)
            gethi = '_mm512_extracti64x4_epi64({in0}, 1)'.format(**fmtspec)
        through_epi32 = \
        '''nsimd_{simd_ext}_v{to_typ} ret;
           ret.v0 = {pre}cvtepi32_ps({pre}cvtep{from_typ}_epi32({getlo}));
           ret.v1 = {pre}cvtepi32_ps({pre}cvtep{from_typ}_epi32({gethi}));
           return ret;'''.format(getlo=getlo, gethi=gethi, **fmtspec)
        emulate = '''{from_typ} in[{le}];
                     f32 out[{leo2}];
                     nsimd_{simd_ext}_vf16 ret;
                     int i;
                     nsimd_storeu_{simd_ext}_{from_typ}(in, {in0});
                     for (i = 0; i < {leo2}; i++) {{
                       out[i] = (f32)in[i];
                     }}
                     ret.v0 = nsimd_loadu_{simd_ext}_f32(out);
                     for (i = 0; i < {leo2}; i++) {{
                       out[i] = (f32)in[i + {leo2}];
                     }}
                     ret.v1 = nsimd_loadu_{simd_ext}_f32(out);
                     return ret;'''.format(leo2=int(fmtspec['le']) // 2,
                                           **fmtspec)
        if simd_ext in ['sse42', 'avx2']:
            return through_epi32
        if simd_ext in ['sse2', 'avx']:
            return emulate
        if simd_ext in avx512:
            return through_epi32
    if from_typ == 'f16':
        return '''f32 in[{leo2}];
                  {to_typ} out[{le}];
                  int i;
                  nsimd_storeu_{simd_ext}_f32(in, {in0}.v0);
                  for (i = 0; i < {leo2}; i++) {{
                    out[i] = ({to_typ})in[i];
                  }}
                  nsimd_storeu_{simd_ext}_f32(in, {in0}.v1);
                  for (i = 0; i < {leo2}; i++) {{
                    out[i + {leo2}] = ({to_typ})in[i];
                  }}
                  return nsimd_loadu_{simd_ext}_{to_typ}(out);'''. \
                  format(leo2=int(fmtspec['le']) // 2, **fmtspec)
    emulate = '''{from_typ} in[{le}];
                 {to_typ} out[{le}];
                 int i;
                 nsimd_storeu_{simd_ext}_{from_typ}(in, {in0});
                 for (i = 0; i < {le}; i++) {{
                   out[i] = ({to_typ})in[i];
                 }}
                 return nsimd_loadu_{simd_ext}_{to_typ}(out);'''. \
                 format(**fmtspec)
    if to_typ == 'f64' or from_typ == 'f64':
        if simd_ext == 'avx512_skylake':
            return 'return _mm512_cvt{from_suf}{to_suf}({in0});'. \
                   format(from_suf=suf_ep(from_typ)[1:], to_suf=suf_ep(to_typ),
                          **fmtspec)
        else:
            return emulate
    if to_typ == 'f32' and from_typ == 'i32':
        return 'return {pre}cvtepi32_ps({in0});'.format(**fmtspec)
    if to_typ == 'f32' and from_typ == 'u32':
        if simd_ext in sse + avx:
            return emulate
        if simd_ext in avx512:
            return 'return _mm512_cvtepu32_ps({in0});'.format(**fmtspec)
    if to_typ == 'i32' and from_typ == 'f32':
        return 'return {pre}cvtps_epi32({in0});'.format(**fmtspec)
    if to_typ == 'u32' and from_typ == 'f32':
        if simd_ext in sse + avx:
            return emulate
        if simd_ext in avx512:
            return 'return _mm512_cvtps_epu32({in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
## Reciprocal (at least 11 bits of precision)

def rec11_rsqrt11(func, simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_{func}11_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_{func}11_{simd_ext}_f32({in0}.v1);
                  return ret;'''. \
                  format(func='rec' if func == 'rcp' else 'rsqrt', **fmtspec)
    if typ == 'f32':
        if simd_ext in sse + avx:
            return 'return {pre}{func}_ps({in0});'.format(func=func, **fmtspec)
        if simd_ext in avx512:
            return 'return _mm512_{func}14_ps({in0});'. \
                   format(func=func, **fmtspec)
    if typ == 'f64':
        if simd_ext in sse + avx:
            return \
            'return {pre}cvtps_pd(_mm_{func}_ps({pre}cvtpd_ps({in0})));'. \
            format(func=func, **fmtspec)
        if simd_ext in avx512:
            return 'return _mm512_{func}14_pd({in0});'. \
                   format(func=func, **fmtspec)

# -----------------------------------------------------------------------------
## Reciprocal (IEEE)

def rec1(simd_ext, typ):
    one = '{pre}set1_ps(1.0f)'.format(**fmtspec) if typ in ['f16', 'f32'] \
          else '{pre}set1_pd(1.0)'.format(**fmtspec)
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  nsimd_{simd_ext}_vf32 one = {one};
                  ret.v0 = {pre}div_ps(one, {in0}.v0);
                  ret.v1 = {pre}div_ps(one, {in0}.v1);
                  return ret;'''.format(one=one, **fmtspec)
    return 'return {pre}div{suf}({one}, {in0});'.format(one=one, **fmtspec)

# -----------------------------------------------------------------------------
## Negative

def neg1(simd_ext, typ):
    cte = '0x80000000' if typ in ['f16', 'f32'] else '0x8000000000000000'
    fsuf = '_ps' if typ in ['f16', 'f32'] else '_pd'
    utyp = 'u32' if typ in ['f16', 'f32'] else 'u64'
    vmask = '{pre}castsi{nbits}{fsuf}(nsimd_set1_{simd_ext}_{utyp}({cte}))'. \
            format(cte=cte, utyp=utyp, fsuf=fsuf, **fmtspec)
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  nsimd_{simd_ext}_vf32 mask = {vmask};
                  ret.v0 = nsimd_xorb_{simd_ext}_f32(mask, {in0}.v0);
                  ret.v1 = nsimd_xorb_{simd_ext}_f32(mask, {in0}.v1);
                  return ret;'''.format(vmask=vmask, **fmtspec)
    if typ in ['f32', 'f64']:
        return 'return nsimd_xorb_{simd_ext}_{typ}({vmask}, {in0});'. \
               format(vmask=vmask, **fmtspec)
    return '''return nsimd_sub_{simd_ext}_{typ}(
                  {pre}setzero_si{nbits}(), {in0});'''. \
              format(**fmtspec)

# -----------------------------------------------------------------------------
## nbtrue

def nbtrue1(simd_ext, typ):
    if typ == 'f16':
        return '''return nsimd_nbtrue_{simd_ext}_f32({in0}.v0) +
                         nsimd_nbtrue_{simd_ext}_f32({in0}.v1);'''. \
                         format(**fmtspec)
    if typ in ['i8', 'u8']:
        code = 'return nsimd_popcnt32_((u32){pre}movemask_epi8({in0}));'. \
               format(**fmtspec)
    elif typ in ['i16', 'u16']:
        code = 'return nsimd_popcnt32_((u32){pre}movemask_epi8({in0})) >> 1;'. \
               format(**fmtspec)
    elif typ in ['i32', 'u32', 'i64', 'u64']:
        code = '''return nsimd_popcnt32_((u32){pre}movemask{fsuf}(
                      {pre}castsi{nbits}{fsuf}({in0})));'''. \
                      format(fsuf='_ps' if typ in ['i32', 'u32'] else '_pd',
                             **fmtspec)
    else:
        code = 'return nsimd_popcnt32_((u32){pre}movemask{suf}({in0}));'. \
               format(**fmtspec)
    if simd_ext in sse:
        return code
    if simd_ext in avx:
        if typ in ['i32', 'u32', 'i64', 'u64', 'f32', 'f64']:
            return code
        else:
            if simd_ext == 'avx2':
                return code
            else:
                return \
                '''return nsimd_nbtrue_sse42_{typ}(
                            _mm256_castsi256_si128({in0})) +
                              nsimd_nbtrue_sse42_{typ}(
                                _mm256_extractf128_si256({in0}, 1));'''. \
                                format(**fmtspec)
    if simd_ext in avx512:
        return 'return nsimd_popcnt64_((u64){in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
## reverse

def reverse1(simd_ext, typ):
    ## 8-bit int
    if typ in ['i8', 'u8']:
        if simd_ext == 'sse2':
            return '''{in0} = _mm_shufflehi_epi16({in0}, _MM_SHUFFLE(0,1,2,3));
                      {in0} = _mm_shufflelo_epi16({in0}, _MM_SHUFFLE(0,1,2,3));
                      {in0} = _mm_castpd_si128(_mm_shuffle_pd(
                                _mm_castsi128_pd({in0}), _mm_castsi128_pd(
                                  {in0}), 1));
                      nsimd_{simd_ext}_v{typ} r0 = _mm_srli_epi16({in0}, 8);
                      nsimd_{simd_ext}_v{typ} r1 = _mm_slli_epi16({in0}, 8);
                      return _mm_or_si128(r0, r1);'''.format(**fmtspec)
        elif simd_ext == 'sse42':
            return '''nsimd_{simd_ext}_v{typ} mask = _mm_set_epi8(
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                      return _mm_shuffle_epi8({in0}, mask);'''. \
                      format(**fmtspec)
        elif simd_ext == 'avx':
            return \
            '''nsimd_sse42_v{typ} r0 = _mm_shuffle_epi8(
                 _mm256_extractf128_si256({in0}, 0), _mm_set_epi8(
                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
               nsimd_sse42_v{typ} r1 = _mm_shuffle_epi8(
                 _mm256_extractf128_si256({in0}, 1), _mm_set_epi8(
                   0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
               {in0} = _mm256_insertf128_si256({in0}, r0, 1);
               return _mm256_insertf128_si256({in0}, r1, 0);'''. \
               format(**fmtspec)
        elif simd_ext == 'avx2':
             return \
             '''{in0} = _mm256_shuffle_epi8({in0}, _mm256_set_epi8(
                   0,  1,  2,  3,  4,  5,  6,  7,
                   8,  9, 10, 11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20, 21, 22, 23,
                  24, 25, 26, 27, 28, 29, 30, 31));
                return _mm256_permute2x128_si256({in0}, {in0}, 1);'''. \
                format(**fmtspec)
        ## AVX-512F and above.
        else:
             return \
             '''nsimd_avx2_v{typ} r0 = _mm512_extracti64x4_epi64({in0}, 0);
                nsimd_avx2_v{typ} r1 = _mm512_extracti64x4_epi64({in0}, 1);
                r0 = _mm256_shuffle_epi8(r0, _mm256_set_epi8(
                     0,  1,  2,  3,  4,  5,  6,  7,
                     8,  9, 10, 11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20, 21, 22, 23,
                    24, 25, 26, 27, 28, 29, 30, 31));
                r1 = _mm256_shuffle_epi8(r1, _mm256_set_epi8(
                      0,  1,  2,  3,  4,  5,  6,  7,
                      8,  9, 10, 11, 12, 13, 14, 15,
                     16, 17, 18, 19, 20, 21, 22, 23,
                     24, 25, 26, 27, 28, 29, 30, 31));
                r0 = _mm256_permute2x128_si256(r0, r0, 1);
                r1 = _mm256_permute2x128_si256(r1, r1, 1);
                {in0} = _mm512_insertf64x4({in0}, r0, 1);
                return _mm512_insertf64x4({in0}, r1, 0);'''.format(**fmtspec)
    ## 16-bit int
    elif typ in ['i16', 'u16']:
        if simd_ext == 'sse2':
            return '''{in0} = _mm_shufflehi_epi16( {in0}, _MM_SHUFFLE(0,1,2,3) );
                      {in0} = _mm_shufflelo_epi16( {in0}, _MM_SHUFFLE(0,1,2,3) );
                      return _mm_castpd_si128(_mm_shuffle_pd(
                               _mm_castsi128_pd({in0}),
                               _mm_castsi128_pd({in0}), 1));'''. \
                               format(**fmtspec)
        elif simd_ext == 'sse42':
            return \
            '''return _mm_shuffle_epi8({in0}, _mm_set_epi8(
                        1,  0,  3,  2,  5,  4,  7, 6,
                        9,  8, 11, 10, 13, 12, 15, 14));'''.format(**fmtspec)
        elif simd_ext == 'avx':
            return \
            '''nsimd_sse42_v{typ} r0 = _mm_shuffle_epi8(
                 _mm256_extractf128_si256({in0}, 0), _mm_set_epi8(
                   1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
               nsimd_sse42_v{typ} r1 = _mm_shuffle_epi8(
                 _mm256_extractf128_si256({in0}, 1), _mm_set_epi8(
                   1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
               {in0} = _mm256_insertf128_si256( {in0}, r0, 1);
               return _mm256_insertf128_si256({in0}, r1, 0);'''. \
               format(**fmtspec)
        elif simd_ext == 'avx2':
            return \
            '''{in0} = _mm256_shuffle_epi8({in0}, _mm256_set_epi8(
                           1,  0,  3,  2,  5,  4,  7,  6,
                           9,  8, 11, 10, 13, 12, 15, 14,
                          17, 16, 19, 18, 21, 20, 23, 22,
                          25, 24, 27, 26, 29, 28, 31, 30));
               return _mm256_permute2x128_si256({in0}, {in0}, 1);'''. \
               format(**fmtspec)
        ## AVX-512F
        elif simd_ext == 'avx512_knl':
            return \
            '''{in0} = _mm512_permutexvar_epi32(_mm512_set_epi32(
                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                 {in0});
               nsimd_{simd_ext}_v{typ} r0 = _mm512_srli_epi32({in0}, 16);
               nsimd_{simd_ext}_v{typ} r1 = _mm512_slli_epi32({in0}, 16);
               return _mm512_or_si512(r0, r1);'''.format(**fmtspec)
        ## AVX-512F+BW (Skylake) + WORKAROUND GCC<=8
        else:
            return \
            '''return _mm512_permutexvar_epi16(_mm512_set_epi32(
                 (0<<16)  | 1,  (2<<16)  | 3,  (4<<16)  | 5,  (6<<16)  | 7,
                 (8<<16)  | 9,  (10<<16) | 11, (12<<16) | 13, (14<<16) | 15,
                 (16<<16) | 17, (18<<16) | 19, (20<<16) | 21, (22<<16) | 23,
                 (24<<16) | 25, (26<<16) | 27, (28<<16) | 29, (30<<16) | 31),
                 {in0} );'''.format(**fmtspec)
    ## 32-bit int
    elif typ in ['i32', 'u32']:
        if simd_ext in ['sse2', 'sse42']:
            return 'return _mm_shuffle_epi32({in0}, _MM_SHUFFLE(0,1,2,3));'. \
                   format(**fmtspec)
        elif simd_ext == 'avx':
            return '''{in0} = _mm256_castps_si256(_mm256_shuffle_ps(
                                _mm256_castsi256_ps({in0}),
                                _mm256_castsi256_ps({in0}),
                                _MM_SHUFFLE(0,1,2,3)));
                      return _mm256_permute2f128_si256({in0}, {in0}, 1);'''. \
                      format(**fmtspec)
        elif simd_ext == 'avx2':
            return \
            '''{in0} = _mm256_shuffle_epi32({in0}, _MM_SHUFFLE(0,1,2,3));
               return _mm256_permute2x128_si256({in0}, {in0}, 1);'''. \
               format(**fmtspec)
        else:
            return \
            '''return _mm512_permutexvar_epi32(_mm512_set_epi32(
                 0, 1,  2,  3,  4,  5,  6,  7,
                 8, 9, 10, 11, 12, 13, 14, 15), {in0});'''. \
                 format(**fmtspec)
    elif typ in ['i64', 'u64']:
        if simd_ext in ['sse2', 'sse42']:
            return '''return _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(
                               {in0}), _mm_castsi128_pd({in0}), 1));'''. \
                               format(**fmtspec)
        elif simd_ext == 'avx':
            return '''{in0} = _mm256_castpd_si256(
                                  _mm256_shuffle_pd(
                                     _mm256_castsi256_pd({in0}),
                                     _mm256_castsi256_pd({in0}),
                                     (1<<2) | 1
                                  )
                              );
                       return _mm256_permute2f128_si256({in0}, {in0}, 1);'''. \
                       format(**fmtspec)
        elif simd_ext == 'avx2':
           return '''return _mm256_permute4x64_epi64({in0},
                              _MM_SHUFFLE(0, 1, 2, 3));'''.format(**fmtspec)
        else:
           return '''return _mm512_permutexvar_epi64(_mm512_set_epi64(
                              0, 1, 2, 3, 4, 5, 6, 7), {in0});'''. \
                              format(**fmtspec)
    ## 16-bit float
    elif typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_reverse_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_reverse_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(**fmtspec)
    ## 32-bit float
    elif typ == 'f32':
        if simd_ext in ['sse2', 'sse42']:
            return '''return _mm_shuffle_ps({in0}, {in0},
                               _MM_SHUFFLE(0, 1, 2, 3));'''.format(**fmtspec)
        elif simd_ext in ['avx', 'avx2']:
            return '''{in0} = _mm256_shuffle_ps({in0}, {in0},
                                _MM_SHUFFLE(0, 1, 2, 3));
                      return _mm256_permute2f128_ps({in0}, {in0}, 1);'''. \
                      format(**fmtspec)
        else:
            return \
            '''return _mm512_permutexvar_ps(_mm512_set_epi32(
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                        {in0} );'''.format(**fmtspec)
    ## 64-bit float
    else:
        if simd_ext in ['sse2', 'sse42']:
            return 'return _mm_shuffle_pd({in0}, {in0}, 1);'.format(**fmtspec)
        elif simd_ext == 'avx':
            return '''{in0} = _mm256_shuffle_pd({in0}, {in0}, (1<<2) | 1);
                      return _mm256_permute2f128_pd({in0}, {in0}, 1);'''. \
                      format(**fmtspec)
        elif simd_ext == 'avx2':
            return '''return _mm256_permute4x64_pd({in0},
                               _MM_SHUFFLE(0, 1, 2, 3));'''.format(**fmtspec)
        else:
            return '''return _mm512_permute_mm512_set_epi64(
                               0, 1, 2, 3, 4, 5, 6, 7), {in0});'''. \
                               format(**fmtspec)

# -----------------------------------------------------------------------------
## addv

def addv(simd_ext, typ):
    if simd_ext in sse:
        if typ == 'f64':
            return \
            '''return _mm_cvtsd_f64(_mm_add_pd({in0},
                                    _mm_shuffle_pd({in0}, {in0}, 0x01)));'''. \
                                    format(**fmtspec)
        elif typ == 'f32':
            return \
            '''nsimd_{simd_ext}_vf32 tmp = _mm_add_ps({in0}, _mm_shuffle_ps(
                                             {in0}, {in0}, 0xb1));
               return _mm_cvtss_f32(_mm_add_ps(tmp, _mm_shuffle_ps(
                        tmp, tmp, 0x4e)));''' .format(**fmtspec)
        elif typ == 'f16':
            return \
            '''nsimd_{simd_ext}_vf32 tmp0 = _mm_add_ps({in0}.v0,
                 _mm_shuffle_ps({in0}.v0, {in0}.v0, 0xb1));
               nsimd_{simd_ext}_vf32 tmp1 = _mm_add_ps({in0}.v1,
                 _mm_shuffle_ps({in0}.v1, {in0}.v1, 0xb1));
               return nsimd_f32_to_f16(_mm_cvtss_f32(_mm_add_ps(
                 tmp0, _mm_shuffle_ps(tmp0, tmp0, 0x4e))) +
                   _mm_cvtss_f32(_mm_add_ps(tmp1, _mm_shuffle_ps(
                     tmp1, tmp1, 0x4e))));''' .format(**fmtspec)
    elif simd_ext in avx:
        if typ == 'f64':
            return \
            '''__m128d tmp = _mm_add_pd(_mm256_extractf128_pd({in0}, 1),
                                        _mm256_extractf128_pd({in0}, 0));
               return _mm_cvtsd_f64(_mm_add_pd(tmp, _mm_shuffle_pd(
                        tmp, tmp, 0x01)));''' .format(**fmtspec)
        elif typ == 'f32':
            return \
            '''__m128 tmp0 = _mm_add_ps(_mm256_extractf128_ps({in0}, 1),
                                        _mm256_extractf128_ps({in0}, 0));
               __m128 tmp1 = _mm_add_ps(tmp0, _mm_shuffle_ps(tmp0, tmp0, 0xb1));
               return _mm_cvtss_f32(_mm_add_ps(tmp1, _mm_shuffle_ps(
                        tmp1, tmp1, 0x4e)));''' .format(**fmtspec)
        elif typ == 'f16':
            return \
            '''__m128 tmp00 = _mm_add_ps(_mm256_extractf128_ps({in0}.v0, 1),
                                         _mm256_extractf128_ps({in0}.v0, 0));
               __m128 tmp01 = _mm_add_ps(tmp00, _mm_shuffle_ps(
                                tmp00, tmp00, 0xb1));
               __m128 tmp10 = _mm_add_ps(_mm256_extractf128_ps({in0}.v1, 1),
                                         _mm256_extractf128_ps({in0}.v1, 0));
               __m128 tmp11 = _mm_add_ps(tmp10, _mm_shuffle_ps(
                                tmp10, tmp10, 0xb1));
               return nsimd_f32_to_f16(_mm_cvtss_f32(_mm_add_ps(
                        tmp01, _mm_shuffle_ps(tmp01, tmp01, 0x4e))) +
                          _mm_cvtss_f32(_mm_add_ps(tmp11, _mm_shuffle_ps(
                            tmp11, tmp11, 0x4e))));
                    ''' .format(**fmtspec)
    elif simd_ext in avx512:
        if typ == 'f64':
            return \
            '''__m256d tmp0 = _mm256_add_pd(_mm512_extractf64x4_pd({in0}, 0),
                                            _mm512_extractf64x4_pd({in0}, 1));
               __m128d tmp1 = _mm_add_pd(_mm256_extractf128_pd(tmp0, 1),
                                         _mm256_extractf128_pd(tmp0, 0));
               return _mm_cvtsd_f64(_mm_add_pd(tmp1, _mm_shuffle_pd(
                        tmp1, tmp1, 0x01)));''' .format(**fmtspec)
        elif typ == 'f32':
            return \
            '''__m128 tmp0 = _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(
                               {in0}, 0), _mm512_extractf32x4_ps({in0}, 1)),
                               _mm_add_ps(_mm512_extractf32x4_ps({in0}, 2),
                               _mm512_extractf32x4_ps({in0}, 3)));
               __m128 tmp1 = _mm_add_ps(tmp0, _mm_shuffle_ps(
                               tmp0, tmp0, 0xb1));
               return _mm_cvtss_f32(_mm_add_ps(tmp1, _mm_shuffle_ps(
                        tmp1, tmp1, 0x4e)));''' .format(**fmtspec)
        elif typ == 'f16':
            return \
            '''f32 res;
               __m128 tmp0 = _mm_add_ps(
                   _mm_add_ps(_mm512_extractf32x4_ps({in0}.v0, 0),
                               _mm512_extractf32x4_ps({in0}.v0, 1)),
                   _mm_add_ps(_mm512_extractf32x4_ps({in0}.v0, 2),
                               _mm512_extractf32x4_ps({in0}.v0, 3)));
               __m128 tmp1 = _mm_add_ps(tmp0, _mm_shuffle_ps(
                               tmp0, tmp0, 0xb1));
               res = _mm_cvtss_f32(_mm_add_ps(tmp1, _mm_shuffle_ps(
                       tmp1, tmp1, 0x4e)));
               tmp0 = _mm_add_ps(
                   _mm_add_ps(_mm512_extractf32x4_ps({in0}.v1, 0),
                               _mm512_extractf32x4_ps({in0}.v1, 1)),
                   _mm_add_ps(_mm512_extractf32x4_ps({in0}.v1, 2),
                               _mm512_extractf32x4_ps({in0}.v1, 3)));
               tmp1 = _mm_add_ps(tmp0, _mm_shuffle_ps(tmp0, tmp0, 0xb1));
               return nsimd_f32_to_f16(res + _mm_cvtss_f32(_mm_add_ps(
                        tmp1, _mm_shuffle_ps(tmp1, tmp1, 0x4e))));''' . \
                        format(**fmtspec)

# -----------------------------------------------------------------------------
## upconvert

def upcvt1(simd_ext, from_typ, to_typ):
    # From f16 is easy
    if from_typ == 'f16':
        if to_typ == 'f32':
            return \
            '''nsimd_{simd_ext}_vf32x2 ret;
               ret.v0 = {in0}.v0;
               ret.v1 = {in0}.v1;
               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_{simd_ext}_v{to_typ}x2 ret;
               ret.v0 = nsimd_cvt_{simd_ext}_{to_typ}_f32({in0}.v0);
               ret.v1 = nsimd_cvt_{simd_ext}_{to_typ}_f32({in0}.v1);
               return ret;'''.format(**fmtspec)

    # To f16 is easy
    if to_typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vf16x2 ret;
           nsimd_{simd_ext}_v{iu}16x2 buf;
           buf = nsimd_upcvt_{simd_ext}_{iu}16_{iu}8({in0});
           ret.v0 = nsimd_cvt_{simd_ext}_f16_{iu}16(buf.v0);
           ret.v1 = nsimd_cvt_{simd_ext}_f16_{iu}16(buf.v1);
           return ret;'''.format(iu=from_typ[0], **fmtspec)

    # For integer upcast, due to 2's complement representation
    # epi_epi : signed   -> bigger signed
    # epi_epi : signed   -> bigger unsigned
    # epu_epi : unsigned -> bigger signed
    # epu_epi : unsigned -> bigger unsigned
    if from_typ in common.iutypes:
        suf_epep = 'ep{ui}{typnbits}_epi{typnbits2}'. \
                   format(ui='u' if from_typ in common.utypes else 'i',
                          typnbits2=str(int(fmtspec['typnbits']) * 2),
                          **fmtspec)
    else:
        suf_epep = 'ps_pd'

    # compute lower half
    if simd_ext in sse:
        lower_half = '{in0}'.format(**fmtspec)
    else:
        lower_half = extract(simd_ext, from_typ, LO, fmtspec['in0'])

    # compute upper half
    if simd_ext in sse:
        if from_typ in common.iutypes:
            upper_half = \
            '''{pre}castpd_si{nbits}({pre}shuffle_pd(
                   {pre}castsi{nbits}_pd({in0}),
                   {pre}castsi{nbits}_pd({in0}), 1))'''. \
                   format(**fmtspec)
        else:
            upper_half = '''{pre}castpd_ps({pre}shuffle_pd(
                                {pre}castps_pd({in0}),
                                {pre}castps_pd({in0}), 1))'''.format(**fmtspec)
    else:
        upper_half = extract(simd_ext, from_typ, HI, fmtspec['in0'])

    # When intrinsics are provided
    # for conversions integers <-> floating point, there is no intrinsics, so
    # we use cvt's
    if from_typ == 'i32' and to_typ == 'f64':
        with_intrinsic = \
        '''nsimd_{simd_ext}_vf64x2 ret;
           ret.v0 = {pre}cvtepi32_pd({lower_half});
           ret.v1 = {pre}cvtepi32_pd({upper_half});
           return ret;'''.format(upper_half=upper_half,
                                 lower_half=lower_half, **fmtspec)
    elif (from_typ in common.iutypes and to_typ in common.iutypes) or \
         (from_typ == 'f32' and to_typ == 'f64'):
        with_intrinsic = \
        '''nsimd_{simd_ext}_v{to_typ}x2 ret;
           ret.v0 = {pre}cvt{suf_epep}({lower_half});
           ret.v1 = {pre}cvt{suf_epep}({upper_half});
           return ret;'''.format(upper_half=upper_half, lower_half=lower_half,
                                 suf_epep=suf_epep, **fmtspec)
    else:
        from_typ2 = from_typ[0] + str(int(fmtspec['typnbits']) * 2)
        if from_typ not in common.iutypes:
            # getting here means that from_typ=f32 and to_typ=f64
            with_intrinsic = \
            '''nsimd_{simd_ext}_vf64x2 ret;
               ret.v0 = nsimd_cvt_{simd_ext}_{to_typ}_f64({pre}cvtps_pd(
                            {lower_half}));
               ret.v1 = nsimd_cvt_{simd_ext}_{to_typ}_f64({pre}cvtps_pd(
                            {upper_half}));
               return ret;'''. \
               format(upper_half=upper_half, lower_half=lower_half,
                      from_typ2=from_typ2, suf_epep=suf_epep, **fmtspec)

    # When no intrinsic is given for going from integers to floating or
    # from floating to integer we can go through a cvt
    if to_typ in common.ftypes:
        int_float = \
        '''nsimd_{simd_ext}_v{to_typ}x2 ret;
           nsimd_{simd_ext}_v{int_typ}x2 tmp;
           tmp = nsimd_upcvt_{simd_ext}_{int_typ}_{from_typ}({in0});
           ret.v0 = nsimd_cvt_{simd_ext}_{to_typ}_{int_typ}(tmp.v0);
           ret.v1 = nsimd_cvt_{simd_ext}_{to_typ}_{int_typ}(tmp.v1);
           return ret;'''. \
           format(int_typ=from_typ[0] + to_typ[1:], lower_half=lower_half,
                  upper_half=upper_half, **fmtspec)
    else:
        int_float = \
        '''return nsimd_upcvt_{simd_ext}_{to_typ}_{int_typ}(
                      nsimd_cvt_{simd_ext}_{int_typ}_{from_typ}({in0}));'''. \
                      format(int_typ=to_typ[0] + from_typ[1:],
                             lower_half=lower_half, upper_half=upper_half,
                             **fmtspec)

    # When no intrinsic is given we can use the trick of falling back to
    # the lower SIMD extension
    split_trick = \
    '''nsimd_{simd_ext}_v{to_typ}x2 ret;
       nsimd_{simd_ext2}_v{to_typ}x2 ret2;
       ret2 = nsimd_upcvt_{simd_ext2}_{to_typ}_{from_typ}({lo});
       ret.v0 = {merge};
       ret2 = nsimd_upcvt_{simd_ext2}_{to_typ}_{from_typ}({hi});
       ret.v1 = {merge};
       return ret;'''. \
       format(simd_ext2='sse42' if simd_ext == 'avx' else 'avx2',
              lo=extract(simd_ext, from_typ, LO, common.in0),
              hi=extract(simd_ext, from_typ, HI, common.in0),
              merge=setr(simd_ext, to_typ, 'ret2.v0', 'ret2.v1'), **fmtspec)

    # return C code
    if from_typ == 'i32' and to_typ == 'f64':
        return with_intrinsic
    if (from_typ in common.ftypes and to_typ in common.iutypes) or \
       (from_typ in common.iutypes and to_typ in common.ftypes):
        return int_float
    if simd_ext == 'sse2':
        if from_typ in common.itypes and to_typ in common.iutypes:
            return \
            '''nsimd_{simd_ext}_v{to_typ}x2 ret;
               __m128i mask = _mm_cmpgt{suf}(_mm_setzero_si128(), {in0});
               ret.v0 = _mm_unpacklo{suf}({in0}, mask);
               ret.v1 = _mm_unpackhi{suf}({in0}, mask);
               return ret;'''.format(**fmtspec)
        elif from_typ in common.utypes and to_typ in common.iutypes:
            return \
            '''nsimd_{simd_ext}_v{to_typ}x2 ret;
               ret.v0 = _mm_unpacklo{suf}({in0}, _mm_setzero_si128());
               ret.v1 = _mm_unpackhi{suf}({in0}, _mm_setzero_si128());
               return ret;'''.format(**fmtspec)
        else:
            return with_intrinsic
    elif simd_ext == 'sse42':
        return with_intrinsic
    elif simd_ext == 'avx':
        if from_typ == 'i32' and to_typ == 'f64':
            return with_intrinsic
        else:
            return split_trick
    elif simd_ext == 'avx2':
        return with_intrinsic
    elif simd_ext == 'avx512_knl':
        if from_typ in ['i16', 'u16', 'i32', 'u32', 'f32']:
            return with_intrinsic
        else:
            return split_trick
    else:
        return with_intrinsic

# -----------------------------------------------------------------------------
## downconvert

def downcvt1(simd_ext, from_typ, to_typ):
    # From f16 is easy
    if from_typ == 'f16':
        le_to_typ = int(fmtspec['le']) * 2
        le_1f32 = le_to_typ // 4
        le_2f32 = 2 * le_to_typ // 4
        le_3f32 = 3 * le_to_typ // 4
        return \
        '''{to_typ} dst[{le_to_typ}];
           f32 src[{le_to_typ}];
           int i;
           {pre}storeu_ps(src, {in0}.v0);
           {pre}storeu_ps(src + {le_1f32}, {in0}.v1);
           {pre}storeu_ps(src + {le_2f32}, {in1}.v0);
           {pre}storeu_ps(src + {le_3f32}, {in1}.v1);
           for (i = 0; i < {le_to_typ}; i++) {{
             dst[i] = ({to_typ})src[i];
           }}
           return {pre}loadu_si{nbits}(({nat_typ}*)dst);'''. \
           format(le_to_typ=le_to_typ, le_1f32=le_1f32, le_2f32=le_2f32,
                  le_3f32=le_3f32, nat_typ=get_type(simd_ext, to_typ),
                  **fmtspec)

    # To f16 is easy
    if to_typ == 'f16':
        if from_typ == 'f32':
            return \
            '''nsimd_{simd_ext}_vf16 ret;
               ret.v0 = {in0};
               ret.v1 = {in1};
               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_{simd_ext}_vf16 ret;
               ret.v0 = nsimd_cvt_{simd_ext}_f32_{from_typ}({in0});
               ret.v1 = nsimd_cvt_{simd_ext}_f32_{from_typ}({in1});
               return ret;'''.format(**fmtspec)

    # f64 --> f32 have intrinsics
    if from_typ == 'f64' and to_typ == 'f32':
        if simd_ext in sse:
            return '''return _mm_movelh_ps(_mm_cvtpd_ps({in0}),
                                           _mm_cvtpd_ps({in1}));'''. \
                                           format(**fmtspec)
        else:
            return 'return {};'.format(setr(simd_ext, 'f32',
                                '{pre}cvtpd_ps({in0})'.format(**fmtspec),
                                '{pre}cvtpd_ps({in1})'.format(**fmtspec)))

    # integer conversions intrinsics are only available with AVX-512
    if simd_ext in avx512:
        if (from_typ in ['i32', 'i64'] and to_typ in common.itypes) or \
           (simd_ext == 'avx512_skylake' and from_typ == 'i16' and \
            to_typ == 'i8'):
            return 'return {};'.format(setr(simd_ext, to_typ,
                   '{pre}cvtep{from_typ}_ep{to_typ}({in0})'.format(**fmtspec),
                   '{pre}cvtep{from_typ}_ep{to_typ}({in1})'.format(**fmtspec)))
        elif from_typ == 'i64' and to_typ == 'f32':
            return 'return nsimd_cvt_{simd_ext}_f32_i32({});'. \
                   format(setr(simd_ext, from_typ,
                          '{pre}cvtepi64_epi32({in0})'.format(**fmtspec),
                          '{pre}cvtepi64_epi32({in1})'.format(**fmtspec)),
                          **fmtspec)

    # and then emulation
    le_to_typ = 2 * int(fmtspec['le'])
    return \
    '''{to_typ} dst[{le_to_typ}];
       {from_typ} src[{le_to_typ}];
       int i;
       {pre}storeu{sufsi}({cast_src}src, {in0});
       {pre}storeu{sufsi}({cast_src}(src + {le}), {in1});
       for (i = 0; i < {le_to_typ}; i++) {{
         dst[i] = ({to_typ})src[i];
       }}
       return {pre}loadu{sufsi_to_typ}({cast_dst}dst);'''. \
       format(cast_src='({}*)'.format(get_type(simd_ext, from_typ)) \
              if from_typ in common.iutypes else '',
              cast_dst='({}*)'.format(get_type(simd_ext, to_typ)) \
              if to_typ in common.iutypes else '',
              le_to_typ=le_to_typ, sufsi_to_typ=suf_si(simd_ext, to_typ),
              **fmtspec)

# -----------------------------------------------------------------------------
## to_mask

def to_mask1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_to_mask_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_to_mask_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(**fmtspec)
    if simd_ext in sse + avx:
        return 'return {in0};'.format(**fmtspec)
    elif simd_ext == 'avx512_skylake':
        if typ in common.iutypes:
            return 'return _mm512_movm_epi{typnbits}({in0});'. \
                   format(**fmtspec)
        elif typ in ['f32', 'f64']:
            return '''return _mm512_castsi512{suf}(
                               _mm512_movm_epi{typnbits}({in0}));'''. \
                               format(**fmtspec)
    else:
        if typ in ['i32', 'u32', 'i64', 'u64']:
            return '''return _mm512_mask_mov{suf}(_mm512_setzero_si512(),
                                 {in0}, _mm512_set1_epi32(-1));'''. \
                                 format(**fmtspec)
        elif typ in ['f32', 'f64']:
            return '''return _mm512_mask_mov{suf}(_mm512_castsi512{suf}(
                               _mm512_setzero_si512()), {in0},
                                 _mm512_castsi512{suf}(
                                   _mm512_set1_epi32(-1)));'''. \
                                   format(**fmtspec)
        else:
            return '''{typ} buf[{le}];
                      int i;
                      for (i = 0; i < {le}; i++) {{
                        if (({in0} >> i) & 1) {{
                          buf[i] = ({typ})-1;
                        }} else {{
                          buf[i] = ({typ})0;
                        }}
                      }}
                      return _mm512_loadu_si512(buf);'''.format(**fmtspec)

# -----------------------------------------------------------------------------
## to_logical

def to_logical1(simd_ext, typ):
    if typ in common.iutypes:
        return '''return nsimd_ne_{simd_ext}_{typ}(
                           {in0}, {pre}setzero{sufsi}());'''.format(**fmtspec)
    elif typ in ['f32', 'f64']:
        return '''return nsimd_reinterpretl_{simd_ext}_{typ}_{utyp}(
                           nsimd_ne_{simd_ext}_{utyp}(
                             {pre}cast{suf2}_si{nbits}({in0}),
                               {pre}setzero_si{nbits}()));'''. \
                               format(suf2=suf_si(simd_ext, typ)[1:],
                                      utyp='u{}'.format(fmtspec['typnbits']),
                                      **fmtspec)
    else:
        return '''nsimd_{simd_ext}_vlf16 ret;
                  ret.v0 = nsimd_to_logical_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_to_logical_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(**fmtspec)

# -----------------------------------------------------------------------------
## zip functions

def zip_half(func, simd_ext, typ):
    if func == 'ziplo':
        get_half0 = '{cast_low}({in0})'
        get_half1 = '{cast_low}({in1})'
    else:
        get_half0 = '{extract}({in0}, 0x01)'
        get_half1 = '{extract}({in1}, 0x01)'

    if simd_ext in ['sse2', 'sse42']:
        if typ == 'f16':
            return '''nsimd_{simd_ext}_v{typ} ret;
            ret.v0 = _mm_unpacklo_ps({in0}.v{k}, {in1}.v{k});
            ret.v1 = _mm_unpackhi_ps({in0}.v{k}, {in1}.v{k});
            return ret;'''.format(k = '0' if func == 'ziplo' else '1',
                                  **fmtspec)
        else:
            return '''return {pre}unpack{lo}{suf}({in0}, {in1});'''.\
                format(k = '0' if func == 'ziplo' else '1',
                       lo = 'lo' if func == 'ziplo' else 'hi',
                       **fmtspec)
    elif simd_ext in ['avx', 'avx2']:
        # Currently, 256 and 512 bits vectors are splitted into 128 bits vectors
        # in order to perform the ziplo/hi operation using the unpacklo/hi sse
        # operations.
        epi = suf_ep(typ)
        if typ in common.iutypes:
            i='i'
            cast_low = '_mm256_castsi256_si128'
            cast_high = '_mm256_castsi128_si256'
            extract = '_mm256_extractf128_si256'
            insert = '_mm256_insertf128_si256'
        elif typ in ['f32', 'f16']:
            i=''
            cast_low = '_mm256_castps256_ps128'
            cast_high = '_mm256_castps128_ps256'
            extract = '_mm256_extractf128_ps'
            insert = '_mm256_insertf128_ps'
        elif typ == 'f64':
            i='d'
            cast_low = '_mm256_castpd256_pd128'
            cast_high = '_mm256_castpd128_pd256'
            extract = '_mm256_extractf128_pd'
            insert = '_mm256_insertf128_pd'
        if typ == 'f16':
            return'''\
            nsimd_{simd_ext}_v{typ} ret;
            __m128 v_tmp0 = {cast_low}({in0}.v{k});
            __m128 v_tmp1 = {cast_low}({in1}.v{k});
            __m128 v_tmp2 = {extract}({in0}.v{k}, 0x01);
            __m128 v_tmp3 = {extract}({in1}.v{k}, 0x01);
            __m128 vres_lo0 = _mm_unpacklo_ps(v_tmp0, v_tmp1);
            __m128 vres_hi0 = _mm_unpackhi_ps(v_tmp0, v_tmp1);
            ret.v0 = {insert}({cast_high}(vres_lo0), vres_hi0, 0x01);
            __m128 vres_lo1 = _mm_unpacklo_ps(v_tmp2, v_tmp3);
            __m128 vres_hi1 = _mm_unpackhi_ps(v_tmp2, v_tmp3);
            ret.v1 = {insert}({cast_high}(vres_lo1), vres_hi1, 0x01);
            return ret;
            '''.format(cast_low=cast_low, cast_high=cast_high,
                       extract=extract, epi=epi, insert=insert,
                       k = '0' if func == 'ziplo' else '1',
                       **fmtspec)
        else:
            return '''\
            __m128{i} v_tmp0 = {get_half0};
            __m128{i} v_tmp1 = {get_half1};
            __m128{i} vres_lo = _mm_unpacklo{epi}(v_tmp0, v_tmp1);
            __m128{i} vres_hi = _mm_unpackhi{epi}(v_tmp0, v_tmp1);
            return {insert}({cast_high}(vres_lo), vres_hi, 0x01);
            '''.format(get_half0=get_half0.format(
                cast_low=cast_low, extract=extract, **fmtspec),
                       get_half1=get_half1.format(
                           cast_low=cast_low, extract=extract, **fmtspec),
                       cast_low=cast_low, cast_high=cast_high,
                       extract=extract, epi=epi, insert=insert, i=i,**fmtspec)
    else:
        if typ in common.iutypes:
            i = 'i'
            cast_low = '_mm512_castsi512_si256'
            cast_high = '_mm512_castsi256_si512'
            extract = '_mm512_extracti32x8_epi32'
            insert = '_mm512_inserti32x8'
        elif typ in ['f32', 'f16']:
            i = ''
            cast_low = '_mm512_castps512_ps256'
            cast_high = '_mm512_castps256_ps512'
            extract = '_mm512_extractf32x8_ps'
            insert = '_mm512_insertf32x8'
        elif typ == 'f64':
            i = 'd'
            cast_low = '_mm512_castpd512_pd256'
            cast_high = '_mm512_castpd256_pd512'
            extract = '_mm512_extractf64x4_pd'
            insert = '_mm512_insertf64x4'

        if typ == 'f16':
            return '''\
            nsimd_{simd_ext}_v{typ} ret;
            __m512 v0 = {in0}.v{k};
            __m512 v1 = {in1}.v{k};
            __m512 vres;
            __m256 v_tmp0, v_tmp1, vres_lo, vres_hi;
            // Low part
            v_tmp0 = _mm512_castps512_ps256(v0);
            v_tmp1 = _mm512_castps512_ps256(v1);
            vres_lo = nsimd_ziplo_avx2_f32(v_tmp0, v_tmp1);
            vres_hi = nsimd_ziphi_avx2_f32(v_tmp0, v_tmp1);
            vres = _mm512_castps256_ps512(vres_lo);
            ret.v0 = _mm512_insertf32x8(vres, vres_hi, 1);
            // High part
            v_tmp0 = _mm512_extractf32x8_ps(v0, 0x1);
            v_tmp1 = _mm512_extractf32x8_ps(v1, 0x1);
            vres_lo = nsimd_ziplo_avx2_f32(v_tmp0, v_tmp1);
            vres_hi = nsimd_ziphi_avx2_f32(v_tmp0, v_tmp1);
            vres = _mm512_castps256_ps512(vres_lo);
            ret.v1 = _mm512_insertf32x8(vres, vres_hi, 1);
            return ret;
            '''.format(**fmtspec, k = '0' if func == 'ziplo' else '1')
        else:
            return '''\
            __m256{i} v_tmp0, v_tmp1;
            v_tmp0 = {get_half0};
            v_tmp1 = {get_half1};
            __m256{i} vres_lo = nsimd_ziplo_avx2_{typ}(v_tmp0, v_tmp1);
            __m256{i} vres_hi = nsimd_ziphi_avx2_{typ}(v_tmp0, v_tmp1);
            __m512{i} vres = {cast_high}(vres_lo);
            return {insert}(vres, vres_hi, 1);
            '''.format(extract=extract,
                       get_half0=get_half0.format(
                           cast_low=cast_low, extract=extract, **fmtspec),
                       get_half1=get_half1.format(
                           cast_low=cast_low, extract=extract, **fmtspec),
                       cast_high=cast_high,
                       cast_low=cast_low,
                       insert=insert, i=i, **fmtspec)

def zip(simd_ext, typ):
    return '// Not implemented yet'

# -----------------------------------------------------------------------------
## unzip functions

def unzip_half(func, simd_ext, typ):
    tab_size = 2 * int(fmtspec['le'])
    vec_size = int(fmtspec['le'])
    loop = '''\
    {typ} tab[{tab_size}];
    {typ} res[{vec_size}];
    int i;
    nsimd_storeu_{simd_ext}_{typ}(tab, {in0});
    nsimd_storeu_{simd_ext}_{typ}(tab + {vec_size}, {in1});
    for(i = 0; i < {vec_size}; i++) {{
    res[i] = tab[2 * i + {offset}];
    }}
    return nsimd_loadu_{simd_ext}_{typ}(res);
    '''.format(tab_size=tab_size, vec_size=vec_size,
               cast='({}*)'.format(get_type(simd_ext, typ)) \
               if typ in common.iutypes else '',
               offset = '0' if func == 'unziplo' else '1', **fmtspec)
    ## SSE ------------------------------------------------------------
    if simd_ext in ['sse2', 'sse42']:
        if typ in ['f32', 'i32', 'u32']:
            v0 = '_mm_castsi128_ps({in0})' if typ in ['i32', 'u32'] else '{in0}'
            v1 = '_mm_castsi128_ps({in1})' if typ in ['i32', 'u32'] else '{in1}'
            ret = '_mm_castps_si128(v_res)' if typ in ['i32', 'u32'] else 'v_res'
            return '''\
            __m128 v_res;
            v_res = _mm_shuffle_ps({v0}, {v1}, {mask});
            return {ret};
            '''.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' if func == 'unziplo' \
                       else '_MM_SHUFFLE(3, 1, 3, 1)',
                       v0=v0.format(**fmtspec), v1=v1.format(**fmtspec), ret=ret,
                       **fmtspec)
        elif typ == 'f16':
            return '''\
            nsimd_{simd_ext}_v{typ} v_res;
            v_res.v0 = _mm_shuffle_ps({in0}.v0, {in0}.v1, {mask});
            v_res.v1 = _mm_shuffle_ps({in1}.v0, {in1}.v1, {mask});
            return v_res;
            '''.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' if func == 'unziplo' \
                       else '_MM_SHUFFLE(3, 1, 3, 1)',
                       **fmtspec)
        elif typ in ['f64', 'i64', 'u64']:
            v0 = '_mm_castsi128_pd({in0})' if typ in ['i64', 'u64'] else '{in0}'
            v1 = '_mm_castsi128_pd({in1})' if typ in ['i64', 'u64'] else '{in1}'
            ret = '_mm_castpd_si128(v_res)' if typ in ['i64', 'u64'] else 'v_res'
            return '''\
            __m128d v_res;
            v_res = _mm_shuffle_pd({v0}, {v1}, {mask});
            return {ret};
            '''.format(mask='0x00' if func == 'unziplo' else '0x03',
                       v0=v0.format(**fmtspec),v1=v1.format(**fmtspec), ret=ret,
                       **fmtspec)
        elif typ in ['i16', 'u16']:
            return '''\
            __m128i v_tmp0 = _mm_shufflelo_epi16({in0}, _MM_SHUFFLE(3, 1, 2, 0));
            v_tmp0 = _mm_shufflehi_epi16(v_tmp0, _MM_SHUFFLE(3, 1, 2, 0));
            __m128i v_tmp1 = _mm_shufflelo_epi16({in1}, _MM_SHUFFLE(3, 1, 2, 0));
            v_tmp1 = _mm_shufflehi_epi16(v_tmp1, _MM_SHUFFLE(3, 1, 2, 0));
            __m128 v_res = _mm_shuffle_ps(
            _mm_castsi128_ps(v_tmp0), _mm_castsi128_ps(v_tmp1), {mask});
            return _mm_castps_si128(v_res);
            '''.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' if func == 'unziplo' \
                       else '_MM_SHUFFLE(3, 1, 3, 1)',
                       **fmtspec)
        else:
            return loop
    ## AVX, AVX2 ----------------------------------------------------
    elif simd_ext in ['avx', 'avx2']:
        ret_template ='''\
        v_tmp0 = _mm256_permute2f128_{t}({v0}, {v0}, 0x01);
        v_tmp0 = _mm256_shuffle_{t}({v0}, v_tmp0, {mask});
        v_tmp1 = _mm256_permute2f128_{t}({v1}, {v1}, 0x01);
        v_tmp1 = _mm256_shuffle_{t}({v1}, v_tmp1, {mask});
        v_res  = _mm256_permute2f128_{t}(v_tmp0, v_tmp1, 0x20);
        {ret} = {v_res};'''
        if typ in ['f32', 'i32', 'u32']:
            v0 = '_mm256_castsi256_ps({in0})' if typ in ['i32', 'u32'] else '{in0}'
            v1 = '_mm256_castsi256_ps({in1})' if typ in ['i32', 'u32'] else '{in1}'
            v_res = '_mm256_castps_si256(v_res)' if typ in ['i32', 'u32'] else 'v_res'
            ret = 'ret'
            src = ret_template .\
                format(mask='_MM_SHUFFLE(2, 0, 2, 0)' if func == 'unziplo' \
                       else '_MM_SHUFFLE(3, 1, 3, 1)',
                       v0=v0, v1=v1, v_res=v_res, ret=ret, t='ps', **fmtspec)
            return '''\
            {styp} ret;
            __m256 v_res, v_tmp0, v_tmp1;
            {src}
            return ret;
            '''.format(src=src.format(**fmtspec), **fmtspec)
        elif typ == 'f16':
            src0 = ret_template.format(
                mask='_MM_SHUFFLE(2, 0, 2, 0)' if func == 'unziplo' \
                else '_MM_SHUFFLE(3, 1, 3, 1)',
                v0='{in0}.v0', v1='{in0}.v1', v_res='v_res', ret='ret.v0', t='ps')
            src1 = ret_template.format(
                mask='_MM_SHUFFLE(2, 0, 2, 0)' if func == 'unziplo' \
                else '_MM_SHUFFLE(3, 1, 3, 1)',
                v0='{in1}.v0', v1='{in1}.v1', v_res='v_res', ret='ret.v1', t='ps')
            return '''\
            nsimd_{simd_ext}_v{typ} ret;
            __m256 v_res, v_tmp0, v_tmp1;
            {src0}
            {src1}
            return ret;
            '''.format(src0=src0.format(**fmtspec), src1=src1.format(**fmtspec),
                       **fmtspec)
        elif typ in ['f64', 'i64', 'u64']:
            v0 = '_mm256_castsi256_pd({in0})' if typ in ['i64', 'u64'] else '{in0}'
            v1 = '_mm256_castsi256_pd({in1})' if typ in ['i64', 'u64'] else '{in1}'
            v_res = '_mm256_castpd_si256(v_res)' if typ in ['i64', 'u64'] else 'v_res'
            src = ret_template . \
                format(mask='0x00' if func == 'unziplo' else '0x03',
                       v0=v0, v1=v1, ret='ret', v_res=v_res, t='pd')
            return '''\
            {styp} ret;
            __m256d v_res, v_tmp0, v_tmp1;
            {src}
            return ret;
            '''.format(src=src.format(**fmtspec), **fmtspec)
        elif typ in ['i16', 'u16']:
            return '''\
            __m128i v_tmp0_hi = _mm256_extractf128_si256({in0}, 0x01);
            __m128i v_tmp0_lo = _mm256_castsi256_si128({in0});
            __m128i v_tmp1_hi = _mm256_extractf128_si256({in1}, 0x01);
            __m128i v_tmp1_lo = _mm256_castsi256_si128({in1});
            v_tmp0_lo = nsimd_{func}_sse2_{typ}(v_tmp0_lo, v_tmp0_hi);
            v_tmp1_lo = nsimd_{func}_sse2_{typ}(v_tmp1_lo, v_tmp1_hi);
            __m256i v_res = _mm256_castsi128_si256(v_tmp0_lo);
            v_res = _mm256_insertf128_si256(v_res, v_tmp1_lo, 0x01);
            return v_res;
            '''.format(func=func, **fmtspec)
        else:
            return loop
        ## AVX 512 --------------------------------------------------
    else:
        if typ == 'f16':
            return '''\
            nsimd_{simd_ext}_v{typ} ret;
            __m512 v_res;
            __m256 v_tmp0, v_tmp1, v_res_lo, v_res_hi;
            v_tmp0 = _mm512_castps512_ps256({in0}.v0);
            v_tmp1 = _mm512_extractf32x8_ps({in0}.v0, 0x01);
            v_res_lo = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
            v_tmp0 = _mm512_castps512_ps256({in0}.v1);
            v_tmp1 = _mm512_extractf32x8_ps({in0}.v1, 0x01);
            v_res_hi = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
            v_res = _mm512_castps256_ps512(v_res_lo);
            v_res = _mm512_insertf32x8(v_res, v_res_hi, 0x01);
            ret.v0 = v_res;
            v_tmp0 = _mm512_castps512_ps256({in1}.v0);
            v_tmp1 = _mm512_extractf32x8_ps({in1}.v0, 0x01);
            v_res_lo = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
            v_tmp0 = _mm512_castps512_ps256({in1}.v1);
            v_tmp1 = _mm512_extractf32x8_ps({in1}.v1, 0x01);
            v_res_hi = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
            v_res = _mm512_castps256_ps512(v_res_lo);
            v_res = _mm512_insertf32x8(v_res, v_res_hi, 0x01);
            ret.v1 = v_res;
            return ret;
            '''.format(func=func, **fmtspec)
        else:
            # return split_opn(func, simd_ext, typ, 2)
            return '''\
            nsimd_avx2_v{typ} v00 = {extract_lo0};
            nsimd_avx2_v{typ} v01 = {extract_hi0};
            nsimd_avx2_v{typ} v10 = {extract_lo1};
            nsimd_avx2_v{typ} v11 = {extract_hi1};
            v00 = nsimd_{func}_avx2_{typ}(v00, v01);
            v01 = nsimd_{func}_avx2_{typ}(v10, v11);
            return {merge};
            '''.format(func=func,
                extract_lo0=extract(simd_ext, typ, LO, common.in0),
                extract_lo1=extract(simd_ext, typ, LO, common.in1),
                extract_hi0=extract(simd_ext, typ, HI, common.in0),
                extract_hi1=extract(simd_ext, typ, HI, common.in1),
                merge=setr(simd_ext, typ, 'v00', 'v01'), **fmtspec)

# -----------------------------------------------------------------------------
## get_impl function

def get_impl(func, simd_ext, from_typ, to_typ):
    global fmtspec

    fmtspec = {
      'simd_ext': simd_ext,
      'typ': from_typ,
      'styp': get_type(simd_ext, from_typ),
      'from_typ': from_typ,
      'to_typ': to_typ,
      'pre': pre(simd_ext),
      'suf': suf_ep(from_typ),
      'sufsi': suf_si(simd_ext, from_typ),
      'in0': common.in0,
      'in1': common.in1,
      'in2': common.in2,
      'in3': common.in3,
      'in4': common.in4,
      'in5': common.in5,
      'nbits': nbits(simd_ext),
      'le': int(nbits(simd_ext)) // int(from_typ[1:]),
      'typnbits': from_typ[1:]
    }

    impls = {
        'loada': lambda: load(simd_ext, from_typ, True),
        'load2a': lambda: load_deg234(simd_ext, from_typ, True, 2),
        'load3a': lambda: load_deg234(simd_ext, from_typ, True, 3),
        'load4a': lambda: load_deg234(simd_ext, from_typ, True, 4),
        'loadu': lambda: load(simd_ext, from_typ, False),
        'load2u': lambda: load_deg234(simd_ext, from_typ, False, 2),
        'load3u': lambda: load_deg234(simd_ext, from_typ, False, 3),
        'load4u': lambda: load_deg234(simd_ext, from_typ, False, 4),
        'storea': lambda: store(simd_ext, from_typ, True),
        'store2a': lambda: store_deg234(simd_ext, from_typ, True, 2),
        'store3a': lambda: store_deg234(simd_ext, from_typ, True, 3),
        'store4a': lambda: store_deg234(simd_ext, from_typ, True, 4),
        'storeu': lambda: store(simd_ext, from_typ, False),
        'store2u': lambda: store_deg234(simd_ext, from_typ, False, 2),
        'store3u': lambda: store_deg234(simd_ext, from_typ, False, 3),
        'store4u': lambda: store_deg234(simd_ext, from_typ, False, 4),
        'andb': lambda: binop2('andb', simd_ext, from_typ),
        'xorb': lambda: binop2('xorb', simd_ext, from_typ),
        'orb': lambda: binop2('orb', simd_ext, from_typ),
        'andl': lambda: binlop2('andl', simd_ext, from_typ),
        'xorl': lambda: binlop2('xorl', simd_ext, from_typ),
        'orl': lambda: binlop2('orl', simd_ext, from_typ),
        'notb': lambda: not1(simd_ext, from_typ),
        'notl': lambda: lnot1(simd_ext, from_typ),
        'andnotb': lambda: andnot2(simd_ext, from_typ),
        'andnotl': lambda: landnot2(simd_ext, from_typ),
        'add': lambda: addsub('add', simd_ext, from_typ),
        'sub': lambda: addsub('sub', simd_ext, from_typ),
        'div': lambda: div2(simd_ext, from_typ),
        'sqrt': lambda: sqrt1(simd_ext, from_typ),
        'len': lambda: len1(simd_ext, from_typ),
        'mul': lambda: mul2(simd_ext, from_typ),
        'shl': lambda: shl_shr('shl', simd_ext, from_typ),
        'shr': lambda: shl_shr('shr', simd_ext, from_typ),
        'set1': lambda: set1(simd_ext, from_typ),
        'eq': lambda: eq2(simd_ext, from_typ),
        'ne': lambda: neq2(simd_ext, from_typ),
        'gt': lambda: gt2(simd_ext, from_typ),
        'lt': lambda: lt2(simd_ext, from_typ),
        'ge': lambda: geq2(simd_ext, from_typ),
        'le': lambda: leq2(simd_ext, from_typ),
        'if_else1': lambda: if_else1(simd_ext, from_typ),
        'min': lambda: minmax('min', simd_ext, from_typ),
        'max': lambda: minmax('max', simd_ext, from_typ),
        'loadla': lambda: loadl(simd_ext, from_typ, True),
        'loadlu': lambda: loadl(simd_ext, from_typ, False),
        'storela': lambda: storel(simd_ext, from_typ, True),
        'storelu': lambda: storel(simd_ext, from_typ, False),
        'abs': lambda: abs1(simd_ext, from_typ),
        'fma': lambda: fma_fms('fma', simd_ext, from_typ),
        'fnma': lambda: fma_fms('fnma', simd_ext, from_typ),
        'fms': lambda: fma_fms('fms', simd_ext, from_typ),
        'fnms': lambda: fma_fms('fnms', simd_ext, from_typ),
        'ceil': lambda: round1('ceil', simd_ext, from_typ),
        'floor': lambda: round1('floor', simd_ext, from_typ),
        'trunc': lambda: trunc1(simd_ext, from_typ),
        'round_to_even': lambda: round_to_even1(simd_ext, from_typ),
        'all': lambda: all_any('all', simd_ext, from_typ),
        'any': lambda: all_any('any', simd_ext, from_typ),
        'reinterpret': lambda: reinterpret1(simd_ext, from_typ, to_typ),
        'reinterpretl': lambda: reinterpretl1(simd_ext, from_typ, to_typ),
        'cvt': lambda: convert1(simd_ext, from_typ, to_typ),
        'rec11': lambda: rec11_rsqrt11('rcp', simd_ext, from_typ),
        'rec8': lambda: rec11_rsqrt11('rcp', simd_ext, from_typ),
        'rsqrt11': lambda: rec11_rsqrt11('rsqrt', simd_ext, from_typ),
        'rsqrt8': lambda: rec11_rsqrt11('rsqrt', simd_ext, from_typ),
        'rec': lambda: rec1(simd_ext, from_typ),
        'neg': lambda: neg1(simd_ext, from_typ),
        'nbtrue': lambda: nbtrue1(simd_ext, from_typ),
        'reverse': lambda: reverse1(simd_ext, from_typ),
        'addv': lambda: addv(simd_ext, from_typ),
        'upcvt': lambda: upcvt1(simd_ext, from_typ, to_typ),
        'downcvt': lambda: downcvt1(simd_ext, from_typ, to_typ),
        'to_mask': lambda: to_mask1(simd_ext, from_typ),
        'to_logical': lambda: to_logical1(simd_ext, from_typ),
        'ziplo': lambda: zip_half('ziplo', simd_ext, from_typ),
        'ziphi': lambda: zip_half('ziphi', simd_ext, from_typ),
        'unziplo': lambda: unzip_half('unziplo', simd_ext, from_typ),
        'unziphi': lambda: unzip_half('unziphi', simd_ext, from_typ)
    }
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if not from_typ in common.types:
        raise ValueError('Unknown type "{}"'.format(from_typ))
    if not func in impls:
        return common.NOT_IMPLEMENTED
    else:
        return impls[func]()
