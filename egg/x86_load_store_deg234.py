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

import platform_x86 as x86
import common

sse = ['sse2', 'sse42']
avx = ['avx', 'avx2']
avx512 = ['avx512_knl', 'avx512_skylake']

###############################################################################
# Helper

def perm64(var1, var2, ind1, ind2):
    return '''_mm_castpd_si128(_mm_shuffle_pd(
                _mm_castsi128_pd({}), _mm_castsi128_pd(
                  {}), _MM_SHUFFLE2({}, {})))'''.format(var1, var2, ind1, ind2)

###############################################################################

def get_load_v0v1(simd_ext, typ, align, fmtspec):
    load = '{pre}load{a}{sufsi}'.format(a='' if align else 'u', **fmtspec)
    if typ in ['f32', 'f64']:
        return '''{styp} v0 = {load}(a0);
                  {styp} v1 = {load}(a0 + {le});'''. \
                  format(load=load, **fmtspec)
    else:
        return '''{styp} v0 = {load}(({styp}*)a0);
                  {styp} v1 = {load}(({styp}*)a0 + 1);'''. \
                  format(load=load, **fmtspec)

###############################################################################

def load2_sse(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['load_v0v1'] = get_load_v0v1('sse', typ, align, fmtspec)
    if typ in ['i8', 'u8']:
        if simd_ext == 'sse42':
            return \
            '''nsimd_sse42_v{typ}x2 ret;
               {load_v0v1}
               __m128i mask = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14,
                                           12, 10, 8, 6, 4, 2, 0);
               __m128i A0 = _mm_shuffle_epi8(v0, mask);
               __m128i B0 = _mm_shuffle_epi8(v1, mask);
               ret.v0 = {perm0};
               ret.v1 = {perm1};
               return ret;'''. \
               format(perm0=perm64('A0', 'B0', '0', '0'),
                      perm1=perm64('A0', 'B0', '1', '1'), **fmtspec)
        else:
            return \
            '''nsimd_sse2_v{typ}x2 ret;
               {load_v0v1}
               __m128i A1 = _mm_unpacklo_epi8(v0, v1);
               __m128i B2 = _mm_unpackhi_epi8(v0, v1);
               __m128i A3 = _mm_unpacklo_epi8(A1, B2);
               __m128i B4 = _mm_unpackhi_epi8(A1, B2);
               __m128i A5 = _mm_unpacklo_epi8(A3, B4);
               __m128i B6 = _mm_unpackhi_epi8(A3, B4);
               ret.v0 = _mm_unpacklo_epi8(A5, B6);
               ret.v1 = _mm_unpackhi_epi8(A5, B6);
               return ret;'''.format(**fmtspec)
    if typ in ['i16', 'u16']:
        if simd_ext == 'sse42':
            return \
            '''nsimd_sse42_v{typ}x2 ret;
               {load_v0v1}
               __m128i mask = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2,
                                           13, 12, 9, 8, 5, 4, 1, 0);
               __m128i A0 = _mm_shuffle_epi8(v0, mask);
               __m128i B0 = _mm_shuffle_epi8(v1, mask);
               ret.v0 = {perm0};
               ret.v1 = {perm1};
               return ret;'''. \
               format(perm0=perm64('A0', 'B0', '0', '0'),
                      perm1=perm64('A0', 'B0', '1', '1'), **fmtspec)
        else:
            return \
            '''nsimd_sse2_v{typ}x2 ret;
               {load_v0v1}
               __m128i v2 = _mm_unpacklo_epi16(v0, v1);
               __m128i v3 = _mm_unpackhi_epi16(v0, v1);
               __m128i v5 = _mm_unpacklo_epi16(v2, v3);
               __m128i v6 = _mm_unpackhi_epi16(v2, v3);
               ret.v0 = _mm_unpacklo_epi16(v5, v6);
               ret.v1 = _mm_unpackhi_epi16(v5, v6);
               return ret;'''.format(**fmtspec)
    if typ in ['i32', 'u32', 'f32']:
        return '''nsimd_{simd_ext}_v{typ}x2 ret;
                  {load_v0v1}
                  {styp} A0 = _mm_unpacklo{suf}(v0, v1);
                  {styp} B0 = _mm_unpackhi{suf}(v0, v1);
                  ret.v0 = _mm_unpacklo{suf}(A0, B0);
                  ret.v1 = _mm_unpackhi{suf}(A0, B0);
                  return ret;'''.format(**fmtspec)
    if typ in ['i64', 'u64', 'f64']:
        return '''nsimd_{simd_ext}_v{typ}x2 ret;
                  {load_v0v1}
                  ret.v0 = _mm_unpacklo{suf}(v0, v1);
                  ret.v1 = _mm_unpackhi{suf}(v0, v1);
                  return ret;'''.format(**fmtspec)

###############################################################################

def load2_avx(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['exlo_v0'] = x86.extract('avx', typ, x86.LO, 'v0')
    fmtspec['exhi_v0'] = x86.extract('avx', typ, x86.HI, 'v0')
    fmtspec['exlo_v1'] = x86.extract('avx', typ, x86.LO, 'v1')
    fmtspec['exhi_v1'] = x86.extract('avx', typ, x86.HI, 'v1')
    fmtspec['load_v0v1'] = get_load_v0v1('avx', typ, align, fmtspec)
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        if simd_ext == 'avx2':
            return \
            '''nsimd_avx2_v{typ}x2 ret;
               {load_v0v1}

               __m256i mask = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,
                                               1, 3, 5, 7, 9, 11, 13, 15,
                                               0, 2, 4, 6, 8, 10, 12, 14,
                                               1, 3, 5, 7, 9, 11, 13, 15);

               __m256i A1 = _mm256_shuffle_epi8(v0, mask);
               __m256i B1 = _mm256_shuffle_epi8(v1, mask);

               __m256i A2 = _mm256_permute4x64_epi64(A1, _MM_SHUFFLE(3,1,2,0));
               __m256i B2 = _mm256_permute4x64_epi64(B1, _MM_SHUFFLE(3,1,2,0));

               ret.v0 = _mm256_permute2f128_si256(A2, B2, 2 << 4);
               ret.v1 = _mm256_permute2f128_si256(A2, B2, (3 << 4) | 1);
               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_avx_v{typ}x2 ret;
               {load_v0v1}

               __m128i v0a = {exlo_v0};
               __m128i v0b = {exhi_v0};
               __m128i v1a = {exlo_v1};
               __m128i v1b = {exhi_v1};

               __m128i mask = _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1,
                                           14, 12, 10, 8, 6, 4, 2, 0);

               __m128i A0a = _mm_shuffle_epi8(v0a, mask);
               __m128i B0a = _mm_shuffle_epi8(v1a, mask);
               __m128i A1a = {perm_a0};
               __m128i B1a = {perm_a1};

               __m128i A0b = _mm_shuffle_epi8(v0b, mask);
               __m128i B0b = _mm_shuffle_epi8(v1b, mask);
               __m128i A1b = {perm_b0};
               __m128i B1b = {perm_b1};

               ret.v0 = {merge_A1};
               ret.v1 = {merge_B1};
               return ret;'''. \
               format(merge_A1=x86.setr('avx', typ, 'A1a', 'A1b'),
                      merge_B1=x86.setr('avx', typ, 'B1a', 'B1b'),
                      perm_a0=perm64('A0a', 'B0a', '0', '0'),
                      perm_a1=perm64('A0a', 'B0a', '1', '1'),
                      perm_b0=perm64('A0b', 'B0b', '0', '0'),
                      perm_b1=perm64('A0b', 'B0b', '1', '1'), **fmtspec)
    if typ in ['i16', 'u16']:
        if simd_ext == 'avx2':
            return \
            '''nsimd_avx2_v{typ}x2 ret;
               {load_v0v1}

               __m256i A1 = _mm256_unpacklo_epi16(v0, v1);
               __m256i B1 = _mm256_unpackhi_epi16(v0, v1);
               __m256i A2 = _mm256_unpacklo_epi16(A1, B1);
               __m256i B2 = _mm256_unpackhi_epi16(A1, B1);
               ret.v0 = _mm256_unpacklo_epi16(A2, B2);
               ret.v1 = _mm256_unpackhi_epi16(A2, B2);
               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_avx_v{typ}x2 ret;
               {load_v0v1}

               __m128i Aa = {exlo_v0};
               __m128i Ba = {exhi_v0};
               __m128i Ab = {exlo_v1};
               __m128i Bb = {exhi_v1};

               __m128i mask = _mm_set_epi8(15, 14, 11, 10, 7, 6, 3, 2,
                                           13, 12, 9, 8, 5, 4, 1, 0);

               __m128i XY0 = _mm_shuffle_epi8(Aa, mask);
               __m128i XY1 = _mm_shuffle_epi8(Ba, mask);
               __m128i Xa = {perm0};
               __m128i Ya = {perm1};

               XY0 = _mm_shuffle_epi8(Ab, mask);
               XY1 = _mm_shuffle_epi8(Bb, mask);
               __m128i Xb = {perm0};
               __m128i Yb = {perm1};

               ret.v0 = {mergeX};
               ret.v1 = {mergeY};

               return ret;'''. \
               format(perm0=perm64('XY0', 'XY1', '0', '0'),
                      perm1=perm64('XY0', 'XY1', '1', '1'),
                      mergeX=x86.setr('avx', typ, 'Xa', 'Xb'),
                      mergeY=x86.setr('avx', typ, 'Ya', 'Yb'), **fmtspec)
    if typ == 'f32':
        return '''nsimd_{simd_ext}_vf32x2 ret;
                  {load_v0v1}
                  __m256 A1 = _mm256_unpacklo_ps(v0, v1);
                  __m256 B1 = _mm256_unpackhi_ps(v0, v1);
                  ret.v0 = _mm256_unpacklo_ps(A1, B1);
                  ret.v1 = _mm256_unpackhi_ps(A1, B1);
                  return ret;'''.format(**fmtspec)
    if typ in ['i32', 'u32']:
        if simd_ext == 'avx2':
            return \
            '''nsimd_avx2_v{typ}x2 ret;
               {load_v0v1}
               __m256i A1 = _mm256_unpacklo_epi32(v0, v1);
               __m256i B1 = _mm256_unpackhi_epi32(v0, v1);
               ret.v0 = _mm256_unpacklo_epi32(A1, B1);
               ret.v1 = _mm256_unpackhi_epi32(A1, B1);
               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_avx_v{typ}x2 ret;
               nsimd_avx_vf32x2 retf32 = nsimd_load2{a}_avx_f32((f32 *){in0});
               ret.v0 = _mm256_castps_si256(retf32.v0);
               ret.v1 = _mm256_castps_si256(retf32.v1);
               return ret;'''.format(**fmtspec)
    if typ == 'f64':
        return '''nsimd_{simd_ext}_vf64x2 ret;
                  {load_v0v1}
                  ret.v0 = _mm256_unpacklo_pd(v0, v1);
                  ret.v1 = _mm256_unpackhi_pd(v0, v1);
                  return ret;'''.format(**fmtspec)
    if typ in ['i64', 'u64']:
        if simd_ext == 'avx2':
            return \
            '''nsimd_avx2_v{typ}x2 ret;
               {load_v0v1}
               ret.v0 = _mm256_unpacklo_epi64(v0, v1);
               ret.v1 = _mm256_unpackhi_epi64(v0, v1);
               return ret;'''.format(**fmtspec)
        else:
            return \
             '''nsimd_avx_v{typ}x2 ret;
                nsimd_avx_vf64x2 retf64 = nsimd_load2{a}_avx_f64((f64 *){in0});
                ret.v0 = _mm256_castpd_si256(retf64.v0);
                ret.v1 = _mm256_castpd_si256(retf64.v1);
                return ret;'''.format(**fmtspec)

###############################################################################

def load2_avx512(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['exlo_v0'] = x86.extract(simd_ext, typ, x86.LO, 'v0')
    fmtspec['exhi_v0'] = x86.extract(simd_ext, typ, x86.HI, 'v0')
    fmtspec['exlo_v1'] = x86.extract(simd_ext, typ, x86.LO, 'v1')
    fmtspec['exhi_v1'] = x86.extract(simd_ext, typ, x86.HI, 'v1')
    fmtspec['load_v0v1'] = get_load_v0v1(simd_ext, typ, align, fmtspec)
    if typ in ['i8', 'u8']:
        return \
        '''nsimd_{simd_ext}_v{typ}x2 ret;
           {load_v0v1}

           __m256i A0 = {exlo_v0};
           __m256i B0 = {exhi_v0};
           __m256i C0 = {exlo_v1};
           __m256i D0 = {exhi_v1};

           __m256i mask = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,
                                           1, 3, 5, 7, 9, 11, 13, 15,
                                           0, 2, 4, 6, 8, 10, 12, 14,
                                           1, 3, 5, 7, 9, 11, 13, 15);

           __m256i A1 = _mm256_shuffle_epi8(A0, mask);
           __m256i B1 = _mm256_shuffle_epi8(B0, mask);
           __m256i C1 = _mm256_shuffle_epi8(C0, mask);
           __m256i D1 = _mm256_shuffle_epi8(D0, mask);

           __m256i A2 = _mm256_permute4x64_epi64(A1, _MM_SHUFFLE(3,1,2,0));
           __m256i B2 = _mm256_permute4x64_epi64(B1, _MM_SHUFFLE(3,1,2,0));
           __m256i C2 = _mm256_permute4x64_epi64(C1, _MM_SHUFFLE(3,1,2,0));
           __m256i D2 = _mm256_permute4x64_epi64(D1, _MM_SHUFFLE(3,1,2,0));

           __m256i A3 = _mm256_permute2f128_si256(A2, B2, 2 << 4);
           __m256i B3 = _mm256_permute2f128_si256(A2, B2, (3 << 4) | 1);
           __m256i C3 = _mm256_permute2f128_si256(C2, D2, 2 << 4);
           __m256i D3 = _mm256_permute2f128_si256(C2, D2, (3 << 4) | 1);

           ret.v0 = {mergeAC};
           ret.v1 = {mergeBD};
           return ret;'''.format(mergeAC=x86.setr(simd_ext, typ, 'A3', 'C3'),
                                 mergeBD=x86.setr(simd_ext, typ, 'B3', 'D3'),
                                 **fmtspec)
    if typ in ['i16', 'u16']:
        return \
        '''nsimd_{simd_ext}_v{typ}x2 ret;
           {load_v0v1}

           __m256i A0a = {exlo_v0};
           __m256i B0a = {exhi_v0};
           __m256i A0b = {exlo_v1};
           __m256i B0b = {exhi_v1};

           __m256i A1 = _mm256_unpacklo_epi16(A0a, B0a);
           __m256i B1 = _mm256_unpackhi_epi16(A0a, B0a);
           __m256i A2 = _mm256_unpacklo_epi16(A1, B1);
           __m256i B2 = _mm256_unpackhi_epi16(A1, B1);
           __m256i A3a = _mm256_unpacklo_epi16(A2, B2);
           __m256i B3a = _mm256_unpackhi_epi16(A2, B2);

           A1 = _mm256_unpacklo_epi16(A0b, B0b);
           B1 = _mm256_unpackhi_epi16(A0b, B0b);
           A2 = _mm256_unpacklo_epi16(A1, B1);
           B2 = _mm256_unpackhi_epi16(A1, B1);
           __m256i A3b = _mm256_unpacklo_epi16(A2, B2);
           __m256i B3b = _mm256_unpackhi_epi16(A2, B2);

           ret.v0 = {mergeA};
           ret.v1 = {mergeB};
           return ret;'''.format(mergeA=x86.setr(simd_ext, typ, 'A3a', 'A3b'),
                                 mergeB=x86.setr(simd_ext, typ, 'B3a', 'B3b'),
                                 **fmtspec)
    if typ in ['f32', 'i32', 'u32']:
        return \
        '''nsimd_{simd_ext}_v{typ}x2 ret;
           {load_v0v1}
           __m512i mask1 = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14,
                                             16, 18, 20, 22, 24, 26, 28, 30);
           __m512i mask2 = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15,
                                             17, 19, 21, 23, 25, 27, 29, 31);
           ret.v0 = _mm512_permutex2var{suf}(v0, mask1, v1);
           ret.v1 = _mm512_permutex2var{suf}(v0, mask2, v1);
           return ret;'''.format(**fmtspec)
    if typ in ['f64', 'i64', 'u64']:
        return \
        '''nsimd_{simd_ext}_v{typ}x2 ret;
           {load_v0v1}
           ret.v0 = _mm512_unpacklo{suf}(v0, v1);
           ret.v1 = _mm512_unpackhi{suf}(v0, v1);
           return ret;'''.format(**fmtspec)

###############################################################################

def store2(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['store'] = '{pre}store{a}{sufsi}'.format(a='' if align else 'u',
                                                     **fmtspec)
    if typ in ['f32', 'f64']:
        dest1 = '{in0}'.format(**fmtspec)
        dest2 = '{in0} + {le}'.format(**fmtspec)
    else:
        dest1 = '(__m{nbits}i *){in0}'.format(**fmtspec)
        dest2 = '(__m{nbits}i *){in0} + 1'.format(**fmtspec)
    normal = '''{store}({dest1}, {pre}unpacklo{suf}({in1}, {in2}));
                {store}({dest2}, {pre}unpackhi{suf}({in1}, {in2}));'''. \
                format(dest1=dest1, dest2=dest2, **fmtspec)
    if simd_ext in sse:
        return normal
    fmtspec['exlo_in1'] = x86.extract(simd_ext, typ, x86.LO, common.in1)
    fmtspec['exhi_in1'] = x86.extract(simd_ext, typ, x86.HI, common.in1)
    fmtspec['exlo_in2'] = x86.extract(simd_ext, typ, x86.LO, common.in2)
    fmtspec['exhi_in2'] = x86.extract(simd_ext, typ, x86.HI, common.in2)
    fmtspec['normal'] = normal
    fmtspec['dest1'] = dest1
    fmtspec['dest2'] = dest2
    if simd_ext == 'avx2':
        if typ in ['i8', 'u8']:
            return \
            '''__m256i A1 = _mm256_permute2f128_si256({in1}, {in2}, 2 << 4);
               __m256i B1 = _mm256_permute2f128_si256(
                              {in1}, {in2}, (3 << 4) | 1);

               __m256i A2 = _mm256_permute4x64_epi64(A1, _MM_SHUFFLE(3,1,2,0));
               __m256i B2 = _mm256_permute4x64_epi64(B1, _MM_SHUFFLE(3,1,2,0));

               __m256i mask = _mm256_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11,
                                               4, 12, 5, 13, 6, 14, 7, 15,
                                               0, 8, 1, 9, 2, 10, 3, 11,
                                               4, 12, 5, 13, 6, 14, 7, 15);

               {store}({dest1}, _mm256_shuffle_epi8(A2, mask));
               {store}({dest2}, _mm256_shuffle_epi8(B2, mask));'''. \
               format(**fmtspec)
        if typ in ['i16', 'u16']:
            return normal
    if simd_ext == 'avx':
        if typ in ['i8', 'u8']:
            return \
            '''__m128i v0a = {exlo_in1};
               __m128i v0b = {exhi_in1};
               __m128i v1a = {exlo_in2};
               __m128i v1b = {exhi_in2};

               __m128i A1a = _mm_unpacklo_epi8(v0a, v1a);
               __m128i B1a = _mm_unpackhi_epi8(v0a, v1a);
               __m128i A1b = _mm_unpacklo_epi8(v0b, v1b);
               __m128i B1b = _mm_unpackhi_epi8(v0b, v1b);

               __m256i A1 = {mergeA1};
               __m256i B1 = {mergeB1};

               {store}({dest1}, A1);
               {store}({dest2}, B1);'''. \
               format(mergeA1=x86.setr('avx', typ, 'A1a', 'A1b'),
                      mergeB1=x86.setr('avx', typ, 'B1a', 'B1b'),
                      **fmtspec)
        if typ in ['i16', 'u16']:
            return \
            '''__m128i Xa = {exlo_in1};
               __m128i Xb = {exhi_in1};
               __m128i Ya = {exlo_in2};
               __m128i Yb = {exhi_in2};

               __m128i A0 = _mm_unpacklo_epi16(Xa, Ya);
               __m128i B0 = _mm_unpackhi_epi16(Xa, Ya);
               __m128i A1 = _mm_unpacklo_epi16(Xb, Yb);
               __m128i B1 = _mm_unpackhi_epi16(Xb, Yb);

               __m256i A = {merge0};
               __m256i B = {merge1};

               {store}({dest1}, A);
               {store}({dest2}, B);'''. \
               format(merge0=x86.setr('avx', typ, 'A0', 'B0'),
                      merge1=x86.setr('avx', typ, 'A1', 'B1'),
                      **fmtspec)
    if (simd_ext in avx and typ in ['f32', 'f64']) or \
       simd_ext == 'avx2' and typ in ['i32', 'u32', 'i64', 'u64']:
        return normal
    if simd_ext == 'avx' and typ in ['i32', 'u32', 'i64', 'u64']:
        ftyp = '__m256' if typ in ['i32', 'u32'] else '__m256d'
        fsuf = 'ps' if typ in ['i32', 'u32'] else 'pd'
        return '''{ftyp} v0 = _mm256_castsi256_{fsuf}({in1});
                  {ftyp} v1 = _mm256_castsi256_{fsuf}({in2});
                  {store}({dest1}, _mm256_cast{fsuf}_si256(
                          _mm256_unpacklo_{fsuf}(v0, v1)));
                  {store}({dest2}, _mm256_cast{fsuf}_si256(
                          _mm256_unpackhi_{fsuf}(v0, v1)));'''. \
                          format(ftyp=ftyp, fsuf=fsuf, **fmtspec)
    if simd_ext in avx512:
        if typ in ['i8', 'u8']:
            return \
            '''__m256i A1 = {exlo_in1};
               __m256i B1 = {exhi_in1};
               __m256i C1 = {exlo_in2};
               __m256i D1 = {exhi_in2};

               __m256i A2 = _mm256_permute2f128_si256(A1, C1, 2 << 4);
               __m256i B2 = _mm256_permute2f128_si256(A1, C1, (3 << 4) | 1);
               __m256i C2 = _mm256_permute2f128_si256(B1, D1, 2 << 4);
               __m256i D2 = _mm256_permute2f128_si256(B1, D1, (3 << 4) | 1);

               __m256i A3 = _mm256_permute4x64_epi64(A2, _MM_SHUFFLE(3,1,2,0));
               __m256i B3 = _mm256_permute4x64_epi64(B2, _MM_SHUFFLE(3,1,2,0));
               __m256i C3 = _mm256_permute4x64_epi64(C2, _MM_SHUFFLE(3,1,2,0));
               __m256i D3 = _mm256_permute4x64_epi64(D2, _MM_SHUFFLE(3,1,2,0));

               __m256i mask = _mm256_setr_epi8(0, 8, 1, 9, 2, 10, 3, 11,
                                               4, 12, 5, 13, 6, 14, 7, 15,
                                               0, 8, 1, 9, 2, 10, 3, 11,
                                               4, 12, 5, 13, 6, 14, 7, 15);

               __m256i A4 = _mm256_shuffle_epi8(A3, mask);
               __m256i B4 = _mm256_shuffle_epi8(B3, mask);
               __m256i C4 = _mm256_shuffle_epi8(C3, mask);
               __m256i D4 = _mm256_shuffle_epi8(D3, mask);

               {store}({dest1}, {mergeAB});
               {store}({dest2}, {mergeCD});'''. \
               format(mergeAB=x86.setr(simd_ext, typ, 'A4', 'B4'),
                      mergeCD=x86.setr(simd_ext, typ, 'C4', 'D4'),
                      **fmtspec)
        if typ in ['i16', 'u16']:
            return \
            '''__m256i A0a = {exlo_in1};
               __m256i A0b = {exhi_in1};
               __m256i B0a = {exlo_in2};
               __m256i B0b = {exhi_in2};

               __m256i A1a = _mm256_unpacklo_epi16(A0a, B0a);
               __m256i B1a = _mm256_unpackhi_epi16(A0a, B0a);
               __m256i A1b = _mm256_unpacklo_epi16(A0b, B0b);
               __m256i B1b = _mm256_unpackhi_epi16(A0b, B0b);

               {store}({dest1}, {mergea});
               {store}({dest2}, {mergeb});'''.\
               format(mergea=x86.setr(simd_ext, typ, 'A1a', 'B1a'),
                      mergeb=x86.setr(simd_ext, typ, 'A1b', 'B1b'),
                      **fmtspec)
        if typ in ['i32', 'f32', 'u32']:
            return \
            '''__m512i mask1 = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19,
                                                 4, 20, 5, 21, 6, 22, 7, 23);
               __m512i mask2 = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27,
                                                 12, 28, 13, 29, 14, 30, 15,
                                                 31);
               {store}({dest1}, _mm512_permutex2var{suf}({in1}, mask1, {in2}));
               {store}({dest2}, _mm512_permutex2var{suf}(
                   {in1}, mask2, {in2}));'''.format(**fmtspec)
        if typ in ['i64', 'u64', 'f64']:
            return \
            '''{store}({dest1}, _mm512_unpacklo{suf}({in1}, {in2}));
               {store}({dest2}, _mm512_unpackhi{suf}({in1}, {in2}));'''. \
               format(**fmtspec)

###############################################################################

def get_load_v0v1v2v3(simd_ext, typ, align, fmtspec):
    load = '{pre}load{a}{sufsi}'.format(a='' if align else 'u', **fmtspec)
    if typ in ['f32', 'f64']:
        return '''{styp} v0 = {load}(a0);
                  {styp} v1 = {load}(a0 + {le});
                  {styp} v2 = {load}(a0 + (2 * {le}));
                  {styp} v3 = {load}(a0 + (3 * {le}));'''. \
                  format(load=load, **fmtspec)
    else:
        return '''{styp} v0 = {load}(({styp}*)a0);
                  {styp} v1 = {load}(({styp}*)a0 + 1);
                  {styp} v2 = {load}(({styp}*)a0 + 2);
                  {styp} v3 = {load}(({styp}*)a0 + 3);'''. \
                  format(load=load, **fmtspec)

###############################################################################

def load4_sse(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['load_v0v1v2v3'] = get_load_v0v1v2v3('sse', typ, align, fmtspec)
    if typ in ['i8', 'u8']:
        if simd_ext == 'sse42':
            return \
            '''nsimd_sse42_v{typ}x4 ret;
               {load_v0v1v2v3}
               __m128i mask = _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2,
                                           13,  9, 5, 1, 12,  8, 4, 0);
               __m128d A1 = _mm_castsi128_pd(_mm_shuffle_epi8(v0, mask));
               __m128d B1 = _mm_castsi128_pd(_mm_shuffle_epi8(v1, mask));
               __m128d C1 = _mm_castsi128_pd(_mm_shuffle_epi8(v2, mask));
               __m128d D1 = _mm_castsi128_pd(_mm_shuffle_epi8(v3, mask));

               __m128 A2 = _mm_castpd_ps(_mm_shuffle_pd(A1, B1,
                                                        _MM_SHUFFLE2(0, 0)));
               __m128 A3 = _mm_castpd_ps(_mm_shuffle_pd(C1, D1,
                                                        _MM_SHUFFLE2(0, 0)));
               __m128 C2 = _mm_castpd_ps(_mm_shuffle_pd(A1, B1,
                                                        _MM_SHUFFLE2(1, 1)));
               __m128 C3 = _mm_castpd_ps(_mm_shuffle_pd(C1, D1,
                                                        _MM_SHUFFLE2(1, 1)));

               ret.v0 = _mm_castps_si128(_mm_shuffle_ps(
                            A2, A3, _MM_SHUFFLE(2, 0, 2, 0)));
               ret.v1 = _mm_castps_si128(_mm_shuffle_ps(
                            A2, A3, _MM_SHUFFLE(3, 1, 3, 1)));
               ret.v2 = _mm_castps_si128(_mm_shuffle_ps(
                            C2, C3, _MM_SHUFFLE(2, 0, 2, 0)));
               ret.v3 = _mm_castps_si128(_mm_shuffle_ps(
                            C2, C3, _MM_SHUFFLE(3, 1, 3, 1)));
               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_sse2_v{typ}x4 ret;
               {load_v0v1v2v3}
               __m128i A1 = _mm_unpacklo_epi8(v0, v2);
               __m128i B1 = _mm_unpackhi_epi8(v0, v2);
               __m128i C1 = _mm_unpacklo_epi8(v1, v3);
               __m128i D1 = _mm_unpackhi_epi8(v1, v3);

               __m128i A2 = _mm_unpacklo_epi8(A1, C1);
               __m128i B2 = _mm_unpackhi_epi8(A1, C1);
               __m128i C2 = _mm_unpacklo_epi8(B1, D1);
               __m128i D2 = _mm_unpackhi_epi8(B1, D1);

               __m128i A3 = _mm_unpacklo_epi8(A2, C2);
               __m128i B3 = _mm_unpackhi_epi8(A2, C2);
               __m128i C3 = _mm_unpacklo_epi8(B2, D2);
               __m128i D3 = _mm_unpackhi_epi8(B2, D2);

               ret.v0 = _mm_unpacklo_epi8(A3, C3);
               ret.v1 = _mm_unpackhi_epi8(A3, C3);
               ret.v2 = _mm_unpacklo_epi8(B3, D3);
               ret.v3 = _mm_unpackhi_epi8(B3, D3);
               return ret;'''.format(**fmtspec)
    if typ in ['i16', 'u16']:
        return \
        '''nsimd_{simd_ext}_v{typ}x4 ret;
           {load_v0v1v2v3}
           __m128i E = _mm_unpacklo_epi16(v0,v1);
           __m128i F = _mm_unpackhi_epi16(v0,v1);
           __m128i G = _mm_unpacklo_epi16(v2,v3);
           __m128i H = _mm_unpackhi_epi16(v2,v3);

           __m128i I = _mm_unpacklo_epi16(E,F);
           __m128i J = _mm_unpackhi_epi16(E,F);
           __m128i K = _mm_unpacklo_epi16(G,H);
           __m128i L = _mm_unpackhi_epi16(G,H);

           ret.v0 = _mm_unpacklo_epi64(I,K);
           ret.v1 = _mm_unpackhi_epi64(I,K);
           ret.v2 = _mm_unpacklo_epi64(J,L);
           ret.v3 = _mm_unpackhi_epi64(J,L);
           return ret;'''.format(**fmtspec)
    if typ in ['f32', 'i32', 'u32']:
        return \
        '''nsimd_{simd_ext}_v{typ}x4 ret;
           {load_v0v1v2v3}
           {styp} A1 = _mm_unpacklo{suf}(v0, v2);
           {styp} B1 = _mm_unpackhi{suf}(v0, v2);
           {styp} C1 = _mm_unpacklo{suf}(v1, v3);
           {styp} D1 = _mm_unpackhi{suf}(v1, v3);

           ret.v0 = _mm_unpacklo{suf}(A1, C1);
           ret.v1 = _mm_unpackhi{suf}(A1, C1);
           ret.v2 = _mm_unpacklo{suf}(B1, D1);
           ret.v3 = _mm_unpackhi{suf}(B1, D1);

           return ret;'''.format(**fmtspec)
    if typ in ['f64', 'i64', 'u64']:
        return \
        '''nsimd_{simd_ext}_v{typ}x4 ret;
           {load_v0v1v2v3}
           ret.v0 = _mm_unpacklo{suf}(v0, v2);
           ret.v1 = _mm_unpackhi{suf}(v0, v2);
           ret.v2 = _mm_unpacklo{suf}(v1, v3);
           ret.v3 = _mm_unpackhi{suf}(v1, v3);
           return ret;'''.format(**fmtspec)

###############################################################################

def load4_avx(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['load_v0v1v2v3'] = get_load_v0v1v2v3('avx', typ, align, fmtspec)
    fmtspec['exlo_v0'] = x86.extract('avx', typ, x86.LO, 'v0')
    fmtspec['exhi_v0'] = x86.extract('avx', typ, x86.HI, 'v0')
    fmtspec['exlo_v1'] = x86.extract('avx', typ, x86.LO, 'v1')
    fmtspec['exhi_v1'] = x86.extract('avx', typ, x86.HI, 'v1')
    fmtspec['exlo_v2'] = x86.extract('avx', typ, x86.LO, 'v2')
    fmtspec['exhi_v2'] = x86.extract('avx', typ, x86.HI, 'v2')
    fmtspec['exlo_v3'] = x86.extract('avx', typ, x86.LO, 'v3')
    fmtspec['exhi_v3'] = x86.extract('avx', typ, x86.HI, 'v3')
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        if simd_ext == 'avx2':
            return \
            '''nsimd_avx2_v{typ}x4 ret;
               {load_v0v1v2v3}

               __m256i mask = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13,
                                               2, 6, 10, 14, 3, 7, 11, 15,
                                               0, 4, 8, 12, 1, 5, 9, 13, 2,
                                               6, 10, 14, 3, 7, 11, 15);
               __m256i mask2 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

               __m256i A1 = _mm256_shuffle_epi8(v0, mask);
               __m256i B1 = _mm256_shuffle_epi8(v1, mask);
               __m256i C1 = _mm256_shuffle_epi8(v2, mask);
               __m256i D1 = _mm256_shuffle_epi8(v3, mask);

               __m256i A2 = _mm256_permutevar8x32_epi32(A1, mask2);
               __m256i B2 = _mm256_permutevar8x32_epi32(B1, mask2);
               __m256i C2 = _mm256_permutevar8x32_epi32(C1, mask2);
               __m256i D2 = _mm256_permutevar8x32_epi32(D1, mask2);

               __m256i A3 = _mm256_permute2x128_si256(A2, C2, 2 << 4);
               __m256i C3 = _mm256_permute2x128_si256(B2, D2, 2 << 4);
               __m256i B3 = _mm256_permute2x128_si256(A2, C2, (3 << 4) | 1);
               __m256i D3 = _mm256_permute2x128_si256(B2, D2, (3 << 4) | 1);

               ret.v0 = _mm256_unpacklo_epi64(A3, C3);
               ret.v1 = _mm256_unpackhi_epi64(A3, C3);
               ret.v2 = _mm256_unpacklo_epi64(B3, D3);
               ret.v3 = _mm256_unpackhi_epi64(B3, D3);
               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_avx_v{typ}x4 ret;
               {load_v0v1v2v3}

               __m128i Aa = {exlo_v0};
               __m128i Ba = {exhi_v0};
               __m128i Ca = {exlo_v1};
               __m128i Da = {exhi_v1};
               __m128i Ab = {exlo_v2};
               __m128i Bb = {exhi_v2};
               __m128i Cb = {exlo_v3};
               __m128i Db = {exhi_v3};

               __m128i mask = _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2,
                                           13,  9, 5, 1, 12,  8, 4, 0);

               __m128d A1 = _mm_castsi128_pd(_mm_shuffle_epi8(Aa, mask));
               __m128d B1 = _mm_castsi128_pd(_mm_shuffle_epi8(Ba, mask));
               __m128d C1 = _mm_castsi128_pd(_mm_shuffle_epi8(Ca, mask));
               __m128d D1 = _mm_castsi128_pd(_mm_shuffle_epi8(Da, mask));

               __m128 A2 = _mm_castpd_ps(_mm_shuffle_pd(A1, B1,
                                                        _MM_SHUFFLE2(0, 0)));
               __m128 A3 = _mm_castpd_ps(_mm_shuffle_pd(C1, D1,
                                                        _MM_SHUFFLE2(0, 0)));
               __m128 C2 = _mm_castpd_ps(_mm_shuffle_pd(A1, B1,
                                                        _MM_SHUFFLE2(1, 1)));
               __m128 C3 = _mm_castpd_ps(_mm_shuffle_pd(C1, D1,
                                                        _MM_SHUFFLE2(1, 1)));

               __m128i Wa = _mm_castps_si128(_mm_shuffle_ps(A2, A3,
                                             _MM_SHUFFLE(2, 0, 2, 0)));
               __m128i Xa = _mm_castps_si128(_mm_shuffle_ps(A2, A3,
                                             _MM_SHUFFLE(3, 1, 3, 1)));
               __m128i Ya = _mm_castps_si128(_mm_shuffle_ps(C2, C3,
                                             _MM_SHUFFLE(2, 0, 2, 0)));
               __m128i Za = _mm_castps_si128(_mm_shuffle_ps(C2, C3,
                                             _MM_SHUFFLE(3, 1, 3, 1)));

               A1 = _mm_castsi128_pd(_mm_shuffle_epi8(Ab, mask));
               B1 = _mm_castsi128_pd(_mm_shuffle_epi8(Bb, mask));
               C1 = _mm_castsi128_pd(_mm_shuffle_epi8(Cb, mask));
               D1 = _mm_castsi128_pd(_mm_shuffle_epi8(Db, mask));

               A2 = _mm_castpd_ps(_mm_shuffle_pd(A1, B1, _MM_SHUFFLE2(0, 0)));
               A3 = _mm_castpd_ps(_mm_shuffle_pd(C1, D1, _MM_SHUFFLE2(0, 0)));
               C2 = _mm_castpd_ps(_mm_shuffle_pd(A1, B1, _MM_SHUFFLE2(1, 1)));
               C3 = _mm_castpd_ps(_mm_shuffle_pd(C1, D1, _MM_SHUFFLE2(1, 1)));

               __m128i Wb = _mm_castps_si128(_mm_shuffle_ps(A2, A3,
                                             _MM_SHUFFLE(2, 0, 2, 0)));
               __m128i Xb = _mm_castps_si128(_mm_shuffle_ps(A2, A3,
                                             _MM_SHUFFLE(3, 1, 3, 1)));
               __m128i Yb = _mm_castps_si128(_mm_shuffle_ps(C2, C3,
                                             _MM_SHUFFLE(2, 0, 2, 0)));
               __m128i Zb = _mm_castps_si128(_mm_shuffle_ps(C2, C3,
                                             _MM_SHUFFLE(3, 1, 3, 1)));

               ret.v0 = {mergeW};
               ret.v1 = {mergeX};
               ret.v2 = {mergeY};
               ret.v3 = {mergeZ};

               return ret;'''.format(mergeW=x86.setr('avx', typ, 'Wa', 'Wb'),
                                     mergeX=x86.setr('avx', typ, 'Xa', 'Xb'),
                                     mergeY=x86.setr('avx', typ, 'Ya', 'Yb'),
                                     mergeZ=x86.setr('avx', typ, 'Za', 'Zb'),
                                     **fmtspec)
    if typ in ['i16', 'u16']:
        if simd_ext == 'avx2':
            return \
            '''nsimd_avx2_v{typ}x4 ret;
               {load_v0v1v2v3}

               __m256i A1 = _mm256_unpacklo_epi16(v0, v2);
               __m256i B1 = _mm256_unpackhi_epi16(v0, v2);
               __m256i C1 = _mm256_unpacklo_epi16(v1, v3);
               __m256i D1 = _mm256_unpackhi_epi16(v1, v3);

               __m256i A2 = _mm256_unpacklo_epi16(A1, C1);
               __m256i B2 = _mm256_unpackhi_epi16(A1, C1);
               __m256i C2 = _mm256_unpacklo_epi16(B1, D1);
               __m256i D2 = _mm256_unpackhi_epi16(B1, D1);

               ret.v0 = _mm256_unpacklo_epi16(A2, C2);
               ret.v1 = _mm256_unpackhi_epi16(A2, C2);
               ret.v2 = _mm256_unpacklo_epi16(B2, D2);
               ret.v3 = _mm256_unpackhi_epi16(B2, D2);

               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_avx_v{typ}x4 ret;
               {load_v0v1v2v3}

               __m128i Aa = {exlo_v0};
               __m128i Ba = {exhi_v0};
               __m128i Ca = {exlo_v1};
               __m128i Da = {exhi_v1};
               __m128i Ab = {exlo_v2};
               __m128i Bb = {exhi_v2};
               __m128i Cb = {exlo_v3};
               __m128i Db = {exhi_v3};

               __m128i mask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4,
                                           11, 10, 3, 2,  9,  8, 1, 0);
               __m128d A1 = _mm_castsi128_pd(_mm_shuffle_epi8(Aa, mask));
               __m128d B1 = _mm_castsi128_pd(_mm_shuffle_epi8(Ba, mask));
               __m128d C1 = _mm_castsi128_pd(_mm_shuffle_epi8(Ca, mask));
               __m128d D1 = _mm_castsi128_pd(_mm_shuffle_epi8(Da, mask));

               __m128 A2 = _mm_castpd_ps(_mm_shuffle_pd(A1, B1,
                                                        _MM_SHUFFLE2(0, 0)));
               __m128 A3 = _mm_castpd_ps(_mm_shuffle_pd(C1, D1,
                                                        _MM_SHUFFLE2(0, 0)));
               __m128 C2 = _mm_castpd_ps(_mm_shuffle_pd(A1, B1,
                                                        _MM_SHUFFLE2(1, 1)));
               __m128 C3 = _mm_castpd_ps(_mm_shuffle_pd(C1, D1,
                                                        _MM_SHUFFLE2(1, 1)));

               __m128i Wa = _mm_castps_si128(_mm_shuffle_ps(A2, A3,
                                             _MM_SHUFFLE(2, 0, 2, 0)));
               __m128i Xa = _mm_castps_si128(_mm_shuffle_ps(A2, A3,
                                             _MM_SHUFFLE(3, 1, 3, 1)));
               __m128i Ya = _mm_castps_si128(_mm_shuffle_ps(C2, C3,
                                             _MM_SHUFFLE(2, 0, 2, 0)));
               __m128i Za = _mm_castps_si128(_mm_shuffle_ps(C2, C3,
                                             _MM_SHUFFLE(3, 1, 3, 1)));

               A1 = _mm_castsi128_pd(_mm_shuffle_epi8(Ab, mask));
               B1 = _mm_castsi128_pd(_mm_shuffle_epi8(Bb, mask));
               C1 = _mm_castsi128_pd(_mm_shuffle_epi8(Cb, mask));
               D1 = _mm_castsi128_pd(_mm_shuffle_epi8(Db, mask));

               A2 = _mm_castpd_ps(_mm_shuffle_pd(A1, B1, _MM_SHUFFLE2(0, 0)));
               A3 = _mm_castpd_ps(_mm_shuffle_pd(C1, D1, _MM_SHUFFLE2(0, 0)));
               C2 = _mm_castpd_ps(_mm_shuffle_pd(A1, B1, _MM_SHUFFLE2(1, 1)));
               C3 = _mm_castpd_ps(_mm_shuffle_pd(C1, D1, _MM_SHUFFLE2(1, 1)));

               __m128i Wb = _mm_castps_si128(_mm_shuffle_ps(A2, A3,
                                             _MM_SHUFFLE(2, 0, 2, 0)));
               __m128i Xb = _mm_castps_si128(_mm_shuffle_ps(A2, A3,
                                             _MM_SHUFFLE(3, 1, 3, 1)));
               __m128i Yb = _mm_castps_si128(_mm_shuffle_ps(C2, C3,
                                             _MM_SHUFFLE(2, 0, 2, 0)));
               __m128i Zb = _mm_castps_si128(_mm_shuffle_ps(C2, C3,
                                             _MM_SHUFFLE(3, 1, 3, 1)));

               ret.v0 = {mergeW};
               ret.v1 = {mergeX};
               ret.v2 = {mergeY};
               ret.v3 = {mergeZ};

               return ret;'''.format(mergeW=x86.setr('avx', typ, 'Wa', 'Wb'),
                                     mergeX=x86.setr('avx', typ, 'Xa', 'Xb'),
                                     mergeY=x86.setr('avx', typ, 'Ya', 'Yb'),
                                     mergeZ=x86.setr('avx', typ, 'Za', 'Zb'),
                                     **fmtspec)
    if typ == 'f32':
        return '''nsimd_{simd_ext}_vf32x4 ret;
                  {load_v0v1v2v3}
                  __m256 A1 = _mm256_unpacklo_ps(v0, v2);
                  __m256 B1 = _mm256_unpackhi_ps(v0, v2);
                  __m256 C1 = _mm256_unpacklo_ps(v1, v3);
                  __m256 D1 = _mm256_unpackhi_ps(v1, v3);

                  ret.v0 = _mm256_unpacklo_ps(A1, C1);
                  ret.v1 = _mm256_unpackhi_ps(A1, C1);
                  ret.v2 = _mm256_unpacklo_ps(B1, D1);
                  ret.v3 = _mm256_unpackhi_ps(B1, D1);
                  return ret;'''.format(**fmtspec)
    if typ in ['i32', 'u32']:
        if simd_ext == 'avx2':
            return \
            '''nsimd_avx2_v{typ}x4 ret;
               {load_v0v1v2v3}

               __m256i A1 = _mm256_unpacklo_epi32(v0, v2);
               __m256i B1 = _mm256_unpackhi_epi32(v0, v2);
               __m256i C1 = _mm256_unpacklo_epi32(v1, v3);
               __m256i D1 = _mm256_unpackhi_epi32(v1, v3);

               ret.v0 = _mm256_unpacklo_epi32(A1, C1);
               ret.v1 = _mm256_unpackhi_epi32(A1, C1);
               ret.v2 = _mm256_unpacklo_epi32(B1, D1);
               ret.v3 = _mm256_unpackhi_epi32(B1, D1);

               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_avx_v{typ}x4 ret;
               nsimd_avx_vf32x4 retf32 = nsimd_load4{a}_avx_f32((f32 *){in0});
               ret.v0 = _mm256_castps_si256(retf32.v0);
               ret.v1 = _mm256_castps_si256(retf32.v1);
               ret.v2 = _mm256_castps_si256(retf32.v2);
               ret.v3 = _mm256_castps_si256(retf32.v3);
               return ret;'''.format(**fmtspec)
    if typ == 'f64':
        return \
        '''nsimd_{simd_ext}_vf64x4 ret;
           {load_v0v1v2v3}

           __m256d A1 = _mm256_permute2f128_pd(v0, v2, 2 << 4);
           __m256d B1 = _mm256_permute2f128_pd(v0, v2, (3 << 4) | 1);
           __m256d C1 = _mm256_permute2f128_pd(v1, v3, 2 << 4);
           __m256d D1 = _mm256_permute2f128_pd(v1, v3, (3 << 4) | 1);

           ret.v0 = _mm256_unpacklo_pd(A1, C1);
           ret.v1 = _mm256_unpackhi_pd(A1, C1);
           ret.v2 = _mm256_unpacklo_pd(B1, D1);
           ret.v3 = _mm256_unpackhi_pd(B1, D1);
           return ret;'''.format(**fmtspec)
    if typ in ['i64', 'u64']:
        if simd_ext == 'avx2':
            return \
            '''nsimd_avx2_v{typ}x4 ret;
               {load_v0v1v2v3}

               __m256i A1 = _mm256_permute2f128_si256(v0, v2, 2 << 4);
               __m256i B1 = _mm256_permute2f128_si256(v0, v2, (3 << 4) | 1);
               __m256i C1 = _mm256_permute2f128_si256(v1, v3, 2 << 4);
               __m256i D1 = _mm256_permute2f128_si256(v1, v3, (3 << 4) | 1);

               ret.v0 = _mm256_unpacklo_epi64(A1, C1);
               ret.v1 = _mm256_unpackhi_epi64(A1, C1);
               ret.v2 = _mm256_unpacklo_epi64(B1, D1);
               ret.v3 = _mm256_unpackhi_epi64(B1, D1);

               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_avx_vf64x4 retf64 = nsimd_load4{a}_avx_f64((f64 *){in0});
               nsimd_avx_v{typ}x4 ret;
               ret.v0 = _mm256_castpd_si256(retf64.v0);
               ret.v1 = _mm256_castpd_si256(retf64.v1);
               ret.v2 = _mm256_castpd_si256(retf64.v2);
               ret.v3 = _mm256_castpd_si256(retf64.v3);
               return ret;'''.format(**fmtspec)

###############################################################################

def load4_avx512(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['load_v0v1v2v3'] = get_load_v0v1v2v3(simd_ext, typ, align, fmtspec)
    fmtspec['exlo_v0'] = x86.extract(simd_ext, typ, x86.LO, 'v0')
    fmtspec['exhi_v0'] = x86.extract(simd_ext, typ, x86.HI, 'v0')
    fmtspec['exlo_v1'] = x86.extract(simd_ext, typ, x86.LO, 'v1')
    fmtspec['exhi_v1'] = x86.extract(simd_ext, typ, x86.HI, 'v1')
    fmtspec['exlo_v2'] = x86.extract(simd_ext, typ, x86.LO, 'v2')
    fmtspec['exhi_v2'] = x86.extract(simd_ext, typ, x86.HI, 'v2')
    fmtspec['exlo_v3'] = x86.extract(simd_ext, typ, x86.LO, 'v3')
    fmtspec['exhi_v3'] = x86.extract(simd_ext, typ, x86.HI, 'v3')
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        return \
        '''nsimd_{simd_ext}_v{typ}x4 ret;

           {load_v0v1v2v3}

           __m256i A0a = {exlo_v0};
           __m256i B0a = {exhi_v0};
           __m256i C0a = {exlo_v1};
           __m256i D0a = {exhi_v1};
           __m256i A0b = {exlo_v2};
           __m256i B0b = {exhi_v2};
           __m256i C0b = {exlo_v3};
           __m256i D0b = {exhi_v3};

           __m256i mask = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13,
                                           2, 6, 10, 14, 3, 7, 11, 15,
                                           0, 4, 8, 12, 1, 5, 9, 13,
                                           2, 6, 10, 14, 3, 7, 11, 15);
           __m256i mask2 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

           __m256i A1 = _mm256_shuffle_epi8(A0a, mask);
           __m256i B1 = _mm256_shuffle_epi8(B0a, mask);
           __m256i C1 = _mm256_shuffle_epi8(C0a, mask);
           __m256i D1 = _mm256_shuffle_epi8(D0a, mask);

           __m256i A2 = _mm256_permutevar8x32_epi32(A1, mask2);
           __m256i B2 = _mm256_permutevar8x32_epi32(B1, mask2);
           __m256i C2 = _mm256_permutevar8x32_epi32(C1, mask2);
           __m256i D2 = _mm256_permutevar8x32_epi32(D1, mask2);

           __m256i A3 = _mm256_permute2x128_si256(A2, C2, 2 << 4);
           __m256i C3 = _mm256_permute2x128_si256(B2, D2, 2 << 4);
           __m256i B3 = _mm256_permute2x128_si256(A2, C2, (3 << 4) | 1);
           __m256i D3 = _mm256_permute2x128_si256(B2, D2, (3 << 4) | 1);

           __m256i A4a = _mm256_unpacklo_epi64(A3, C3);
           __m256i B4a = _mm256_unpackhi_epi64(A3, C3);
           __m256i C4a = _mm256_unpacklo_epi64(B3, D3);
           __m256i D4a = _mm256_unpackhi_epi64(B3, D3);

           A1 = _mm256_shuffle_epi8(A0b, mask);
           B1 = _mm256_shuffle_epi8(B0b, mask);
           C1 = _mm256_shuffle_epi8(C0b, mask);
           D1 = _mm256_shuffle_epi8(D0b, mask);

           A2 = _mm256_permutevar8x32_epi32(A1, mask2);
           B2 = _mm256_permutevar8x32_epi32(B1, mask2);
           C2 = _mm256_permutevar8x32_epi32(C1, mask2);
           D2 = _mm256_permutevar8x32_epi32(D1, mask2);

           A3 = _mm256_permute2x128_si256(A2, C2, 2 << 4);
           C3 = _mm256_permute2x128_si256(B2, D2, 2 << 4);
           B3 = _mm256_permute2x128_si256(A2, C2, (3 << 4) | 1);
           D3 = _mm256_permute2x128_si256(B2, D2, (3 << 4) | 1);

           __m256i A4b = _mm256_unpacklo_epi64(A3, C3);
           __m256i B4b = _mm256_unpackhi_epi64(A3, C3);
           __m256i C4b = _mm256_unpacklo_epi64(B3, D3);
           __m256i D4b = _mm256_unpackhi_epi64(B3, D3);

           ret.v0 = {mergeA};
           ret.v1 = {mergeB};
           ret.v2 = {mergeC};
           ret.v3 = {mergeD};

           return ret;'''.format(mergeA=x86.setr(simd_ext, typ, 'A4a', 'A4b'),
                                 mergeB=x86.setr(simd_ext, typ, 'B4a', 'B4b'),
                                 mergeC=x86.setr(simd_ext, typ, 'C4a', 'C4b'),
                                 mergeD=x86.setr(simd_ext, typ, 'D4a', 'D4b'),
                                 **fmtspec)
    if typ in ['i16', 'u16']:
        return \
        '''nsimd_{simd_ext}_v{typ}x4 ret;

           {load_v0v1v2v3}

           __m256i A0a = {exlo_v0};
           __m256i B0a = {exhi_v0};
           __m256i C0a = {exlo_v1};
           __m256i D0a = {exhi_v1};
           __m256i A0b = {exlo_v2};
           __m256i B0b = {exhi_v2};
           __m256i C0b = {exlo_v3};
           __m256i D0b = {exhi_v3};

           __m256i A1 = _mm256_unpacklo_epi16(A0a, C0a);
           __m256i B1 = _mm256_unpackhi_epi16(A0a, C0a);
           __m256i C1 = _mm256_unpacklo_epi16(B0a, D0a);
           __m256i D1 = _mm256_unpackhi_epi16(B0a, D0a);

           __m256i A2 = _mm256_unpacklo_epi16(A1, C1);
           __m256i B2 = _mm256_unpackhi_epi16(A1, C1);
           __m256i C2 = _mm256_unpacklo_epi16(B1, D1);
           __m256i D2 = _mm256_unpackhi_epi16(B1, D1);

           __m256i A3a = _mm256_unpacklo_epi16(A2, C2);
           __m256i B3a = _mm256_unpackhi_epi16(A2, C2);
           __m256i C3a = _mm256_unpacklo_epi16(B2, D2);
           __m256i D3a = _mm256_unpackhi_epi16(B2, D2);

           A1 = _mm256_unpacklo_epi16(A0b, C0b);
           B1 = _mm256_unpackhi_epi16(A0b, C0b);
           C1 = _mm256_unpacklo_epi16(B0b, D0b);
           D1 = _mm256_unpackhi_epi16(B0b, D0b);

           A2 = _mm256_unpacklo_epi16(A1, C1);
           B2 = _mm256_unpackhi_epi16(A1, C1);
           C2 = _mm256_unpacklo_epi16(B1, D1);
           D2 = _mm256_unpackhi_epi16(B1, D1);

           __m256i A3b = _mm256_unpacklo_epi16(A2, C2);
           __m256i B3b = _mm256_unpackhi_epi16(A2, C2);
           __m256i C3b = _mm256_unpacklo_epi16(B2, D2);
           __m256i D3b = _mm256_unpackhi_epi16(B2, D2);

           ret.v0 = {mergeA};
           ret.v1 = {mergeB};
           ret.v2 = {mergeC};
           ret.v3 = {mergeD};

           return ret;'''.format(mergeA=x86.setr(simd_ext, typ, 'A3a', 'A3b'),
                                 mergeB=x86.setr(simd_ext, typ, 'B3a', 'B3b'),
                                 mergeC=x86.setr(simd_ext, typ, 'C3a', 'C3b'),
                                 mergeD=x86.setr(simd_ext, typ, 'D3a', 'D3b'),
                                 **fmtspec)
    if typ in ['f32', 'i32', 'u32']:
        return \
        '''nsimd_{simd_ext}_v{typ}x4 ret;

           {load_v0v1v2v3}

           __m512i WXm = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28,
                                           1, 5, 9, 13, 17, 21, 25, 29);
           __m512i YZm = _mm512_setr_epi32(2, 6, 10, 14, 18, 22, 26, 30,
                                           3, 7, 11, 15, 19, 23, 27, 31);
           __m512i Wm = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                          16, 17, 18, 19, 20, 21, 22, 23);
           __m512i Xm = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15,
                                          24, 25, 26, 27, 28, 29, 30, 31);

           {styp} WXa = _mm512_permutex2var{suf}(v0, WXm, v1);
           {styp} WXb = _mm512_permutex2var{suf}(v2, WXm, v3);
           {styp} YZa = _mm512_permutex2var{suf}(v0, YZm, v1);
           {styp} YZb = _mm512_permutex2var{suf}(v2, YZm, v3);

           ret.v0 = _mm512_permutex2var{suf}(WXa, Wm, WXb);
           ret.v1 = _mm512_permutex2var{suf}(WXa, Xm, WXb);
           ret.v2 = _mm512_permutex2var{suf}(YZa, Wm, YZb);
           ret.v3 = _mm512_permutex2var{suf}(YZa, Xm, YZb);

           return ret;'''.format(**fmtspec)
    if typ in ['f64', 'i64', 'u64']:
        return \
        '''nsimd_{simd_ext}_v{typ}x4 ret;

           {load_v0v1v2v3}

           {styp} A1 = _mm512_unpacklo{suf}(v0, v1);
           {styp} B1 = _mm512_unpacklo{suf}(v2, v3);
           {styp} C1 = _mm512_unpackhi{suf}(v0, v1);
           {styp} D1 = _mm512_unpackhi{suf}(v2, v3);

           __m512i A_mask = _mm512_set_epi64(13, 9, 12, 8, 5, 1, 4, 0);
           __m512i B_mask = _mm512_set_epi64(15, 11, 14, 10, 7, 3, 6, 2);

           ret.v0 = _mm512_permutex2var{suf}(A1, A_mask, B1);
           ret.v1 = _mm512_permutex2var{suf}(C1, A_mask, D1);
           ret.v2 = _mm512_permutex2var{suf}(A1, B_mask, B1);
           ret.v3 = _mm512_permutex2var{suf}(C1, B_mask, D1);

           return ret;'''.format(**fmtspec)

###############################################################################

def store4(simd_ext, typ, align, fmtspec2, v0, v1, v2, v3):
    fmtspec = fmtspec2.copy()
    fmtspec['a'] = '' if align else 'u'
    store = '{pre}store{a}{sufsi}'.format(**fmtspec)
    fmtspec['store'] = store
    fmtspec['v0'] = v0
    fmtspec['v1'] = v1
    fmtspec['v2'] = v2
    fmtspec['v3'] = v3
    if typ in ['f32', 'f64']:
        return \
        '''{store}({in0}, {v0});
           {store}({in0} + {le}, {v1});
           {store}({in0} + (2 * {le}), {v2});
           {store}({in0} + (3 * {le}), {v3});'''.format(**fmtspec)
    else:
        return \
        '''{store}(({styp} *){in0}, {v0});
           {store}(({styp} *){in0} + 1, {v1});
           {store}(({styp} *){in0} + 2, {v2});
           {store}(({styp} *){in0} + 3, {v3});'''.format(**fmtspec)

###############################################################################

def store4_sse(typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    if typ in ['i8', 'u8']:
        return \
        '''__m128i A5 = _mm_unpacklo_epi8({in1}, {in3});
           __m128i B5 = _mm_unpackhi_epi8({in1}, {in3});
           __m128i C5 = _mm_unpacklo_epi8({in2}, {in4});
           __m128i D5 = _mm_unpackhi_epi8({in2}, {in4});

           __m128i A6 = _mm_unpacklo_epi8(A5, C5);
           __m128i B6 = _mm_unpackhi_epi8(A5, C5);
           __m128i C6 = _mm_unpacklo_epi8(B5, D5);
           __m128i D6 = _mm_unpackhi_epi8(B5, D5);

           {store}'''.format(store=store4('sse', typ, align, fmtspec,
                                          'A6', 'B6', 'C6', 'D6'), **fmtspec)
    if typ in ['i16', 'u16']:
        return \
        '''__m128i Q = _mm_unpacklo_epi16({in1}, {in2});
           __m128i R = _mm_unpackhi_epi16({in1}, {in2});
           __m128i S = _mm_unpacklo_epi16({in3}, {in4});
           __m128i T = _mm_unpackhi_epi16({in3}, {in4});

           __m128i U = _mm_unpacklo_epi32(Q, S);
           __m128i V = _mm_unpackhi_epi32(Q, S);
           __m128i W = _mm_unpacklo_epi32(R, T);
           __m128i X = _mm_unpackhi_epi32(R, T);

           {store}'''.format(store=store4('sse', typ, align, fmtspec,
                                          'U', 'V', 'W', 'X'), **fmtspec)
    if typ in ['f32', 'i32', 'u32']:
        return \
        '''{styp} A3 = _mm_unpacklo{suf}({in1}, {in3});
           {styp} B3 = _mm_unpackhi{suf}({in1}, {in3});
           {styp} C3 = _mm_unpacklo{suf}({in2}, {in4});
           {styp} D3 = _mm_unpackhi{suf}({in2}, {in4});

           {styp} A4 = _mm_unpacklo{suf}(A3, C3);
           {styp} B4 = _mm_unpackhi{suf}(A3, C3);
           {styp} C4 = _mm_unpacklo{suf}(B3, D3);
           {styp} D4 = _mm_unpackhi{suf}(B3, D3);

           {store}'''.format(store=store4('sse', typ, align, fmtspec,
                                          'A4', 'B4', 'C4', 'D4'), **fmtspec)
    if typ in ['f64', 'u64', 'i64']:
        return \
        '''{styp} A0 = _mm_unpacklo{suf}({in1}, {in2});
           {styp} B0 = _mm_unpacklo{suf}({in3}, {in4});
           {styp} C0 = _mm_unpackhi{suf}({in1}, {in2});
           {styp} D0 = _mm_unpackhi{suf}({in3}, {in4});
           {store}'''.format(store=store4('sse', typ, align, fmtspec,
                                          'A0', 'B0', 'C0', 'D0'), **fmtspec)

###############################################################################

def store4_avx(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['exlo_in1'] = x86.extract('avx', typ, x86.LO, common.in1)
    fmtspec['exhi_in1'] = x86.extract('avx', typ, x86.HI, common.in1)
    fmtspec['exlo_in2'] = x86.extract('avx', typ, x86.LO, common.in2)
    fmtspec['exhi_in2'] = x86.extract('avx', typ, x86.HI, common.in2)
    fmtspec['exlo_in3'] = x86.extract('avx', typ, x86.LO, common.in3)
    fmtspec['exhi_in3'] = x86.extract('avx', typ, x86.HI, common.in3)
    fmtspec['exlo_in4'] = x86.extract('avx', typ, x86.LO, common.in4)
    fmtspec['exhi_in4'] = x86.extract('avx', typ, x86.HI, common.in4)
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        if simd_ext == 'avx2':
            return \
            '''__m256i A1 = _mm256_unpacklo_epi8({in1}, {in3});
               __m256i B1 = _mm256_unpackhi_epi8({in1}, {in3});
               __m256i C1 = _mm256_unpacklo_epi8({in2}, {in4});
               __m256i D1 = _mm256_unpackhi_epi8({in2}, {in4});

               __m256i A2 = _mm256_permute4x64_epi64(A1, _MM_SHUFFLE(3,1,2,0));
               __m256i B2 = _mm256_permute4x64_epi64(B1, _MM_SHUFFLE(3,1,2,0));
               __m256i C2 = _mm256_permute4x64_epi64(C1, _MM_SHUFFLE(3,1,2,0));
               __m256i D2 = _mm256_permute4x64_epi64(D1, _MM_SHUFFLE(3,1,2,0));

               __m256i A = _mm256_unpacklo_epi8(A2, C2);
               __m256i B = _mm256_unpacklo_epi8(B2, D2);
               __m256i C = _mm256_unpackhi_epi8(A2, C2);
               __m256i D = _mm256_unpackhi_epi8(B2, D2);

               {store}'''.format(store=store4('avx', typ, align, fmtspec,
                                 'A', 'B', 'C', 'D'), **fmtspec)
        else:
            return \
            '''__m128i Wa = {exlo_in1};
               __m128i Wb = {exhi_in1};
               __m128i Xa = {exlo_in2};
               __m128i Xb = {exhi_in2};
               __m128i Ya = {exlo_in3};
               __m128i Yb = {exhi_in3};
               __m128i Za = {exlo_in4};
               __m128i Zb = {exhi_in4};

               __m128i AA = _mm_unpacklo_epi8(Wa, Ya);
               __m128i BB = _mm_unpackhi_epi8(Wa, Ya);
               __m128i CC = _mm_unpacklo_epi8(Xa, Za);
               __m128i DD = _mm_unpackhi_epi8(Xa, Za);

               __m128i A0 = _mm_unpacklo_epi8(AA, CC);
               __m128i B0 = _mm_unpackhi_epi8(AA, CC);
               __m128i C0 = _mm_unpacklo_epi8(BB, DD);
               __m128i D0 = _mm_unpackhi_epi8(BB, DD);

               AA = _mm_unpacklo_epi8(Wb, Yb);
               BB = _mm_unpackhi_epi8(Wb, Yb);
               CC = _mm_unpacklo_epi8(Xb, Zb);
               DD = _mm_unpackhi_epi8(Xb, Zb);

               __m128i A1 = _mm_unpacklo_epi8(AA, CC);
               __m128i B1 = _mm_unpackhi_epi8(AA, CC);
               __m128i C1 = _mm_unpacklo_epi8(BB, DD);
               __m128i D1 = _mm_unpackhi_epi8(BB, DD);

               __m256i A = {mergeAB0};
               __m256i B = {mergeCD0};
               __m256i C = {mergeAB1};
               __m256i D = {mergeCD1};

               {store}'''.format(mergeAB0=x86.setr('avx', typ, 'A0', 'B0'),
                                 mergeCD0=x86.setr('avx', typ, 'C0', 'D0'),
                                 mergeAB1=x86.setr('avx', typ, 'A1', 'B1'),
                                 mergeCD1=x86.setr('avx', typ, 'C1', 'D1'),
                                 store=store4('avx', typ, align, fmtspec,
                                         'A', 'B', 'C', 'D'), **fmtspec)
    if typ in ['i16', 'u16']:
        if simd_ext == 'avx2':
            return \
            '''__m256i A3 = _mm256_unpacklo_epi16({in1}, {in3});
               __m256i B3 = _mm256_unpackhi_epi16({in1}, {in3});
               __m256i C3 = _mm256_unpacklo_epi16({in2}, {in4});
               __m256i D3 = _mm256_unpackhi_epi16({in2}, {in4});

               __m256i A = _mm256_unpacklo_epi16(A3, C3);
               __m256i B = _mm256_unpackhi_epi16(A3, C3);
               __m256i C = _mm256_unpacklo_epi16(B3, D3);
               __m256i D = _mm256_unpackhi_epi16(B3, D3);

               {store}'''.format(store=store4('avx', typ, align, fmtspec,
                                              'A', 'B', 'C', 'D'), **fmtspec)
        else:
            return \
            '''__m128i Wa = {exlo_in1};
               __m128i Wb = {exhi_in1};
               __m128i Xa = {exlo_in2};
               __m128i Xb = {exhi_in2};
               __m128i Ya = {exlo_in3};
               __m128i Yb = {exhi_in3};
               __m128i Za = {exlo_in4};
               __m128i Zb = {exhi_in4};

               __m128i AA = _mm_unpacklo_epi16(Wa, Ya);
               __m128i BB = _mm_unpackhi_epi16(Wa, Ya);
               __m128i CC = _mm_unpacklo_epi16(Xa, Za);
               __m128i DD = _mm_unpackhi_epi16(Xa, Za);

               __m128i A0 = _mm_unpacklo_epi16(AA, CC);
               __m128i B0 = _mm_unpackhi_epi16(AA, CC);
               __m128i C0 = _mm_unpacklo_epi16(BB, DD);
               __m128i D0 = _mm_unpackhi_epi16(BB, DD);

               AA = _mm_unpacklo_epi16(Wb, Yb);
               BB = _mm_unpackhi_epi16(Wb, Yb);
               CC = _mm_unpacklo_epi16(Xb, Zb);
               DD = _mm_unpackhi_epi16(Xb, Zb);

               __m128i A1 = _mm_unpacklo_epi16(AA, CC);
               __m128i B1 = _mm_unpackhi_epi16(AA, CC);
               __m128i C1 = _mm_unpacklo_epi16(BB, DD);
               __m128i D1 = _mm_unpackhi_epi16(BB, DD);

               __m256i A = {mergeAB0};
               __m256i B = {mergeCD0};
               __m256i C = {mergeAB1};
               __m256i D = {mergeCD1};

               {store}'''.format(mergeAB0=x86.setr('avx', typ, 'A0', 'B0'),
                                 mergeCD0=x86.setr('avx', typ, 'C0', 'D0'),
                                 mergeAB1=x86.setr('avx', typ, 'A1', 'B1'),
                                 mergeCD1=x86.setr('avx', typ, 'C1', 'D1'),
                                 store=store4('avx', typ, align, fmtspec,
                                              'A', 'B', 'C', 'D'), **fmtspec)
    if typ == 'f32':
        return \
        '''__m256 A3 = _mm256_unpacklo_ps({in1}, {in3});
           __m256 B3 = _mm256_unpackhi_ps({in1}, {in3});
           __m256 C3 = _mm256_unpacklo_ps({in2}, {in4});
           __m256 D3 = _mm256_unpackhi_ps({in2}, {in4});

           __m256 A = _mm256_unpacklo_ps(A3, C3);
           __m256 B = _mm256_unpackhi_ps(A3, C3);
           __m256 C = _mm256_unpacklo_ps(B3, D3);
           __m256 D = _mm256_unpackhi_ps(B3, D3);

           {store}'''.format(store=store4('avx', typ, align, fmtspec,
                                          'A', 'B', 'C', 'D'), **fmtspec)
    if typ in ['i32', 'u32']:
        if simd_ext == 'avx2':
            return \
            '''__m256i A3 = _mm256_unpacklo_epi32({in1}, {in3});
               __m256i B3 = _mm256_unpackhi_epi32({in1}, {in3});
               __m256i C3 = _mm256_unpacklo_epi32({in2}, {in4});
               __m256i D3 = _mm256_unpackhi_epi32({in2}, {in4});

               __m256i A = _mm256_unpacklo_epi32(A3, C3);
               __m256i B = _mm256_unpackhi_epi32(A3, C3);
               __m256i C = _mm256_unpacklo_epi32(B3, D3);
               __m256i D = _mm256_unpackhi_epi32(B3, D3);

               {store}'''.format(store=store4('avx', typ, align, fmtspec,
                                 'A', 'B', 'C', 'D'), **fmtspec)
        else:
            return \
            '''nsimd_store4{a}_avx_f32((f32 *){in0},
                                       _mm256_castsi256_ps({in1}),
                                       _mm256_castsi256_ps({in2}),
                                       _mm256_castsi256_ps({in3}),
                                       _mm256_castsi256_ps({in4}));'''. \
               format(**fmtspec)
    if typ == 'f64':
        return \
        '''__m256d A3 = _mm256_permute2f128_pd({in1}, {in3}, 2 << 4);
           __m256d B3 = _mm256_permute2f128_pd({in2}, {in4}, 2 << 4);
           __m256d C3 = _mm256_permute2f128_pd({in1}, {in3}, (3 << 4) | 1);
           __m256d D3 = _mm256_permute2f128_pd({in2}, {in4}, (3 << 4) | 1);

           __m256d A = _mm256_unpacklo_pd(A3, B3);
           __m256d B = _mm256_unpackhi_pd(A3, B3);
           __m256d C = _mm256_unpacklo_pd(C3, D3);
           __m256d D = _mm256_unpackhi_pd(C3, D3);

           {store}'''.format(store=store4('avx', typ, align, fmtspec,
                                          'A', 'B', 'C', 'D'), **fmtspec)

    if typ in ['i64', 'u64']:
        if simd_ext == 'avx2':
            return \
            '''__m256i A3 = _mm256_permute2f128_si256({in1}, {in3}, 2 << 4);
               __m256i B3 = _mm256_permute2f128_si256({in2}, {in4}, 2 << 4);
               __m256i C3 = _mm256_permute2f128_si256(
                              {in1}, {in3}, (3 << 4) | 1);
               __m256i D3 = _mm256_permute2f128_si256(
                              {in2}, {in4}, (3 << 4) | 1);

               __m256i A = _mm256_unpacklo_epi64(A3, B3);
               __m256i B = _mm256_unpackhi_epi64(A3, B3);
               __m256i C = _mm256_unpacklo_epi64(C3, D3);
               __m256i D = _mm256_unpackhi_epi64(C3, D3);

               {store}'''.format(store=store4('avx', typ, align, fmtspec,
                                 'A', 'B', 'C', 'D'), **fmtspec)
        else:
            return \
            '''nsimd_store4{a}_avx_f64((f64 *){in0},
                                       _mm256_castsi256_pd({in1}),
                                       _mm256_castsi256_pd({in2}),
                                       _mm256_castsi256_pd({in3}),
                                       _mm256_castsi256_pd({in4}));'''. \
               format(**fmtspec)

###############################################################################

def store4_avx512(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['exlo_in1'] = x86.extract(simd_ext, typ, x86.LO, common.in1)
    fmtspec['exhi_in1'] = x86.extract(simd_ext, typ, x86.HI, common.in1)
    fmtspec['exlo_in2'] = x86.extract(simd_ext, typ, x86.LO, common.in2)
    fmtspec['exhi_in2'] = x86.extract(simd_ext, typ, x86.HI, common.in2)
    fmtspec['exlo_in3'] = x86.extract(simd_ext, typ, x86.LO, common.in3)
    fmtspec['exhi_in3'] = x86.extract(simd_ext, typ, x86.HI, common.in3)
    fmtspec['exlo_in4'] = x86.extract(simd_ext, typ, x86.LO, common.in4)
    fmtspec['exhi_in4'] = x86.extract(simd_ext, typ, x86.HI, common.in4)
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        return \
        '''__m256i A0a = {exlo_in1};
           __m256i A0b = {exhi_in1};
           __m256i B0a = {exlo_in2};
           __m256i B0b = {exhi_in2};
           __m256i C0a = {exlo_in3};
           __m256i C0b = {exhi_in3};
           __m256i D0a = {exlo_in4};
           __m256i D0b = {exhi_in4};

           __m256i A1 = _mm256_unpacklo_epi8(A0a, C0a);
           __m256i B1 = _mm256_unpackhi_epi8(A0a, C0a);
           __m256i C1 = _mm256_unpacklo_epi8(B0a, D0a);
           __m256i D1 = _mm256_unpackhi_epi8(B0a, D0a);

           __m256i A2 = _mm256_permute4x64_epi64(A1, _MM_SHUFFLE(3,1,2,0));
           __m256i B2 = _mm256_permute4x64_epi64(B1, _MM_SHUFFLE(3,1,2,0));
           __m256i C2 = _mm256_permute4x64_epi64(C1, _MM_SHUFFLE(3,1,2,0));
           __m256i D2 = _mm256_permute4x64_epi64(D1, _MM_SHUFFLE(3,1,2,0));

           __m256i A3a = _mm256_unpacklo_epi8(A2, C2);
           __m256i B3a = _mm256_unpacklo_epi8(B2, D2);
           __m256i C3a = _mm256_unpackhi_epi8(A2, C2);
           __m256i D3a = _mm256_unpackhi_epi8(B2, D2);

           A1 = _mm256_unpacklo_epi8(A0b, C0b);
           B1 = _mm256_unpackhi_epi8(A0b, C0b);
           C1 = _mm256_unpacklo_epi8(B0b, D0b);
           D1 = _mm256_unpackhi_epi8(B0b, D0b);

           A2 = _mm256_permute4x64_epi64(A1, _MM_SHUFFLE(3,1,2,0));
           B2 = _mm256_permute4x64_epi64(B1, _MM_SHUFFLE(3,1,2,0));
           C2 = _mm256_permute4x64_epi64(C1, _MM_SHUFFLE(3,1,2,0));
           D2 = _mm256_permute4x64_epi64(D1, _MM_SHUFFLE(3,1,2,0));

           __m256i A3b = _mm256_unpacklo_epi8(A2, C2);
           __m256i B3b = _mm256_unpacklo_epi8(B2, D2);
           __m256i C3b = _mm256_unpackhi_epi8(A2, C2);
           __m256i D3b = _mm256_unpackhi_epi8(B2, D2);

           __m512i A = {mergeABa};
           __m512i B = {mergeCDa};
           __m512i C = {mergeABb};
           __m512i D = {mergeCDb};

           {store}'''.format(mergeABa=x86.setr(simd_ext, typ, 'A3a', 'B3a'),
                             mergeCDa=x86.setr(simd_ext, typ, 'C3a', 'D3a'),
                             mergeABb=x86.setr(simd_ext, typ, 'A3b', 'B3b'),
                             mergeCDb=x86.setr(simd_ext, typ, 'C3b', 'D3b'),
                             store=store4(simd_ext, typ, align, fmtspec,
                                          'A', 'B', 'C', 'D'), **fmtspec)
    if typ in ['i16', 'u16']:
        return \
        '''__m256i A0a = {exlo_in1};
           __m256i A0b = {exhi_in1};
           __m256i B0a = {exlo_in2};
           __m256i B0b = {exhi_in2};
           __m256i C0a = {exlo_in3};
           __m256i C0b = {exhi_in3};
           __m256i D0a = {exlo_in4};
           __m256i D0b = {exhi_in4};

           __m256i A3 = _mm256_unpacklo_epi16(A0a, C0a);
           __m256i B3 = _mm256_unpackhi_epi16(A0a, C0a);
           __m256i C3 = _mm256_unpacklo_epi16(B0a, D0a);
           __m256i D3 = _mm256_unpackhi_epi16(B0a, D0a);

           __m256i A4a = _mm256_unpacklo_epi16(A3, C3);
           __m256i B4a = _mm256_unpackhi_epi16(A3, C3);
           __m256i C4a = _mm256_unpacklo_epi16(B3, D3);
           __m256i D4a = _mm256_unpackhi_epi16(B3, D3);

           A3 = _mm256_unpacklo_epi16(A0b, C0b);
           B3 = _mm256_unpackhi_epi16(A0b, C0b);
           C3 = _mm256_unpacklo_epi16(B0b, D0b);
           D3 = _mm256_unpackhi_epi16(B0b, D0b);

           __m256i A4b = _mm256_unpacklo_epi16(A3, C3);
           __m256i B4b = _mm256_unpackhi_epi16(A3, C3);
           __m256i C4b = _mm256_unpacklo_epi16(B3, D3);
           __m256i D4b = _mm256_unpackhi_epi16(B3, D3);

           __m512i A = {mergeABa};
           __m512i B = {mergeCDa};
           __m512i C = {mergeABb};
           __m512i D = {mergeCDb};

           {store}'''.format(mergeABa=x86.setr(simd_ext, typ, 'A4a', 'B4a'),
                             mergeCDa=x86.setr(simd_ext, typ, 'C4a', 'D4a'),
                             mergeABb=x86.setr(simd_ext, typ, 'A4b', 'B4b'),
                             mergeCDb=x86.setr(simd_ext, typ, 'C4b', 'D4b'),
                             store=store4(simd_ext, typ, align, fmtspec,
                                          'A', 'B', 'C', 'D'), **fmtspec)
    if typ in ['f32', 'i32', 'u32']:
        return \
        '''__m512i m1 = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19,
                                          4, 5, 6, 7, 20, 21, 22, 23);
           __m512i m2 = _mm512_setr_epi32(8, 9, 10, 11, 24, 25, 26, 27,
                                          12, 13, 14, 15, 28, 29, 30, 31);
           __m512i m3 = _mm512_setr_epi32(0, 4, 16, 20, 1, 5, 17, 21,
                                          2, 6, 18, 22, 3, 7, 19, 23);
           __m512i m4 = _mm512_setr_epi32(8, 12, 24, 28, 9, 13, 25, 29,
                                          10, 14, 26, 30, 11, 15, 27, 31);

           {styp} WXa = _mm512_permutex2var{suf}({in1}, m1, {in2});
           {styp} WXb = _mm512_permutex2var{suf}({in1}, m2, {in2});
           {styp} YZa = _mm512_permutex2var{suf}({in3}, m1, {in4});
           {styp} YZb = _mm512_permutex2var{suf}({in3}, m2, {in4});

           {styp} A = _mm512_permutex2var{suf}(WXa, m3, YZa);
           {styp} B = _mm512_permutex2var{suf}(WXa, m4, YZa);
           {styp} C = _mm512_permutex2var{suf}(WXb, m3, YZb);
           {styp} D = _mm512_permutex2var{suf}(WXb, m4, YZb);

           {store}'''.format(store=store4(simd_ext, typ, align, fmtspec,
                                          'A', 'B', 'C', 'D'), **fmtspec)
    if typ in ['f64', 'i64', 'u64']:
        return \
        '''__m512i A_mask = _mm512_setr_epi64(0, 1,  2,  3,  8,  9, 10, 11);
           __m512i B_mask = _mm512_setr_epi64(4, 5,  6,  7, 12, 13, 14, 15);
           __m512i C_mask = _mm512_setr_epi64(0, 4,  8, 12,  1,  5,  9, 13);
           __m512i D_mask = _mm512_setr_epi64(2, 6, 10, 14,  3,  7, 11, 15);

           {styp} A1 = _mm512_permutex2var{suf}({in1}, A_mask, {in2});
           {styp} B1 = _mm512_permutex2var{suf}({in1}, B_mask, {in2});
           {styp} C1 = _mm512_permutex2var{suf}({in3}, A_mask, {in4});
           {styp} D1 = _mm512_permutex2var{suf}({in3}, B_mask, {in4});

           {styp} A = _mm512_permutex2var{suf}(A1, C_mask, C1);
           {styp} B = _mm512_permutex2var{suf}(A1, D_mask, C1);
           {styp} C = _mm512_permutex2var{suf}(B1, C_mask, D1);
           {styp} D = _mm512_permutex2var{suf}(B1, D_mask, D1);

           {store}'''.format(store=store4(simd_ext, typ, align, fmtspec,
                                          'A', 'B', 'C', 'D'), **fmtspec)

###############################################################################

def get_load_v0v1v2(simd_ext, typ, align, fmtspec):
    load = '{pre}load{a}{sufsi}'.format(a='' if align else 'u', **fmtspec)
    if typ in ['f32', 'f64']:
        return '''{styp} v0 = {load}(a0);
                  {styp} v1 = {load}(a0 + {le});
                  {styp} v2 = {load}(a0 + (2 * {le}));'''. \
                  format(load=load, **fmtspec)
    else:
        return '''{styp} v0 = {load}(({styp}*)a0);
                  {styp} v1 = {load}(({styp}*)a0 + 1);
                  {styp} v2 = {load}(({styp}*)a0 + 2);'''. \
                  format(load=load, **fmtspec)

###############################################################################

def load3_sse(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['load_v0v1v2'] = get_load_v0v1v2('sse', typ, align, fmtspec)
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        if simd_ext == 'sse42':
            return \
            '''nsimd_sse42_v{typ}x3 ret;
               {load_v0v1v2}

               __m128i A1_mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                              -1, -1, 15, 12,  9,  6,  3,  0);
               __m128i A2_mask = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11,  8,
                                               5,  2, -1, -1, -1, -1, -1, -1);
               __m128i A3_mask = _mm_set_epi8(13, 10,  7,  4,  1, -1, -1, -1,
                                              -1, -1, -1, -1, -1, -1, -1, -1);
               __m128i A4 = _mm_shuffle_epi8(v0, A1_mask);
               __m128i A5 = _mm_shuffle_epi8(v1, A2_mask);
               __m128i A6 = _mm_shuffle_epi8(v2, A3_mask);
               A4 = _mm_or_si128(A4, A5);
               ret.v0 = _mm_or_si128(A4, A6);

               __m128i B1_mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                              -1, -1, -1, 13, 10,  7,  4,  1);
               __m128i B2_mask = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12,  9,
                                               6,  3,  0, -1, -1, -1, -1, -1);
               __m128i B3_mask = _mm_set_epi8(14, 11,  8,  5,  2, -1, -1, -1,
                                              -1, -1, -1, -1, -1, -1, -1, -1);
               __m128i B4 = _mm_shuffle_epi8(v0, B1_mask);
               __m128i B5 = _mm_shuffle_epi8(v1, B2_mask);
               __m128i B6 = _mm_shuffle_epi8(v2, B3_mask);
               B4 = _mm_or_si128(B4, B5);
               ret.v1 = _mm_or_si128(B4, B6);

               __m128i C1_mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                              -1, -1, -1, 14, 11,  8,  5,  2);
               __m128i C2_mask = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10,
                                               7,  4,  1, -1, -1, -1, -1, -1);
               __m128i C3_mask = _mm_set_epi8(15, 12,  9,  6,  3,  0, -1, -1,
                                              -1, -1, -1, -1, -1, -1, -1, -1);
               __m128i C4 = _mm_shuffle_epi8(v0, C1_mask);
               __m128i C5 = _mm_shuffle_epi8(v1, C2_mask);
               __m128i C6 = _mm_shuffle_epi8(v2, C3_mask);
               C4 = _mm_or_si128(C4, C5);
               ret.v2 = _mm_or_si128(C4, C6);

               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_sse2_v{typ}x3 ret;
               {load_v0v1v2}

               __m128i A0 = v0;
               __m128i B0 = v1;
               __m128i C0 = v2;
               int k;

               for (k = 0; k < 4; ++k) {{
                 __m128d B0_pd = _mm_castsi128_pd(B0);
                 __m128d C0_pd = _mm_castsi128_pd(C0);

                 __m128d B1_pd = _mm_shuffle_pd(B0_pd, B0_pd, 1);
                 __m128d C2_pd = _mm_shuffle_pd(C0_pd, C0_pd, 1);

                 __m128i B1 = _mm_castpd_si128(B1_pd);
                 __m128i C2 = _mm_castpd_si128(C2_pd);

                 __m128i B3 = _mm_unpackhi_epi8(A0, C2);
                 __m128i A4 = _mm_unpacklo_epi8(A0, B1);
                 __m128i C5 = _mm_unpackhi_epi8(B1, C0);
                 A0 = A4;
                 B0 = B3;
                 C0 = C5;
               }}
               ret.v0 = A0;
               ret.v1 = B0;
               ret.v2 = C0;
               return ret;'''.format(**fmtspec)
    if typ in ['i16', 'u16']:
        return \
        '''nsimd_{simd_ext}_v{typ}x3 ret;

           {load_v0v1v2}

           int k;

           for (k = 0; k < 3; ++k) {{
             __m128d B1_pd = _mm_castsi128_pd(v1);
             __m128d C1_pd = _mm_castsi128_pd(v2);
             __m128d B2_pd = _mm_shuffle_pd(B1_pd, B1_pd, 1);
             __m128d C3_pd = _mm_shuffle_pd(C1_pd, C1_pd, 1);
             __m128i B2 = _mm_castpd_si128(B2_pd);
             __m128i C3 = _mm_castpd_si128(C3_pd);

             __m128i B4 = _mm_unpackhi_epi16(v0, C3);
             __m128i A5 = _mm_unpacklo_epi16(v0, B2);
             __m128i C7 = _mm_unpackhi_epi16(B2, v2);

             v0 = A5;
             v1 = B4;
             v2 = C7;
           }}
           ret.v0 = v0;
           ret.v1 = v1;
           ret.v2 = v2;
           return ret;'''.format(**fmtspec)
    if typ == 'f32':
        return \
        '''nsimd_{simd_ext}_v{typ}x3 ret;
           {load_v0v1v2}

           __m128 A1 = _mm_shuffle_ps(v2, v1, _MM_SHUFFLE(3,2,1,0));
           __m128 B2 = _mm_shuffle_ps(v0, v2, _MM_SHUFFLE(3,2,1,0));
           __m128 C3 = _mm_shuffle_ps(v1, v0, _MM_SHUFFLE(3,2,1,0));

           ret.v0 = _mm_shuffle_ps(v0, A1, _MM_SHUFFLE(1,2,3,0));
           __m128 B5 = _mm_shuffle_ps(B2, v1, _MM_SHUFFLE(0,3,2,1));
           ret.v2 = _mm_shuffle_ps(C3, v2, _MM_SHUFFLE(3,0,1,2));

           ret.v1 = _mm_shuffle_ps(B5, B5, _MM_SHUFFLE(1,2,3,0));

           return ret;'''.format(**fmtspec)
    if typ in ['i32', 'u32']:
        return \
        '''nsimd_{simd_ext}_v{typ}x3 ret;
           nsimd_{simd_ext}_vf32x3 retf32 =
               nsimd_load3{a}_{simd_ext}_f32((f32 *){in0});
           ret.v0 = _mm_castps_si128(retf32.v0);
           ret.v1 = _mm_castps_si128(retf32.v1);
           ret.v2 = _mm_castps_si128(retf32.v2);
           return ret;'''.format(**fmtspec)
    if typ == 'f64':
        return \
        '''nsimd_{simd_ext}_vf64x3 ret;
           {load_v0v1v2}
           ret.v0 = _mm_shuffle_pd(v0, v1, 2);
           ret.v1 = _mm_shuffle_pd(v0, v2, 1);
           ret.v2 = _mm_shuffle_pd(v1, v2, 2);
           return ret;'''.format(**fmtspec)
    if typ in ['i64', 'u64']:
        return \
        '''nsimd_{simd_ext}_v{typ}x3 ret;
           nsimd_{simd_ext}_vf64x3 retf64 =
               nsimd_load3{a}_{simd_ext}_f64((f64 *){in0});
           ret.v0 = _mm_castpd_si128(retf64.v0);
           ret.v1 = _mm_castpd_si128(retf64.v1);
           ret.v2 = _mm_castpd_si128(retf64.v2);
           return ret;'''.format(**fmtspec)

###############################################################################

def store3(simd_ext, typ, align, fmtspec2, v0, v1, v2):
    fmtspec = fmtspec2.copy()
    fmtspec['a'] = '' if align else 'u'
    store = '{pre}store{a}{sufsi}'.format(**fmtspec)
    fmtspec['store'] = store
    fmtspec['v0'] = v0
    fmtspec['v1'] = v1
    fmtspec['v2'] = v2
    if typ in ['f32', 'f64']:
        return \
        '''{store}({in0}, {v0});
           {store}({in0} + {le}, {v1});
           {store}({in0} + (2 * {le}), {v2});'''.format(**fmtspec)
    else:
        return \
        '''{store}(({styp} *){in0}, {v0});
           {store}(({styp} *){in0} + 1, {v1});
           {store}(({styp} *){in0} + 2, {v2});'''.format(**fmtspec)

###############################################################################

def store3_sse(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        if simd_ext == 'sse42':
            return \
            '''__m128i A1_mask = _mm_set_epi8( 5, -1, -1,  4, -1, -1,  3, -1,
                                              -1,  2, -1, -1,  1, -1, -1,  0);
               __m128i A2_mask = _mm_set_epi8(-1, -1,  4, -1, -1,  3, -1, -1,
                                               2, -1, -1,  1, -1, -1,  0, -1);
               __m128i A3_mask = _mm_set_epi8(-1,  4, -1, -1,  3, -1, -1,  2,
                                              -1, -1,  1, -1, -1,  0, -1, -1);
               __m128i A4 = _mm_shuffle_epi8({in1}, A1_mask);
               __m128i A5 = _mm_shuffle_epi8({in2}, A2_mask);
               __m128i A6 = _mm_shuffle_epi8({in3}, A3_mask);
               A4 = _mm_or_si128(A4, A5);
               A4 = _mm_or_si128(A4, A6);

               __m128i B1_mask = _mm_set_epi8(-1, 10, -1, -1,  9, -1, -1,  8,
                                              -1, -1,  7, -1, -1,  6, -1, -1);
               __m128i B2_mask = _mm_set_epi8(10, -1, -1,  9, -1, -1,  8, -1,
                                              -1,  7, -1, -1,  6, -1, -1,  5);
               __m128i B3_mask = _mm_set_epi8(-1, -1,  9, -1, -1,  8, -1, -1,
                                               7, -1, -1,  6, -1, -1,  5, -1);
               __m128i B4 = _mm_shuffle_epi8({in1}, B1_mask);
               __m128i B5 = _mm_shuffle_epi8({in2}, B2_mask);
               __m128i B6 = _mm_shuffle_epi8({in3}, B3_mask);
               B4 = _mm_or_si128(B4, B5);
               B4 = _mm_or_si128(B4, B6);

               __m128i C1_mask = _mm_set_epi8(-1, -1, 15, -1, -1, 14, -1, -1,
                                              13, -1, -1, 12, -1, -1, 11, -1);
               __m128i C2_mask = _mm_set_epi8(-1, 15, -1, -1, 14, -1, -1, 13,
                                              -1, -1, 12, -1, -1, 11, -1, -1);
               __m128i C3_mask = _mm_set_epi8(15, -1, -1, 14, -1, -1, 13, -1,
                                              -1, 12, -1, -1, 11, -1, -1, 10);
               __m128i C4 = _mm_shuffle_epi8({in1}, C1_mask);
               __m128i C5 = _mm_shuffle_epi8({in2}, C2_mask);
               __m128i C6 = _mm_shuffle_epi8({in3}, C3_mask);
               C4 = _mm_or_si128(C4, C5);
               C4 = _mm_or_si128(C4, C6);

               {store4}'''.format(store4=store3('sse', typ, align, fmtspec,
                                  'A4', 'B4', 'C4'), **fmtspec)
        else:
            return \
            '''__m128i A0 = {in1};
               __m128i B0 = {in2};
               __m128i C0 = {in3};
               int k;

               for (k = 0; k < 4; ++k) {{
                 __m128i A1 = _mm_unpacklo_epi8(A0, B0);
                 __m128i A2 = _mm_unpackhi_epi8(A0, B0);
                 __m128i A3 = _mm_unpacklo_epi8(A1, A2);
                 __m128i A4 = _mm_unpackhi_epi8(A1, A2);
                 __m128i A5 = _mm_unpacklo_epi8(A3, A4);
                 __m128i A6 = _mm_unpackhi_epi8(A3, A4);
                 __m128i A7 = _mm_unpacklo_epi8(A5, A6);
                 __m128i B8 = _mm_unpackhi_epi8(A5, A6);

                 __m128i C9  = _mm_castpd_si128(_mm_shuffle_pd(
                                   _mm_castsi128_pd(C0),
                                   _mm_castsi128_pd(C0), 1));
                 __m128i C10 = _mm_unpacklo_epi8(C0, C9);
                 __m128i C11 = _mm_castpd_si128(_mm_shuffle_pd(
                                   _mm_castsi128_pd(C10),
                                   _mm_castsi128_pd(C10), 1));
                 __m128i C12 = _mm_unpacklo_epi8(C10, C11);
                 __m128i C13 = _mm_castpd_si128(_mm_shuffle_pd(
                                   _mm_castsi128_pd(C12),
                                   _mm_castsi128_pd(C12), 1));
                 __m128i C14 = _mm_unpacklo_epi8(C12, C13);

                 __m128i B15 = _mm_castpd_si128(_mm_shuffle_pd(
                                   _mm_castsi128_pd(C14),
                                   _mm_castsi128_pd(B8), 0));
                 __m128i C16 = _mm_castpd_si128(_mm_shuffle_pd(
                                   _mm_castsi128_pd(B8),
                                   _mm_castsi128_pd(C14), 3));

                 A0 = A7;
                 B0 = B15;
                 C0 = C16;
               }}
               {store0}'''.format(store0=store3('sse', typ, align, fmtspec,
                                  'A0', 'B0', 'C0'), **fmtspec)
    if typ in ['i16', 'u16']:
        if simd_ext == 'avx2':
            return \
            '''__m128i A0 = {in1};
               __m128i B0 = {in2};
               __m128i C0 = {in3};

               __m128i A1_mask = _mm_set_epi8(-1, -1,  5,  4, -1, -1, -1, -1,
                                               3,  2, -1, -1, -1, -1,  1,  0);
               __m128i A2_mask = _mm_set_epi8( 5,  4, -1, -1, -1, -1,  3,  2,
                                              -1, -1, -1, -1,  1,  0, -1, -1);
               __m128i A3_mask = _mm_set_epi8(-1, -1, -1, -1,  3,  2, -1, -1,
                                              -1, -1,  1,  0, -1, -1, -1, -1);
               __m128i A4 = _mm_shuffle_epi8(A0, A1_mask);
               __m128i A5 = _mm_shuffle_epi8(B0, A2_mask);
               __m128i A6 = _mm_shuffle_epi8(C0, A3_mask);
               A4 = _mm_or_si128(A4, A5);
               A4 = _mm_or_si128(A4, A6);

               __m128i B1_mask = _mm_set_epi8(11, 10, -1, -1, -1, -1,  9,  8,
                                              -1, -1, -1, -1,  7,  6, -1, -1);
               __m128i B2_mask = _mm_set_epi8(-1, -1, -1, -1,  9,  8, -1, -1,
                                              -1, -1,  7,  6, -1, -1, -1, -1);
               __m128i B3_mask = _mm_set_epi8(-1, -1,  9,  8, -1, -1, -1, -1,
                                               7,  6, -1, -1, -1, -1,  5,  4);
               __m128i B4 = _mm_shuffle_epi8(A0, B1_mask);
               __m128i B5 = _mm_shuffle_epi8(B0, B2_mask);
               __m128i B6 = _mm_shuffle_epi8(C0, B3_mask);
               B4 = _mm_or_si128(B4, B5);
               B4 = _mm_or_si128(B4, B6);

               __m128i C1_mask = _mm_set_epi8(-1, -1, -1, -1, 15, 14, -1, -1,
                                              -1, -1, 13, 12, -1, -1, -1, -1);
               __m128i C2_mask = _mm_set_epi8(-1, -1, 15, 14, -1, -1, -1, -1,
                                              13, 12, -1, -1, -1, -1, 11, 10);
               __m128i C3_mask = _mm_set_epi8(15, 14, -1, -1, -1, -1, 13, 12,
                                              -1, -1, -1, -1, 11, 10, -1, -1);
               __m128i C4 = _mm_shuffle_epi8(A0, C1_mask);
               __m128i C5 = _mm_shuffle_epi8(B0, C2_mask);
               __m128i C6 = _mm_shuffle_epi8(C0, C3_mask);
               C4 = _mm_or_si128(C4, C5);
               C4 = _mm_or_si128(C4, C6);

               {store4};'''.format(store4=store3('sse', typ, align, fmtspec,
                                   'A4', 'B4', 'C4'), **fmtspec)
        else:
            return \
            '''__m128i A0 = {in1};
               __m128i B0 = {in2};
               __m128i C0 = {in3};
               int k;

               for (k = 0; k < 3; ++k) {{
                 __m128i A1 = _mm_shufflelo_epi16(A0, _MM_SHUFFLE(3, 1, 2, 0));
                 __m128i A2 = _mm_shufflehi_epi16(A1, _MM_SHUFFLE(3, 1, 2, 0));
                 __m128i B3 = _mm_shufflelo_epi16(B0, _MM_SHUFFLE(3, 1, 2, 0));
                 __m128i B4 = _mm_shufflehi_epi16(B3, _MM_SHUFFLE(3, 1, 2, 0));
                 __m128i C5 = _mm_shufflelo_epi16(C0, _MM_SHUFFLE(3, 1, 2, 0));
                 __m128i C6 = _mm_shufflehi_epi16(C5, _MM_SHUFFLE(3, 1, 2, 0));

                 __m128 A2_ps = _mm_castsi128_ps(A2);
                 __m128 B4_ps = _mm_castsi128_ps(B4);
                 __m128 C6_ps = _mm_castsi128_ps(C6);

                 __m128 A0_ps = _mm_shuffle_ps(A2_ps, B4_ps,
                                               _MM_SHUFFLE(2, 0, 2, 0));
                 __m128 B0_ps = _mm_shuffle_ps(C6_ps, A2_ps,
                                               _MM_SHUFFLE(3, 1, 2, 0));
                 __m128 C0_ps = _mm_shuffle_ps(B4_ps, C6_ps,
                                               _MM_SHUFFLE(3, 1, 3, 1));

                 A0 = _mm_castps_si128(A0_ps);
                 B0 = _mm_castps_si128(B0_ps);
                 C0 = _mm_castps_si128(C0_ps);
               }}

               {store0}'''.format(store0=store3('sse', typ, align, fmtspec,
                                  'A0', 'B0', 'C0'), **fmtspec)
    if typ == 'f32':
        return \
        '''__m128 A1 = _mm_shuffle_ps({in1}, {in2}, _MM_SHUFFLE(2,0,2,0));
           __m128 B2 = _mm_shuffle_ps({in3}, {in1}, _MM_SHUFFLE(3,1,2,0));
           __m128 C3 = _mm_shuffle_ps({in2}, {in3}, _MM_SHUFFLE(3,1,3,1));

           __m128 A4 = _mm_shuffle_ps(A1, B2, _MM_SHUFFLE(2,0,2,0));
           __m128 B5 = _mm_shuffle_ps(C3, A1, _MM_SHUFFLE(3,1,2,0));
           __m128 C6 = _mm_shuffle_ps(B2, C3, _MM_SHUFFLE(3,1,3,1));

           {store};'''. \
           format(store=store3('sse', typ, align, fmtspec, 'A4', 'B5', 'C6'),
                  **fmtspec)
    if typ in ['i32', 'u32']:
        return \
        '''nsimd_store3{a}_{simd_ext}_f32((f32 *){in0},
                                          _mm_castsi128_ps({in1}),
                                          _mm_castsi128_ps({in2}),
                                          _mm_castsi128_ps({in3}));'''. \
                                          format(**fmtspec)
    if typ == 'f64':
        return \
        '''__m128d A0 = _mm_unpacklo_pd({in1}, {in2});
           __m128d B0 = _mm_shuffle_pd({in3}, {in1}, 2);
           __m128d C0 = _mm_unpackhi_pd({in2}, {in3});
           {store}'''. \
           format(store=store3('sse', typ, align, fmtspec, 'A0', 'B0', 'C0'),
                  **fmtspec)
    if typ in ['i64', 'u64']:
        return \
        '''nsimd_store3{a}_{simd_ext}_f64((f64 *){in0},
                                          _mm_castsi128_pd({in1}),
                                          _mm_castsi128_pd({in2}),
                                          _mm_castsi128_pd({in3}));'''. \
                                          format(**fmtspec)

###############################################################################

def load3_avx(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['load_v0v1v2'] = get_load_v0v1v2('avx', typ, align, fmtspec)
    fmtspec['exlo_v0'] = x86.extract('avx', typ, x86.LO, 'v0')
    fmtspec['exhi_v0'] = x86.extract('avx', typ, x86.HI, 'v0')
    fmtspec['exlo_v1'] = x86.extract('avx', typ, x86.LO, 'v1')
    fmtspec['exhi_v1'] = x86.extract('avx', typ, x86.HI, 'v1')
    fmtspec['exlo_v2'] = x86.extract('avx', typ, x86.LO, 'v2')
    fmtspec['exhi_v2'] = x86.extract('avx', typ, x86.HI, 'v2')
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        if simd_ext == 'avx2':
            return \
            '''nsimd_avx2_v{typ}x3 ret;
               {load_v0v1v2}

               __m256i ARmask = _mm256_setr_epi8( 0,  3,  6,  9, 12, 15, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1,  2,  5,
                                                  8, 11, 14, -1, -1, -1, -1, -1);
               __m256i BRmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1,  1,  4,  7, 10, 13,
                                                  0,  3,  6,  9, 12, 15, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1);
               __m256i CRmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1,  2,  5,
                                                  8, 11, 14, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1,  1,  4,  7, 10, 13);

               __m256i AR = _mm256_shuffle_epi8(v0, ARmask);
               __m256i BR = _mm256_shuffle_epi8(v1, BRmask);
               __m256i CR = _mm256_shuffle_epi8(v2, CRmask);
               __m256i DR = _mm256_permute2f128_si256(AR, CR, (2 << 4) | 1);

               __m256i R0 = _mm256_or_si256(AR, BR);
               __m256i R1 = _mm256_or_si256(BR, CR);
               __m256i R2 = _mm256_permute2f128_si256(R0, R1, 3 << 4);
               ret.v0 = _mm256_or_si256(DR, R2);


               __m256i AGmask = _mm256_setr_epi8( 1,  4,  7, 10, 13, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1,  0,  3,  6,
                                                  9, 12, 15, -1, -1, -1, -1, -1);
               __m256i BGmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1,  2,  5,  8, 11, 14,
                                                  1,  4,  7, 10, 13, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1);
               __m256i CGmask = _mm256_setr_epi8(-1, -1, -1, -1, -1,  0,  3,  6,
                                                  9, 12, 15, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1,  2,  5,  8, 11, 14);

               __m256i AG = _mm256_shuffle_epi8(v0, AGmask);
               __m256i BG = _mm256_shuffle_epi8(v1, BGmask);
               __m256i CG = _mm256_shuffle_epi8(v2, CGmask);
               __m256i DG = _mm256_permute2f128_si256(AG, CG, (2 << 4) | 1);

               __m256i G0 = _mm256_or_si256(AG, BG);
               __m256i G1 = _mm256_or_si256(BG, CG);
               __m256i G2 = _mm256_permute2f128_si256(G0, G1, 3 << 4);
               ret.v1 = _mm256_or_si256(DG, G2);

               __m256i ABmask = _mm256_setr_epi8( 2,  5,  8, 11, 14, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1,  1,  4,  7,
                                                 10, 13, -1, -1, -1, -1, -1, -1);
               __m256i BBmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1,  0,  3,  6,  9, 12, 15,
                                                  2,  5,  8, 11, 14, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1);
               __m256i CBmask = _mm256_setr_epi8(-1, -1, -1, -1, -1,  1,  4,  7,
                                                 10, 13, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1,  0,  3,  6,  9, 12, 15);

               __m256i AB = _mm256_shuffle_epi8(v0, ABmask);
               __m256i BB = _mm256_shuffle_epi8(v1, BBmask);
               __m256i CB = _mm256_shuffle_epi8(v2, CBmask);
               __m256i DB = _mm256_permute2f128_si256(AB, CB, (2 << 4) | 1);

               __m256i B0 = _mm256_or_si256(AB, BB);
               __m256i B1 = _mm256_or_si256(BB, CB);
               __m256i B2 = _mm256_permute2f128_si256(B0, B1, 3 << 4);
               ret.v2 = _mm256_or_si256(DB, B2);
               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_avx_v{typ}x3 ret;
               {load_v0v1v2}

               __m128i Aa = {exlo_v0};
               __m128i Ba = {exhi_v0};
               __m128i Ca = {exlo_v1};
               __m128i Ab = {exhi_v1};
               __m128i Bb = {exlo_v2};
               __m128i Cb = {exhi_v2};

               __m128i ARm = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                          -1, -1, 15, 12,  9,  6,  3,  0);
               __m128i BRm = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11,  8,
                                           5,  2, -1, -1, -1, -1, -1, -1);
               __m128i CRm = _mm_set_epi8(13, 10,  7,  4,  1, -1, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1);
               __m128i AR = _mm_shuffle_epi8(Aa, ARm);
               __m128i BR = _mm_shuffle_epi8(Ba, BRm);
               __m128i CR = _mm_shuffle_epi8(Ca, CRm);
               __m128i R0 = _mm_or_si128(AR, BR);
               R0 = _mm_or_si128(R0, CR);

               AR = _mm_shuffle_epi8(Ab, ARm);
               BR = _mm_shuffle_epi8(Bb, BRm);
               CR = _mm_shuffle_epi8(Cb, CRm);
               __m128i R1 = _mm_or_si128(AR, BR);
               R1 = _mm_or_si128(R1, CR);

               __m128i AGm = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                          -1, -1, -1, 13, 10,  7,  4,  1);
               __m128i BGm = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12,  9,
                                           6,  3,  0, -1, -1, -1, -1, -1);
               __m128i CGm = _mm_set_epi8(14, 11,  8,  5,  2, -1, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1);
               __m128i AG = _mm_shuffle_epi8(Aa, AGm);
               __m128i BG = _mm_shuffle_epi8(Ba, BGm);
               __m128i CG = _mm_shuffle_epi8(Ca, CGm);
               __m128i G0 = _mm_or_si128(AG, BG);
               G0 = _mm_or_si128(G0, CG);

               AG = _mm_shuffle_epi8(Ab, AGm);
               BG = _mm_shuffle_epi8(Bb, BGm);
               CG = _mm_shuffle_epi8(Cb, CGm);
               __m128i G1 = _mm_or_si128(AG, BG);
               G1 = _mm_or_si128(G1, CG);

               __m128i ABm = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                          -1, -1, -1, 14, 11,  8,  5,  2);
               __m128i BBm = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10,
                                           7,  4,  1, -1, -1, -1, -1, -1);
               __m128i CBm = _mm_set_epi8(15, 12,  9,  6,  3,  0, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1);
               __m128i AB = _mm_shuffle_epi8(Aa, ABm);
               __m128i BB = _mm_shuffle_epi8(Ba, BBm);
               __m128i CB = _mm_shuffle_epi8(Ca, CBm);
               __m128i B0 = _mm_or_si128(AB, BB);
               B0 = _mm_or_si128(B0, CB);

               AB = _mm_shuffle_epi8(Ab, ABm);
               BB = _mm_shuffle_epi8(Bb, BBm);
               CB = _mm_shuffle_epi8(Cb, CBm);
               __m128i B1 = _mm_or_si128(AB, BB);
               B1 = _mm_or_si128(B1, CB);

               ret.v0 = {mergeR};
               ret.v1 = {mergeG};
               ret.v2 = {mergeB};

               return ret;'''.format(mergeR=x86.setr('avx', typ, 'R0', 'R1'),
                                     mergeG=x86.setr('avx', typ, 'G0', 'G1'),
                                     mergeB=x86.setr('avx', typ, 'B0', 'B1'),
                                     **fmtspec)
    if typ in ['i16', 'u16']:
        if simd_ext == 'avx2':
            return \
            '''nsimd_avx2_v{typ}x3 ret;
               {load_v0v1v2}
               __m256i ARmask = _mm256_setr_epi8( 0,  1,  6,  7, 12, 13, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1,  2,  3,
                                                  8,  9, 14, 15, -1, -1, -1, -1);
               __m256i BRmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1,  4,  5, 10, 11,
                                                  0,  1,  6,  7, 12, 13, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1);
               __m256i CRmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1,  2,  3,
                                                  8,  9, 14, 15, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1,  4,  5, 10, 11);

               __m256i AR = _mm256_shuffle_epi8(v0, ARmask);
               __m256i BR = _mm256_shuffle_epi8(v1, BRmask);
               __m256i CR = _mm256_shuffle_epi8(v2, CRmask);
               __m256i DR = _mm256_permute2f128_si256(AR, CR, (2 << 4) | 1);

               __m256i R0 = _mm256_or_si256(AR, BR);
               __m256i R1 = _mm256_or_si256(BR, CR);
               __m256i R2 = _mm256_permute2f128_si256(R0, R1, 3 << 4);
               ret.v0 = _mm256_or_si256(DR, R2);


               __m256i AGmask = _mm256_setr_epi8( 2,  3,  8,  9, 14, 15, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1,  4,  5,
                                                 10, 11, -1, -1, -1, -1, -1, -1);
               __m256i BGmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1,  0,  1,  6,  7, 12, 13,
                                                  2,  3,  8,  9, 14, 15, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1);
               __m256i CGmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1,  4,  5,
                                                 10, 11, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1,  0,  1,  6,  7, 12, 13);

               __m256i AG = _mm256_shuffle_epi8(v0, AGmask);
               __m256i BG = _mm256_shuffle_epi8(v1, BGmask);
               __m256i CG = _mm256_shuffle_epi8(v2, CGmask);
               __m256i DG = _mm256_permute2f128_si256(AG, CG, (2 << 4) | 1);

               __m256i G0 = _mm256_or_si256(AG, BG);
               __m256i G1 = _mm256_or_si256(BG, CG);
               __m256i G2 = _mm256_permute2f128_si256(G0, G1, 3 << 4);
               ret.v1 = _mm256_or_si256(DG, G2);

               __m256i ABmask = _mm256_setr_epi8( 4,  5, 10, 11, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1,  0,  1,  6,  7,
                                                 12, 13, -1, -1, -1, -1, -1, -1);
               __m256i BBmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1,  2,  3,  8,  9, 14, 15,
                                                  4,  5, 10, 11, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1);
               __m256i CBmask = _mm256_setr_epi8(-1, -1, -1, -1,  0,  1,  6,  7,
                                                 12, 13, -1, -1, -1, -1, -1, -1,
                                                 -1, -1, -1, -1, -1, -1, -1, -1,
                                                 -1, -1,  2,  3,  8,  9, 14, 15);

               __m256i AB = _mm256_shuffle_epi8(v0, ABmask);
               __m256i BB = _mm256_shuffle_epi8(v1, BBmask);
               __m256i CB = _mm256_shuffle_epi8(v2, CBmask);
               __m256i DB = _mm256_permute2f128_si256(AB, CB, (2 << 4) | 1);

               __m256i B0 = _mm256_or_si256(AB, BB);
               __m256i B1 = _mm256_or_si256(BB, CB);
               __m256i B2 = _mm256_permute2f128_si256(B0, B1, 3 << 4);
               ret.v2 = _mm256_or_si256(DB, B2);
               return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_avx_v{typ}x3 ret;
               {load_v0v1v2}

               __m128i Aa = {exlo_v0};
               __m128i Ba = {exhi_v0};
               __m128i Ca = {exlo_v1};
               __m128i Ab = {exhi_v1};
               __m128i Bb = {exlo_v2};
               __m128i Cb = {exhi_v2};

               __m128i ARm = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                          -1, -1, 13, 12,  7,  6,  1,  0);
               __m128i BRm = _mm_set_epi8(-1, -1, -1, -1, 15, 14,  9,  8,
                                           3,  2, -1, -1, -1, -1, -1, -1);
               __m128i CRm = _mm_set_epi8(11, 10,  5,  4, -1, -1, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1);
               __m128i AR = _mm_shuffle_epi8(Aa, ARm);
               __m128i BR = _mm_shuffle_epi8(Ba, BRm);
               __m128i CR = _mm_shuffle_epi8(Ca, CRm);
               __m128i R0 = _mm_or_si128(AR, BR);
               R0 = _mm_or_si128(R0, CR);

               AR = _mm_shuffle_epi8(Ab, ARm);
               BR = _mm_shuffle_epi8(Bb, BRm);
               CR = _mm_shuffle_epi8(Cb, CRm);
               __m128i R1 = _mm_or_si128(AR, BR);
               R1 = _mm_or_si128(R1, CR);

               __m128i AGm = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                          -1, -1, 15, 14,  9,  8,  3,  2);
               __m128i BGm = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 11, 10,
                                           5,  4, -1, -1, -1, -1, -1, -1);
               __m128i CGm = _mm_set_epi8(13, 12,  7,  6,  1,  0, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1);
               __m128i AG = _mm_shuffle_epi8(Aa, AGm);
               __m128i BG = _mm_shuffle_epi8(Ba, BGm);
               __m128i CG = _mm_shuffle_epi8(Ca, CGm);
               __m128i G0 = _mm_or_si128(AG, BG);
               G0 = _mm_or_si128(G0, CG);

               AG = _mm_shuffle_epi8(Ab, AGm);
               BG = _mm_shuffle_epi8(Bb, BGm);
               CG = _mm_shuffle_epi8(Cb, CGm);
               __m128i G1 = _mm_or_si128(AG, BG);
               G1 = _mm_or_si128(G1, CG);

               __m128i ABm = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                          -1, -1, -1, -1, 11, 10,  5,  4);
               __m128i BBm = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 12,
                                           7,  6,  1,  0, -1, -1, -1, -1);
               __m128i CBm = _mm_set_epi8(15, 14,  9,  8,  3,  2, -1, -1,
                                          -1, -1, -1, -1, -1, -1, -1, -1);
               __m128i AB = _mm_shuffle_epi8(Aa, ABm);
               __m128i BB = _mm_shuffle_epi8(Ba, BBm);
               __m128i CB = _mm_shuffle_epi8(Ca, CBm);
               __m128i B0 = _mm_or_si128(AB, BB);
               B0 = _mm_or_si128(B0, CB);

               AB = _mm_shuffle_epi8(Ab, ABm);
               BB = _mm_shuffle_epi8(Bb, BBm);
               CB = _mm_shuffle_epi8(Cb, CBm);
               __m128i B1 = _mm_or_si128(AB, BB);
               B1 = _mm_or_si128(B1, CB);

               ret.v0 = {mergeR};
               ret.v1 = {mergeG};
               ret.v2 = {mergeB};
               return ret;'''.format(mergeR=x86.setr('avx', typ, 'R0', 'R1'),
                                     mergeG=x86.setr('avx', typ, 'G0', 'G1'),
                                     mergeB=x86.setr('avx', typ, 'B0', 'B1'),
                                     **fmtspec)
    avx2_template = \
    '''nsimd_avx2_v{typ}x3 ret;
       {load_v0v1v2}

       __m256i RAm = _mm256_setr_epi32( 0,  3,  6, -1, -1, -1, -1, -1);
       __m256i RBm = _mm256_setr_epi32(-1, -1, -1,  1,  4,  7, -1, -1);
       __m256i RCm = _mm256_setr_epi32(-1, -1, -1, -1, -1, -1,  2,  5);

       __m256i GAm = _mm256_setr_epi32( 1,  4,  7, -1, -1, -1, -1, -1);
       __m256i GBm = _mm256_setr_epi32(-1, -1, -1,  2,  5, -1, -1, -1);
       __m256i GCm = _mm256_setr_epi32(-1, -1, -1, -1, -1,  0,  3,  6);

       __m256i BAm = _mm256_setr_epi32( 2,  5, -1, -1, -1, -1, -1, -1);
       __m256i BBm = _mm256_setr_epi32(-1, -1,  0,  3,  6, -1, -1, -1);
       __m256i BCm = _mm256_setr_epi32(-1, -1, -1, -1, -1,  1,  4,  7);

       {styp} RA = _mm256_permutevar8x32{suf}(v0, RAm);
       {styp} RB = _mm256_permutevar8x32{suf}(v1, RBm);
       {styp} RC = _mm256_permutevar8x32{suf}(v2, RCm);

       {styp} R = _mm256_blend{suf}(RA, RB, 8 + 16 + 32);
       ret.v0 = _mm256_blend{suf}(R, RC, 64 + 128);

       {styp} GA = _mm256_permutevar8x32{suf}(v0, GAm);
       {styp} GB = _mm256_permutevar8x32{suf}(v1, GBm);
       {styp} GC = _mm256_permutevar8x32{suf}(v2, GCm);
       {styp} G = _mm256_blend{suf}(GA, GB, 8 + 16);
       ret.v1 = _mm256_blend{suf}(G, GC, 32 + 64 + 128);

       {styp} BA = _mm256_permutevar8x32{suf}(v0, BAm);
       {styp} BB = _mm256_permutevar8x32{suf}(v1, BBm);
       {styp} BC = _mm256_permutevar8x32{suf}(v2, BCm);
       {styp} B = _mm256_blend{suf}(BA, BB, 4 + 8 + 16);
       ret.v2 = _mm256_blend{suf}(B, BC, 32 + 64 + 128);

       return ret;'''.format(**fmtspec)
    if typ == 'f32':
        if simd_ext == 'avx2':
            return avx2_template
        else:
            return \
            '''nsimd_avx_v{typ}x3 ret;
               {load_v0v1v2}

               __m256i RAm = _mm256_setr_epi32( 0,  3, -1, -1, -1, -1,  2, -1);
               __m256i RBm = _mm256_setr_epi32(-1, -1, -1,  1,  0,  3, -1, -1);
               __m256i RCm = _mm256_setr_epi32( 0,  0,  2,  0,  1,  1,  1,  1);

               __m256i GAm = _mm256_setr_epi32( 1, -1, -1, -1, -1,  0,  3, -1);
               __m256i GBm = _mm256_setr_epi32(-1, -1, -1,  2,  5, -1, -1, -1);
               __m256i GCm = _mm256_setr_epi32(-1,  0,  3, -1, -1, -1, -1,  6);

               __m256i BAm = _mm256_setr_epi32( 2, -1, -1, -1, -1,  1, -1, -1);
               __m256i BBm = _mm256_setr_epi32(-1, -1,  0,  3,  6, -1, -1, -1);
               __m256i BCm = _mm256_setr_epi32(-1,  1, -1, -1, -1, -1,  4,  7);

               __m256 RA = _mm256_permutevar_ps(v0, RAm);
               __m256 RAi = _mm256_permute2f128_ps(RA, RA, (2 << 4) | 1);
               RA = _mm256_blend_ps(RAi, RA, 1 + 2);
               __m256 RB = _mm256_permutevar_ps(v1, RBm);
               __m256 RC = _mm256_permutevar_ps(v2, RCm);
               __m256 RCi = _mm256_permute2f128_ps(RC, RC, 2 << 4);
               RC = _mm256_blend_ps(RC, RCi, 64);
               __m256 R = _mm256_blend_ps(RA, RB, 8 + 16 + 32);
               ret.v0 = _mm256_blend_ps(R, RC, 64 + 128);

               __m256 GA = _mm256_permutevar_ps(v0, GAm);
               __m256 GAi = _mm256_permute2f128_ps(GA, GA, (2 << 4) | 1);
               GA = _mm256_blend_ps(GA, GAi, 2 + 4);
               __m256 GB = _mm256_permutevar_ps(v1, GBm);
               __m256 GC = _mm256_permutevar_ps(v2, GCm);
               __m256 GCi = _mm256_permute2f128_ps(GC, GC, 2 << 4);
               GC = _mm256_blend_ps(GC, GCi, 32 + 64);
               __m256 G = _mm256_blend_ps(GA, GB, 8 + 16);
               ret.v1 = _mm256_blend_ps(G, GC, 32 + 64 + 128);

               __m256 BA = _mm256_permutevar_ps(v0, BAm);
               __m256 BAi = _mm256_permute2f128_ps(BA, BA, (2 << 4) | 1);
               BA = _mm256_blend_ps(BA, BAi, 2);
               __m256 BB = _mm256_permutevar_ps(v1, BBm);
               __m256 BC = _mm256_permutevar_ps(v2, BCm);
               __m256 BCi = _mm256_permute2f128_ps(BC, BC, 2 << 4);
               BC = _mm256_blend_ps(BC, BCi, 32);
               __m256 B = _mm256_blend_ps(BA, BB, 4 + 8 + 16);
               ret.v2 = _mm256_blend_ps(B, BC, 32 + 64 + 128);

               return ret;'''.format(**fmtspec)
    if typ in ['i32', 'u32', 'f32']:
        if simd_ext == 'avx2':
            return avx2_template
        else:
            return \
            '''nsimd_avx_v{typ}x3 ret;
               nsimd_avx_vf32x3 retf32 = nsimd_load3{a}_avx_f32((f32 *){in0});
               ret.v0 = _mm256_castps_si256(retf32.v0);
               ret.v1 = _mm256_castps_si256(retf32.v1);
               ret.v2 = _mm256_castps_si256(retf32.v2);
               return ret;'''.format(**fmtspec)
    avx2_template = \
    '''nsimd_avx2_v{typ}x3 ret;
       {load_v0v1v2}
       {styp} A1 = _mm256_permute4x64{suf}(v0, _MM_SHUFFLE(2, 1, 3, 0));
       {styp} C2 = _mm256_permute4x64{suf}(v2, _MM_SHUFFLE(3, 0, 2, 1));
       {styp} B3 = _mm256_permute2f128{sufsi}(A1, v1, (2 << 4) | 1);
       {styp} B4 = _mm256_permute2f128{sufsi}(v1, C2, (2 << 4) | 1);
       {styp} B5 = _mm256_permute4x64{suf}(B3, _MM_SHUFFLE(3, 1, 2, 0));
       {styp} B6 = _mm256_permute4x64{suf}(B4, _MM_SHUFFLE(3, 1, 2, 0));
       ret.v0 = _mm256_permute2f128{sufsi}(A1, B6, 2 << 4);
       ret.v1 = _mm256_permute2f128{sufsi}(B5, B6, 3 << 4);
       ret.v2 = _mm256_permute2f128{sufsi}(B5, C2, (3 << 4 ) | 1);
       return ret;'''.format(**fmtspec)
    if typ == 'f64':
        if simd_ext == 'avx2':
            return avx2_template
        else:
            return \
            '''nsimd_avx_v{typ}x3 ret;
               {load_v0v1v2}

               __m256d R1 = _mm256_permute2f128_pd(v0, v2, (2 << 4) | 1);
               __m256d R2 = _mm256_permute2f128_pd(v0, v1, 3 << 4);
               ret.v0  = _mm256_blend_pd(R1, R2, 1 + 4);

               __m256d G1 = _mm256_permute2f128_pd(v0, v1, 3 << 4);
               __m256d G2 = _mm256_permute2f128_pd(v1, v2, 3 << 4);
               __m256d G  = _mm256_blend_pd(G1, G2, 1 + 4);
               ret.v1 = _mm256_permute_pd(G, 1 + 4);

               __m256d B1 = _mm256_permute2f128_pd(v0, v2, (2 << 4) | 1);
               __m256d B2 = _mm256_permute2f128_pd(v1, v2, 3 << 4);
               ret.v2  = _mm256_blend_pd(B1, B2, 2 + 8);

               return ret;'''.format(**fmtspec)
    if typ in ['i64', 'u64']:
        if simd_ext == 'avx2':
            return avx2_template
        else:
            return \
            '''nsimd_avx_v{typ}x3 ret;
               nsimd_avx_vf64x3 retf64 = nsimd_load3{a}_avx_f64((f64 *){in0});
               ret.v0 = _mm256_castpd_si256(retf64.v0);
               ret.v1 = _mm256_castpd_si256(retf64.v1);
               ret.v2 = _mm256_castpd_si256(retf64.v2);
               return ret;'''.format(**fmtspec)

###############################################################################

def store3_avx(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['exlo_in1'] = x86.extract('avx', typ, x86.LO, common.in1)
    fmtspec['exhi_in1'] = x86.extract('avx', typ, x86.HI, common.in1)
    fmtspec['exlo_in2'] = x86.extract('avx', typ, x86.LO, common.in2)
    fmtspec['exhi_in2'] = x86.extract('avx', typ, x86.HI, common.in2)
    fmtspec['exlo_in3'] = x86.extract('avx', typ, x86.LO, common.in3)
    fmtspec['exhi_in3'] = x86.extract('avx', typ, x86.HI, common.in3)
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        if simd_ext == 'avx2':
            return \
            '''__m256i RACm = _mm256_setr_epi8( 0, -1, -1,  1, -1, -1,  2, -1,
                                               -1,  3, -1, -1,  4, -1, -1,  5,
                                               -1, 27, -1, -1, 28, -1, -1, 29,
                                               -1, -1, 30, -1, -1, 31, -1, -1);
               __m256i RBBm = _mm256_setr_epi8(-1, 11, -1, -1, 12, -1, -1, 13,
                                               -1, -1, 14, -1, -1, 15, -1, -1,
                                               16, -1, -1, 17, -1, -1, 18, -1,
                                               -1, 19, -1, -1, 20, -1, -1, 21);
               __m256i RCAm = _mm256_setr_epi8(-1, -1, 22, -1, -1, 23, -1, -1,
                                               24, -1, -1, 25, -1, -1, 26, -1,
                                               -1, -1,  6, -1, -1,  7, -1, -1,
                                                8, -1, -1,  9, -1, -1, 10, -1);

               __m256i GACm = _mm256_setr_epi8(-1,  0, -1, -1,  1, -1, -1,  2,
                                               -1, -1,  3, -1, -1,  4, -1, -1,
                                               -1, -1, 27, -1, -1, 28, -1, -1,
                                               29, -1, -1, 30, -1, -1, 31, -1);
               __m256i GBBm = _mm256_setr_epi8(-1, -1, 11, -1, -1, 12, -1, -1,
                                               13, -1, -1, 14, -1, -1, 15, -1,
                                               -1, 16, -1, -1, 17, -1, -1, 18,
                                               -1, -1, 19, -1, -1, 20, -1, -1);
               __m256i GCAm = _mm256_setr_epi8(21, -1, -1, 22, -1, -1, 23, -1,
                                               -1, 24, -1, -1, 25, -1, -1, 26,
                                                5, -1, -1,  6, -1, -1,  7, -1,
                                               -1,  8, -1, -1,  9, -1, -1, 10);

               __m256i BACm = _mm256_setr_epi8(-1, -1,  0, -1, -1,  1, -1, -1,
                                                2, -1, -1,  3, -1, -1,  4, -1,
                                               26, -1, -1, 27, -1, -1, 28, -1,
                                               -1, 29, -1, -1, 30, -1, -1, 31);
               __m256i BBBm = _mm256_setr_epi8(10, -1, -1, 11, -1, -1, 12, -1,
                                               -1, 13, -1, -1, 14, -1, -1, 15,
                                               -1, -1, 16, -1, -1, 17, -1, -1,
                                               18, -1, -1, 19, -1, -1, 20, -1);
               __m256i BCAm = _mm256_setr_epi8(-1, 21, -1, -1, 22, -1, -1, 23,
                                               -1, -1, 24, -1, -1, 25, -1, -1,
                                               -1,  5, -1, -1,  6, -1, -1,  7,
                                               -1, -1,  8, -1, -1,  9, -1, -1);

               __m256i RAC = _mm256_shuffle_epi8({in1}, RACm);
               __m256i GAC = _mm256_shuffle_epi8({in2}, GACm);
               __m256i BAC = _mm256_shuffle_epi8({in3}, BACm);

               __m256i RBB = _mm256_shuffle_epi8({in1}, RBBm);
               __m256i GBB = _mm256_shuffle_epi8({in2}, GBBm);
               __m256i BBB = _mm256_shuffle_epi8({in3}, BBBm);

               __m256i RCA = _mm256_shuffle_epi8({in1}, RCAm);
               __m256i GCA = _mm256_shuffle_epi8({in2}, GCAm);
               __m256i BCA = _mm256_shuffle_epi8({in3}, BCAm);

               __m256i AC = _mm256_or_si256(RAC, GAC);
               AC = _mm256_or_si256(AC, BAC);

               __m256i B = _mm256_or_si256(RBB, GBB);
               B = _mm256_or_si256(B, BBB);

               __m256i CA = _mm256_or_si256(RCA, GCA);
               CA = _mm256_or_si256(CA, BCA);

               __m256i A = _mm256_permute2f128_si256(AC, CA, 2 << 4);
               __m256i C = _mm256_permute2f128_si256(AC, CA, (1 << 4) | 3);

               {store}'''.format(store=store3('avx', typ, align, fmtspec,
                                              'A', 'B', 'C'), **fmtspec)
        else:
            return \
            '''__m128i Ra = {exlo_in1};
               __m128i Rb = {exhi_in1};
               __m128i Ga = {exlo_in2};
               __m128i Gb = {exhi_in2};
               __m128i Ba = {exlo_in3};
               __m128i Bb = {exhi_in3};

               __m128i RAm = _mm_set_epi8( 5, -1, -1,  4, -1, -1,  3, -1,
                                          -1,  2, -1, -1,  1, -1, -1,  0);
               __m128i GAm = _mm_set_epi8(-1, -1,  4, -1, -1,  3, -1, -1,
                                           2, -1, -1,  1, -1, -1,  0, -1);
               __m128i BAm = _mm_set_epi8(-1,  4, -1, -1,  3, -1, -1,  2,
                                          -1, -1,  1, -1, -1,  0, -1, -1);
               __m128i RA = _mm_shuffle_epi8(Ra, RAm);
               __m128i GA = _mm_shuffle_epi8(Ga, GAm);
               __m128i BA = _mm_shuffle_epi8(Ba, BAm);
               __m128i A0 = _mm_or_si128(RA, GA);
               A0 = _mm_or_si128(A0, BA);

               RA = _mm_shuffle_epi8(Rb, RAm);
               GA = _mm_shuffle_epi8(Gb, GAm);
               BA = _mm_shuffle_epi8(Bb, BAm);
               __m128i A1 = _mm_or_si128(RA, GA);
               A1 = _mm_or_si128(A1, BA);

               __m128i RBm = _mm_set_epi8(-1, 10, -1, -1,  9, -1, -1,  8,
                                          -1, -1,  7, -1, -1,  6, -1, -1);
               __m128i GBm = _mm_set_epi8(10, -1, -1,  9, -1, -1,  8, -1,
                                          -1,  7, -1, -1,  6, -1, -1,  5);
               __m128i BBm = _mm_set_epi8(-1, -1,  9, -1, -1,  8, -1, -1,
                                           7, -1, -1,  6, -1, -1,  5, -1);
               __m128i RB = _mm_shuffle_epi8(Ra, RBm);
               __m128i GB = _mm_shuffle_epi8(Ga, GBm);
               __m128i BB = _mm_shuffle_epi8(Ba, BBm);
               __m128i B0 = _mm_or_si128(RB, GB);
               B0 = _mm_or_si128(B0, BB);

               RB = _mm_shuffle_epi8(Rb, RBm);
               GB = _mm_shuffle_epi8(Gb, GBm);
               BB = _mm_shuffle_epi8(Bb, BBm);
               __m128i B1 = _mm_or_si128(RB, GB);
               B1 = _mm_or_si128(B1, BB);

               __m128i RCm = _mm_set_epi8(-1, -1, 15, -1, -1, 14, -1, -1,
                                          13, -1, -1, 12, -1, -1, 11, -1);
               __m128i GCm = _mm_set_epi8(-1, 15, -1, -1, 14, -1, -1, 13,
                                          -1, -1, 12, -1, -1, 11, -1, -1);
               __m128i BCm = _mm_set_epi8(15, -1, -1, 14, -1, -1, 13, -1,
                                          -1, 12, -1, -1, 11, -1, -1, 10);
               __m128i RC = _mm_shuffle_epi8(Ra, RCm);
               __m128i GC = _mm_shuffle_epi8(Ga, GCm);
               __m128i BC = _mm_shuffle_epi8(Ba, BCm);
               __m128i C0 = _mm_or_si128(RC, GC);
               C0 = _mm_or_si128(C0, BC);

               RC = _mm_shuffle_epi8(Rb, RCm);
               GC = _mm_shuffle_epi8(Gb, GCm);
               BC = _mm_shuffle_epi8(Bb, BCm);
               __m128i C1 = _mm_or_si128(RC, GC);
               C1 = _mm_or_si128(C1, BC);

               __m256i A = {mergeA0B0};
               __m256i B = {mergeC0A1};
               __m256i C = {mergeB1C1};

               {store}'''.format(mergeA0B0=x86.setr('avx', typ, 'A0', 'B0'),
                                 mergeC0A1=x86.setr('avx', typ, 'C0', 'A1'),
                                 mergeB1C1=x86.setr('avx', typ, 'B1', 'C1'),
                                 store=store3('avx', typ, align, fmtspec,
                                              'A', 'B', 'C'), **fmtspec)
    if typ in ['i16', 'u16']:
        if simd_ext == 'avx2':
            return \
            '''__m256i RACm = _mm256_setr_epi8( 0,  1, -1, -1, -1, -1,  2,  3,
                                               -1, -1, -1, -1,  4,  5, -1, -1,
                                               -1, -1, -1, -1, 12, 13, -1, -1,
                                               -1, -1, 14, 15, -1, -1, -1, -1);
               __m256i RBBm = _mm256_setr_epi8(-1, -1, -1, -1, 12, 13, -1, -1,
                                               -1, -1, 14, 15, -1, -1, -1, -1,
                                                0,  1, -1, -1, -1, -1,  2,  3,
                                               -1, -1, -1, -1,  4,  5, -1, -1);
               __m256i RCAm = _mm256_setr_epi8(-1, -1,  6,  7, -1, -1, -1, -1,
                                                8,  9, -1, -1, -1, -1, 10, 11,
                                               -1, -1,  6,  7, -1, -1, -1, -1,
                                                8,  9, -1, -1, -1, -1, 10, 11);

               __m256i GACm = _mm256_setr_epi8(-1, -1,  0,  1, -1, -1, -1, -1,
                                                2,  3, -1, -1, -1, -1,  4,  5,
                                               10, 11, -1, -1, -1, -1, 12, 13,
                                               -1, -1, -1, -1, 14, 15, -1, -1);
               __m256i GBBm = _mm256_setr_epi8(10, 11, -1, -1, -1, -1, 12, 13,
                                               -1, -1, -1, -1, 14, 15, -1, -1,
                                               -1, -1,  0,  1, -1, -1, -1, -1,
                                                2,  3, -1, -1, -1, -1,  4,  5);
               __m256i GCAm = _mm256_setr_epi8(-1, -1, -1, -1,  6,  7, -1, -1,
                                               -1, -1,  8,  9, -1, -1, -1, -1,
                                               -1, -1, -1, -1,  6,  7, -1, -1,
                                               -1, -1,  8,  9, -1, -1, -1, -1);

               __m256i BACm = _mm256_setr_epi8(-1, -1, -1, -1,  0,  1, -1, -1,
                                               -1, -1,  2,  3, -1, -1, -1, -1,
                                               -1, -1, 10, 11, -1, -1, -1, -1,
                                               12, 13, -1, -1, -1, -1, 14, 15);
               __m256i BBBm = _mm256_setr_epi8(-1, -1, 10, 11, -1, -1, -1, -1,
                                               12, 13, -1, -1, -1, -1, 14, 15,
                                               -1, -1, -1, -1,  0,  1, -1, -1,
                                               -1, -1,  2,  3, -1, -1, -1, -1);
               __m256i BCAm = _mm256_setr_epi8( 4,  5, -1, -1, -1, -1,  6,  7,
                                               -1, -1, -1, -1,  8,  9, -1, -1,
                                                4,  5, -1, -1, -1, -1,  6,  7,
                                                -1, -1, -1, -1,  8,  9, -1, -1);

               __m256i RAC = _mm256_shuffle_epi8({in1}, RACm);
               __m256i GAC = _mm256_shuffle_epi8({in2}, GACm);
               __m256i BAC = _mm256_shuffle_epi8({in3}, BACm);

               __m256i RBB = _mm256_shuffle_epi8({in1}, RBBm);
               __m256i GBB = _mm256_shuffle_epi8({in2}, GBBm);
               __m256i BBB = _mm256_shuffle_epi8({in3}, BBBm);

               __m256i RCA = _mm256_shuffle_epi8({in1}, RCAm);
               __m256i GCA = _mm256_shuffle_epi8({in2}, GCAm);
               __m256i BCA = _mm256_shuffle_epi8({in3}, BCAm);

               __m256i AC = _mm256_or_si256(RAC, GAC);
               AC = _mm256_or_si256(AC, BAC);

               __m256i B = _mm256_or_si256(RBB, GBB);
               B = _mm256_or_si256(B, BBB);

               __m256i CA = _mm256_or_si256(RCA, GCA);
               CA = _mm256_or_si256(CA, BCA);

               __m256i A = _mm256_permute2f128_si256(AC, CA, 2 << 4);
               __m256i C = _mm256_permute2f128_si256(AC, CA, (1 << 4) | 3);

               {store}'''.format(store=store3('avx', typ, align, fmtspec,
                                              'A', 'B', 'C'), **fmtspec)
        else:
            return \
            '''__m128i Ra = {exlo_in1};
               __m128i Rb = {exhi_in1};
               __m128i Ga = {exlo_in2};
               __m128i Gb = {exhi_in2};
               __m128i Ba = {exlo_in3};
               __m128i Bb = {exhi_in3};

               __m128i RAm = _mm_set_epi8(-1, -1,  5,  4, -1, -1, -1, -1,
                                           3,  2, -1, -1, -1, -1,  1,  0);
               __m128i GAm = _mm_set_epi8( 5,  4, -1, -1, -1, -1,  3,  2,
                                          -1, -1, -1, -1,  1,  0, -1, -1);
               __m128i BAm = _mm_set_epi8(-1, -1, -1, -1,  3,  2, -1, -1,
                                          -1, -1,  1,  0, -1, -1, -1, -1);
               __m128i RA = _mm_shuffle_epi8(Ra, RAm);
               __m128i GA = _mm_shuffle_epi8(Ga, GAm);
               __m128i BA = _mm_shuffle_epi8(Ba, BAm);
               __m128i A0 = _mm_or_si128(RA, GA);
               A0 = _mm_or_si128(A0, BA);

               RA = _mm_shuffle_epi8(Rb, RAm);
               GA = _mm_shuffle_epi8(Gb, GAm);
               BA = _mm_shuffle_epi8(Bb, BAm);
               __m128i A1 = _mm_or_si128(RA, GA);
               A1 = _mm_or_si128(A1, BA);

               __m128i RBm = _mm_set_epi8(11, 10, -1, -1, -1, -1,  9,  8,
                                          -1, -1, -1, -1,  7,  6, -1, -1);
               __m128i GBm = _mm_set_epi8(-1, -1, -1, -1,  9,  8, -1, -1,
                                          -1, -1,  7,  6, -1, -1, -1, -1);
               __m128i BBm = _mm_set_epi8(-1, -1,  9,  8, -1, -1, -1, -1,
                                           7,  6, -1, -1, -1, -1,  5,  4);
               __m128i RB = _mm_shuffle_epi8(Ra, RBm);
               __m128i GB = _mm_shuffle_epi8(Ga, GBm);
               __m128i BB = _mm_shuffle_epi8(Ba, BBm);
               __m128i B0 = _mm_or_si128(RB, GB);
               B0 = _mm_or_si128(B0, BB);

               RB = _mm_shuffle_epi8(Rb, RBm);
               GB = _mm_shuffle_epi8(Gb, GBm);
               BB = _mm_shuffle_epi8(Bb, BBm);
               __m128i B1 = _mm_or_si128(RB, GB);
               B1 = _mm_or_si128(B1, BB);

               __m128i RCm = _mm_set_epi8(-1, -1, -1, -1, 15, 14, -1, -1,
                                          -1, -1, 13, 12, -1, -1, -1, -1);
               __m128i GCm = _mm_set_epi8(-1, -1, 15, 14, -1, -1, -1, -1,
                                          13, 12, -1, -1, -1, -1, 11, 10);
               __m128i BCm = _mm_set_epi8(15, 14, -1, -1, -1, -1, 13, 12,
                                          -1, -1, -1, -1, 11, 10, -1, -1);
               __m128i RC = _mm_shuffle_epi8(Ra, RCm);
               __m128i GC = _mm_shuffle_epi8(Ga, GCm);
               __m128i BC = _mm_shuffle_epi8(Ba, BCm);
               __m128i C0 = _mm_or_si128(RC, GC);
               C0 = _mm_or_si128(C0, BC);

               RC = _mm_shuffle_epi8(Rb, RCm);
               GC = _mm_shuffle_epi8(Gb, GCm);
               BC = _mm_shuffle_epi8(Bb, BCm);
               __m128i C1 = _mm_or_si128(RC, GC);
               C1 = _mm_or_si128(C1, BC);

               __m256i A = {mergeA0B0};
               __m256i B = {mergeC0A1};
               __m256i C = {mergeB1C1};

               {store}'''.format(mergeA0B0=x86.setr('avx', typ, 'A0', 'B0'),
                                 mergeC0A1=x86.setr('avx', typ, 'C0', 'A1'),
                                 mergeB1C1=x86.setr('avx', typ, 'B1', 'C1'),
                                 store=store3('avx', typ, align, fmtspec,
                                              'A', 'B', 'C'), **fmtspec)
    avx2_template = \
    '''__m256i RAm = _mm256_setr_epi32( 0, -1, -1,  1, -1, -1,  2, -1);
       __m256i RBm = _mm256_setr_epi32(-1,  3, -1, -1,  4, -1, -1,  5);
       __m256i RCm = _mm256_setr_epi32(-1, -1,  6, -1, -1,  7, -1, -1);

       __m256i GAm = _mm256_setr_epi32(-1,  0, -1, -1,  1, -1, -1,  2);
       __m256i GBm = _mm256_setr_epi32(-1, -1,  3, -1, -1,  4, -1, -1);
       __m256i GCm = _mm256_setr_epi32( 5, -1, -1,  6, -1, -1,  7, -1);

       __m256i BAm = _mm256_setr_epi32(-1, -1,  0, -1, -1,  1, -1, -1);
       __m256i BBm = _mm256_setr_epi32( 2, -1, -1,  3, -1, -1,  4, -1);
       __m256i BCm = _mm256_setr_epi32(-1,  5, -1, -1,  6, -1, -1,  7);

       {styp} RA = _mm256_permutevar8x32{suf}({in1}, RAm);
       {styp} RB = _mm256_permutevar8x32{suf}({in1}, RBm);
       {styp} RC = _mm256_permutevar8x32{suf}({in1}, RCm);

       {styp} GA = _mm256_permutevar8x32{suf}({in2}, GAm);
       {styp} GB = _mm256_permutevar8x32{suf}({in2}, GBm);
       {styp} GC = _mm256_permutevar8x32{suf}({in2}, GCm);

       {styp} BA = _mm256_permutevar8x32{suf}({in3}, BAm);
       {styp} BB = _mm256_permutevar8x32{suf}({in3}, BBm);
       {styp} BC = _mm256_permutevar8x32{suf}({in3}, BCm);

       {styp} A = _mm256_blend{suf}(RA, GA, 2 + 16 + 128);
       A = _mm256_blend{suf}(A, BA, 4 + 32);

       {styp} B = _mm256_blend{suf}(RB, GB, 4 + 32);
       B = _mm256_blend{suf}(B, BB, 1 + 8 + 64);

       {styp} C = _mm256_blend{suf}(RC, GC, 1 + 8 + 64);
       C = _mm256_blend{suf}(C, BC, 2 + 16 + 128);

       {store}'''.format(store=store3('avx', typ, align, fmtspec,
                                      'A', 'B', 'C'), **fmtspec)
    if typ == 'f32':
        if simd_ext == 'avx2':
            return avx2_template
        else:
            return \
            '''__m256i RAm = _mm256_setr_epi32( 0, -1, -1,  1, -1, -1,  2, -1);
               __m256i RBm = _mm256_setr_epi32(-1,  3, -1, -1,  4, -1, -1,  5);
               __m256i RCm = _mm256_setr_epi32(-1, -1,  6, -1, -1,  7, -1, -1);

               __m256i GAm = _mm256_setr_epi32(-1,  0, -1, -1,  1, -1, -1,  2);
               __m256i GBm = _mm256_setr_epi32(-1, -1,  3, -1, -1,  4, -1, -1);
               __m256i GCm = _mm256_setr_epi32( 5, -1, -1,  6, -1, -1,  7, -1);

               __m256i BAm = _mm256_setr_epi32(-1, -1,  0, -1, -1,  1, -1, -1);
               __m256i BBm = _mm256_setr_epi32( 2, -1, -1,  3, -1, -1,  4, -1);
               __m256i BCm = _mm256_setr_epi32(-1,  5, -1, -1,  6, -1, -1,  7);

               __m256 RA = _mm256_permutevar_ps({in1}, RAm);
               __m256 RB = _mm256_permutevar_ps({in1}, RBm);
               __m256 RC = _mm256_permutevar_ps({in1}, RCm);

               __m256 GA = _mm256_permutevar_ps({in2}, GAm);
               __m256 GB = _mm256_permutevar_ps({in2}, GBm);
               __m256 GC = _mm256_permutevar_ps({in2}, GCm);

               __m256 BA = _mm256_permutevar_ps({in3}, BAm);
               __m256 BB = _mm256_permutevar_ps({in3}, BBm);
               __m256 BC = _mm256_permutevar_ps({in3}, BCm);

               __m256 A1 = _mm256_blend_ps(RA, GA, 2 + 16 + 128);
               A1 = _mm256_blend_ps(A1, BA, 4 + 32);

               __m256 B = _mm256_blend_ps(RB, GB, 4 + 32);
               B = _mm256_blend_ps(B, BB, 1 + 8 + 64);

               __m256 C1 = _mm256_blend_ps(RC, GC, 1 + 8 + 64);
               C1 = _mm256_blend_ps(C1, BC, 2 + 16 + 128);

               __m256 A = _mm256_permute2f128_ps(A1, C1, 2 << 4);
               __m256 C = _mm256_permute2f128_ps(A1, C1, (3 << 4) | 1);

               {store}'''.format(avx2_template=avx2_template,
                                 store=store3('avx', typ, align, fmtspec,
                                              'A', 'B', 'C'), **fmtspec)
    if typ in ['i32', 'u32']:
        if simd_ext == 'avx2':
            return avx2_template
        else:
            return \
            '''nsimd_store3{a}_avx_f32((f32 *){in0},
                                       _mm256_castsi256_ps({in1}),
                                       _mm256_castsi256_ps({in2}),
                                       _mm256_castsi256_ps({in3}));'''. \
                                       format(**fmtspec)
    if typ == 'f64':
        return \
        '''__m256d invv1 = _mm256_permute_pd({in2}, 1 + 4);
           __m256d A1C0 = _mm256_blend_pd({in1}, {in3}, 1 + 4);
           __m256d A0B1 = _mm256_blend_pd({in1}, invv1, 2 + 8);
           __m256d B0C1 = _mm256_blend_pd(invv1, {in3}, 2 + 8);

           __m256d A = _mm256_permute2f128_pd(A0B1, A1C0, 2 << 4);
           __m256d B = _mm256_blend_pd(B0C1, A0B1, 4 + 8);
           __m256d C = _mm256_permute2f128_pd(A1C0, B0C1, (3 << 4) |  1);

           {store}'''.format(store=store3('avx', typ, align, fmtspec,
                                          'A', 'B', 'C'), **fmtspec)
    if typ in ['i64', 'u64']:
        return \
        '''nsimd_store3{a}_{simd_ext}_f64((f64 *){in0},
                                          _mm256_castsi256_pd({in1}),
                                          _mm256_castsi256_pd({in2}),
                                          _mm256_castsi256_pd({in3}));'''. \
                                          format(**fmtspec)

###############################################################################

def load3_avx512(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['load_v0v1v2'] = get_load_v0v1v2(simd_ext, typ, align, fmtspec)
    fmtspec['exlo_v0'] = x86.extract(simd_ext, typ, x86.LO, 'v0')
    fmtspec['exhi_v0'] = x86.extract(simd_ext, typ, x86.HI, 'v0')
    fmtspec['exlo_v1'] = x86.extract(simd_ext, typ, x86.LO, 'v1')
    fmtspec['exhi_v1'] = x86.extract(simd_ext, typ, x86.HI, 'v1')
    fmtspec['exlo_v2'] = x86.extract(simd_ext, typ, x86.LO, 'v2')
    fmtspec['exhi_v2'] = x86.extract(simd_ext, typ, x86.HI, 'v2')
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        return \
        '''nsimd_{simd_ext}_v{typ}x3 ret;

           {load_v0v1v2}

           __m256i A0in = {exlo_v0};
           __m256i B0in = {exhi_v0};
           __m256i C0in = {exlo_v1};
           __m256i A1in = {exhi_v1};
           __m256i B1in = {exlo_v2};
           __m256i C1in = {exhi_v2};

	   __m256i ARmask = _mm256_setr_epi8( 0,  3,  6,  9, 12, 15, -1, -1,
	                                     -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1,  2,  5,
                                              8, 11, 14, -1, -1, -1, -1, -1);
           __m256i BRmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1,  1,  4,  7, 10, 13,
                                              0,  3,  6,  9, 12, 15, -1, -1,
                                              -1, -1, -1, -1, -1, -1, -1, -1);
           __m256i CRmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1,  2,  5,
                                              8, 11, 14, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1,  1,  4,  7, 10, 13);

           __m256i AR = _mm256_shuffle_epi8(A0in, ARmask);
           __m256i BR = _mm256_shuffle_epi8(B0in, BRmask);
           __m256i CR = _mm256_shuffle_epi8(C0in, CRmask);
           __m256i DR = _mm256_permute2f128_si256(AR, CR, (2 << 4) | 1);

           __m256i R0 = _mm256_or_si256(AR, BR);
           __m256i R1 = _mm256_or_si256(BR, CR);
           __m256i R2 = _mm256_permute2f128_si256(R0, R1, 3 << 4);
           __m256i R3 = _mm256_or_si256(DR, R2);

           AR = _mm256_shuffle_epi8(A1in, ARmask);
           BR = _mm256_shuffle_epi8(B1in, BRmask);
           CR = _mm256_shuffle_epi8(C1in, CRmask);
           DR = _mm256_permute2f128_si256(AR, CR, (2 << 4) | 1);

           R0 = _mm256_or_si256(AR, BR);
           R1 = _mm256_or_si256(BR, CR);
           R2 = _mm256_permute2f128_si256(R0, R1, 3 << 4);
           __m256i R3b = _mm256_or_si256(DR, R2);

           __m256i AGmask = _mm256_setr_epi8( 1,  4,  7, 10, 13, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1,  0,  3,  6,
                                              9, 12, 15, -1, -1, -1, -1, -1);
           __m256i BGmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1,  2,  5,  8, 11, 14,
                                              1,  4,  7, 10, 13, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1);
           __m256i CGmask = _mm256_setr_epi8(-1, -1, -1, -1, -1,  0,  3,  6,
                                              9, 12, 15, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1,  2,  5,  8, 11, 14);

           __m256i AG = _mm256_shuffle_epi8(A0in, AGmask);
           __m256i BG = _mm256_shuffle_epi8(B0in, BGmask);
           __m256i CG = _mm256_shuffle_epi8(C0in, CGmask);
           __m256i DG = _mm256_permute2f128_si256(AG, CG, (2 << 4) | 1);

           __m256i G0 = _mm256_or_si256(AG, BG);
           __m256i G1 = _mm256_or_si256(BG, CG);
           __m256i G2 = _mm256_permute2f128_si256(G0, G1, 3 << 4);
           __m256i G3 = _mm256_or_si256(DG, G2);

           AG = _mm256_shuffle_epi8(A1in, AGmask);
           BG = _mm256_shuffle_epi8(B1in, BGmask);
           CG = _mm256_shuffle_epi8(C1in, CGmask);
           DG = _mm256_permute2f128_si256(AG, CG, (2 << 4) | 1);

           G0 = _mm256_or_si256(AG, BG);
           G1 = _mm256_or_si256(BG, CG);
           G2 = _mm256_permute2f128_si256(G0, G1, 3 << 4);
           __m256i G3b = _mm256_or_si256(DG, G2);

           __m256i ABmask = _mm256_setr_epi8( 2,  5,  8, 11, 14, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1,  1,  4,  7,
                                             10, 13, -1, -1, -1, -1, -1, -1);
           __m256i BBmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1,  0,  3,  6,  9, 12, 15,
                                              2,  5,  8, 11, 14, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1);
           __m256i CBmask = _mm256_setr_epi8(-1, -1, -1, -1, -1,  1,  4,  7,
                                             10, 13, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1,  0,  3,  6,  9, 12, 15);

           __m256i AB = _mm256_shuffle_epi8(A0in, ABmask);
           __m256i BB = _mm256_shuffle_epi8(B0in, BBmask);
           __m256i CB = _mm256_shuffle_epi8(C0in, CBmask);
           __m256i DB = _mm256_permute2f128_si256(AB, CB, (2 << 4) | 1);

           __m256i B0 = _mm256_or_si256(AB, BB);
           __m256i B1 = _mm256_or_si256(BB, CB);
           __m256i B2 = _mm256_permute2f128_si256(B0, B1, 3 << 4);
           __m256i B3 = _mm256_or_si256(DB, B2);

           AB = _mm256_shuffle_epi8(A1in, ABmask);
           BB = _mm256_shuffle_epi8(B1in, BBmask);
           CB = _mm256_shuffle_epi8(C1in, CBmask);
           DB = _mm256_permute2f128_si256(AB, CB, (2 << 4) | 1);

           B0 = _mm256_or_si256(AB, BB);
           B1 = _mm256_or_si256(BB, CB);
           B2 = _mm256_permute2f128_si256(B0, B1, 3 << 4);
           __m256i B3b = _mm256_or_si256(DB, B2);

           ret.v0 = {mergeR};
           ret.v1 = {mergeG};
           ret.v2 = {mergeB};

           return ret;'''. \
           format(mergeR=x86.setr(simd_ext, typ, 'R3', 'R3b'),
                  mergeG=x86.setr(simd_ext, typ, 'G3', 'G3b'),
                  mergeB=x86.setr(simd_ext, typ, 'B3', 'B3b'),
                  **fmtspec)
    if typ in ['i16', 'u16']:
        return \
        '''nsimd_{simd_ext}_v{typ}x3 ret;

           {load_v0v1v2}

           __m256i A0a = {exlo_v0};
           __m256i B0a = {exhi_v0};
           __m256i C0a = {exlo_v1};
           __m256i A0b = {exhi_v1};
           __m256i B0b = {exlo_v2};
           __m256i C0b = {exhi_v2};

           __m256i ARmask = _mm256_setr_epi8( 0,  1,  6,  7, 12, 13, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1,  2,  3,
                                              8,  9, 14, 15, -1, -1, -1, -1);
           __m256i BRmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1,  4,  5, 10, 11,
                                              0,  1,  6,  7, 12, 13, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1);
           __m256i CRmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1,  2,  3,
                                              8,  9, 14, 15, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1,  4,  5, 10, 11);

           __m256i AR = _mm256_shuffle_epi8(A0a, ARmask);
           __m256i BR = _mm256_shuffle_epi8(B0a, BRmask);
           __m256i CR = _mm256_shuffle_epi8(C0a, CRmask);
           __m256i DR = _mm256_permute2f128_si256(AR, CR, (2 << 4) | 1);

           __m256i R0 = _mm256_or_si256(AR, BR);
           __m256i R1 = _mm256_or_si256(BR, CR);
           __m256i R2 = _mm256_permute2f128_si256(R0, R1, 3 << 4);
           __m256i R3a = _mm256_or_si256(DR, R2);

           AR = _mm256_shuffle_epi8(A0b, ARmask);
           BR = _mm256_shuffle_epi8(B0b, BRmask);
           CR = _mm256_shuffle_epi8(C0b, CRmask);
           DR = _mm256_permute2f128_si256(AR, CR, (2 << 4) | 1);

           R0 = _mm256_or_si256(AR, BR);
           R1 = _mm256_or_si256(BR, CR);
           R2 = _mm256_permute2f128_si256(R0, R1, 3 << 4);
           __m256i R3b = _mm256_or_si256(DR, R2);

           __m256i AGmask = _mm256_setr_epi8( 2,  3,  8,  9, 14, 15, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1,  4,  5,
                                             10, 11, -1, -1, -1, -1, -1, -1);
           __m256i BGmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1,  0,  1,  6,  7, 12, 13,
                                              2,  3,  8,  9, 14, 15, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1);
           __m256i CGmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1,  4,  5,
                                             10, 11, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1,  0,  1,  6,  7, 12, 13);

           __m256i AG = _mm256_shuffle_epi8(A0a, AGmask);
           __m256i BG = _mm256_shuffle_epi8(B0a, BGmask);
           __m256i CG = _mm256_shuffle_epi8(C0a, CGmask);
           __m256i DG = _mm256_permute2f128_si256(AG, CG, (2 << 4) | 1);

           __m256i G0 = _mm256_or_si256(AG, BG);
           __m256i G1 = _mm256_or_si256(BG, CG);
           __m256i G2 = _mm256_permute2f128_si256(G0, G1, 3 << 4);
           __m256i G3a = _mm256_or_si256(DG, G2);

           AG = _mm256_shuffle_epi8(A0b, AGmask);
           BG = _mm256_shuffle_epi8(B0b, BGmask);
           CG = _mm256_shuffle_epi8(C0b, CGmask);
           DG = _mm256_permute2f128_si256(AG, CG, (2 << 4) | 1);

           G0 = _mm256_or_si256(AG, BG);
           G1 = _mm256_or_si256(BG, CG);
           G2 = _mm256_permute2f128_si256(G0, G1, 3 << 4);
           __m256i G3b = _mm256_or_si256(DG, G2);

           __m256i ABmask = _mm256_setr_epi8( 4,  5, 10, 11, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1,  0,  1,  6,  7,
                                             12, 13, -1, -1, -1, -1, -1, -1);
           __m256i BBmask = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1,  2,  3,  8,  9, 14, 15,
                                              4,  5, 10, 11, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1);
           __m256i CBmask = _mm256_setr_epi8(-1, -1, -1, -1,  0,  1,  6,  7,
                                             12, 13, -1, -1, -1, -1, -1, -1,
                                             -1, -1, -1, -1, -1, -1, -1, -1,
                                             -1, -1,  2,  3,  8,  9, 14, 15);

           __m256i AB = _mm256_shuffle_epi8(A0a, ABmask);
           __m256i BB = _mm256_shuffle_epi8(B0a, BBmask);
           __m256i CB = _mm256_shuffle_epi8(C0a, CBmask);
           __m256i DB = _mm256_permute2f128_si256(AB, CB, (2 << 4) | 1);

           __m256i B0 = _mm256_or_si256(AB, BB);
           __m256i B1 = _mm256_or_si256(BB, CB);
           __m256i B2 = _mm256_permute2f128_si256(B0, B1, 3 << 4);
           __m256i B3a = _mm256_or_si256(DB, B2);

           AB = _mm256_shuffle_epi8(A0b, ABmask);
           BB = _mm256_shuffle_epi8(B0b, BBmask);
           CB = _mm256_shuffle_epi8(C0b, CBmask);
           DB = _mm256_permute2f128_si256(AB, CB, (2 << 4) | 1);

           B0 = _mm256_or_si256(AB, BB);
           B1 = _mm256_or_si256(BB, CB);
           B2 = _mm256_permute2f128_si256(B0, B1, 3 << 4);
           __m256i B3b = _mm256_or_si256(DB, B2);

           ret.v0 = {mergeR};
           ret.v1 = {mergeG};
           ret.v2 = {mergeB};

           return ret;'''. \
           format(mergeR=x86.setr(simd_ext, typ, 'R3a', 'R3b'),
                  mergeG=x86.setr(simd_ext, typ, 'G3a', 'G3b'),
                  mergeB=x86.setr(simd_ext, typ, 'B3a', 'B3b'),
                  **fmtspec)
    if typ in ['f32', 'i32', 'u32']:
        return \
        '''nsimd_{simd_ext}_v{typ}x3 ret;

           {load_v0v1v2}

           __m512i RABm  = _mm512_setr_epi32( 0,  3,  6,  9, 12, 15, 18, 21,
                                             24, 27, 30,  0,  0,  0 , 0,  0);
           __m512i RABCm = _mm512_setr_epi32( 0,  1,  2,  3,  4,  5,  6 , 7,
                                              8,  9, 10, 17, 20, 23, 26, 29);
           __m512i GABm  = _mm512_setr_epi32( 1,  4,  7, 10, 13, 16, 19, 22,
                                             25, 28, 31,  0,  0,  0 , 0,  0);
           __m512i GABCm = _mm512_setr_epi32( 0,  1,  2,  3,  4,  5,  6 , 7,
                                              8,  9, 10, 18, 21, 24, 27, 30);
           __m512i BABm  = _mm512_setr_epi32( 2,  5,  8, 11, 14, 17, 20, 23,
                                             26, 29,  0,  0,  0,  0 , 0,  0);
           __m512i BABCm = _mm512_setr_epi32( 0,  1,  2,  3,  4,  5,  6 , 7,
                                              8,  9, 16, 19, 22, 25, 28, 31);

           {styp} R = _mm512_permutex2var{suf}(v0, RABm, v1);
           ret.v0 = _mm512_permutex2var{suf}(R, RABCm, v2);
           {styp} G = _mm512_permutex2var{suf}(v0, GABm, v1);
           ret.v1 = _mm512_permutex2var{suf}(G, GABCm, v2);
           {styp} B = _mm512_permutex2var{suf}(v0, BABm, v1);
           ret.v2 = _mm512_permutex2var{suf}(B, BABCm, v2);

           return ret;'''.format(**fmtspec)
    if typ in ['f64', 'i64', 'u64']:
        return \
        '''nsimd_{simd_ext}_v{typ}x3 ret;

           {load_v0v1v2}

           __m512i R_mask0 = _mm512_set_epi64( 0,  0, 15, 12, 9, 6, 3, 0);
           __m512i R_mask1 = _mm512_set_epi64(13, 10,  5,  4, 3, 2, 1, 0);
           {styp} A1 = _mm512_permutex2var{suf}(v0, R_mask0, v1);
           ret.v0 = _mm512_permutex2var{suf}(A1, R_mask1, v2);

           __m512i G_mask0 = _mm512_set_epi64( 0,  0,  0, 13, 10, 7, 4, 1);
           __m512i G_mask1 = _mm512_set_epi64(14, 11,  8,  4,  3, 2, 1, 0);
           {styp} B1 = _mm512_permutex2var{suf}(v0, G_mask0, v1);
           ret.v1 = _mm512_permutex2var{suf}(B1, G_mask1, v2);

           __m512i B_mask0 = _mm512_set_epi64( 0,  0,  0, 14, 11, 8, 5, 2);
           __m512i B_mask1 = _mm512_set_epi64(15, 12,  9,  4,  3, 2, 1, 0);
           {styp} C1 = _mm512_permutex2var{suf}(v0, B_mask0, v1);
           ret.v2 = _mm512_permutex2var{suf}(C1, B_mask1, v2);

           return ret;'''.format(**fmtspec)

###############################################################################

def store3_avx512(simd_ext, typ, align, fmtspec2):
    fmtspec = fmtspec2.copy()
    fmtspec['exlo_in1'] = x86.extract(simd_ext, typ, x86.LO, common.in1)
    fmtspec['exhi_in1'] = x86.extract(simd_ext, typ, x86.HI, common.in1)
    fmtspec['exlo_in2'] = x86.extract(simd_ext, typ, x86.LO, common.in2)
    fmtspec['exhi_in2'] = x86.extract(simd_ext, typ, x86.HI, common.in2)
    fmtspec['exlo_in3'] = x86.extract(simd_ext, typ, x86.LO, common.in3)
    fmtspec['exhi_in3'] = x86.extract(simd_ext, typ, x86.HI, common.in3)
    fmtspec['a'] = 'a' if align else 'u'
    if typ in ['i8', 'u8']:
        return \
        '''__m256i R0 = {exlo_in1};
           __m256i R1 = {exhi_in1};
           __m256i G0 = {exlo_in2};
           __m256i G1 = {exhi_in2};
           __m256i B0 = {exlo_in3};
           __m256i B1 = {exhi_in3};


           __m256i RACm = _mm256_setr_epi8( 0, -1, -1,  1, -1, -1,  2, -1,
                                           -1,  3, -1, -1,  4, -1, -1,  5,
                                           -1, 27, -1, -1, 28, -1, -1, 29,
                                           -1, -1, 30, -1, -1, 31, -1, -1);
           __m256i RBBm = _mm256_setr_epi8(-1, 11, -1, -1, 12, -1, -1, 13,
                                           -1, -1, 14, -1, -1, 15, -1, -1,
                                           16, -1, -1, 17, -1, -1, 18, -1,
                                           -1, 19, -1, -1, 20, -1, -1, 21);
           __m256i RCAm = _mm256_setr_epi8(-1, -1, 22, -1, -1, 23, -1, -1,
                                           24, -1, -1, 25, -1, -1, 26, -1,
                                           -1, -1,  6, -1, -1,  7, -1, -1,
                                            8, -1, -1,  9, -1, -1, 10, -1);

           __m256i GACm = _mm256_setr_epi8(-1,  0, -1, -1,  1, -1, -1,  2,
                                           -1, -1,  3, -1, -1,  4, -1, -1,
                                           -1, -1, 27, -1, -1, 28, -1, -1,
                                           29, -1, -1, 30, -1, -1, 31, -1);
           __m256i GBBm = _mm256_setr_epi8(-1, -1, 11, -1, -1, 12, -1, -1,
                                           13, -1, -1, 14, -1, -1, 15, -1,
                                           -1, 16, -1, -1, 17, -1, -1, 18,
                                           -1, -1, 19, -1, -1, 20, -1, -1);
           __m256i GCAm = _mm256_setr_epi8(21, -1, -1, 22, -1, -1, 23, -1,
                                           -1, 24, -1, -1, 25, -1, -1, 26,
                                           05, -1, -1,  6, -1, -1,  7, -1,
                                           -1,  8, -1, -1,  9, -1, -1, 10);

           __m256i BACm = _mm256_setr_epi8(-1, -1,  0, -1, -1,  1, -1, -1,
                                            2, -1, -1,  3, -1, -1,  4, -1,
                                           26, -1, -1, 27, -1, -1, 28, -1,
                                           -1, 29, -1, -1, 30, -1, -1, 31);
           __m256i BBBm = _mm256_setr_epi8(10, -1, -1, 11, -1, -1, 12, -1,
                                           -1, 13, -1, -1, 14, -1, -1, 15,
                                           -1, -1, 16, -1, -1, 17, -1, -1,
                                           18, -1, -1, 19, -1, -1, 20, -1);
           __m256i BCAm = _mm256_setr_epi8(-1, 21, -1, -1, 22, -1, -1, 23,
                                           -1, -1, 24, -1, -1, 25, -1, -1,
                                           -1,  5, -1, -1,  6, -1, -1,  7,
                                           -1, -1,  8, -1, -1,  9, -1, -1);

           __m256i RAC = _mm256_shuffle_epi8(R0, RACm);
           __m256i GAC = _mm256_shuffle_epi8(G0, GACm);
           __m256i BAC = _mm256_shuffle_epi8(B0, BACm);

           __m256i AC0 = _mm256_or_si256(RAC, GAC);
           AC0 = _mm256_or_si256(AC0, BAC);

           __m256i RBB = _mm256_shuffle_epi8(R0, RBBm);
           __m256i GBB = _mm256_shuffle_epi8(G0, GBBm);
           __m256i BBB = _mm256_shuffle_epi8(B0, BBBm);

           __m256i BB0 = _mm256_or_si256(RBB, GBB);
           BB0 = _mm256_or_si256(BB0, BBB);

           __m256i RCA = _mm256_shuffle_epi8(R0, RCAm);
           __m256i GCA = _mm256_shuffle_epi8(G0, GCAm);
           __m256i BCA = _mm256_shuffle_epi8(B0, BCAm);

           __m256i CA0 = _mm256_or_si256(RCA, GCA);
           CA0 = _mm256_or_si256(CA0, BCA);

           __m256i AA0 = _mm256_permute2f128_si256(AC0, CA0, 2 << 4);
           __m256i CC0 = _mm256_permute2f128_si256(AC0, CA0, (1 << 4) | 3);

           RAC = _mm256_shuffle_epi8(R1, RACm);
           GAC = _mm256_shuffle_epi8(G1, GACm);
           BAC = _mm256_shuffle_epi8(B1, BACm);

           __m256i AC1 = _mm256_or_si256(RAC, GAC);
           AC1 = _mm256_or_si256(AC1, BAC);

           RBB = _mm256_shuffle_epi8(R1, RBBm);
           GBB = _mm256_shuffle_epi8(G1, GBBm);
           BBB = _mm256_shuffle_epi8(B1, BBBm);

           __m256i BB1 = _mm256_or_si256(RBB, GBB);
           BB1 = _mm256_or_si256(BB1, BBB);

           RCA = _mm256_shuffle_epi8(R1, RCAm);
           GCA = _mm256_shuffle_epi8(G1, GCAm);
           BCA = _mm256_shuffle_epi8(B1, BCAm);

           __m256i CA1 = _mm256_or_si256(RCA, GCA);
           CA1 = _mm256_or_si256(CA1, BCA);

           __m256i AA1 = _mm256_permute2f128_si256(AC1, CA1, 2 << 4);
           __m256i CC1 = _mm256_permute2f128_si256(AC1, CA1, (1 << 4) | 3);

           __m512i A = {mergeA0B0};
           __m512i B = {mergeC0A1};
           __m512i C = {mergeB1C1};

           {store}'''. \
           format(mergeA0B0=x86.setr(simd_ext, typ, 'AA0', 'BB0'),
                  mergeC0A1=x86.setr(simd_ext, typ, 'CC0', 'AA1'),
                  mergeB1C1=x86.setr(simd_ext, typ, 'BB1', 'CC1'),
                  store=store3(simd_ext, typ, align, fmtspec, 'A', 'B', 'C'),
                  **fmtspec)
    if typ in ['i16', 'u16']:
        return \
        '''__m256i R0a = {exlo_in1};
           __m256i R0b = {exhi_in1};
           __m256i G0a = {exlo_in2};
           __m256i G0b = {exhi_in2};
           __m256i B0a = {exlo_in3};
           __m256i B0b = {exhi_in3};

           __m256i RACm = _mm256_setr_epi8( 0,  1, -1, -1, -1, -1,  2,  3,
                                           -1, -1, -1, -1,  4,  5, -1, -1,
                                           -1, -1, -1, -1, 12, 13, -1, -1,
                                           -1, -1, 14, 15, -1, -1, -1, -1);
           __m256i RBBm = _mm256_setr_epi8(-1, -1, -1, -1, 12, 13, -1, -1,
                                           -1, -1, 14, 15, -1, -1, -1, -1,
                                            0,  1, -1, -1, -1, -1,  2,  3,
                                           -1, -1, -1, -1,  4,  5, -1, -1);
           __m256i RCAm = _mm256_setr_epi8(-1, -1,  6,  7, -1, -1, -1, -1,
                                            8,  9, -1, -1, -1, -1, 10, 11,
                                           -1, -1,  6,  7, -1, -1, -1, -1,
                                            8,  9, -1, -1, -1, -1, 10, 11);

           __m256i GACm = _mm256_setr_epi8(-1, -1,  0,  1, -1, -1, -1, -1,
                                            2,  3, -1, -1, -1, -1,  4,  5,
                                           10, 11, -1, -1, -1, -1, 12, 13,
                                           -1, -1, -1, -1, 14, 15, -1, -1);
           __m256i GBBm = _mm256_setr_epi8(10, 11, -1, -1, -1, -1, 12, 13,
                                           -1, -1, -1, -1, 14, 15, -1, -1,
                                           -1, -1,  0,  1, -1, -1, -1, -1,
                                            2,  3, -1, -1, -1, -1,  4,  5);
           __m256i GCAm = _mm256_setr_epi8(-1, -1, -1, -1,  6,  7, -1, -1,
                                           -1, -1,  8,  9, -1, -1, -1, -1,
                                           -1, -1, -1, -1,  6,  7, -1, -1,
                                           -1, -1,  8,  9, -1, -1, -1, -1);

           __m256i BACm = _mm256_setr_epi8(-1, -1, -1, -1,  0,  1, -1, -1,
                                           -1, -1,  2,  3, -1, -1, -1, -1,
                                           -1, -1, 10, 11, -1, -1, -1, -1,
                                           12, 13, -1, -1, -1, -1, 14, 15);
           __m256i BBBm = _mm256_setr_epi8(-1, -1, 10, 11, -1, -1, -1, -1,
                                           12, 13, -1, -1, -1, -1, 14, 15,
                                           -1, -1, -1, -1,  0,  1, -1, -1,
                                           -1, -1,  2,  3, -1, -1, -1, -1);
           __m256i BCAm = _mm256_setr_epi8( 4,  5, -1, -1, -1, -1,  6,  7,
                                           -1, -1, -1, -1,  8,  9, -1, -1,
                                            4,  5, -1, -1, -1, -1,  6,  7,
                                           -1, -1, -1, -1,  8,  9, -1, -1);

           __m256i RAC = _mm256_shuffle_epi8(R0a, RACm);
           __m256i GAC = _mm256_shuffle_epi8(G0a, GACm);
           __m256i BAC = _mm256_shuffle_epi8(B0a, BACm);

           __m256i RBB = _mm256_shuffle_epi8(R0a, RBBm);
           __m256i GBB = _mm256_shuffle_epi8(G0a, GBBm);
           __m256i BBB = _mm256_shuffle_epi8(B0a, BBBm);

           __m256i RCA = _mm256_shuffle_epi8(R0a, RCAm);
           __m256i GCA = _mm256_shuffle_epi8(G0a, GCAm);
           __m256i BCA = _mm256_shuffle_epi8(B0a, BCAm);

           __m256i AC = _mm256_or_si256(RAC, GAC);
           AC = _mm256_or_si256(AC, BAC);

           __m256i BBa = _mm256_or_si256(RBB, GBB);
           BBa = _mm256_or_si256(BBa, BBB);

           __m256i CA = _mm256_or_si256(RCA, GCA);
           CA = _mm256_or_si256(CA, BCA);

           __m256i AAa = _mm256_permute2f128_si256(AC, CA, 2 << 4);
           __m256i CCa = _mm256_permute2f128_si256(AC, CA, (1 << 4) | 3);

           RAC = _mm256_shuffle_epi8(R0b, RACm);
           GAC = _mm256_shuffle_epi8(G0b, GACm);
           BAC = _mm256_shuffle_epi8(B0b, BACm);

           RBB = _mm256_shuffle_epi8(R0b, RBBm);
           GBB = _mm256_shuffle_epi8(G0b, GBBm);
           BBB = _mm256_shuffle_epi8(B0b, BBBm);

           RCA = _mm256_shuffle_epi8(R0b, RCAm);
           GCA = _mm256_shuffle_epi8(G0b, GCAm);
           BCA = _mm256_shuffle_epi8(B0b, BCAm);

           AC = _mm256_or_si256(RAC, GAC);
           AC = _mm256_or_si256(AC, BAC);

           __m256i BBb = _mm256_or_si256(RBB, GBB);
           BBb = _mm256_or_si256(BBb, BBB);

           CA = _mm256_or_si256(RCA, GCA);
           CA = _mm256_or_si256(CA, BCA);

           __m256i AAb = _mm256_permute2f128_si256(AC, CA, 2 << 4);
           __m256i CCb = _mm256_permute2f128_si256(AC, CA, (1 << 4) | 3);

           __m512i A = {mergeAaBa};
           __m512i B = {mergeCaAb};
           __m512i C = {mergeBbCb};

           {store}'''. \
           format(mergeAaBa=x86.setr(simd_ext, typ, 'AAa', 'BBa'),
                  mergeCaAb=x86.setr(simd_ext, typ, 'CCa', 'AAb'),
                  mergeBbCb=x86.setr(simd_ext, typ, 'BBb', 'CCb'),
                  store=store3(simd_ext, typ, align, fmtspec, 'A', 'B', 'C'),
                  **fmtspec)
    if typ in ['f32', 'i32', 'u32']:
        return \
        '''__m512i ARGm  = _mm512_setr_epi32( 0, 16,  0,  1, 17,  0,  2, 18,
                                              0,  3, 19,  0,  4, 20,  0,  5);
           __m512i ARGBm = _mm512_setr_epi32( 0,  1, 16,  3,  4, 17,  6,  7,
                                             18,  9, 10, 19, 12, 13, 20, 15);
           __m512i BRGm  = _mm512_setr_epi32(21,  0,  6, 22,  0,  7, 23,  0,
                                              8, 24,  0,  9, 25,  0, 10, 26);
           __m512i BRGBm = _mm512_setr_epi32( 0, 21,  2,  3, 22,  5,  6, 23,
                                              8,  9, 24, 11, 12, 25, 14, 15);
           __m512i CRGm  = _mm512_setr_epi32( 0, 11, 27,  0, 12, 28,  0, 13,
                                             29,  0, 14, 30,  0, 15, 31,  0);
           __m512i CRGBm = _mm512_setr_epi32(26,  1,  2, 27,  4,  5, 28,  7,
                                              8, 29, 10, 11, 30, 13, 14, 31);

           {styp} A = _mm512_permutex2var{suf}({in1}, ARGm, {in2});
           A = _mm512_permutex2var{suf}(A, ARGBm, {in3});
           {styp} B = _mm512_permutex2var{suf}({in1}, BRGm, {in2});
           B = _mm512_permutex2var{suf}(B, BRGBm, {in3});
           {styp} C = _mm512_permutex2var{suf}({in1}, CRGm, {in2});
           C = _mm512_permutex2var{suf}(C, CRGBm, {in3});

           {store}'''. \
           format(store=store3(simd_ext, typ, align, fmtspec, 'A', 'B', 'C'),
                  **fmtspec)
    if typ in ['f64', 'i64', 'u64']:
        return \
        '''__m512i A_mask0 = _mm512_set_epi64(10,  2,  0,  9,  1,  0,  8,  0);
           __m512i A_mask1 = _mm512_set_epi64( 7,  6,  9,  4,  3,  8,  1,  0);
           {styp} A1 = _mm512_permutex2var{suf}({in1}, A_mask0, {in2});
           {styp} A2 = _mm512_permutex2var{suf}(A1, A_mask1, {in3});

           __m512i B_mask0 = _mm512_set_epi64( 5,  0, 12,  4,  0, 11,  3,  0);
           __m512i B_mask1 = _mm512_set_epi64( 7, 12,  5,  4, 11,  2,  1, 10);
           {styp} B1 = _mm512_permutex2var{suf}({in1}, B_mask0, {in2});
           {styp} B2 = _mm512_permutex2var{suf}(B1, B_mask1, {in3});

           __m512i C_mask0 = _mm512_set_epi64( 0, 15,  7,  0, 14,  6,  0, 13);
           __m512i C_mask1 = _mm512_set_epi64(15,  6,  5, 14,  3,  2, 13,  0);
           {styp} C1 = _mm512_permutex2var{suf}({in1}, C_mask0, {in2});
           {styp} C2 = _mm512_permutex2var{suf}(C1, C_mask1, {in3});

           {store}'''. \
           format(store=store3(simd_ext, typ, align, fmtspec,
                  'A2', 'B2', 'C2'), **fmtspec)
