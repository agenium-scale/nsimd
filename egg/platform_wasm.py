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

# -----------------------------------------------------------------------------
# Implementation of mandatory functions for this module


def get_simd_exts():
    return ['wasm_simd128']


def emulate_fp16(simd_ext):
    if not simd_ext in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    return True


def get_native_typ(simd_ext, typ):
    if simd_ext != 'wasm_simd128':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    return 'v128_t'


def get_type(opts, simd_ext, typ, nsimd_typ):
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    if typ == 'f16':
        return 'typedef struct {{ {t} v0; {t} v1; }} {nsimd_typ};'. \
               format(t=get_native_typ(simd_ext, 'f32'), nsimd_typ=nsimd_typ)
    else:
        return 'typedef {} {};'.format(get_native_typ(simd_ext, typ),
                                       nsimd_typ)


def get_logical_type(opts, simd_ext, typ, nsimd_typ):
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    return get_type(opts, simd_ext, typ, nsimd_typ)


def get_nb_registers(simd_ext):
    return '1'


def has_compatible_SoA_types(simd_ext):
    if simd_ext != 'wasm_simd128':
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    else:
        return False


def get_additional_include(func, platform, simd_ext):
    ret = ''
    if simd_ext == 'wasm_simd128':
        ret += '''#include <nsimd/cpu/cpu/{}.h>
                  '''.format(func)
    return ret

# -----------------------------------------------------------------------------
# Function prefixes

def pre(simd_ext):
    return 'wasm_v128_'

def pretyp(simd_ext, typ):
    return 'wasm_{}x{}_'.format(typ, 128 // int(typ[1:]))

def nbits(simd_ext):
    return '128'

# -----------------------------------------------------------------------------
# Other helper functions

fmtspec = {}

def set_lane(simd_ext, typ, var_name, scalar, i):
    # No code for f16's
    if typ == 'f16':
        return ''
    if typ in common.itypes + common.ftypes:
        return '{} = {}replace_lane({}, {}, {});'. \
               format(var_name, fmtspec['pre'], var_name, i, scalar)
    else :
        return \
        '{} = {}replace_lane({}, {}, nsimd_scalar_reinterpret_i{}_{}({}))'. \
        format(var_name, fmtspec['pre'], var_name, i, fmtspec['typnbits'], typ,
               scalar)

def get_lane(simd_ext, typ, var_name, i):
    # No code for f16's
    if typ == 'f16':
        return ''
    if typ in common.itypes + ['u8', 'u16'] + common.ftypes:
        return '{}extract_lane({}, {})'.format(fmtspec['pre'], var_name, i)
    return 'nsimd_scalar_reinterpret_u{}_{}({}replace_lane({}, {}))'. \
           format(fmtspec['typnbits'], typ, pretyp(simd_ext,
                  'i' + fmtspec['typnbits']), var_name, i)

# Signature must be a list of 'v', 's'
#   'v' means vector so code to extract has to be emitted
#   's' means base type so no need to write code for extraction
def get_emulation_code(func, signature, simd_ext, typ):
    ret = 'nsimd_{simd_ext}_v{typ} ret;\n'.format(**fmtspec)
    arity = len(signature)
    ret += typ + ' ' + \
           ', '.join(['tmp{}'.format(i) \
                      for i in range(arity) if signature[i] == 'v']) + ';\n'
    args = ', '.join(['{{in{}}}'.format(i).format(**fmtspec) \
                      if signature[i] == 's' else 'tmp{}'.format(i) \
                      for i in range(arity)])
    for i in range(fmtspec['le']):
        ret += '\n'.join(['tmp{} = {};'. \
               format(j, get_lane(simd_ext, typ,
                      '{{in{}}}'.format(j).format(**fmtspec), i)) \
                      for j in range(arity) if signature[j] == 'v']) + '\n'
        ret += set_lane(simd_ext, typ, 'ret',
                        'nsimd_scalar_{func}_{typ}({args})'. \
                        format(func=func, args=args, **fmtspec), i) + '\n'
    ret += 'return ret;'
    return ret

def how_it_should_be_op1(func, intrin, simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = {pre}{func}_ps({in0}.v0);
                  ret.v1 = {pre}{func}_ps({in0}.v1);
                  return ret;'''.format(func=func, **fmtspec)
    else:
        return 'return {pre}{intrin}({in0});'. \
               format(intrin=intrin, **fmtspec)

def how_it_should_be_op2(func, intrin, simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = {pre}{func}_ps({in0}.v0, {in1}.v0);
                  ret.v1 = {pre}{func}_ps({in0}.v1, {in1}.v1);
                  return ret;'''.format(func=func, **fmtspec)
    else:
        return 'return {pre}{intrin}({in0}, {in1});'. \
               format(intrin=intrin, **fmtspec)

# -----------------------------------------------------------------------------
# Returns C code for func

# Load

def load(simd_ext, typ):
    if typ == 'f16':
        def helper(var_name, i_src, i_dst):
            scalar = 'nsimd_u16_to_f32({})'. \
                     format(get_lane(simd_ext, 'u16', 'buf', i_src))
            return set_lane(simd_ext, 'f32', var_name, scalar, i_dst)
        return \
        '''nsimd_{simd_ext}_vf16 ret = ret;
           v128_t buf = wasm_v128_load((void *){in0});
           {fill_v0}
           {fill_v1}
           return ret;'''.format(
           fill_v0='\n'.join([helper('ret.v0', i, i) for i in range(4)]),
           fill_v1='\n'.join([helper('ret.v1', i + 4, i) for i in range(4)]),
           **fmtspec)
    else:
        return 'wasm_v128_load((void *){in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# masked loads

def maskoz_load(simd_ext, typ, oz, aligned):
    if typ == 'f16':
        def helper(dst, i_dst, i_src):
            mask = get_lane(simd_ext, typ, common.in0, i_src)
            set_zero = set_lane(simd_ext, typ, dst, '0.0', i_dst)
            set_other = set_lane(simd_ext, typ, dst, common.in2, i_dst)
            set_value = set_lane(simd_ext, typ, dst,
                                 'nsimd_f32_to_u16({}[{}])'. \
                                 format(common.in1, i_src), i_dst)
            return '''if ({}) {{
                        {}
                      }} else {{
                        {}
                      }}'''.format(mask, set_value,
                                   set_zero if oz == 'z' else set_other)
        return \
        '''nsimd_{simd_ext}_vf16 ret = ret;
           {fill_ret_v0}
           {fill_ret_v1}
           return ret'''.format(
           fill_ret0='\n'.join([helper('ret.v0', i, i) for i in range(4)]),
           fill_ret1='\n'.join([helper('ret.v1', i, i + 4) for i in range(4)]),
           **fmtspec)
    else:
        def helper(i):
            mask = get_lane(simd_ext, typ, common.in0, i)
            set_zero = set_lane(simd_ext, typ, 'ret', '({})0'.format(typ), i)
            set_other = set_lane(simd_ext, typ, 'ret', common.in2, i)
            set_value = set_lane(simd_ext, typ, 'ret',
                                 '{}[{}]'.format(common.in1, i), i)
            return '''if ({}) {{
                        {}
                      }} else {{
                        {}
                      }}'''.format(mask, set_value,
                                   set_zero if oz == 'z' else set_other)
        return '''nsimd_{simd_ext}_v{typ} ret = ret;
                  {fill_ret}
                  return ret'''. \
                  format(fill_ret='\n'.join([helper(i) for i in range(4)]),
                         **fmtspec)

# -----------------------------------------------------------------------------
# Loads of degree 2, 3 and 4

def load_deg234(simd_ext, typ, align, deg):
    if deg == 2:
        seq = list(range(128 // int(typ[1:])))
        seq_even = ', '.join([2 * i for i in seq])
        seq_odd = ', '.join([2 * i + 1 for i in seq])
        return '''nsimd_{simd_ext}_v{typ}x2 ret;
                  v128_t a = wasm_v128_load((void *){in0});
                  v128_t b = wasm_v128_load((void *)({in0} + {le}));
                  ret.v0 = wasm_v{typnbits}x{le}_shuffle(a, b, {seq_even});
                  ret.v1 = wasm_v{typnbits}x{le}_shuffle(a, b, {seq_odd});
                  return ret;'''.format(seq_even=seq_even, seq_odd=seq_odd,
                                        **fmtspec)
    if deg == 3:
        load_block = '''v128_t a = wasm_v128_load((void *){in0});
                        v128_t b = wasm_v128_load((void *)({in0} + 2));
                        v128_t c = wasm_v128_load((void *)({in0} + 4));'''
        shuffle64 = '''ret.v0 = wasm_v64x2_shuffle({a}, {b}, 0, 3);
                       ret.v1 = wasm_v64x2_shuffle({a}, {c}, 1, 2);
                       ret.v2 = wasm_v64x2_shuffle({b}, {c}, 0, 3);'''
        shuffle32 = \
        '''v128_t rrgg = wasm_v32x4_shuffle({a}, {b}, 0, 3, 1, 4);
           v128_t bbrr = wasm_v32x4_shuffle({a}, {b}, 2, 5, 6, 6);
           v128_t bbrr = wasm_v32x4_shuffle(bbrr, {c}, 0, 1, 2, 5);
           v128_t ggbb = wasm_v32x4_shuffle({b}, {c}, 3, 6, 4, 7);'''
        shuffle16 = \
        '''v128_t rrggbbrr = wasm_v16x8_shuffle({a}, {b},
                                                0, 3, 1, 4, 2, 5, 6, 9);
           v128_t ggbbrrgg = wasm_v16x8_shuffle({b}, {c},
                                                0, 2, 0, 3, 4, 7, 5, 8);
           v128_t ggbbrrgg = wasm_v16x8_shuffle({a}, ggbbrrgg,
                                                7, 9, 10, 11, 12, 13, 14, 15);
           v128_t bbrrggbb = wasm_v16x8_shuffle(
                                 {b}, {c}, 6, 9, 10, 13, 11, 14, 12, 15);'''
        shuffle8 = \
        '''v128_t rrggbbrrggbbrrgg = wasm_v8x16_shuffle({a}, {b},
             0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11, 12, 15, 13, 16);
           v128_t rrggbbrrggbbrrgg = wasm_v8x16_shuffle({b}, {c},
             2, 5, 3, 6, 4, 7, 8, 11, 9, 12, 10, 13, 14, 17, 15, 18);
           '''
    if deg == 4:
        if typ in ['i64', 'u64', 'f64']:
            return '''nsimd_{simd_ext}_v{typ}x4 ret;
                      v128_t a = wasm_v128_load((void *){in0});
                      v128_t b = wasm_v128_load((void *)({in0} + 2));
                      v128_t c = wasm_v128_load((void *)({in0} + 4));
                      v128_t d = wasm_v128_load((void *)({in0} + 6));
                      ret.v0 = wasm_v64x2_shuffle(a, c, 0, 2);
                      ret.v1 = wasm_v64x2_shuffle(a, c, 1, 3);
                      ret.v2 = wasm_v64x2_shuffle(b, d, 0, 2);
                      ret.v3 = wasm_v64x2_shuffle(b, d, 1, 3);
                      return ret;'''.format(seq_even=seq_even, seq_odd=seq_odd,
                                            **fmtspec)
        else:
            seq = list(range(128 // int(typ[1:]) // 2))
            seq_ab = ', '.join([4 * i for i in seq] + [4 * i + 1 for i in seq])
            seq_cd = ', '.join([4 * i + 2 for i in seq] + \
                               [4 * i + 3 for i in seq])
            lex2 = 2 * (128 // int(typ[1:]))
            lex3 = 3 * (128 // int(typ[1:]))
            return \
            '''nsimd_{simd_ext}_v{typ}x4 ret;
               v128_t a = wasm_v128_load((void *){in0});
               v128_t b = wasm_v128_load((void *)({in0} + {le}));
               v128_t c = wasm_v128_load((void *)({in0} + {lex2}));
               v128_t d = wasm_v128_load((void *)({in0} + {lex3}));
               v128_t ab0 = wasm_v{typnbits}x{le}_shuffle(a, b, {seq_ab});
               v128_t ab1 = wasm_v{typnbits}x{le}_shuffle(c, d, {seq_ab});
               v128_t cd0 = wasm_v{typnbits}x{le}_shuffle(a, b, {seq_cd});
               v128_t cd1 = wasm_v{typnbits}x{le}_shuffle(c, d, {seq_cd});
               ret.v0 = wasm_v64x2_shuffle(ab0, ab1, 0, 2);
               ret.v1 = wasm_v64x2_shuffle(ab0, ab1, 1, 3);
               ret.v2 = wasm_v64x2_shuffle(cd0, cd1, 0, 2);
               ret.v3 = wasm_v64x2_shuffle(cd0, cd1, 1, 3);
               return ret;'''.format(seq_ab=seq_ab, seq_cd=seq_cd, lex2=lex2,
                                     lex3=lex3, **fmtspec)

# -----------------------------------------------------------------------------
# Stores of degree 2, 3 and 4

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
# Store

# TODO to implements
def store(simd_ext, typ, aligned):
    align = '' if aligned else 'u'
    cast = castsi(simd_ext, typ)
    if typ == 'f16':
        if simd_ext in sse:
            return \
            '''#ifdef NSIMD_FP16
                 __m128i v0 = _mm_cvtps_ph({in1}.v0, 4);
                 __m128i v1 = _mm_cvtps_ph({in1}.v1, 4);
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
                   _mm256_cvtps_ph({in1}.v0, 4));
                 _mm_store{align}_si128((__m128i*){in0} + 1,
                   _mm256_cvtps_ph({in1}.v1, 4));
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
                   _mm512_cvtps_ph({in1}.v0, 4));
               _mm256_store{align}_si256((__m256i*){in0} + 1,
                   _mm512_cvtps_ph({in1}.v1, 4));'''. \
                        format(align=align, **fmtspec)
    else:
        return '{pre}store{align}{sufsi}({cast}{in0}, {in1});'. \
               format(align=align, cast=cast, **fmtspec)

# masked store

def mask_store(simd_ext, typ, aligned):
    if typ == 'f16':
        le2 = fmtspec['le'] // 2
        if simd_ext in sse + avx:
            store_mask = '''{pre}storeu_ps(mask, {in0}.v0);
                            {pre}storeu_ps(mask + {le2}, {in0}.v1);'''. \
                            format(le2=le2, **fmtspec)
        else:
            store_mask = '''_mm512_storeu_ps(mask, _mm512_maskz_mov_ps(
                              {in0}.v0, _mm512_set1_ps(1.0f)));
                            _mm512_storeu_ps(mask + {le2}, _mm512_maskz_mov_ps(
                              {in0}.v1, _mm512_set1_ps(1.0f)));'''. \
                            format(le2=le2, **fmtspec)
        return '''f32 mask[{le}], buf[{le}];
                  int i;
                  {store_mask}
                  {pre}storeu_ps(buf, {in2}.v0);
                  {pre}storeu_ps(buf + {le2}, {in2}.v1);
                  for (i = 0; i < {le}; i++) {{
                    if (nsimd_scalar_reinterpret_u32_f32(mask[i]) != (u32)0) {{
                      {in1}[i] = nsimd_f32_to_f16(buf[i]);
                    }}
                  }}'''.format(store_mask=store_mask, le2=le2, **fmtspec)
    suf2 = 'ps' if typ[1:] == '32' else 'pd'
    if simd_ext in sse:
        if typ in common.iutypes:
            return '_mm_maskmoveu_si128({in2}, {in0}, (char *){in1});'. \
                   format(**fmtspec)
        else:
            return '''_mm_maskmoveu_si128(_mm_cast{suf2}_si128({in2}),
                                          _mm_cast{suf2}_si128({in0}),
                                          (char *){in1});'''. \
                                          format(suf2=suf2, **fmtspec)
    if typ in ['i8', 'u8', 'i16', 'u16'] and simd_ext != 'avx512_skylake':
        if simd_ext == 'avx512_knl':
            return \
            '''int i;
               u64 mask;
               {typ} buf[{le}];
               {pre}storeu{sufsi}((__m512i *)buf, {in2});
               mask = (u64){in0};
               for (i = 0; i < {le}; i++) {{
                 if ((mask >> i) & 1) {{
                   {in1}[i] = buf[i];
                 }}
               }}'''.format(utyp='u' + typ[1:], **fmtspec)
        else:
            return \
            '''nsimd_{op_name}_sse42_{typ}({mask_lo}, {in1}, {val_lo});
               nsimd_{op_name}_sse42_{typ}({mask_hi}, {in1} + {le2},
                                           {val_hi});
               '''.format(le2=fmtspec['le'] // 2,
               op_name='mask_store{}1'.format('a' if  aligned else 'u'),
               mask_lo=extract(simd_ext, typ, LO, common.in0),
               mask_hi=extract(simd_ext, typ, HI, common.in0),
               val_lo=extract(simd_ext, typ, LO, common.in2),
               val_hi=extract(simd_ext, typ, HI, common.in2), **fmtspec)
    # Here typ is 32 of 64-bits wide except
    if simd_ext in avx:
        if typ in common.ftypes:
            return '''{pre}maskstore{suf}({in1},
                          {pre}cast{suf2}_si256({in0}), {in2});'''. \
                          format(suf2=suf2, **fmtspec)
        else:
            if simd_ext == 'avx2':
                return '{pre}maskstore{suf}({cast}{in1}, {in0}, {in2});'. \
                       format(cast='(nsimd_longlong *)' \
                              if typ in ['i64', 'u64'] \
                              else '(int *)', **fmtspec)
            else:
                return '''{pre}maskstore_{suf2}(({ftyp}*){in1}, {in0},
                            {pre}castsi256_{suf2}({in2}));'''. \
                            format(suf2=suf2, ftyp='f' + typ[1:], **fmtspec)
    # getting here means avx512 with intrinsics
    code = '{pre}mask_store{{}}{suf}((void*){in1}, {in0}, {in2});'. \
           format(**fmtspec)
    if typ in ['i32', 'u32', 'f32', 'i64', 'u64', 'f64']:
        return code.format('' if aligned else 'u')
    else:
        return code.format('u')

# -----------------------------------------------------------------------------
# Code for binary operators: and, or, xor

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
# Code for logical binary operators: andl, orl, xorl

def binlop2(func, simd_ext, typ):
    op = { 'orl': '|', 'xorl': '^', 'andl': '&' }
    op_fct = { 'orl': 'kor', 'xorl': 'kxor', 'andl': 'kand' }
    if simd_ext not in avx512:
        if typ == 'f16':
            return binop2(func, simd_ext, typ, True)
        else:
            return binop2(func, simd_ext, typ)
    elif simd_ext == 'avx512_knl':
        if typ == 'f16':
            return '''nsimd_{simd_ext}_vlf16 ret;
                      ret.v0 = _{op_fct}_mask16({in0}.v0, {in1}.v0);
                      ret.v1 = _{op_fct}_mask16({in0}.v1, {in1}.v1);
                      return ret;'''. \
                      format(op_fct=op_fct[func], **fmtspec)
        elif typ in ['f32', 'u32', 'i32']:
            return 'return _{op_fct}_mask16({in0}, {in1});'. \
                   format(op_fct=op_fct[func], **fmtspec)
        else:
            return 'return (__mmask{le})({in0} {op} {in1});'. \
                   format(op=op[func], **fmtspec)
    elif simd_ext == 'avx512_skylake':
        if typ == 'f16':
            return '''nsimd_{simd_ext}_vlf16 ret;
                      #if defined(NSIMD_IS_GCC) || defined(NSIMD_IS_CLANG)
                        ret.v0 = (__mmask16)({in0}.v0 {op} {in1}.v0);
                        ret.v1 = (__mmask16)({in0}.v1 {op} {in1}.v1);
                      #else
                        ret.v0 = _{op_fct}_mask16({in0}.v0, {in1}.v0);
                        ret.v1 = _{op_fct}_mask16({in0}.v1, {in1}.v1);
                      #endif
                      return ret;'''. \
                      format(op_fct=op_fct[func], op=op[func], **fmtspec)
        else:
            return '''#if defined(NSIMD_IS_GCC) || defined(NSIMD_IS_CLANG)
                        return (__mmask{le})({in0} {op} {in1});
                      #else
                        return _{op_fct}_mask{le}({in0}, {in1});
                      #endif'''.format(op_fct=op_fct[func], op=op[func],
                                       **fmtspec)

# -----------------------------------------------------------------------------
# andnot

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
# logical andnot

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
# Code for unary not

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
# Code for unary logical lnot

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
# Addition and substraction

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
# Len

def len1(simd_ext, typ):
    return 'return {le};'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Division

def div2(opts, simd_ext, typ):
    if typ in common.ftypes:
        return how_it_should_be_op2('div', simd_ext, typ)
    return emulate_op2(opts, '/', simd_ext, typ)

# -----------------------------------------------------------------------------
# Multiplication

def mul2(opts, simd_ext, typ):
    emulate = emulate_op2(opts, '*', simd_ext, typ)
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
        return emulate_op2(opts, '*', simd_ext, typ)
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
# Shift left and right

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
# Arithmetic shift right

def shra(opts, simd_ext, typ):
    if typ in common.utypes:
        # For unsigned type, logical shift
        return 'return nsimd_shr_{simd_ext}_{typ}({in0}, {in1});'. \
               format(**fmtspec)

    intrinsic = 'return {pre}sra{suf}({in0}, _mm_set1_epi64x((i64){in1}));'. \
                format(**fmtspec)

    simd_ext2 = 'sse42' if simd_ext in avx else 'avx2'
    split = '''nsimd_{simd_ext2}_v{typ} v0 = {extract_lo};
               nsimd_{simd_ext2}_v{typ} v1 = {extract_hi};
               v0 = nsimd_shra_{simd_ext2}_{typ}(v0, {in1});
               v1 = nsimd_shra_{simd_ext2}_{typ}(v1, {in1});
               return {merge};'''. \
               format(simd_ext2=simd_ext2,
                      extract_lo=extract(simd_ext, typ, LO, common.in0),
                      extract_hi=extract(simd_ext, typ, HI, common.in0),
                      merge=setr(simd_ext, typ, 'v0', 'v1'), **fmtspec)

    trick_for_i8 = \
    '''__m128i count = _mm_set1_epi64x((i64){in1});
       nsimd_{simd_ext}_vi16 lo, hi;
       hi = {pre}andnot{sufsi}({pre}set1_epi16(255),
                               {pre}sra_epi16({in0}, count));
       lo = {pre}srli_epi16({pre}sra_epi16(
                {pre}slli_epi16({in0}, 8), count), 8);
       return {pre}or{sufsi}(hi, lo);'''.format(**fmtspec)

    emulation = get_emulation_code('shra', ['v', 's'], simd_ext, typ)

    if simd_ext in sse + ['avx2']:
        if typ == 'i8':
            return trick_for_i8
        elif typ in ['i16', 'i32']:
            return intrinsic
        elif typ == 'i64':
            return emulation
    elif simd_ext == 'avx':
        if typ in ['i8', 'i16', 'i32']:
            return split
        elif typ == 'i64':
            return emulation
    elif simd_ext == 'avx512_knl':
        if typ in ['i8', 'i16']:
            return split
        elif typ in ['i32', 'i64']:
            return intrinsic
    elif simd_ext == 'avx512_skylake':
        if typ == 'i8':
            return trick_for_i8
        elif typ in ['i16', 'i32', 'i64']:
            return intrinsic

# -----------------------------------------------------------------------------
# set1 or splat function

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
# set1l or splat function for logical

def set1l(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vlf16 ret;
                  ret.v0 = nsimd_set1l_{simd_ext}_f32({in0});
                  ret.v1 = ret.v0;
                  return ret;'''.format(**fmtspec)
    if simd_ext in sse + avx:
        if simd_ext in sse:
            ones = '_mm_cmpeq_pd(_mm_setzero_pd(), _mm_setzero_pd())'
        else:
            ones = '_mm256_cmp_pd(_mm256_setzero_pd(), _mm256_setzero_pd(), ' \
                   '_CMP_EQ_OQ)'
        if typ != 'f64':
            ones = '{pre}castpd{sufsi}({ones})'.format(ones=ones, **fmtspec)
        return '''if ({in0}) {{
                    return {ones};
                  }} else {{
                    return {pre}setzero{sufsi}();
                  }}'''.format(ones=ones, **fmtspec)
    else:
        return '''if ({in0}) {{
                    return (__mmask{le})(~(__mmask{le})(0));
                  }} else {{
                    return (__mmask{le})(0);
                  }}'''.format(**fmtspec)

# -----------------------------------------------------------------------------
# Equality

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
# not equal

def neq2(simd_ext, typ):
    if typ == 'f16':
        return f16_cmp2('ne', simd_ext)
    if simd_ext in sse and typ in ['f32', 'f64']:
        return how_it_should_be_op2('cmpneq', simd_ext, typ)
    if simd_ext in avx and typ in ['f32', 'f64']:
        return 'return _mm256_cmp{suf}({in0}, {in1}, _CMP_NEQ_UQ);'. \
               format(**fmtspec)
    if simd_ext in avx512 and typ in ['f32', 'f64']:
        return 'return _mm512_cmp{suf}_mask({in0}, {in1}, _CMP_NEQ_UQ);'. \
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
# Greater than

def gt2(simd_ext, typ):
    if typ == 'f16':
        return f16_cmp2('gt', simd_ext)
    if simd_ext in sse:
        if typ in ['f32', 'f64', 'i8', 'i16', 'i32']:
            return how_it_should_be_op2('cmpgt', simd_ext, typ)
        if typ == 'i64':
            if simd_ext == 'sse42':
                return how_it_should_be_op2('cmpgt', simd_ext, typ)
            #return '''return _mm_sub_epi64(_mm_setzero_si128(), _mm_srli_epi64(
            #                   _mm_sub_epi64({in1}, {in0}), 63));'''. \
            #                   format(**fmtspec)
            return '''{typ} buf0[2], buf1[2];

                      _mm_storeu_si128((__m128i*)buf0, {in0});
                      _mm_storeu_si128((__m128i*)buf1, {in1});

                      buf0[0] = -(buf0[0] > buf1[0]);
                      buf0[1] = -(buf0[1] > buf1[1]);

                      return _mm_loadu_si128((__m128i*)buf0);'''.format(**fmtspec)
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
# lesser than

def lt2(simd_ext, typ):
    return 'return nsimd_gt_{simd_ext}_{typ}({in1}, {in0});'. \
           format(**fmtspec)

# -----------------------------------------------------------------------------
# greater or equal

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
# lesser or equal

def leq2(simd_ext, typ):
    if typ == 'f16':
        return f16_cmp2('le', simd_ext)
    notgt = '''return nsimd_notl_{simd_ext}_{typ}(
                        nsimd_gt_{simd_ext}_{typ}({in0}, {in1}));'''. \
                        format(**fmtspec)
    if simd_ext in sse and typ in ['f32', 'f64']:
        return 'return _mm_cmple{suf}({in0}, {in1});'.format(**fmtspec)
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
# if_else1 function

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
# min and max functions

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
# sqrt

def sqrt1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = {pre}sqrt_ps({in0}.v0);
                  ret.v1 = {pre}sqrt_ps({in0}.v1);
                  return ret;'''.format(**fmtspec)
    return 'return {pre}sqrt{suf}({in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Load logical

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
# Store logical

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
                        ((u16*){in0})[i] = (u16)((({in1}.v0 >> i) & 1) ? one
                                                                       : 0);
                      }}
                      for (i = 0; i < 16; i++) {{
                        ((u16*){in0})[i + 16] = (u16)((({in1}.v1 >> i) & 1)
                                                      ? one : 0);
                      }}'''.format(**fmtspec)
        return '''/* This can surely be improved but it is not our priority. */
                  int i;
                  for (i = 0; i < {le}; i++) {{
                    {in0}[i] = ({typ})((({in1} >> i) & 1) ? 1 : 0);
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
# Absolute value

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
    '''return _mm256_castpd_si256(_mm256_blendv_pd(
        _mm256_castsi256_pd({in0}),
        _mm256_castsi256_pd(_mm256_sub_epi64(_mm256_setzero_si256(), {in0})),
        _mm256_castsi256_pd({in0})));'''.format(**fmtspec)
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
# FMA and FMS

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
                              nsimd_mul_{simd_ext}_{typ}(
                                nsimd_neg_{simd_ext}_{typ}({in0}), {in1}),
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
                      # else
                        {emulate}
                      # endif'''.format(op=op, neg=neg, emulate=emulate,
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
# Ceil and floor

def round1(opts, func, simd_ext, typ):
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
            return emulate_op1(opts, func, simd_ext, typ)
    return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Trunc

def trunc1(opts, simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_trunc_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_trunc_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(**fmtspec)
    if typ in ['f32', 'f64']:
        normal = '''return {pre}round{suf}({in0}, _MM_FROUND_TO_ZERO |
                               _MM_FROUND_NO_EXC);'''.format(**fmtspec)
        if simd_ext == 'sse2':
            return emulate_op1(opts, 'trunc', simd_ext, typ)
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

def round_to_even1(opts, simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_round_to_even_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_round_to_even_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(**fmtspec)
    if typ in ['f32', 'f64']:
        normal = '''return {pre}round{suf}({in0}, _MM_FROUND_TO_NEAREST_INT |
                               _MM_FROUND_NO_EXC);'''.format(**fmtspec)
        if simd_ext == 'sse2':
            return emulate_op1(opts, 'round_to_even', simd_ext, typ)
        if simd_ext == 'sse42':
            return normal
        if simd_ext in avx:
            return normal
        if simd_ext in avx512:
            return 'return _mm512_roundscale{suf}({in0}, 0);'.format(**fmtspec)
    return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------
# All and any functions

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
            if simd_ext == 'avx2':
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
                 '{pre}cvtps_ph({in0}.v0, 4)'.format(**fmtspec),
                 '{pre}cvtps_ph({in0}.v1, 4)'.format(**fmtspec)))
        if simd_ext in sse:
            return \
            '''#ifdef NSIMD_FP16
                 __m128i lo = _mm_cvtps_ph({in0}.v0, 4);
                 __m128i hi = _mm_cvtps_ph({in0}.v1, 4);
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
# Reciprocal (at least 11 bits of precision)

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
            one = '{pre}set1_pd(1.0)'.format(**fmtspec)
            if func == 'rcp':
                return 'return {pre}div{suf}({one}, {in0});'.format(one=one, **fmtspec)
            else:
                return 'return {pre}div{suf}({one}, {pre}sqrt{suf}({in0}));'. \
                        format(one=one, **fmtspec)
            format(func=func, **fmtspec)
        if simd_ext in avx512:
            return 'return _mm512_{func}14_pd({in0});'. \
                   format(func=func, **fmtspec)

# -----------------------------------------------------------------------------
# Reciprocal (IEEE)

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
# Negative

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
# nbtrue

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
# reverse

def reverse1(simd_ext, typ):
    # 8-bit int
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
        # AVX-512F and above.
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
    # 16-bit int
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
        # AVX-512F
        elif simd_ext == 'avx512_knl':
            return \
            '''{in0} = _mm512_permutexvar_epi32(_mm512_set_epi32(
                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                 {in0});
               nsimd_{simd_ext}_v{typ} r0 = _mm512_srli_epi32({in0}, 16);
               nsimd_{simd_ext}_v{typ} r1 = _mm512_slli_epi32({in0}, 16);
               return _mm512_or_si512(r0, r1);'''.format(**fmtspec)
        # AVX-512F+BW (Skylake) + WORKAROUND GCC<=8
        else:
            return \
            '''return _mm512_permutexvar_epi16(_mm512_set_epi32(
                 (0<<16)  | 1,  (2<<16)  | 3,  (4<<16)  | 5,  (6<<16)  | 7,
                 (8<<16)  | 9,  (10<<16) | 11, (12<<16) | 13, (14<<16) | 15,
                 (16<<16) | 17, (18<<16) | 19, (20<<16) | 21, (22<<16) | 23,
                 (24<<16) | 25, (26<<16) | 27, (28<<16) | 29, (30<<16) | 31),
                 {in0} );'''.format(**fmtspec)
    # 32-bit int
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
    # 16-bit float
    elif typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_reverse_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_reverse_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(**fmtspec)
    # 32-bit float
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
    # 64-bit float
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
# addv

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
# upconvert

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
            upper_half = '_mm_shuffle_epi32({in0}, 14 /* 2 | 3 */)'. \
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
    # if simd_ext == 'sse2':
    if simd_ext in sse:
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
    # elif simd_ext == 'sse42':
    #    return with_intrinsic
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
# downconvert

def downcvt1(opts, simd_ext, from_typ, to_typ):
    # From f16 is easy
    if from_typ == 'f16':
        le_to_typ = int(fmtspec['le']) * 2
        le_1f32 = le_to_typ // 4
        le_2f32 = 2 * le_to_typ // 4
        le_3f32 = 3 * le_to_typ // 4
        cast = castsi(simd_ext, to_typ)
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
           return {pre}loadu_si{nbits}({cast}dst);'''. \
           format(le_to_typ=le_to_typ, le_1f32=le_1f32, le_2f32=le_2f32,
                  le_3f32=le_3f32, cast=cast, **fmtspec)

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
    cast_src = '(__m{nbits}i *)'.format(**fmtspec) \
               if from_typ in common.iutypes else ''
    cast_dst = '(__m{nbits}i *)'.format(**fmtspec) \
               if to_typ in common.iutypes else ''
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
       format(cast_src=cast_src, cast_dst=cast_dst, le_to_typ=le_to_typ,
              sufsi_to_typ=suf_si(simd_ext, to_typ), **fmtspec)

# -----------------------------------------------------------------------------
# adds / subs helper

def adds_subs_intrinsic_instructions_i8_i16_u8_u16(which_op, simd_ext, typ):

    valid_types = ('i8', 'i16', 'u8', 'u16')
    if typ not in valid_types:
        raise TypeError(
    '''def adds_subs_intrinsic_instructions_i8_i16_u8_u16(...):
     {typ} must belong to the following types set: {valid_types}'''.\
        format(typ=typ, valid_types=valid_types)
    )
    if 'sse2' in simd_ext or 'sse42' in simd_ext:
        return'''
        return _mm_{which_op}_ep{typ}({in0}, {in1});
        '''.format(which_op=which_op, **fmtspec)
    if 'avx' == simd_ext:
        return split_opn(which_op, simd_ext, typ, 2)
    if simd_ext in ('avx2', 'avx512_skylake'):
        return 'return {pre}{which_op}_ep{typ}({in0}, {in1});'. \
            format(which_op=which_op, **fmtspec)
    if 'avx512_knl' == simd_ext:
        return split_opn(which_op, simd_ext, typ, 2)

def get_avx512_sse2_i32_i64_dependent_code(simd_ext, typ):
    if 'avx512' in simd_ext or 'sse2' in simd_ext:
        mask_processing = \
        '''/* For avx512/sse2 */
           const nsimd_{simd_ext}_vu{typnbits} mask_strong_bit =
               nsimd_shr_{simd_ext}_u{typnbits}(
                   mask, sizeof(u{typnbits}) * CHAR_BIT - 1);
           const nsimd_{simd_ext}_vi{typnbits} imask_strong_bit =
               nsimd_reinterpret_{simd_ext}_i{typnbits}_u{typnbits}(
                   mask_strong_bit);
           const nsimd_{simd_ext}_vli{typnbits} limask_strong_bit =
               nsimd_to_logical_{simd_ext}_i{typnbits}(imask_strong_bit);'''. \
               format(**fmtspec)
        if_else = \
        '''/* For avx512/sse2 */
           return nsimd_if_else1_{simd_ext}_i{typnbits}(
                      limask_strong_bit, ires, i_max_min);'''. \
                      format(**fmtspec)
    else:
        mask_processing = '/* Before avx512: is_same(__m128i, ' \
                          'vector<signed>, vector<unsigned>, ' \
                          'vector<logical>) */'
        suf2 = 'ps' if typ in ['i32', 'u32'] else 'pd'
        if_else = '''return {pre}cast{suf2}_si{nbits}({pre}blendv_{suf2}(
                                {pre}castsi{nbits}_{suf2}(i_max_min),
                                {pre}castsi{nbits}_{suf2}(ires),
                                {pre}castsi{nbits}_{suf2}(mask)));
                                '''.format(suf2=suf2, **fmtspec)

    return { 'mask_processing': mask_processing, 'if_else': if_else }

# -----------------------------------------------------------------------------
# adds

def adds(simd_ext, typ):

    if typ in common.ftypes:
        return 'return nsimd_add_{simd_ext}_{typ}({in0}, {in1});'. \
               format(**fmtspec)

    if typ in ('i8', 'i16', 'u8', 'u16'):
        return adds_subs_intrinsic_instructions_i8_i16_u8_u16(
                   'adds', simd_ext, typ)

    if typ in common.utypes:
        return \
        '''/* Algo pseudo code: */
           /* ures = a + b */
           /* if overflow then ures < a && ures < b */
           /* --> test against a single value: if(ures < a){{ overflow ; }} */
           /* return ures < a ? {type_max} : ures */

           const nsimd_{simd_ext}_v{typ} ures =
               nsimd_add_{simd_ext}_{typ}({in0}, {in1});
           const nsimd_{simd_ext}_v{typ} type_max =
               nsimd_set1_{simd_ext}_{typ}(({typ}){type_max});
           return nsimd_if_else1_{simd_ext}_{typ}(
                    nsimd_lt_{simd_ext}_{typ}(ures, {in0}),
                    type_max, ures);'''. \
                    format(type_max=common.limits[typ]['max'], **fmtspec)

    avx512_sse2_i32_i64_dependent_code = \
        get_avx512_sse2_i32_i64_dependent_code(simd_ext, typ)

    return \
    '''/* Algo pseudo code: */

       /* if ( ( same_sign(ux, uy) && same_sign(uy, res) ) || */
       /*      ! same_sign(ux, uy) ): */
       /*     neither overflow nor underflow happened */
       /* else: */
       /*     if(ux > 0 && uy > 0): res = MAX // overflow */
       /*     else: res = MIN // underflow */

       /* Step 1: reinterpret to unsigned to work with the bits */

       nsimd_{simd_ext}_vu{typnbits} ux =
           nsimd_reinterpret_{simd_ext}_u{typnbits}_i{typnbits}({in0});
       const nsimd_{simd_ext}_vu{typnbits} uy =
           nsimd_reinterpret_{simd_ext}_u{typnbits}_i{typnbits}({in1});
       const nsimd_{simd_ext}_vu{typnbits} ures =
           nsimd_add_{simd_ext}_u{typnbits}(ux, uy);

       /* Step 2: check signs different: ux, uy, res */

       /* xor_ux_uy's most significant bit will be zero if both ux and */
       /* uy have same sign */

       const nsimd_{simd_ext}_vu{typnbits} xor_ux_uy =
           nsimd_xorb_{simd_ext}_u{typnbits}(ux, uy);

       /* xor_uy_res's most significant bit will be zero if both uy and */
       /* ures have same sign */

       const nsimd_{simd_ext}_vu{typnbits} xor_uy_res =
           nsimd_xorb_{simd_ext}_u{typnbits}(uy, ures);

       /* Step 3: Construct the MIN/MAX vector */

       /* Pseudo code: */

       /* Both positive --> overflow possible */
       /* --> get the MAX: */

       /* (signed)ux >= 0 && (signed)uy >= 0 */
       /* <=> ((unsigned)ux | (unsigned)uy) >> 31 == 0 */
       /* --> MAX + ( (ux | uy) >> 31 ) == MAX + 0 == MAX */

       /* At least one negative */
       /* --> overflow not possible / underflow possible if both negative */
       /* --> get the MIN: */

       /* unsigned tmp = (unsigned)MAX + */
       /*                ( ( (ux | uy) >> 31 ) == (unsigned)MAX + 1 ) */
       /* --> MIN = (reinterpret signed)tmp */

       /* ux | uy */
       const nsimd_{simd_ext}_vu{typnbits} ux_uy_orb =
           nsimd_orb_{simd_ext}_u{typnbits}(ux, uy);

       /* (ux | uy) >> 31 --> Vector of 0's and 1's */
       const nsimd_{simd_ext}_vu{typnbits} u_zeros_ones =
           nsimd_shr_{simd_ext}_u{typnbits}(
               ux_uy_orb, sizeof(u{typnbits}) * CHAR_BIT - 1);

       /* MIN/MAX vector */

       /* i{typnbits} tmp = sMAX + 1 --> undefined behavior */
       /* u{typnbits} tmp = (u{typnbits})sMAX + 1 */
       /* i{typnbits} sMIN = *(i{typnbits}*)(&tmp) */

       const nsimd_{simd_ext}_vu{typnbits} u_max =
           nsimd_set1_{simd_ext}_u{typnbits}((u{typnbits}){type_max});
       const nsimd_{simd_ext}_vu{typnbits} u_max_min =
           nsimd_add_{simd_ext}_u{typnbits}(u_max, u_zeros_ones);
       const nsimd_{simd_ext}_vi{typnbits} i_max_min =
           nsimd_reinterpret_{simd_ext}_i{typnbits}_u{typnbits}(u_max_min);

       /* Step 4: Construct the mask vector */

       /* mask == ( 8ot_same_sign(ux, uy) || same_sign(uy, res) ) */
       /* mask: True (no underflow/overflow) / False (underflow/overflow) */
       /* mask = xor_ux_uy | ~ xor_uy_res */

       const nsimd_{simd_ext}_vu{typnbits} not_xor_uy_res =
           nsimd_notb_{simd_ext}_u{typnbits}(xor_uy_res);
       const nsimd_{simd_ext}_vu{typnbits} mask =
           nsimd_orb_{simd_ext}_u{typnbits}(xor_ux_uy, not_xor_uy_res);

       {avx512_sse2_dependent_mask_processing}

       /* Step 5: Apply the Mask */

       const nsimd_{simd_ext}_vi{typnbits} ires =
           nsimd_reinterpret_{simd_ext}_i{typnbits}_u{typnbits}(ures);

       {avx512_sse2_dependent_if_else}'''. \
       format(type_max = common.limits[typ]['max'],
              avx512_sse2_dependent_mask_processing = \
                  avx512_sse2_i32_i64_dependent_code['mask_processing'],
              avx512_sse2_dependent_if_else = \
                  avx512_sse2_i32_i64_dependent_code['if_else'], **fmtspec)

# -----------------------------------------------------------------------------
# subs

def subs(simd_ext, typ):

    if typ in common.ftypes:
        return 'return nsimd_sub_{simd_ext}_{typ}({in0}, {in1});'. \
               format(**fmtspec)

    if typ in ('i8', 'i16', 'u8', 'u16'):
        return adds_subs_intrinsic_instructions_i8_i16_u8_u16(
                   'subs', simd_ext, typ)

    if typ in common.itypes:
        return 'return nsimd_adds_{simd_ext}_{typ}({in0}, ' \
               'nsimd_neg_{simd_ext}_{typ}({in1}));'.format(**fmtspec)

    min_ = common.limits[typ]['min']

    return \
    '''/* Algo pseudo code: */

       /* unsigned only */
       /* a > 0; b > 0 ==> a - b --> possibility for underflow only */
       /* if b > a --> underflow */

       const nsimd_{simd_ext}_v{typ} ures =
           nsimd_sub_{simd_ext}_{typ}({in0}, {in1});
       const nsimd_{simd_ext}_vl{typ} is_underflow =
           nsimd_gt_{simd_ext}_{typ}({in1}, {in0});
       const nsimd_{simd_ext}_v{typ} umin =
           nsimd_set1_{simd_ext}_{typ}(({typ}){min_});
       return nsimd_if_else1_{simd_ext}_{typ}(is_underflow, umin, ures);'''. \
       format(min_=min_, **fmtspec)

# -----------------------------------------------------------------------------
# to_mask

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
# to_logical

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
# zip functions

def zip_half(func, simd_ext, typ):
    simd_ext2 = 'sse42' if simd_ext in avx else 'avx2'
    if simd_ext in sse:
        if typ == 'f16':
            return '''nsimd_{simd_ext}_v{typ} ret;
                      ret.v0 = _mm_unpacklo_ps({in0}.v{k}, {in1}.v{k});
                      ret.v1 = _mm_unpackhi_ps({in0}.v{k}, {in1}.v{k});
                      return ret;'''. \
                      format(k='0' if func == 'ziplo' else '1', **fmtspec)
        else:
            return 'return {pre}unpack{lo}{suf}({in0}, {in1});'. \
                   format(lo='lo' if func == 'ziplo' else 'hi', **fmtspec)
    elif simd_ext in avx:
        # Currently, 256 and 512 bits vectors are splitted into 128 bits
        # vectors in order to perform the ziplo/hi operation using the
        # unpacklo/hi sse operations.
        if typ == 'f16':
            in0vk = '{in0}.v{k}'.format(k='0' if func == 'ziplo' else '1',
                                        **fmtspec)
            in1vk = '{in1}.v{k}'.format(k='0' if func == 'ziplo' else '1',
                                        **fmtspec)
            return \
            '''nsimd_{simd_ext}_v{typ} ret;
               __m128 v_tmp0 = {get_low_in0vk};
               __m128 v_tmp1 = {get_low_in1vk};
               __m128 v_tmp2 = {get_high_in0vk};
               __m128 v_tmp3 = {get_high_in1vk};
               __m128 vres_lo0 = _mm_unpacklo_ps(v_tmp0, v_tmp1);
               __m128 vres_hi0 = _mm_unpackhi_ps(v_tmp0, v_tmp1);
               ret.v0 = {merge0};
               __m128 vres_lo1 = _mm_unpacklo_ps(v_tmp2, v_tmp3);
               __m128 vres_hi1 = _mm_unpackhi_ps(v_tmp2, v_tmp3);
               ret.v1 = {merge1};
               return ret;
               '''.format(get_low_in0vk=extract(simd_ext, 'f32', LO, in0vk),
                          get_low_in1vk=extract(simd_ext, 'f32', LO, in1vk),
                          get_high_in0vk=extract(simd_ext, 'f32', HI, in0vk),
                          get_high_in1vk=extract(simd_ext, 'f32', HI, in1vk),
                          merge0=setr(simd_ext, 'f32', 'vres_lo0', 'vres_hi0'),
                          merge1=setr(simd_ext, 'f32', 'vres_lo1', 'vres_hi1'),
                          **fmtspec)
        else:
            hl = LO if func == 'ziplo' else HI
            return \
            '''{nat} v_tmp0 = {half_in0};
               {nat} v_tmp1 = {half_in1};
               {nat} vres_lo = _mm_unpacklo{suf}(v_tmp0, v_tmp1);
               {nat} vres_hi = _mm_unpackhi{suf}(v_tmp0, v_tmp1);
               return {merge};
               '''.format(nat=get_native_typ(simd_ext2, typ),
                          half_in0=extract(simd_ext, typ, hl, common.in0),
                          half_in1=extract(simd_ext, typ, hl, common.in1),
                          merge=setr(simd_ext, typ, 'vres_lo', 'vres_hi'),
                          **fmtspec)
    else:
        if typ == 'f16':
            return \
            '''nsimd_{simd_ext}_v{typ} ret;
               __m512 v0 = {in0}.v{k};
               __m512 v1 = {in1}.v{k};
               __m256 v_tmp0, v_tmp1, vres_lo, vres_hi;
               /* Low part */
               v_tmp0 = {low_v0};
               v_tmp1 = {low_v1};
               vres_lo = nsimd_ziplo_avx2_f32(v_tmp0, v_tmp1);
               vres_hi = nsimd_ziphi_avx2_f32(v_tmp0, v_tmp1);
               ret.v0 = {merge};
               /* High part */
               v_tmp0 = {high_v0};
               v_tmp1 = {high_v1};
               vres_lo = nsimd_ziplo_avx2_f32(v_tmp0, v_tmp1);
               vres_hi = nsimd_ziphi_avx2_f32(v_tmp0, v_tmp1);
               ret.v1 = {merge};
               return ret;'''. \
               format(k='0' if func == 'ziplo' else '1',
                      low_v0=extract(simd_ext, 'f32', LO, 'v0'),
                      low_v1=extract(simd_ext, 'f32', LO, 'v1'),
                      high_v0=extract(simd_ext, 'f32', HI, 'v0'),
                      high_v1=extract(simd_ext, 'f32', HI, 'v1'),
                      merge=setr(simd_ext, 'f32', 'vres_lo', 'vres_hi'),
                      **fmtspec)
        else:
            hl = LO if func == 'ziplo' else HI
            return \
            '''{nat} v_tmp0, v_tmp1;
               v_tmp0 = {half_in0};
               v_tmp1 = {half_in1};
               {nat} vres_lo = nsimd_ziplo_avx2_{typ}(v_tmp0, v_tmp1);
               {nat} vres_hi = nsimd_ziphi_avx2_{typ}(v_tmp0, v_tmp1);
               return {merge};'''. \
               format(nat=get_native_typ(simd_ext2, typ),
                      half_in0=extract(simd_ext, typ, hl, common.in0),
                      half_in1=extract(simd_ext, typ, hl, common.in1),
                      merge=setr(simd_ext, typ, 'vres_lo', 'vres_hi'),
                      **fmtspec)

def zip(simd_ext, typ):
    return '''nsimd_{simd_ext}_v{typ}x2 ret;
              ret.v0 = nsimd_ziplo_{simd_ext}_{typ}({in0}, {in1});
              ret.v1 = nsimd_ziphi_{simd_ext}_{typ}({in0}, {in1});
              return ret;
              '''.format(**fmtspec)

# -----------------------------------------------------------------------------
# unzip functions

def unzip_half(opts, func, simd_ext, typ):
    loop = '''{typ} tab[{lex2}];
              {typ} res[{le}];
              int i;
              nsimd_storeu_{simd_ext}_{typ}(tab, {in0});
              nsimd_storeu_{simd_ext}_{typ}(tab + {le}, {in1});
              for(i = 0; i < {le}; i++) {{
                res[i] = tab[2 * i + {offset}];
              }}
              return nsimd_loadu_{simd_ext}_{typ}(res);
              '''.format(lex2=2 * int(fmtspec['le']),
                         offset='0' if func == 'unziplo' else '1', **fmtspec)

    if simd_ext in sse:
        if typ in ['f32', 'i32', 'u32']:
            v0 = ('_mm_castsi128_ps({in0})' if typ in ['i32', 'u32'] \
                                            else '{in0}').format(**fmtspec)
            v1 = ('_mm_castsi128_ps({in1})' if typ in ['i32', 'u32'] \
                                            else '{in1}').format(**fmtspec)
            ret = ('_mm_castps_si128(v_res)' if typ in ['i32', 'u32'] \
                                             else 'v_res').format(**fmtspec)
            return '''__m128 v_res;
                      v_res = _mm_shuffle_ps({v0}, {v1}, {mask});
                      return {ret};'''.format(
                      mask='_MM_SHUFFLE(2, 0, 2, 0)' if func == 'unziplo' \
                      else '_MM_SHUFFLE(3, 1, 3, 1)',
                      v0=v0, v1=v1, ret=ret, **fmtspec)
        elif typ == 'f16':
            return \
            '''nsimd_{simd_ext}_v{typ} v_res;
               v_res.v0 = _mm_shuffle_ps({in0}.v0, {in0}.v1, {mask});
               v_res.v1 = _mm_shuffle_ps({in1}.v0, {in1}.v1, {mask});
               return v_res;'''.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' \
                                       if func == 'unziplo' \
                                       else '_MM_SHUFFLE(3, 1, 3, 1)',
                                       **fmtspec)
        elif typ in ['f64', 'i64', 'u64']:
            v0 = ('_mm_castsi128_pd({in0})' if typ in ['i64', 'u64'] \
                                            else '{in0}').format(**fmtspec)
            v1 = ('_mm_castsi128_pd({in1})' if typ in ['i64', 'u64'] \
                                            else '{in1}').format(**fmtspec)
            ret = ('_mm_castpd_si128(v_res)' if typ in ['i64', 'u64'] \
                                             else 'v_res').format(**fmtspec)
            return '''__m128d v_res;
                      v_res = _mm_shuffle_pd({v0}, {v1}, {mask});
                      return {ret};
                      '''.format(mask='0' if func == 'unziplo' else '3',
                                 v0=v0, v1=v1, ret=ret, **fmtspec)
        elif typ in ['i16', 'u16']:
            return '''__m128i v_tmp0 = _mm_shufflelo_epi16(
                                           {in0}, _MM_SHUFFLE(3, 1, 2, 0));
                      v_tmp0 = _mm_shufflehi_epi16(v_tmp0,
                                   _MM_SHUFFLE(3, 1, 2, 0));
                      __m128i v_tmp1 = _mm_shufflelo_epi16({in1},
                                   _MM_SHUFFLE(3, 1, 2, 0));
                      v_tmp1 = _mm_shufflehi_epi16(v_tmp1,
                                   _MM_SHUFFLE(3, 1, 2, 0));
                      __m128 v_res = _mm_shuffle_ps(_mm_castsi128_ps(v_tmp0),
                                         _mm_castsi128_ps(v_tmp1), {mask});
                      return _mm_castps_si128(v_res);
                      '''.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' \
                                 if func == 'unziplo' \
                                 else '_MM_SHUFFLE(3, 1, 3, 1)', **fmtspec)
        else:
            return loop
    elif simd_ext in avx:
        ret_template = \
        '''v_tmp0 = _mm256_permute2f128_{t}({v0}, {v0}, 0x01);
           v_tmp0 = _mm256_shuffle_{t}({v0}, v_tmp0, {mask});
           v_tmp1 = _mm256_permute2f128_{t}({v1}, {v1}, 0x01);
           v_tmp1 = _mm256_shuffle_{t}({v1}, v_tmp1, {mask});
           v_res  = _mm256_permute2f128_{t}(v_tmp0, v_tmp1, 0x20);
           {ret} = {v_res};'''
        if typ in ['f32', 'i32', 'u32']:
            v0 = '_mm256_castsi256_ps({in0})' \
                 if typ in ['i32', 'u32'] else '{in0}'
            v1 = '_mm256_castsi256_ps({in1})' \
                 if typ in ['i32', 'u32'] else '{in1}'
            v_res = '_mm256_castps_si256(v_res)' \
                    if typ in ['i32', 'u32'] else 'v_res'
            ret = 'ret'
            src = ret_template.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' \
                      if func == 'unziplo' else '_MM_SHUFFLE(3, 1, 3, 1)',
                      v0=v0, v1=v1, v_res=v_res, ret=ret, t='ps', **fmtspec)
            return '''nsimd_{simd_ext}_v{typ} ret;
                      __m256 v_res, v_tmp0, v_tmp1;
                      {src}
                      return ret;'''. \
                      format(src=src.format(**fmtspec), **fmtspec)
        elif typ == 'f16':
            src0 = ret_template.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' \
                       if func == 'unziplo' else '_MM_SHUFFLE(3, 1, 3, 1)',
                       v0='{in0}.v0', v1='{in0}.v1', v_res='v_res',
                       ret='ret.v0', t='ps')
            src1 = ret_template.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' \
                       if func == 'unziplo' else '_MM_SHUFFLE(3, 1, 3, 1)',
                       v0='{in1}.v0', v1='{in1}.v1', v_res='v_res',
                       ret='ret.v1', t='ps')
            return '''nsimd_{simd_ext}_v{typ} ret;
                      __m256 v_res, v_tmp0, v_tmp1;
                      {src0}
                      {src1}
                      return ret;'''.format(src0=src0.format(**fmtspec),
                                            src1=src1.format(**fmtspec),
                                            **fmtspec)
        elif typ in ['f64', 'i64', 'u64']:
            v0 = ('_mm256_castsi256_pd({in0})' \
                      if typ in ['i64', 'u64'] else '{in0}').format(**fmtspec)
            v1 = ('_mm256_castsi256_pd({in1})' \
                      if typ in ['i64', 'u64'] else '{in1}').format(**fmtspec)
            v_res = ('_mm256_castpd_si256(v_res)' \
                         if typ in ['i64', 'u64'] else 'v_res'). \
                         format(**fmtspec)
            src = ret_template.format(mask='0x00' if func == 'unziplo' \
                      else '0x03', v0=v0, v1=v1, ret='ret', v_res=v_res,
                      t='pd')
            return '''nsimd_{simd_ext}_v{typ} ret;
                      __m256d v_res, v_tmp0, v_tmp1;
                      {src}
                      return ret;'''.format(src=src.format(**fmtspec),
                                            **fmtspec)
        elif typ in ['i16', 'u16']:
            return \
            '''__m128i v_tmp0_hi = {hi0};
               __m128i v_tmp0_lo = {lo0};
               __m128i v_tmp1_hi = {hi1};
               __m128i v_tmp1_lo = {lo1};
               v_tmp0_lo = nsimd_{func}_sse2_{typ}(v_tmp0_lo, v_tmp0_hi);
               v_tmp1_lo = nsimd_{func}_sse2_{typ}(v_tmp1_lo, v_tmp1_hi);
               return {merge};'''. \
               format(hi0=extract(simd_ext, typ, HI, common.in0),
                      lo0=extract(simd_ext, typ, LO, common.in0),
                      hi1=extract(simd_ext, typ, HI, common.in1),
                      lo1=extract(simd_ext, typ, LO, common.in1),
                      merge=setr(simd_ext, typ, 'v_tmp0_lo', 'v_tmp1_lo'),
                      func=func, **fmtspec)
        else:
            return loop
    else:
        if typ == 'f16':
            return \
            '''nsimd_{simd_ext}_v{typ} ret;
               __m256 v_tmp0, v_tmp1, v_res_lo, v_res_hi;
               v_tmp0 = {loin0v0};
               v_tmp1 = {hiin0v0};
               v_res_lo = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
               v_tmp0 = {loin0v1};
               v_tmp1 = {hiin0v1};
               v_res_hi = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
               ret.v0 = {merge};
               v_tmp0 = {loin1v0};
               v_tmp1 = {hiin1v0};
               v_res_lo = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
               v_tmp0 = {loin1v1};
               v_tmp1 = {hiin1v1};
               v_res_hi = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
               ret.v1 = {merge};
               return ret;'''.format(
                   loin0v0=extract(simd_ext, 'f32', LO, common.in0 + '.v0'),
                   hiin0v0=extract(simd_ext, 'f32', HI, common.in0 + '.v0'),
                   loin0v1=extract(simd_ext, 'f32', LO, common.in0 + '.v1'),
                   hiin0v1=extract(simd_ext, 'f32', HI, common.in0 + '.v1'),
                   loin1v0=extract(simd_ext, 'f32', LO, common.in1 + '.v0'),
                   hiin1v0=extract(simd_ext, 'f32', HI, common.in1 + '.v0'),
                   loin1v1=extract(simd_ext, 'f32', LO, common.in1 + '.v1'),
                   hiin1v1=extract(simd_ext, 'f32', HI, common.in1 + '.v1'),
                   merge=setr(simd_ext, 'f32', 'v_res_lo', 'v_res_hi'),
                   func=func, **fmtspec)
        else:
            return '''nsimd_avx2_v{typ} v00 = {extract_lo0};
                      nsimd_avx2_v{typ} v01 = {extract_hi0};
                      nsimd_avx2_v{typ} v10 = {extract_lo1};
                      nsimd_avx2_v{typ} v11 = {extract_hi1};
                      v00 = nsimd_{func}_avx2_{typ}(v00, v01);
                      v01 = nsimd_{func}_avx2_{typ}(v10, v11);
                      return {merge};'''.format(
                          func=func,
                          extract_lo0=extract(simd_ext, typ, LO, common.in0),
                          extract_lo1=extract(simd_ext, typ, LO, common.in1),
                          extract_hi0=extract(simd_ext, typ, HI, common.in0),
                          extract_hi1=extract(simd_ext, typ, HI, common.in1),
                          merge=setr(simd_ext, typ, 'v00', 'v01'), **fmtspec)

def unzip(simd_ext, typ):
    return '''nsimd_{simd_ext}_v{typ}x2 ret;
              ret.v0 = nsimd_unziplo_{simd_ext}_{typ}({in0}, {in1});
              ret.v1 = nsimd_unziphi_{simd_ext}_{typ}({in0}, {in1});
              return ret;'''.format(**fmtspec)

# -----------------------------------------------------------------------------
# mask_for_loop_tail

def mask_for_loop_tail(simd_ext, typ):
    if typ == 'f16':
        fill_n = '''n.v0 = {pre}set1_ps((f32)({in1} - {in0}));
                    n.v1 = n.v0;'''.format(**fmtspec)
    else:
        fill_n = 'n = nsimd_set1_{simd_ext}_{typ}(({typ})({in1} - {in0}));'. \
                 format(**fmtspec)
    return '''if ({in0} >= {in1}) {{
                return nsimd_set1l_{simd_ext}_{typ}(0);
              }}
              if ({in1} - {in0} < {le}) {{
                nsimd_{simd_ext}_v{typ} n;
                {fill_n}
                return nsimd_lt_{simd_ext}_{typ}(
                         nsimd_iota_{simd_ext}_{typ}(), n);
              }} else {{
                return nsimd_set1l_{simd_ext}_{typ}(1);
              }}'''.format(fill_n=fill_n, **fmtspec)

# -----------------------------------------------------------------------------
# iota

def iota(simd_ext, typ):
    typ2 = 'f32' if typ == 'f16' else typ
    iota = ', '.join(['({typ2}){i}'.format(typ2=typ2, i=i) \
                      for i in range(int(fmtspec['le']))])
    if typ == 'f16':
        return '''f32 buf[{le}] = {{ {iota} }};
                  nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = {pre}loadu_ps(buf);
                  ret.v1 = {pre}loadu_ps(buf + {le2});
                  return ret;'''. \
                  format(iota=iota, le2=fmtspec['le'] // 2, **fmtspec)
    return '''{typ} buf[{le}] = {{ {iota} }};
              return {pre}loadu{sufsi}({cast}buf);'''. \
              format(iota=iota, cast='(__m{nbits}i*)'.format(**fmtspec) \
                                if typ in common.iutypes else '', **fmtspec)

# -----------------------------------------------------------------------------
# scatter

def scatter(simd_ext, typ):
    if typ == 'f16':
        return '''int i;
                  f32 buf[{le}];
                  i16 offset_buf[{le}];
                  {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in1});
                  {pre}storeu_ps(buf, {in2}.v0);
                  {pre}storeu_ps(buf + {leo2}, {in2}.v1);
                  for (i = 0; i < {le}; i++) {{
                    {in0}[offset_buf[i]] = nsimd_f32_to_f16(buf[i]);
                  }}'''.format(leo2=int(fmtspec['le']) // 2, **fmtspec)
    if simd_ext in (sse + avx) or typ in ['i8', 'u8', 'i16', 'u16']:
        cast = castsi(simd_ext, typ)
        return '''int i;
                  {typ} buf[{le}];
                  {ityp} offset_buf[{le}];
                  {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in1});
                  {pre}storeu{sufsi}({cast}buf, {in2});
                  for (i = 0; i < {le}; i++) {{
                    {in0}[offset_buf[i]] = buf[i];
                  }}'''.format(ityp='i' + typ[1:], cast=cast, **fmtspec)
    # getting here means 32 and 64-bits types for avx512
    return '''{pre}i{typnbits}scatter{suf}(
                  (void *){in0}, {in1}, {in2}, {scale});'''. \
                  format(scale=int(typ[1:]) // 8, **fmtspec)

# -----------------------------------------------------------------------------
# linear scatter

def scatter_linear(simd_ext, typ):
    if typ == 'f16':
        return '''int i;
                  f32 buf[{le}];
                  {pre}storeu_ps(buf, {in2}.v0);
                  {pre}storeu_ps(buf + {leo2}, {in2}.v1);
                  for (i = 0; i < {le}; i++) {{
                    {in0}[i * {in1}] = nsimd_f32_to_f16(buf[i]);
                  }}'''.format(leo2=int(fmtspec['le']) // 2, **fmtspec)
    if simd_ext in avx512:
        return '''nsimd_scatter_linear_avx2_{typ}({in0}, {in1}, {lo});
                  nsimd_scatter_linear_avx2_{typ}({in0} + ({leo2} * {in1}),
                                                  {in1}, {hi});'''. \
                  format(leo2=int(fmtspec['le']) // 2,
                         lo=extract(simd_ext, typ, LO, fmtspec['in2']),
                         hi=extract(simd_ext, typ, HI, fmtspec['in2']),
                         **fmtspec)
    emulation = '''int i;
                   {typ} buf[{le}];
                   {pre}storeu{sufsi}({cast}buf, {in2});
                   for (i = 0; i < {le}; i++) {{
                     {in0}[i * {in1}] = buf[i];
                   }}'''.format(cast=castsi(simd_ext, typ), **fmtspec)
    if (simd_ext == 'sse2' and typ in ['i16', 'u16']) or \
       (simd_ext == 'avx' and \
        typ in ['i32', 'u32', 'f32', 'i64', 'u64', 'f64']) or \
       (simd_ext in ['sse42', 'avx2']):
        trick = '\n'.join([
        '{in0}[{i} * {in1}] = {get_lane};'.format(i=i,
        get_lane=get_lane(simd_ext, typ, '{in2}'.format(**fmtspec), i),
        **fmtspec) for i in range(int(fmtspec['le']))])
        return '''#if NSIMD_WORD_SIZE == 32
                    {}
                  #else
                    {}
                  #endif'''.format(emulation, trick)
    else:
        return emulation

# -----------------------------------------------------------------------------
# mask_scatter

def mask_scatter(simd_ext, typ):
    if typ == 'f16':
        le2 = fmtspec['le'] // 2
        if simd_ext in sse + avx:
            store_mask = '''{pre}storeu_ps(mask, {in0}.v0);
                            {pre}storeu_ps(mask + {le2}, {in0}.v1);'''. \
                            format(le2=le2, **fmtspec)
        else:
            store_mask = '''_mm512_storeu_ps(mask, _mm512_maskz_mov_ps(
                              {in0}.v0, _mm512_set1_ps(1.0f)));
                            _mm512_storeu_ps(mask + {le2}, _mm512_maskz_mov_ps(
                              {in0}.v1, _mm512_set1_ps(1.0f)));'''. \
                            format(le2=le2, **fmtspec)
        return '''int i;
                  f32 mask[{le}], buf[{le}];
                  i16 offset_buf[{le}];
                  {store_mask}
                  {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in2});
                  {pre}storeu_ps(buf, {in3}.v0);
                  {pre}storeu_ps(buf + {le2}, {in3}.v1);
                  for (i = 0; i < {le}; i++) {{
                    if (nsimd_scalar_reinterpret_u32_f32(mask[i]) != (u32)0) {{
                      {in1}[offset_buf[i]] = nsimd_f32_to_f16(buf[i]);
                    }}
                  }}'''.format(le2=le2, store_mask=store_mask, **fmtspec)
    if simd_ext in (sse + avx) or typ in ['i8', 'u8', 'i16', 'u16']:
        cast = castsi(simd_ext, typ)
        if simd_ext in avx512:
            mask_decl = 'u64 mask;'
            store_mask = 'mask = (u64){in0};'.format(**fmtspec)
            cond = '(mask >> i) & 1'
        else:
            mask_decl = '{typ} mask[{le}];'.format(**fmtspec)
            store_mask = '{pre}storeu{sufsi}({cast}mask, {in0});'. \
                         format(cast=cast, **fmtspec)
            cond = 'nsimd_scalar_reinterpret_{utyp}_{typ}(mask[i]) != '\
                   '({utyp})0'.format(utyp='u' + typ[1:], **fmtspec)
        return '''int i;
                  {typ} buf[{le}];
                  {mask_decl}
                  {ityp} offset_buf[{le}];
                  {store_mask}
                  {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in2});
                  {pre}storeu{sufsi}({cast}buf, {in3});
                  for (i = 0; i < {le}; i++) {{
                    if ({cond}) {{
                      {in1}[offset_buf[i]] = buf[i];
                    }}
                  }}'''.format(ityp='i' + typ[1:], cast=cast, cond=cond,
                               mask_decl=mask_decl, store_mask=store_mask,
                               **fmtspec)
    # getting here means 32 and 64-bits types for avx512
    return '''{pre}mask_i{typnbits}scatter{suf}(
                  (void *){in1}, {in0}, {in2}, {in3}, {scale});'''. \
                  format(scale=int(typ[1:]) // 8, **fmtspec)

# -----------------------------------------------------------------------------
# gather

def gather(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  int i;
                  f32 buf[{le}];
                  i16 offset_buf[{le}];
                  {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in1});
                  for (i = 0; i < {le}; i++) {{
                    buf[i] = nsimd_f16_to_f32({in0}[offset_buf[i]]);
                  }}
                  ret.v0 = {pre}loadu_ps(buf);
                  ret.v1 = {pre}loadu_ps(buf + {leo2});
                  return ret;'''.format(leo2=int(fmtspec['le']) // 2,
                                        **fmtspec)
    if simd_ext in (sse + ['avx']) or typ in ['i8', 'u8', 'i16', 'u16']:
        cast = castsi(simd_ext, typ)
        return '''int i;
                  {typ} buf[{le}];
                  {ityp} offset_buf[{le}];
                  {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in1});
                  for (i = 0; i < {le}; i++) {{
                    buf[i] = {in0}[offset_buf[i]];
                  }}
                  return {pre}loadu{sufsi}({cast}buf);'''. \
                  format(ityp='i' + typ[1:], cast=cast, **fmtspec)
    # getting here means 32 and 64-bits types for avx2 and avx512
    if simd_ext == 'avx2':
        if typ in ['i64', 'u64']:
            cast = '(nsimd_longlong *)'
        elif typ in ['i32', 'u32']:
            cast = '(int *)'
        else:
            cast = '({typ} *)'.format(**fmtspec)
        return '''return {pre}i{typnbits}gather{suf}(
                             {cast}{in0}, {in1}, {scale});'''. \
                             format(scale=int(typ[1:]) // 8, cast=cast,
                                    **fmtspec)
    elif simd_ext in avx512:
        return 'return {pre}i{typnbits}gather{suf}({in1}, ' \
                      '(const void *){in0}, {scale});'. \
                      format(scale=int(typ[1:]) // 8, **fmtspec)

# -----------------------------------------------------------------------------
# linear gather

def gather_linear(simd_ext, typ):
    le = int(fmtspec['le'])
    cast = castsi(simd_ext, typ)
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  f32 buf[{le}];
                  int i;
                  for (i = 0; i < {le}; i++) {{
                    buf[i] = nsimd_f16_to_f32({in0}[i * {in1}]);
                  }}
                  ret.v0 = {pre}loadu_ps(buf);
                  ret.v1 = {pre}loadu_ps(buf + {leo2});
                  return ret;'''.format(leo2=le // 2, **fmtspec)
    emulation = '''{typ} buf[{le}];
                   int i;
                   for (i = 0; i < {le}; i++) {{
                     buf[i] = {in0}[i * {in1}];
                   }}
                   return {pre}loadu{sufsi}({cast}buf);'''. \
                   format(cast=cast, **fmtspec)
    if simd_ext == 'sse2' and typ not in ['i16', 'u16']:
        return emulation
    if simd_ext in sse + avx:
        trick = \
        '''nsimd_{simd_ext}_v{typ} ret;
           ret = {pre}undefined{sufsi}();
           '''.format(**fmtspec) + ''.join([
           set_lane(simd_ext, typ, 'ret', '{in0}[{i} * {in1}]'. \
                                          format(i=i, **fmtspec), i) + '\n' \
                                          for i in range(le)]) + \
        '''return ret;'''
        return '''#if NSIMD_WORD_SIZE == 32
                    {}
                  #else
                    {}
                  #endif
                  '''.format(emulation, trick)
    # getting here means AVX-512
    return \
    '''nsimd_avx2_v{typ} lo = _mm256_undefined{sufsi2}();
       nsimd_avx2_v{typ} hi = _mm256_undefined{sufsi2}();
       lo = nsimd_gather_linear_avx2_{typ}({in0}, {in1});
       hi = nsimd_gather_linear_avx2_{typ}({in0} + ({leo2} * {in1}), {in1});
       return {merge};'''.format(merge=setr(simd_ext, typ, 'lo', 'hi'),
                                 sufsi2=suf_si('avx2', typ),
                                 leo2=le // 2, **fmtspec)

# -----------------------------------------------------------------------------
# maksed gather

def maskoz_gather(oz, simd_ext, typ):
    if typ == 'f16':
        le2 = fmtspec['le'] // 2
        if simd_ext in sse + avx:
            store_mask = '''{pre}storeu_ps(mask, {in0}.v0);
                            {pre}storeu_ps(mask + {le2}, {in0}.v1);'''. \
                            format(le2=le2, **fmtspec)
        else:
            store_mask = '''_mm512_storeu_ps(mask, _mm512_maskz_mov_ps(
                              {in0}.v0, _mm512_set1_ps(1.0f)));
                            _mm512_storeu_ps(mask + {le2}, _mm512_maskz_mov_ps(
                              {in0}.v1, _mm512_set1_ps(1.0f)));'''. \
                            format(le2=le2, **fmtspec)
        if oz == 'z':
            store_oz = '''{pre}storeu_ps(buf, {pre}setzero_ps());
                          {pre}storeu_ps(buf + {le2}, {pre}setzero_ps());'''. \
                          format(le2=le2, **fmtspec)
        else:
            store_oz = '''{pre}storeu_ps(buf, {in3}.v0);
                          {pre}storeu_ps(buf + {le2}, {in3}.v1);'''. \
                          format(le2=le2, **fmtspec)
        return '''nsimd_{simd_ext}_vf16 ret;
                  int i;
                  f32 buf[{le}], mask[{le}];
                  i16 offset_buf[{le}];
                  {store_mask}
                  {store_oz}
                  {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in2});
                  for (i = 0; i < {le}; i++) {{
                    if (nsimd_scalar_reinterpret_u32_f32(mask[i]) != (u32)0) {{
                      buf[i] = nsimd_f16_to_f32({in1}[offset_buf[i]]);
                    }}
                  }}
                  ret.v0 = {pre}loadu_ps(buf);
                  ret.v1 = {pre}loadu_ps(buf + {leo2});
                  return ret;'''.format(leo2=le2, store_mask=store_mask,
                                        store_oz=store_oz, **fmtspec)
    if simd_ext in (sse + ['avx']) or typ in ['i8', 'u8', 'i16', 'u16']:
        cast = castsi(simd_ext, typ)
        if simd_ext in sse + avx:
            mask_decl = '{typ} mask[{le}];'.format(**fmtspec)
            store_mask = '{pre}storeu{sufsi}({cast}mask, {in0});'. \
                         format(cast=cast, **fmtspec)
            if typ in common.iutypes:
                comp = 'mask[i]'
            else:
                comp = 'nsimd_scalar_reinterpret_u{typnbits}_{typ}(mask[i])'. \
                       format(**fmtspec)
        else:
            mask_decl = 'u64 mask;'
            store_mask = 'mask = (u64){in0};'.format(**fmtspec)
            comp = '(mask >> i) & 1'
        if oz == 'z':
            store_oz = '''{pre}storeu{sufsi}({cast}buf,
                                             {pre}setzero{sufsi}());'''. \
                                             format(cast=cast, **fmtspec)
        else:
            store_oz = '{pre}storeu{sufsi}({cast}buf, {in3});'. \
                       format(cast=cast, **fmtspec)
        return '''int i;
                  {typ} buf[{le}];
                  {mask_decl}
                  {ityp} offset_buf[{le}];
                  {store_mask}
                  {store_oz}
                  {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in2});
                  for (i = 0; i < {le}; i++) {{
                    if ({comp}) {{
                      buf[i] = {in1}[offset_buf[i]];
                    }}
                  }}
                  return {pre}loadu{sufsi}({cast}buf);'''. \
                  format(ityp='i' + typ[1:], cast=cast, store_mask=store_mask,
                         store_oz=store_oz, comp=comp, mask_decl=mask_decl,
                         **fmtspec)
    # getting here means 32 and 64-bits types for avx2 and avx512
    if oz == 'o':
        src = '{in3}'.format(**fmtspec)
    else:
        src = '{pre}setzero{sufsi}()'.format(**fmtspec)
    if simd_ext == 'avx2':
        if typ in ['i64', 'u64']:
            cast = '(nsimd_longlong *)'
        elif typ in ['i32', 'u32']:
            cast = '(int *)'
        else:
            cast = '({typ} *)'.format(**fmtspec)
        return '''return {pre}mask_i{typnbits}gather{suf}({src},
                             {cast}{in1}, {in2}, {in0}, {scale});'''. \
                             format(scale=int(typ[1:]) // 8, cast=cast,
                                    src=src, **fmtspec)
    elif simd_ext in avx512:
        return 'return {pre}mask_i{typnbits}gather{suf}({src}, {in0}, ' \
                      '{in2}, (const void *){in1}, {scale});'. \
                      format(src=src, scale=int(typ[1:]) // 8, **fmtspec)


# -----------------------------------------------------------------------------
# get_impl function

def get_impl(opts, func, simd_ext, from_typ, to_typ):
    global fmtspec

    fmtspec = {
      'simd_ext': simd_ext,
      'typ': from_typ,
      'styp': get_native_typ(simd_ext, from_typ),
      'from_typ': from_typ,
      'to_typ': to_typ,
      'pre': pre(simd_ext),
    #  'suf': suf_ep(from_typ),
    #  'sufsi': suf_si(simd_ext, from_typ),
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
        'masko_loada1': lambda: maskoz_load(simd_ext, from_typ, 'o', True),
        'maskz_loada1': lambda: maskoz_load(simd_ext, from_typ, 'z', True),
        'load2a': lambda: load_deg234(simd_ext, from_typ, True, 2),
        'load3a': lambda: load_deg234(simd_ext, from_typ, True, 3),
        'load4a': lambda: load_deg234(simd_ext, from_typ, True, 4),
        'loadu': lambda: load(simd_ext, from_typ, False),
        'masko_loadu1': lambda: maskoz_load(simd_ext, from_typ, 'o', False),
        'maskz_loadu1': lambda: maskoz_load(simd_ext, from_typ, 'z', False),
        'load2u': lambda: load_deg234(simd_ext, from_typ, False, 2),
        'load3u': lambda: load_deg234(simd_ext, from_typ, False, 3),
        'load4u': lambda: load_deg234(simd_ext, from_typ, False, 4),
        'storea': lambda: store(simd_ext, from_typ, True),
        'mask_storea1': lambda: mask_store(simd_ext, from_typ, True),
        'store2a': lambda: store_deg234(simd_ext, from_typ, True, 2),
        'store3a': lambda: store_deg234(simd_ext, from_typ, True, 3),
        'store4a': lambda: store_deg234(simd_ext, from_typ, True, 4),
        'storeu': lambda: store(simd_ext, from_typ, False),
        'mask_storeu1': lambda: mask_store(simd_ext, from_typ, False),
        'store2u': lambda: store_deg234(simd_ext, from_typ, False, 2),
        'store3u': lambda: store_deg234(simd_ext, from_typ, False, 3),
        'store4u': lambda: store_deg234(simd_ext, from_typ, False, 4),
        'gather': lambda: gather(simd_ext, from_typ),
        'gather_linear': lambda: gather_linear(simd_ext, from_typ),
        'masko_gather': lambda: maskoz_gather('o', simd_ext, from_typ),
        'maskz_gather': lambda: maskoz_gather('z', simd_ext, from_typ),
        'scatter': lambda: scatter(simd_ext, from_typ),
        'scatter_linear': lambda: scatter_linear(simd_ext, from_typ),
        'mask_scatter': lambda: mask_scatter(simd_ext, from_typ),
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
        'adds': lambda: adds(simd_ext, from_typ),
        'subs': lambda: subs(simd_ext, from_typ),
        'div': lambda: div2(opts, simd_ext, from_typ),
        'sqrt': lambda: sqrt1(simd_ext, from_typ),
        'len': lambda: len1(simd_ext, from_typ),
        'mul': lambda: mul2(opts, simd_ext, from_typ),
        'shl': lambda: shl_shr('shl', simd_ext, from_typ),
        'shr': lambda: shl_shr('shr', simd_ext, from_typ),
        'shra': lambda: shra(opts, simd_ext, from_typ),
        'set1': lambda: set1(simd_ext, from_typ),
        'set1l': lambda: set1l(simd_ext, from_typ),
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
        'ceil': lambda: round1(opts, 'ceil', simd_ext, from_typ),
        'floor': lambda: round1(opts, 'floor', simd_ext, from_typ),
        'trunc': lambda: trunc1(opts, simd_ext, from_typ),
        'round_to_even': lambda: round_to_even1(opts, simd_ext, from_typ),
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
        'downcvt': lambda: downcvt1(opts, simd_ext, from_typ, to_typ),
        'to_mask': lambda: to_mask1(simd_ext, from_typ),
        'to_logical': lambda: to_logical1(simd_ext, from_typ),
        'ziplo': lambda: zip_half('ziplo', simd_ext, from_typ),
        'ziphi': lambda: zip_half('ziphi', simd_ext, from_typ),
        'unziplo': lambda: unzip_half(opts, 'unziplo', simd_ext, from_typ),
        'unziphi': lambda: unzip_half(opts, 'unziphi', simd_ext, from_typ),
        'zip' : lambda : zip(simd_ext, from_typ),
        'unzip' : lambda : unzip(simd_ext, from_typ),
        'mask_for_loop_tail': lambda : mask_for_loop_tail(simd_ext, from_typ),
        'iota': lambda : iota(simd_ext, from_typ)
    }
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if not from_typ in common.types:
        raise ValueError('Unknown type "{}"'.format(from_typ))
    if not func in impls:
        return common.NOT_IMPLEMENTED
    else:
        return impls[func]()
