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
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABIœLITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This file gives the implementation of platform wasm, i.e. web-assembly
# Reading this file is NOT straightforward. X86 SIMD extensions is a mess.
# This script implements wasm-simd128

import common

# -----------------------------------------------------------------------------
# Implementation of mandatory functions for this module


def get_simd_exts():
    return ['wasm']


def emulate_fp16(simd_ext):
    if not simd_ext in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    return True


def get_native_typ(simd_ext, typ):
    if simd_ext not in get_simd_exts():
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
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    else:
        return False


def get_additional_include(func, platform, simd_ext):
    ret = '''#include <nsimd/cpu/cpu/{}.h>
                  '''.format(func)

    if func in ['fma', 'fms', 'fnma', 'fnms']:
        ret += '''#include <nsimd/wasm/{simd_ext}/add.h>
                      #include <nsimd/wasm/{simd_ext}/sub.h>
                      #include <nsimd/wasm/{simd_ext}/mul.h>
                      #include <nsimd/wasm/{simd_ext}/div.h>
                      '''.format(**fmtspec)

    if func == 'mask_for_loop_tail':
        ret += '''#include <nsimd/wasm/{simd_ext}/set1.h>
                  #include <nsimd/wasm/{simd_ext}/set1l.h>
                  #include <nsimd/wasm/{simd_ext}/iota.h>
                  #include <nsimd/wasm/{simd_ext}/lt.h>
                  '''.format(simd_ext=simd_ext)

    if func in ['store2u', 'store3u', 'store4u', 'store2a', 'store3a',
                'store4a', 'storeu', 'storea']:
        deg = func[5]
        deg = deg if deg not in "lua" else 1
        args = ','.join(['nsimd_{simd_ext}_vu16'.format(simd_ext=simd_ext)
                         for i in range(1, int(deg) + 1)])
        ret += """
               # include <nsimd/wasm/{simd_ext}/loadu.h>
               # include <nsimd/wasm/{simd_ext}/storeu.h>

               # if NSIMD_CXX > 0
               extern "C" {{
               # endif

               NSIMD_INLINE void NSIMD_VECTORCALL
               nsimd_{func}_{simd_ext}_u16(u16*, {args});

               # if NSIMD_CXX > 0
               }} // extern "C"
               # endif
              """.format(func=func, args=args, simd_ext=simd_ext)

    if func in ['load2u', 'load3u', 'load4u', 'load2a', 'load3a', 'load4a']:
        ret += '''
                  # include <nsimd/wasm/{simd_ext}/loadu.h>
                  # include <nsimd/wasm/{simd_ext}/storeu.h>

                  # if NSIMD_CXX > 0
                  extern "C" {{
                  # endif

                  NSIMD_INLINE nsimd_{simd_ext}_vu16x{deg} NSIMD_VECTORCALL
                  nsimd_{func}_{simd_ext}_u16(const u16*);

                  # if NSIMD_CXX > 0
                  }} // extern "C"
                  # endif
                  '''.format(func=func, deg=func[4], simd_ext=simd_ext)

    if func in ['loadlu', 'loadla']:
        ret += '''#include <nsimd/wasm/{simd_ext}/loadu.h>
                    #include <nsimd/wasm/{simd_ext}/loada.h>
                    #include <nsimd/wasm/{simd_ext}/to_logical.h>
                    #include <nsimd/wasm/{simd_ext}/eq.h>
                      #include <nsimd/wasm/{simd_ext}/set1.h>
                      #include <nsimd/wasm/{simd_ext}/{load}.h>
                      #include <nsimd/wasm/{simd_ext}/notl.h>
                      '''.format(load='load' + func[5], **fmtspec)

    elif func in ['storelu']:
        ret += '''#include <nsimd/wasm/{simd_ext}/if_else1.h>
                     #include <nsimd/wasm/{simd_ext}/set1.h>
                     '''.format(**fmtspec)
    elif func in ['to_logical']:
        ret += '''#include <nsimd/wasm/{simd_ext}/reinterpret.h>
        #include <nsimd/wasm/{simd_ext}/reinterpretl.h>
        #include <nsimd/wasm/{simd_ext}/ne.h>
                     '''.format(**fmtspec)

    return ret

# Signature must be a list of 'v', 's'
#   'v' means vector so code to extract has to be emitted
#   's' means base type so no need to write code for extraction
def get_emulation_code(func, signature, simd_ext, typ):
    # Trick using insert and extract
    trick = 'nsimd_{simd_ext}_v{typ} ret = {undef};\n'. \
           format(undef=get_undefined(simd_ext, typ), **fmtspec)
    arity = len(signature)
    trick += typ + ' ' + \
            ', '.join(['tmp{}'.format(i) \
                       for i in range(arity) if signature[i] == 'v']) + ';\n'
    args = ', '.join(['{{in{}}}'.format(i).format(**fmtspec) \
                      if signature[i] == 's' else 'tmp{}'.format(i) \
                      for i in range(arity)])
    for i in range(fmtspec['le']):
        trick += '\n'.join(['tmp{} = {};'. \
                format(j, get_lane(simd_ext, typ,
                       '{{in{}}}'.format(j).format(**fmtspec), i)) \
                       for j in range(arity) if signature[j] == 'v']) + '\n'
        trick += set_lane(simd_ext, typ, 'ret',
                         'nsimd_scalar_{func}_{typ}({args})'. \
                         format(func=func, args=args, **fmtspec), i) + '\n'
    trick += 'return ret;'

    # but in 32-bits mode insert and extract instrinsics are almost never
    # available so we emulate
    emulation = 'int i;\n{typ} ret[{le}];\n'.format(**fmtspec)
    emulation += typ + ' ' + \
                 ', '.join(['buf{}[{}]'.format(i, fmtspec['le']) \
                            for i in range(arity) if signature[i] == 'v']) + \
                            ';\n'
    emulation += '\n'.join(['{{pre}}store{{sufsi}}({cast}buf{i}, {{in{i}}});'. \
                            format(i=i, cast=castsi(simd_ext, typ)). \
                            format(**fmtspec) \
                            for i in range(arity) if signature[i] == 'v']) + \
                            '\n'
    args = ', '.join(['{{in{}}}'.format(i).format(**fmtspec) \
                      if signature[i] == 's' else 'buf{}[i]'.format(i) \
                      for i in range(arity)])
    emulation += '''for (i = 0; i < {le}; i++) {{
                      ret[i] = nsimd_scalar_{func}_{typ}({args});
                    }}
                    return {pre}loadu{sufsi}({cast}ret);'''. \
                    format(args=args, cast=castsi(simd_ext, typ), func=func,
                           **fmtspec)

    if typ in ['i8', 'u8', 'i16', 'u16', 'i32', 'u32', 'f32']:
        return trick
    else:
        return '''#if NSIMD_WORD_SIZE == 32
                    {}
                  #else
                    {}
                  #endif'''.format(emulation, trick)


def emulate_op2(opts, op, simd_ext, typ):
    func = {'/': 'div', '*': 'mul'}
    return get_emulation_code(func[op], ['v', 'v'], simd_ext, typ)

# -----------------------------------------------------------------------------
# Function prefixes

def get_len(typ):
    return 128 // int(typ[1:])

def pre(simd_ext):
    return 'wasm_v128_'


def pretyp(simd_ext, typ):
    return 'wasm_{}x{}'.format(typ, get_len(typ))

def pretyp2(simd_ext, typ):
    if typ in common.itypes + common.ftypes:
        return pretyp(simd_ext, typ)
    return 'wasm_u{}x{}'.format(typ[1:], get_len(typ))


def nbits(simd_ext):
    return '128'

# -----------------------------------------------------------------------------
# Other helper functions

fmtspec = {}


def set_lane(simd_ext, typ, var_name, scalar, i):
    # No code for f16's
    if typ == 'f16':
        return ''
    pt = pretyp(simd_ext, 'i'+ typ[1:]) if typ[0] == 'u' else fmtspec['pretyp']
    if typ in common.itypes + common.ftypes:
        return f'{var_name} = {pt}_replace_lane({var_name}, {i}, {scalar});'
    else:
        if typ[0] == 'u':
            typ2 = 'i' + typ[1:]
        return \
            '{} = {}_replace_lane({}, {}, nsimd_scalar_reinterpret_i{}_{}({}))'. \
                format(var_name, pt, var_name, i, fmtspec['typnbits'], typ2,
                       scalar)


def get_lane(simd_ext, typ, var_name, i):
    # No code for f16's
    if typ == 'f16':
        return ''
    if typ in common.itypes + ['u8', 'u16'] + common.ftypes:
        return f'{pretyp(simd_ext, typ)}_extract_lane({var_name}, {i})'
    return 'nsimd_scalar_reinterpret_u{}_{}({}_extract_lane({}, {}))'. \
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
                        format(func=func, args=args, **fmtspec), i) + ';\n'
    ret += 'return ret;'
    return ret


def how_it_should_be_op1(op, simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = wasm_f32x4_{op}({in0}.v0);
                  ret.v1 = wasm_f32x4_{op}({in0}.v1);
                  return ret;'''.format(**fmtspec, op=op)
    return 'return {pretyp}_{op}({in0});'. \
        format(**fmtspec, op=op)


def how_it_should_be_op2(func, simd_ext, typ):
    if typ == 'f16':
        pt = pretyp(simd_ext, "f32")
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = {pt}_{func}({in0}.v0, {in1}.v0);
                  ret.v1 = {pt}_{func}({in0}.v1, {in1}.v1);
                  return ret;'''.format(**fmtspec, pt=pt, func=func)
    else:
        return 'return {pretyp2}_{func}({in0}, {in1});'. \
            format(**fmtspec, pretyp2=pretyp(simd_ext, typ), func=func)


# -----------------------------------------------------------------------------
# Returns C code for func

# Load
def load(simd_ext, typ, aligned):
    if typ == 'f16':
        return """
               nsimd_{simd_ext}_vf16 ret;

               ret.v0 = wasm_v128_load(a0);
               ret.v1 = wasm_v128_load(a0 + {le}/2);
               return ret;
               """.format(**fmtspec)

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
        return '''
               return wasm_v128_load((void *){in0});

               '''.format(**fmtspec)


# -----------------------------------------------------------------------------
# masked loads

# -----------------------------------------------------------------------------
# Loads of degree 2, 3 and 4
def load_deg234(simd_ext, typ, align, deg):
    if not simd_ext in get_simd_exts():
        return common.NOT_IMPLEMENTED
    if typ == "f16":
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

    seq = list(range(128 // int(typ[1:])))
    seq_even = ', '.join([str(2 * i) for i in seq])
    seq_odd = ', '.join([str(2 * i + 1) for i in seq])

    if deg == 2:
        return '''nsimd_{simd_ext}_v{typ}x2 ret;
        
                  /* don't know why but loads doesn't work... */
                  v128_t a = wasm_v128_load((void *){in0});
                  v128_t b = wasm_v128_load((void *)({in0} + {le}));

                  ret.v0 = wasm_v{typnbits}x{le}_shuffle(a, b, {seq_even});
                  ret.v1 = wasm_v{typnbits}x{le}_shuffle(a, b, {seq_odd});

                  return ret;'''.format(seq_even=seq_even, seq_odd=seq_odd, nbits2=get_len(typ),
                                        **fmtspec)
    if deg == 3:
        shuffle = ""
        load_block = """
                     v128_t a = wasm_v128_load((void *){in0});
                     v128_t b = wasm_v128_load((void *)({in0} + {nbits2}));
                     v128_t c = wasm_v128_load((void *)({in0} + {nbits2}*2));
                     """.format(**fmtspec, nbits2=get_len(typ))

        if typ[1:] == "64":
            shuffle = """
                      ret.v0 = wasm_v64x2_shuffle(a, b, 0, 3);
                      ret.v1 = wasm_v64x2_shuffle(a, c, 1, 2);
                      ret.v2 = wasm_v64x2_shuffle(b, c, 0, 3);
                      """
        elif typ[1:] == "32":
            shuffle = """
                      v128_t rrgg12 = wasm_v32x4_shuffle(a, b, 0, 3, 1, 4);
                      v128_t rrgg34 = wasm_v32x4_shuffle(b, c, 2, 5, 3, 6);
                      
                      ret.v0 = wasm_v32x4_shuffle(rrgg12, rrgg34, 0,1,4,5);
                      ret.v1 = wasm_v32x4_shuffle(rrgg12, rrgg34, 2,3,6,7);
                      
                      v128_t bb = wasm_v32x4_shuffle(a, b, 2, 5, 0, 0);
                      ret.v2 = wasm_v32x4_shuffle(bb, c, 0,1,4,7);
                      """
        elif typ[1:] == "16":
            shuffle = """
                      v128_t rg1234 = wasm_v16x8_shuffle(a, b, 0, 3, 6, 9, 1, 4, 7, 10);
                      v128_t rg5678 = wasm_v16x8_shuffle(b, c, 4, 7, 10, 13, 5, 8, 11, 14);
                      
                      ret.v0 = wasm_v16x8_shuffle(rg1234, rg5678, 0,1,2,3,8,9,10,11);
                      ret.v1 = wasm_v16x8_shuffle(rg1234, rg5678, 4,5,6,7,12,13,14,15);
                      
                      v128_t bx5 = wasm_v16x8_shuffle(a, b, 2, 5, 8, 11, 14, 0 ,0 ,0);
                      ret.v2 = wasm_v16x8_shuffle(bx5, c, 0,1,2,3,4,9, 12, 15);
                      """
        elif typ[1:] == "8":
            shuffle = """
                      v128_t rx11_gx5 = wasm_v8x16_shuffle(a, b, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 1, 4, 7, 10, 13);
                      v128_t rx5_gx11 = wasm_v8x16_shuffle(b, c, 17, 20, 23, 26, 29, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30);
                      
                      ret.v0 = wasm_v8x16_shuffle(rx11_gx5, rx5_gx11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 18, 19, 20);
                      ret.v1 = wasm_v8x16_shuffle(rx11_gx5, rx5_gx11, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
                      
                      v128_t bx10 = wasm_v8x16_shuffle(a, b, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
                      ret.v2 = wasm_v8x16_shuffle(bx10, c, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);
                      """

        return """
                nsimd_{simd_ext}_v{typ}x3 ret;
                {load}
                {shuffle}
                return ret;
                """.format(**fmtspec, shuffle=shuffle, load=load_block, nbits2=get_len(typ))

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
                      return ret;'''.format(seq_even=seq_even, seq_odd=seq_odd, nbits2=get_len(typ),
                                            **fmtspec)
        else:
            seq = list(range(128 // int(typ[1:]) // 2))
            seq_ab = ', '.join([str(4 * i) for i in seq] + [str(4 * i + 1) for i in seq])
            seq_cd = ', '.join([str(4 * i + 2) for i in seq] + \
                               [str(4 * i + 3) for i in seq])
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
                  return ret;'''.format(seq_ab=seq_ab, seq_cd=seq_cd, lex2=lex2, nbits2=get_len(typ),
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
                format(variables=variables, code=code, a=a, deg=deg if deg != 1 else "", **fmtspec)

    if deg == 2:
        list_unpack = []
        for i in range(get_len(typ)):
            list_unpack.append(i)
            list_unpack.append(get_len(typ) + i)

        middle = get_len(typ)
        seq_unpacklo = ', '.join(str(i) for i in list_unpack[:middle])
        seq_unpackhi = ', '.join(str(i) for i in list_unpack[middle:])

        return """
                  nsimd_{simd_ext}_v{typ} buf0 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, {seq_unpacklo});
                  wasm_v128_store({in0}, buf0);
                  nsimd_{simd_ext}_v{typ} buf1 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, {seq_unpackhi});
                  wasm_v128_store({in0} + {nbits2}, buf1);
               """.format(**fmtspec, seq_unpacklo=seq_unpacklo, seq_unpackhi=seq_unpackhi,
                          suf_align="a" if align else "u",
                          nbits2=get_len(typ), native_typ=get_native_typ(simd_ext, typ),
                          typ1=typ[1:])
    if deg == 3:
        if typ[1:] == "8":
            return """
                   nsimd_{simd_ext}_v{typ} rg1_6 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 0, 0, 0, 0, 0); /* without g6 */
                   nsimd_{simd_ext}_v{typ} rg6_11 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 21, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 0, 0, 0, 0); /* without r6 */
                   nsimd_{simd_ext}_v{typ} rg12_16 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31, 0, 0, 0, 0, 0, 0);
                   
                   wasm_v128_store({in0}, wasm_v{typnbits}x{le}_shuffle(rg1_6 , {in3}, 0, 1, 16, 2, 3, 17, 4, 5, 18, 6, 7, 19, 8, 9, 20, 10));
                   wasm_v128_store({in0} + {nbits2}, wasm_v{typnbits}x{le}_shuffle(rg6_11 , {in3}, 1, 21, 2, 3, 22, 4, 5, 23, 6, 7, 24, 8, 9, 25, 10, 11));
                   wasm_v128_store({in0} + 2*{nbits2}, wasm_v{typnbits}x{le}_shuffle(rg12_16 , {in3}, 26, 0, 1, 27, 2, 3, 28, 4, 5, 29, 6, 7, 30, 8, 9, 31));
                   """.format(**fmtspec, nbits2=get_len(typ))
        if typ[1:] == "16":
            return """
                   nsimd_{simd_ext}_v{typ} rg123_ = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 0,8,1,9,2,10,10,10);
                   nsimd_{simd_ext}_v{typ} rg456_ = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 3, 11, 4, 12, 5, 5,5,5);
                   nsimd_{simd_ext}_v{typ} rg678_ = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 13, 13, 6, 14, 7, 15,15,15);
                   
                   wasm_v128_store({in0}, wasm_v{typnbits}x{le}_shuffle(rg123_ , {in3}, 0,1,8,2,3,9,4,5));
                   wasm_v128_store({in0} + {nbits2}, wasm_v{typnbits}x{le}_shuffle(rg456_ , {in3}, 10, 0,1, 11, 2, 3, 12, 4));
                   wasm_v128_store({in0} + 2*{nbits2}, wasm_v{typnbits}x{le}_shuffle(rg678_ , {in3}, 1, 13, 2,3, 14, 4, 5, 15));
                   """.format(**fmtspec, nbits2=get_len(typ))
        elif typ[1:] == "32":
            return """
                   nsimd_{simd_ext}_v{typ} rg1r_2 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 0, 4, 1, 1);
                   nsimd_{simd_ext}_v{typ} bb23r3_ = wasm_v{typnbits}x{le}_shuffle({in3}, {in1}, 1, 2, 6, 3);
                    
                   wasm_v128_store({in0}, wasm_v{typnbits}x{le}_shuffle(rg1r_2, {in3}, 0, 1, 4, 2));
                   wasm_v128_store({in0} + {nbits2}, wasm_v{typnbits}x{le}_shuffle(bb23r3_, {in2}, 5, 0, 2, 6));
                   
                   nsimd_{simd_ext}_v{typ} r4g4__ = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 3, 7, 7, 7);
                   wasm_v128_store({in0} + 2*{nbits2}, wasm_v{typnbits}x{le}_shuffle(r4g4__, {in3}, 6, 0, 1, 7));
                   """.format(**fmtspec, nbits2=get_len(typ))
        elif typ[1:] == "64":
            return """
                   nsimd_{simd_ext}_v{typ} buf0 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 0, 2);
                   wasm_v128_store({in0}, buf0);
                   

                   nsimd_{simd_ext}_v{typ} buf1 = wasm_v{typnbits}x{le}_shuffle({in3}, {in1}, 0, 3);
                   wasm_v128_store({in0} + {nbits2}, buf1);
                   
                   nsimd_{simd_ext}_v{typ} buf2 = wasm_v{typnbits}x{le}_shuffle({in2}, {in3}, 1, 3);
                   wasm_v128_store({in0} + 2*{nbits2}, buf2);
                   
                   """.format(**fmtspec, nbits2=get_len(typ))

    if deg == 4:

        le = get_len(typ)

        # indexes to shuffle intermediate var
        list_unpack = []

        # indexes to shuffle ret var according to the return type
        dic_unpack_retlo = {
            "64": [0, 1],
            "32": [0, 1, 4, 5],
            "16": [0, 1, 8, 9, 2, 3, 10, 11],
            "8" : [0, 1, 16, 17, 2, 3, 18, 19, 4,5, 20, 21,  6, 7, 22, 23]
        }
        list_unpack_retlo = dic_unpack_retlo[typ[1:]]

        list_unpack_rethi = []
        for i in range(len(list_unpack_retlo)):
            inc = 1 if typ[1:] == "64" else 0
            list_unpack_rethi.append(list_unpack_retlo[i] + le // 2 + inc)

        for i in range(get_len(typ)):
            list_unpack.append(i)
            list_unpack.append(le + i)

        seq_unpacklo = ', '.join(str(i) for i in list_unpack[:le])
        seq_unpackhi = ', '.join(str(i) for i in list_unpack[le:])

        seq_unpack_retlo = ', '.join(str(i) for i in list_unpack_retlo)
        seq_unpack_rethi = ', '.join(str(i) for i in list_unpack_rethi)

        return """
                  nsimd_{simd_ext}_v{typ} rg1 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, {seq_unpacklo});
                  nsimd_{simd_ext}_v{typ} rg2 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, {seq_unpackhi});
                  nsimd_{simd_ext}_v{typ} ba1 = wasm_v{typnbits}x{le}_shuffle({in3}, {in4}, {seq_unpacklo});
                  nsimd_{simd_ext}_v{typ} ba2 = wasm_v{typnbits}x{le}_shuffle({in3}, {in4}, {seq_unpackhi});
               
                  wasm_v128_store({in0}, wasm_v{typnbits}x{le}_shuffle(rg1, ba1, {seq_unpack_retlo}));
                  wasm_v128_store({in0} + {nbits2}, wasm_v{typnbits}x{le}_shuffle(rg1, ba1, {seq_unpack_rethi}));
                  wasm_v128_store({in0} + 2 * {nbits2}, wasm_v{typnbits}x{le}_shuffle(rg2, ba2, {seq_unpack_retlo}));
                  wasm_v128_store({in0} + 3 * {nbits2}, wasm_v{typnbits}x{le}_shuffle(rg2, ba2, {seq_unpack_rethi}));
               """.format(**fmtspec, nbits2=le,
                          seq_unpacklo=seq_unpacklo, seq_unpackhi=seq_unpackhi,
                          seq_unpack_rethi=seq_unpack_rethi, seq_unpack_retlo=seq_unpack_retlo)

    return common.NOT_IMPLEMENTED


# -----------------------------------------------------------------------------
# Store

def store(simd_ext, typ, aligned):
    align = '' if aligned else 'u'
    if typ == 'f16': #todo still experimental
        return """
               /*{pre}store{le}_lane({in0}, {in1}.v0, 0);
               {pre}store{le}_lane({in0} + {le} /2, {in1}.v1, 0);*/
               wasm_v128_store({in0}, {in1}.v0);
               wasm_v128_store({in0} + {le}/2, {in1}.v1);
               """.format(**fmtspec)
    return """
           wasm_v128_store({in0}, {in1});
           """.format(**fmtspec)


#
# # masked store

# # -----------------------------------------------------------------------------
# # Code for binary operators: and, or, xor

# # -----------------------------------------------------------------------------
# # Code for comparisons

def eq2(simd_ext, typ):
    if typ == 'f16':
        return '/*TODO*/'
    if typ in ['i64', 'u64']:
        return \
        '''v128_t t = wasm_i32x4_eq({in0}, {in1});
            return wasm_v128_and(t,
                    wasm_i32x4_shuffle(t, t, 2,3,0,1) /* = 2|3|0|1 */);'''. \
                    format(**fmtspec)
    else:
        return '/*TODO*/'

def comp(op, simd_ext, typ):
    if op == "andnot":
        func = "{pre}andnot".format(**fmtspec)
    else:
        func = "{pretyp1}_{op}".format(pretyp1=pretyp(simd_ext,typ), op=op)
    if typ == "f16":
        return '/*TODO*/'

    return 'return {func}({in0}, {in1});'.format(**fmtspec,op=op,func=func)

def comp_bin(op, simd_ext, typ):
    if op[-1] == 'l':
        op = op[:-1]
    # arity: 1
    op_map_1 = {
        "not": "~",
    }
    op_map_2 = {
        "and": "&",
        "or": "|",
        "xor": "^",
        "andnot": "& ~"
    }
    if op in op_map_1:
        if typ == "f16":
            return """
                   nsimd_{simd_ext}_vf16 ret;
                   ret.v0 =  {op}{in0}.v0;
                   ret.v1 = {op}{in0}.v1;
                   return ret;
                   """.format(**fmtspec, op=op_map_1[op])
        return """
               return {op}{in0};
               """.format(**fmtspec, op=op_map_1[op])
    if typ == "f16":
        return """
               nsimd_{simd_ext}_vf16 ret;
               ret.v0 =  {in0}.v0 {op} {in1}.v0;
               ret.v1 = {in0}.v1 {op} {in1}.v1;
               return ret;
               """.format(**fmtspec,
                          op=op_map_2[op])
    return """
           return {in0} {op} {in1};
           """.format(**fmtspec,
                      op=op_map_2[op],
                      )

# -----------------------------------------------------------------------------
# Code for logical binary operators: andl, orl, xorl

def binlop2(func, simd_ext, typ):
    op = { 'orl': '|', 'xorl': '^', 'andl': '&' }
    op_fct = { 'orl': 'kor', 'xorl': 'kxor', 'andl': 'kand' }
    if typ == 'f16':
        return '/*TODO*/'
    else:
        return comp_bin(func, simd_ext, typ)


# # -----------------------------------------------------------------------------
# # andnot

# # -----------------------------------------------------------------------------
# # logical andnot

# # -----------------------------------------------------------------------------
# # Code for unary not

# -----------------------------------------------------------------------------
# Code for unary logical lnot

def lnot1(simd_ext, typ):
    ntyp = get_native_typ(simd_ext, typ)
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vlf16 ret;
                  ret.v0 = ~{in0}.v0;
                  ret.v1 = ~{in0}.v1;
                  return ret;'''.format(**fmtspec)
    else:
        return '''
               int i;

               nsimd_{simd_ext}_v{typ} ret = ~a0;
               for(i=0; i<{le}; ++i)
                  ret[i] = ~{in0}[i];
               
               ret[1] = (u64) -1;
               return ret; /* ~{in0} */
               '''.format(**fmtspec)

# -----------------------------------------------------------------------------
# Addition and substraction

def addsub(op, simd_ext, typ):
    if (op in ['adds', 'subs']) and (typ in ['i8', 'u8', 'i16', 'u16']):
        return 'return wasm_{typ}x{len}_{op}_sat({in0}, {in1});'. \
               format(**fmtspec, op=op[:-1], len=get_len(typ))
    elif (op in ['add', 'sub']) and (typ in ['u8', 'u16', 'u32', 'u64']):
        return 'return wasm_i{typ2}x{len}_{op}({in0}, {in1});'. \
               format(**fmtspec, typ2=typ[1:], op=op, len=get_len(typ))
    elif (op in ['adds', 'subs']) and (typ in ['u32', 'u64', 'i32', 'i64']):
        typ2 = 'i' + typ[1:]
        len_ = get_len(typ)
        in0 = fmtspec['in0']
        in1 = fmtspec['in1']
        ret = f'nsimd_{simd_ext}_v{typ} ret;'
        ret+= '\n'
        ret+= f'''wasm_i64x2_const_splat(0);'''
        ret+= '\n'
        for i in range(len_):
            tmp = f'nsimd_scalar_{op}_{typ}(({typ}){get_lane(simd_ext, typ2, in0, i)},({typ}){get_lane(simd_ext, typ2, in1, i)})'
            ret+= set_lane(simd_ext, typ, 'ret', tmp, i) + ';'
            ret+= '\n'
        ret+= 'return ret;'
        return ret
    elif (op in ['adds', 'subs']) and (typ in ['i32', 'i64']):
        return 'return wasm_i{typ2}x{len}_{op}({in0}, {in1});'. \
               format(**fmtspec, typ2=typ[1:], op=op[:-1], len=get_len(typ))
    else:
        if op in ['adds', 'subs']:
            op = op[:-1]
        return how_it_should_be_op2(op, simd_ext, typ)


# -----------------------------------------------------------------------------
# Multiplication

def mul2(simd_ext, typ):
    if typ == "i8" or typ == "u8":
        return '''
                nsimd_{simd_ext}_v{typ} lo =
                    wasm_i16x8_mul({in0}, {in1});
                nsimd_{simd_ext}_v{typ} hi = wasm_i16x8_shl(
                    wasm_i16x8_mul(wasm_u16x8_shr({in0}, 8),
                        wasm_u16x8_shr({in1}, 8)), 8);
                return wasm_v128_or(wasm_v128_and(
                            lo, wasm_i16x8_splat(255)),hi);'''.format(**fmtspec)
    else:
        typ2 = typ
        if typ[0] == "u":
            typ2 = "i" + typ[1:]
        return how_it_should_be_op2('mul', simd_ext, typ)

# -----------------------------------------------------------------------------
# Division

def div2(opts, simd_ext, typ):
    if typ in ['f32', 'f64', 'f16']:
        return how_it_should_be_op2('div', simd_ext, typ)
    return emulate_op2(opts, '/', simd_ext, typ)

# -----------------------------------------------------------------------------
# Len

def len1(simd_ext, typ):
    return 'return {le};'.format(**fmtspec)


# -----------------------------------------------------------------------------
# Shift left and right
def shl_shr(func, simd_ext, typ):
    if typ[0] == "u":
        return """
        return nsimd_{func}_{simd_ext}_{typ}({in0}, {in1});
        """.format(**fmtspec, func=func)
    if typ == "f16":
        return """
               nsimd_{simd_ext}_v{typ} ret;
               ret.v0 = nsimd_{func}_{nsimd_ext}_f32({in0}.v0, {in1});
               ret.v1 = nsimd_{func}_{nsimd_ext}_f32({in0}.v1, {in1});
               return ret;
               """.format(func=func, **fmtspec)
    return """
           return {pretyp}_{func}({in0}, {in1});
           """.format(**fmtspec, func=func)


# # -----------------------------------------------------------------------------
# # Arithmetic shift right

def shra(opts, simd_ext, typ):
    if typ in common.utypes:
        # For unsigned type, logical shift
        return 'return nsimd_shr_{simd_ext}_{typ}({in0}, {in1});'. \
            format(**fmtspec)
    # TODO
    return """
           
           """.format(**fmtspec)


# # -----------------------------------------------------------------------------
# # set1 or splat function
#
def set1(simd_ext, typ):
    #if typ[0] == "u":
    #    typ = "i" + typ[1:]
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  f32 f = nsimd_f16_to_f32({in0});
                  ret.v0 = nsimd_set1_{simd_ext}_f32(f);
                  ret.v1 = ret.v0;
                  return ret;'''.format(**fmtspec)
    return 'return {pretyp1}_splat({in0});'.format(**fmtspec,
                            pretyp1=pretyp(simd_ext, typ))


#
# # -----------------------------------------------------------------------------
# # set1l or splat function for logical
#
def set1l(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vlf16 ret;
                  ret.v0 = nsimd_set1l_{simd_ext}_f32({in0});
                  ret.v1 = ret.v0;
                  return ret;'''.format(**fmtspec)
    return """
           /*if({in0})
              return {pretyp}eq((u{bits})-1);
           return {pretyp}_splat(0);*/
           """.format(**fmtspec, bits=typ[1:])


# -----------------------------------------------------------------------------
# if_else1 function
def if_else1(simd_ext, typ):
    if typ == 'f16':
        return """
               nsimd_{simd_ext}_vf16 ret;
               ret.v0 = nsimd_if_else1_{simd_ext}_f32(
               {in0}.v0, {in1}.v0, {in2}.v0);
               ret.v1 = nsimd_if_else1_{simd_ext}_f32(
               {in0}.v1, {in1}.v1, {in2}.v1);
               return ret;
               """.format(**fmtspec)
    return """
           nsimd_{simd_ext}_v{typ} mask0 = {in1} & {in0};
           nsimd_{simd_ext}_v{typ} mask1 = {in2} & ~{in0};
           return mask0 | mask1;
           """.format(**fmtspec)


# -----------------------------------------------------------------------------
# min and max functions
def minmax(func, simd_ext, typ):
    return ""
    #todo: fix intrinsics
    if typ =="f16":
        return """
               nsimd_{simd_ext}_vf16 ret;
               ret.v0 = {pretyp1}_{op}({in0}.v0);
               ret.v1 = {pretyp1}_{op}({in0}.v1);
               return ret;
               """.format(**fmtspec, op=func, pretyp1=pretyp(simd_ext, "f32"))
    return """
           return {pretyp}_{op}({in0}, {in1});
           """.format(**fmtspec, op=func)

# -----------------------------------------------------------------------------
# sqrt
def sqrt1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = wasm_f32x4_sqrt({in0}.v0);
                  ret.v1 = wasm_f32x4_sqrt({in0}.v1);
                  return ret;'''.format(**fmtspec)
    return 'return {pretyp}_sqrt({in0});'.format(**fmtspec)


# -----------------------------------------------------------------------------
# Load logical

def loadl(simd_ext, typ, aligned):
    return \
        '''/* This can surely be improved but it is not our priority. */
           return nsimd_notl_{simd_ext}_{typ}(nsimd_eq_{simd_ext}_{typ}(
                    nsimd_load{align}_{simd_ext}_{typ}(
                      {in0}), nsimd_set1_{simd_ext}_{typ}({zero})));'''. \
            format(align='a' if aligned else 'u',
                   zero='nsimd_f32_to_f16(0.0f)' if typ == 'f16'
                   else '({})0'.format(typ), **fmtspec)


#def loadl(simd_ext, typ, aligned):
#    return \
#        '''/* This can surely be improved but it is not our priority. */
#           return nsimd_to_logical_{simd_ext}_{typ}(nsimd_load{au}_{simd_ext}_{typ}({in0}));'''. \
#            format(au='a' if aligned else 'u', **fmtspec)


# -----------------------------------------------------------------------------
# Store logical

def storel(simd_ext, typ, aligned):
    return \
        '''/* This can surely be improved but it is not our priority. */
           nsimd_store{align}_{simd_ext}_{typ}({in0},
             nsimd_if_else1_{simd_ext}_{typ}({in1},
               nsimd_set1_{simd_ext}_{typ}({one}),
               nsimd_set1_{simd_ext}_{typ}({zero})));'''. \
            format(align='a' if aligned else 'u',
                   one='nsimd_f32_to_f16(1.0f)' if typ == 'f16'
                   else '({})1'.format(typ),
                   zero='nsimd_f32_to_f16(0.0f)' if typ == 'f16'
                   else '({})0'.format(typ), **fmtspec)


# -----------------------------------------------------------------------------
# Absolute value
def abs1(simd_ext, typ):
    if typ == "f16":
        pretyp1 = pretyp(simd_ext, "f32")
        return """
               nsimd_{simd_ext}_vf16 ret;
               ret.v0 = {pretyp1}_abs({in0}.v0);
               ret.v1 = {pretyp1}_abs({in0}.v1);
               return ret;
               """.format(**fmtspec, pretyp1=pretyp1)
    elif typ[0] == 'u':
        return 'return {in0};'.format(**fmtspec)
    return """
           return {pretyp}_abs({in0});
           """.format(**fmtspec)


#
# # -----------------------------------------------------------------------------
# # FMA and FMS

def fma_fms(op, simd_ext, typ):
    if typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vf16 ret;
           ret.v0 = nsimd_{func}_{simd_ext}_f32({in0}.v0, {in1}.v0, {in2}.v0);
           ret.v1 = nsimd_{func}_{simd_ext}_f32({in0}.v1, {in1}.v1, {in2}.v1);
           return ret;'''.format(func=op, **fmtspec)
    else:
        if op == 'fma':
            return \
            'return nsimd_add_{simd_ext}_{typ}(nsimd_mul_{simd_ext}_{typ}({in0}, {in1}), {in2});'.format(**fmtspec)
        elif op == 'fms':
            return \
            'return nsimd_sub_{simd_ext}_{typ}(nsimd_mul_{simd_ext}_{typ}({in0}, {in1}), {in2});'.format(**fmtspec)
        elif op == 'fnma':
            return \
            'return nsimd_sub_{simd_ext}_{typ}({in2}, nsimd_mul_{simd_ext}_{typ}({in0}, {in1}));'.format(**fmtspec)
        elif op == 'fnms':
            return '''return nsimd_sub_{simd_ext}_{typ}(nsimd_neg_{simd_ext}_{typ}({in2}),
                                 nsimd_mul_{simd_ext}_{typ}({in0}, {in1}));'''.format(**fmtspec)


# -----------------------------------------------------------------------------
# Ceil floor trunc and round_to_even
def round1(opts, func, simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_{func}_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_{func}_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(func=func, **fmtspec)
    if func == "round_to_even":
        func = "nearest"
    if typ in ['f32', 'f64']:
        return """
               return {pretyp}_{func}({in0});
               """.format(func=func, **fmtspec)

    return 'return {in0};'.format(**fmtspec)


# -----------------------------------------------------------------------------
# All and any functions

def all_any(func, simd_ext, typ):
    typ1 = "i" + typ[1:] if typ != "f16" else "i32"
    pret = pretyp(simd_ext, typ1)
    op = pret + "_all_true" if func == "all" else "" + pre(simd_ext) + "any_true"
    if typ == 'f16':
        return """
               return {op}({in0}.v0) && {op}({in0}.v1);
               """.format(**fmtspec, op=op)
    return """
           return {op}({in0});
           """.format(**fmtspec, op = op)

# # -----------------------------------------------------------------------------
# # Reinterpret (bitwise_cast)
# # Reinterpretl, i.e. reinterpret on logicals

def  reinterpret1(simd_ext, from_typ, to_typ):
    if to_typ == 'f16' or from_typ == 'f16':
        return "/* TODO */"
    return 'return {in0};'.format(**fmtspec)

# # -----------------------------------------------------------------------------
# # Convert

# # -----------------------------------------------------------------------------
# # Reciprocal (at least 11 bits of precision)

# # -----------------------------------------------------------------------------
# # Reciprocal (IEEE)

# # -----------------------------------------------------------------------------
# # Negative
def neg1(simd_ext, typ):
    
    if typ in ['u8', 'u16', 'u32', 'u64']:
        typ2 = 'i' + typ[1:]
        return 'return {pretyp2}_neg({in0});' . \
               format(**fmtspec, pretyp2=pretyp(simd_ext, typ2))
    return how_it_should_be_op1('neg', simd_ext, typ)

# # -----------------------------------------------------------------------------
# # nbtrue

# # -----------------------------------------------------------------------------
# # reverse

# # -----------------------------------------------------------------------------
# # addv

def addv(simd_ext, typ):
    if typ == 'f64':
        return \
        '''return (f64)(wasm_f64x2_extract_lane({in0}, 0) + wasm_f64x2_extract_lane({in0}, 1));'''. \
                                format(**fmtspec)
    elif typ == 'f32':
        return '''return (f32)(wasm_f32x4_extract_lane({in0}, 0)
                + wasm_f32x4_extract_lane({in0}, 1)
                + wasm_f32x4_extract_lane({in0}, 2)
                + wasm_f32x4_extract_lane({in0}, 3));'''.format(**fmtspec)
    elif typ == 'f16':
        return '''return nsimd_f32_to_f16(
                    nsimd_addv_{simd_ext}_f32({in0}.v0) +
                    nsimd_addv_{simd_ext}_f32({in0}.v1));'''.format(**fmtspec)

# # -----------------------------------------------------------------------------
# # upconvert

# # -----------------------------------------------------------------------------
# # downconvert

# # -----------------------------------------------------------------------------
# # to_mask

# #
# # -----------------------------------------------------------------------------
# # to_logical

# # -----------------------------------------------------------------------------
# # zip functions

# # -----------------------------------------------------------------------------
# # unzip functions

# -----------------------------------------------------------------------------
# mask_for_loop_tail
def mask_for_loop_tail(simd_ext, typ):
    if typ == 'f16':
        threshold = 'nsimd_f32_to_f16((f32)({in1} - {in0}))'.format(**fmtspec)
    else:
        threshold = '({typ})({in1} - {in0})'.format(**fmtspec)
    return '''
              if ({in0} >= {in1}) {{
                return nsimd_set1l_{simd_ext}_{typ}(0);
              }}
              if ({in1} - {in0} < {le}) {{
                nsimd_{simd_ext}_v{typ} n =
                      nsimd_set1_{simd_ext}_{typ}({threshold});
                return nsimd_lt_{simd_ext}_{typ}(
                           nsimd_iota_{simd_ext}_{typ}(), n);
              }} else {{
                return nsimd_set1l_{simd_ext}_{typ}(1);
              }}'''.format(threshold=threshold, **fmtspec)


#
# # -----------------------------------------------------------------------------
# # iota

def iota(simd_ext, typ):
    typ2 = 'f32' if typ == 'f16' else typ
    iota = ', '.join(['({typ2}){i}'.format(typ2=typ2, i=i) \
                      for i in range(int(fmtspec['le']))])
    if typ == 'f16':
        return '''f32 buf[{le}] = {{ {iota} }};
                  nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = wasm_v128_load(buf);
                  ret.v1 = wasm_v128_load(buf + {le2});
                  return ret;'''. \
                  format(iota=iota, le2=fmtspec['le'] // 2, **fmtspec)
    return '''{typ} buf[{le}] = {{ {iota} }};
              return wasm_v128_load(buf);'''. \
              format(iota=iota, **fmtspec)

# # -----------------------------------------------------------------------------
# # scatter

# # -----------------------------------------------------------------------------
# # linear scatter

# # -----------------------------------------------------------------------------
# # mask_scatter

# # -----------------------------------------------------------------------------
# # gather

# # -----------------------------------------------------------------------------
# # linear gather

# # -----------------------------------------------------------------------------
# # maksed gather

def void(simd_ext, typ, func):
    """
    in wasm, you almost cannot know what operation wasn't implemented with abort
    so here is this function when not implemented
    """
    return f"""
    //printf("ATTENTION, VOID HAS BEEN CALLED\\n");
           //printf("func not implemented: {func}\\n");
           """


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
        'pretyp': pretyp(simd_ext, from_typ),
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
        # 'masko_loada1': lambda: maskoz_load(simd_ext, from_typ, 'o', True),
        # 'maskz_loada1': lambda: maskoz_load(simd_ext, from_typ, 'z', True),
        'load2a': lambda: load_deg234(simd_ext, from_typ, True, 2),
        'load3a': lambda: load_deg234(simd_ext, from_typ, True, 3),
        'load4a': lambda: load_deg234(simd_ext, from_typ, True, 4),
        'loadu': lambda: load(simd_ext, from_typ, False),
        # 'masko_loadu1': lambda: maskoz_load(simd_ext, from_typ, 'o', False),
        # 'maskz_loadu1': lambda: maskoz_load(simd_ext, from_typ, 'z', False),
        'load2u': lambda: load_deg234(simd_ext, from_typ, False, 2),
        'load3u': lambda: load_deg234(simd_ext, from_typ, False, 3),
        'load4u': lambda: load_deg234(simd_ext, from_typ, False, 4),
        'storea': lambda: store(simd_ext, from_typ, True),
        # 'mask_storea1': lambda: mask_store(simd_ext, from_typ, True),
        'store2a': lambda: store_deg234(simd_ext, from_typ, True, 2),
        'store3a': lambda: store_deg234(simd_ext, from_typ, True, 3),
        'store4a': lambda: store_deg234(simd_ext, from_typ, True, 4),
        'storeu': lambda: store(simd_ext, from_typ, False),
        # 'mask_storeu1': lambda: mask_store(simd_ext, from_typ, False),
        'store2u': lambda: store_deg234(simd_ext, from_typ, False, 2),
        'store3u': lambda: store_deg234(simd_ext, from_typ, False, 3),
        'store4u': lambda: store_deg234(simd_ext, from_typ, False, 4),
        # 'gather': lambda: gather(simd_ext, from_typ),
        # 'gather_linear': lambda: gather_linear(simd_ext, from_typ),
        # 'masko_gather': lambda: maskoz_gather('o', simd_ext, from_typ),
        # 'maskz_gather': lambda: maskoz_gather('z', simd_ext, from_typ),
        # 'scatter': lambda: scatter(simd_ext, from_typ),
        # 'scatter_linear': lambda: scatter_linear(simd_ext, from_typ),
        # 'mask_scatter': lambda: mask_scatter(simd_ext, from_typ),
        'andb': lambda: comp_bin('and', simd_ext, from_typ),
        'xorb': lambda: comp_bin('xor', simd_ext, from_typ),
        'orb': lambda: comp_bin('or', simd_ext, from_typ),
        'andl': lambda: binlop2('andl', simd_ext, from_typ),
        'xorl': lambda: binlop2('xorl', simd_ext, from_typ),
        'orl': lambda: binlop2('orl', simd_ext, from_typ),
        'notb': lambda: comp_bin("not", simd_ext, from_typ),
        'notl': lambda: lnot1(simd_ext, from_typ),
        'andnotb': lambda: comp_bin("andnot", simd_ext, from_typ),
        'andnotl': lambda: comp("andnot", simd_ext, from_typ),
        'add': lambda: addsub('add', simd_ext, from_typ),
        'sub': lambda: addsub('sub', simd_ext, from_typ),
        'adds': lambda: addsub('adds', simd_ext, from_typ),
        'subs': lambda: addsub('subs', simd_ext, from_typ),
        'div': lambda: div2(opts, simd_ext, from_typ),
        'sqrt': lambda: sqrt1(simd_ext, from_typ),
        'len': lambda: len1(simd_ext, from_typ),
        #'mul': lambda: mul2(simd_ext, from_typ),
        'shl': lambda: shl_shr('shl', simd_ext, from_typ),
        'shr': lambda: shl_shr('shr', simd_ext, from_typ),
        # 'shra': lambda: shra(opts, simd_ext, from_typ),
        'set1': lambda: set1(simd_ext, from_typ),
        'set1l': lambda: set1l(simd_ext, from_typ),
        #'eq': lambda: comp("eq", simd_ext, from_typ),
        #'ne': lambda: comp("ne", simd_ext, from_typ),
        #'gt': lambda: comp("gt", simd_ext, from_typ),
        #'lt': lambda: comp("lt", simd_ext, from_typ),
        #'ge': lambda: comp("ge", simd_ext, from_typ),
        #'le': lambda: comp("le", simd_ext, from_typ),
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
        'trunc': lambda: round1(opts, 'trunc', simd_ext, from_typ),
        'round_to_even': lambda: round1(opts, 'round_to_even', simd_ext, from_typ),
        'all': lambda: all_any('all', simd_ext, from_typ),
        'any': lambda: all_any('any', simd_ext, from_typ),
        'reinterpret': lambda: reinterpret1(simd_ext, from_typ, to_typ),
        'reinterpretl': lambda: reinterpret1(simd_ext, from_typ, to_typ),
        # 'cvt': lambda: convert1(simd_ext, from_typ, to_typ),
        # 'rec11': lambda: rec11_rsqrt11('rcp', simd_ext, from_typ),
        # 'rec8': lambda: rec11_rsqrt11('rcp', simd_ext, from_typ),
        # 'rsqrt11': lambda: rec11_rsqrt11('rsqrt', simd_ext, from_typ),
        # 'rsqrt8': lambda: rec11_rsqrt11('rsqrt', simd_ext, from_typ),
        # 'rec': lambda: rec1(simd_ext, from_typ),
        'neg': lambda: neg1(simd_ext, from_typ),
        # 'nbtrue': lambda: nbtrue1(simd_ext, from_typ),
        # 'reverse': lambda: reverse1(simd_ext, from_typ),
        'addv': lambda: addv(simd_ext, from_typ),
        # 'upcvt': lambda: upcvt1(simd_ext, from_typ, to_typ),
        # 'downcvt': lambda: downcvt1(opts, simd_ext, from_typ, to_typ),
        # 'to_mask': lambda: to_mask1(simd_ext, from_typ),
        #'to_logical': lambda: to_logical1(simd_ext, from_typ),
        # 'ziplo': lambda: zip_half('ziplo', simd_ext, from_typ),
        # 'ziphi': lambda: zip_half('ziphi', simd_ext, from_typ),
        # 'unziplo': lambda: unzip_half(opts, 'unziplo', simd_ext, from_typ),
        # 'unziphi': lambda: unzip_half(opts, 'unziphi', simd_ext, from_typ),
        # 'zip' : lambda : zip(simd_ext, from_typ),
        # 'unzip' : lambda : unzip(simd_ext, from_typ),
        'mask_for_loop_tail': lambda: mask_for_loop_tail(simd_ext, from_typ),
        'iota': lambda : iota(simd_ext, from_typ)
    }
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if not from_typ in common.types:
        raise ValueError('Unknown type "{}"'.format(from_typ))
    if not func in impls:
        return void(simd_ext, from_typ, func)
        return common.NOT_IMPLEMENTED # while implementing wams, abort() is the worst case
    else:
        return impls[func]()
