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

DEBUG = True


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
    ret = ''
    if simd_ext == 'wasm_simd128':
        ret += '''#include <nsimd/cpu/cpu/{}.h>
                  '''.format(func)
    if DEBUG == True:
        ret += """#include <stdio.h>
        """

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
        ret += '''#include <nsimd/wasm/{simd_ext}/eq.h>
                      #include <nsimd/wasm/{simd_ext}/set1.h>
                      #include <nsimd/wasm/{simd_ext}/{load}.h>
                      #include <nsimd/wasm/{simd_ext}/notl.h>
                      '''.format(load='load' + func[5], **fmtspec)

    elif func in ['storelu']:
        ret += '''#include <nsimd/wasm/{simd_ext}/if_else1.h>
                     #include <nsimd/wasm/{simd_ext}/set1.h>
                     '''.format(**fmtspec)

    return ret


def printf2(func):
    """
    debugging purposes
    juste decorate the function with it and when executed on test, it will print the environnements
    """
    import inspect

    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = f"{func.__name__} called on {fmtspec['typ']}\\n" + ", ".join(
            "{} = {!r}".format(*item) for item in func_args.items())
        ret = func(*args)
        if not DEBUG:
            return ret

        elif func.__name__ == "store1234":
            ret += """
                   int k;
                   printf("element to store:");
                   for(k=0;k<{nbits};k++)printf(" %lx", {in1}[k]);
                   printf("\\n");
                   """.format(**fmtspec, nbits=get_nbits(fmtspec["typ"]))
        return f"""
               printf("\\n---------------\\n");
               printf("{func.__module__}.{func.__qualname__} ( {func_args_str} )\\n");
               """ + ret

    return wrapper


# -----------------------------------------------------------------------------
# Function prefixes

def pre(simd_ext):
    return 'wasm_v128_'


def pretyp(simd_ext, typ):
    return 'wasm_{}x{}'.format(typ, 128 // int(typ[1:]))


def nbits(simd_ext):
    return '128'


def get_nbits(typ):
    # equivalent to fmtspec[len]
    return int(nbits(fmtspec["simd_ext"])) // int(typ[1:])


# -----------------------------------------------------------------------------
# Other helper functions

fmtspec = {}


def set_lane(simd_ext, typ, var_name, scalar, i):
    # No code for f16's
    if typ == 'f16':
        return ''
    if typ in common.itypes + common.ftypes:
        return f'{var_name} = {pretyp(simd_ext, typ)}_replace_lane({var_name}, {i}, {scalar});'
    else:
        return \
            '{} = {}_replace_lane({}, {}, nsimd_scalar_reinterpret_i{}_{}({}))'. \
                format(var_name, fmtspec['pre'], var_name, i, fmtspec['typnbits'], typ,
                       scalar)


def get_lane(simd_ext, typ, var_name, i):
    # No code for f16's
    if typ == 'f16':
        return ''
    if typ in common.itypes + ['u8', 'u16'] + common.ftypes:
        return f'{pretyp(simd_ext, typ)}_extract_lane({var_name}, {i})'
    return 'nsimd_scalar_reinterpret_u{}_{}({}_replace_lane({}, {}))'. \
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
@printf2
def load(simd_ext, typ, aligned):
    if typ == 'f16':
        return """
               nsimd_{simd_ext}_vf16 ret;

               ret.v0 = wasm_v128_load(a0);
               ret.v1 = wasm_v128_load(a0 + {le}/2);

               int i;
               {typ} buf[{le}/2];
               printf("----------------------\\n");
               printf("value in load\\n");

               wasm_v128_store((void *)buf, ret.v0);
               printf("value of ret.v0");
               for(i=0; i<{le}/2; ++i)printf(" %#010x", buf[i]);
               printf("\\nvalue of ret.v1");
               wasm_v128_store((void *)buf, ret.v1);
               for(i=0; i<{le}/2; ++i)printf(" %#010x", buf[i]);
               printf("\\n");


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
            
                          int i;
               {typ} buf1[{le}/2];
               printf("----------------------\\n");
               printf("value in load\\n");
               
               wasm_v128_store((void *)buf1, ret.v0);
               printf("value of ret.v0");
               for(i=0; i<{le}/2; ++i)printf(" %#010x", buf1[i]);
               printf("\\nvalue of ret.v1");
               wasm_v128_store((void *)buf1, ret.v1);
               for(i=0; i<{le}/2; ++i)printf(" %#010x", buf1[i]);
               printf("\\n");
               
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

# def maskoz_load(simd_ext, typ, oz, aligned):
#     if typ == 'f16':
#         def helper(dst, i_dst, i_src):
#             mask = get_lane(simd_ext, typ, common.in0, i_src)
#             set_zero = set_lane(simd_ext, typ, dst, '0.0', i_dst)
#             set_other = set_lane(simd_ext, typ, dst, common.in2, i_dst)
#             set_value = set_lane(simd_ext, typ, dst,
#                                  'nsimd_f32_to_u16({}[{}])'. \
#                                  format(common.in1, i_src), i_dst)
#             return '''if ({}) {{
#                         {}
#                       }} else {{
#                         {}
#                       }}'''.format(mask, set_value,
#                                    set_zero if oz == 'z' else set_other)
#
#         return \
#             '''nsimd_{simd_ext}_vf16 ret = ret;
#                {fill_ret_v0}
#                {fill_ret_v1}
#                return ret;'''.format(
#                 fill_ret_v0='\n'.join([helper('ret.v0', i, i) for i in range(4)]),
#                 fill_ret_v1='\n'.join([helper('ret.v1', i, i + 4) for i in range(4)]),
#                 **fmtspec)
#     else:
#         def helper(i):
#             mask = get_lane(simd_ext, typ, common.in0, i)
#             set_zero = set_lane(simd_ext, typ, 'ret', '({})0'.format(typ), i)
#             set_other = set_lane(simd_ext, typ, 'ret', common.in2, i)
#             set_value = set_lane(simd_ext, typ, 'ret',
#                                  '{}[{}]'.format(common.in1, i), i)
#             return '''if ({}) {{
#                         {}
#                       }} else {{
#                         {}
#                       }}'''.format(mask, set_value,
#                                    set_zero if oz == 'z' else set_other)
#
#         return '''nsimd_{simd_ext}_v{typ} ret = ret;
#                   {fill_ret}
#                   return ret;'''. \
#             format(fill_ret='\n'.join([helper(i) for i in range(4)]),
#                    **fmtspec)


# -----------------------------------------------------------------------------
# Loads of degree 2, 3 and 4
@printf2
def load_deg234(simd_ext, typ, align, deg):
    if simd_ext != "wasm_simd128":
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
                  
                  int i;
                  {typ} buf[{nbits2}];
                  printf("----------------------\\n");
                  printf("value in load2\\n");
                  printf("value of in0 ");
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", a0[i]);
                  printf("\\n");
                  printf("value of in1 ");
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", a0[i + {le}]);
                  printf("\\n");


                  wasm_v128_store((void *)buf, ret.v0);
                  printf("value of ret.v0");
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\nvalue of ret.v1");
                   wasm_v128_store((void *)buf, ret.v1);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  
                  return ret;'''.format(seq_even=seq_even, seq_odd=seq_odd, nbits2=get_nbits(typ),
                                        **fmtspec)
    if deg == 3:
        shuffle = ""
        load_block = """
                     v128_t a = wasm_v128_load((void *){in0});
                     v128_t b = wasm_v128_load((void *)({in0} + {nbits2}));
                     v128_t c = wasm_v128_load((void *)({in0} + {nbits2}*2));
                     """.format(**fmtspec, nbits2=get_nbits(typ))

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
               
               
               int i;
                  {typ} buf[{nbits2}];
                  printf("----------------------\\n");
                  printf("value in load2\\n");
                  printf("value of in0 ");
                  for(i=0; i<{nbits2}*3; ++i)printf(" %lu", a0[i]);
                  printf("\\n");

               
               
                   printf("value of ret ");
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", ret.v0[i]);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", ret.v1[i]);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", ret.v2[i]);
                  printf("\\n");
               
               return ret;
               """.format(**fmtspec, shuffle=shuffle, load=load_block, nbits2=get_nbits(typ))

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
                      
                      
                      
                  int i;
                  {typ} buf[{nbits2}];
                  printf("----------------------\\n");
                  printf("value in load4\\n");

                  wasm_v128_store((void *)buf, ret.v0);
                  printf("value of ret.v0");
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\nvalue of ret.v1");
                   wasm_v128_store((void *)buf, ret.v1);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  printf("\\nvalue of ret.v2");
                   wasm_v128_store((void *)buf, ret.v2);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  printf("\\nvalue of ret.v3");
                   wasm_v128_store((void *)buf, ret.v3);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                      
                      
                      return ret;'''.format(seq_even=seq_even, seq_odd=seq_odd, nbits2=get_nbits(typ),
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
                   
                   
                   int i;
                  {typ} buf[{nbits2}];
                  printf("----------------------\\n");
                  printf("value in load4\\n");

                  wasm_v128_store((void *)buf, ret.v0);
                  printf("value of ret.v0");
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\nvalue of ret.v1");
                   wasm_v128_store((void *)buf, ret.v1);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  printf("value of ret.v2");
                   wasm_v128_store((void *)buf, ret.v2);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  printf("value of ret.v3");
                   wasm_v128_store((void *)buf, ret.v3);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                   
                   
                   
                   return ret;'''.format(seq_ab=seq_ab, seq_cd=seq_cd, lex2=lex2, nbits2=get_nbits(typ),
                                         lex3=lex3, **fmtspec)


# -----------------------------------------------------------------------------
# Stores of degree 2, 3 and 4
@printf2
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
        for i in range(get_nbits(typ)):
            list_unpack.append(i)
            list_unpack.append(get_nbits(typ) + i)

        middle = get_nbits(typ)
        seq_unpacklo = ', '.join(str(i) for i in list_unpack[:middle])
        seq_unpackhi = ', '.join(str(i) for i in list_unpack[middle:])

        return """               
               nsimd_{simd_ext}_v{typ} buf0 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, {seq_unpacklo});
               wasm_v128_store({in0}, buf0);
               
               
               nsimd_{simd_ext}_v{typ} buf1 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, {seq_unpackhi});
               wasm_v128_store({in0} + {nbits2}, buf1);
               
               
                  int i;
                  {typ} buf[{nbits2}];
                  printf("---------------\\n");
                  printf("value to store\\n");
                  
                  printf("value of in1: ");
                  wasm_v128_store((void *)buf, a1);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\nvalue of in2: ");
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  
                  wasm_v128_store((void *)buf, buf0);
                  
                  printf("value of buf0: ");
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\nvalue of buf1: ");
                  wasm_v128_store((void *)buf, buf1);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  printf("value of ret");
                  for(i=0; i<{nbits2}*2; ++i)printf(" %lu", a0[i]);
                  printf("\\n");
            
               
               """.format(**fmtspec, seq_unpacklo=seq_unpacklo, seq_unpackhi=seq_unpackhi,
                          suf_align="a" if align else "u",
                          nbits2=get_nbits(typ), native_typ=get_native_typ(simd_ext, typ),
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
                   
                   
                   int i;
                  {typ} buf[{nbits2}];
                  printf("---------------\\n");
                  printf("value to store\\n");
                  
                  printf("value of in1: ");
                  wasm_v128_store((void *)buf, a1);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\nvalue of in2: ");
                  wasm_v128_store((void *)buf, a2);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  printf("\\nvalue of in3: ");
                   wasm_v128_store((void *)buf, a3);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  
                  
                  printf("value of ret");
                  for(i=0; i<{nbits2}*3; ++i)printf(" %lu", a0[i]);
                  printf("\\n");
                   
                   
                   """.format(**fmtspec, nbits2=get_nbits(typ))
        if typ[1:] == "16":
            return """
                   nsimd_{simd_ext}_v{typ} rg123_ = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 0,8,1,9,2,10,10,10);
                   nsimd_{simd_ext}_v{typ} rg456_ = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 3, 11, 4, 12, 5, 5,5,5);
                   nsimd_{simd_ext}_v{typ} rg678_ = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 13, 13, 6, 14, 7, 15,15,15);
                   
                   wasm_v128_store({in0}, wasm_v{typnbits}x{le}_shuffle(rg123_ , {in3}, 0,1,8,2,3,9,4,5));
                   wasm_v128_store({in0} + {nbits2}, wasm_v{typnbits}x{le}_shuffle(rg456_ , {in3}, 10, 0,1, 11, 2, 3, 12, 4));
                   wasm_v128_store({in0} + 2*{nbits2}, wasm_v{typnbits}x{le}_shuffle(rg678_ , {in3}, 1, 13, 2,3, 14, 4, 5, 15));
                   """.format(**fmtspec, nbits2=get_nbits(typ))
        elif typ[1:] == "32":
            return """
                   nsimd_{simd_ext}_v{typ} rg1r_2 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 0, 4, 1, 1);
                   nsimd_{simd_ext}_v{typ} bb23r3_ = wasm_v{typnbits}x{le}_shuffle({in3}, {in1}, 1, 2, 6, 3);
                    
                   wasm_v128_store({in0}, wasm_v{typnbits}x{le}_shuffle(rg1r_2, {in3}, 0, 1, 4, 2));
                   wasm_v128_store({in0} + {nbits2}, wasm_v{typnbits}x{le}_shuffle(bb23r3_, {in2}, 5, 0, 2, 6));
                   
                   nsimd_{simd_ext}_v{typ} r4g4__ = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 3, 7, 7, 7);
                   wasm_v128_store({in0} + 2*{nbits2}, wasm_v{typnbits}x{le}_shuffle(r4g4__, {in3}, 6, 0, 1, 7));
                   """.format(**fmtspec, nbits2=get_nbits(typ))
        elif typ[1:] == "64":
            return """
                   nsimd_{simd_ext}_v{typ} buf0 = wasm_v{typnbits}x{le}_shuffle({in1}, {in2}, 0, 2);
                   wasm_v128_store({in0}, buf0);
                   

                   nsimd_{simd_ext}_v{typ} buf1 = wasm_v{typnbits}x{le}_shuffle({in3}, {in1}, 0, 3);
                   wasm_v128_store({in0} + {nbits2}, buf1);
                   
                   nsimd_{simd_ext}_v{typ} buf2 = wasm_v{typnbits}x{le}_shuffle({in2}, {in3}, 1, 3);
                   wasm_v128_store({in0} + 2*{nbits2}, buf2);
                   
                   """.format(**fmtspec, nbits2=get_nbits(typ))

    if deg == 4:

        le = get_nbits(typ)

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

        for i in range(get_nbits(typ)):
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
               
               
                                  int i;
                  {typ} buf[{nbits2}];
                  printf("----------------------\\n");
                  printf("value in store4\\n");
                  
                  
                  printf("value of in1: ");
                  wasm_v128_store((void *)buf, a1);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\nvalue of in2: ");
                  wasm_v128_store((void *)buf, a2);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  printf("value of in3: ");
                   wasm_v128_store((void *)buf, a3);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  
                  printf("value of in4: ");
                   wasm_v128_store((void *)buf, a4);
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n\\n");
                  
                  
                  

                  wasm_v128_store((void *)buf, wasm_v128_load(a0));
                  printf("value of ret.v0");
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\nvalue of ret.v1");
                   wasm_v128_store((void *)buf, wasm_v128_load(a0 + {nbits2}));
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  printf("value of ret.v2");
                   wasm_v128_store((void *)buf, wasm_v128_load(a0 + {nbits2} * 2));
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
                  printf("value of ret.v3");
                   wasm_v128_store((void *)buf, wasm_v128_load(a0 + {nbits2} * 3));
                  for(i=0; i<{nbits2}; ++i)printf(" %lu", buf[i]);
                  printf("\\n");
               
               
               """.format(**fmtspec, nbits2=le,
                          seq_unpacklo=seq_unpacklo, seq_unpackhi=seq_unpackhi,
                          seq_unpack_rethi=seq_unpack_rethi, seq_unpack_retlo=seq_unpack_retlo)

    return common.NOT_IMPLEMENTED


# -----------------------------------------------------------------------------
# Store

@printf2
def store(simd_ext, typ, aligned):
    align = '' if aligned else 'u'
    if typ == 'f16': #todo still experimental
        return """
               /*{pre}store{le}_lane({in0}, {in1}.v0, 0);
               {pre}store{le}_lane({in0} + {le} /2, {in1}.v1, 0);*/
               wasm_v128_store({in0}, {in1}.v0);
               wasm_v128_store({in0} + {le}/2, {in1}.v1);
               
               int i;
               {typ} buf[{le}];
               printf("----------------------\\n");
               printf("value in store\\n");
               for(i=0; i<8; ++i)printf(" %#010x", a0[i]);
               printf("\\n");
               
               
               """.format(**fmtspec)
    return """
           wasm_v128_store({in0}, {in1});
           
           int i;
           {typ} buf[{le}];
           printf("----------------------\\n");
           printf("value in store\\n");
           printf("value stored");
           for(i=0; i<{le}; ++i)printf(" %lu", a0[i]);
           printf("\\n");
           
           """.format(**fmtspec)


#
# # masked store
#
# def mask_store(simd_ext, typ, aligned):
#     if typ == 'f16':
#         le2 = fmtspec['le'] // 2
#         if simd_ext in sse + avx:
#             store_mask = '''{pre}storeu_ps(mask, {in0}.v0);
#                             {pre}storeu_ps(mask + {le2}, {in0}.v1);'''. \
#                             format(le2=le2, **fmtspec)
#         else:
#             store_mask = '''_mm512_storeu_ps(mask, _mm512_maskz_mov_ps(
#                               {in0}.v0, _mm512_set1_ps(1.0f)));
#                             _mm512_storeu_ps(mask + {le2}, _mm512_maskz_mov_ps(
#                               {in0}.v1, _mm512_set1_ps(1.0f)));'''. \
#                             format(le2=le2, **fmtspec)
#         return '''f32 mask[{le}], buf[{le}];
#                   int i;
#                   {store_mask}
#                   {pre}storeu_ps(buf, {in2}.v0);
#                   {pre}storeu_ps(buf + {le2}, {in2}.v1);
#                   for (i = 0; i < {le}; i++) {{
#                     if (nsimd_scalar_reinterpret_u32_f32(mask[i]) != (u32)0) {{
#                       {in1}[i] = nsimd_f32_to_f16(buf[i]);
#                     }}
#                   }}'''.format(store_mask=store_mask, le2=le2, **fmtspec)
#     suf2 = 'ps' if typ[1:] == '32' else 'pd'
#     if simd_ext in sse:
#         if typ in common.iutypes:
#             return '_mm_maskmoveu_si128({in2}, {in0}, (char *){in1});'. \
#                    format(**fmtspec)
#         else:
#             return '''_mm_maskmoveu_si128(_mm_cast{suf2}_si128({in2}),
#                                           _mm_cast{suf2}_si128({in0}),
#                                           (char *){in1});'''. \
#                                           format(suf2=suf2, **fmtspec)
#     if typ in ['i8', 'u8', 'i16', 'u16'] and simd_ext != 'avx512_skylake':
#         if simd_ext == 'avx512_knl':
#             return \
#             '''int i;
#                u64 mask;
#                {typ} buf[{le}];
#                {pre}storeu{sufsi}((__m512i *)buf, {in2});
#                mask = (u64){in0};
#                for (i = 0; i < {le}; i++) {{
#                  if ((mask >> i) & 1) {{
#                    {in1}[i] = buf[i];
#                  }}
#                }}'''.format(utyp='u' + typ[1:], **fmtspec)
#         else:
#             return \
#             '''nsimd_{op_name}_sse42_{typ}({mask_lo}, {in1}, {val_lo});
#                nsimd_{op_name}_sse42_{typ}({mask_hi}, {in1} + {le2},
#                                            {val_hi});
#                '''.format(le2=fmtspec['le'] // 2,
#                op_name='mask_store{}1'.format('a' if  aligned else 'u'),
#                mask_lo=extract(simd_ext, typ, LO, common.in0),
#                mask_hi=extract(simd_ext, typ, HI, common.in0),
#                val_lo=extract(simd_ext, typ, LO, common.in2),
#                val_hi=extract(simd_ext, typ, HI, common.in2), **fmtspec)
#     # Here typ is 32 of 64-bits wide except
#     if simd_ext in avx:
#         if typ in common.ftypes:
#             return '''{pre}maskstore{suf}({in1},
#                           {pre}cast{suf2}_si256({in0}), {in2});'''. \
#                           format(suf2=suf2, **fmtspec)
#         else:
#             if simd_ext == 'avx2':
#                 return '{pre}maskstore{suf}({cast}{in1}, {in0}, {in2});'. \
#                        format(cast='(nsimd_longlong *)' \
#                               if typ in ['i64', 'u64'] \
#                               else '(int *)', **fmtspec)
#             else:
#                 return '''{pre}maskstore_{suf2}(({ftyp}*){in1}, {in0},
#                             {pre}castsi256_{suf2}({in2}));'''. \
#                             format(suf2=suf2, ftyp='f' + typ[1:], **fmtspec)
#     # getting here means avx512 with intrinsics
#     code = '{pre}mask_store{{}}{suf}((void*){in1}, {in0}, {in2});'. \
#            format(**fmtspec)
#     if typ in ['i32', 'u32', 'f32', 'i64', 'u64', 'f64']:
#         return code.format('' if aligned else 'u')
#     else:
#         return code.format('u')
#
# # -----------------------------------------------------------------------------
# # Code for binary operators: and, or, xor

# # -----------------------------------------------------------------------------
# # Code for comparisons

@printf2
def comp(op, simd_ext, typ):
    if typ == "f16":
        return ""
    if typ[0] == "u":
        typ = "i" + typ[1:]
    if op == "andnot":
        func = "{pre}andnot".format(**fmtspec)
    else:
        func = "{pretyp1}_{op}".format(pretyp1=pretyp(simd_ext,typ), op=op)


    return """
           int i;
    
           {typ} buf[{le}];
           printf("value of in0 ");
           for(i=0; i<{le}; ++i)printf(" %u", a0[i]);
           printf("\\n");
           printf("value of in1 ");
           for(i=0; i<{le}; ++i)printf(" %u", a1[i]);
           printf("\\n");
           
           nsimd_{simd_ext}_v{typ} ret = {func}({in0}, {in1});
           wasm_v128_store((void *)buf, ret);
           printf("value of ret ");
           for(i=0; i<{le}; ++i)printf(" %lu", buf[i]);
           printf("\\n");
           
           return {func}({in0}, {in1});
           """.format(**fmtspec,
                      op=op,
                      func=func
                      )

@printf2
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
           int i;
    
           {typ} buf[{le}];
           printf("value of in0 ");
           for(i=0; i<{le}; ++i)printf(" %u", a0[i]);
           printf("\\n");
           printf("value of in1 ");
           for(i=0; i<{le}; ++i)printf(" %u", a1[i]);
           printf("\\n");
           
           
           return {in0} {op} {in1};
           """.format(**fmtspec,
                      op=op_map_2[op],
                      )


#
# def binop2(func, simd_ext, typ, logical=False):
#     logical = 'l' if logical else ''
#     func = func[0:-1]
#     if typ == 'f16':
#         return \
#         '''nsimd_{simd_ext}_v{logi}f16 ret;
#            ret.v0 = nsimd_{func}{logi2}_{simd_ext}_f32({in0}.v0, {in1}.v0);
#            ret.v1 = nsimd_{func}{logi2}_{simd_ext}_f32({in0}.v1, {in1}.v1);
#            return ret;'''.format(logi='l' if logical else '', func=func,
#                                  logi2='l' if logical else 'b', **fmtspec)
#     normal = 'return {pre}{func}{sufsi}({in0}, {in1});'. \
#              format(func=func, **fmtspec)
#     if simd_ext in sse:
#         return normal
#     if simd_ext in avx:
#         if simd_ext == 'avx2' or typ in ['f32', 'f64']:
#             return normal
#         else:
#             return '''return _mm256_castpd_si256(_mm256_{func}_pd(
#                                _mm256_castsi256_pd({in0}),
#                                  _mm256_castsi256_pd({in1})));'''. \
#                                  format(func=func, **fmtspec)
#     if simd_ext in avx512:
#         if simd_ext == 'avx512_skylake' or typ in common.iutypes:
#             return normal
#         else:
#             return \
#             '''return _mm512_castsi512{suf}(_mm512_{func}_si512(
#                         _mm512_cast{typ2}_si512({in0}),
#                           _mm512_cast{typ2}_si512({in1})));'''. \
#                           format(func=func, typ2=suf_ep(typ)[1:], **fmtspec)
#
# -----------------------------------------------------------------------------
# Code for logical binary operators: andl, orl, xorl

# def binlop2(func, simd_ext, typ):
#     op = { 'orl': '|', 'xorl': '^', 'andl': '&' }
#     op_fct = { 'orl': 'kor', 'xorl': 'kxor', 'andl': 'kand' }
#     if simd_ext not in avx512:
#         if typ == 'f16':
#             return binop2(func, simd_ext, typ, True)
#         else:
#             return binop2(func, simd_ext, typ)
#     elif simd_ext == 'avx512_knl':
#         if typ == 'f16':
#             return '''nsimd_{simd_ext}_vlf16 ret;
#                       ret.v0 = _{op_fct}_mask16({in0}.v0, {in1}.v0);
#                       ret.v1 = _{op_fct}_mask16({in0}.v1, {in1}.v1);
#                       return ret;'''. \
#                       format(op_fct=op_fct[func], **fmtspec)
#         elif typ in ['f32', 'u32', 'i32']:
#             return 'return _{op_fct}_mask16({in0}, {in1});'. \
#                    format(op_fct=op_fct[func], **fmtspec)
#         else:
#             return 'return (__mmask{le})({in0} {op} {in1});'. \
#                    format(op=op[func], **fmtspec)
#     elif simd_ext == 'avx512_skylake':
#         if typ == 'f16':
#             return '''nsimd_{simd_ext}_vlf16 ret;
#                       #if defined(NSIMD_IS_GCC) || defined(NSIMD_IS_CLANG)
#                         ret.v0 = (__mmask16)({in0}.v0 {op} {in1}.v0);
#                         ret.v1 = (__mmask16)({in0}.v1 {op} {in1}.v1);
#                       #else
#                         ret.v0 = _{op_fct}_mask16({in0}.v0, {in1}.v0);
#                         ret.v1 = _{op_fct}_mask16({in0}.v1, {in1}.v1);
#                       #endif
#                       return ret;'''. \
#                       format(op_fct=op_fct[func], op=op[func], **fmtspec)
#         else:
#             return '''#if defined(NSIMD_IS_GCC) || defined(NSIMD_IS_CLANG)
#                         return (__mmask{le})({in0} {op} {in1});
#                       #else
#                         return _{op_fct}_mask{le}({in0}, {in1});
#                       #endif'''.format(op_fct=op_fct[func], op=op[func],
#                                        **fmtspec)
#
# # -----------------------------------------------------------------------------
# # andnot
#
# def andnot2(simd_ext, typ, logical=False):
#     if typ == 'f16':
#         return \
#         '''nsimd_{simd_ext}_v{logi}f16 ret;
#            ret.v0 = nsimd_andnot{logi2}_{simd_ext}_f32({in0}.v0, {in1}.v0);
#            ret.v1 = nsimd_andnot{logi2}_{simd_ext}_f32({in0}.v1, {in1}.v1);
#            return ret;'''.format(logi='l' if logical else '',
#                                  logi2='l' if logical else 'b', **fmtspec)
#     if simd_ext in sse:
#         return 'return _mm_andnot{sufsi}({in1}, {in0});'.format(**fmtspec)
#     if simd_ext in avx:
#         if simd_ext == 'avx2' or typ in ['f32', 'f64']:
#             return 'return _mm256_andnot{sufsi}({in1}, {in0});'. \
#                    format(**fmtspec)
#         else:
#             return '''return _mm256_castpd_si256(_mm256_andnot_pd(
#                                _mm256_castsi256_pd({in1}),
#                                _mm256_castsi256_pd({in0})));'''. \
#                                format(**fmtspec)
#     if simd_ext in avx512:
#         if simd_ext == 'avx512_skylake' or typ in common.iutypes:
#             return 'return _mm512_andnot{sufsi}({in1}, {in0});'. \
#                    format(**fmtspec)
#         else:
#             return '''return _mm512_castsi512{suf}(_mm512_andnot_si512(
#                                _mm512_cast{suf2}_si512({in1}),
#                                _mm512_cast{suf2}_si512({in0})));'''. \
#                                format(suf2=fmtspec['suf'][1:], **fmtspec)
#
# # -----------------------------------------------------------------------------
# # logical andnot
#
# def landnot2(simd_ext, typ):
#     if simd_ext in avx512:
#         if typ == 'f16':
#             return '''nsimd_{simd_ext}_vlf16 ret;
#                       ret.v0 = (__mmask16)({in0}.v0 & (~{in1}.v0));
#                       ret.v1 = (__mmask16)({in0}.v1 & (~{in1}.v1));
#                       return ret;'''.format(**fmtspec)
#         else:
#             return 'return (__mmask{le})({in0} & (~{in1}));'.format(**fmtspec)
#     return andnot2(simd_ext, typ, True)
#
# # -----------------------------------------------------------------------------
# # Code for unary not
#
# def not1(simd_ext, typ, logical=False):
#     if typ == 'f16':
#         return \
#         '''nsimd_{simd_ext}_v{logi}f16 ret;
#            nsimd_{simd_ext}_vf32 cte = {pre}castsi{nbits}_ps(
#                                          {pre}set1_epi8(-1));
#            ret.v0 = nsimd_andnot{logi2}_{simd_ext}_f32(cte, {in0}.v0);
#            ret.v1 = nsimd_andnot{logi2}_{simd_ext}_f32(cte, {in0}.v1);
#            return ret;'''.format(logi='l' if logical else '',
#                                  logi2='l' if logical else 'b', **fmtspec)
#     elif typ in ['f32', 'f64']:
#         return '''return nsimd_andnotb_{simd_ext}_{typ}(
#                            {pre}castsi{nbits}{suf}(
#                              {pre}set1_epi8(-1)), {in0});'''.format(**fmtspec)
#     else:
#         return '''return nsimd_andnotb_{simd_ext}_{typ}(
#                            {pre}set1_epi8(-1), {in0});'''.format(**fmtspec)
#
# -----------------------------------------------------------------------------
# Code for unary logical lnot

@printf2
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
    
               {typ} buf[{le}];
               printf("value of in0 ");
               for(i=0; i<{le}; ++i)printf(" %#lx", a0[i]);
               printf("\\n");
               wasm_v128_store((void *)buf, {pre}not({in0}));
               printf("value of ret ");
               for(i=0; i<{le}; ++i)printf(" %#lx", buf[i]);
               printf("\\n");
               
               nsimd_{simd_ext}_v{typ} ret = ~a0;
               for(i=0; i<{le}; ++i)
                  ret[i] = ~{in0}[i];
               
               ret[1] = (u64) -1;
               return ret; /* ~{in0} */
               '''.format(**fmtspec)

# -----------------------------------------------------------------------------
# Addition and substraction

@printf2
def addsub(op, simd_ext, typ):
    """
    :param op: can be either add adds sub subs
    """
    pre_sat = ''
    if op[-1] == 's':
        if typ[1:] == "16" and typ[0] != 'f':
            pre_sat = '_sat'
        op = op[:-1]
    fmtspec2 = fmtspec.copy()
    if typ[0] == "u":
        fmtspec2["pretyp"] = pretyp(simd_ext, "i" + typ[1:])
    if typ == "f16":
        fmtspec2["pretyp"] = pretyp(simd_ext, "f32")
        return """
               nsimd_{simd_ext}_vf16 ret;
               ret.v0 = {pretyp}_{op}{pre_sat}({in0}.v0,{in1}.v0);
               ret.v1 = {pretyp}_{op}{pre_sat}({in0}.v1,{in1}.v1);
               
               int i;
               {typ} buf1[{le}/2];
               printf("----------------------\\n");
               printf("value of ret\\n");
               
               wasm_v128_store((void *)buf1, a0.v0);
               printf("value of in0.v0");
               for(i=0; i<{le}/2; ++i)printf(" %#010x", buf1[i]);
               printf("\\nvalue of in0.v1");
               wasm_v128_store((void *)buf1, a0.v1);
               for(i=0; i<{le}/2; ++i)printf(" %#010x", buf1[i]);
               printf("\\n");
               
               return ret;
               """.format(
            **fmtspec2, op=op,
            pre_sat=pre_sat
        )
    return """
           return {pretyp}_{op}{pre_sat}({in0},{in1});
           """.format(
        **fmtspec2, op=op,
        pre_sat=pre_sat
    )


# -----------------------------------------------------------------------------
# Len

def len1(simd_ext, typ):
    return 'return {le};'.format(**fmtspec)


# -----------------------------------------------------------------------------
# Shift left and right
@printf2
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

@printf2
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
@printf2
def set1(simd_ext, typ):
    if typ[0] == "u":
        typ = "i" + typ[1:]
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  f32 f = nsimd_f16_to_f32({in0});
                  ret.v0 = nsimd_set1_{simd_ext}_f32(f);
                  ret.v1 = ret.v0;
                  return ret;'''.format(**fmtspec)
    return """
           nsimd_{simd_ext}_v{typ} ret = {pretyp1}_splat({in0});
           printf("value to set %lu\\n", {in0});
           int i;
           {typ} buf[{le}];
           wasm_v128_store((void *)buf, ret);
           printf("value of ret ");
           for(i=0; i<{le}; ++i)printf(" %lu", buf[i]);
           printf("\\n");
           
    
    
           return {pretyp1}_splat({in0});
           """.format(**fmtspec,
                      pretyp1=pretyp(simd_ext, typ))


#
# # -----------------------------------------------------------------------------
# # set1l or splat function for logical
#
@printf2
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
@printf2
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
           nsimd_{simd_ext}_v{typ} mask0 = {in1} & ~{in0};
           nsimd_{simd_ext}_v{typ} mask1 = {in2} & {in0};
           return mask0 | mask1;
           """.format(**fmtspec)


# -----------------------------------------------------------------------------
# min and max functions
@printf2
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
@printf2
def sqrt1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = wasm_f32x4_sqrt({in0}.v0);
                  ret.v1 = wasm_f32x4_sqrt({in0}.v1);
                  return ret;'''.format(**fmtspec)
    return 'return {pretyp}_sqrt({in0});'.format(**fmtspec)


# -----------------------------------------------------------------------------
# Load logical

@printf2
def loadl(simd_ext, typ, aligned):
    return \
        '''/* This can surely be improved but it is not our priority. */
           return nsimd_notl_{simd_ext}_{typ}(nsimd_eq_{simd_ext}_{typ}(
                    nsimd_load{align}_{simd_ext}_{typ}(
                      {in0}), nsimd_set1_{simd_ext}_{typ}({zero})));'''. \
            format(align='a' if aligned else 'u',
                   zero='nsimd_f32_to_f16(0.0f)' if typ == 'f16'
                   else '({})0'.format(typ), **fmtspec)


# -----------------------------------------------------------------------------
# Store logical

@printf2
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
@printf2
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
#
# def fma_fms(func, simd_ext, typ):
#     op = 'add' if func in ['fma', 'fnma'] else 'sub'
#     neg = 'n' if func in ['fnma', 'fnms'] else ''
#     if typ == 'f16':
#         return \
#         '''nsimd_{simd_ext}_vf16 ret;
#            ret.v0 = nsimd_{func}_{simd_ext}_f32({in0}.v0, {in1}.v0, {in2}.v0);
#            ret.v1 = nsimd_{func}_{simd_ext}_f32({in0}.v1, {in1}.v1, {in2}.v1);
#            return ret;'''.format(neg=neg, func=func, **fmtspec)
#     if neg == '':
#         emulate = '''return nsimd_{op}_{simd_ext}_{typ}(
#                               nsimd_mul_{simd_ext}_{typ}({in0}, {in1}),
#                                 {in2});'''.format(op=op, **fmtspec)
#     else:
#         emulate = '''return nsimd_{op}_{simd_ext}_{typ}(
#                               nsimd_mul_{simd_ext}_{typ}(
#                                 nsimd_neg_{simd_ext}_{typ}({in0}), {in1}),
#                                     {in2});'''.format(op=op, **fmtspec)
#     # One could use only emulate and no split. But to avoid splitting and
#     # merging SIMD register for each operation: sub, mul and add, we use
#     # emulation only for SIMD extensions that have natively add, sub and mul
#     # intrinsics.
#     split = split_opn(func, simd_ext, typ, 3)
#     if typ in ['f32', 'f64']:
#         if simd_ext in sse + avx:
#             return '''#ifdef NSIMD_FMA
#                         return {pre}f{neg}m{op}{suf}({in0}, {in1}, {in2});
#                       # else
#                         {emulate}
#                       # endif'''.format(op=op, neg=neg, emulate=emulate,
#                                        **fmtspec)
#         else:
#             return 'return {pre}f{neg}m{op}{suf}({in0}, {in1}, {in2});'. \
#                    format(op=op, neg=neg, **fmtspec)
#     if simd_ext in avx:
#         return emulate if simd_ext == 'avx2' else split
#     if simd_ext in avx512:
#         return emulate if simd_ext == 'avx512_skylake' else split
#     return emulate
#
# -----------------------------------------------------------------------------
# Ceil floor trunc and round_to_even
@printf2
def round1(opts, func, simd_ext, typ):
    if func == "round_to_even":
        func = "nearest"
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_{func}_{simd_ext}_f32({in0}.v0);
                  ret.v1 = nsimd_{func}_{simd_ext}_f32({in0}.v1);
                  return ret;'''.format(func=func, **fmtspec)
    if typ in ['f32', 'f64']:
        return """
               return {pretyp}_{func}({in0});
               """.format(func=func, **fmtspec)

    return 'return {in0};'.format(**fmtspec)


# -----------------------------------------------------------------------------
# All and any functions

@printf2
def all_any(func, simd_ext, typ):
    typ1 = "i" + typ[1:] if typ != "f16" else "i32"
    pret = pretyp(simd_ext, typ1)
    op = pret + "_all_true" if func == "all" else "" + pre(simd_ext) + "any_true"
    if typ == 'f16':
        return """
               return {op}({in0}.v0) && {op}({in0}.v1);
               """.format(**fmtspec, op=op)
    return """
           int i;
    
           {typ} buf[{le}];
           printf("value of in0 ");
           for(i=0; i<{le}; ++i)printf(" %u", a0[i]);
           printf("\\n");
           printf("value of ret: %u\\n", {op}({in0}));
        
    
           return {op}({in0});
           """.format(**fmtspec, op = op)

# # -----------------------------------------------------------------------------
# # Reinterpret (bitwise_cast)
#
# def reinterpret1(simd_ext, from_typ, to_typ):
#     if from_typ == to_typ:
#         return 'return {in0};'.format(**fmtspec)
#     if to_typ == 'f16':
#         emulate = '''{from_typ} buf[{le}];
#                      nsimd_storeu_{simd_ext}_{from_typ}(buf, {in0});
#                      return nsimd_loadu_{simd_ext}_f16((f16*)buf);'''. \
#                      format(**fmtspec)
#         native = '''nsimd_{simd_ext}_vf16 ret;
#                     ret.v0 = {pre}cvtph_ps({extract_lo});
#                     ret.v1 = {pre}cvtph_ps({extract_hi});
#                     return ret;'''.format(
#                     extract_lo=extract(simd_ext, 'u16', LO, common.in0),
#                     extract_hi=extract(simd_ext, 'u16', HI, common.in0),
#                     **fmtspec)
#         if simd_ext in sse:
#             return \
#             '''#ifdef NSIMD_FP16
#                  nsimd_{simd_ext}_vf16 ret;
#                  ret.v0 = _mm_cvtph_ps({in0});
#                  {in0} = _mm_shuffle_epi32({in0}, 14); /* = (3 << 2) | (2 << 0) */
#                  ret.v1 = _mm_cvtph_ps({in0});
#                  return ret;
#                #else
#                  {emulate}
#                #endif'''.format(emulate=emulate, **fmtspec)
#         if simd_ext in avx:
#             return \
#             '''#ifdef NSIMD_FP16
#                  {}
#                #else
#                  {}
#                #endif'''.format(native, emulate)
#         if simd_ext in avx512:
#             return native
#     if from_typ == 'f16':
#         emulate = \
#         '''u16 buf[{le}];
#            nsimd_storeu_{simd_ext}_f16((f16*)buf, {in0});
#            return nsimd_loadu_{simd_ext}_{to_typ}(({to_typ}*)buf);'''. \
#            format(**fmtspec)
#         native = 'return {};'.format(setr(simd_ext, 'u16',
#                  '{pre}cvtps_ph({in0}.v0, 4)'.format(**fmtspec),
#                  '{pre}cvtps_ph({in0}.v1, 4)'.format(**fmtspec)))
#         if simd_ext in sse:
#             return \
#             '''#ifdef NSIMD_FP16
#                  __m128i lo = _mm_cvtps_ph({in0}.v0, 4);
#                  __m128i hi = _mm_cvtps_ph({in0}.v1, 4);
#                  return _mm_castpd_si128(_mm_shuffle_pd(
#                           _mm_castsi128_pd(lo), _mm_castsi128_pd(hi), 0));
#                #else
#                  {emulate}
#                #endif'''.format(emulate=emulate, **fmtspec)
#         if simd_ext in avx:
#             return \
#             '''#ifdef NSIMD_FP16
#                  {}
#                #else
#                  {}
#                #endif'''.format(native, emulate)
#         if simd_ext in avx512:
#             return native
#     if from_typ in common.iutypes and to_typ in common.iutypes:
#         return 'return {in0};'.format(**fmtspec)
#     if to_typ in ['f32', 'f64']:
#         return 'return {pre}castsi{nbits}{to_suf}({in0});'. \
#                format(to_suf=suf_ep(to_typ), **fmtspec)
#     if from_typ in ['f32', 'f64']:
#         return 'return {pre}cast{from_suf}_si{nbits}({in0});'. \
#                format(from_suf=suf_ep(from_typ)[1:], **fmtspec)
#
# # -----------------------------------------------------------------------------
# # Reinterpretl, i.e. reinterpret on logicals
#
# def reinterpretl1(simd_ext, from_typ, to_typ):
#     if from_typ == to_typ:
#         return 'return {in0};'.format(**fmtspec)
#     if to_typ == 'f16':
#         if simd_ext in sse:
#             return \
#             '''nsimd_{simd_ext}_vlf16 ret;
#                ret.v0 = _mm_castsi128_ps(_mm_unpacklo_epi16({in0}, {in0}));
#                ret.v1 = _mm_castsi128_ps(_mm_unpackhi_epi16({in0}, {in0}));
#                return ret;'''.format(**fmtspec)
#         if simd_ext == 'avx':
#             return \
#             '''nsimd_{simd_ext}_vlf16 ret;
#                nsimd_sse42_vlf16 tmp1 =
#                    nsimd_reinterpretl_sse42_f16_{from_typ}(
#                      _mm256_castsi256_si128({in0}));
#                nsimd_sse42_vlf16 tmp2 =
#                    nsimd_reinterpretl_sse42_f16_{from_typ}(
#                       _mm256_extractf128_si256({in0}, 1));
#                ret.v0 = {setr_tmp1};
#                ret.v1 = {setr_tmp2};
#                return ret;'''. \
#                format(setr_tmp1=setr('avx', 'f32', 'tmp1.v0', 'tmp1.v1'),
#                       setr_tmp2=setr('avx', 'f32', 'tmp2.v0', 'tmp2.v1'),
#                       **fmtspec)
#         if simd_ext == 'avx2':
#             return \
#             '''nsimd_{simd_ext}_vlf16 ret;
#                ret.v0 = _mm256_castsi256_ps(_mm256_cvtepi16_epi32(
#                           _mm256_castsi256_si128({in0})));
#                ret.v1 = _mm256_castsi256_ps(_mm256_cvtepi16_epi32(
#                           _mm256_extractf128_si256({in0}, 1)));
#                return ret;'''.format(**fmtspec)
#         if simd_ext in avx512:
#             return '''nsimd_{simd_ext}_vlf16 ret;
#                       ret.v0 = (__mmask16)({in0} & 0xFFFF);
#                       ret.v1 = (__mmask16)(({in0} >> 16) & 0xFFFF);
#                       return ret;'''.format(**fmtspec)
#     if from_typ == 'f16':
#         if simd_ext in sse + avx:
#             return '''f32 in[{le}];
#                       {to_typ} out[{le}];
#                       int i;
#                       nsimd_storeu_{simd_ext}_f32(in, {in0}.v0);
#                       nsimd_storeu_{simd_ext}_f32(in + {leo2}, {in0}.v1);
#                       for (i = 0; i < {le}; i++) {{
#                         out[i] = ({to_typ})(in[i] != 0.0f ? -1 : 0);
#                       }}
#                       return nsimd_loadu_{simd_ext}_{to_typ}(out);'''. \
#                       format(leo2=int(fmtspec['le']) // 2, **fmtspec)
#         if simd_ext in avx512:
#             return \
#             'return (__mmask32){in0}.v0 | ((__mmask32){in0}.v1 << 16);'. \
#             format(**fmtspec)
#     if simd_ext in sse + avx:
#         return reinterpret1(simd_ext, from_typ, to_typ)
#     else:
#         return 'return {in0};'.format(**fmtspec)
#
# # -----------------------------------------------------------------------------
# # Convert
#
# def convert1(simd_ext, from_typ, to_typ):
#     if to_typ == from_typ or \
#        to_typ in common.iutypes and from_typ in common.iutypes:
#         return 'return {in0};'.format(**fmtspec)
#     if to_typ == 'f16':
#         if simd_ext in sse:
#             getlo = '{in0}'.format(**fmtspec)
#             gethi = '_mm_unpackhi_epi64({in0}, {in0})'.format(**fmtspec)
#         if simd_ext in avx:
#             getlo = '_mm256_castsi256_si128({in0})'.format(**fmtspec)
#             gethi = '_mm256_extractf128_si256({in0}, 1)'.format(**fmtspec)
#         if simd_ext in avx512:
#             getlo = '_mm512_castsi512_si256({in0})'.format(**fmtspec)
#             gethi = '_mm512_extracti64x4_epi64({in0}, 1)'.format(**fmtspec)
#         through_epi32 = \
#         '''nsimd_{simd_ext}_v{to_typ} ret;
#            ret.v0 = {pre}cvtepi32_ps({pre}cvtep{from_typ}_epi32({getlo}));
#            ret.v1 = {pre}cvtepi32_ps({pre}cvtep{from_typ}_epi32({gethi}));
#            return ret;'''.format(getlo=getlo, gethi=gethi, **fmtspec)
#         emulate = '''{from_typ} in[{le}];
#                      f32 out[{leo2}];
#                      nsimd_{simd_ext}_vf16 ret;
#                      int i;
#                      nsimd_storeu_{simd_ext}_{from_typ}(in, {in0});
#                      for (i = 0; i < {leo2}; i++) {{
#                        out[i] = (f32)in[i];
#                      }}
#                      ret.v0 = nsimd_loadu_{simd_ext}_f32(out);
#                      for (i = 0; i < {leo2}; i++) {{
#                        out[i] = (f32)in[i + {leo2}];
#                      }}
#                      ret.v1 = nsimd_loadu_{simd_ext}_f32(out);
#                      return ret;'''.format(leo2=int(fmtspec['le']) // 2,
#                                            **fmtspec)
#         if simd_ext in ['sse42', 'avx2']:
#             return through_epi32
#         if simd_ext in ['sse2', 'avx']:
#             return emulate
#         if simd_ext in avx512:
#             return through_epi32
#     if from_typ == 'f16':
#         return '''f32 in[{leo2}];
#                   {to_typ} out[{le}];
#                   int i;
#                   nsimd_storeu_{simd_ext}_f32(in, {in0}.v0);
#                   for (i = 0; i < {leo2}; i++) {{
#                     out[i] = ({to_typ})in[i];
#                   }}
#                   nsimd_storeu_{simd_ext}_f32(in, {in0}.v1);
#                   for (i = 0; i < {leo2}; i++) {{
#                     out[i + {leo2}] = ({to_typ})in[i];
#                   }}
#                   return nsimd_loadu_{simd_ext}_{to_typ}(out);'''. \
#                   format(leo2=int(fmtspec['le']) // 2, **fmtspec)
#     emulate = '''{from_typ} in[{le}];
#                  {to_typ} out[{le}];
#                  int i;
#                  nsimd_storeu_{simd_ext}_{from_typ}(in, {in0});
#                  for (i = 0; i < {le}; i++) {{
#                    out[i] = ({to_typ})in[i];
#                  }}
#                  return nsimd_loadu_{simd_ext}_{to_typ}(out);'''. \
#                  format(**fmtspec)
#     if to_typ == 'f64' or from_typ == 'f64':
#         if simd_ext == 'avx512_skylake':
#             return 'return _mm512_cvt{from_suf}{to_suf}({in0});'. \
#                    format(from_suf=suf_ep(from_typ)[1:], to_suf=suf_ep(to_typ),
#                           **fmtspec)
#         else:
#             return emulate
#     if to_typ == 'f32' and from_typ == 'i32':
#         return 'return {pre}cvtepi32_ps({in0});'.format(**fmtspec)
#     if to_typ == 'f32' and from_typ == 'u32':
#         if simd_ext in sse + avx:
#             return emulate
#         if simd_ext in avx512:
#             return 'return _mm512_cvtepu32_ps({in0});'.format(**fmtspec)
#     if to_typ == 'i32' and from_typ == 'f32':
#         return 'return {pre}cvtps_epi32({in0});'.format(**fmtspec)
#     if to_typ == 'u32' and from_typ == 'f32':
#         if simd_ext in sse + avx:
#             return emulate
#         if simd_ext in avx512:
#             return 'return _mm512_cvtps_epu32({in0});'.format(**fmtspec)
#
# # -----------------------------------------------------------------------------
# # Reciprocal (at least 11 bits of precision)
#
# def rec11_rsqrt11(func, simd_ext, typ):
#     if typ == 'f16':
#         return '''nsimd_{simd_ext}_vf16 ret;
#                   ret.v0 = nsimd_{func}11_{simd_ext}_f32({in0}.v0);
#                   ret.v1 = nsimd_{func}11_{simd_ext}_f32({in0}.v1);
#                   return ret;'''. \
#                   format(func='rec' if func == 'rcp' else 'rsqrt', **fmtspec)
#     if typ == 'f32':
#         if simd_ext in sse + avx:
#             return 'return {pre}{func}_ps({in0});'.format(func=func, **fmtspec)
#         if simd_ext in avx512:
#             return 'return _mm512_{func}14_ps({in0});'. \
#                    format(func=func, **fmtspec)
#     if typ == 'f64':
#         if simd_ext in sse + avx:
#             one = '{pre}set1_pd(1.0)'.format(**fmtspec)
#             if func == 'rcp':
#                 return 'return {pre}div{suf}({one}, {in0});'.format(one=one, **fmtspec)
#             else:
#                 return 'return {pre}div{suf}({one}, {pre}sqrt{suf}({in0}));'. \
#                         format(one=one, **fmtspec)
#             format(func=func, **fmtspec)
#         if simd_ext in avx512:
#             return 'return _mm512_{func}14_pd({in0});'. \
#                    format(func=func, **fmtspec)
#
# # -----------------------------------------------------------------------------
# # Reciprocal (IEEE)
#
# def rec1(simd_ext, typ):
#     one = '{pre}set1_ps(1.0f)'.format(**fmtspec) if typ in ['f16', 'f32'] \
#           else '{pre}set1_pd(1.0)'.format(**fmtspec)
#     if typ == 'f16':
#         return '''nsimd_{simd_ext}_vf16 ret;
#                   nsimd_{simd_ext}_vf32 one = {one};
#                   ret.v0 = {pre}div_ps(one, {in0}.v0);
#                   ret.v1 = {pre}div_ps(one, {in0}.v1);
#                   return ret;'''.format(one=one, **fmtspec)
#     return 'return {pre}div{suf}({one}, {in0});'.format(one=one, **fmtspec)
#
# # -----------------------------------------------------------------------------
# # Negative
#
# def neg1(simd_ext, typ):
#     cte = '0x80000000' if typ in ['f16', 'f32'] else '0x8000000000000000'
#     fsuf = '_ps' if typ in ['f16', 'f32'] else '_pd'
#     utyp = 'u32' if typ in ['f16', 'f32'] else 'u64'
#     vmask = '{pre}castsi{nbits}{fsuf}(nsimd_set1_{simd_ext}_{utyp}({cte}))'. \
#             format(cte=cte, utyp=utyp, fsuf=fsuf, **fmtspec)
#     if typ == 'f16':
#         return '''nsimd_{simd_ext}_vf16 ret;
#                   nsimd_{simd_ext}_vf32 mask = {vmask};
#                   ret.v0 = nsimd_xorb_{simd_ext}_f32(mask, {in0}.v0);
#                   ret.v1 = nsimd_xorb_{simd_ext}_f32(mask, {in0}.v1);
#                   return ret;'''.format(vmask=vmask, **fmtspec)
#     if typ in ['f32', 'f64']:
#         return 'return nsimd_xorb_{simd_ext}_{typ}({vmask}, {in0});'. \
#                format(vmask=vmask, **fmtspec)
#     return '''return nsimd_sub_{simd_ext}_{typ}(
#                   {pre}setzero_si{nbits}(), {in0});'''. \
#               format(**fmtspec)
#
# # -----------------------------------------------------------------------------
# # nbtrue
#
# def nbtrue1(simd_ext, typ):
#     if typ == 'f16':
#         return '''return nsimd_nbtrue_{simd_ext}_f32({in0}.v0) +
#                          nsimd_nbtrue_{simd_ext}_f32({in0}.v1);'''. \
#                          format(**fmtspec)
#     if typ in ['i8', 'u8']:
#         code = 'return nsimd_popcnt32_((u32){pre}movemask_epi8({in0}));'. \
#                format(**fmtspec)
#     elif typ in ['i16', 'u16']:
#         code = 'return nsimd_popcnt32_((u32){pre}movemask_epi8({in0})) >> 1;'. \
#                format(**fmtspec)
#     elif typ in ['i32', 'u32', 'i64', 'u64']:
#         code = '''return nsimd_popcnt32_((u32){pre}movemask{fsuf}(
#                       {pre}castsi{nbits}{fsuf}({in0})));'''. \
#                       format(fsuf='_ps' if typ in ['i32', 'u32'] else '_pd',
#                              **fmtspec)
#     else:
#         code = 'return nsimd_popcnt32_((u32){pre}movemask{suf}({in0}));'. \
#                format(**fmtspec)
#     if simd_ext in sse:
#         return code
#     if simd_ext in avx:
#         if typ in ['i32', 'u32', 'i64', 'u64', 'f32', 'f64']:
#             return code
#         else:
#             if simd_ext == 'avx2':
#                 return code
#             else:
#                 return \
#                 '''return nsimd_nbtrue_sse42_{typ}(
#                             _mm256_castsi256_si128({in0})) +
#                               nsimd_nbtrue_sse42_{typ}(
#                                 _mm256_extractf128_si256({in0}, 1));'''. \
#                                 format(**fmtspec)
#     if simd_ext in avx512:
#         return 'return nsimd_popcnt64_((u64){in0});'.format(**fmtspec)
#
# # -----------------------------------------------------------------------------
# # reverse
#
# def reverse1(simd_ext, typ):
#     # 8-bit int
#     if typ in ['i8', 'u8']:
#         if simd_ext == 'sse2':
#             return '''{in0} = _mm_shufflehi_epi16({in0}, _MM_SHUFFLE(0,1,2,3));
#                       {in0} = _mm_shufflelo_epi16({in0}, _MM_SHUFFLE(0,1,2,3));
#                       {in0} = _mm_castpd_si128(_mm_shuffle_pd(
#                                 _mm_castsi128_pd({in0}), _mm_castsi128_pd(
#                                   {in0}), 1));
#                       nsimd_{simd_ext}_v{typ} r0 = _mm_srli_epi16({in0}, 8);
#                       nsimd_{simd_ext}_v{typ} r1 = _mm_slli_epi16({in0}, 8);
#                       return _mm_or_si128(r0, r1);'''.format(**fmtspec)
#         elif simd_ext == 'sse42':
#             return '''nsimd_{simd_ext}_v{typ} mask = _mm_set_epi8(
#                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
#                       return _mm_shuffle_epi8({in0}, mask);'''. \
#                       format(**fmtspec)
#         elif simd_ext == 'avx':
#             return \
#             '''nsimd_sse42_v{typ} r0 = _mm_shuffle_epi8(
#                  _mm256_extractf128_si256({in0}, 0), _mm_set_epi8(
#                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
#                nsimd_sse42_v{typ} r1 = _mm_shuffle_epi8(
#                  _mm256_extractf128_si256({in0}, 1), _mm_set_epi8(
#                    0,  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
#                {in0} = _mm256_insertf128_si256({in0}, r0, 1);
#                return _mm256_insertf128_si256({in0}, r1, 0);'''. \
#                format(**fmtspec)
#         elif simd_ext == 'avx2':
#              return \
#              '''{in0} = _mm256_shuffle_epi8({in0}, _mm256_set_epi8(
#                    0,  1,  2,  3,  4,  5,  6,  7,
#                    8,  9, 10, 11, 12, 13, 14, 15,
#                   16, 17, 18, 19, 20, 21, 22, 23,
#                   24, 25, 26, 27, 28, 29, 30, 31));
#                 return _mm256_permute2x128_si256({in0}, {in0}, 1);'''. \
#                 format(**fmtspec)
#         # AVX-512F and above.
#         else:
#              return \
#              '''nsimd_avx2_v{typ} r0 = _mm512_extracti64x4_epi64({in0}, 0);
#                 nsimd_avx2_v{typ} r1 = _mm512_extracti64x4_epi64({in0}, 1);
#                 r0 = _mm256_shuffle_epi8(r0, _mm256_set_epi8(
#                      0,  1,  2,  3,  4,  5,  6,  7,
#                      8,  9, 10, 11, 12, 13, 14, 15,
#                     16, 17, 18, 19, 20, 21, 22, 23,
#                     24, 25, 26, 27, 28, 29, 30, 31));
#                 r1 = _mm256_shuffle_epi8(r1, _mm256_set_epi8(
#                       0,  1,  2,  3,  4,  5,  6,  7,
#                       8,  9, 10, 11, 12, 13, 14, 15,
#                      16, 17, 18, 19, 20, 21, 22, 23,
#                      24, 25, 26, 27, 28, 29, 30, 31));
#                 r0 = _mm256_permute2x128_si256(r0, r0, 1);
#                 r1 = _mm256_permute2x128_si256(r1, r1, 1);
#                 {in0} = _mm512_insertf64x4({in0}, r0, 1);
#                 return _mm512_insertf64x4({in0}, r1, 0);'''.format(**fmtspec)
#     # 16-bit int
#     elif typ in ['i16', 'u16']:
#         if simd_ext == 'sse2':
#             return '''{in0} = _mm_shufflehi_epi16( {in0}, _MM_SHUFFLE(0,1,2,3) );
#                       {in0} = _mm_shufflelo_epi16( {in0}, _MM_SHUFFLE(0,1,2,3) );
#                       return _mm_castpd_si128(_mm_shuffle_pd(
#                                _mm_castsi128_pd({in0}),
#                                _mm_castsi128_pd({in0}), 1));'''. \
#                                format(**fmtspec)
#         elif simd_ext == 'sse42':
#             return \
#             '''return _mm_shuffle_epi8({in0}, _mm_set_epi8(
#                         1,  0,  3,  2,  5,  4,  7, 6,
#                         9,  8, 11, 10, 13, 12, 15, 14));'''.format(**fmtspec)
#         elif simd_ext == 'avx':
#             return \
#             '''nsimd_sse42_v{typ} r0 = _mm_shuffle_epi8(
#                  _mm256_extractf128_si256({in0}, 0), _mm_set_epi8(
#                    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
#                nsimd_sse42_v{typ} r1 = _mm_shuffle_epi8(
#                  _mm256_extractf128_si256({in0}, 1), _mm_set_epi8(
#                    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
#                {in0} = _mm256_insertf128_si256( {in0}, r0, 1);
#                return _mm256_insertf128_si256({in0}, r1, 0);'''. \
#                format(**fmtspec)
#         elif simd_ext == 'avx2':
#             return \
#             '''{in0} = _mm256_shuffle_epi8({in0}, _mm256_set_epi8(
#                            1,  0,  3,  2,  5,  4,  7,  6,
#                            9,  8, 11, 10, 13, 12, 15, 14,
#                           17, 16, 19, 18, 21, 20, 23, 22,
#                           25, 24, 27, 26, 29, 28, 31, 30));
#                return _mm256_permute2x128_si256({in0}, {in0}, 1);'''. \
#                format(**fmtspec)
#         # AVX-512F
#         elif simd_ext == 'avx512_knl':
#             return \
#             '''{in0} = _mm512_permutexvar_epi32(_mm512_set_epi32(
#                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
#                  {in0});
#                nsimd_{simd_ext}_v{typ} r0 = _mm512_srli_epi32({in0}, 16);
#                nsimd_{simd_ext}_v{typ} r1 = _mm512_slli_epi32({in0}, 16);
#                return _mm512_or_si512(r0, r1);'''.format(**fmtspec)
#         # AVX-512F+BW (Skylake) + WORKAROUND GCC<=8
#         else:
#             return \
#             '''return _mm512_permutexvar_epi16(_mm512_set_epi32(
#                  (0<<16)  | 1,  (2<<16)  | 3,  (4<<16)  | 5,  (6<<16)  | 7,
#                  (8<<16)  | 9,  (10<<16) | 11, (12<<16) | 13, (14<<16) | 15,
#                  (16<<16) | 17, (18<<16) | 19, (20<<16) | 21, (22<<16) | 23,
#                  (24<<16) | 25, (26<<16) | 27, (28<<16) | 29, (30<<16) | 31),
#                  {in0} );'''.format(**fmtspec)
#     # 32-bit int
#     elif typ in ['i32', 'u32']:
#         if simd_ext in ['sse2', 'sse42']:
#             return 'return _mm_shuffle_epi32({in0}, _MM_SHUFFLE(0,1,2,3));'. \
#                    format(**fmtspec)
#         elif simd_ext == 'avx':
#             return '''{in0} = _mm256_castps_si256(_mm256_shuffle_ps(
#                                 _mm256_castsi256_ps({in0}),
#                                 _mm256_castsi256_ps({in0}),
#                                 _MM_SHUFFLE(0,1,2,3)));
#                       return _mm256_permute2f128_si256({in0}, {in0}, 1);'''. \
#                       format(**fmtspec)
#         elif simd_ext == 'avx2':
#             return \
#             '''{in0} = _mm256_shuffle_epi32({in0}, _MM_SHUFFLE(0,1,2,3));
#                return _mm256_permute2x128_si256({in0}, {in0}, 1);'''. \
#                format(**fmtspec)
#         else:
#             return \
#             '''return _mm512_permutexvar_epi32(_mm512_set_epi32(
#                  0, 1,  2,  3,  4,  5,  6,  7,
#                  8, 9, 10, 11, 12, 13, 14, 15), {in0});'''. \
#                  format(**fmtspec)
#     elif typ in ['i64', 'u64']:
#         if simd_ext in ['sse2', 'sse42']:
#             return '''return _mm_castpd_si128(_mm_shuffle_pd(_mm_castsi128_pd(
#                                {in0}), _mm_castsi128_pd({in0}), 1));'''. \
#                                format(**fmtspec)
#         elif simd_ext == 'avx':
#             return '''{in0} = _mm256_castpd_si256(
#                                   _mm256_shuffle_pd(
#                                      _mm256_castsi256_pd({in0}),
#                                      _mm256_castsi256_pd({in0}),
#                                      (1<<2) | 1
#                                   )
#                               );
#                        return _mm256_permute2f128_si256({in0}, {in0}, 1);'''. \
#                        format(**fmtspec)
#         elif simd_ext == 'avx2':
#            return '''return _mm256_permute4x64_epi64({in0},
#                               _MM_SHUFFLE(0, 1, 2, 3));'''.format(**fmtspec)
#         else:
#            return '''return _mm512_permutexvar_epi64(_mm512_set_epi64(
#                               0, 1, 2, 3, 4, 5, 6, 7), {in0});'''. \
#                               format(**fmtspec)
#     # 16-bit float
#     elif typ == 'f16':
#         return '''nsimd_{simd_ext}_vf16 ret;
#                   ret.v0 = nsimd_reverse_{simd_ext}_f32({in0}.v0);
#                   ret.v1 = nsimd_reverse_{simd_ext}_f32({in0}.v1);
#                   return ret;'''.format(**fmtspec)
#     # 32-bit float
#     elif typ == 'f32':
#         if simd_ext in ['sse2', 'sse42']:
#             return '''return _mm_shuffle_ps({in0}, {in0},
#                                _MM_SHUFFLE(0, 1, 2, 3));'''.format(**fmtspec)
#         elif simd_ext in ['avx', 'avx2']:
#             return '''{in0} = _mm256_shuffle_ps({in0}, {in0},
#                                 _MM_SHUFFLE(0, 1, 2, 3));
#                       return _mm256_permute2f128_ps({in0}, {in0}, 1);'''. \
#                       format(**fmtspec)
#         else:
#             return \
#             '''return _mm512_permutexvar_ps(_mm512_set_epi32(
#                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
#                         {in0} );'''.format(**fmtspec)
#     # 64-bit float
#     else:
#         if simd_ext in ['sse2', 'sse42']:
#             return 'return _mm_shuffle_pd({in0}, {in0}, 1);'.format(**fmtspec)
#         elif simd_ext == 'avx':
#             return '''{in0} = _mm256_shuffle_pd({in0}, {in0}, (1<<2) | 1);
#                       return _mm256_permute2f128_pd({in0}, {in0}, 1);'''. \
#                       format(**fmtspec)
#         elif simd_ext == 'avx2':
#             return '''return _mm256_permute4x64_pd({in0},
#                                _MM_SHUFFLE(0, 1, 2, 3));'''.format(**fmtspec)
#         else:
#             return '''return _mm512_permute_mm512_set_epi64(
#                                0, 1, 2, 3, 4, 5, 6, 7), {in0});'''. \
#                                format(**fmtspec)
#
# # -----------------------------------------------------------------------------
# # addv
#
# def addv(simd_ext, typ):
#     if simd_ext in sse:
#         if typ == 'f64':
#             return \
#             '''return _mm_cvtsd_f64(_mm_add_pd({in0},
#                                     _mm_shuffle_pd({in0}, {in0}, 0x01)));'''. \
#                                     format(**fmtspec)
#         elif typ == 'f32':
#             return \
#             '''nsimd_{simd_ext}_vf32 tmp = _mm_add_ps({in0}, _mm_shuffle_ps(
#                                              {in0}, {in0}, 0xb1));
#                return _mm_cvtss_f32(_mm_add_ps(tmp, _mm_shuffle_ps(
#                         tmp, tmp, 0x4e)));''' .format(**fmtspec)
#         elif typ == 'f16':
#             return \
#             '''nsimd_{simd_ext}_vf32 tmp0 = _mm_add_ps({in0}.v0,
#                  _mm_shuffle_ps({in0}.v0, {in0}.v0, 0xb1));
#                nsimd_{simd_ext}_vf32 tmp1 = _mm_add_ps({in0}.v1,
#                  _mm_shuffle_ps({in0}.v1, {in0}.v1, 0xb1));
#                return nsimd_f32_to_f16(_mm_cvtss_f32(_mm_add_ps(
#                  tmp0, _mm_shuffle_ps(tmp0, tmp0, 0x4e))) +
#                    _mm_cvtss_f32(_mm_add_ps(tmp1, _mm_shuffle_ps(
#                      tmp1, tmp1, 0x4e))));''' .format(**fmtspec)
#     elif simd_ext in avx:
#         if typ == 'f64':
#             return \
#             '''__m128d tmp = _mm_add_pd(_mm256_extractf128_pd({in0}, 1),
#                                         _mm256_extractf128_pd({in0}, 0));
#                return _mm_cvtsd_f64(_mm_add_pd(tmp, _mm_shuffle_pd(
#                         tmp, tmp, 0x01)));''' .format(**fmtspec)
#         elif typ == 'f32':
#             return \
#             '''__m128 tmp0 = _mm_add_ps(_mm256_extractf128_ps({in0}, 1),
#                                         _mm256_extractf128_ps({in0}, 0));
#                __m128 tmp1 = _mm_add_ps(tmp0, _mm_shuffle_ps(tmp0, tmp0, 0xb1));
#                return _mm_cvtss_f32(_mm_add_ps(tmp1, _mm_shuffle_ps(
#                         tmp1, tmp1, 0x4e)));''' .format(**fmtspec)
#         elif typ == 'f16':
#             return \
#             '''__m128 tmp00 = _mm_add_ps(_mm256_extractf128_ps({in0}.v0, 1),
#                                          _mm256_extractf128_ps({in0}.v0, 0));
#                __m128 tmp01 = _mm_add_ps(tmp00, _mm_shuffle_ps(
#                                 tmp00, tmp00, 0xb1));
#                __m128 tmp10 = _mm_add_ps(_mm256_extractf128_ps({in0}.v1, 1),
#                                          _mm256_extractf128_ps({in0}.v1, 0));
#                __m128 tmp11 = _mm_add_ps(tmp10, _mm_shuffle_ps(
#                                 tmp10, tmp10, 0xb1));
#                return nsimd_f32_to_f16(_mm_cvtss_f32(_mm_add_ps(
#                         tmp01, _mm_shuffle_ps(tmp01, tmp01, 0x4e))) +
#                           _mm_cvtss_f32(_mm_add_ps(tmp11, _mm_shuffle_ps(
#                             tmp11, tmp11, 0x4e))));
#                     ''' .format(**fmtspec)
#     elif simd_ext in avx512:
#         if typ == 'f64':
#             return \
#             '''__m256d tmp0 = _mm256_add_pd(_mm512_extractf64x4_pd({in0}, 0),
#                                             _mm512_extractf64x4_pd({in0}, 1));
#                __m128d tmp1 = _mm_add_pd(_mm256_extractf128_pd(tmp0, 1),
#                                          _mm256_extractf128_pd(tmp0, 0));
#                return _mm_cvtsd_f64(_mm_add_pd(tmp1, _mm_shuffle_pd(
#                         tmp1, tmp1, 0x01)));''' .format(**fmtspec)
#         elif typ == 'f32':
#             return \
#             '''__m128 tmp0 = _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(
#                                {in0}, 0), _mm512_extractf32x4_ps({in0}, 1)),
#                                _mm_add_ps(_mm512_extractf32x4_ps({in0}, 2),
#                                _mm512_extractf32x4_ps({in0}, 3)));
#                __m128 tmp1 = _mm_add_ps(tmp0, _mm_shuffle_ps(
#                                tmp0, tmp0, 0xb1));
#                return _mm_cvtss_f32(_mm_add_ps(tmp1, _mm_shuffle_ps(
#                         tmp1, tmp1, 0x4e)));''' .format(**fmtspec)
#         elif typ == 'f16':
#             return \
#             '''f32 res;
#                __m128 tmp0 = _mm_add_ps(
#                    _mm_add_ps(_mm512_extractf32x4_ps({in0}.v0, 0),
#                                _mm512_extractf32x4_ps({in0}.v0, 1)),
#                    _mm_add_ps(_mm512_extractf32x4_ps({in0}.v0, 2),
#                                _mm512_extractf32x4_ps({in0}.v0, 3)));
#                __m128 tmp1 = _mm_add_ps(tmp0, _mm_shuffle_ps(
#                                tmp0, tmp0, 0xb1));
#                res = _mm_cvtss_f32(_mm_add_ps(tmp1, _mm_shuffle_ps(
#                        tmp1, tmp1, 0x4e)));
#                tmp0 = _mm_add_ps(
#                    _mm_add_ps(_mm512_extractf32x4_ps({in0}.v1, 0),
#                                _mm512_extractf32x4_ps({in0}.v1, 1)),
#                    _mm_add_ps(_mm512_extractf32x4_ps({in0}.v1, 2),
#                                _mm512_extractf32x4_ps({in0}.v1, 3)));
#                tmp1 = _mm_add_ps(tmp0, _mm_shuffle_ps(tmp0, tmp0, 0xb1));
#                return nsimd_f32_to_f16(res + _mm_cvtss_f32(_mm_add_ps(
#                         tmp1, _mm_shuffle_ps(tmp1, tmp1, 0x4e))));''' . \
#                         format(**fmtspec)
#
# # -----------------------------------------------------------------------------
# # upconvert
#
# def upcvt1(simd_ext, from_typ, to_typ):
#     # From f16 is easy
#     if from_typ == 'f16':
#         if to_typ == 'f32':
#             return \
#             '''nsimd_{simd_ext}_vf32x2 ret;
#                ret.v0 = {in0}.v0;
#                ret.v1 = {in0}.v1;
#                return ret;'''.format(**fmtspec)
#         else:
#             return \
#             '''nsimd_{simd_ext}_v{to_typ}x2 ret;
#                ret.v0 = nsimd_cvt_{simd_ext}_{to_typ}_f32({in0}.v0);
#                ret.v1 = nsimd_cvt_{simd_ext}_{to_typ}_f32({in0}.v1);
#                return ret;'''.format(**fmtspec)
#
#     # To f16 is easy
#     if to_typ == 'f16':
#         return \
#         '''nsimd_{simd_ext}_vf16x2 ret;
#            nsimd_{simd_ext}_v{iu}16x2 buf;
#            buf = nsimd_upcvt_{simd_ext}_{iu}16_{iu}8({in0});
#            ret.v0 = nsimd_cvt_{simd_ext}_f16_{iu}16(buf.v0);
#            ret.v1 = nsimd_cvt_{simd_ext}_f16_{iu}16(buf.v1);
#            return ret;'''.format(iu=from_typ[0], **fmtspec)
#
#     # For integer upcast, due to 2's complement representation
#     # epi_epi : signed   -> bigger signed
#     # epi_epi : signed   -> bigger unsigned
#     # epu_epi : unsigned -> bigger signed
#     # epu_epi : unsigned -> bigger unsigned
#     if from_typ in common.iutypes:
#         suf_epep = 'ep{ui}{typnbits}_epi{typnbits2}'. \
#                    format(ui='u' if from_typ in common.utypes else 'i',
#                           typnbits2=str(int(fmtspec['typnbits']) * 2),
#                           **fmtspec)
#     else:
#         suf_epep = 'ps_pd'
#
#     # compute lower half
#     if simd_ext in sse:
#         lower_half = '{in0}'.format(**fmtspec)
#     else:
#         lower_half = extract(simd_ext, from_typ, LO, fmtspec['in0'])
#
#     # compute upper half
#     if simd_ext in sse:
#         if from_typ in common.iutypes:
#             upper_half = '_mm_shuffle_epi32({in0}, 14 /* 2 | 3 */)'. \
#                          format(**fmtspec)
#         else:
#             upper_half = '''{pre}castpd_ps({pre}shuffle_pd(
#                                 {pre}castps_pd({in0}),
#                                 {pre}castps_pd({in0}), 1))'''.format(**fmtspec)
#     else:
#         upper_half = extract(simd_ext, from_typ, HI, fmtspec['in0'])
#
#     # When intrinsics are provided
#     # for conversions integers <-> floating point, there is no intrinsics, so
#     # we use cvt's
#     if from_typ == 'i32' and to_typ == 'f64':
#         with_intrinsic = \
#         '''nsimd_{simd_ext}_vf64x2 ret;
#            ret.v0 = {pre}cvtepi32_pd({lower_half});
#            ret.v1 = {pre}cvtepi32_pd({upper_half});
#            return ret;'''.format(upper_half=upper_half,
#                                  lower_half=lower_half, **fmtspec)
#     elif (from_typ in common.iutypes and to_typ in common.iutypes) or \
#          (from_typ == 'f32' and to_typ == 'f64'):
#         with_intrinsic = \
#         '''nsimd_{simd_ext}_v{to_typ}x2 ret;
#            ret.v0 = {pre}cvt{suf_epep}({lower_half});
#            ret.v1 = {pre}cvt{suf_epep}({upper_half});
#            return ret;'''.format(upper_half=upper_half, lower_half=lower_half,
#                                  suf_epep=suf_epep, **fmtspec)
#     else:
#         from_typ2 = from_typ[0] + str(int(fmtspec['typnbits']) * 2)
#         if from_typ not in common.iutypes:
#             # getting here means that from_typ=f32 and to_typ=f64
#             with_intrinsic = \
#             '''nsimd_{simd_ext}_vf64x2 ret;
#                ret.v0 = nsimd_cvt_{simd_ext}_{to_typ}_f64({pre}cvtps_pd(
#                             {lower_half}));
#                ret.v1 = nsimd_cvt_{simd_ext}_{to_typ}_f64({pre}cvtps_pd(
#                             {upper_half}));
#                return ret;'''. \
#                format(upper_half=upper_half, lower_half=lower_half,
#                       from_typ2=from_typ2, suf_epep=suf_epep, **fmtspec)
#
#     # When no intrinsic is given for going from integers to floating or
#     # from floating to integer we can go through a cvt
#     if to_typ in common.ftypes:
#         int_float = \
#         '''nsimd_{simd_ext}_v{to_typ}x2 ret;
#            nsimd_{simd_ext}_v{int_typ}x2 tmp;
#            tmp = nsimd_upcvt_{simd_ext}_{int_typ}_{from_typ}({in0});
#            ret.v0 = nsimd_cvt_{simd_ext}_{to_typ}_{int_typ}(tmp.v0);
#            ret.v1 = nsimd_cvt_{simd_ext}_{to_typ}_{int_typ}(tmp.v1);
#            return ret;'''. \
#            format(int_typ=from_typ[0] + to_typ[1:], lower_half=lower_half,
#                   upper_half=upper_half, **fmtspec)
#     else:
#         int_float = \
#         '''return nsimd_upcvt_{simd_ext}_{to_typ}_{int_typ}(
#                       nsimd_cvt_{simd_ext}_{int_typ}_{from_typ}({in0}));'''. \
#                       format(int_typ=to_typ[0] + from_typ[1:],
#                              lower_half=lower_half, upper_half=upper_half,
#                              **fmtspec)
#
#     # When no intrinsic is given we can use the trick of falling back to
#     # the lower SIMD extension
#     split_trick = \
#     '''nsimd_{simd_ext}_v{to_typ}x2 ret;
#        nsimd_{simd_ext2}_v{to_typ}x2 ret2;
#        ret2 = nsimd_upcvt_{simd_ext2}_{to_typ}_{from_typ}({lo});
#        ret.v0 = {merge};
#        ret2 = nsimd_upcvt_{simd_ext2}_{to_typ}_{from_typ}({hi});
#        ret.v1 = {merge};
#        return ret;'''. \
#        format(simd_ext2='sse42' if simd_ext == 'avx' else 'avx2',
#               lo=extract(simd_ext, from_typ, LO, common.in0),
#               hi=extract(simd_ext, from_typ, HI, common.in0),
#               merge=setr(simd_ext, to_typ, 'ret2.v0', 'ret2.v1'), **fmtspec)
#
#     # return C code
#     if from_typ == 'i32' and to_typ == 'f64':
#         return with_intrinsic
#     if (from_typ in common.ftypes and to_typ in common.iutypes) or \
#        (from_typ in common.iutypes and to_typ in common.ftypes):
#         return int_float
#     # if simd_ext == 'sse2':
#     if simd_ext in sse:
#         if from_typ in common.itypes and to_typ in common.iutypes:
#             return \
#             '''nsimd_{simd_ext}_v{to_typ}x2 ret;
#                __m128i mask = _mm_cmpgt{suf}(_mm_setzero_si128(), {in0});
#                ret.v0 = _mm_unpacklo{suf}({in0}, mask);
#                ret.v1 = _mm_unpackhi{suf}({in0}, mask);
#                return ret;'''.format(**fmtspec)
#         elif from_typ in common.utypes and to_typ in common.iutypes:
#             return \
#             '''nsimd_{simd_ext}_v{to_typ}x2 ret;
#                ret.v0 = _mm_unpacklo{suf}({in0}, _mm_setzero_si128());
#                ret.v1 = _mm_unpackhi{suf}({in0}, _mm_setzero_si128());
#                return ret;'''.format(**fmtspec)
#         else:
#             return with_intrinsic
#     # elif simd_ext == 'sse42':
#     #    return with_intrinsic
#     elif simd_ext == 'avx':
#         if from_typ == 'i32' and to_typ == 'f64':
#             return with_intrinsic
#         else:
#             return split_trick
#     elif simd_ext == 'avx2':
#         return with_intrinsic
#     elif simd_ext == 'avx512_knl':
#         if from_typ in ['i16', 'u16', 'i32', 'u32', 'f32']:
#             return with_intrinsic
#         else:
#             return split_trick
#     else:
#         return with_intrinsic
#
# # -----------------------------------------------------------------------------
# # downconvert
#
# def downcvt1(opts, simd_ext, from_typ, to_typ):
#     # From f16 is easy
#     if from_typ == 'f16':
#         le_to_typ = int(fmtspec['le']) * 2
#         le_1f32 = le_to_typ // 4
#         le_2f32 = 2 * le_to_typ // 4
#         le_3f32 = 3 * le_to_typ // 4
#         cast = castsi(simd_ext, to_typ)
#         return \
#         '''{to_typ} dst[{le_to_typ}];
#            f32 src[{le_to_typ}];
#            int i;
#            {pre}storeu_ps(src, {in0}.v0);
#            {pre}storeu_ps(src + {le_1f32}, {in0}.v1);
#            {pre}storeu_ps(src + {le_2f32}, {in1}.v0);
#            {pre}storeu_ps(src + {le_3f32}, {in1}.v1);
#            for (i = 0; i < {le_to_typ}; i++) {{
#              dst[i] = ({to_typ})src[i];
#            }}
#            return {pre}loadu_si{nbits}({cast}dst);'''. \
#            format(le_to_typ=le_to_typ, le_1f32=le_1f32, le_2f32=le_2f32,
#                   le_3f32=le_3f32, cast=cast, **fmtspec)
#
#     # To f16 is easy
#     if to_typ == 'f16':
#         if from_typ == 'f32':
#             return \
#             '''nsimd_{simd_ext}_vf16 ret;
#                ret.v0 = {in0};
#                ret.v1 = {in1};
#                return ret;'''.format(**fmtspec)
#         else:
#             return \
#             '''nsimd_{simd_ext}_vf16 ret;
#                ret.v0 = nsimd_cvt_{simd_ext}_f32_{from_typ}({in0});
#                ret.v1 = nsimd_cvt_{simd_ext}_f32_{from_typ}({in1});
#                return ret;'''.format(**fmtspec)
#
#     # f64 --> f32 have intrinsics
#     if from_typ == 'f64' and to_typ == 'f32':
#         if simd_ext in sse:
#             return '''return _mm_movelh_ps(_mm_cvtpd_ps({in0}),
#                                            _mm_cvtpd_ps({in1}));'''. \
#                                            format(**fmtspec)
#         else:
#             return 'return {};'.format(setr(simd_ext, 'f32',
#                                 '{pre}cvtpd_ps({in0})'.format(**fmtspec),
#                                 '{pre}cvtpd_ps({in1})'.format(**fmtspec)))
#
#     # integer conversions intrinsics are only available with AVX-512
#     if simd_ext in avx512:
#         if (from_typ in ['i32', 'i64'] and to_typ in common.itypes) or \
#            (simd_ext == 'avx512_skylake' and from_typ == 'i16' and \
#             to_typ == 'i8'):
#             return 'return {};'.format(setr(simd_ext, to_typ,
#                    '{pre}cvtep{from_typ}_ep{to_typ}({in0})'.format(**fmtspec),
#                    '{pre}cvtep{from_typ}_ep{to_typ}({in1})'.format(**fmtspec)))
#         elif from_typ == 'i64' and to_typ == 'f32':
#             return 'return nsimd_cvt_{simd_ext}_f32_i32({});'. \
#                    format(setr(simd_ext, from_typ,
#                           '{pre}cvtepi64_epi32({in0})'.format(**fmtspec),
#                           '{pre}cvtepi64_epi32({in1})'.format(**fmtspec)),
#                           **fmtspec)
#
#     # and then emulation
#     le_to_typ = 2 * int(fmtspec['le'])
#     cast_src = '(__m{nbits}i *)'.format(**fmtspec) \
#                if from_typ in common.iutypes else ''
#     cast_dst = '(__m{nbits}i *)'.format(**fmtspec) \
#                if to_typ in common.iutypes else ''
#     return \
#     '''{to_typ} dst[{le_to_typ}];
#        {from_typ} src[{le_to_typ}];
#        int i;
#        {pre}storeu{sufsi}({cast_src}src, {in0});
#        {pre}storeu{sufsi}({cast_src}(src + {le}), {in1});
#        for (i = 0; i < {le_to_typ}; i++) {{
#          dst[i] = ({to_typ})src[i];
#        }}
#        return {pre}loadu{sufsi_to_typ}({cast_dst}dst);'''. \
#        format(cast_src=cast_src, cast_dst=cast_dst, le_to_typ=le_to_typ,
#               sufsi_to_typ=suf_si(simd_ext, to_typ), **fmtspec)
#

# # -----------------------------------------------------------------------------
# # to_mask
#
# def to_mask1(simd_ext, typ):
#     if typ == 'f16':
#         return '''nsimd_{simd_ext}_vf16 ret;
#                   ret.v0 = nsimd_to_mask_{simd_ext}_f32({in0}.v0);
#                   ret.v1 = nsimd_to_mask_{simd_ext}_f32({in0}.v1);
#                   return ret;'''.format(**fmtspec)
#     if simd_ext in sse + avx:
#         return 'return {in0};'.format(**fmtspec)
#     elif simd_ext == 'avx512_skylake':
#         if typ in common.iutypes:
#             return 'return _mm512_movm_epi{typnbits}({in0});'. \
#                    format(**fmtspec)
#         elif typ in ['f32', 'f64']:
#             return '''return _mm512_castsi512{suf}(
#                                _mm512_movm_epi{typnbits}({in0}));'''. \
#                                format(**fmtspec)
#     else:
#         if typ in ['i32', 'u32', 'i64', 'u64']:
#             return '''return _mm512_mask_mov{suf}(_mm512_setzero_si512(),
#                                  {in0}, _mm512_set1_epi32(-1));'''. \
#                                  format(**fmtspec)
#         elif typ in ['f32', 'f64']:
#             return '''return _mm512_mask_mov{suf}(_mm512_castsi512{suf}(
#                                _mm512_setzero_si512()), {in0},
#                                  _mm512_castsi512{suf}(
#                                    _mm512_set1_epi32(-1)));'''. \
#                                    format(**fmtspec)
#         else:
#             return '''{typ} buf[{le}];
#                       int i;
#                       for (i = 0; i < {le}; i++) {{
#                         if (({in0} >> i) & 1) {{
#                           buf[i] = ({typ})-1;
#                         }} else {{
#                           buf[i] = ({typ})0;
#                         }}
#                       }}
#                       return _mm512_loadu_si512(buf);'''.format(**fmtspec)
#
# # -----------------------------------------------------------------------------
# # to_logical
#
# def to_logical1(simd_ext, typ):
#     if typ in common.iutypes:
#         return '''return nsimd_ne_{simd_ext}_{typ}(
#                            {in0}, {pre}setzero{sufsi}());'''.format(**fmtspec)
#     elif typ in ['f32', 'f64']:
#         return '''return nsimd_reinterpretl_{simd_ext}_{typ}_{utyp}(
#                            nsimd_ne_{simd_ext}_{utyp}(
#                              {pre}cast{suf2}_si{nbits}({in0}),
#                                {pre}setzero_si{nbits}()));'''. \
#                                format(suf2=suf_si(simd_ext, typ)[1:],
#                                       utyp='u{}'.format(fmtspec['typnbits']),
#                                       **fmtspec)
#     else:
#         return '''nsimd_{simd_ext}_vlf16 ret;
#                   ret.v0 = nsimd_to_logical_{simd_ext}_f32({in0}.v0);
#                   ret.v1 = nsimd_to_logical_{simd_ext}_f32({in0}.v1);
#                   return ret;'''.format(**fmtspec)
#
# # -----------------------------------------------------------------------------
# # zip functions
#
# def zip_half(func, simd_ext, typ):
#     simd_ext2 = 'sse42' if simd_ext in avx else 'avx2'
#     if simd_ext in sse:
#         if typ == 'f16':
#             return '''nsimd_{simd_ext}_v{typ} ret;
#                       ret.v0 = _mm_unpacklo_ps({in0}.v{k}, {in1}.v{k});
#                       ret.v1 = _mm_unpackhi_ps({in0}.v{k}, {in1}.v{k});
#                       return ret;'''. \
#                       format(k='0' if func == 'ziplo' else '1', **fmtspec)
#         else:
#             return 'return {pre}unpack{lo}{suf}({in0}, {in1});'. \
#                    format(lo='lo' if func == 'ziplo' else 'hi', **fmtspec)
#     elif simd_ext in avx:
#         # Currently, 256 and 512 bits vectors are splitted into 128 bits
#         # vectors in order to perform the ziplo/hi operation using the
#         # unpacklo/hi sse operations.
#         if typ == 'f16':
#             in0vk = '{in0}.v{k}'.format(k='0' if func == 'ziplo' else '1',
#                                         **fmtspec)
#             in1vk = '{in1}.v{k}'.format(k='0' if func == 'ziplo' else '1',
#                                         **fmtspec)
#             return \
#             '''nsimd_{simd_ext}_v{typ} ret;
#                __m128 v_tmp0 = {get_low_in0vk};
#                __m128 v_tmp1 = {get_low_in1vk};
#                __m128 v_tmp2 = {get_high_in0vk};
#                __m128 v_tmp3 = {get_high_in1vk};
#                __m128 vres_lo0 = _mm_unpacklo_ps(v_tmp0, v_tmp1);
#                __m128 vres_hi0 = _mm_unpackhi_ps(v_tmp0, v_tmp1);
#                ret.v0 = {merge0};
#                __m128 vres_lo1 = _mm_unpacklo_ps(v_tmp2, v_tmp3);
#                __m128 vres_hi1 = _mm_unpackhi_ps(v_tmp2, v_tmp3);
#                ret.v1 = {merge1};
#                return ret;
#                '''.format(get_low_in0vk=extract(simd_ext, 'f32', LO, in0vk),
#                           get_low_in1vk=extract(simd_ext, 'f32', LO, in1vk),
#                           get_high_in0vk=extract(simd_ext, 'f32', HI, in0vk),
#                           get_high_in1vk=extract(simd_ext, 'f32', HI, in1vk),
#                           merge0=setr(simd_ext, 'f32', 'vres_lo0', 'vres_hi0'),
#                           merge1=setr(simd_ext, 'f32', 'vres_lo1', 'vres_hi1'),
#                           **fmtspec)
#         else:
#             hl = LO if func == 'ziplo' else HI
#             return \
#             '''{nat} v_tmp0 = {half_in0};
#                {nat} v_tmp1 = {half_in1};
#                {nat} vres_lo = _mm_unpacklo{suf}(v_tmp0, v_tmp1);
#                {nat} vres_hi = _mm_unpackhi{suf}(v_tmp0, v_tmp1);
#                return {merge};
#                '''.format(nat=get_native_typ(simd_ext2, typ),
#                           half_in0=extract(simd_ext, typ, hl, common.in0),
#                           half_in1=extract(simd_ext, typ, hl, common.in1),
#                           merge=setr(simd_ext, typ, 'vres_lo', 'vres_hi'),
#                           **fmtspec)
#     else:
#         if typ == 'f16':
#             return \
#             '''nsimd_{simd_ext}_v{typ} ret;
#                __m512 v0 = {in0}.v{k};
#                __m512 v1 = {in1}.v{k};
#                __m256 v_tmp0, v_tmp1, vres_lo, vres_hi;
#                /* Low part */
#                v_tmp0 = {low_v0};
#                v_tmp1 = {low_v1};
#                vres_lo = nsimd_ziplo_avx2_f32(v_tmp0, v_tmp1);
#                vres_hi = nsimd_ziphi_avx2_f32(v_tmp0, v_tmp1);
#                ret.v0 = {merge};
#                /* High part */
#                v_tmp0 = {high_v0};
#                v_tmp1 = {high_v1};
#                vres_lo = nsimd_ziplo_avx2_f32(v_tmp0, v_tmp1);
#                vres_hi = nsimd_ziphi_avx2_f32(v_tmp0, v_tmp1);
#                ret.v1 = {merge};
#                return ret;'''. \
#                format(k='0' if func == 'ziplo' else '1',
#                       low_v0=extract(simd_ext, 'f32', LO, 'v0'),
#                       low_v1=extract(simd_ext, 'f32', LO, 'v1'),
#                       high_v0=extract(simd_ext, 'f32', HI, 'v0'),
#                       high_v1=extract(simd_ext, 'f32', HI, 'v1'),
#                       merge=setr(simd_ext, 'f32', 'vres_lo', 'vres_hi'),
#                       **fmtspec)
#         else:
#             hl = LO if func == 'ziplo' else HI
#             return \
#             '''{nat} v_tmp0, v_tmp1;
#                v_tmp0 = {half_in0};
#                v_tmp1 = {half_in1};
#                {nat} vres_lo = nsimd_ziplo_avx2_{typ}(v_tmp0, v_tmp1);
#                {nat} vres_hi = nsimd_ziphi_avx2_{typ}(v_tmp0, v_tmp1);
#                return {merge};'''. \
#                format(nat=get_native_typ(simd_ext2, typ),
#                       half_in0=extract(simd_ext, typ, hl, common.in0),
#                       half_in1=extract(simd_ext, typ, hl, common.in1),
#                       merge=setr(simd_ext, typ, 'vres_lo', 'vres_hi'),
#                       **fmtspec)
#
# def zip(simd_ext, typ):
#     return '''nsimd_{simd_ext}_v{typ}x2 ret;
#               ret.v0 = nsimd_ziplo_{simd_ext}_{typ}({in0}, {in1});
#               ret.v1 = nsimd_ziphi_{simd_ext}_{typ}({in0}, {in1});
#               return ret;
#               '''.format(**fmtspec)
#
# # -----------------------------------------------------------------------------
# # unzip functions
#
# def unzip_half(opts, func, simd_ext, typ):
#     loop = '''{typ} tab[{lex2}];
#               {typ} res[{le}];
#               int i;
#               nsimd_storeu_{simd_ext}_{typ}(tab, {in0});
#               nsimd_storeu_{simd_ext}_{typ}(tab + {le}, {in1});
#               for(i = 0; i < {le}; i++) {{
#                 res[i] = tab[2 * i + {offset}];
#               }}
#               return nsimd_loadu_{simd_ext}_{typ}(res);
#               '''.format(lex2=2 * int(fmtspec['le']),
#                          offset='0' if func == 'unziplo' else '1', **fmtspec)
#
#     if simd_ext in sse:
#         if typ in ['f32', 'i32', 'u32']:
#             v0 = ('_mm_castsi128_ps({in0})' if typ in ['i32', 'u32'] \
#                                             else '{in0}').format(**fmtspec)
#             v1 = ('_mm_castsi128_ps({in1})' if typ in ['i32', 'u32'] \
#                                             else '{in1}').format(**fmtspec)
#             ret = ('_mm_castps_si128(v_res)' if typ in ['i32', 'u32'] \
#                                              else 'v_res').format(**fmtspec)
#             return '''__m128 v_res;
#                       v_res = _mm_shuffle_ps({v0}, {v1}, {mask});
#                       return {ret};'''.format(
#                       mask='_MM_SHUFFLE(2, 0, 2, 0)' if func == 'unziplo' \
#                       else '_MM_SHUFFLE(3, 1, 3, 1)',
#                       v0=v0, v1=v1, ret=ret, **fmtspec)
#         elif typ == 'f16':
#             return \
#             '''nsimd_{simd_ext}_v{typ} v_res;
#                v_res.v0 = _mm_shuffle_ps({in0}.v0, {in0}.v1, {mask});
#                v_res.v1 = _mm_shuffle_ps({in1}.v0, {in1}.v1, {mask});
#                return v_res;'''.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' \
#                                        if func == 'unziplo' \
#                                        else '_MM_SHUFFLE(3, 1, 3, 1)',
#                                        **fmtspec)
#         elif typ in ['f64', 'i64', 'u64']:
#             v0 = ('_mm_castsi128_pd({in0})' if typ in ['i64', 'u64'] \
#                                             else '{in0}').format(**fmtspec)
#             v1 = ('_mm_castsi128_pd({in1})' if typ in ['i64', 'u64'] \
#                                             else '{in1}').format(**fmtspec)
#             ret = ('_mm_castpd_si128(v_res)' if typ in ['i64', 'u64'] \
#                                              else 'v_res').format(**fmtspec)
#             return '''__m128d v_res;
#                       v_res = _mm_shuffle_pd({v0}, {v1}, {mask});
#                       return {ret};
#                       '''.format(mask='0' if func == 'unziplo' else '3',
#                                  v0=v0, v1=v1, ret=ret, **fmtspec)
#         elif typ in ['i16', 'u16']:
#             return '''__m128i v_tmp0 = _mm_shufflelo_epi16(
#                                            {in0}, _MM_SHUFFLE(3, 1, 2, 0));
#                       v_tmp0 = _mm_shufflehi_epi16(v_tmp0,
#                                    _MM_SHUFFLE(3, 1, 2, 0));
#                       __m128i v_tmp1 = _mm_shufflelo_epi16({in1},
#                                    _MM_SHUFFLE(3, 1, 2, 0));
#                       v_tmp1 = _mm_shufflehi_epi16(v_tmp1,
#                                    _MM_SHUFFLE(3, 1, 2, 0));
#                       __m128 v_res = _mm_shuffle_ps(_mm_castsi128_ps(v_tmp0),
#                                          _mm_castsi128_ps(v_tmp1), {mask});
#                       return _mm_castps_si128(v_res);
#                       '''.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' \
#                                  if func == 'unziplo' \
#                                  else '_MM_SHUFFLE(3, 1, 3, 1)', **fmtspec)
#         else:
#             return loop
#     elif simd_ext in avx:
#         ret_template = \
#         '''v_tmp0 = _mm256_permute2f128_{t}({v0}, {v0}, 0x01);
#            v_tmp0 = _mm256_shuffle_{t}({v0}, v_tmp0, {mask});
#            v_tmp1 = _mm256_permute2f128_{t}({v1}, {v1}, 0x01);
#            v_tmp1 = _mm256_shuffle_{t}({v1}, v_tmp1, {mask});
#            v_res  = _mm256_permute2f128_{t}(v_tmp0, v_tmp1, 0x20);
#            {ret} = {v_res};'''
#         if typ in ['f32', 'i32', 'u32']:
#             v0 = '_mm256_castsi256_ps({in0})' \
#                  if typ in ['i32', 'u32'] else '{in0}'
#             v1 = '_mm256_castsi256_ps({in1})' \
#                  if typ in ['i32', 'u32'] else '{in1}'
#             v_res = '_mm256_castps_si256(v_res)' \
#                     if typ in ['i32', 'u32'] else 'v_res'
#             ret = 'ret'
#             src = ret_template.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' \
#                       if func == 'unziplo' else '_MM_SHUFFLE(3, 1, 3, 1)',
#                       v0=v0, v1=v1, v_res=v_res, ret=ret, t='ps', **fmtspec)
#             return '''nsimd_{simd_ext}_v{typ} ret;
#                       __m256 v_res, v_tmp0, v_tmp1;
#                       {src}
#                       return ret;'''. \
#                       format(src=src.format(**fmtspec), **fmtspec)
#         elif typ == 'f16':
#             src0 = ret_template.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' \
#                        if func == 'unziplo' else '_MM_SHUFFLE(3, 1, 3, 1)',
#                        v0='{in0}.v0', v1='{in0}.v1', v_res='v_res',
#                        ret='ret.v0', t='ps')
#             src1 = ret_template.format(mask='_MM_SHUFFLE(2, 0, 2, 0)' \
#                        if func == 'unziplo' else '_MM_SHUFFLE(3, 1, 3, 1)',
#                        v0='{in1}.v0', v1='{in1}.v1', v_res='v_res',
#                        ret='ret.v1', t='ps')
#             return '''nsimd_{simd_ext}_v{typ} ret;
#                       __m256 v_res, v_tmp0, v_tmp1;
#                       {src0}
#                       {src1}
#                       return ret;'''.format(src0=src0.format(**fmtspec),
#                                             src1=src1.format(**fmtspec),
#                                             **fmtspec)
#         elif typ in ['f64', 'i64', 'u64']:
#             v0 = ('_mm256_castsi256_pd({in0})' \
#                       if typ in ['i64', 'u64'] else '{in0}').format(**fmtspec)
#             v1 = ('_mm256_castsi256_pd({in1})' \
#                       if typ in ['i64', 'u64'] else '{in1}').format(**fmtspec)
#             v_res = ('_mm256_castpd_si256(v_res)' \
#                          if typ in ['i64', 'u64'] else 'v_res'). \
#                          format(**fmtspec)
#             src = ret_template.format(mask='0x00' if func == 'unziplo' \
#                       else '0x03', v0=v0, v1=v1, ret='ret', v_res=v_res,
#                       t='pd')
#             return '''nsimd_{simd_ext}_v{typ} ret;
#                       __m256d v_res, v_tmp0, v_tmp1;
#                       {src}
#                       return ret;'''.format(src=src.format(**fmtspec),
#                                             **fmtspec)
#         elif typ in ['i16', 'u16']:
#             return \
#             '''__m128i v_tmp0_hi = {hi0};
#                __m128i v_tmp0_lo = {lo0};
#                __m128i v_tmp1_hi = {hi1};
#                __m128i v_tmp1_lo = {lo1};
#                v_tmp0_lo = nsimd_{func}_sse2_{typ}(v_tmp0_lo, v_tmp0_hi);
#                v_tmp1_lo = nsimd_{func}_sse2_{typ}(v_tmp1_lo, v_tmp1_hi);
#                return {merge};'''. \
#                format(hi0=extract(simd_ext, typ, HI, common.in0),
#                       lo0=extract(simd_ext, typ, LO, common.in0),
#                       hi1=extract(simd_ext, typ, HI, common.in1),
#                       lo1=extract(simd_ext, typ, LO, common.in1),
#                       merge=setr(simd_ext, typ, 'v_tmp0_lo', 'v_tmp1_lo'),
#                       func=func, **fmtspec)
#         else:
#             return loop
#     else:
#         if typ == 'f16':
#             return \
#             '''nsimd_{simd_ext}_v{typ} ret;
#                __m256 v_tmp0, v_tmp1, v_res_lo, v_res_hi;
#                v_tmp0 = {loin0v0};
#                v_tmp1 = {hiin0v0};
#                v_res_lo = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
#                v_tmp0 = {loin0v1};
#                v_tmp1 = {hiin0v1};
#                v_res_hi = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
#                ret.v0 = {merge};
#                v_tmp0 = {loin1v0};
#                v_tmp1 = {hiin1v0};
#                v_res_lo = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
#                v_tmp0 = {loin1v1};
#                v_tmp1 = {hiin1v1};
#                v_res_hi = nsimd_{func}_avx2_f32(v_tmp0, v_tmp1);
#                ret.v1 = {merge};
#                return ret;'''.format(
#                    loin0v0=extract(simd_ext, 'f32', LO, common.in0 + '.v0'),
#                    hiin0v0=extract(simd_ext, 'f32', HI, common.in0 + '.v0'),
#                    loin0v1=extract(simd_ext, 'f32', LO, common.in0 + '.v1'),
#                    hiin0v1=extract(simd_ext, 'f32', HI, common.in0 + '.v1'),
#                    loin1v0=extract(simd_ext, 'f32', LO, common.in1 + '.v0'),
#                    hiin1v0=extract(simd_ext, 'f32', HI, common.in1 + '.v0'),
#                    loin1v1=extract(simd_ext, 'f32', LO, common.in1 + '.v1'),
#                    hiin1v1=extract(simd_ext, 'f32', HI, common.in1 + '.v1'),
#                    merge=setr(simd_ext, 'f32', 'v_res_lo', 'v_res_hi'),
#                    func=func, **fmtspec)
#         else:
#             return '''nsimd_avx2_v{typ} v00 = {extract_lo0};
#                       nsimd_avx2_v{typ} v01 = {extract_hi0};
#                       nsimd_avx2_v{typ} v10 = {extract_lo1};
#                       nsimd_avx2_v{typ} v11 = {extract_hi1};
#                       v00 = nsimd_{func}_avx2_{typ}(v00, v01);
#                       v01 = nsimd_{func}_avx2_{typ}(v10, v11);
#                       return {merge};'''.format(
#                           func=func,
#                           extract_lo0=extract(simd_ext, typ, LO, common.in0),
#                           extract_lo1=extract(simd_ext, typ, LO, common.in1),
#                           extract_hi0=extract(simd_ext, typ, HI, common.in0),
#                           extract_hi1=extract(simd_ext, typ, HI, common.in1),
#                           merge=setr(simd_ext, typ, 'v00', 'v01'), **fmtspec)
#
# def unzip(simd_ext, typ):
#     return '''nsimd_{simd_ext}_v{typ}x2 ret;
#               ret.v0 = nsimd_unziplo_{simd_ext}_{typ}({in0}, {in1});
#               ret.v1 = nsimd_unziphi_{simd_ext}_{typ}({in0}, {in1});
#               return ret;'''.format(**fmtspec)
#
# -----------------------------------------------------------------------------
# mask_for_loop_tail
@printf2
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
#
# def iota(simd_ext, typ):
#     typ2 = 'f32' if typ == 'f16' else typ
#     iota = ', '.join(['({typ2}){i}'.format(typ2=typ2, i=i) \
#                       for i in range(int(fmtspec['le']))])
#     if typ == 'f16':
#         return '''f32 buf[{le}] = {{ {iota} }};
#                   nsimd_{simd_ext}_vf16 ret;
#                   ret.v0 = {pre}loadu_ps(buf);
#                   ret.v1 = {pre}loadu_ps(buf + {le2});
#                   return ret;'''. \
#                   format(iota=iota, le2=fmtspec['le'] // 2, **fmtspec)
#     return '''{typ} buf[{le}] = {{ {iota} }};
#               return {pre}loadu{sufsi}({cast}buf);'''. \
#               format(iota=iota, cast='(__m{nbits}i*)'.format(**fmtspec) \
#                                 if typ in common.iutypes else '', **fmtspec)
#
# # -----------------------------------------------------------------------------
# # scatter
#
# def scatter(simd_ext, typ):
#     if typ == 'f16':
#         return '''int i;
#                   f32 buf[{le}];
#                   i16 offset_buf[{le}];
#                   {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in1});
#                   {pre}storeu_ps(buf, {in2}.v0);
#                   {pre}storeu_ps(buf + {leo2}, {in2}.v1);
#                   for (i = 0; i < {le}; i++) {{
#                     {in0}[offset_buf[i]] = nsimd_f32_to_f16(buf[i]);
#                   }}'''.format(leo2=int(fmtspec['le']) // 2, **fmtspec)
#     if simd_ext in (sse + avx) or typ in ['i8', 'u8', 'i16', 'u16']:
#         cast = castsi(simd_ext, typ)
#         return '''int i;
#                   {typ} buf[{le}];
#                   {ityp} offset_buf[{le}];
#                   {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in1});
#                   {pre}storeu{sufsi}({cast}buf, {in2});
#                   for (i = 0; i < {le}; i++) {{
#                     {in0}[offset_buf[i]] = buf[i];
#                   }}'''.format(ityp='i' + typ[1:], cast=cast, **fmtspec)
#     # getting here means 32 and 64-bits types for avx512
#     return '''{pre}i{typnbits}scatter{suf}(
#                   (void *){in0}, {in1}, {in2}, {scale});'''. \
#                   format(scale=int(typ[1:]) // 8, **fmtspec)
#
# # -----------------------------------------------------------------------------
# # linear scatter
#
# def scatter_linear(simd_ext, typ):
#     if typ == 'f16':
#         return '''int i;
#                   f32 buf[{le}];
#                   {pre}storeu_ps(buf, {in2}.v0);
#                   {pre}storeu_ps(buf + {leo2}, {in2}.v1);
#                   for (i = 0; i < {le}; i++) {{
#                     {in0}[i * {in1}] = nsimd_f32_to_f16(buf[i]);
#                   }}'''.format(leo2=int(fmtspec['le']) // 2, **fmtspec)
#     if simd_ext in avx512:
#         return '''nsimd_scatter_linear_avx2_{typ}({in0}, {in1}, {lo});
#                   nsimd_scatter_linear_avx2_{typ}({in0} + ({leo2} * {in1}),
#                                                   {in1}, {hi});'''. \
#                   format(leo2=int(fmtspec['le']) // 2,
#                          lo=extract(simd_ext, typ, LO, fmtspec['in2']),
#                          hi=extract(simd_ext, typ, HI, fmtspec['in2']),
#                          **fmtspec)
#     emulation = '''int i;
#                    {typ} buf[{le}];
#                    {pre}storeu{sufsi}({cast}buf, {in2});
#                    for (i = 0; i < {le}; i++) {{
#                      {in0}[i * {in1}] = buf[i];
#                    }}'''.format(cast=castsi(simd_ext, typ), **fmtspec)
#     if (simd_ext == 'sse2' and typ in ['i16', 'u16']) or \
#        (simd_ext == 'avx' and \
#         typ in ['i32', 'u32', 'f32', 'i64', 'u64', 'f64']) or \
#        (simd_ext in ['sse42', 'avx2']):
#         trick = '\n'.join([
#         '{in0}[{i} * {in1}] = {get_lane};'.format(i=i,
#         get_lane=get_lane(simd_ext, typ, '{in2}'.format(**fmtspec), i),
#         **fmtspec) for i in range(int(fmtspec['le']))])
#         return '''#if NSIMD_WORD_SIZE == 32
#                     {}
#                   #else
#                     {}
#                   #endif'''.format(emulation, trick)
#     else:
#         return emulation
#
# # -----------------------------------------------------------------------------
# # mask_scatter
#
# def mask_scatter(simd_ext, typ):
#     if typ == 'f16':
#         le2 = fmtspec['le'] // 2
#         if simd_ext in sse + avx:
#             store_mask = '''{pre}storeu_ps(mask, {in0}.v0);
#                             {pre}storeu_ps(mask + {le2}, {in0}.v1);'''. \
#                             format(le2=le2, **fmtspec)
#         else:
#             store_mask = '''_mm512_storeu_ps(mask, _mm512_maskz_mov_ps(
#                               {in0}.v0, _mm512_set1_ps(1.0f)));
#                             _mm512_storeu_ps(mask + {le2}, _mm512_maskz_mov_ps(
#                               {in0}.v1, _mm512_set1_ps(1.0f)));'''. \
#                             format(le2=le2, **fmtspec)
#         return '''int i;
#                   f32 mask[{le}], buf[{le}];
#                   i16 offset_buf[{le}];
#                   {store_mask}
#                   {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in2});
#                   {pre}storeu_ps(buf, {in3}.v0);
#                   {pre}storeu_ps(buf + {le2}, {in3}.v1);
#                   for (i = 0; i < {le}; i++) {{
#                     if (nsimd_scalar_reinterpret_u32_f32(mask[i]) != (u32)0) {{
#                       {in1}[offset_buf[i]] = nsimd_f32_to_f16(buf[i]);
#                     }}
#                   }}'''.format(le2=le2, store_mask=store_mask, **fmtspec)
#     if simd_ext in (sse + avx) or typ in ['i8', 'u8', 'i16', 'u16']:
#         cast = castsi(simd_ext, typ)
#         if simd_ext in avx512:
#             mask_decl = 'u64 mask;'
#             store_mask = 'mask = (u64){in0};'.format(**fmtspec)
#             cond = '(mask >> i) & 1'
#         else:
#             mask_decl = '{typ} mask[{le}];'.format(**fmtspec)
#             store_mask = '{pre}storeu{sufsi}({cast}mask, {in0});'. \
#                          format(cast=cast, **fmtspec)
#             cond = 'nsimd_scalar_reinterpret_{utyp}_{typ}(mask[i]) != '\
#                    '({utyp})0'.format(utyp='u' + typ[1:], **fmtspec)
#         return '''int i;
#                   {typ} buf[{le}];
#                   {mask_decl}
#                   {ityp} offset_buf[{le}];
#                   {store_mask}
#                   {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in2});
#                   {pre}storeu{sufsi}({cast}buf, {in3});
#                   for (i = 0; i < {le}; i++) {{
#                     if ({cond}) {{
#                       {in1}[offset_buf[i]] = buf[i];
#                     }}
#                   }}'''.format(ityp='i' + typ[1:], cast=cast, cond=cond,
#                                mask_decl=mask_decl, store_mask=store_mask,
#                                **fmtspec)
#     # getting here means 32 and 64-bits types for avx512
#     return '''{pre}mask_i{typnbits}scatter{suf}(
#                   (void *){in1}, {in0}, {in2}, {in3}, {scale});'''. \
#                   format(scale=int(typ[1:]) // 8, **fmtspec)
#
# # -----------------------------------------------------------------------------
# # gather
#
# def gather(simd_ext, typ):
#     if typ == 'f16':
#         return '''nsimd_{simd_ext}_vf16 ret;
#                   int i;
#                   f32 buf[{le}];
#                   i16 offset_buf[{le}];
#                   {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in1});
#                   for (i = 0; i < {le}; i++) {{
#                     buf[i] = nsimd_f16_to_f32({in0}[offset_buf[i]]);
#                   }}
#                   ret.v0 = {pre}loadu_ps(buf);
#                   ret.v1 = {pre}loadu_ps(buf + {leo2});
#                   return ret;'''.format(leo2=int(fmtspec['le']) // 2,
#                                         **fmtspec)
#     if simd_ext in (sse + ['avx']) or typ in ['i8', 'u8', 'i16', 'u16']:
#         cast = castsi(simd_ext, typ)
#         return '''int i;
#                   {typ} buf[{le}];
#                   {ityp} offset_buf[{le}];
#                   {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in1});
#                   for (i = 0; i < {le}; i++) {{
#                     buf[i] = {in0}[offset_buf[i]];
#                   }}
#                   return {pre}loadu{sufsi}({cast}buf);'''. \
#                   format(ityp='i' + typ[1:], cast=cast, **fmtspec)
#     # getting here means 32 and 64-bits types for avx2 and avx512
#     if simd_ext == 'avx2':
#         if typ in ['i64', 'u64']:
#             cast = '(nsimd_longlong *)'
#         elif typ in ['i32', 'u32']:
#             cast = '(int *)'
#         else:
#             cast = '({typ} *)'.format(**fmtspec)
#         return '''return {pre}i{typnbits}gather{suf}(
#                              {cast}{in0}, {in1}, {scale});'''. \
#                              format(scale=int(typ[1:]) // 8, cast=cast,
#                                     **fmtspec)
#     elif simd_ext in avx512:
#         return 'return {pre}i{typnbits}gather{suf}({in1}, ' \
#                       '(const void *){in0}, {scale});'. \
#                       format(scale=int(typ[1:]) // 8, **fmtspec)
#
# # -----------------------------------------------------------------------------
# # linear gather
#
# def gather_linear(simd_ext, typ):
#     le = int(fmtspec['le'])
#     cast = castsi(simd_ext, typ)
#     if typ == 'f16':
#         return '''nsimd_{simd_ext}_vf16 ret;
#                   f32 buf[{le}];
#                   int i;
#                   for (i = 0; i < {le}; i++) {{
#                     buf[i] = nsimd_f16_to_f32({in0}[i * {in1}]);
#                   }}
#                   ret.v0 = {pre}loadu_ps(buf);
#                   ret.v1 = {pre}loadu_ps(buf + {leo2});
#                   return ret;'''.format(leo2=le // 2, **fmtspec)
#     emulation = '''{typ} buf[{le}];
#                    int i;
#                    for (i = 0; i < {le}; i++) {{
#                      buf[i] = {in0}[i * {in1}];
#                    }}
#                    return {pre}loadu{sufsi}({cast}buf);'''. \
#                    format(cast=cast, **fmtspec)
#     if simd_ext == 'sse2' and typ not in ['i16', 'u16']:
#         return emulation
#     if simd_ext in sse + avx:
#         trick = \
#         '''nsimd_{simd_ext}_v{typ} ret;
#            ret = {pre}undefined{sufsi}();
#            '''.format(**fmtspec) + ''.join([
#            set_lane(simd_ext, typ, 'ret', '{in0}[{i} * {in1}]'. \
#                                           format(i=i, **fmtspec), i) + '\n' \
#                                           for i in range(le)]) + \
#         '''return ret;'''
#         return '''#if NSIMD_WORD_SIZE == 32
#                     {}
#                   #else
#                     {}
#                   #endif
#                   '''.format(emulation, trick)
#     # getting here means AVX-512
#     return \
#     '''nsimd_avx2_v{typ} lo = _mm256_undefined{sufsi2}();
#        nsimd_avx2_v{typ} hi = _mm256_undefined{sufsi2}();
#        lo = nsimd_gather_linear_avx2_{typ}({in0}, {in1});
#        hi = nsimd_gather_linear_avx2_{typ}({in0} + ({leo2} * {in1}), {in1});
#        return {merge};'''.format(merge=setr(simd_ext, typ, 'lo', 'hi'),
#                                  sufsi2=suf_si('avx2', typ),
#                                  leo2=le // 2, **fmtspec)
#
# # -----------------------------------------------------------------------------
# # maksed gather
#
# def maskoz_gather(oz, simd_ext, typ):
#     if typ == 'f16':
#         le2 = fmtspec['le'] // 2
#         if simd_ext in sse + avx:
#             store_mask = '''{pre}storeu_ps(mask, {in0}.v0);
#                             {pre}storeu_ps(mask + {le2}, {in0}.v1);'''. \
#                             format(le2=le2, **fmtspec)
#         else:
#             store_mask = '''_mm512_storeu_ps(mask, _mm512_maskz_mov_ps(
#                               {in0}.v0, _mm512_set1_ps(1.0f)));
#                             _mm512_storeu_ps(mask + {le2}, _mm512_maskz_mov_ps(
#                               {in0}.v1, _mm512_set1_ps(1.0f)));'''. \
#                             format(le2=le2, **fmtspec)
#         if oz == 'z':
#             store_oz = '''{pre}storeu_ps(buf, {pre}setzero_ps());
#                           {pre}storeu_ps(buf + {le2}, {pre}setzero_ps());'''. \
#                           format(le2=le2, **fmtspec)
#         else:
#             store_oz = '''{pre}storeu_ps(buf, {in3}.v0);
#                           {pre}storeu_ps(buf + {le2}, {in3}.v1);'''. \
#                           format(le2=le2, **fmtspec)
#         return '''nsimd_{simd_ext}_vf16 ret;
#                   int i;
#                   f32 buf[{le}], mask[{le}];
#                   i16 offset_buf[{le}];
#                   {store_mask}
#                   {store_oz}
#                   {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in2});
#                   for (i = 0; i < {le}; i++) {{
#                     if (nsimd_scalar_reinterpret_u32_f32(mask[i]) != (u32)0) {{
#                       buf[i] = nsimd_f16_to_f32({in1}[offset_buf[i]]);
#                     }}
#                   }}
#                   ret.v0 = {pre}loadu_ps(buf);
#                   ret.v1 = {pre}loadu_ps(buf + {leo2});
#                   return ret;'''.format(leo2=le2, store_mask=store_mask,
#                                         store_oz=store_oz, **fmtspec)
#     if simd_ext in (sse + ['avx']) or typ in ['i8', 'u8', 'i16', 'u16']:
#         cast = castsi(simd_ext, typ)
#         if simd_ext in sse + avx:
#             mask_decl = '{typ} mask[{le}];'.format(**fmtspec)
#             store_mask = '{pre}storeu{sufsi}({cast}mask, {in0});'. \
#                          format(cast=cast, **fmtspec)
#             if typ in common.iutypes:
#                 comp = 'mask[i]'
#             else:
#                 comp = 'nsimd_scalar_reinterpret_u{typnbits}_{typ}(mask[i])'. \
#                        format(**fmtspec)
#         else:
#             mask_decl = 'u64 mask;'
#             store_mask = 'mask = (u64){in0};'.format(**fmtspec)
#             comp = '(mask >> i) & 1'
#         if oz == 'z':
#             store_oz = '''{pre}storeu{sufsi}({cast}buf,
#                                              {pre}setzero{sufsi}());'''. \
#                                              format(cast=cast, **fmtspec)
#         else:
#             store_oz = '{pre}storeu{sufsi}({cast}buf, {in3});'. \
#                        format(cast=cast, **fmtspec)
#         return '''int i;
#                   {typ} buf[{le}];
#                   {mask_decl}
#                   {ityp} offset_buf[{le}];
#                   {store_mask}
#                   {store_oz}
#                   {pre}storeu_si{nbits}((__m{nbits}i *)offset_buf, {in2});
#                   for (i = 0; i < {le}; i++) {{
#                     if ({comp}) {{
#                       buf[i] = {in1}[offset_buf[i]];
#                     }}
#                   }}
#                   return {pre}loadu{sufsi}({cast}buf);'''. \
#                   format(ityp='i' + typ[1:], cast=cast, store_mask=store_mask,
#                          store_oz=store_oz, comp=comp, mask_decl=mask_decl,
#                          **fmtspec)
#     # getting here means 32 and 64-bits types for avx2 and avx512
#     if oz == 'o':
#         src = '{in3}'.format(**fmtspec)
#     else:
#         src = '{pre}setzero{sufsi}()'.format(**fmtspec)
#     if simd_ext == 'avx2':
#         if typ in ['i64', 'u64']:
#             cast = '(nsimd_longlong *)'
#         elif typ in ['i32', 'u32']:
#             cast = '(int *)'
#         else:
#             cast = '({typ} *)'.format(**fmtspec)
#         return '''return {pre}mask_i{typnbits}gather{suf}({src},
#                              {cast}{in1}, {in2}, {in0}, {scale});'''. \
#                              format(scale=int(typ[1:]) // 8, cast=cast,
#                                     src=src, **fmtspec)
#     elif simd_ext in avx512:
#         return 'return {pre}mask_i{typnbits}gather{suf}({src}, {in0}, ' \
#                       '{in2}, (const void *){in1}, {scale});'. \
#                       format(src=src, scale=int(typ[1:]) // 8, **fmtspec)

@printf2
def void(simd_ext, typ, func):
    """
    in wasm, you almost cannot know what operation wasn't implemented with abort
    so here is this function when not implemented
    """
    return f"""
    printf("ATTENTION, VOID HAS BEEN CALLED\\n");
           printf("func not implemented: {func}\\n");
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
        # 'andl': lambda: binlop2('andl', simd_ext, from_typ),
        # 'xorl': lambda: binlop2('xorl', simd_ext, from_typ),
        # 'orl': lambda: binlop2('orl', simd_ext, from_typ),
        'notb': lambda: comp_bin("not", simd_ext, from_typ),
        'notl': lambda: lnot1(simd_ext, from_typ),
        'andnotb': lambda: comp_bin("andnot", simd_ext, from_typ),
        'andnotl': lambda: comp("andnot", simd_ext, from_typ),
        'add': lambda: addsub('add', simd_ext, from_typ),
        'sub': lambda: addsub('sub', simd_ext, from_typ),
        'adds': lambda: addsub('adds', simd_ext, from_typ),
        'subs': lambda: addsub('subs', simd_ext, from_typ),
        # 'div': lambda: div2(opts, simd_ext, from_typ),
        'sqrt': lambda: sqrt1(simd_ext, from_typ),
        'len': lambda: len1(simd_ext, from_typ),
        # 'mul': lambda: mul2(opts, simd_ext, from_typ),
        'shl': lambda: shl_shr('shl', simd_ext, from_typ),
        'shr': lambda: shl_shr('shr', simd_ext, from_typ),
        # 'shra': lambda: shra(opts, simd_ext, from_typ),
        'set1': lambda: set1(simd_ext, from_typ),
        'set1l': lambda: set1l(simd_ext, from_typ),
        'eq': lambda: comp("eq", simd_ext, from_typ),
        'ne': lambda: comp("ne", simd_ext, from_typ),
        'gt': lambda: comp("gt", simd_ext, from_typ),
        'lt': lambda: comp("lt", simd_ext, from_typ),
        'ge': lambda: comp("ge", simd_ext, from_typ),
        'le': lambda: comp("le", simd_ext, from_typ),
        'if_else1': lambda: if_else1(simd_ext, from_typ),
        'min': lambda: minmax('min', simd_ext, from_typ),
        'max': lambda: minmax('max', simd_ext, from_typ),
        'loadla': lambda: loadl(simd_ext, from_typ, True),
        'loadlu': lambda: loadl(simd_ext, from_typ, False),
        'storela': lambda: storel(simd_ext, from_typ, True),
        'storelu': lambda: storel(simd_ext, from_typ, False),
        'abs': lambda: abs1(simd_ext, from_typ),
        # 'fma': lambda: fma_fms('fma', simd_ext, from_typ),
        # 'fnma': lambda: fma_fms('fnma', simd_ext, from_typ),
        # 'fms': lambda: fma_fms('fms', simd_ext, from_typ),
        # 'fnms': lambda: fma_fms('fnms', simd_ext, from_typ),
        # 'ceil': lambda: round1(opts, 'ceil', simd_ext, from_typ),
        # 'floor': lambda: round1(opts, 'floor', simd_ext, from_typ),
        # 'trunc': lambda: trunc1(opts, simd_ext, from_typ),
        # 'round_to_even': lambda: round_to_even1(opts, simd_ext, from_typ),
        'all': lambda: all_any('all', simd_ext, from_typ),
        'any': lambda: all_any('any', simd_ext, from_typ),
        # 'reinterpret': lambda: reinterpret1(simd_ext, from_typ, to_typ),
        # 'reinterpretl': lambda: reinterpretl1(simd_ext, from_typ, to_typ),
        # 'cvt': lambda: convert1(simd_ext, from_typ, to_typ),
        # 'rec11': lambda: rec11_rsqrt11('rcp', simd_ext, from_typ),
        # 'rec8': lambda: rec11_rsqrt11('rcp', simd_ext, from_typ),
        # 'rsqrt11': lambda: rec11_rsqrt11('rsqrt', simd_ext, from_typ),
        # 'rsqrt8': lambda: rec11_rsqrt11('rsqrt', simd_ext, from_typ),
        # 'rec': lambda: rec1(simd_ext, from_typ),
        # 'neg': lambda: neg1(simd_ext, from_typ),
        # 'nbtrue': lambda: nbtrue1(simd_ext, from_typ),
        # 'reverse': lambda: reverse1(simd_ext, from_typ),
        # 'addv': lambda: addv(simd_ext, from_typ),
        # 'upcvt': lambda: upcvt1(simd_ext, from_typ, to_typ),
        # 'downcvt': lambda: downcvt1(opts, simd_ext, from_typ, to_typ),
        # 'to_mask': lambda: to_mask1(simd_ext, from_typ),
        # 'to_logical': lambda: to_logical1(simd_ext, from_typ),
        # 'ziplo': lambda: zip_half('ziplo', simd_ext, from_typ),
        # 'ziphi': lambda: zip_half('ziphi', simd_ext, from_typ),
        # 'unziplo': lambda: unzip_half(opts, 'unziplo', simd_ext, from_typ),
        # 'unziphi': lambda: unzip_half(opts, 'unziphi', simd_ext, from_typ),
        # 'zip' : lambda : zip(simd_ext, from_typ),
        # 'unzip' : lambda : unzip(simd_ext, from_typ),
        'mask_for_loop_tail': lambda: mask_for_loop_tail(simd_ext, from_typ),
        # 'iota': lambda : iota(simd_ext, from_typ)
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
