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

# This file gives the implementation for the Power PC platform.
# This script tries to be as readable as possible. It implements VMX and VSX.

# Documentation found from:
# https://www.nxp.com/docs/en/reference-manual/ALTIVECPIM.pdf
# https://www.ibm.com/docs/en/xl-c-and-cpp-linux/13.1.6?topic=functions-vector-built-in
# https://gcc.gnu.org/onlinedocs/gcc-9.1.0/gcc/PowerPC-AltiVec-Built-in-Functions-Available-on-ISA-2_002e06.html

import common

fmtspec = {}

# -----------------------------------------------------------------------------
# Helpers

def has_to_be_emulated(simd_ext, typ):
    if typ == 'f16':
        return True
    if simd_ext == 'vmx' and typ in ['f64', 'i64', 'u64']:
        return True
    return False

# Returns the power pc type corresponding to the nsimd type
def native_type(typ):
    if typ == 'u8':
        return '__vector unsigned char'
    elif typ == 'i8':
        return '__vector signed char'
    elif typ == 'u16':
        return '__vector unsigned short'
    elif typ == 'i16':
        return '__vector signed short'
    elif typ == 'u32':
        return '__vector unsigned int'
    elif typ == 'u64':
        return '__vector unsigned long long'
    elif typ == 'i32':
        return '__vector signed int'
    elif typ == 'i64':
        return '__vector signed long long'
    elif typ == 'f32':
        return '__vector float'
    elif typ == 'f64':
        return '__vector double'
    else:
        raise ValueError('Type "{}" not supported'.format(typ))

# Returns the logical power pc type corresponding to the nsimd type
def native_typel(typ):
    if typ in ['i8', 'u8']:
        return '__vector __bool char'
    elif typ in ['i16', 'u16']:
        return '__vector __bool short'
    elif typ in ['i32', 'u32', 'f32']:
        return '__vector __bool int'
    elif typ in ['f64', 'i64', 'u64']:
        return '__vector __bool long long'
    else:
        raise ValueError('Type "{}" not supported'.format(typ))

# Length of a vector with elements of type typ
def get_len(typ):
    return 128 // int(typ[1:])

# Emulate 64 bits types for vmx only
def emulate_64(op, typ, params):
    def arg(param, i):
        if param == 'v':
            return '{}.v{{i}}'.format(common.get_arg(i))
        elif param == 'l':
            return '(int)({}.v{{i}} & ((u64)1))'.format(common.get_arg(i))
        else:
            return common.get_arg(i)
    args = ', '.join(arg(params[i + 1], i) for i in range(len(params[1:])))
    args0 = args.format(i=0)
    args1 = args.format(i=1)
    if params[0] == 'v':
        return '''nsimd_vmx_v{typ} ret;
                  ret.v0 = nsimd_scalar_{op}_{typ}({args0});
                  ret.v1 = nsimd_scalar_{op}_{typ}({args1});
                  return ret;'''. \
                  format(typ=typ, op=op, args0=args0, args1=args1)
    else:
        return \
        '''nsimd_vmx_vl{typ} ret;
           ret.v0 = (u64)(nsimd_scalar_{op}{suf}({args0}) ? -1 : 0);
           ret.v1 = (u64)(nsimd_scalar_{op}{suf}({args1}) ? -1 : 0);
           return ret;'''. \
           format(suf='' if params == ['l'] * len(params) else '_' + typ,
                  typ=typ, op=op, args0=args0, args1=args1)

def emulate_f16(op, simd_ext, params):
    tmpl = ', '.join(['{{in{}}}.v{{{{i}}}}'.format(i).format(**fmtspec) \
                      for i in range(len(params[1:]))])
    args1 = tmpl.format(i=0)
    args2 = tmpl.format(i=1)
    l = 'l' if params[0] == 'l' else ''
    return '''nsimd_{simd_ext}_v{l}f16 ret;
              ret.v0 = nsimd_{op}_{simd_ext}_f32({args1});
              ret.v1 = nsimd_{op}_{simd_ext}_f32({args2});
              return ret;'''. \
              format(l=l, op=op, args1=args1, args2=args2, **fmtspec)

def emulation_code(op, simd_ext, typ, params):
    if typ == 'f16':
        return emulate_f16(op, simd_ext, params)
    elif simd_ext == 'vmx' and typ in ['f64', 'i64', 'u64']:
        return emulate_64(op, typ, params)
    else:
        raise ValueError('Automatic emulation for {}/{}/{} is not supported'. \
                         format(func, simd_ext, typ))

# -----------------------------------------------------------------------------
# Implementation of mandatory functions for this module

def emulate_fp16(simd_ext):
    return True

def get_simd_exts():
    return ['vmx', 'vsx']

def get_type(opts, simd_ext, typ, nsimd_typ):
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    if typ == 'f16':
        struct = 'struct {__vector float v0; __vector float v1;}'
    elif simd_ext == 'vmx' and typ in ['i64', 'u64', 'f64']:
        struct = 'struct {{ {} v0; {} v1; }}'.format(typ, typ)
    else:
        struct = native_type(typ)
    return 'typedef {} {};'.format(struct, nsimd_typ)

def get_logical_type(opts, simd_ext, typ, nsimd_typ):
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    if typ == 'f16':
        struct = 'struct {__vector __bool int v0; __vector __bool int v1;}'
    elif simd_ext == 'vmx' and typ in ['i64', 'u64', 'f64']:
        struct = 'struct { u64 v0; u64 v1; }'
    else:
        struct = native_typel(typ)
    return 'typedef {} {};'.format(struct, nsimd_typ)

def get_nb_registers(simd_ext):
    if simd_ext == 'vsx':
        return '64'
    elif simd_ext == 'vmx':
        return '32'
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def has_compatible_SoA_types(simd_ext):
    if simd_ext in get_simd_exts():
        return False
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_additional_include(func, platform, simd_ext):
    ret = '''#include <nsimd/cpu/cpu/{}.h>
             '''.format(func)

    DEBUG = False
    if DEBUG == True:
        ret += '''#include <nsimd/ppc/{simd_ext}/put.h>
                  #if NSIMD_CXX > 0
                  #include <cstdio>
                  #else
                  #include <stdio.h>
                  #endif
                  '''.format(simd_ext=simd_ext)

    if func == 'neq':
        ret += '''#include <nsimd/ppc/{simd_ext}/eq.h>
                  #include <nsimd/ppc/{simd_ext}/notl.h>
                  '''.format(simd_ext=simd_ext)

    elif func in ['loadlu', 'loadla']:
        ret += '''#include <nsimd/ppc/{simd_ext}/eq.h>
                  #include <nsimd/ppc/{simd_ext}/set1.h>
                  #include <nsimd/ppc/{simd_ext}/{load}.h>
                  #include <nsimd/ppc/{simd_ext}/notl.h>
                  '''.format(load='load' + func[5], **fmtspec)

    elif func in ['storelu']:
        ret += '''#include <nsimd/ppc/{simd_ext}/if_else1.h>
                  #include <nsimd/ppc/{simd_ext}/set1.h>
                  '''.format(**fmtspec)

    elif func in ['shr', 'shl']:
        ret += '''#include <nsimd/ppc/{simd_ext}/set1.h>
                  '''.format(**fmtspec)

    elif func == "shra":
        ret += '''#include <nsimd/scalar_utilities.h>
                  '''

    elif func[:11] == 'reinterpret' and DEBUG == True:
        ret += '''#if NSIMD_CXX > 0
                  #include <cstring>
                  #else
                  #include <string.h>
                  #endif
                  '''

    elif func in ['zip', 'unzip']:
        ret += '''#include <nsimd/ppc/{simd_ext}/{unzip_prefix}ziplo.h>
                  #include <nsimd/ppc/{simd_ext}/{unzip_prefix}ziphi.h>
                  '''.format(unzip_prefix="" if func == "zip" else "un",
                             **fmtspec)

    elif func in ['unziplo', 'unziphi']:
        ret += '''#include <nsimd/ppc/{simd_ext}/ziplo.h>
                  #include <nsimd/ppc/{simd_ext}/ziphi.h>
                  #include <math.h>
                  '''.format(**fmtspec)

    elif func[:5] in ['masko', 'maskz']:
        ret += '''#include <nsimd/scalar_utilities.h>
                  '''

    elif func == 'mask_for_loop_tail':
        ret += '''#include <nsimd/ppc/{simd_ext}/set1.h>
                  #include <nsimd/ppc/{simd_ext}/set1l.h>
                  #include <nsimd/ppc/{simd_ext}/iota.h>
                  #include <nsimd/ppc/{simd_ext}/lt.h>
                  '''.format(simd_ext=simd_ext)

    elif func[:4] == 'load':
        ret += '''

        #define NSIMD_PERMUTE_MASK_64(a, b)                        \
                {(unsigned char)(8 * a), (unsigned char)(8 * a + 1), \
                (unsigned char)(8 * b), (unsigned char)(8 * b + 1)}


        #define NSIMD_PERMUTE_MASK_32(a, b, c, d)                        \
                {(unsigned char)(4 * a), (unsigned char)(4 * a + 1),     \
                (unsigned char)(4 * a + 2), (unsigned char)(4 * a + 3),  \
                (unsigned char)(4 * b), (unsigned char)(4 * b + 1),      \
                (unsigned char)(4 * b + 2), (unsigned char)(4 * b + 3),  \
                (unsigned char)(4 * c), (unsigned char)(4 * c + 1),      \
                (unsigned char)(4 * c + 2), (unsigned char)(4 * c + 3),  \
                (unsigned char)(4 * d), (unsigned char)(4 * d + 1),      \
                (unsigned char)(4 * d + 2), (unsigned char)(4 * d + 3)}

         #define NSIMD_PERMUTE_MASK_16(a, b, c, d, e, f, g, h)           \
               {(unsigned char)(2 * a + 0), (unsigned char)(2 * a + 1),  \
                (unsigned char)(2 * b + 0), (unsigned char)(2 * b + 1),  \
                (unsigned char)(2 * c + 0), (unsigned char)(2 * c + 1),  \
                (unsigned char)(2 * d + 0), (unsigned char)(2 * d + 1),  \
                (unsigned char)(2 * e + 0), (unsigned char)(2 * e + 1),  \
                (unsigned char)(2 * f + 0), (unsigned char)(2 * f + 1),  \
                (unsigned char)(2 * g + 0), (unsigned char)(2 * g + 1),  \
                (unsigned char)(2 * h + 0), (unsigned char)(2 * h + 1)}

         #define NSIMD_PERMUTE_MASK_8(a, b, c, d, e, f, g, h,            \
                                      i, j, k, l, m, n, o, p)            \
              { (unsigned char)(a), (unsigned char)(b),                  \
                (unsigned char)(c), (unsigned char)(d),                  \
                (unsigned char)(e), (unsigned char)(f),                  \
                (unsigned char)(g), (unsigned char)(h),                  \
                (unsigned char)(i), (unsigned char)(j),                  \
                (unsigned char)(k), (unsigned char)(l),                  \
                (unsigned char)(m), (unsigned char)(n),                  \
                (unsigned char)(o), (unsigned char)(p) }
        '''

    return ret

# -----------------------------------------------------------------------------

def printf2(*args0):
    """
    debugging purposes
    decorate the function with it and when executed on test, it will print the
    environnements *args0 are the name of var to printf
    """
    to_print = []
    for arg in args0:
        if isinstance(arg, str):
            to_print.append(arg)

    def decorator(func):
        import inspect

        def wrapper(*args, **kwargs):
            func_args = inspect.signature(func).bind(*args, **kwargs).arguments
            func_args_str = '{} called on {}\\n'. \
                            format(func.__name__, fmtspec['typ']) + \
                            ', "'.join('{} = {!r}'.format(*item) \
                                       for item in func_args.items())
            ret = ''
            if not DEBUG:
                return func(*args)
            typ = ''
            if 'typ' in func_args:
                typ = func_args['typ']
            else:
                typ = func_args['from_typ']
            ret += 'int k;\n'
            if func.__name__ == 'store1234' and typ in ['f64', 'i64', 'u64']:
                ret += '''
                       printf("element to store: %ld %ld", {in1}{suf0},
                              {in1}{suf1});
                       printf("\\n");
                       '''.format(**fmtspec, **get_suf64(typ))
            elif func.__name__ == 'store1234' and typ[1:] == '32':
                ret += '''
                       printf("element to store:");
                       for (k = 0; k < 4; k++) {{
                         printf(" %lx", {in1}[k]);
                       }}
                       printf("\\n");
                       '''.format(**fmtspec, nbits=get_len(typ))
            #print var passed as parameter on printf2
            for var in to_print:
                if ppc_is_vec_type(typ):
                    ret += '''
                           printf("values of {var}:");
                           for (k = 0; k < {nbits}; k++) {{
                             printf(" %lld", {var}[k]);
                           }}
                           printf("\\n");
                           '''.format(var=var, **fmtspec, nbits=get_len(typ))
            return '''
                   printf("\\n---------------\\n");
                   printf("{}.{} ( {} )\\n");
                   '''.format(func.__module__, func.__qualname__,
                              func_args_str) + ret + func(*args)

        return wrapper

    return decorator


# -----------------------------------------------------------------------------
# Loads of degree 1, 2, 3 and 4
# About unaligned loads/stores for Altivec:
# https://developer.ibm.com/technologies/systems/articles/pa-dalign/

def load1234(simd_ext, typ, deg, aligned):
    if typ in ['f64', 'i64', 'u64']:
        if deg == 1:
            if simd_ext == 'vmx':
                return '''nsimd_{simd_ext}_v{typ} ret;
                          ret.v0 = {in0}[0];
                          ret.v1 = {in0}[1];
                          return ret;'''.format(**fmtspec)
            else:
                return '''nsimd_{simd_ext}_v{typ} ret;
                          ret = vec_splats({in0}[0]);
                          ret = vec_insert({in0}[1], ret, 1);
                          return ret;'''.format(**fmtspec)
        else:
            if simd_ext == 'vmx':
                return \
                'nsimd_{simd_ext}_v{typ}x{} ret;\n'.format(deg, **fmtspec) + \
                '\n'.join(['ret.v{i}.v0 = *({in0} + {i});'. \
                           format(i=i, **fmtspec) \
                           for i in range(0, deg)]) + \
                '\n'.join(['ret.v{i}.v1 = *({in0} + {ipd});'. \
                           format(i=i, ipd=i + deg, **fmtspec) \
                           for i in range(0, deg)]) + \
                '\nreturn ret;'
            else:
                return \
                'nsimd_{simd_ext}_v{typ}x{} ret;\n'.format(deg, **fmtspec) + \
                '\n'.join('ret.v{i} = vec_splats(*({in0} + {i}));'. \
                          format(i=i, **fmtspec) \
                          for i in range(0, deg)) + \
                '\n'.join('ret.v{i} = vec_insert(*({in0} + {ipd}), ret, 1);'. \
                          format(i=i, ipd=i + deg, **fmtspec) \
                          for i in range(0, deg)) + \
                '\nreturn ret;'
    if typ == 'f16':
        if deg == 1:
            return \
            '''nsimd_{simd_ext}_vf16 ret;
               u16 *ptr = (u16 *){in0};
               ret.v0 = vec_splats(nsimd_u16_to_f32(ptr[0]));
               ret.v0 = vec_insert(nsimd_u16_to_f32(ptr[1]), ret.v0, 1);
               ret.v0 = vec_insert(nsimd_u16_to_f32(ptr[2]), ret.v0, 2);
               ret.v0 = vec_insert(nsimd_u16_to_f32(ptr[3]), ret.v0, 3);
               ret.v1 = vec_splats(nsimd_u16_to_f32(ptr[4]));
               ret.v1 = vec_insert(nsimd_u16_to_f32(ptr[5]), ret.v1, 1);
               ret.v1 = vec_insert(nsimd_u16_to_f32(ptr[6]), ret.v1, 2);
               ret.v1 = vec_insert(nsimd_u16_to_f32(ptr[7]), ret.v1, 3);
               return ret;'''.format(**fmtspec)
        else:
            ret = '''nsimd_{simd_ext}_vf16x{deg} ret;
                     u16 *ptr = (u16 *){in0};
                     '''.format(deg=deg, **fmtspec)

            for i in range(0, deg):
                for k in range(0, 2):
                    ret += 'ret.v{}.v{} = vec_splats(' \
                           'nsimd_u16_to_f32(ptr[{}]));\n'. \
                           format(i, k, i + k * 4 * deg)
                    for j in range(1, 4):
                        ret += 'ret.v{i}.v{k} = vec_insert(nsimd_u16_to_f32(' \
                               'ptr[{o}]), ret.v{i}.v{k}, {j});\n'. \
                               format(i=i, k=k, j=j,
                                      o=i + k * 4 * deg + j * deg)
            ret += 'return ret;'
            return ret
    if deg == 1:
        if aligned:
            return 'return vec_ld(0, {in0});'.format(**fmtspec)
        else:
            return 'return *({ppc_typ}*){in0};'. \
                   format(ppc_typ=native_type(typ), **fmtspec)

    # From here deg >= 2

    if aligned:
        load = 'nsimd_{simd_ext}_v{typ}x{deg} ret;\n'. \
               format(deg=deg, **fmtspec) + \
               '\n'.join(
                 'nsimd_{simd_ext}_v{typ} in{i} = vec_ld({o}, {in0});'. \
                 format(i=i, o=i * 16, **fmtspec) for i in range(deg))
    else:
        load = 'nsimd_{simd_ext}_v{typ}x{deg} ret;\n'. \
               format(deg=deg, **fmtspec) + \
               '\n'.join(
                 'nsimd_{simd_ext}_v{typ} in{i} = *({ppc_typ}*){in0};'. \
                 format(i=i, ppc_typ=native_type(typ), **fmtspec) \
                        for i in range(0, deg))
    if deg == 2:
        if typ in ['i32', 'u32', 'f32']:
            return \
            '''{load}

               nsimd_{simd_ext}_v{typ} tmp0 = vec_mergeh(in0, in1);
               nsimd_{simd_ext}_v{typ} tmp1 = vec_mergel(in0, in1);

               ret.v0 = vec_mergeh(tmp0, tmp1);
               ret.v1 = vec_mergel(tmp0, tmp1);

               return ret;'''.format(load=load, **fmtspec)
        elif typ in ['i16', 'u16']:
            return \
            '''{load}

               nsimd_{simd_ext}_v{typ} tmp0 = vec_mergeh(in0, in1);
               nsimd_{simd_ext}_v{typ} tmp1 = vec_mergel(in0, in1);

               in0 = vec_mergeh(tmp0, tmp1);
               in1 = vec_mergel(tmp0, tmp1);

               ret.v0 = vec_mergeh(in0, in1);
               ret.v1 = vec_mergel(in0, in1);

               return ret;'''.format(load=load, **fmtspec)
        elif typ in ['i8', 'u8']:
            return \
            '''__vector unsigned char perm1 = NSIMD_PERMUTE_MASK_8(
                 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
               __vector unsigned char perm2 = NSIMD_PERMUTE_MASK_8(
                 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

               {load}

               ret.v0 = vec_perm(in0, in1, perm1);
               ret.v1 = vec_perm(in0, in1, perm2);

               return ret;'''.format(load=load, **fmtspec)
    elif deg == 3:
        if typ in ['i32', 'u32', 'f32']:
            return \
            '''__vector char perm1 = NSIMD_PERMUTE_MASK_32(0, 3, 6, 0);

               {load}

               nsimd_{simd_ext}_v{typ} tmp0 = vec_perm(in0, in1, perm1);
               nsimd_{simd_ext}_v{typ} tmp1 = vec_perm(in1, in2, perm1);
               nsimd_{simd_ext}_v{typ} tmp2 = vec_perm(in2, in0, perm1);

               __vector char perm2 = NSIMD_PERMUTE_MASK_32(0, 1, 2, 5);
               __vector char perm3 = NSIMD_PERMUTE_MASK_32(5, 0, 1, 2);
               __vector char perm4 = NSIMD_PERMUTE_MASK_32(2, 5, 0, 1);

               ret.v0 = vec_perm(tmp0, in2, perm2);
               ret.v1 = vec_perm(tmp1, in0, perm3);
               ret.v2 = vec_perm(tmp2, in1, perm4);

               return ret;'''.format(load=load, **fmtspec)
        elif typ in ['i16', 'u16']:
            return \
            '''{load}

               __vector char permRAB = NSIMD_PERMUTE_MASK_16(
                                           0, 3, 6, 9, 12, 15, 0, 0);
               __vector char permRDC = NSIMD_PERMUTE_MASK_16(
                                           0, 1, 2, 3, 4, 5, 10, 13);

               nsimd_{simd_ext}_v{typ} tmp0 = vec_perm(in0, in1, permRAB);
               ret.v0 = vec_perm(tmp0, in2, permRDC);

               __vector char permGAB = NSIMD_PERMUTE_MASK_16(
                                           1, 4, 7, 10, 13, 0, 0, 0);
               __vector char permGEC = NSIMD_PERMUTE_MASK_16(
                                           0, 1, 2, 3, 4, 8, 11, 14);

               nsimd_{simd_ext}_v{typ} tmp1 = vec_perm(in0, in1, permGAB);
               ret.v1 = vec_perm(tmp1, in2, permGEC);

               __vector char permBAB = NSIMD_PERMUTE_MASK_16(
                                           2, 5, 8, 11, 14, 0, 0, 0);
               __vector char permBFC = NSIMD_PERMUTE_MASK_16(
                                           0, 1, 2, 3, 4, 9, 12, 15);

               nsimd_{simd_ext}_v{typ} tmp2 = vec_perm(in0, in1, permBAB);
               ret.v2 = vec_perm(tmp2, in2, permBFC);

               return ret;'''.format(load=load, **fmtspec)
        elif typ in ['i8', 'u8']:
            return \
            '''{load}

               __vector char permRAB = NSIMD_PERMUTE_MASK_8(
                   0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
               __vector char permRDC = NSIMD_PERMUTE_MASK_8(
                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);

               nsimd_{simd_ext}_v{typ} tmp0 = vec_perm(in0, in1, permRAB);
               ret.v0 = vec_perm(tmp0, in2, permRDC);

               __vector char permGAB = NSIMD_PERMUTE_MASK_8(
                   1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 0, 0, 0, 0, 0);
               __vector char permGEC = NSIMD_PERMUTE_MASK_8(
                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18, 21, 24, 27, 30);

               nsimd_{simd_ext}_v{typ} tmp1 = vec_perm(in0, in1, permGAB);
               ret.v1 = vec_perm(tmp1, in2, permGEC);

               __vector char permBAB = NSIMD_PERMUTE_MASK_8(
                   2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0, 0, 0, 0, 0, 0);
               __vector char permBFC = NSIMD_PERMUTE_MASK_8(
                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 19, 22, 25, 28, 31);

               nsimd_{simd_ext}_v{typ} tmp2 = vec_perm(in0, in1, permBAB);
               ret.v2 = vec_perm(tmp2, in2, permBFC);

               return ret;'''.format(load=load, **fmtspec)
    else:
        if typ in ['i32', 'u32', 'f32']:
            return \
            '''{load}

               nsimd_{simd_ext}_v{typ} tmp0 = vec_mergeh(in0, in2);
               nsimd_{simd_ext}_v{typ} tmp1 = vec_mergel(in0, in2);
               nsimd_{simd_ext}_v{typ} tmp2 = vec_mergeh(in1, in3);
               nsimd_{simd_ext}_v{typ} tmp3 = vec_mergel(in1, in3);

               ret.v0 = vec_mergeh(tmp0, tmp2);
               ret.v1 = vec_mergel(tmp0, tmp2);
               ret.v2 = vec_mergeh(tmp1, tmp3);
               ret.v3 = vec_mergel(tmp1, tmp3);

               return ret;'''.format(load=load, **fmtspec)
        elif typ in ['i16', 'u16']:
            return \
            '''{load}

               ret.v0 = vec_mergeh(in0, in2);
               ret.v1 = vec_mergel(in0, in2);
               ret.v2 = vec_mergeh(in1, in3);
               ret.v3 = vec_mergel(in1, in3);

               nsimd_{simd_ext}_v{typ} tmp0 = vec_mergeh(ret.v0, ret.v2);
               nsimd_{simd_ext}_v{typ} tmp1 = vec_mergel(ret.v0, ret.v2);
               nsimd_{simd_ext}_v{typ} tmp2 = vec_mergeh(ret.v1, ret.v3);
               nsimd_{simd_ext}_v{typ} tmp3 = vec_mergel(ret.v1, ret.v3);

               ret.v0 = vec_mergeh(tmp0, tmp2);
               ret.v1 = vec_mergel(tmp0, tmp2);
               ret.v2 = vec_mergeh(tmp1, tmp3);
               ret.v3 = vec_mergel(tmp1, tmp3);

               return ret;'''.format(load=load, **fmtspec)
        elif typ in ['i8', 'u8']:
            return \
            '''{load}

               nsimd_{simd_ext}_v{typ} tmp0 = vec_mergeh(in0, in2);
               nsimd_{simd_ext}_v{typ} tmp1 = vec_mergel(in0, in2);
               nsimd_{simd_ext}_v{typ} tmp2 = vec_mergeh(in1, in3);
               nsimd_{simd_ext}_v{typ} tmp3 = vec_mergel(in1, in3);

               ret.v0 = vec_mergeh(tmp0, tmp2);
               ret.v1 = vec_mergel(tmp0, tmp2);
               ret.v2 = vec_mergeh(tmp1, tmp3);
               ret.v3 = vec_mergel(tmp1, tmp3);

               tmp0 = vec_mergeh(ret.v0, ret.v2);
               tmp1 = vec_mergel(ret.v0, ret.v2);
               tmp2 = vec_mergeh(ret.v1, ret.v3);
               tmp3 = vec_mergel(ret.v1, ret.v3);

               ret.v0 = vec_mergeh(tmp0, tmp2);
               ret.v1 = vec_mergel(tmp0, tmp2);
               ret.v2 = vec_mergeh(tmp1, tmp3);
               ret.v3 = vec_mergel(tmp1, tmp3);

               return ret;'''.format(load=load, **fmtspec)

# -----------------------------------------------------------------------------
# Stores of degree 1, 2, 3 and 4

def store1234(simd_ext, typ, deg, aligned):
    if typ in ['f64', 'i64', 'u64']:
        if simd_ext == 'vmx':
            return '\n'.join('{}[{}] = {}.v0;'. \
                             format(common.in0, i, common.get_arg(i + 1)) \
                             for i in range(deg)) + '\n' + \
                   '\n'.join('{}[{}] = {}.v1;'. \
                             format(common.in0, i + deg,
                                    common.get_arg(i + 1)) for i in range(deg))
        else:
            return '\n'.join('{}[{}] = vec_extract({}, 0);'. \
                             format(common.in0, i, common.get_arg(i + 1)) \
                             for i in range(deg)) + '\n' + \
                   '\n'.join('{}[{}] = vec_extract({}, 1);'. \
                             format(common.in0, i + deg,
                                    common.get_arg(i + 1)) for i in range(deg))
    if typ == 'f16':
        if deg == 1:
            return \
            '''u16 *ptr = (u16 *){in0};
               ptr[0] = nsimd_f32_to_u16(vec_extract({in1}.v0, 0));
               ptr[1] = nsimd_f32_to_u16(vec_extract({in1}.v0, 1));
               ptr[2] = nsimd_f32_to_u16(vec_extract({in1}.v0, 2));
               ptr[3] = nsimd_f32_to_u16(vec_extract({in1}.v0, 3));
               ptr[4] = nsimd_f32_to_u16(vec_extract({in1}.v1, 0));
               ptr[5] = nsimd_f32_to_u16(vec_extract({in1}.v1, 1));
               ptr[6] = nsimd_f32_to_u16(vec_extract({in1}.v1, 2));
               ptr[7] = nsimd_f32_to_u16(vec_extract({in1}.v1, 3));'''. \
               format(**fmtspec)
        else:
            ret = 'u16 *ptr = (u16 *){in0};\n'.format(**fmtspec)
            for i in range(0, deg):
                for k in range(0, 2):
                    for j in range(0, 4):
                        ret += 'ptr[{o}] = nsimd_f32_to_u16(' \
                               'vec_extract({a}.v{k}, {j}));\n'. \
                               format(a=common.get_arg(i + 1), j=j, k=k,
                                      o=i + k * 4 * deg + j * deg, **fmtspec)
            return ret
    if deg == 1:
        if aligned:
            return 'vec_st({in1}, 0, {in0});'.format(**fmtspec)
        else:
            return '*({ppc_typ} *){in0} = {in1};'. \
                   format(ppc_typ=native_type(typ), **fmtspec)

    # From here deg >= 2

    if aligned:
        store = '\n'.join('vec_st(ret{i}, {o}, {in0});'. \
                          format(i=i, o=i * 16, **fmtspec) \
                          for i in range(0, deg))
    else:
        store = '\n'.join('*({ppc_typ} *)({in0} + {o}) = ret{i};'. \
                          format(o=i * get_len(typ), ppc_typ=native_type(typ),
                                 i=i, **fmtspec) for i in range(deg))
    if deg == 2:
        return \
        '''nsimd_{simd_ext}_v{typ} ret0 = vec_mergeh({in1}, {in2});
           nsimd_{simd_ext}_v{typ} ret1 = vec_mergel({in1}, {in2});

           {store}'''.format(store=store, **fmtspec)
    elif deg == 3:
        if typ in ['i32', 'u32', 'f32']:
            return \
            '''__vector char perm1 = NSIMD_PERMUTE_MASK_32(0, 2, 4, 6);
               __vector char perm2 = NSIMD_PERMUTE_MASK_32(0, 2, 5, 7);
               __vector char perm3 = NSIMD_PERMUTE_MASK_32(1, 3, 5, 7);

               nsimd_{simd_ext}_v{typ} tmp0 = vec_perm({in1}, {in2}, perm1);
               nsimd_{simd_ext}_v{typ} tmp1 = vec_perm({in3}, {in1}, perm2);
               nsimd_{simd_ext}_v{typ} tmp2 = vec_perm({in2}, {in3}, perm3);

               nsimd_{simd_ext}_v{typ} ret0 = vec_perm(tmp0, tmp1, perm1);
               nsimd_{simd_ext}_v{typ} ret1 = vec_perm(tmp2, tmp0, perm2);
               nsimd_{simd_ext}_v{typ} ret2 = vec_perm(tmp1, tmp2, perm3);

               {store}'''.format(store=store, **fmtspec)
        elif typ in ['i16', 'u16']:
            return \
            '''__vector char permARG = NSIMD_PERMUTE_MASK_16(
                                           0, 8, 0, 1, 9, 0, 2, 10);
               __vector char permAXB = NSIMD_PERMUTE_MASK_16(
                                           0, 1, 8, 3, 4, 9, 6, 7);

               nsimd_{simd_ext}_v{typ} tmp0 = vec_perm({in1}, {in2}, permARG);
               nsimd_{simd_ext}_v{typ} ret0 = vec_perm(tmp0, {in3}, permAXB);

               __vector char permBRG = NSIMD_PERMUTE_MASK_16(
                                           0, 3, 11, 0, 4, 12, 0, 5);
               __vector char permBYB = NSIMD_PERMUTE_MASK_16(
                                           10, 1, 2, 11, 4, 5, 12, 7);

               nsimd_{simd_ext}_v{typ} tmp1 = vec_perm({in1}, {in2}, permBRG);
               nsimd_{simd_ext}_v{typ} ret1 = vec_perm(tmp1, {in3}, permBYB);

               __vector char permCRG = NSIMD_PERMUTE_MASK_16(
                                           13, 0, 6, 14, 0, 7, 15, 0);
               __vector char permCZB = NSIMD_PERMUTE_MASK_16(
                                           0, 13, 2, 3, 14, 5, 6, 15);

               nsimd_{simd_ext}_v{typ} tmp2 = vec_perm({in1}, {in2}, permCRG);
               nsimd_{simd_ext}_v{typ} ret2 = vec_perm(tmp2, {in3}, permCZB);

               {store}'''.format(store=store, **fmtspec)
        elif typ in ['i8', 'u8']:
            return \
            '''__vector char mARG = NSIMD_PERMUTE_MASK_8(
                   0, 16, 0, 1, 17, 0, 2, 18, 0, 3, 19, 0, 4, 20, 0, 5);
               __vector char mAXB = NSIMD_PERMUTE_MASK_8(
                   0, 1, 16, 3, 4, 17, 6, 7, 18, 9, 10, 19, 12, 13, 20, 15);

               nsimd_{simd_ext}_v{typ} tmp0 = vec_perm({in1}, {in2}, mARG);
               nsimd_{simd_ext}_v{typ} ret0 = vec_perm(tmp0, {in3}, mAXB);

               __vector char mBRG = NSIMD_PERMUTE_MASK_8(
                   21, 0, 6, 22, 0, 7, 23, 0, 8, 24, 0, 9, 25, 0, 10, 26);
               __vector char mBYB = NSIMD_PERMUTE_MASK_8(
                   0, 21, 2, 3, 22, 5, 6, 23, 8, 9, 24, 11, 12, 25, 14, 15);

               nsimd_{simd_ext}_v{typ} tmp1 = vec_perm({in1}, {in2}, mBRG);
               nsimd_{simd_ext}_v{typ} ret1 = vec_perm(tmp1, {in3}, mBYB);

               __vector char mCRG = NSIMD_PERMUTE_MASK_8(
                   0, 11, 27, 0, 12, 28, 0, 13, 29, 0, 14, 30, 0, 15, 31, 0);
               __vector char mCZB = NSIMD_PERMUTE_MASK_8(
                   26, 1, 2, 27, 4, 5, 28, 7, 8, 29, 10, 11, 30, 13, 14, 31);

               nsimd_{simd_ext}_v{typ} tmp2 = vec_perm({in1}, {in2}, mCRG);
               nsimd_{simd_ext}_v{typ} ret2 = vec_perm(tmp2, {in3}, mCZB);

               {store}'''.format(store=store, **fmtspec)
    else:
        if typ in ['i32', 'u32', 'f32']:
            return \
            '''nsimd_{simd_ext}_v{typ} tmp0 = vec_mergeh({in1}, {in3});
               nsimd_{simd_ext}_v{typ} tmp1 = vec_mergel({in1}, {in3});
               nsimd_{simd_ext}_v{typ} tmp2 = vec_mergeh({in2}, {in4});
               nsimd_{simd_ext}_v{typ} tmp3 = vec_mergel({in2}, {in4});

               nsimd_{simd_ext}_v{typ} ret0 = vec_mergeh(tmp0, tmp2);
               nsimd_{simd_ext}_v{typ} ret1 = vec_mergel(tmp0, tmp2);
               nsimd_{simd_ext}_v{typ} ret2 = vec_mergeh(tmp1, tmp3);
               nsimd_{simd_ext}_v{typ} ret3 = vec_mergel(tmp1, tmp3);

               {store}'''.format(store=store, **fmtspec)
        elif typ in ['i16', 'u16']:
            return \
            '''nsimd_{simd_ext}_v{typ} tmp0 = vec_mergeh({in1}, {in3});
               nsimd_{simd_ext}_v{typ} tmp1 = vec_mergel({in1}, {in3});
               nsimd_{simd_ext}_v{typ} tmp2 = vec_mergeh({in2}, {in4});
               nsimd_{simd_ext}_v{typ} tmp3 = vec_mergel({in2}, {in4});

               nsimd_{simd_ext}_v{typ} ret0 = vec_mergeh(tmp0, tmp2);
               nsimd_{simd_ext}_v{typ} ret1 = vec_mergel(tmp0, tmp2);
               nsimd_{simd_ext}_v{typ} ret2 = vec_mergeh(tmp1, tmp3);
               nsimd_{simd_ext}_v{typ} ret3 = vec_mergel(tmp1, tmp3);

               {store}'''.format(store=store, **fmtspec)
        elif typ in ['i8', 'u8']:
            return \
            '''nsimd_{simd_ext}_v{typ} tmp0 = vec_mergeh({in1}, {in3});
               nsimd_{simd_ext}_v{typ} tmp1 = vec_mergel({in1}, {in3});
               nsimd_{simd_ext}_v{typ} tmp2 = vec_mergeh({in2}, {in4});
               nsimd_{simd_ext}_v{typ} tmp3 = vec_mergel({in2}, {in4});

               nsimd_{simd_ext}_v{typ} ret0 = vec_mergeh(tmp0, tmp2);
               nsimd_{simd_ext}_v{typ} ret1 = vec_mergel(tmp0, tmp2);
               nsimd_{simd_ext}_v{typ} ret2 = vec_mergeh(tmp1, tmp3);
               nsimd_{simd_ext}_v{typ} ret3 = vec_mergel(tmp1, tmp3);

               {store}'''.format(store=store, **fmtspec)

# -----------------------------------------------------------------------------
# Length

def len1(simd_ext, typ):
    return 'return {};'.format(128 // int(typ[1:]))

# -----------------------------------------------------------------------------
# Other helper functions

def simple_op2(op, simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code(op, simd_ext, typ, ['v', 'v', 'v'])
    return 'return vec_{op}({in0}, {in1});'.format(op=op, **fmtspec)

# Binary operators: and, or, xor, andnot
def binary_op2(op, simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code(op, simd_ext, typ, ['v', 'v', 'v'])
    else:
        ppcop = {'orb': 'or', 'xorb': 'xor', 'andb': 'and', 'andnotb': 'andc'}
        return 'return vec_{op}({in0}, {in1});'.format(op=ppcop[op], **fmtspec)

# Logical operators: and, or, xor, andnot
def logical_op2(op, simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code(op, simd_ext, typ, ['l', 'l', 'l'])
    ppcop = {'orl': 'or', 'xorl': 'xor', 'andl': 'and', 'andnotl': 'andc'}
    return 'return vec_{op}({in0}, {in1});'.format(op=ppcop[op], **fmtspec)

# -----------------------------------------------------------------------------

def div2(simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code('div', simd_ext, typ, ['v', 'v', 'v'])
    elif typ in common.ftypes:
        return 'return vec_div({in0}, {in1});'.format(**fmtspec)
    elif typ in common.iutypes:
        return '''nsimd_{simd_ext}_v{typ} ret;
                  ret = vec_splats(({typ})(vec_extract({in0}, 0) /
                                   vec_extract({in1}, 0)));
                  '''.format(**fmtspec) + \
               '\n'.join(
               '''ret = vec_insert(({typ})(vec_extract({in0}, {i}) /
                                   vec_extract({in1}, {i})), ret, {i});'''. \
                                   format(i=i, **fmtspec) \
                                   for i in range(get_len(typ))) + \
               '\nreturn ret;'

# -----------------------------------------------------------------------------

def not1(simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code('notb', simd_ext, typ, ['v', 'v'])
    return 'return vec_nor({in0}, {in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------

def lnot1(simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code('notl', simd_ext, typ, ['l', 'l'])
    return 'return vec_nor({in0}, {in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------

def sqrt1(simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code('sqrt', simd_ext, typ, ['v', 'v'])
    return 'return vec_sqrt({in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------

def shift2(op, simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code(op, simd_ext, typ, ['v', 'v', 'p'])
    return 'return vec_{ppcop}({in0}, vec_splats((u{typnbits}){in1}));'. \
           format(ppcop={'shl': 'sl', 'shr': 'sr', 'shra': 'sra'}[op],
                  **fmtspec)

# -----------------------------------------------------------------------------

def set1(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  f32 tmp = nsimd_f16_to_f32({in0});
                  ret.v0 = vec_splats(tmp);
                  ret.v1 = ret.v0;
                  return ret;'''.format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return '''nsimd_{simd_ext}_v{typ} ret;
                  ret.v0 = {in0};
                  ret.v1 = {in0};
                  return ret;'''.format(**fmtspec)
    else:
        return 'return vec_splats({in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------

def lset1(simd_ext, typ):
    if typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vlf16 ret;
           ret.v0 = (__vector __bool int)vec_splats((u32)({in0} ? -1 : 0));
           ret.v1 = ret.v0;
           return ret;'''.format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return '''nsimd_{simd_ext}_vl{typ} ret;
                  ret.v0 = (u64)({in0} ? -1 : 0);
                  ret.v1 = (u64)({in0} ? -1 : 0);
                  return ret;'''.format(**fmtspec)
    else:
        return '''if ({in0}) {{
                    return ({ppc_typ})vec_splats((u{typnbits})-1);
                  }} else {{
                    return {lzeros};
                  }}'''.format(ppc_typ=native_typel(typ), **fmtspec)

# -----------------------------------------------------------------------------

def cmp2(op, simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code(op, simd_ext, typ, ['l', 'v', 'v'])
    return 'return vec_cmp{op}({in0}, {in1});'.format(op=op, **fmtspec)

# -----------------------------------------------------------------------------

def if_else3(simd_ext, typ):
    if typ == 'f16':
        return emulate_f16('if_else1', simd_ext, ['v', 'l', 'v', 'v'])
    elif has_to_be_emulated(simd_ext, typ):
        return '''nsimd_{simd_ext}_v{typ} ret;
                  ret.v0 = ({in0}.v0 ? {in1}.v0 : {in2}.v0);
                  ret.v1 = ({in0}.v1 ? {in1}.v1 : {in2}.v1);
                  return ret;'''.format(**fmtspec)
    return 'return vec_sel({in2}, {in1}, {in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------

def minmax2(op, simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code(op, simd_ext, typ, ['v', 'v', 'v'])
    return 'return vec_{op}({in0}, {in1});'.format(op=op, **fmtspec)

# -----------------------------------------------------------------------------

def abs1(simd_ext, typ):
    if typ in common.utypes:
        return 'return {in0};'.format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return emulation_code('abs', simd_ext, typ, ['v', 'v'])
    return 'return vec_abs({in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------

def round1(op, simd_ext, typ):
    if typ in common.iutypes:
        return 'return {in0};'.format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return emulation_code(op, simd_ext, typ, ['v', 'v'])
    ppcop = {'trunc': 'trunc', 'ceil': 'ceil', 'floor': 'floor',
             'round_to_even': 'round'}
    return 'return vec_{op}({in0});'.format(op=ppcop[op], **fmtspec)

# -----------------------------------------------------------------------------

def fma(op, simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code(op, simd_ext, typ, ['v', 'v', 'v', 'v'])
    elif typ in common.iutypes:
        if op == 'fma':
            return \
            'return vec_add(vec_mul({in0}, {in1}), {in2});'.format(**fmtspec)
        elif op == 'fms':
            return \
            'return vec_sub(vec_mul({in0}, {in1}), {in2});'.format(**fmtspec)
        elif op == 'fnma':
            return \
            'return vec_sub({in2}, vec_mul({in0}, {in1}));'.format(**fmtspec)
        elif op == 'fnms':
            return '''return vec_sub(nsimd_neg_{simd_ext}_{typ}({in2}),
                                 vec_mul({in0}, {in1}));'''.format(**fmtspec)
    elif typ in common.ftypes:
        ppcop = { 'fma': 'vec_madd', 'fms': 'vec_msub', 'fnms': 'vec_nmadd',
                  'fnma': 'vec_nmsub' }
        return 'return {ppcop}({in0}, {in1}, {in2});'. \
               format(ppcop=ppcop[op], **fmtspec)

# -----------------------------------------------------------------------------

def neg1(simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code('neg', simd_ext, typ, ['v', 'v'])
    elif typ in common.itypes or typ in common.ftypes:
        return 'return vec_neg({in0});'.format(**fmtspec)
    else:
        return 'return vec_sub({zeros}, {in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------

def recs1(op, simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code(op, simd_ext, typ, ['v', 'v'])
    elif op == 'rec':
        return 'return vec_div(vec_splats(({typ})1), {in0});'. \
               format(**fmtspec)
    elif op in ['rec8', 'rec11']:
        return 'return vec_re({in0});'.format(**fmtspec)
    elif op in ['rsqrt8', 'rsqrt11']:
        return 'return vec_rsqrte({in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------

def loadl(aligned, simd_ext, typ):
    return \
    '''/* This can surely be improved but it is not our priority. */
       return nsimd_notl_{simd_ext}_{typ}(nsimd_eq_{simd_ext}_{typ}(
                nsimd_load{align}_{simd_ext}_{typ}(
                  {in0}), nsimd_set1_{simd_ext}_{typ}({zero})));'''. \
                  format(align='a' if aligned else 'u',
                         zero='nsimd_f32_to_f16(0.0f)' if typ == 'f16'
                         else '({})0'.format(typ), **fmtspec)

# -----------------------------------------------------------------------------

def storel(aligned, simd_ext, typ):
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

def allany1(op, simd_ext, typ):
    binop = '&&' if op == 'all' else '||'
    if typ == 'f16':
        return \
        '''return nsimd_{op}_{simd_ext}_f32({in0}.v0) {binop}
                  nsimd_{op}_{simd_ext}_f32({in0}.v1);'''. \
                  format(op=op, binop=binop, **fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return 'return {in0}.v0 {binop} {in0}.v1;'. \
               format(binop=binop, **fmtspec)
    return 'return vec_{op}_ne({in0}, ({lzeros}));'.format(op=op, **fmtspec)

# -----------------------------------------------------------------------------

def nbtrue1(simd_ext, typ):
    if typ == 'f16':
        return \
        '''return nsimd_nbtrue_{simd_ext}_f32({in0}.v0) +
                  nsimd_nbtrue_{simd_ext}_f32({in0}.v1);'''. \
                  format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return 'return -(int)((i64)({in0}.v0) + (i64)({in0}.v1));'. \
               format(**fmtspec)
    return 'return {};'. \
           format(' + '.join('(vec_extract({in0}, {i}) ? 1 : 0)'. \
                             format(i=i, **fmtspec) \
                             for i in range(get_len(typ))))

# -----------------------------------------------------------------------------

def reinterpretl1(simd_ext, from_typ, to_typ):
    if from_typ == to_typ:
        return 'return {in0};'.format(**fmtspec)
    elif simd_ext == 'vmx' and from_typ in ['f64', 'i64', 'u64']:
        return \
        '''nsimd_{simd_ext}_vl{to_typ} ret;
           ret.v0 = {in0}.v0;
           ret.v1 = {in0}.v1;
           return ret;'''.format(**fmtspec)
    elif from_typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vl{to_typ} ret =
               (__vector __bool short)vec_splats(
                   (u16)vec_extract({in0}.v0, 0));
           ret = vec_insert((u16)vec_extract({in0}.v0, 1), ret, 1);
           ret = vec_insert((u16)vec_extract({in0}.v0, 2), ret, 2);
           ret = vec_insert((u16)vec_extract({in0}.v0, 3), ret, 3);
           ret = vec_insert((u16)vec_extract({in0}.v1, 0), ret, 4);
           ret = vec_insert((u16)vec_extract({in0}.v1, 1), ret, 5);
           ret = vec_insert((u16)vec_extract({in0}.v1, 2), ret, 6);
           ret = vec_insert((u16)vec_extract({in0}.v1, 3), ret, 7);
           return ret;'''.format(**fmtspec)
    elif to_typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vlf16 ret;
           ret.v0 = (__vector __bool int)vec_splats(
                        (u32)(vec_extract({in0}, 0) ? -1 : 0));
           ret.v0 = vec_insert(
                        (u32)(vec_extract({in0}, 1) ? -1 : 0), ret.v0, 1);
           ret.v0 = vec_insert(
                        (u32)(vec_extract({in0}, 2) ? -1 : 0), ret.v0, 2);
           ret.v0 = vec_insert(
                        (u32)(vec_extract({in0}, 3) ? -1 : 0), ret.v0, 3);
           ret.v1 = (__vector __bool int)vec_splats(
                        (u32)(vec_extract({in0}, 4) ? -1 : 0));
           ret.v1 = vec_insert(
                        (u32)(vec_extract({in0}, 5) ? -1 : 0), ret.v1, 1);
           ret.v1 = vec_insert(
                        (u32)(vec_extract({in0}, 6) ? -1 : 0), ret.v1, 2);
           ret.v1 = vec_insert(
                        (u32)(vec_extract({in0}, 7) ? -1 : 0), ret.v1, 3);
           return ret;'''.format(**fmtspec)
    else:
        return 'return ({ppc_to_typ}){in0};'. \
               format(ppc_to_typ=native_typel(to_typ), **fmtspec)

# -----------------------------------------------------------------------------

def convert1(simd_ext, from_typ, to_typ):
    if from_typ == to_typ:
        return 'return {in0};'.format(**fmtspec)
    elif from_typ == 'f16' and to_typ == 'u16':
        return \
        '''return vec_pack((__vector unsigned int)vec_ctu({in0}.v0, 0),
                           (__vector unsigned int)vec_ctu({in0}.v1, 0));'''. \
                           format(**fmtspec)
    elif from_typ == 'f16' and to_typ == 'i16':
        return \
        '''return vec_pack((__vector signed int)vec_cts({in0}.v0, 0),
                           (__vector signed int)vec_cts({in0}.v1, 0));'''. \
                           format(**fmtspec)
    elif from_typ == 'u16' and to_typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vf16 ret;
           /* Unpack extends the sign, we need to remove the extra 1s */
           __vector int mask = vec_splats((int)0xFFFF);
           ret.v0 = vec_ctf(vec_and(vec_unpackh((__vector short){in0}), mask),
                            0);
           ret.v1 = vec_ctf(vec_and(vec_unpackl((__vector short){in0}), mask),
                            0);
           return ret;'''.format(**fmtspec)
    elif from_typ == 'i16' and to_typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = vec_ctf(vec_unpackh({in0}), 0);
                  ret.v1 = vec_ctf(vec_unpackl({in0}), 0);
                  return ret;'''.format(**fmtspec)
    elif has_to_be_emulated(simd_ext, to_typ):
        return '''nsimd_{simd_ext}_v{to_typ} ret;
                  ret.v0 = nsimd_scalar_cvt_{to_typ}_{from_typ}({in0}.v0);
                  ret.v1 = nsimd_scalar_cvt_{to_typ}_{from_typ}({in0}.v1);
                  return ret;'''.format(**fmtspec)
    elif from_typ in ['f32', 'f64'] and to_typ in ['i32', 'i64']:
        return 'return vec_cts({in0}, 0);'.format(**fmtspec)
    elif from_typ in ['f32', 'f64'] and to_typ in ['u32', 'u64']:
        return 'return vec_ctu({in0}, 0);'.format(**fmtspec)
    elif from_typ in ['i32', 'i64', 'u32', 'u64'] and to_typ in ['f32', 'f64']:
        return 'return vec_ctf({in0}, 0);'.format(**fmtspec)
    elif from_typ in common.iutypes and to_typ in common.iutypes:
        return 'return ({ppctyp}){in0};'. \
               format(ppctyp=native_type(to_typ), **fmtspec)

# -----------------------------------------------------------------------------

def reinterpret1(simd_ext, from_typ, to_typ):
    if from_typ == to_typ:
        return 'return {in0};'.format(**fmtspec)
    elif simd_ext == 'vmx' and from_typ in ['f64', 'i64', 'u64']:
        return \
        '''nsimd_{simd_ext}_v{to_typ} ret;
           ret.v0 = nsimd_scalar_reinterpret_{to_typ}_{from_typ}({in0}.v0);
           ret.v1 = nsimd_scalar_reinterpret_{to_typ}_{from_typ}({in0}.v1);
           return ret;'''.format(**fmtspec)
    elif from_typ == 'f16' and to_typ == 'u16':
        return \
        '''nsimd_{simd_ext}_vu16 ret;
           ret = vec_splats(nsimd_f32_to_u16(vec_extract({in0}.v0, 0)));
           ret = vec_insert(nsimd_f32_to_u16(vec_extract({in0}.v0, 1)),
                            ret, 1);
           ret = vec_insert(nsimd_f32_to_u16(vec_extract({in0}.v0, 2)),
                            ret, 2);
           ret = vec_insert(nsimd_f32_to_u16(vec_extract({in0}.v0, 3)),
                            ret, 3);
           ret = vec_insert(nsimd_f32_to_u16(vec_extract({in0}.v1, 1)),
                            ret, 4);
           ret = vec_insert(nsimd_f32_to_u16(vec_extract({in0}.v1, 2)),
                            ret, 5);
           ret = vec_insert(nsimd_f32_to_u16(vec_extract({in0}.v1, 3)),
                            ret, 6);
           ret = vec_insert(nsimd_f32_to_u16(vec_extract({in0}.v1, 4)),
                            ret, 7);
           return ret;'''.format(**fmtspec)
    elif from_typ == 'f16' and to_typ == 'i16':
        return \
        '''nsimd_{simd_ext}_vi16 ret;
           ret = vec_splats(nsimd_scalar_reinterpret_i16_u16(
                     nsimd_f32_to_u16(vec_extract({in0}.v0, 0))));
           ret = vec_insert(nsimd_scalar_reinterpret_i16_u16(
                     nsimd_f32_to_u16(vec_extract({in0}.v0, 1))), ret, 1);
           ret = vec_insert(nsimd_scalar_reinterpret_i16_u16(
                     nsimd_f32_to_u16(vec_extract({in0}.v0, 2))), ret, 2);
           ret = vec_insert(nsimd_scalar_reinterpret_i16_u16(
                     nsimd_f32_to_u16(vec_extract({in0}.v0, 3))), ret, 3);
           ret = vec_insert(nsimd_scalar_reinterpret_i16_u16(
                     nsimd_f32_to_u16(vec_extract({in0}.v1, 0))), ret, 4);
           ret = vec_insert(nsimd_scalar_reinterpret_i16_u16(
                     nsimd_f32_to_u16(vec_extract({in0}.v1, 1))), ret, 5);
           ret = vec_insert(nsimd_scalar_reinterpret_i16_u16(
                     nsimd_f32_to_u16(vec_extract({in0}.v1, 2))), ret, 6);
           ret = vec_insert(nsimd_scalar_reinterpret_i16_u16(
                     nsimd_f32_to_u16(vec_extract({in0}.v1, 3))), ret, 7);
           return ret;'''.format(**fmtspec)
    elif from_typ == 'u16' and to_typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vf16 ret;
           ret.v0 = vec_splats(nsimd_u16_to_f32(vec_extract({in0}, 0)));
           ret.v0 = vec_insert(nsimd_u16_to_f32(vec_extract({in0}, 1)),
                               ret.v0, 1);
           ret.v0 = vec_insert(nsimd_u16_to_f32(vec_extract({in0}, 2)),
                               ret.v0, 2);
           ret.v0 = vec_insert(nsimd_u16_to_f32(vec_extract({in0}, 3)),
                               ret.v0, 3);
           ret.v1 = vec_splats(nsimd_u16_to_f32(vec_extract({in0}, 4)));
           ret.v1 = vec_insert(nsimd_u16_to_f32(vec_extract({in0}, 5)),
                               ret.v1, 1);
           ret.v1 = vec_insert(nsimd_u16_to_f32(vec_extract({in0}, 6)),
                               ret.v1, 2);
           ret.v1 = vec_insert(nsimd_u16_to_f32(vec_extract({in0}, 7)),
                               ret.v1, 3);
           return ret;'''.format(**fmtspec)
    elif from_typ == 'i16' and to_typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vf16 ret;
           ret.v0 = vec_splats(nsimd_u16_to_f32(
                        nsimd_scalar_reinterpret_u16_i16(
                            vec_extract({in0}, 0))));
           ret.v0 = vec_insert(nsimd_u16_to_f32(
                        nsimd_scalar_reinterpret_u16_i16(
                            vec_extract({in0}, 1))), ret.v0, 1);
           ret.v0 = vec_insert(nsimd_u16_to_f32(
                        nsimd_scalar_reinterpret_u16_i16(
                            vec_extract({in0}, 2))), ret.v0, 2);
           ret.v0 = vec_insert(nsimd_u16_to_f32(
                        nsimd_scalar_reinterpret_u16_i16(
                            vec_extract({in0}, 3))), ret.v0, 3);
           ret.v1 = vec_splats(nsimd_u16_to_f32(
                        nsimd_scalar_reinterpret_u16_i16(
                            vec_extract({in0}, 4))));
           ret.v1 = vec_insert(nsimd_u16_to_f32(
                        nsimd_scalar_reinterpret_u16_i16(
                            vec_extract({in0}, 5))), ret.v1, 1);
           ret.v1 = vec_insert(nsimd_u16_to_f32(
                        nsimd_scalar_reinterpret_u16_i16(
                            vec_extract({in0}, 6))), ret.v1, 2);
           ret.v1 = vec_insert(nsimd_u16_to_f32(
                        nsimd_scalar_reinterpret_u16_i16(
                            vec_extract({in0}, 7))), ret.v1, 3);
           return ret;'''.format(**fmtspec)
    else:
        return 'return ({ppc_typ}){in0};'. \
               format(ppc_typ=native_type(to_typ), **fmtspec)

# -----------------------------------------------------------------------------

def reverse1(simd_ext, typ):
    if typ == 'f16':
        return emulate_f16('reverse', simd_ext, ['v', 'v'])
    elif has_to_be_emulated(simd_ext, typ):
        return '''nsimd_{simd_ext}_v{typ} ret;
                  ret.v0 = {in0}.v1;
                  ret.v1 = {in0}.v0;
                  return ret;'''.format(**fmtspec)
    elif typ in ['i8', 'u8']:
        return '''return vec_perm({in0}, {in0}, (__vector unsigned char)
                                  {{ 15, 14, 13, 12, 11, 10, 9, 8,
                                      7,  6,  5,  4,  3,  2, 1, 0 }});'''. \
                                      format(**fmtspec)
    elif typ in ['i16', 'u16']:
        return '''return vec_perm({in0}, {in0}, (__vector unsigned char)
                                  {{ 14, 15, 12, 13, 10, 11, 8, 9,
                                      6,  7,  4,  5,  2,  3, 0, 1 }});'''. \
                                      format(**fmtspec)
    elif typ in ['i32', 'u32', 'f32']:
        return '''return vec_perm({in0}, {in0}, (__vector unsigned char)
                                  {{ 12, 13, 14, 15,  8,  9, 10, 11,
                                      4,  5,  6,  7,  0,  1,  2,  3 }});'''. \
                                      format(**fmtspec)
    elif typ in ['f64', 'i64', 'u64']:
        return '''return vec_perm({in0}, {in0}, (__vector unsigned char)
                                  {{ 8, 9, 10, 11, 12, 13, 14, 15,
                                     0, 1,  2,  3,  4,  5,  6,  7  }});'''. \
                                      format(**fmtspec)

# -----------------------------------------------------------------------------

def addv(simd_ext, typ):
    if typ == 'f16':
        return '''return nsimd_f32_to_f16(
                    nsimd_addv_{simd_ext}_f32({in0}.v0) +
                    nsimd_addv_{simd_ext}_f32({in0}.v1));'''.format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return 'return {in0}.v0 + {in0}.v1;'.format(**fmtspec)
    return 'return ({})({});'. \
           format(typ, ' + '.join('vec_extract({in0}, {i})'. \
                                  format(i=i, **fmtspec) \
                                  for i in range(get_len(typ))))

# -----------------------------------------------------------------------------

def add_sub_s(op, simd_ext, typ):
    if has_to_be_emulated(simd_ext, typ):
        return emulation_code(op, simd_ext, typ, ['v', 'v', 'v'])
    if typ in common.ftypes:
        return 'return vec_add({in0}, {in1});'.format(**fmtspec)
    elif typ in ['i64', 'u64']:
        return '''nsimd_{simd_ext}_v{typ} ret;
                  ret = vec_splats(nsimd_scalar_{op}_{typ}(
                            vec_extract({in0}, 0), vec_extract({in1}, 0)));
                  ret = vec_insert(nsimd_scalar_{op}_{typ}(
                            vec_extract({in0}, 1), vec_extract({in1}, 1)),
                            ret, 1);
                  return ret;'''.format(op=op, **fmtspec)
    return 'return vec_{op}({in0}, {in1});'.format(op=op, **fmtspec)

# -----------------------------------------------------------------------------

def upcvt1(simd_ext, from_typ, to_typ):
    if from_typ in ['i8', 'u8'] and to_typ == 'f16':
        return '''nsimd_{simd_ext}_vf16x2 ret;
                  nsimd_{simd_ext}_vi16x2 tmp;
                  tmp = nsimd_upcvt_{simd_ext}_i16_{from_typ}(a0);
                  ret.v0 = nsimd_cvt_{simd_ext}_f16_i16(tmp.v0);
                  ret.v1 = nsimd_cvt_{simd_ext}_f16_i16(tmp.v1);
                  return ret;'''.format(**fmtspec)
    elif from_typ == 'f16' and to_typ == 'f32':
        return '''nsimd_{simd_ext}_v{to_typ}x2 ret;
                  ret.v0 = {in0}.v0;
                  ret.v1 = {in0}.v1;
                  return ret;'''.format(**fmtspec)
    elif from_typ == 'f16' and to_typ in ['i32', 'u32']:
        sign = 'u' if to_typ[0] == 'u' else 's'
        return '''nsimd_{simd_ext}_v{to_typ}x2 ret;
                  ret.v0 = vec_ct{sign}({in0}.v0, 0);
                  ret.v1 = vec_ct{sign}({in0}.v1, 0);
                  return ret;'''.format(sign=sign, **fmtspec)
    elif from_typ == 'f32' and to_typ in ['f64', 'i64', 'u64']:
        if simd_ext == 'vmx':
            return '''nsimd_vmx_v{to_typ}x2 ret;
                      ret.v0.v0 = ({to_typ})vec_extract({in0}, 0);
                      ret.v0.v1 = ({to_typ})vec_extract({in0}, 1);
                      ret.v1.v0 = ({to_typ})vec_extract({in0}, 2);
                      ret.v1.v1 = ({to_typ})vec_extract({in0}, 3);
                      return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_vsx_v{to_typ}x2 ret;
               ret.v0 = vec_splats(({to_typ})vec_extract({in0}, 0));
               ret.v0 = vec_insert(({to_typ})vec_extract({in0}, 1), ret.v0, 1);
               ret.v1 = vec_splats(({to_typ})vec_extract({in0}, 2));
               ret.v1 = vec_insert(({to_typ})vec_extract({in0}, 3), ret.v0, 1);
               return ret;'''.format(**fmtspec)
    elif (from_typ in ['i16', 'u16'] and to_typ == 'f32') or \
         (from_typ in ['i32', 'u32'] and to_typ == 'f64'):
        return '''nsimd_{simd_ext}_v{to_typ}x2 ret;
                  nsimd_{simd_ext}_v{sto_typ}x2 tmp;
                  tmp = nsimd_upcvt_{simd_ext}_{sto_typ}_{from_typ}({in0});
                  ret.v0 = nsimd_cvt_{simd_ext}_{to_typ}_{sto_typ}(tmp.v0);
                  ret.v1 = nsimd_cvt_{simd_ext}_{to_typ}_{sto_typ}(tmp.v1);
                  return ret;'''.format(sto_typ=from_typ[0] + to_typ[1:],
                                        **fmtspec)
    elif from_typ in ['i8', 'u8', 'i16', 'u16']:
        mask='(i{})0x{}'.format(to_typ[1:], 'F' * (int(from_typ[1:]) // 4))
        ppc_sto_typ = native_type('i' + to_typ[1:])
        ppc_sfrom_typ = '({})'.format(native_type('i' + from_typ[1:])) \
                        if from_typ in common.utypes else ''
        ppc_to_typ = '({})'.format(native_type(to_typ)) \
                     if to_typ in common.utypes else ''
        return '''nsimd_{simd_ext}_v{to_typ}x2 ret;
                  {ppc_sto_typ} mask = vec_splats({mask});
                  ret.v0 = {ppc_to_typ}vec_and(
                               vec_unpackl({ppc_sfrom_typ}{in0}), mask);
                  ret.v1 = {ppc_to_typ}vec_and(
                               vec_unpackh({ppc_sfrom_typ}{in0}), mask);
                  return ret;'''.format(mask=mask, ppc_sto_typ=ppc_sto_typ,
                                        ppc_sfrom_typ=ppc_sfrom_typ,
                                        ppc_to_typ=ppc_to_typ, **fmtspec)
    elif from_typ in ['i32', 'u32']:
        if simd_ext == 'vmx':
            return '''nsimd_vmx_v{to_typ}x2 ret;
                      ret.v0.v0 = ({to_typ})vec_extract({in0}, 0);
                      ret.v0.v1 = ({to_typ})vec_extract({in0}, 1);
                      ret.v1.v0 = ({to_typ})vec_extract({in0}, 2);
                      ret.v1.v1 = ({to_typ})vec_extract({in0}, 3);
                      return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_vsx_v{to_typ}x2 ret;
               ret.v0 = vec_splats(({to_typ})vec_extract({in0}, 0));
               ret.v0 = vec_insert(({to_typ})vec_extract({in0}, 1), ret.v0, 1);
               ret.v1 = vec_splats(({to_typ})vec_extract({in0}, 2));
               ret.v1 = vec_insert(({to_typ})vec_extract({in0}, 3), ret.v1, 1);
               return ret;'''.format(**fmtspec)

# -----------------------------------------------------------------------------

def downcvt1(simd_ext, from_typ, to_typ):
    if from_typ in ['f64', 'i64', 'u64']:
        if simd_ext == 'vmx':
            return '''nsimd_vmx_v{to_typ} ret;
                      ret = vec_splats(({to_typ}){in0}.v0);
                      ret = vec_insert(({to_typ}){in0}.v1, ret, 1);
                      ret = vec_insert(({to_typ}){in1}.v0, ret, 2);
                      ret = vec_insert(({to_typ}){in1}.v1, ret, 3);
                      return ret;'''.format(**fmtspec)
        else:
            return \
            '''nsimd_vmx_v{to_typ} ret;
               ret = vec_splats(({to_typ})vec_extract({in0}, 0));
               ret = vec_insert(({to_typ})vec_extract({in0}, 1), ret, 1);
               ret = vec_insert(({to_typ})vec_extract({in1}, 0), ret, 2);
               ret = vec_insert(({to_typ})vec_extract({in1}, 1), ret, 3);
               return ret;'''.format(**fmtspec)
    elif from_typ in common.iutypes and to_typ in common.iutypes:
        return 'return {cast}vec_pack({in0}, {in1});'. \
               format(cast='({})'.format(native_type(to_typ)) \
                      if from_typ[0] != to_typ[0] else '', **fmtspec)
    elif from_typ == 'f32' and to_typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = {in0};
                  ret.v1 = {in1};
                  return ret;'''.format(**fmtspec)
    elif from_typ == 'f32' and to_typ in common.iutypes:
        return 'return vec_pack(vec_ct{s}({in0}, 0), vec_ct{s}({in1}, 0));'. \
               format(s='s' if to_typ == 'i16' else 'u', **fmtspec)
    elif from_typ in common.iutypes and to_typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = vec_ctf({in0}, 0);
                  ret.v1 = vec_ctf({in1}, 0);
                  return ret;'''.format(**fmtspec)
    elif from_typ == 'f16':
        return \
        '''return vec_pack(vec_pack(vec_ct{s}({in0}.v0, 0),
                                    vec_ct{s}({in0}.v1, 0)),
                           vec_pack(vec_ct{s}({in1}.v0, 0),
                                    vec_ct{s}({in1}.v1, 0)));'''. \
                                    format(s='s' if to_typ == 'i8' else 'u',
                                           **fmtspec)

# -----------------------------------------------------------------------------

def unzip(func, simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = vec_splats(vec_extract({in0}.v0, {i0}));
                  ret.v0 = vec_insert(vec_extract({in0}.v0, {i1}), ret.v0, 1);
                  ret.v0 = vec_insert(vec_extract({in0}.v1, {i2}), ret.v0, 2);
                  ret.v0 = vec_insert(vec_extract({in0}.v1, {i3}), ret.v0, 3);
                  ret.v1 = vec_splats(vec_extract({in1}.v0, {i0}));
                  ret.v1 = vec_insert(vec_extract({in1}.v0, {i1}), ret.v1, 1);
                  ret.v1 = vec_insert(vec_extract({in1}.v1, {i2}), ret.v1, 2);
                  ret.v1 = vec_insert(vec_extract({in1}.v1, {i3}), ret.v1, 3);
                  return ret;'''.format(i0=0 if func == 'unziplo' else 1,
                                        i1=2 if func == 'unziplo' else 3,
                                        i2=0 if func == 'unziplo' else 1,
                                        i3=2 if func == 'unziplo' else 3,
                                        **fmtspec)
    elif simd_ext == 'vmx' and typ in ['f64', 'i64', 'u64']:
        return '''nsimd_vmx_v{typ} ret;
                  ret.v0 = {in0}.v{i};
                  ret.v1 = {in1}.v{i};
                  return ret;'''.format(i=0 if func == 'unziplo' else 1,
                                        **fmtspec)
    elif typ in ['i32', 'u32']:
        return 'return vec_merge{suf}({in0}, {in1});'. \
               format(suf='e' if func == 'unziplo' else 'o', **fmtspec)
    else:
        leo2 = get_len(typ) // 2
        return '''nsimd_{simd_ext}_v{typ} ret;
                  ret = vec_splats(vec_extract({in0}, {i}));'''. \
                  format(i=0 if func == 'unziplo' else 1, **fmtspec) + '\n' + \
               '\n'.join(
                   'ret = vec_insert(vec_extract({in0}, {ix2}), ret, {i});'. \
                   format(i=i, ix2=2 * i if func == 'unziplo' else 2 * i + 1,
                          **fmtspec) for i in range(leo2)) + '\n' + \
               '\n'.join(
                   'ret = vec_insert(vec_extract({in1}, {ix2}), ret, {i});'. \
                   format(i=i + leo2, ix2=2 * i if func == 'unziplo' \
                          else 2 * i + 1, **fmtspec) \
                          for i in range(leo2)) + '\n' + \
               'return ret;'

# -----------------------------------------------------------------------------

def zip(op, simd_ext, typ):
    nbits = get_len(typ)
    if typ == 'f16' or (simd_ext == 'vmx' and typ in ['f64', 'i64', 'u64']):
        return '''nsimd_{simd_ext}_v{typ} ret;
                  ret.v0 = {in0}.v{i};
                  ret.v1 = {in1}.v{i};
                  return ret;'''.format(i='1' if op == 'ziphi' else '0',
                                        **fmtspec)
    return 'return vec_merge{suf}({in0}, {in1});'. \
           format(suf='h' if op == 'ziphi' else 'l', **fmtspec)

# -----------------------------------------------------------------------------

def zip_unzip_basic(op, simd_ext, typ):
    return \
    '''nsimd_{simd_ext}_v{typ}x2 ret;
       ret.v0 = nsimd_{pre}ziplo_{simd_ext}_{typ}({in0}, {in1});
       ret.v1 = nsimd_{pre}ziphi_{simd_ext}_{typ}({in0}, {in1});
       return ret;'''.format(pre='un' if op == 'unzip' else '', **fmtspec)

# -----------------------------------------------------------------------------

def to_mask(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = (__vector float){in0}.v0;
                  ret.v1 = (__vector float){in0}.v1;
                  return ret;'''.format(**fmtspec)
    if simd_ext == 'vmx' and typ in ['f64', 'i64']:
        return '''nsimd_{simd_ext}_v{typ} ret;
                  ret.v0 = nsimd_scalar_reinterpret_{typ}_u64({in0}.v0);
                  ret.v1 = nsimd_scalar_reinterpret_{typ}_u64({in0}.v1);
                  return ret;'''.format(**fmtspec)
    elif simd_ext == 'vmx' and typ == 'u64':
        return '''nsimd_{simd_ext}_vu64 ret;
                  ret.v0 = {in0}.v0;
                  ret.v1 = {in0}.v1;
                  return ret;'''.format(**fmtspec)
    return 'return ({ppc_typ}){in0};'. \
           format(ppc_typ=native_type(typ), **fmtspec)

# -----------------------------------------------------------------------------

def iota(simd_ext, typ):
    if typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = vec_splats(0.0f);
                  ret.v0 = vec_insert(1.0f, ret.v0, 1);
                  ret.v0 = vec_insert(2.0f, ret.v0, 2);
                  ret.v0 = vec_insert(3.0f, ret.v0, 3);
                  ret.v1 = vec_splats(4.0f);
                  ret.v1 = vec_insert(5.0f, ret.v1, 1);
                  ret.v1 = vec_insert(6.0f, ret.v1, 2);
                  ret.v1 = vec_insert(7.0f, ret.v1, 3);
                  return ret;'''.format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return '''nsimd_vmx_v{typ} ret;
                  ret.v0 = ({typ})0;
                  ret.v1 = ({typ})1;
                  return ret;'''.format(**fmtspec)
    return 'nsimd_{simd_ext}_v{typ} ret;\n' \
           'ret = vec_splats(({typ})0);\n'.format(**fmtspec) + \
           '\n'.join('ret = vec_insert(({}){}, ret, {});'.format(typ, i, i) \
                     for i in range(1, get_len(typ))) + \
           '\nreturn ret;'

# -----------------------------------------------------------------------------

def mask_for_loop_tail(simd_ext, typ):
    le = get_len(typ)
    if typ == 'f16':
        threshold = 'nsimd_f32_to_f16((f32)({in1} - {in0}))'.format(**fmtspec)
    else:
        threshold = '({typ})({in1} - {in0})'.format(**fmtspec)
    return '''if ({in0} >= {in1}) {{
                return nsimd_set1l_{simd_ext}_{typ}(0);
              }}
              if ({in1} - {in0} < {le}) {{
                nsimd_{simd_ext}_v{typ} n =
                      nsimd_set1_{simd_ext}_{typ}({threshold});
                return nsimd_lt_{simd_ext}_{typ}(
                           nsimd_iota_{simd_ext}_{typ}(), n);
              }} else {{
                return nsimd_set1l_{simd_ext}_{typ}(1);
              }}'''.format(le=le, threshold=threshold, **fmtspec)

# -----------------------------------------------------------------------------

def scatter(simd_ext, typ):
    le = get_len(typ)
    if typ == 'f16':
        return \
        '''{in0}[vec_extract({in1}, 0)] = nsimd_f32_to_f16(
                                              vec_extract({in2}.v0, 0));
           {in0}[vec_extract({in1}, 1)] = nsimd_f32_to_f16(
                                              vec_extract({in2}.v0, 1));
           {in0}[vec_extract({in1}, 2)] = nsimd_f32_to_f16(
                                              vec_extract({in2}.v0, 2));
           {in0}[vec_extract({in1}, 3)] = nsimd_f32_to_f16(
                                              vec_extract({in2}.v0, 3));
           {in0}[vec_extract({in1}, 4)] = nsimd_f32_to_f16(
                                              vec_extract({in2}.v1, 0));
           {in0}[vec_extract({in1}, 5)] = nsimd_f32_to_f16(
                                              vec_extract({in2}.v1, 1));
           {in0}[vec_extract({in1}, 6)] = nsimd_f32_to_f16(
                                              vec_extract({in2}.v1, 2));
           {in0}[vec_extract({in1}, 7)] = nsimd_f32_to_f16(
                                              vec_extract({in2}.v1, 3));'''. \
                                              format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return '''{in0}[{in1}.v0] = {in2}.v0;
                  {in0}[{in1}.v1] = {in2}.v1;'''.format(**fmtspec)
    return '\n'.join(['{in0}[vec_extract({in1}, {i})] = ' \
                      'vec_extract({in2}, {i});'.format(i=i, **fmtspec) \
                      for i in range(get_len(typ))])

# -----------------------------------------------------------------------------

def gather(simd_ext, typ):
    if typ == 'f16':
        return \
        '''nsimd_{simd_ext}_v{typ} ret;
           ret.v0 = vec_splats(nsimd_f16_to_f32({in0}[vec_extract({in1}, 0)]));
           ret.v0 = vec_insert(nsimd_f16_to_f32({in0}[vec_extract({in1}, 0)]),
                               ret.v0, 1);
           ret.v0 = vec_insert(nsimd_f16_to_f32({in0}[vec_extract({in1}, 0)]),
                               ret.v0, 2);
           ret.v0 = vec_insert(nsimd_f16_to_f32({in0}[vec_extract({in1}, 0)]),
                               ret.v0, 3);
           ret.v1 = vec_splats(nsimd_f16_to_f32({in0}[vec_extract({in1}, 0)]));
           ret.v1 = vec_insert(nsimd_f16_to_f32({in0}[vec_extract({in1}, 0)]),
                               ret.v1, 1);
           ret.v1 = vec_insert(nsimd_f16_to_f32({in0}[vec_extract({in1}, 0)]),
                               ret.v1, 2);
           ret.v1 = vec_insert(nsimd_f16_to_f32({in0}[vec_extract({in1}, 0)]),
                               ret.v1, 3);
           return ret;'''.format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return '''nsimd_{simd_ext}_v{typ} ret;
                  ret.v0 = {in0}[{in1}.v0];
                  ret.v0 = {in0}[{in1}.v1];
                  return ret;'''.format(**fmtspec)
    return '''nsimd_{simd_ext}_v{typ} ret;
              ret = vec_splats({in0}[vec_extract({in1}, 0)]);
              '''.format(**fmtspec) + \
           '\n'.join('ret = vec_insert({in1}[vec_extract({in1}, {i})], ' \
                     'ret, {i});'.format(i=i, **fmtspec) \
                     for i in range(1, get_len(typ))) + '\n' + \
           'return ret;'

# -----------------------------------------------------------------------------

def gather_linear(simd_ext, typ):
    if typ == 'f16':
        return \
        '''nsimd_{simd_ext}_v{typ} ret;
           ret.v0 = vec_splats(nsimd_f16_to_f32({in0}[0]));
           ret.v0 = vec_insert(nsimd_f16_to_f32({in0}[{in1}]), ret.v0, 1);
           ret.v0 = vec_insert(nsimd_f16_to_f32({in0}[2 * {in1}]), ret.v0, 2);
           ret.v0 = vec_insert(nsimd_f16_to_f32({in0}[3 * {in1}]), ret.v0, 3);
           ret.v1 = vec_splats(nsimd_f16_to_f32({in0}[4 * {in1}]));
           ret.v1 = vec_insert(nsimd_f16_to_f32({in0}[5 * {in1}]), ret.v1, 1);
           ret.v1 = vec_insert(nsimd_f16_to_f32({in0}[6 * {in1}]), ret.v1, 2);
           ret.v1 = vec_insert(nsimd_f16_to_f32({in0}[7 * {in1}]), ret.v1, 3);
           return ret;'''.format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return '''nsimd_{simd_ext}_v{typ} ret;
                  ret.v0 = {in0}[0];
                  ret.v1 = {in0}[{in1}];
                  return ret;'''.format(**fmtspec)
    return '''nsimd_{simd_ext}_v{typ} ret;
           ret = vec_splats({in0}[0]);
           '''.format(**fmtspec) + \
        '\n'.join('ret = vec_insert({in0}[{in1} * {i}], ret, {i});'. \
                  format(i=i, **fmtspec) for i in range(1, get_len(typ))) + \
        '\nreturn ret;'

# -----------------------------------------------------------------------------

def scatter_linear(simd_ext, typ):
    if typ == 'f16':
        return \
        '''{in0}[0] = nsimd_f32_to_f16(vec_extract({in2}.v0, 0));
           {in0}[{in1}] = nsimd_f32_to_f16(vec_extract({in2}.v0, 1));
           {in0}[2 * {in1}] = nsimd_f32_to_f16(vec_extract({in2}.v0, 2));
           {in0}[3 * {in1}] = nsimd_f32_to_f16(vec_extract({in2}.v0, 3));
           {in0}[4 * {in1}] = nsimd_f32_to_f16(vec_extract({in2}.v1, 0));
           {in0}[5 * {in1}] = nsimd_f32_to_f16(vec_extract({in2}.v1, 1));
           {in0}[6 * {in1}] = nsimd_f32_to_f16(vec_extract({in2}.v1, 2));
           {in0}[7 * {in1}] = nsimd_f32_to_f16(vec_extract({in2}.v1, 3));'''. \
           format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return '''{in0}[0] = {in2}.v0;
                  {in0}[{in1}] = {in2}.v1;'''.format(**fmtspec)
    return '\n'.join(['{in0}[{in1} * {i}] = vec_extract({in2}, {i});'. \
                      format(i=i, **fmtspec) for i in range(get_len(typ))])

# -----------------------------------------------------------------------------

def maskoz_load(oz, simd_ext, typ):
    if typ == 'f16':
        return \
        '''nsimd_{simd_ext}_vf16 ret;
           ret.v0 = vec_splats(0.0f);
           ret.v0 = vec_insert(vec_extract({in0}.v0, 0) ?
                               nsimd_f16_to_f32({in1}[0]) : {oz0}, ret.v0, 0);
           ret.v0 = vec_insert(vec_extract({in0}.v0, 1) ?
                               nsimd_f16_to_f32({in1}[1]) : {oz1}, ret.v0, 1);
           ret.v0 = vec_insert(vec_extract({in0}.v0, 2) ?
                               nsimd_f16_to_f32({in1}[2]) : {oz2}, ret.v0, 2);
           ret.v0 = vec_insert(vec_extract({in0}.v0, 3) ?
                               nsimd_f16_to_f32({in1}[3]) : {oz3}, ret.v0, 3);
           ret.v1 = ret.v0;
           ret.v1 = vec_insert(vec_extract({in0}.v0, 4) ?
                               nsimd_f16_to_f32({in1}[4]) : {oz4}, ret.v1, 0);
           ret.v1 = vec_insert(vec_extract({in0}.v0, 5) ?
                               nsimd_f16_to_f32({in1}[5]) : {oz5}, ret.v1, 1);
           ret.v1 = vec_insert(vec_extract({in0}.v0, 6) ?
                               nsimd_f16_to_f32({in1}[6]) : {oz6}, ret.v1, 2);
           ret.v1 = vec_insert(vec_extract({in0}.v0, 7) ?
                               nsimd_f16_to_f32({in1}[7]) : {oz7}, ret.v1, 3);
           return ret;'''. \
           format(oz0='0.0f' if oz == 'z' else 'vec_extract({in2}.v0, 0)',
                  oz1='0.0f' if oz == 'z' else 'vec_extract({in2}.v0, 1)',
                  oz2='0.0f' if oz == 'z' else 'vec_extract({in2}.v0, 2)',
                  oz3='0.0f' if oz == 'z' else 'vec_extract({in2}.v0, 3)',
                  oz4='0.0f' if oz == 'z' else 'vec_extract({in2}.v1, 0)',
                  oz5='0.0f' if oz == 'z' else 'vec_extract({in2}.v1, 1)',
                  oz6='0.0f' if oz == 'z' else 'vec_extract({in2}.v1, 2)',
                  oz7='0.0f' if oz == 'z' else 'vec_extract({in2}.v1, 3)',
                  **fmtspec).format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        if oz == 'z':
            return '''nsimd_{simd_ext}_v{typ} ret;
                      ret.v0 = {in0}.v0 ? {in1}[0] : ({typ})0;
                      ret.v1 = {in0}.v1 ? {in1}[1] : ({typ})0;
                      return ret;'''.format(**fmtspec)
        else:
            return '''nsimd_{simd_ext}_v{typ} ret;
                      ret.v0 = {in0}.v0 ? {in1}[0] : {in2}.v0;
                      ret.v1 = {in0}.v1 ? {in1}[1] : {in2}.v1;
                      return ret;'''.format(**fmtspec)
    return 'nsimd_{simd_ext}_v{typ} ret = {zeros};\n'.format(**fmtspec) + \
           '\n'.join(
           '''if (vec_extract({in0}, {i})) {{
                ret = vec_insert({in1}[{i}], ret, {i});
              }} else {{
                ret = vec_insert({v}, ret, {i});
              }}'''.format(i=i, v='({})0'.format(typ) if oz == 'z' \
                                else 'vec_extract({in2}, {i})'. \
                                     format(i=i, **fmtspec), **fmtspec) \
                                     for i in range(get_len(typ))) + \
                                     '\nreturn ret;'

# -----------------------------------------------------------------------------

def mask_store(simd_ext, typ):
    if typ == 'f16':
        return \
        '''if (vec_extract({in0}.v0, 0)) {{
             {in1}[0] = nsimd_f32_to_f16(vec_extract({in2}.v0, 0));
           }}
           if (vec_extract({in0}.v0, 1)) {{
             {in1}[1] = nsimd_f32_to_f16(vec_extract({in2}.v0, 1));
           }}
           if (vec_extract({in0}.v0, 2)) {{
             {in1}[2] = nsimd_f32_to_f16(vec_extract({in2}.v0, 2));
           }}
           if (vec_extract({in0}.v0, 3)) {{
             {in1}[3] = nsimd_f32_to_f16(vec_extract({in2}.v0, 3));
           }}
           if (vec_extract({in0}.v1, 0)) {{
             {in1}[4] = nsimd_f32_to_f16(vec_extract({in2}.v1, 0));
           }}
           if (vec_extract({in0}.v1, 1)) {{
             {in1}[5] = nsimd_f32_to_f16(vec_extract({in2}.v1, 1));
           }}
           if (vec_extract({in0}.v1, 2)) {{
             {in1}[6] = nsimd_f32_to_f16(vec_extract({in2}.v1, 2));
           }}
           if (vec_extract({in0}.v1, 3)) {{
             {in1}[7] = nsimd_f32_to_f16(vec_extract({in2}.v1, 3));
           }}'''.format(**fmtspec)
    elif has_to_be_emulated(simd_ext, typ):
        return '''if ({in0}.v0) {{
                    {in1}[0] = {in2}.v0;
                  }}
                  if ({in0}.v1) {{
                    {in1}[1] = {in2}.v1;
                  }}'''.format(**fmtspec)
    return '\n'.join(
           '''if (vec_extract({in0}, {i})) {{
                {in1}[{i}] = vec_extract({in2}, {i});
              }}'''.format(i=i, **fmtspec) for i in range(get_len(typ)))

# -----------------------------------------------------------------------------

def to_logical(simd_ext, typ):
    if typ == 'f16':
        return emulate_f16('to_logical', simd_ext, ['l', 'v'])
    elif has_to_be_emulated(simd_ext, typ):
        if typ in ['i64', 'u64']:
            return '''nsimd_{simd_ext}_vl{typ} ret;
                      ret.v0 = (u64)({in0}.v0 != ({typ})0 ? -1 : 0);
                      ret.v1 = (u64)({in0}.v1 != ({typ})0 ? -1 : 0);
                      return ret;'''.format(**fmtspec)
        elif typ == 'f64':
            return '''nsimd_{simd_ext}_vl{typ} ret;
                      ret.v0 = (u64)(nsimd_scalar_reinterpret_u64_f64(
                                       {in0}.v0) != (u64)0 ? -1 : 0);
                      ret.v1 = (u64)(nsimd_scalar_reinterpret_u64_f64(
                                       {in0}.v1) != (u64)0 ? -1 : 0);
                      return ret;'''.format(**fmtspec)
    elif typ in common.iutypes:
        return 'return nsimd_ne_{simd_ext}_{typ}({in0}, {zeros});'. \
               format(**fmtspec)
    elif typ in ['f32', 'f64']:
        return '''return nsimd_ne_{simd_ext}_u{typnbits}(
                             nsimd_reinterpret_{simd_ext}_u{typnbits}_{typ}(
                                 {in0}), vec_splats((u{typnbits})0));'''. \
                                 format(**fmtspec)

# -----------------------------------------------------------------------------

def get_impl(opts, func, simd_ext, from_typ, to_typ):
    global fmtspec

    fmtspec = {
        'simd_ext': simd_ext,
        'typ': from_typ,
        'styp': get_type(opts, simd_ext, from_typ, to_typ),
        'from_typ': from_typ,
        'to_typ': to_typ,
        'in0': common.in0,
        'in1': common.in1,
        'in2': common.in2,
        'in3': common.in3,
        'in4': common.in4,
        'in5': common.in5,
        'zeros': 'vec_splats(({})0)'.format(from_typ),
        'lzeros': '({})vec_splats((u{})0)'. \
                  format(native_typel(from_typ), from_typ[1:]) \
                  if not has_to_be_emulated(simd_ext, from_typ) else '',
        'typnbits': from_typ[1:]
    }

    impls = {
        'loada': 'load1234(simd_ext, from_typ, 1, True)',
        'load2a': 'load1234(simd_ext, from_typ, 2, True)',
        'load3a': 'load1234(simd_ext, from_typ, 3, True)',
        'load4a': 'load1234(simd_ext, from_typ, 4, True)',
        'loadu': 'load1234(simd_ext, from_typ, 1, False)',
        'load2u': 'load1234(simd_ext, from_typ, 2, False)',
        'load3u': 'load1234(simd_ext, from_typ, 3, False)',
        'load4u': 'load1234(simd_ext, from_typ, 4, False)',
        'storea': 'store1234(simd_ext, from_typ, 1, True)',
        'store2a': 'store1234(simd_ext, from_typ, 2, True)',
        'store3a': 'store1234(simd_ext, from_typ, 3, True)',
        'store4a': 'store1234(simd_ext, from_typ, 4, True)',
        'storeu': 'store1234(simd_ext, from_typ, 1, False)',
        'store2u': 'store1234(simd_ext, from_typ, 2, False)',
        'store3u': 'store1234(simd_ext, from_typ, 3, False)',
        'store4u': 'store1234(simd_ext, from_typ, 4, False)',
        'andb': 'binary_op2("andb", simd_ext, from_typ)',
        'xorb': 'binary_op2("xorb", simd_ext, from_typ)',
        'orb': 'binary_op2("orb", simd_ext, from_typ)',
        'andl': 'logical_op2("andl", simd_ext, from_typ)',
        'xorl': 'logical_op2("xorl", simd_ext, from_typ)',
        'orl': 'logical_op2("orl", simd_ext, from_typ)',
        'notb': 'not1(simd_ext, from_typ)',
        'notl': 'lnot1(simd_ext, from_typ)',
        'andnotb': 'binary_op2("andnotb", simd_ext, from_typ)',
        'andnotl': 'logical_op2("andnotl", simd_ext, from_typ)',
        'add': 'simple_op2("add", simd_ext, from_typ)',
        'adds': 'add_sub_s("adds",simd_ext, from_typ)',
        'sub': 'simple_op2("sub", simd_ext, from_typ)',
        'subs': 'add_sub_s("subs",simd_ext, from_typ)',
        'div': 'div2(simd_ext, from_typ)',
        'sqrt': 'sqrt1(simd_ext, from_typ)',
        'len': 'len1(simd_ext, from_typ)',
        'mul': 'simple_op2("mul", simd_ext, from_typ)',
        'shl': 'shift2("shl", simd_ext, from_typ)',
        'shr': 'shift2("shr", simd_ext, from_typ)',
        'shra': 'shift2("shra", simd_ext, from_typ)',
        'set1': 'set1(simd_ext, from_typ)',
        'set1l': 'lset1(simd_ext, from_typ)',
        'eq': 'cmp2("eq", simd_ext, from_typ)',
        'lt': 'cmp2("lt", simd_ext, from_typ)',
        'le': 'cmp2("le", simd_ext, from_typ)',
        'gt': 'cmp2("gt", simd_ext, from_typ)',
        'ge': 'cmp2("ge", simd_ext, from_typ)',
        'ne': 'cmp2("ne", simd_ext, from_typ)',
        'if_else1': 'if_else3(simd_ext, from_typ)',
        'min': 'minmax2("min", simd_ext, from_typ)',
        'max': 'minmax2("max", simd_ext, from_typ)',
        'loadla': 'loadl(True, simd_ext, from_typ)',
        'loadlu': 'loadl(False, simd_ext, from_typ)',
        'storela': 'storel(True, simd_ext, from_typ)',
        'storelu': 'storel(False, simd_ext, from_typ)',
        'abs': 'abs1(simd_ext, from_typ)',
        'fma': 'fma("fma", simd_ext, from_typ)',
        'fnma': 'fma("fnma", simd_ext, from_typ)',
        'fms': 'fma("fms", simd_ext, from_typ)',
        'fnms': 'fma("fnms", simd_ext, from_typ)',
        'ceil': 'round1("ceil", simd_ext, from_typ)',
        'floor': 'round1("floor", simd_ext, from_typ)',
        'trunc': 'round1("trunc", simd_ext, from_typ)',
        'round_to_even': 'round1("round_to_even", simd_ext, from_typ)',
        'all': 'allany1("all", simd_ext, from_typ)',
        'any': 'allany1("any", simd_ext, from_typ)',
        'reinterpret': 'reinterpret1(simd_ext, from_typ, to_typ)',
        'reinterpretl': 'reinterpretl1(simd_ext, from_typ, to_typ)',
        'cvt': 'convert1(simd_ext, from_typ, to_typ)',
        'rec8': 'recs1("rec8", simd_ext, from_typ)',
        'rec11': 'recs1("rec11", simd_ext, from_typ)',
        'rsqrt8': 'recs1("rsqrt8", simd_ext, from_typ)',
        'rsqrt11': 'recs1("rsqrt11", simd_ext, from_typ)',
        'rec': 'recs1("rec", simd_ext, from_typ)',
        'neg': 'neg1(simd_ext, from_typ)',
        'nbtrue': 'nbtrue1(simd_ext, from_typ)',
        'reverse': 'reverse1(simd_ext, from_typ)',
        'addv': 'addv(simd_ext, from_typ)',
        'upcvt': 'upcvt1(simd_ext, from_typ, to_typ)',
        'downcvt': 'downcvt1(simd_ext, from_typ, to_typ)',
        'iota': 'iota(simd_ext, from_typ)',
        'to_logical': 'to_logical(simd_ext, from_typ)',
        'mask_for_loop_tail': 'mask_for_loop_tail(simd_ext, from_typ)',
        'masko_loadu1': 'maskoz_load("o", simd_ext, from_typ)',
        'maskz_loadu1': 'maskoz_load("z", simd_ext, from_typ)',
        'masko_loada1': 'maskoz_load("o", simd_ext, from_typ)',
        'maskz_loada1': 'maskoz_load("z", simd_ext, from_typ)',
        'mask_storea1': 'mask_store(simd_ext, from_typ)',
        'mask_storeu1': 'mask_store(simd_ext, from_typ)',
        'gather': 'gather(simd_ext, from_typ)',
        'scatter': 'scatter(simd_ext, from_typ)',
        'gather_linear': 'gather_linear(simd_ext, from_typ)',
        'scatter_linear': 'scatter_linear(simd_ext, from_typ)',
        'to_mask': 'to_mask(simd_ext, from_typ)',
        'ziplo': 'zip("ziplo", simd_ext, from_typ)',
        'ziphi': 'zip("ziphi", simd_ext, from_typ)',
        'zip': 'zip_unzip_basic("zip", simd_ext, from_typ)',
        'unzip': 'zip_unzip_basic("unzip", simd_ext, from_typ)',
        'unziplo': 'unzip("unziplo", simd_ext, from_typ)',
        'unziphi': 'unzip("unziphi", simd_ext, from_typ)'
    }
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if not from_typ in common.types:
        raise ValueError('Unknown type "{}"'.format(from_typ))
    if not func in impls:
        return common.NOT_IMPLEMENTED
    else:
        return eval(impls[func])
