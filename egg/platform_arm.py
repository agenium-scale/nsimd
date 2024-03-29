# Copyright (c) 2020 Agenium Scale
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

# This file gives the implementation of platform ARM, i.e. ARM SIMD.
# Reading this file is rather straightforward. ARM SIMD extensions are rather
# coherent and consistent. It implements the following architectures:
#   - ARMv7   -> 128 bits registers without f16 and f64 support
#   - Aarch32 -> 128 bits registers with optional f16 and without f64 support
#   - Aarch64 -> 128 bits registers with optional f16 and f64 support
#   - SVE     -> up to 2048 bits registers
# The first three SIMD extensions are collectively called NEON. Aarch32 and
# Aarch64 correspond respectively to ARMv8 32 and 64 bits chips. Note that
# the ARM documentation says that ARMv7, Aarch32 are different but it seems
# that they differ by only a handful of intrinsics which are not in the scope
# of NSIMD so we have implemented the following:
#
#   - ARMv7   \  -> neon128
#   - Aarch32 /
#   - Aarch64    -> aarch64
#   - SVE        -> sve

import common

# -----------------------------------------------------------------------------
# Helpers


def neon_typ(typ):
    prefix = {'i': 'int', 'u': 'uint', 'f': 'float'}
    return '{}{}x{}_t'.format(prefix[typ[0]], typ[1:], 128 // int(typ[1:]))

def half_neon64_typ(typ):
    prefix = {'i': 'int', 'u': 'uint', 'f': 'float'}
    return '{}{}x{}_t'.format(prefix[typ[0]], typ[1:], 64 // int(typ[1:]))


def sve_typ(typ):
    prefix = {'i': 'svint', 'u': 'svuint', 'f': 'svfloat'}
    return '{}{}_t'.format(prefix[typ[0]], typ[1:])

def suf(typ):
    if typ[0] == 'i':
        return 's{}'.format(typ[1:])
    else:
        return typ

neon = ['neon128', 'aarch64']
fixed_sized_sve = ['sve128', 'sve256', 'sve512', 'sve1024', 'sve2048']
sve = ['sve'] + fixed_sized_sve
fmtspec = {}

def convert_from_predicate(opts, op):
    if opts.sve_emulate_bool:
        return '''svsel({op},
                    svdup_n_u{typnbits}_x({svtrue}, (u{typnbits})~0),
                    svdup_n_u{typnbits}_x({svtrue}, 0))'''. \
                            format(op=op, **fmtspec)
    else:
        return op

def convert_to_predicate(opts, op):
    if opts.sve_emulate_bool:
        # TODO: the casts are a workaround to avoid a bug in gcc trunk for sve
        # it needs to be deleted when the bug is corrected
        return '''svcmpeq({svtrue}, (svuint{typnbits}_t){op},
                          svdup_n_u{typnbits}_x({svtrue},
                          (u{typnbits})~0))'''.format(op=op, **fmtspec)
    else:
        return op

# -----------------------------------------------------------------------------
# Implementation of mandatory functions for this module

def get_simd_exts():
    return ['neon128', 'aarch64', 'sve', 'sve128', 'sve256', 'sve512',
            'sve1024', 'sve2048']

def get_prev_simd_ext(simd_ext):
    if simd_ext in ['neon128', 'aarch64']:
        return 'cpu'
    elif simd_ext in sve:
        return 'aarch64'
    raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def emulate_fp16(simd_ext):
    if not simd_ext in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if simd_ext in sve:
        return False
    else:
        return True

def get_type(opts, simd_ext, typ, nsimd_typ):
    if simd_ext in neon:
        if typ == 'f64':
            if simd_ext == 'neon128':
                return 'typedef struct {{ double v0; double v1; }} {};'. \
                       format(nsimd_typ)
            else:
                return 'typedef {} {};'.format(neon_typ('f64'), nsimd_typ)
        elif typ == 'f16':
            return '''
                   #ifdef NSIMD_ARM_FP16
                     typedef float16x8_t {nsimd_typ};
                   #else
                     typedef struct {{ float32x4_t v0; float32x4_t v1; }}
                         {nsimd_typ};
                   #endif
                   '''.format(nsimd_typ=nsimd_typ) # extra \n are necessary
        else:
            return 'typedef {} {};'.format(neon_typ(typ), nsimd_typ)
    elif simd_ext == 'sve':
        return 'typedef {} {};'.format(sve_typ(typ), nsimd_typ)
    elif simd_ext in fixed_sized_sve:
        return 'typedef {} {} __attribute__((arm_sve_vector_bits({})));'. \
               format(sve_typ(typ), nsimd_typ, simd_ext[3:])
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

def get_logical_type(opts, simd_ext, typ, nsimd_typ):
    if typ not in common.types:
        raise ValueError('Unknown type "{}"'.format(typ))
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))

    if typ in common.ftypes + common.itypes:
        typ2 = 'u{}'.format(typ[1:]);
    else:
        typ2 = typ

    if simd_ext == 'neon128':
        if typ == 'f16':
            return \
            '''
            #ifdef NSIMD_ARM_FP16
              typedef uint16x8_t {nsimd_typ};
            #else
              typedef struct {{ uint32x4_t v0; uint32x4_t v1; }} {nsimd_typ};
            #endif
            '''.format(nsimd_typ=nsimd_typ) # extra \n are necessary
        elif typ == 'f64':
            return 'typedef struct {{ u64 v0; u64 v1; }} {};'.format(nsimd_typ)
        else:
            return get_type(opts, simd_ext, typ2, nsimd_typ)
    if simd_ext == 'aarch64':
        if typ == 'f16':
            return get_logical_type(opts, 'neon128', 'f16', nsimd_typ)
        else:
            return get_type(opts, simd_ext, typ2, nsimd_typ)
    elif simd_ext in sve:
        if opts.sve_emulate_bool:
            return get_type(opts, simd_ext, 'u' + typ[1:], nsimd_typ)
        elif simd_ext in fixed_sized_sve:
            return \
            'typedef svbool_t {} __attribute__((arm_sve_vector_bits({})));'. \
            format(nsimd_typ, simd_ext[3:])
        else:
            return 'typedef svbool_t {};'.format(nsimd_typ)

def get_nb_registers(simd_ext):
    if simd_ext in neon:
        return '16'
    elif simd_ext in sve:
        return '32'
    else:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))


def get_native_soa_typ(simd_ext, typ, deg):
    prefix = { 'i': 'int', 'u': 'uint', 'f': 'float' }[typ[0]]
    if simd_ext in sve:
        return 'sv{}x{}_t'.format(prefix + typ[1:], deg)
    else:
        return '{}{}x{}x{}_t'.format(prefix, typ[1:], 128 // int(typ[1:]),
                                     deg)


def get_SoA_type(simd_ext, typ, deg, nsimd_typ):
    if simd_ext != 'sve':
        raise ValueError('SIMD extension must be "sve"')
    prefix = { 'i': 'int', 'u': 'uint', 'f': 'float' }[typ[0]]
    return 'typedef {} {};'.format(get_native_soa_typ(simd_ext, typ, deg),
                                   nsimd_typ)


def has_compatible_SoA_types(simd_ext):
    if simd_ext not in neon + sve:
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    return False

# -----------------------------------------------------------------------------

def get_additional_include(func, platform, simd_ext):
    ret = '''#include <nsimd/cpu/cpu/{}.h>
             '''.format(func)
    if simd_ext in sve:
        ret += '''#include <nsimd/arm/aarch64/{}.h>
                  '''.format(func)
    if func in ['load2u', 'load3u', 'load4u', 'load2a', 'load3a', 'load4a']:
        deg = func[4]
        ret += '''#if NSIMD_CXX > 0
                  extern "C" {{
                  #endif

                  NSIMD_INLINE nsimd_{simd_ext}_vu16x{deg}
                  nsimd_{func}_{simd_ext}_u16(const u16*);

                  # if NSIMD_CXX > 0
                  }} // extern "C"
                  #endif

                  '''.format(func=func, deg=deg, simd_ext=simd_ext)
    if func in ['mask_storea1', 'mask_storeu1', 'masko_loada1',
                'masko_loadu1', 'maskz_loada1', 'maskz_loadu1'] and \
                simd_ext not in sve:
        ret += '''#include <nsimd/scalar_utilities.h>
                  '''
    if func == 'mask_for_loop_tail' and simd_ext not in sve:
        ret += '''#include <nsimd/arm/{simd_ext}/set1.h>
                  #include <nsimd/arm/{simd_ext}/set1l.h>
                  #include <nsimd/arm/{simd_ext}/iota.h>
                  #include <nsimd/arm/{simd_ext}/lt.h>
                  '''.format(simd_ext=simd_ext)
    if simd_ext == 'neon128' and func == 'notl':
        ret += '''#include <nsimd/arm/neon128/notb.h>
                  '''
    if simd_ext in neon and func == 'ne':
        ret += '''#include <nsimd/arm/{simd_ext}/eq.h>
                  # include <nsimd/arm/{simd_ext}/notl.h>
                  '''.format(simd_ext=simd_ext)
    if simd_ext in neon and func in ['fms', 'fnms']:
        ret += '''#include <nsimd/arm/{simd_ext}/ne.h>
                  #include <nsimd/arm/{simd_ext}/fma.h>
                  #include <nsimd/arm/{simd_ext}/fnma.h>
                  '''.format(simd_ext=simd_ext)
    if func == 'shra':
        ret += '''#include <nsimd/arm/{simd_ext}/shr.h>
        '''.format(simd_ext=simd_ext)

    if func in ['loadlu', 'loadla']:
        ret += '''#include <nsimd/arm/{simd_ext}/eq.h>
                  # include <nsimd/arm/{simd_ext}/set1.h>
                  # include <nsimd/arm/{simd_ext}/{load}.h>
                  # include <nsimd/arm/{simd_ext}/notl.h>
                  '''.format(load='load' + func[5], simd_ext=simd_ext)
    if func in ['storelu', 'storela']:
        ret += '''#include <nsimd/arm/{simd_ext}/if_else1.h>
                  # include <nsimd/arm/{simd_ext}/set1.h>
                  # include <nsimd/arm/{simd_ext}/{store}.h>
                  '''.format(store='store' + func[6], simd_ext=simd_ext)
    if func == 'to_logical':
        ret += '''#include <nsimd/arm/{simd_ext}/reinterpret.h>
                  #include <nsimd/arm/{simd_ext}/ne.h>
                  ''' .format(simd_ext=simd_ext)
    if func == 'zip':
        ret += '''#include <nsimd/arm/{simd_ext}/ziplo.h>
                  #include <nsimd/arm/{simd_ext}/ziphi.h>
                  '''.format(simd_ext=simd_ext)
    if func == 'unzip':
        ret += '''#include <nsimd/arm/{simd_ext}/unziplo.h>
                  #include <nsimd/arm/{simd_ext}/unziphi.h>
                  '''.format(simd_ext=simd_ext)
    if func == 'adds':
        ret += '''#include <nsimd/arm/{simd_ext}/add.h>
                  '''.format(simd_ext=simd_ext)
    if func == 'subs':
        ret += '''#include <nsimd/arm/{simd_ext}/sub.h>
                  '''.format(simd_ext=simd_ext)
    if func in ['gather', 'scatter'] and simd_ext == 'sve':
        ret += '''#include <nsimd/arm/sve/len.h>
                  '''
    return ret

# -----------------------------------------------------------------------------
# Emulators

def emulate_op1(op, simd_ext, typ):
    if simd_ext in neon:
        le = 128 // int(typ[1:]);
        return '''int i;
                  {typ} buf[{le}];
                  vst1q_{suf}(buf, {in0});
                  for (i=0; i < {le}; i += nsimd_len_cpu_{typ}()) {{
                    nsimd_storeu_cpu_{typ}( & buf[i], nsimd_{op}_cpu_{typ}(
                      nsimd_loadu_cpu_{typ}(&buf[i])));}}
                  return vld1q_{suf}(buf); '''. \
                  format(op=op, le=le, **fmtspec)
    if simd_ext in sve:
        le = 2048 // int(typ[1:]);
        return '''int i;
                  {typ} buf[{le}];
                  svst1_{suf}({svtrue}, buf, {in0});
                  for (i=0; i < simd_len_{simd_ext}_{typ}();
                       i += nsimd_len_cpu_{typ}()) {{
                    nsimd_storeu_cpu_{typ}( & buf[i], nsimd_{op}_cpu_{typ}(
                      nsimd_loadu_cpu_{typ}(&buf[i])));}}
                  return svld1_{suf}({svtrue}, buf); '''. \
                  format(op=op, le=le, **fmtspec)

def emulate_op2(op, simd_ext, typ):
    if simd_ext in neon:
        le = 128 // int(typ[1:]);
        return '''int i;
                  {typ} buf0[{le}], buf1[{le}];
                  vst1q_{suf}(buf0, {in0});
                  vst1q_{suf}(buf1, {in1});
                  for (i=0; i < {le}; i++) {{
                    buf0[i] = ({typ})(buf0[i] {op} buf1[i]);}}
                  return vld1q_{suf}(buf0); '''. \
                  format(op=op, le=le, **fmtspec)
    if simd_ext in sve:
        le = 2048 // int(typ[1:]);
        return '''int i;
                  {typ} buf0[{le}], buf1[{le}];
                  svst1_{suf}({svtrue}, buf0, {in0});
                  svst1_{suf}({svtrue}, buf1, {in1});
                  for (i=0; i < nsimd_len_{simd_ext}_{typ}(); i++) {{
                    buf0[i] = ({typ})(buf0[i] {op} buf1[i]);}}
                  return svld1_{suf}({svtrue}, buf0); '''. \
                  format(op=op, le=le, **fmtspec)

def emulate_lop2_neon(opts, op, simd_ext, typ):
    le = 128 // int(typ[1:]);
    ltyp = get_logical_type(opts, simd_ext, typ)
    lsuf = suf(ltyp)
    return '''int i;
              {ltyp} buf0[{le}], buf1[{le}];
              vst1q_{lsuf}(buf0, {in0});
              vst1q_{lsuf}(buf1, {in1});
              for (i = 0; i < {le}; i++) {{
                buf0[i] = buf0[i] {op} buf1[i] ? ({ltyp})-1 : 0;
              }}
              return vld1q_{lsuf}(buf0);'''. \
              format(op=op, le=le, ltyp=ltyp, lsuf=lsuf, **fmtspec)

def emulate_op3_neon(op, simd_ext, typ):
    le = 128 // int(typ[1:]);
    return '''int i;
              {typ} buf0[{le}], buf1[{le}], buf2[{le}];
              vst1q_{suf}(buf0, {in0});
              vst1q_{suf}(buf1, {in1});
              vst1q_{suf}(buf2, {in2});
              for (i = 0; i < {le}; i += nsimd_len_cpu_{typ}()) {{
                nsimd_storeu_cpu_{typ}(&buf0[i], nsimd_{op}_cpu_{typ}(
                  nsimd_loadu_cpu_{typ}(&buf0[i]),
                  nsimd_loadu_cpu_{typ}(&buf1[i]),
                  nsimd_loadu_cpu_{typ}(&buf2[i])));
              }}
              return vld1q_{suf}(buf0);'''.format(op=op, le=le, **fmtspec)

def emulate_f64_neon(simd_ext, op, params):
    fmtspec2 = fmtspec.copy()
    fmtspec2['op'] = op
    fmtspec2['buf_ret_decl'] = 'nsimd_cpu_{}f64 buf_ret;'. \
                               format('v' if params[0] == 'v' else 'vl')
    fmtspec2['buf_decl'] = '\n'.join(['nsimd_cpu_{}f64 buf{};'. \
                           format('v' if p[1] == 'v' else 'vl', p[0]) \
                           for p in common.enum(params[1:])])
    fmtspec2['bufs'] = ','.join(['buf{}'.format(i) \
                                 for i in range(0, len(params) - 1)])
    fmtspec2['ret_decl'] = 'nsimd_{}_{}f64 ret;'. \
                           format(simd_ext, 'v' if params[0] == 'v' else 'vl')
    buf_set = '\n'.join('''buf{i}.v0 = {ini}.v0;
                           buf{i}.v1 = {ini}.v1;'''. \
                           format(i=i, ini=fmtspec['in{}'.format(i)]) \
                           for i in range(0, len(params) - 1))
    return '''{buf_ret_decl}
              {buf_decl}
              {ret_decl}
              {buf_set}
              buf_ret = nsimd_{op}_cpu_f64({bufs});
              ret.v0 = buf_ret.v0;
              ret.v1 = buf_ret.v1;
              return ret;'''.format(buf_set=buf_set, **fmtspec2)

# -----------------------------------------------------------------------------

def f16f64(simd_ext, typ, op, armop, arity, forced_intrinsics = ''):
    fmtspec2 = fmtspec.copy()
    tmpl = ', '.join(['{{in{}}}.v{{{{i}}}}'.format(i).format(**fmtspec) \
                      for i in range(0, arity)])
    fmtspec2['args1'] = tmpl.format(i='0')
    fmtspec2['args2'] = tmpl.format(i='1')
    fmtspec2['armop'] = armop
    fmtspec2['op'] = op
    if simd_ext in neon and typ == 'f16':
        if forced_intrinsics != '':
            fmtspec2['intrinsics'] = forced_intrinsics
        else:
            temp = ', '.join(['{{in{}}}'.format(i).format(**fmtspec) \
                              for i in range(0, arity)])
            fmtspec2['intrinsics'] = 'return v{}q_f16({});'.format(armop, temp)
        return '''#ifdef NSIMD_ARM_FP16
                    {intrinsics}
                  #else
                    nsimd_{simd_ext}_vf16 ret;
                    ret.v0 = nsimd_{op}_{simd_ext}_f32({args1});
                    ret.v1 = nsimd_{op}_{simd_ext}_f32({args2});
                    return ret;
                  #endif'''.format(**fmtspec2)
    elif simd_ext == 'neon128' and typ == 'f64':
        return emulate_f64_neon(simd_ext, op, ['v'] * (arity + 1))
    return ''

# -----------------------------------------------------------------------------
# Lenghts

def max_len(simd_ext, typ):
    if simd_ext == 'sve':
        return 2048 // int(typ[1:])
    elif simd_ext in fixed_sized_sve:
        return int(simd_ext[3:]) // int(typ[1:])
    else:
        return 128 // int(typ[1:])

def real_len(simd_ext, typ):
    if simd_ext == 'sve':
        return 'nsimd_len_sve_{typ}()'.format(**fmtspec)
    else:
        return max_len(simd_ext, typ)

# -----------------------------------------------------------------------------
# Loads of degree 1, 2, 3 and 4

def load1234(opts, simd_ext, typ, deg):
    if simd_ext in neon:
        if deg == 1:
            normal = 'return vld{deg}q_{suf}({in0});'. \
                     format(deg=deg, **fmtspec)
            if typ == 'f16':
                return \
                '''#ifdef NSIMD_ARM_FP16
                     {normal}
                   #else
                     /* Note that we can do much better but is it useful? */
                     nsimd_{simd_ext}_vf16 ret;
                     f32 buf[4];
                     buf[0] = nsimd_u16_to_f32(*(u16*){in0});
                     buf[1] = nsimd_u16_to_f32(*((u16*){in0} + 1));
                     buf[2] = nsimd_u16_to_f32(*((u16*){in0} + 2));
                     buf[3] = nsimd_u16_to_f32(*((u16*){in0} + 3));
                     ret.v0 = vld1q_f32(buf);
                     buf[0] = nsimd_u16_to_f32(*((u16*){in0} + 4));
                     buf[1] = nsimd_u16_to_f32(*((u16*){in0} + 5));
                     buf[2] = nsimd_u16_to_f32(*((u16*){in0} + 6));
                     buf[3] = nsimd_u16_to_f32(*((u16*){in0} + 7));
                     ret.v1 = vld1q_f32(buf);
                     return ret;
                   #endif'''.format(normal=normal, **fmtspec)
            elif typ == 'f64' and simd_ext == 'neon128':
                return \
                '''nsimd_neon128_vf64 ret;
                   ret.v0 = *{in0};
                   ret.v1 = *({in0} + 1);
                   return ret;'''.format(**fmtspec)
            else:
                return normal
        else:
            normal = \
            '''nsimd_{simd_ext}_v{typ}x{deg} ret;
               {soa_typ} buf = vld{deg}q_{suf}({in0});
               {assignment}
               return ret;'''. \
               format(deg=deg, soa_typ=get_native_soa_typ(simd_ext, typ, deg),
                      assignment='\n'.join(['ret.v{i} = buf.val[{i}];'. \
                      format(i=i) for i in range(0, deg)]), **fmtspec)
            if typ == 'f16':
                assignment = \
                '''vst1q_u16(buf, temp.val[{{i}}]);
                   ret.v{{i}} = nsimd_loadu_{simd_ext}_f16((f16 *)buf);'''. \
                   format(**fmtspec)
                return \
                '''{soa_typ} temp = vld{deg}q_u16((u16 *){in0});
                   u16 buf[8];
                   nsimd_{simd_ext}_vf16x{deg} ret;
                   {assignment}
                   return ret;'''. \
                   format(deg=deg, assignment='\n'.join([assignment. \
                          format(i=i) for i in range(0, deg)]),
                          soa_typ=get_native_soa_typ(simd_ext, 'u16', deg),
                          **fmtspec)
            elif typ in 'f64' and simd_ext == 'neon128':
                return \
                'nsimd_neon128_vf64x{} ret;\n'.format(deg) + \
                '\n'.join(['ret.v{i}.v0 = *({in0} + {i});'. \
                           format(i=i, **fmtspec) for i in range(0, deg)]) + \
                '\n'.join(['ret.v{i}.v1 = *({in0} + {ipd});'. \
                           format(i=i, ipd=i + deg, **fmtspec) \
                           for i in range(0, deg)]) + \
                '\nreturn ret;\n'
            elif typ in ['i64', 'u64'] and simd_ext == 'neon128':
                return \
                '''nsimd_neon128_v{typ}x{deg} ret;
                   {typ} buf[2];'''.format(deg=deg, **fmtspec) + \
                '\n'.join(['''buf[0] = *({in0} + {i});
                              buf[1] = *({in0} + {ipd});
                              ret.v{i} = vld1q_{suf}(buf);'''. \
                              format(i=i, ipd=i + deg, **fmtspec) \
                              for i in range(0, deg)]) + \
                '\nreturn ret;\n'
            else:
                return normal
    else:
        if deg == 1:
            return 'return svld{deg}_{suf}({svtrue}, {in0});'. \
                   format(deg=deg, **fmtspec)
        else:
            return \
            '''nsimd_{simd_ext}_v{typ}x{deg} ret;
               {sve_typ} buf = svld{deg}_{suf}({svtrue}, {in0});
               {assignment}
               return ret;'''.format(assignment=\
               '\n'.join(['ret.v{i} = svget{deg}_{suf}(buf, {i});'. \
                          format(i=i, deg=deg, **fmtspec) \
                          for i in range(deg)]),
                          sve_typ=get_native_soa_typ('sve', typ, deg),
                          deg=deg, **fmtspec)

# -----------------------------------------------------------------------------
# Mask loads

def maskoz_load(oz, simd_ext, typ):
    if simd_ext in sve:
        return 'return svsel_{suf}({in0}, svld1_{suf}({in0}, {in1}), {oz});'. \
               format(oz='{in2}'.format(**fmtspec) if oz == 'o' \
                      else 'svdup_n_{suf}(({typ})0)'.format(**fmtspec),
                      **fmtspec)
    if typ == 'f64' and simd_ext == 'neon128':
        return '''nsimd_neon128_vf64 ret;
                  if ({in0}.v0) {{
                    ret.v0 = {in1}[0];
                  }} else {{
                    ret.v0 = {oz0};
                  }}
                  if ({in0}.v1) {{
                    ret.v1 = {in1}[1];
                  }} else {{
                    ret.v1 = {oz1};
                  }}
                  return ret;'''.format(
                  oz0 = '0.0f' if oz == 'z' else '{in2}.v0'.format(**fmtspec),
                  oz1 = '0.0f' if oz == 'z' else '{in2}.v1'.format(**fmtspec),
                  **fmtspec)
    le = 128 // int(typ[1:])
    normal = '''int i;
                {typ} buf[{le}];
                u{typnbits} mask[{le}];
                vst1q_{suf}(buf, {oz});
                vst1q_u{typnbits}(mask, {in0});
                for (i = 0; i < {le}; i++) {{
                  if (mask[i]) {{
                    buf[i] = {in1}[i];
                  }}
                }}
                return vld1q_{suf}(buf);'''. \
                format(oz='vdupq_n_{suf}(({typ})0)'.format(**fmtspec) \
                          if oz == 'z' else '{in2}'.format(**fmtspec),
                          le=le, **fmtspec)
    if typ == 'f16':
        return '''#ifdef NSIMD_ARM_FP16
                    {normal}
                  #else
                    int i;
                    nsimd_{simd_ext}_vf16 ret;
                    f32 buf[8];
                    u32 mask[8];
                    vst1q_f32(buf, {oz0});
                    vst1q_f32(buf + 4, {oz1});
                    vst1q_u32(mask, {in0}.v0);
                    vst1q_u32(mask + 4, {in0}.v1);
                    for (i = 0; i < 8; i++) {{
                      if (mask[i]) {{
                        buf[i] = nsimd_f16_to_f32({in1}[i]);
                      }}
                    }}
                    ret.v0 = vld1q_f32(buf);
                    ret.v1 = vld1q_f32(buf + 4);
                    return ret;
                  #endif'''. \
                  format(oz0='vdupq_n_f32(0.0f)'.format(**fmtspec) \
                             if oz == 'z' else '{in2}.v0'.format(**fmtspec),
                         oz1='vdupq_n_f32(0.0f)'.format(**fmtspec) \
                             if oz == 'z' else '{in2}.v1'.format(**fmtspec),
                             normal=normal, **fmtspec)
    return normal

# -----------------------------------------------------------------------------
# Stores of degree 1, 2, 3 and 4

def store1234(opts, simd_ext, typ, deg):
    if simd_ext in neon:
        if deg == 1:
            normal = 'vst{deg}q_{suf}({in0}, {in1});'. \
                     format(deg=deg, **fmtspec)
            if typ == 'f16':
                return \
                '''#ifdef NSIMD_ARM_FP16
                     {normal}
                   #else
                     f32 buf[4];
                     vst1q_f32(buf, {in1}.v0);
                     *((u16*){in0}    ) = nsimd_f32_to_u16(buf[0]);
                     *((u16*){in0} + 1) = nsimd_f32_to_u16(buf[1]);
                     *((u16*){in0} + 2) = nsimd_f32_to_u16(buf[2]);
                     *((u16*){in0} + 3) = nsimd_f32_to_u16(buf[3]);
                     vst1q_f32(buf, {in1}.v1);
                     *((u16*){in0} + 4) = nsimd_f32_to_u16(buf[0]);
                     *((u16*){in0} + 5) = nsimd_f32_to_u16(buf[1]);
                     *((u16*){in0} + 6) = nsimd_f32_to_u16(buf[2]);
                     *((u16*){in0} + 7) = nsimd_f32_to_u16(buf[3]);
                   #endif'''.format(normal=normal, **fmtspec)
            elif typ == 'f64' and simd_ext == 'neon128':
                return \
                '''*{in0} = {in1}.v0;
                   *({in0} + 1) = {in1}.v1;'''.format(**fmtspec)
            else:
                return normal
        else:
            normal = \
            '''{soa_typ} buf;
               {assignment}
               vst{deg}q_{suf}({in0}, buf);'''. \
               format(deg=deg, assignment='\n'.join([
                      'buf.val[{{}}] = {{in{}}};'.format(i). \
                      format(i - 1, **fmtspec) for i in range(1, deg + 1)]),
                      soa_typ=get_native_soa_typ(simd_ext, typ, deg),
                      **fmtspec)
            if typ == 'f16':
                assignment = \
                '''nsimd_storeu_{{simd_ext}}_f16((f16 *)buf, {{in{}}});
                   temp.val[{{}}] = vld1q_u16(buf);'''
                return \
                '''#ifdef NSIMD_ARM_FP16
                     {normal}
                   #else
                     {soa_typ} temp;
                     u16 buf[8];
                     {assignment}
                     vst{deg}q_u16((u16 *){in0}, temp);
                   #endif'''. \
                   format(assignment='\n'.join([assignment.format(i). \
                          format(i - 1, **fmtspec) \
                          for i in range(1, deg + 1)]),
                          deg=deg, normal=normal,
                          soa_typ=get_native_soa_typ(simd_ext, 'u16', deg),
                          **fmtspec)
            elif typ == 'f64' and simd_ext == 'neon128':
                return \
                '\n'.join(['*({{in0}} + {}) = {{in{}}}.v0;'. \
                           format(i - 1, i).format(**fmtspec) \
                           for i in range(1, deg + 1)]) + '\n' + \
                '\n'.join(['*({{in0}} + {}) = {{in{}}}.v1;'. \
                           format(i + deg - 1, i).format(**fmtspec) \
                           for i in range(1, deg + 1)])
            elif typ in ['i64', 'u64'] and simd_ext == 'neon128':
                return \
                '{typ} buf[{biglen}];'.format(biglen=2 * deg, **fmtspec) + \
                '\n'.join(['vst1q_{{suf}}(buf + {im1x2}, {{in{i}}});'. \
                           format(im1x2=2 * (i - 1), i=i).format(**fmtspec) \
                           for i in range(1, deg + 1)]) + \
                '\n'.join(['''*({in0} + {i}) = buf[{ix2}];
                              *({in0} + {ipd}) = buf[{ix2p1}];'''. \
                              format(i=i, ipd=i + deg, ix2=i * 2,
                                     ix2p1=2 * i + 1, **fmtspec) \
                              for i in range(0, deg)])
            else:
                return normal
    else:
        if deg == 1:
            return 'svst{deg}_{suf}({svtrue}, {in0}, {in1});'. \
                   format(deg=deg, **fmtspec)
        fill_soa_typ = \
            '\n'.join(['tmp = svset{{deg}}_{{suf}}(tmp, {im1}, {{in{i}}});'. \
            format(im1=i - 1, i=i).format(deg=deg, **fmtspec) \
            for i in range(1, deg + 1)])
        return \
        '''{soa_typ} tmp = svundef{deg}_{suf}();
           {fill_soa_typ}
           svst{deg}_{suf}({svtrue}, {in0}, tmp);'''. \
           format(soa_typ=get_native_soa_typ('sve', typ, deg), deg=deg,
                  fill_soa_typ=fill_soa_typ, **fmtspec)

# -----------------------------------------------------------------------------
# Mask stores

def mask_store(simd_ext, typ):
    if simd_ext in sve:
        return 'svst1_{suf}({in0}, {in1}, {in2});'.format(**fmtspec)
    if typ == 'f64' and simd_ext == 'neon128':
        return '''if ({in0}.v0) {{
                    {in1}[0] = {in2}.v0;
                  }}
                  if ({in0}.v1) {{
                    {in1}[1] = {in2}.v1;
                  }}'''.format(**fmtspec)
    le = 128 // int(typ[1:])
    normal = '''int i;
                {typ} buf[{le}];
                u{typnbits} mask[{le}];
                vst1q_{suf}(buf, {in2});
                vst1q_u{typnbits}(mask, {in0});
                for (i = 0; i < {le}; i++) {{
                  if (mask[i]) {{
                    {in1}[i] = buf[i];
                  }}
                }}'''.format(le=le, **fmtspec)
    if typ == 'f16':
        return \
        '''#ifdef NSIMD_ARM_FP16
             {normal}
           #else
             f32 buf[8];
             u32 mask[8];
             int i;
             vst1q_u32(mask, {in0}.v0);
             vst1q_u32(mask + 4, {in0}.v1);
             vst1q_f32(buf, {in2}.v0);
             vst1q_f32(buf + 4, {in2}.v1);
             for (i = 0; i < 8; i++) {{
               if (mask[i]) {{
                 {in1}[i] = nsimd_f32_to_f16(buf[i]);
               }}
             }}
           #endif'''.format(normal=normal, **fmtspec)
    return normal

# -----------------------------------------------------------------------------
# Length

def len1(simd_ext, typ):
    if simd_ext in neon:
        return 'return {};'.format(128 // int(typ[1:]))
    elif simd_ext == 'sve':
        return 'return (int)svcntp_b{typnbits}({svtrue}, {svtrue});'. \
               format(**fmtspec)
    elif simd_ext in fixed_sized_sve:
        return 'return {};'.format(int(simd_ext[3:]) // int(typ[1:]))

# -----------------------------------------------------------------------------
# Add/sub

def addsub(op, simd_ext, typ):
    ret = f16f64(simd_ext, typ, op, op, 2)
    if ret != '':
        return ret
    if simd_ext in neon:
        return 'return v{op}q_{suf}({in0}, {in1});'. \
               format(op=op, **fmtspec)
    else:
        return 'return sv{op}_{suf}_x({svtrue}, {in0}, {in1});'. \
               format(op=op, **fmtspec)

# -----------------------------------------------------------------------------
# Multiplication

def mul2(simd_ext, typ):
    ret = f16f64(simd_ext, typ, 'mul', 'mul', 2)
    if ret != '':
        return ret
    elif simd_ext in neon and typ in ['i64', 'u64']:
        return emulate_op2('*', simd_ext, typ)
    else:
        if simd_ext in neon:
            return 'return vmulq_{suf}({in0}, {in1});'.format(**fmtspec)
        else:
            return 'return svmul_{suf}_x({svtrue}, {in0}, {in1});'. \
                   format(**fmtspec)

# -----------------------------------------------------------------------------
# Division

def div2(simd_ext, typ):
    if simd_ext == 'aarch64' and typ in ['f32', 'f64']:
        return 'return vdivq_{suf}({in0}, {in1});'.format(**fmtspec)
    elif simd_ext in sve and \
         typ in ['f16', 'f32', 'f64', 'i32', 'u32', 'i64', 'u64']:
        return 'return svdiv_{suf}_x({svtrue}, {in0}, {in1});'. \
               format(**fmtspec)
    else:
        ret = f16f64(simd_ext, typ, 'div', 'div', 2)
        if ret != '':
            return ret
    return emulate_op2('/', simd_ext, typ)

# -----------------------------------------------------------------------------
# Binary operators: and, or, xor, andnot

def binop2(op, simd_ext, typ):
    armop = {'orb': 'orr', 'xorb': 'eor', 'andb': 'and', 'andnotb': 'bic'}
    if typ in common.iutypes:
        if simd_ext in neon:
            return 'return v{armop}q_{suf}({in0}, {in1});'. \
                   format(armop=armop[op], **fmtspec)
        else:
            return 'return sv{armop}_{suf}_x({svtrue}, {in0}, {in1});'. \
                   format(armop=armop[op], **fmtspec)
    # From here only float types
    if typ == 'f16':
        intrinsics = \
        '''return vreinterpretq_f16_u16(v{armop}q_u16(vreinterpretq_u16_f16(
                    {in0}), vreinterpretq_u16_f16({in1})));'''. \
                    format(armop=armop[op], **fmtspec)
    else:
        intrinsics = ''
    ret = f16f64(simd_ext, typ, op, armop[op], 2, intrinsics)
    if ret != '':
        return ret
    if simd_ext in neon:
        return \
        '''return vreinterpretq_f{typnbits}_u{typnbits}(v{armop}q_u{typnbits}(
                    vreinterpretq_u{typnbits}_f{typnbits}({in0}),
                      vreinterpretq_u{typnbits}_f{typnbits}({in1})));'''. \
                      format(armop=armop[op], **fmtspec)
    else:
        return \
        '''return svreinterpret_f{typnbits}_u{typnbits}(
                    sv{armop}_u{typnbits}_x({svtrue},
                      svreinterpret_u{typnbits}_f{typnbits}({in0}),
                      svreinterpret_u{typnbits}_f{typnbits}({in1})));'''. \
                      format(armop=armop[op], **fmtspec)

# -----------------------------------------------------------------------------
# Binary not

def not1(simd_ext, typ):
    if typ in common.iutypes:
        if simd_ext in neon:
            if typ in ['i8', 'u8', 'i16', 'u16', 'i32', 'u32']:
                return 'return vmvnq_{suf}({in0});'.format(**fmtspec)
            else:
                return \
                '''return vreinterpretq_{suf}_u32(vmvnq_u32(
                            vreinterpretq_u32_{suf}({in0})));'''. \
                            format(**fmtspec)
        if simd_ext in sve:
            return 'return svnot_{suf}_x({svtrue}, {in0});'.format(**fmtspec)
    # From here only float types
    if typ == 'f16':
        intrinsics = \
        '''return vreinterpretq_f16_u16(vmvnq_u16(vreinterpretq_u16_f16(
                    {in0})));'''.format(**fmtspec)
    else:
        intrinsics = ''
    ret = f16f64(simd_ext, typ, 'notb', 'mvn', 1, intrinsics)
    if ret != '':
        return ret
    if simd_ext in neon:
        return \
        '''return vreinterpretq_{suf}_u32(vmvnq_u32(
                    vreinterpretq_u32_{suf}({in0})));'''. \
                    format(**fmtspec)
    else:
        return \
        '''return svreinterpret_{suf}_u{typnbits}(svnot_u{typnbits}_x(
                    {svtrue}, svreinterpret_u{typnbits}_{suf}({in0})));'''. \
                    format(**fmtspec)

# -----------------------------------------------------------------------------
# Logical operators: and, or, xor, andnot

def lop2(opts, op, simd_ext, typ):
    armop = {'orl': 'orr', 'xorl': 'eor', 'andl': 'and', 'andnotl': 'bic'}
    if simd_ext in neon:
        if typ == 'f16':
            return \
            '''#ifdef NSIMD_ARM_FP16
                 return v{armop}q_u16({in0}, {in1});
               #else
                 nsimd_{simd_ext}_vlf16 ret;
                 ret.v0 = v{armop}q_u32({in0}.v0, {in1}.v0);
                 ret.v1 = v{armop}q_u32({in0}.v1, {in1}.v1);
                 return ret;
               #endif'''.format(armop=armop[op], **fmtspec)
        elif simd_ext == 'neon128' and typ == 'f64':
            if op == 'andnotl':
                return '''nsimd_{simd_ext}_vlf64 ret;
                          ret.v0 = {in0}.v0 & (~{in1}.v0);
                          ret.v1 = {in0}.v1 & (~{in1}.v1);
                          return ret;'''.format(**fmtspec)
            else:
                cpuop = {'orl': '|', 'xorl': '^', 'andl': '&'}
                return '''nsimd_{simd_ext}_vlf64 ret;
                          ret.v0 = {in0}.v0 {cpuop} {in1}.v0;
                          ret.v1 = {in0}.v1 {cpuop} {in1}.v1;
                          return ret;'''.format(cpuop=cpuop[op], **fmtspec)
        else:
            return 'return v{armop}q_u{typnbits}({in0}, {in1});'. \
                   format(armop=armop[op], **fmtspec)
    else:
        if opts.sve_emulate_bool:
            # TODO: the casts are a workaround to avoid a bug in gcc trunk for sve
            # it needs to be deleted when the bug is corrected
            return \
            '''return sv{armop}_x({svtrue},
                                  (svuint{typnbits}_t){in0},
                                  (svuint{typnbits}_t){in1});'''. \
            format(armop=armop[op], **fmtspec)
        else:
            return '''return sv{armop}_z({svtrue}, {in0}, {in1});'''. \
            format(armop=armop[op], **fmtspec)

# -----------------------------------------------------------------------------
# Logical not

def lnot1(opts, simd_ext, typ):
    if simd_ext in neon:
        if typ == 'f16':
            return \
            '''#ifdef NSIMD_ARM_FP16
                 return vmvnq_u16({in0});
               #else
                 nsimd_{simd_ext}_vlf16 ret;
                 ret.v0 = vmvnq_u32({in0}.v0);
                 ret.v1 = vmvnq_u32({in0}.v1);
                 return ret;
               #endif'''.format(**fmtspec)
        elif simd_ext == 'neon128' and typ == 'f64':
            return '''nsimd_neon128_vlf64 ret;
                      ret.v0 = ~{in0}.v0;
                      ret.v1 = ~{in0}.v1;
                      return ret;'''.format(**fmtspec)
        elif typ in ['i64', 'u64', 'f64']:
            return '''return vreinterpretq_u{typnbits}_u32(vmvnq_u32(
                               vreinterpretq_u32_u{typnbits}({in0})));'''. \
                               format(**fmtspec)
        else:
            return 'return vmvnq_u{typnbits}({in0});'.format(**fmtspec)
    elif simd_ext in sve:
        if opts.sve_emulate_bool:
            # TODO: the cast is a workaround to avoid a bug in gcc trunk for sve
            # it needs to be deleted when the bug is corrected
            return 'return svnot_x({svtrue}, (svuint{typnbits}_t){in0});'.format(**fmtspec)
        else:
            return 'return svnot_z({svtrue}, {in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Square root

def sqrt1(simd_ext, typ):
    if simd_ext == 'neon128':
        if typ in 'f16':
            return '''nsimd_neon128_vf16 ret;
                      ret.v0 = nsimd_sqrt_neon128_f32({in0}.v0);
                      ret.v1 = nsimd_sqrt_neon128_f32({in0}.v1);
                      return ret;'''.format(**fmtspec)
        elif typ == 'f64':
            return f16f64('neon128', 'f64', 'sqrt', 'sqrt', 1)
        else:
            return emulate_op1('sqrt', simd_ext, typ)
    elif simd_ext == 'aarch64':
        if typ == 'f16':
            return f16f64('aarch64', 'f16', 'sqrt', 'sqrt', 1)
        else:
            return 'return vsqrtq_{suf}({in0});'.format(**fmtspec)
    else:
        return 'return svsqrt_{suf}_x({svtrue}, {in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Shifts

def shl_shr(op, simd_ext, typ):
    if simd_ext in neon:
        sign = '-' if op == 'shr' else ''
        if typ in common.utypes:
            return '''return vshlq_{suf}({in0}, vdupq_n_s{typnbits}(
                                 (i{typnbits})({sign}{in1})));'''. \
                                 format(sign=sign, **fmtspec)
        else:
            return \
            '''return vreinterpretq_s{typnbits}_u{typnbits}(vshlq_u{typnbits}(
                        vreinterpretq_u{typnbits}_s{typnbits}({in0}),
                          vdupq_n_s{typnbits}((i{typnbits})({sign}{in1}))));'''. \
                          format(sign=sign, **fmtspec)
    else:
       armop = 'lsl' if op == 'shl' else 'lsr'
       if op == 'shr' and typ in common.itypes:
           return \
           '''return svreinterpret_{suf}_{suf2}(sv{armop}_{suf2}_x({svtrue},
                       svreinterpret_{suf2}_{suf}({in0}),
                       svdup_n_u{typnbits}((u{typnbits}){in1})));'''. \
                       format(suf2=common.bitfield_type[typ], armop=armop,
                              **fmtspec)
       else:
           return '''return sv{armop}_{suf}_x({svtrue}, {in0},
                              svdup_n_u{typnbits}((u{typnbits}){in1}));'''. \
                              format(armop=armop, **fmtspec)

def shra(simd_ext, typ):
    if typ in common.utypes:
        return '''return nsimd_shr_{simd_ext}_{typ}({in0}, {in1});'''. \
                format(**fmtspec)

    if simd_ext in neon:
        return  '''return vshlq_{suf}(
        {in0}, vdupq_n_s{typnbits}((i{typnbits})-{in1}));'''.\
            format(**fmtspec)
    elif simd_ext in sve:
        if typ[0] == 'i':
            return '''return svasr_n_{suf}_x({svtrue}, {in0},
                (u{typnbits}){in1});'''.\
                format(**fmtspec)
        elif typ[0] == 'u':
            return 'return svlsl_n_{suf}_x({svtrue}, {in0}, (u64){in1});'.\
                format(**fmtspec)

# -----------------------------------------------------------------------------
# Set1

def set1(simd_ext, typ):
    if simd_ext in neon:
        if typ == 'f16':
            return '''#ifdef NSIMD_ARM_FP16
                        return vdupq_n_f16({in0});
                      #else
                        nsimd_{simd_ext}_vf16 ret;
                        f32 f = nsimd_f16_to_f32({in0});
                        ret.v0 = nsimd_set1_{simd_ext}_f32(f);
                        ret.v1 = nsimd_set1_{simd_ext}_f32(f);
                        return ret;
                      #endif'''.format(**fmtspec)
        elif simd_ext == 'neon128' and typ == 'f64':
            return '''nsimd_neon128_vf64 ret;
                      ret.v0 = {in0};
                      ret.v1 = {in0};
                      return ret;'''.format(**fmtspec)
        else:
            return 'return vdupq_n_{suf}({in0});'.format(**fmtspec)
    else:
        return 'return svdup_n_{suf}({in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Set1l

def lset1(simd_ext, typ):
    if simd_ext in sve:
        return '''if ({in0}) {{
                    return svptrue_b{typnbits}();
                  }} else {{
                    return svpfalse_b();
                  }}'''.format(**fmtspec)
    # getting here means no NEON and AARCH64 only
    mask = 'vdupq_n_u{typnbits}((u{typnbits}){{}})'.format(**fmtspec)
    normal = '''if ({in0}) {{
                  return {ones};
                }} else {{
                  return {zeros};
                }}'''.format(ones=mask.format('-1'), zeros=mask.format('0'),
                             **fmtspec)
    if typ == 'f16':
        return '''#ifdef NSIMD_ARM_FP16
                    {normal}
                  #else
                    nsimd_{simd_ext}_vlf16 ret;
                    ret.v0 = nsimd_set1l_{simd_ext}_f32({in0});
                    ret.v1 = ret.v0;
                    return ret;
                  #endif'''.format(normal=normal, **fmtspec)
    if typ == 'f64' and simd_ext == 'neon128':
        return '''nsimd_neon128_vlf64 ret;
                  ret.v0 = (u64)({in0} ? -1 : 0);
                  ret.v1 = ret.v0;
                  return ret;'''.format(**fmtspec)
    return normal

# -----------------------------------------------------------------------------
# Comparison operators: ==, <, <=, >, >=

def cmp2(opts, op, simd_ext, typ):
    binop = {'eq': '==', 'lt': '<', 'le': '<=', 'gt': '>', 'ge': '>='}
    armop = {'eq': 'eq', 'lt': 'lt', 'le': 'le', 'gt': 'gt', 'ge': 'ge'}
    if simd_ext in neon:
        emul_f16 = '''nsimd_{simd_ext}_vlf16 ret;
                      ret.v0 = nsimd_{op}_{simd_ext}_f32({in0}.v0, {in1}.v0);
                      ret.v1 = nsimd_{op}_{simd_ext}_f32({in0}.v1, {in1}.v1);
                      return ret;'''.format(op=op, **fmtspec)
        normal = 'return vc{armop}q_{suf}({in0}, {in1});'. \
                 format(armop=armop[op], **fmtspec)
        if typ == 'f16':
            if simd_ext == 'neon128':
                return emul_f16
            else:
                return \
                '''#ifdef NSIMD_ARM_FP16
                     {}
                   #else
                     {}
                   #endif'''.format(normal, emul_f16)
        if simd_ext == 'neon128' and typ == 'f64':
            return '''nsimd_{simd_ext}_vl{typ} ret;
                      ret.v0 = {in0}.v0 {op} {in1}.v0 ? (u64)-1 : 0;
                      ret.v1 = {in0}.v1 {op} {in1}.v1 ? (u64)-1 : 0;
                      return ret;'''.format(op=binop[op], **fmtspec)
        elif simd_ext == 'neon128' and typ in ['i64', 'u64']:
            return '''{typ} buf0[2], buf1[2];
                      u64 ret[2];
                      vst1q_{suf}(buf0, {in0});
                      vst1q_{suf}(buf1, {in1});
                      ret[0] = buf0[0] {op} buf1[0] ? (u64)-1 : 0;
                      ret[1] = buf0[1] {op} buf1[1] ? (u64)-1 : 0;
                      return vld1q_u64(ret);'''. \
                      format(op=binop[op], **fmtspec)
        else:
            return normal
    elif simd_ext in sve:
        if opts.sve_emulate_bool:
            # TODO: the casts are a workaround to avoid a bug in gcc trunk for sve
            # it needs to be deleted when the bug is corrected
            comp = 'svcmp{op}_{suf}({svtrue}, ({svetyp}){in0}, ({svetyp}){in1})'. \
                format(op=armop[op], **fmtspec)
            return 'return {};'.format(convert_from_predicate(opts, comp))
        else:
            return 'return svcmp{op}_{suf}({svtrue}, {in0}, {in1});'. \
                    format(op=armop[op], **fmtspec)

# -----------------------------------------------------------------------------
# Not equal

def neq2(opts, simd_ext, typ):
    if simd_ext in neon:
        return '''return nsimd_notl_{simd_ext}_{typ}(
                      nsimd_eq_{simd_ext}_{typ}({in0}, {in1}));'''. \
                      format(**fmtspec)
    elif simd_ext in sve:
        comp='svcmpne_{suf}({svtrue}, {in0}, {in1})'. \
                format(**fmtspec)
        return 'return {};'.format(convert_from_predicate(opts, comp))


# -----------------------------------------------------------------------------
# If_else

def if_else3(opts, simd_ext, typ):
    if simd_ext in neon:
        intrinsic = 'return vbslq_{suf}({in0}, {in1}, {in2});'. \
                    format(**fmtspec)
        if typ == 'f16':
            return \
            '''#ifdef NSIMD_ARM_FP16
                 {intrinsic}
               #else
                 nsimd_{simd_ext}_vf16 ret;
                 ret.v0 = nsimd_if_else1_{simd_ext}_f32(
                            {in0}.v0, {in1}.v0, {in2}.v0);
                 ret.v1 = nsimd_if_else1_{simd_ext}_f32(
                            {in0}.v1, {in1}.v1, {in2}.v1);
                 return ret;
               #endif'''.format(intrinsic=intrinsic, **fmtspec)
        elif simd_ext == 'neon128' and typ == 'f64':
            return '''nsimd_neon128_vf64 ret;
                      ret.v0 = {in0}.v0 != 0u ? {in1}.v0 : {in2}.v0;
                      ret.v1 = {in0}.v1 != 0u ? {in1}.v1 : {in2}.v1;
                      return ret;'''.format(**fmtspec)
        else:
            return intrinsic
    elif simd_ext in sve:
        if opts.sve_emulate_bool:
            # TODO: the casts are a workaround to avoid a bug in gcc trunk for sve
            # it needs to be deleted when the bug is corrected
            return 'return svsel_{suf}({cond}, ({svetyp}){in1}, ({svetyp}){in2});' \
                    .format(cond=convert_to_predicate(opts,
                                '{in0}'.format(**fmtspec)),
                            **fmtspec)
        else:
            return 'return svsel_{suf}({in0}, {in1}, {in2});' \
                    .format(**fmtspec)

# -----------------------------------------------------------------------------
# Minimum and maximum

def minmax2(op, simd_ext, typ):
    ret = f16f64(simd_ext, typ, op, op, 2)
    if ret != '':
        return ret
    if simd_ext in neon:
        if typ in ['i64', 'u64']:
            binop = '<' if op == 'min' else '>'
            return '''{typ} buf0[2], buf1[2];
                      vst1q_{suf}(buf0, {in0});
                      vst1q_{suf}(buf1, {in1});
                      buf0[0] = buf0[0] {binop} buf1[0] ? buf0[0] : buf1[0];
                      buf0[1] = buf0[1] {binop} buf1[1] ? buf0[1] : buf1[1];
                      return vld1q_{suf}(buf0);'''. \
                      format(binop=binop, **fmtspec)
        else:
            return 'return v{op}q_{suf}({in0}, {in1});'. \
                   format(op=op, **fmtspec)
    else:
        return 'return sv{op}_{suf}_x({svtrue}, {in0}, {in1});'. \
               format(op=op, **fmtspec)

# -----------------------------------------------------------------------------
# Abs

def abs1(simd_ext, typ):
    if typ in common.utypes:
        return 'return {in0};'.format(**fmtspec)
    elif simd_ext in neon:
        if typ == 'f16':
            return f16f64(simd_ext, 'f16', 'abs', 'abs', 1)
        elif (typ in ['i8', 'i16', 'i32', 'f32']) or \
             (simd_ext == 'aarch64' and typ in ['i64', 'f64']):
            return 'return vabsq_{suf}({in0});'.format(**fmtspec)
        elif typ == 'i64':
            return emulate_op1('abs', 'neon128', 'i64')
        else:
            return f16f64(simd_ext, 'f64', 'abs', 'abs', 1)
    else:
        return 'return svabs_{suf}_x({svtrue}, {in0});'. \
               format(**fmtspec)

# -----------------------------------------------------------------------------
# Round, trunc, ceil and round_to_even

def round1(op, simd_ext, typ):
    if typ in common.iutypes:
        return 'return {in0};'.format(**fmtspec)
    armop = {'floor': 'rndm', 'ceil': 'rndp', 'trunc': 'rnd',
             'round_to_even': 'rndn'}
    if simd_ext == 'neon128':
        ret = f16f64('neon128', typ, op, 'v{armop}q_{suf}'. \
                     format(armop=armop, **fmtspec), 1)
        if ret != '':
            return ret
        return emulate_op1(op, 'neon128', typ);
    elif simd_ext == 'aarch64':
        if typ == 'f16':
            return f16f64('aarch64', 'f16', op, armop[op], 1)
        else:
            return 'return v{armop}q_{suf}({in0});'. \
                   format(armop=armop[op], **fmtspec)
    else:
        armop = {'floor': 'rintm', 'ceil': 'rintp', 'trunc': 'rintz',
                 'round_to_even': 'rintn'}
        return 'return sv{armop}_{suf}_x({svtrue}, {in0});'. \
               format(armop=armop[op], **fmtspec)

# -----------------------------------------------------------------------------
# FMA and FNMA

def fmafnma3(op, simd_ext, typ):
    if typ in common.ftypes and simd_ext == 'aarch64':
        armop = {'fma': 'fma', 'fnma': 'fms'}
    else:
        armop = {'fma': 'mla', 'fnma': 'mls'}
    if simd_ext in neon:
        normal = 'return v{armop}q_{suf}({in2}, {in1}, {in0});'. \
                 format(armop=armop[op], **fmtspec)
        emul = emulate_op3_neon(op, simd_ext, typ)
        if typ == 'f16':
            using_f32 = \
            '''nsimd_{simd_ext}_vf16 ret;
               ret.v0 = nsimd_{op}_{simd_ext}_f32({in0}.v0, {in1}.v0, {in2}.v0);
               ret.v1 = nsimd_{op}_{simd_ext}_f32({in0}.v1, {in1}.v1, {in2}.v1);
               return ret;'''.format(op=op, **fmtspec)
            if simd_ext == 'aarch64':
                return \
                '''#ifdef NSIMD_ARM_FP16
                     {}
                   #else
                     {}
                   #endif'''.format(emul, using_f32)
            else:
                return using_f32
        elif simd_ext == 'neon128' and typ == 'f64':
            return emulate_f64_neon('neon128', op, ['v'] * 4)
        elif simd_ext == 'aarch64' and typ == 'f64':
            return normal
        elif typ in ['i64', 'u64']:
            return emul
        else:
            return normal
    else:
        return 'return sv{armop}_{suf}_x({svtrue}, {in2}, {in1}, {in0});'. \
               format(armop=armop[op], **fmtspec)

# -----------------------------------------------------------------------------
# FMS and FNMS

def fmsfnms3(op, simd_ext, typ):
    if typ in common.iutypes:
        return \
        '''return nsimd_neg_{simd_ext}_{typ}(nsimd_{op2}_{simd_ext}_{typ}(
                      {in0}, {in1}, {in2}));'''. \
                      format(op2='fma' if op == 'fnms' else 'fnma', **fmtspec)
    if simd_ext in neon:
        return \
        '''return nsimd_{op2}_{simd_ext}_{typ}({in0}, {in1},
                      nsimd_neg_{simd_ext}_{typ}({in2}));'''. \
                      format(op2='fma' if op == 'fms' else 'fnma', **fmtspec)
    else:
        armop = {'fnms': 'nmla', 'fms': 'nmls'}
        return 'return sv{armop}_{suf}_x({svtrue}, {in2}, {in1}, {in0});'. \
               format(armop=armop[op], **fmtspec)

# -----------------------------------------------------------------------------
# Neg

def neg1(simd_ext, typ):
    if simd_ext in neon:
        normal = 'return vnegq_{suf}({in0});'.format(**fmtspec)
        if typ == 'f16':
            return f16f64(simd_ext, 'f16', 'neg', 'neg', 1)
        elif typ in ['i8', 'i16', 'i32', 'f32']:
            return normal
        elif typ in ['u8', 'u16', 'u32']:
            return \
            '''return vreinterpretq_{suf}_s{typnbits}(
                        vnegq_s{typnbits}(
                          vreinterpretq_s{typnbits}_{suf}({in0})));'''. \
                          format(**fmtspec)
        elif simd_ext == 'neon128' and typ in ['i64', 'u64']:
            return emulate_op1('neg', simd_ext, typ)
        elif simd_ext == 'neon128' and typ == 'f64':
            return \
            '''nsimd_neon128_vf64 ret;
               ret.v0 = -{in0}.v0;
               ret.v1 = -{in0}.v1;
               return ret;'''.format(**fmtspec)
        elif simd_ext == 'aarch64' and typ in ['f64', 'i64']:
            return normal
        elif simd_ext == 'aarch64' and typ == 'u64':
            return \
            '''return vreinterpretq_u64_s64(vnegq_s64(
                          vreinterpretq_s64_u64({in0})));'''. \
                          format(**fmtspec)
    else:
        if typ in common.utypes:
            return \
            '''return svreinterpret_{suf}_s{typnbits}(
                        svneg_s{typnbits}_x({svtrue},
                          svreinterpret_s{typnbits}_{suf}({in0})));'''. \
                          format(**fmtspec)
        else:
            return 'return svneg_{suf}_x({svtrue}, {in0});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Reciprocals

def recs1(op, simd_ext, typ):
    cte = '({typ})1'.format(**fmtspec) if typ != 'f16' \
          else 'nsimd_f32_to_f16(1.0f)'
    if op in ['rec', 'rec11']:
        return \
        '''return nsimd_div_{simd_ext}_{typ}(
                      nsimd_set1_{simd_ext}_{typ}({cte}), {in0});'''. \
                      format(cte=cte, **fmtspec)
    elif op == 'rsqrt11':
        return \
        '''return nsimd_div_{simd_ext}_{typ}(
                      nsimd_set1_{simd_ext}_{typ}({cte}),
                      nsimd_sqrt_{simd_ext}_{typ}({in0}));'''. \
                      format(cte=cte, **fmtspec)
    elif op in ['rec8', 'rsqrt8']:
        armop = 'recpe' if op == 'rec8' else 'rsqrte'
        if simd_ext in sve:
            return 'return sv{armop}_{suf}({in0});'. \
            format(armop=armop, **fmtspec)
        else:
            ret = f16f64(simd_ext, typ, op, armop, 1)
            if ret != '':
                return ret
            return 'return v{armop}q_{suf}({in0});'. \
            format(armop=armop, **fmtspec)

# Rec11 and rsqrt11
# According to http://infocenter.arm.com/help/topic/com.arm.doc.faqs/ka14282.html
# reciprocal estimates only work when inputs is restrained in some small
# interval so we comment these for now and return full-precision reciprocals.

# def rec11rsqrt11(op, simd_ext, typ):
#    armop = {'rec11': 'recpe', 'rsqrt11': 'rsqrte'}
#    if simd_ext in neon:
#        ret = f16f64(simd_ext, typ, op, armop[op], 1)
#        if ret != '':
#            return ret
#        return 'return v{armop}q_{suf}({in0});'. \
#               format(armop=armop[op], **fmtspec)
#    else:
#        return 'return sv{armop}_{suf}({in0});'. \
#               format(armop=armop[op], **fmtspec)

# -----------------------------------------------------------------------------
# Load of logicals

def loadl(aligned, simd_ext, typ):
    return \
    '''/* This can surely be improved but it is not our priority. */
       return nsimd_notl_{simd_ext}_{typ}(nsimd_eq_{simd_ext}_{typ}(
                nsimd_load{align}_{simd_ext}_{typ}(
                  {in0}), nsimd_set1_{simd_ext}_{typ}({zero})));'''. \
       format(align='a' if aligned else 'u',
              zero = 'nsimd_f32_to_f16(0.0f)' if typ == 'f16'
              else '({})0'.format(typ), **fmtspec)

# -----------------------------------------------------------------------------
# Store of logicals

def storel(aligned, simd_ext, typ):
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
# All and any

def allany1(opts, op, simd_ext, typ):
    binop = '&&' if  op == 'all' else '||'
    if simd_ext == 'neon128':
        if typ == 'f16':
            return \
            '''return nsimd_{op}_neon128_f32({in0}.v0) {binop}
                      nsimd_{op}_neon128_f32({in0}.v1);'''. \
                      format(op=op, binop=binop, **fmtspec)
        elif typ == 'f64':
            return 'return {in0}.v0 {binop} {in0}.v1;'. \
                   format(binop=binop, **fmtspec)
        else:
            return 'return ' + \
            binop.join(['vgetq_lane_u{typnbits}({in0}, {i})'. \
                        format(i=i, **fmtspec) \
                        for i in range(0, 128 // int(fmtspec['typnbits']))]) + \
                        ';'
    elif simd_ext == 'aarch64':
        armop = {'all': 'min', 'any': 'max'}
        normal = 'return v{armop}vq_u{typnbits}({in0}) != 0;'. \
                 format(armop=armop[op], **fmtspec)
        if typ == 'f16':
            return \
            '''#ifdef NSIMD_ARM_FP16
                 {normal}
               #else
                 return nsimd_{op}_aarch64_f32({in0}.v0) {binop}
                        nsimd_{op}_aarch64_f32({in0}.v1);
               #endif'''.format(normal=normal, op=op, binop=binop, **fmtspec)
        elif typ in ['i64', 'u64', 'f64']:
            return \
            'return v{armop}vq_u32(vreinterpretq_u32_u64({in0})) != 0;'. \
            format(armop=armop[op], **fmtspec)
        else:
            return normal
    elif simd_ext in sve:
        if op == 'any':
            operand= convert_to_predicate(opts, '{in0}'.format(**fmtspec))
            return '''return svptest_any({svtrue}, {operand});'''. \
                    format(operand=operand, **fmtspec)
        else:
            operand='svnot_z({svtrue}, {op})'. \
            format(op=convert_to_predicate(opts, '{in0}'.format(**fmtspec)),
                   **fmtspec)

            return '''return !svptest_any({svtrue}, {operand});'''. \
                    format(operand=operand, **fmtspec)

# -----------------------------------------------------------------------------
# nbtrue

def nbtrue1(opts, simd_ext, typ):
    if simd_ext == 'neon128':
        if typ == 'f16':
            return \
            '''return nsimd_nbtrue_neon128_f32({in0}.v0) +
                      nsimd_nbtrue_neon128_f32({in0}.v1);'''. \
                      format(**fmtspec)
        elif typ == 'f64':
            return 'return -(int)((i64){in0}.v0 + (i64){in0}.v1);'. \
                   format(**fmtspec)
        else:
            return \
            '''nsimd_neon128_vi{typnbits} temp =
                   vreinterpretq_s{typnbits}_u{typnbits}({in0});
               return -(int)('''.format(**fmtspec) + \
            '+'.join(['vgetq_lane_s{typnbits}(temp, {i})'. \
                      format(i=i, **fmtspec) \
                      for i in range(0, 128 // int(fmtspec['typnbits']))]) + \
                      ');'
    elif simd_ext == 'aarch64':
        normal = \
        '''return -(int)vaddvq_s{typnbits}(
                          vreinterpretq_s{typnbits}_u{typnbits}({in0}));'''. \
                     format(**fmtspec)
        if typ == 'f16':
            return \
            '''#ifdef NSIMD_ARM_FP16
                 {normal}
               #else
                 return nsimd_nbtrue_aarch64_f32({in0}.v0) +
                        nsimd_nbtrue_aarch64_f32({in0}.v1);
               #endif'''.format(normal=normal, **fmtspec)
        elif typ in ['i64', 'u64', 'f64']:
            return \
            '''return -(vaddvq_s32(vreinterpretq_s32_u64({in0})) >> 1);'''. \
                         format(**fmtspec)
        else:
            return normal
    elif simd_ext in sve:
        return 'return (int)svcntp_b{typnbits}({svtrue}, {op});'. \
               format(op=convert_to_predicate(opts, '{in0}'.format(**fmtspec)),
                      **fmtspec)

# -----------------------------------------------------------------------------
# Reinterpret logical

def reinterpretl1(simd_ext, from_typ, to_typ):
    if from_typ == to_typ or simd_ext in sve:
        return 'return {in0};'.format(**fmtspec)
    to_f16_with_f32 = \
    '''nsimd_{simd_ext}_vlf16 ret;
       u32 buf[4];
       buf[0] = (vgetq_lane_u16({in0}, 0) ? (u32)-1 : 0);
       buf[1] = (vgetq_lane_u16({in0}, 1) ? (u32)-1 : 0);
       buf[2] = (vgetq_lane_u16({in0}, 2) ? (u32)-1 : 0);
       buf[3] = (vgetq_lane_u16({in0}, 3) ? (u32)-1 : 0);
       ret.v0 = vld1q_u32(buf);
       buf[0] = (vgetq_lane_u16({in0}, 4) ? (u32)-1 : 0);
       buf[1] = (vgetq_lane_u16({in0}, 5) ? (u32)-1 : 0);
       buf[2] = (vgetq_lane_u16({in0}, 6) ? (u32)-1 : 0);
       buf[3] = (vgetq_lane_u16({in0}, 7) ? (u32)-1 : 0);
       ret.v1 = vld1q_u32(buf);
       return ret;'''.format(**fmtspec)
    from_f16_with_f32 = \
    '''u16 buf[8];
       buf[0] = (vgetq_lane_u32({in0}.v0, 0) ? (u16)-1 : 0);
       buf[1] = (vgetq_lane_u32({in0}.v0, 1) ? (u16)-1 : 0);
       buf[2] = (vgetq_lane_u32({in0}.v0, 2) ? (u16)-1 : 0);
       buf[3] = (vgetq_lane_u32({in0}.v0, 3) ? (u16)-1 : 0);
       buf[4] = (vgetq_lane_u32({in0}.v1, 0) ? (u16)-1 : 0);
       buf[5] = (vgetq_lane_u32({in0}.v1, 1) ? (u16)-1 : 0);
       buf[6] = (vgetq_lane_u32({in0}.v1, 2) ? (u16)-1 : 0);
       buf[7] = (vgetq_lane_u32({in0}.v1, 3) ? (u16)-1 : 0);
       return vld1q_u16(buf);'''.format(**fmtspec)
    if simd_ext == 'neon128':
        if to_typ == 'f16':
            return to_f16_with_f32
        elif from_typ == 'f16':
            return from_f16_with_f32
        elif to_typ == 'f64':
            return '''nsimd_neon128_vlf64 ret;
                      ret.v0 = vgetq_lane_u64({in0}, 0);
                      ret.v1 = vgetq_lane_u64({in0}, 1);
                      return ret;'''.format(**fmtspec)
        elif from_typ == 'f64':
            return '''u64 buf[2];
                      buf[0] = {in0}.v0;
                      buf[1] = {in0}.v1;
                      return vld1q_u64(buf);'''.format(**fmtspec)
        else:
            return 'return {in0};'.format(**fmtspec)
    elif simd_ext == 'aarch64':
        if to_typ == 'f16':
            return '''#ifdef NSIMD_ARM_FP16
                        return {in0};
                      #else
                        {using_f32}
                      #endif'''.format(using_f32=to_f16_with_f32, **fmtspec)
        elif from_typ == 'f16':
            return '''#ifdef NSIMD_ARM_FP16
                        return {in0};
                      #else
                        {using_f32}
                      #endif'''.format(using_f32=from_f16_with_f32, **fmtspec)
        else:
            return 'return {in0};'.format(**fmtspec)

# -----------------------------------------------------------------------------
# Convert

def convert1(simd_ext, from_typ, to_typ):
    fmtspec2 = fmtspec.copy()
    fmtspec2['to_suf'] = suf(to_typ)
    fmtspec2['from_suf'] = suf(from_typ)
    if from_typ == to_typ:
        return 'return {in0};'.format(**fmtspec)
    if from_typ in common.iutypes and to_typ in common.iutypes:
        if simd_ext in neon:
            return 'return vreinterpretq_{to_suf}_{from_suf}({in0});'. \
                   format(**fmtspec2)
        else:
            return 'return svreinterpret_{to_suf}_{from_suf}({in0});'. \
                   format(**fmtspec2)
    if simd_ext in sve:
        return 'return svcvt_{to_suf}_{from_suf}_x({svtrue}, {in0});'. \
               format(**fmtspec2)
    to_f16_with_f32 = \
    '''nsimd_{simd_ext}_vf16 ret;
       f32 buf[4];
       buf[0] = (f32)vgetq_lane_{from_suf}({in0}, 0);
       buf[1] = (f32)vgetq_lane_{from_suf}({in0}, 1);
       buf[2] = (f32)vgetq_lane_{from_suf}({in0}, 2);
       buf[3] = (f32)vgetq_lane_{from_suf}({in0}, 3);
       ret.v0 = vld1q_f32(buf);
       buf[0] = (f32)vgetq_lane_{from_suf}({in0}, 4);
       buf[1] = (f32)vgetq_lane_{from_suf}({in0}, 5);
       buf[2] = (f32)vgetq_lane_{from_suf}({in0}, 6);
       buf[3] = (f32)vgetq_lane_{from_suf}({in0}, 7);
       ret.v1 = vld1q_f32(buf);
       return ret;'''.format(**fmtspec2)
    from_f16_with_f32 = \
    '''{to_typ} buf[8];
       buf[0] = ({to_typ})vgetq_lane_f32({in0}.v0, 0);
       buf[1] = ({to_typ})vgetq_lane_f32({in0}.v0, 1);
       buf[2] = ({to_typ})vgetq_lane_f32({in0}.v0, 2);
       buf[3] = ({to_typ})vgetq_lane_f32({in0}.v0, 3);
       buf[4] = ({to_typ})vgetq_lane_f32({in0}.v1, 0);
       buf[5] = ({to_typ})vgetq_lane_f32({in0}.v1, 1);
       buf[6] = ({to_typ})vgetq_lane_f32({in0}.v1, 2);
       buf[7] = ({to_typ})vgetq_lane_f32({in0}.v1, 3);
       return vld1q_{to_suf}(buf);'''.format(**fmtspec2)
    if simd_ext == 'neon128':
        if to_typ == 'f16':
            return to_f16_with_f32
        elif from_typ == 'f16':
            return from_f16_with_f32
        elif to_typ == 'f64':
            return '''nsimd_neon128_vf64 ret;
                      ret.v0 = (f64)vgetq_lane_{from_suf}({in0}, 0);
                      ret.v1 = (f64)vgetq_lane_{from_suf}({in0}, 1);
                      return ret;'''.format(**fmtspec2)
        elif from_typ == 'f64':
            return '''{to_typ} buf[2];
                      buf[0] = ({to_typ}){in0}.v0;
                      buf[1] = ({to_typ}){in0}.v1;
                      return vld1q_{to_suf}(buf);'''.format(**fmtspec2)
        else:
            return 'return vcvtq_{to_suf}_{from_suf}({in0});'. \
                   format(**fmtspec2)
    elif simd_ext == 'aarch64':
        if to_typ == 'f16':
            return '''#ifdef NSIMD_ARM_FP16
                        return vcvtq_{to_suf}_{from_suf}({in0});
                      #else
                        {using_f32}
                      #endif'''.format(using_f32=to_f16_with_f32, **fmtspec2)
        elif from_typ == 'f16':
            return '''#ifdef NSIMD_ARM_FP16
                        return vcvtq_{to_suf}_{from_suf}({in0});
                      #else
                        {using_f32}
                      #endif'''.format(using_f32=from_f16_with_f32, **fmtspec2)
        else:
            return 'return vcvtq_{to_suf}_{from_suf}({in0});'. \
                   format(**fmtspec2)

# -----------------------------------------------------------------------------
# Reinterpret

def reinterpret1(simd_ext, from_typ, to_typ):
    fmtspec2 = fmtspec.copy()
    fmtspec2['to_suf'] = suf(to_typ)
    fmtspec2['from_suf'] = suf(from_typ)
    if from_typ == to_typ:
        return 'return {in0};'.format(**fmtspec)
    if simd_ext in sve:
        return 'return svreinterpret_{to_suf}_{from_suf}({in0});'. \
               format(**fmtspec2)
    to_f16_with_f32 = \
    '''nsimd_{simd_ext}_vf16 ret;
       f32 buf[4];
       buf[0] = nsimd_u16_to_f32((u16)vgetq_lane_{from_suf}({in0}, 0));
       buf[1] = nsimd_u16_to_f32((u16)vgetq_lane_{from_suf}({in0}, 1));
       buf[2] = nsimd_u16_to_f32((u16)vgetq_lane_{from_suf}({in0}, 2));
       buf[3] = nsimd_u16_to_f32((u16)vgetq_lane_{from_suf}({in0}, 3));
       ret.v0 = vld1q_f32(buf);
       buf[0] = nsimd_u16_to_f32((u16)vgetq_lane_{from_suf}({in0}, 4));
       buf[1] = nsimd_u16_to_f32((u16)vgetq_lane_{from_suf}({in0}, 5));
       buf[2] = nsimd_u16_to_f32((u16)vgetq_lane_{from_suf}({in0}, 6));
       buf[3] = nsimd_u16_to_f32((u16)vgetq_lane_{from_suf}({in0}, 7));
       ret.v1 = vld1q_f32(buf);
       return ret;'''.format(**fmtspec2)
    from_f16_with_f32 = \
    '''{to_typ} buf[8];
       buf[0] = ({to_typ})nsimd_f32_to_u16(vgetq_lane_f32({in0}.v0, 0));
       buf[1] = ({to_typ})nsimd_f32_to_u16(vgetq_lane_f32({in0}.v0, 1));
       buf[2] = ({to_typ})nsimd_f32_to_u16(vgetq_lane_f32({in0}.v0, 2));
       buf[3] = ({to_typ})nsimd_f32_to_u16(vgetq_lane_f32({in0}.v0, 3));
       buf[4] = ({to_typ})nsimd_f32_to_u16(vgetq_lane_f32({in0}.v1, 0));
       buf[5] = ({to_typ})nsimd_f32_to_u16(vgetq_lane_f32({in0}.v1, 1));
       buf[6] = ({to_typ})nsimd_f32_to_u16(vgetq_lane_f32({in0}.v1, 2));
       buf[7] = ({to_typ})nsimd_f32_to_u16(vgetq_lane_f32({in0}.v1, 3));
       return vld1q_{to_suf}(buf);'''.format(**fmtspec2)
    if simd_ext == 'neon128':
        if to_typ == 'f16':
            return to_f16_with_f32
        elif from_typ == 'f16':
            return from_f16_with_f32
        elif to_typ == 'f64':
            return '''nsimd_neon128_vf64 ret;
                      union {{ f64 to; {from_typ} from; }} buf;
                      buf.from = vgetq_lane_{from_suf}({in0}, 0);
                      ret.v0 = buf.to;
                      buf.from = vgetq_lane_{from_suf}({in0}, 1);
                      ret.v1 = buf.to;
                      return ret;'''.format(**fmtspec2)
        elif from_typ == 'f64':
            return '''union {{ f64 from; {to_typ} to; }} buf_;
                      {to_typ} buf[2];
                      buf_.from = {in0}.v0;
                      buf[0] = buf_.to;
                      buf_.from = {in0}.v1;
                      buf[1] = buf_.to;
                      return vld1q_{to_suf}(buf);'''.format(**fmtspec2)
        else:
            return 'return vreinterpretq_{to_suf}_{from_suf}({in0});'. \
                   format(**fmtspec2)
    elif simd_ext == 'aarch64':
        if to_typ == 'f16':
            return '''#ifdef NSIMD_ARM_FP16
                        return vreinterpretq_{to_suf}_{from_suf}({in0});
                      #else
                        {using_f32}
                      #endif'''.format(using_f32=to_f16_with_f32, **fmtspec2)
        elif from_typ == 'f16':
            return '''#ifdef NSIMD_ARM_FP16
                        return vreinterpretq_{to_suf}_{from_suf}({in0});
                      #else
                        {using_f32}
                      #endif'''.format(using_f32=from_f16_with_f32, **fmtspec2)
        else:
            return 'return vreinterpretq_{to_suf}_{from_suf}({in0});'. \
                   format(**fmtspec2)

# -----------------------------------------------------------------------------
# reverse

def reverse1(simd_ext, typ):
    armtyp = suf(typ)
    if simd_ext in sve:
        return '''return svrev_{suf}( {in0} );'''.format(**fmtspec)
    elif simd_ext == 'neon128' and typ == 'f64':
        return '''nsimd_neon128_vf64 ret;
                  ret.v0 = {in0}.v1;
                  ret.v1 = {in0}.v0;
                  return ret;'''.format(**fmtspec)
    elif typ in [ 'i64', 'u64', 'f64' ]:
        return '''return vcombine_{armtyp}(vget_high_{armtyp}({in0}),
                                           vget_low_{armtyp}({in0}));'''. \
                                           format(armtyp=armtyp, **fmtspec)
    elif typ == 'f16':
        return '''nsimd_{simd_ext}_vf16 ret;
                  ret.v0 = nsimd_reverse_{simd_ext}_f32(a0.v1);
                  ret.v1 = nsimd_reverse_{simd_ext}_f32(a0.v0);
                  return ret;'''.format(**fmtspec)
    else:
        return '''{in0} = vrev64q_{armtyp}({in0});
                  return vcombine_{armtyp}(vget_high_{armtyp}({in0}),
                                           vget_low_{armtyp}({in0}));'''. \
                                           format(armtyp=armtyp, **fmtspec)

# -----------------------------------------------------------------------------
# Horizontal sum

def addv(simd_ext, typ):

    if simd_ext == 'neon128':
        if typ == 'f64':
            return 'return ({typ})({in0}.v0 + {in0}.v1);'.format(**fmtspec)
        elif typ == 'f16':
            return \
            '''#ifdef NSIMD_ARM_FP16
                 {t} tmp = vadd_{suf}(vget_low_{suf}({in0}),
                                      vget_high_{suf}({in0}));
                 tmp = vadd_{suf}(tmp, vext_{suf}(tmp, tmp, 3));
                 tmp = vadd_{suf}(tmp, vext_{suf}(tmp, tmp, 0));
                 return vget_lane_{suf}(tmp, 0);
               #else
                 float32x2_t tmp0 = vadd_f32(vget_low_f32({in0}.v0),
                                             vget_high_f32({in0}.v0));
                 tmp0 = vadd_f32(tmp0, vext_f32(tmp0, tmp0, 1));
                 float32x2_t tmp1 = vadd_f32(vget_low_f32({in0}.v1),
                                             vget_high_f32({in0}.v1));
                 tmp1 = vadd_f32(tmp1, vext_f32(tmp1, tmp1, 1));
                 return nsimd_f32_to_f16(vget_lane_f32(tmp0, 0) +
                                         vget_lane_f32(tmp1, 0));
               #endif''' .format(t=half_neon64_typ(typ), **fmtspec)
        elif typ == 'f32':
            return \
            '''{t} tmp = vadd_{suf}(vget_low_{suf}({in0}),
                                    vget_high_{suf}({in0}));
               tmp = vadd_{suf}(tmp, vext_{suf}(tmp, tmp, 1));
               return vget_lane_{suf}(tmp, 0);'''. \
               format(t=half_neon64_typ(typ), **fmtspec)
        elif typ[0] in ['i', 'u']:
            le = 128 // int(typ[1:]);
            return \
            '''{typ} res = ({typ})0;
               {typ} buf[{le}];
               vst1q_{suf}(buf, {in0});
               for (int i = 0; i < {le}; i++) {{
                 res += buf[i];
               }}
               return res;'''. \
               format(le=le, **fmtspec)
    elif simd_ext == 'aarch64':
        if typ == 'f16':
            return \
            '''#ifdef NSIMD_ARM_FP16
                 {t} tmp = vadd_{suf}(vget_low_{suf}({in0}),
                                      vget_high_{suf}({in0}));
                 tmp = vadd_{suf}(tmp, vext_{suf}(tmp, tmp, 3));
                 tmp = vadd_{suf}(tmp, vext_{suf}(tmp, tmp, 0));
                 return vget_lane_{suf}(tmp, 0);
               #else
                 float32x2_t tmp0 = vadd_f32(vget_low_f32({in0}.v0),
                                             vget_high_f32({in0}.v0));
                 tmp0 = vadd_f32(tmp0, vext_f32(tmp0, tmp0, 1));
                 float32x2_t tmp1 = vadd_f32(vget_low_f32({in0}.v1),
                                             vget_high_f32({in0}.v1));
                 tmp1 = vadd_f32(tmp1, vext_f32(tmp1, tmp1, 1));
                 return nsimd_f32_to_f16(vget_lane_f32(tmp0, 0) +
                                         vget_lane_f32(tmp1, 0));
               #endif''' .format(t=half_neon64_typ(typ), **fmtspec)
        elif typ in ['f32', 'f64']:
            return 'return vaddvq_{suf}({in0});'.format(**fmtspec)
    elif simd_ext in sve:
        return 'return svaddv_{suf}({svtrue}, {in0});' .format(**fmtspec)

# -----------------------------------------------------------------------------
# Up convert

def upcvt1(simd_ext, from_typ, to_typ):
    # For integer upcast, due to 2's complement representation
    # _s : signed   -> bigger signed
    # _s : signed   -> bigger unsigned
    # _u : unsigned -> bigger signed
    # _u : unsigned -> bigger unsigned
    if simd_ext in neon:
        if from_typ == 'f16' and to_typ == 'f32':
            return \
            '''#ifdef NSIMD_ARM_FP16
                 nsimd_{simd_ext}_vf32x2 ret;
                 ret.v0 = vcvt_f32_f16(vget_low_{suf}({in0}));
                 ret.v1 = vcvt_f32_f16(vget_high_{suf}({in0}));
                 return ret;
               #else
                 nsimd_{simd_ext}_vf32x2 ret;
                 ret.v0 = {in0}.v0;
                 ret.v1 = {in0}.v1;
                 return ret;
               #endif'''.format(**fmtspec)
        elif from_typ == 'f32' and to_typ == 'f64':
            if simd_ext == 'neon128':
                return \
                '''nsimd_neon128_vf64x2 ret;
                   f32 buf[4];
                   vst1q_f32(buf, {in0});
                   ret.v0.v0 = (f64)buf[0];
                   ret.v0.v1 = (f64)buf[1];
                   ret.v1.v0 = (f64)buf[2];
                   ret.v1.v1 = (f64)buf[3];
                   return ret;'''.format(**fmtspec)
            else:
                return \
                '''nsimd_aarch64_vf64x2 ret;
                   ret.v0 = vcvt_f64_f32(vget_low_{suf}({in0}));
                   ret.v1 = vcvt_f64_f32(vget_high_{suf}({in0}));
                   return ret;'''.format(**fmtspec)
        elif (from_typ in common.itypes and to_typ in common.itypes) or \
             (from_typ in common.utypes and to_typ in common.utypes):
            return '''nsimd_{simd_ext}_v{to_typ}x2 ret;
                      ret.v0 = vmovl_{suf}(vget_low_{suf}({in0}));
                      ret.v1 = vmovl_{suf}(vget_high_{suf}({in0}));
                      return ret;'''.format(**fmtspec)
        elif (from_typ in common.itypes and to_typ in common.utypes) or \
             (from_typ in common.utypes and to_typ in common.itypes):
            return '''nsimd_{simd_ext}_v{to_typ}x2 ret;
                      ret.v0 = vreinterpretq_{suf_to_typ}_{suf_int_typ}(
                                 vmovl_{suf}(vget_low_{suf}({in0})));
                      ret.v1 = vreinterpretq_{suf_to_typ}_{suf_int_typ}(
                                 vmovl_{suf}(vget_high_{suf}({in0})));
                      return ret;'''. \
                      format(suf_to_typ=suf(to_typ),
                             suf_int_typ=suf(from_typ[0] + to_typ[1:]),
                             **fmtspec)
        else:
            return \
            '''nsimd_{simd_ext}_v{to_typ}x2 ret;
               nsimd_{simd_ext}_v{int_typ}x2 tmp;
               tmp = nsimd_upcvt_{simd_ext}_{int_typ}_{from_typ}({in0});
               ret.v0 = nsimd_cvt_{simd_ext}_{to_typ}_{int_typ}(tmp.v0);
               ret.v1 = nsimd_cvt_{simd_ext}_{to_typ}_{int_typ}(tmp.v1);
               return ret;'''. \
               format(int_typ=from_typ[0] + to_typ[1:], **fmtspec)

    # Getting here means that we deal with SVE
    if (from_typ in common.itypes and to_typ in common.itypes) or \
       (from_typ in common.utypes and to_typ in common.utypes):
        return '''nsimd_{simd_ext}_v{to_typ}x2 ret;
                  ret.v0 = svunpklo_{suf_to_typ}({in0});
                  ret.v1 = svunpkhi_{suf_to_typ}({in0});
                  return ret;'''.format(suf_to_typ=suf(to_typ), **fmtspec)
    elif (from_typ in common.itypes and to_typ in common.utypes) or \
         (from_typ in common.utypes and to_typ in common.itypes):
        return \
        '''nsimd_{simd_ext}_v{to_typ}x2 ret;
           ret.v0 = svreinterpret_{suf_to_typ}_{suf_int_typ}(
                      svunpklo_{suf_int_typ}({in0}));
           ret.v1 = svreinterpret_{suf_to_typ}_{suf_int_typ}(
                      svunpkhi_{suf_int_typ}({in0}));
           return ret;'''. \
           format(suf_to_typ=suf(to_typ),
                  suf_int_typ=suf(from_typ[0] + to_typ[1:]), **fmtspec)
    elif from_typ in common.iutypes and to_typ in common.ftypes:
        return \
        '''nsimd_{simd_ext}_v{to_typ}x2 ret;
           ret.v0 = svcvt_{suf_to_typ}_{suf_int_typ}_x(
                      {svtrue}, svunpklo_{suf_int_typ}({in0}));
           ret.v1 = svcvt_{suf_to_typ}_{suf_int_typ}_x(
                      {svtrue}, svunpkhi_{suf_int_typ}({in0}));
           return ret;'''. \
           format(suf_to_typ=suf(to_typ),
                  suf_int_typ=suf(from_typ[0] + to_typ[1:]), **fmtspec)
    else:
        return \
        '''nsimd_{simd_ext}_v{to_typ}x2 ret;
           ret.v0 = svcvt_{suf_to_typ}_{suf}_x({svtrue}, svzip1_{suf}(
                      {in0}, {in0}));
           ret.v1 = svcvt_{suf_to_typ}_{suf}_x({svtrue}, svzip2_{suf}(
                      {in0}, {in0}));
           return ret;'''.format(suf_to_typ=suf(to_typ), **fmtspec)

# -----------------------------------------------------------------------------
# Down convert

def downcvt1(simd_ext, from_typ, to_typ):
    if simd_ext in neon:
        if from_typ == 'f64' and to_typ == 'f32':
            if simd_ext == 'neon128':
                return '''f32 buf[4];
                          buf[0] = (f32){in0}.v0;
                          buf[1] = (f32){in0}.v1;
                          buf[2] = (f32){in1}.v0;
                          buf[3] = (f32){in1}.v1;
                          return vld1q_f32(buf);'''.format(**fmtspec)
            else:
                return '''return vcombine_f32(vcvt_f32_f64({in0}),
                                              vcvt_f32_f64({in1}));'''. \
                                              format(**fmtspec)
        elif from_typ == 'f32' and to_typ == 'f16':
            return '''#ifdef NSIMD_ARM_FP16
                        return vcombine_f16(vcvt_f16_f32({in0}),
                                            vcvt_f16_f32({in1}));
                      #else
                        nsimd_{simd_ext}_vf16 ret;
                        ret.v0 = {in0};
                        ret.v1 = {in1};
                        return ret;
                      #endif'''.format(**fmtspec)
        elif (from_typ in common.itypes and to_typ in common.itypes) or \
             (from_typ in common.utypes and to_typ in common.utypes):
            return '''return vcombine_{suf_to_typ}(vmovn_{suf}({in0}),
                               vmovn_{suf}({in1}));'''. \
                               format(suf_to_typ=suf(to_typ), **fmtspec)
        elif (from_typ in common.itypes and to_typ in common.itypes) or \
             (from_typ in common.utypes and to_typ in common.utypes):
            return '''return vreinterpretq_{suf_to_typ}(
                               vcombine_{suf_to_typ}(vmovn_{suf}({in0}),
                                 vmovn_{suf}({in1}));'''. \
                                 format(suf_to_typ=suf(to_typ), **fmtspec)
        else:
            return \
            '''return nsimd_downcvt_{simd_ext}_{to_typ}_{int_typ}(
                        nsimd_cvt_{simd_ext}_{int_typ}_{from_typ}({in0}),
                        nsimd_cvt_{simd_ext}_{int_typ}_{from_typ}({in1}));'''.\
                        format(int_typ=to_typ[0] + from_typ[1:], **fmtspec)

    # Getting here means that we deal with SVE
    if from_typ in common.iutypes and to_typ in common.iutypes:
        return '''return svuzp1_{suf_to_typ}(
                           svreinterpret_{suf_to_typ}_{suf}({in0}),
                           svreinterpret_{suf_to_typ}_{suf}({in1}));'''. \
                           format(suf_to_typ=suf(to_typ), **fmtspec)
    elif from_typ in common.ftypes and to_typ in common.iutypes:
        return \
        '''return svuzp1_{suf_to_typ}(svreinterpret_{suf_to_typ}_{suf_int_typ}(
                    svcvt_{suf_int_typ}_{suf}_x({svtrue}, {in0})),
                      svreinterpret_{suf_to_typ}_{suf_int_typ}(
                        svcvt_{suf_int_typ}_{suf}_x({svtrue}, {in1})));'''. \
                        format(suf_to_typ=suf(to_typ),
                               suf_int_typ=suf(to_typ[0] + from_typ[1:]),
                               **fmtspec)
    else:
        return \
        '''return svuzp1_{suf_to_typ}(svcvt_{suf_to_typ}_{suf}_x(
                    {svtrue}, {in0}), svcvt_{suf_to_typ}_{suf}_x(
                      {svtrue}, {in1}));'''. \
                    format(suf_to_typ=suf(to_typ), **fmtspec)

# -----------------------------------------------------------------------------
# adds

def adds(simd_ext, from_typ):
    if from_typ in common.ftypes:
        return 'return nsimd_add_{simd_ext}_{from_typ}({in0}, {in1});'. \
               format(**fmtspec)
    if simd_ext in neon:
        return 'return vqaddq_{suf}({in0}, {in1});'.format(**fmtspec)
    else:
        return 'return svqadd_{suf}({in0}, {in1});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# subs

def subs(simd_ext, from_typ):
    if from_typ in common.ftypes:
        return 'return nsimd_sub_{simd_ext}_{from_typ}({in0}, {in1});'. \
               format(**fmtspec)
    elif simd_ext in neon:
        return 'return vqsubq_{suf}({in0}, {in1});'.format(**fmtspec)
    else:
        return 'return svqsub_{suf}({in0}, {in1});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# to_mask

def to_mask1(opts, simd_ext, typ):
    if typ in common.itypes + common.ftypes:
        normal = 'return vreinterpretq_{suf}_u{typnbits}({in0});'. \
                 format(**fmtspec)
    else:
        normal = 'return {in0};'.format(**fmtspec)
    emulate_f16 = '''nsimd_{simd_ext}_vf16 ret;
                     ret.v0 = nsimd_to_mask_{simd_ext}_f32({in0}.v0);
                     ret.v1 = nsimd_to_mask_{simd_ext}_f32({in0}.v1);
                     return ret;'''.format(**fmtspec)
    if simd_ext == 'neon128' and typ == 'f16':
        return emulate_f16
    elif simd_ext == 'neon128' and typ == 'f64':
        return '''nsimd_neon128_vf64 ret;
                  ret.v0 = nsimd_scalar_reinterpret_f64_u64({in0}.v0);
                  ret.v1 = nsimd_scalar_reinterpret_f64_u64({in0}.v1);
                  return ret;'''.format(**fmtspec)
    elif simd_ext == 'aarch64' and typ == 'f16':
        return '''#ifdef NSIMD_ARM_FP16
                    {normal}
                  #else
                    {emulate_f16}
                  #endif'''.format(normal=normal, emulate_f16=emulate_f16)
    elif simd_ext in sve:
        if opts.sve_emulate_bool:
            return 'return svreinterpret_{suf}_u{typnbits}({in0});'. \
                    format(**fmtspec)
        else:
           utyp = 'u{}'.format(fmtspec['typnbits'])
           return '''return svreinterpret_{suf}_{utyp}(svsel_{utyp}(
                          {in0}, svdup_n_{utyp}(({utyp})-1),
                          svdup_n_{utyp}(({utyp})0)));'''. \
                          format(utyp=utyp, **fmtspec)
    else:
        return normal

# -----------------------------------------------------------------------------
# iota

def iota(simd_ext, typ):
    if simd_ext in sve:
        if typ in common.iutypes:
            return 'return svindex_{suf}(0, 1);'.format(**fmtspec)
        else:
            return \
            '''return svcvt_{suf}_s{typnbits}_x({svtrue},
                        svindex_s{typnbits}(0, 1));'''.format(**fmtspec)
    if typ == 'f64' and simd_ext == 'neon128':
        return '''nsimd_neon128_vf64 ret;
                  ret.v0 = 0.0;
                  ret.v1 = 1.0;
                  return ret;'''.format(**fmtspec)
    typ2 = 'f32' if typ == 'f16' else typ
    le = 128 // int(typ[1:])
    iota = ', '.join(['({typ2}){i}'.format(typ2=typ2, i=i) \
                      for i in range(le)])
    normal = '''{typ} buf[{le}] = {{ {iota} }};
                return vld1q_{suf}(buf);'''. \
                format(le=le, iota=iota, **fmtspec)
    if typ == 'f16':
        return '''#ifdef NSIMD_ARM_FP16
                    {normal}
                  #else
                    f32 buf[8] = {{ {iota} }};
                    nsimd_{simd_ext}_vf16 ret;
                    ret.v0 = vld1q_f32(buf);
                    ret.v1 = vld1q_f32(buf + 4);
                    return ret;
                  #endif'''.format(iota=iota, normal=normal, **fmtspec)
    return normal

# -----------------------------------------------------------------------------
# mask_for_loop_tail

def mask_for_loop_tail(simd_ext, typ):
    if typ == 'f16':
        threshold = 'nsimd_f32_to_f16((f32)({in1} - {in0}))'.format(**fmtspec)
    else:
        threshold = '({typ})({in1} - {in0})'.format(**fmtspec)
    if simd_ext == 'sve':
        le = 'nsimd_len_sve_{typ}()'.format(**fmtspec)
    elif simd_ext in fixed_sized_sve:
        le = int(simd_ext[3:]) // int(typ[1:])
    else:
        le = 128 // int(typ[1:])
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
# to_logical

def to_logical1(opts, simd_ext, typ):
    if typ in common.iutypes:
        return '''return nsimd_ne_{simd_ext}_{typ}({in0},
                           nsimd_set1_{simd_ext}_{typ}(({typ})0));'''. \
                           format(**fmtspec)
    normal_fp = \
    '''return nsimd_reinterpretl_{simd_ext}_{suf}_{utyp}(
                nsimd_ne_{simd_ext}_{utyp}(
                  nsimd_reinterpret_{simd_ext}_{utyp}_{typ}(
                    {in0}), nsimd_set1_{simd_ext}_{utyp}(({utyp})0)));'''. \
                    format(utyp='u{}'.format(fmtspec['typnbits']), **fmtspec)
    if typ in ['f32', 'f64'] or (typ == 'f16' and simd_ext in sve):
        return normal_fp
    emulate_fp16 = \
    '''nsimd_{simd_ext}_vlf16 ret;
       ret.v0 = nsimd_to_logical_{simd_ext}_f32({in0}.v0);
       ret.v1 = nsimd_to_logical_{simd_ext}_f32({in0}.v1);
       return ret;'''.format(**fmtspec)
    if simd_ext == 'aarch64':
        return '''#ifdef NSIMD_ARM_FP16
                    {normal_fp}
                  #else
                    {emulate_fp16}
                  #endif'''.format(normal_fp=normal_fp,
                                   emulate_fp16=emulate_fp16)
    elif simd_ext == 'neon128':
        return emulate_fp16

# -----------------------------------------------------------------------------
# unpack functions

def zip_unzip_half(func, simd_ext, typ):
    if simd_ext == 'aarch64' or simd_ext in sve:
        if typ =='f16' and simd_ext == 'aarch64':
            if func in ['zip1', 'zip2']:
                return '''\
                #ifdef NSIMD_ARM_FP16
                  return {s}v{op}{q}_{suf}({in0}, {in1});
                #else
                  nsimd_{simd_ext}_v{typ} ret;
                  ret.v0 = {s}vzip1{q}_f32({in0}.v{i}, {in1}.v{i});
                  ret.v1 = {s}vzip2{q}_f32({in0}.v{i}, {in1}.v{i});
                  return ret;
                #endif
                '''.format(op=func,
                           i = '0' if func in ['zip1', 'uzp1'] else '1',
                           s = 's' if simd_ext in sve else '',
                           q = '' if simd_ext in sve else 'q', **fmtspec)
            else:
                return '''\
                #ifdef NSIMD_ARM_FP16
                  return {s}v{op}{q}_{suf}({in0}, {in1});
                #else
                  nsimd_{simd_ext}_v{typ} ret;
                  ret.v0 = {s}v{func}{q}_f32({in0}.v0, {in0}.v1);
                  ret.v1 = {s}v{func}{q}_f32({in1}.v0, {in1}.v1);
                  return ret;
                #endif'''.format(op=func, func=func,
                          s = 's' if simd_ext in sve else '',
                          q = '' if simd_ext in sve else 'q', **fmtspec)
        else:
            return 'return {s}v{op}{q}_{suf}({in0}, {in1});'. \
                format(op=func, s = 's' if simd_ext in sve else '',
                       q = '' if simd_ext in sve else 'q', **fmtspec)
    elif simd_ext == 'neon128':
        armop = {'zip1': 'zipq', 'zip2': 'zipq', 'uzp1': 'uzpq',
                 'uzp2': 'uzpq'}
        prefix = { 'i': 'int', 'u': 'uint', 'f': 'float' }
        neon_typ = '{}{}x{}x2_t'. \
            format(prefix[typ[0]], typ[1:], 128 // int(typ[1:]))
        if typ == 'f16':
            if func in ['zip1', 'zip2']:
                return '''\
                nsimd_{simd_ext}_v{typ} ret;
                float32x4x2_t tmp = v{op}_f32({in0}.v{i}, {in1}.v{i});
                ret.v0 = tmp.val[0];
                ret.v1 = tmp.val[1];
                return ret;
                '''.format(i = '0' if func == 'zip1' else '1',
                           op=armop[func], **fmtspec)
            else:
                return '''\
                nsimd_{simd_ext}_v{typ} ret;
                float32x4x2_t tmp0 = vuzpq_f32({in0}.v0, {in0}.v1);
                float32x4x2_t tmp1 = vuzpq_f32({in1}.v0, {in1}.v1);
                ret.v0 = tmp0.val[{i}];
                ret.v1 = tmp1.val[{i}];
                return ret;
                '''.format(i = '0' if func == 'uzp1' else '1', **fmtspec)
        elif typ in ['i64', 'u64']:
            return '''\
            {typ} buf0[2], buf1[2];
            {typ} ret[2];
            vst1q_{suf}(buf0, {in0});
            vst1q_{suf}(buf1, {in1});
            ret[0] = buf0[{i}];
            ret[1] = buf1[{i}];
            return vld1q_{suf}(ret);'''. \
                format(**fmtspec, i= '0' if func in ['zip1', 'uzp1'] else '1')
        elif  typ == 'f64' :
            return '''\
            nsimd_{simd_ext}_v{typ} ret;
            ret.v0 = {in0}.v{i};
            ret.v1 = {in1}.v{i};
            return ret;'''. \
                format(**fmtspec, i= '0' if func in ['zip1', 'uzp1'] else '1')
        else :
            return '''\
            {neon_typ} res;
            res = v{op}_{suf}({in0}, {in1});
            return res.val[{i}];'''. \
                format(neon_typ=neon_typ, op=armop[func], **fmtspec,
                       i = '0' if func in ['zip1', 'uzp1'] else '1')

def zip_unzip(func, simd_ext, typ):
    lo_hi = '''\
    nsimd_{simd_ext}_v{typ}x2 ret;
    ret.v0 = nsimd_{func}lo_{simd_ext}_{typ}({in0}, {in1});
    ret.v1 = nsimd_{func}hi_{simd_ext}_{typ}({in0}, {in1});
    return ret;
    '''.format(func='zip' if func == 'zip' else 'unzip', **fmtspec)
    if simd_ext == 'aarch64' or simd_ext in sve:
        content = '''\
        nsimd_{simd_ext}_v{typ}x2 ret;
        ret.v0 = {s}v{func}1{q}_{suf}({in0}, {in1});
        ret.v1 = {s}v{func}2{q}_{suf}({in0}, {in1});
        return ret;'''.format(s = 's' if simd_ext in sve else '',
                              q = '' if simd_ext in sve else 'q',
                              func=func, **fmtspec)
        if typ == 'f16':
            return '''\
            #ifdef NSIMD_ARM_FP16
            {c}
            #else
            {default}
            #endif'''.\
                format(c=content, default=lo_hi, s = 's' if simd_ext in sve else '',
                       **fmtspec)
        else:
            return content
    else:
       prefix = { 'i': 'int', 'u': 'uint', 'f': 'float' }
       neon_typ = '{}{}x{}x2_t'.\
               format(prefix[typ[0]], typ[1:], 128 // int(typ[1:]))
       content = '''\
       nsimd_{simd_ext}_v{typ}x2 ret;
       {neon_typ} tmp = v{func}q_{suf}({in0}, {in1});
       ret.v0 = tmp.val[0];
       ret.v1 = tmp.val[1];
       return ret;'''\
           .format(func=func, neon_typ=neon_typ, **fmtspec)
       if typ in ['u64', 'i64', 'f64']:
           return lo_hi
       elif typ == 'f16':
           return '''\
           #ifdef NSIMD_ARM_FP16
           {content}
           #else
           {default}
           #endif'''.\
               format(content=content, default=lo_hi,
                      f='zip' if func == 'zip' else 'unzip', **fmtspec)
       else:
           return content

# -----------------------------------------------------------------------------
# gather

def gather(simd_ext, typ):
    le = max_len(simd_ext, typ)
    real_le = real_len(simd_ext, typ)

    if simd_ext in sve:
        emul = '''int i;
                  {typ} buf[{le}];
                  i{typnbits} offset_buf[{le}];
                  svst1_s{typnbits}({svtrue}, offset_buf, {in1});
                  for (i = 0; i < {real_le}; i++) {{
                    buf[i] = {in0}[offset_buf[i]];
                  }}
                  return svld1_{suf}({svtrue}, buf);'''. \
                  format(le=le, real_le=real_le, **fmtspec)
    else:
        emul = \
        '''nsimd_{simd_ext}_v{typ} ret;
           ret = vdupq_n_{suf}({in0}[vgetq_lane_s{typnbits}({in1}, 0)]);'''. \
           format(**fmtspec) + ''.join([
        '''ret = vsetq_lane_{suf}({in0}[
                     vgetq_lane_s{typnbits}({in1}, {i})], ret, {i});\n'''. \
                     format(i=i, **fmtspec) for i in range(1, le)]) + \
          'return ret;'
    if typ == 'f16':
        if simd_ext in sve:
            return emul
        return '''#ifdef NSIMD_ARM_FP16
                    {emul}
                  #else
                    nsimd_{simd_ext}_vf16 ret;
                    f32 buf[8];
                  '''.format(emul=emul, **fmtspec) + \
                  ''.join(['buf[{i}] = nsimd_f16_to_f32({in0}[' \
                           'vgetq_lane_s16({in1}, {i})]);\n'. \
                           format(i=i, **fmtspec) for i in range(4)]) + \
                  ''.join(['buf[4 + {i}] = nsimd_f16_to_f32({in0}[' \
                           'vgetq_lane_s16({in1}, 4 + {i})]);\n'. \
                           format(i=i, **fmtspec) for i in range(4)]) + \
               '''  ret.v0 = vld1q_f32(buf);
                    ret.v1 = vld1q_f32(buf + 4);
                    return ret;
                  #endif'''.format(**fmtspec)
    if simd_ext == 'neon128' and typ == 'f64':
        return '''nsimd_neon128_vf64 ret;
                  i64 offset_buf[2];
                  vst1q_s64(offset_buf, {in1});
                  ret.v0 = {in0}[offset_buf[0]];
                  ret.v1 = {in0}[offset_buf[1]];
                  return ret;'''.format(**fmtspec)
    if simd_ext in neon or typ in ['i8', 'u8', 'i16', 'u16']:
        return emul
    # getting here means SVE
    return 'return svld1_gather_s{typnbits}index_{suf}({svtrue}, {in0}, ' \
           '{in1});'.format(**fmtspec)

# -----------------------------------------------------------------------------
# linear gather

def gather_linear(simd_ext, typ):
    if simd_ext in sve:
        if typ in ['i8', 'u8', 'i16', 'u16', 'f16']:
            le = max_len(simd_ext, typ)
            real_le = real_len(simd_ext, typ)
            return '''{typ} buf[{le}];
                      int i;
                      for (i = 0; i < {real_le}; i++) {{
                        buf[i] = {in0}[i * {in1}];
                      }}
                      return svld1_{suf}({svtrue}, buf);'''. \
                      format(le=le, real_le=real_le, **fmtspec)
        else:
            return 'return svld1_gather_s{typnbits}index_{suf}({svtrue}, ' \
                   '{in0}, svindex_s{typnbits}(0, (i{typnbits}){in1}));'. \
                   format(**fmtspec)
    # getting here means neon128 and aarch64
    intrinsic = '''nsimd_{simd_ext}_v{typ} ret;
                   ret = vdupq_n_{suf}({in0}[0]);
                '''.format(**fmtspec) + ''.join([
                  'ret = vsetq_lane_{suf}({in0}[{i} * {in1}], ret, {i});\n'. \
                  format(i=i, **fmtspec) \
                  for i in range(1, 128 // int(fmtspec['typnbits']))]) + \
               '''return ret;'''
    if typ == 'f16':
        return '''#ifdef NSIMD_ARM_FP16
                    {intrinsic}
                  #else
                    nsimd_{simd_ext}_vf16 ret;
                    f32 buf[8];
                    int i;
                    for (i = 0; i < 8; i++) {{
                      buf[i] = nsimd_f16_to_f32({in0}[i * {in1}]);
                    }}
                    ret.v0 = vld1q_f32(buf);
                    ret.v1 = vld1q_f32(buf + 4);
                    return ret;
                  #endif'''.format(intrinsic=intrinsic, **fmtspec)
    if typ == 'f64' and simd_ext == 'neon128':
        return '''nsimd_neon128_vf64 ret;
                  ret.v0 = {in0}[0];
                  ret.v1 = {in0}[{in1}];
                  return ret;'''.format(**fmtspec)
    return intrinsic

# -----------------------------------------------------------------------------
# masked gather

def maskoz_gather(oz, simd_ext, typ):
    le = max_len(simd_ext, typ)
    real_le = real_len(simd_ext, typ)

    if simd_ext in sve:
        utyp = 'u{typnbits}'.format(**fmtspec)
        store = '''svst1_s{typnbits}({svtrue}, offset_buf, {in2});
                   svst1_{utyp}({svtrue}, mask, svsel_{utyp}(
                       {in0}, svdup_n_{utyp}(({utyp})-1), svdup_n_{utyp}(
                         ({utyp})0)));
                         '''.format(utyp=utyp, **fmtspec)
        if oz == 'z':
            store += 'svst1_{suf}({svtrue}, buf, svdup_n_{suf}(({typ})0));'. \
                     format(**fmtspec)
        else:
            store += 'svst1_{suf}({svtrue}, buf, {in3});'.format(**fmtspec)
        load = 'svld1_{suf}({svtrue}, buf)'.format(**fmtspec)
    else:
        store = '''vst1q_s{typnbits}(offset_buf, {in2});
                   vst1q_u{typnbits}(mask, {in0});'''.format(**fmtspec)
        if oz == 'z':
            store += 'vst1q_{suf}(buf, vdupq_n_{suf}(({typ})0));'. \
                     format(**fmtspec)
        else:
            store += 'vst1q_{suf}(buf, {in3});'.format(**fmtspec)
        load = 'vld1q_{suf}(buf)'.format(**fmtspec)

    emul = '''int i;
              {typ} buf[{le}];
              u{typnbits} mask[{le}];
              i{typnbits} offset_buf[{le}];
              {store}
              for (i = 0; i < {real_le}; i++) {{
                if (mask[i]) {{
                  buf[i] = {in1}[offset_buf[i]];
                }}
              }}
              return {load};'''. \
              format(le=le, real_le=real_le, store=store, load=load, **fmtspec)
    if typ == 'f16':
        if simd_ext in sve:
            return emul
        if oz == 'z':
            oz0 = 'vdupq_n_f32(0.0f)'
            oz1 = oz0
        else:
            oz0 = '{in3}.v0'.format(**fmtspec)
            oz1 = '{in3}.v1'.format(**fmtspec)
        return '''#ifdef NSIMD_ARM_FP16
                    {emul}
                  #else
                    nsimd_{simd_ext}_vf16 ret;
                    int i;
                    f32 buf[{le}];
                    u32 mask[{le}];
                    i16 offset_buf[{le}];
                    vst1q_s16(offset_buf, {in2});
                    vst1q_f32(buf, {oz0});
                    vst1q_f32(buf + {leo2}, {oz1});
                    vst1q_u32(mask, {in0}.v0);
                    vst1q_u32(mask + {leo2}, {in0}.v1);
                    for (i = 0; i < {le}; i++) {{
                      if (mask[i]) {{
                        buf[i] = nsimd_f16_to_f32({in1}[offset_buf[i]]);
                      }}
                    }}
                    ret.v0 = vld1q_f32(buf);
                    ret.v1 = vld1q_f32(buf + {leo2});
                    return ret;
                  #endif'''.format(emul=emul, leo2=le // 2, le=le, oz0=oz0,
                                   oz1=oz1, **fmtspec)
    if simd_ext == 'neon128' and typ == 'f64':
        oz0 = '0.0' if oz == 'z' else '{in3}.v0'.format(**fmtspec)
        oz1 = '0.0' if oz == 'z' else '{in3}.v1'.format(**fmtspec)
        return '''nsimd_neon128_vf64 ret;
                  i64 offset_buf[2];
                  vst1q_s64(offset_buf, {in2});
                  if ({in0}.v0) {{
                    ret.v0 = {in1}[offset_buf[0]];
                  }} else {{
                    ret.v0 = {oz0};
                  }}
                  if ({in0}.v1) {{
                    ret.v1 = {in1}[offset_buf[1]];
                  }} else {{
                    ret.v1 = {oz1};
                  }}
                  return ret;'''.format(oz0=oz0, oz1=oz1, **fmtspec)
    if simd_ext in neon or typ in ['i8', 'u8', 'i16', 'u16']:
        return emul
    # getting here means SVE
    oz0 = 'svdup_n_{suf}(({typ})0)'.format(**fmtspec) if oz == 'z' \
          else '{in3}'.format(**fmtspec)
    return '''return svsel_{suf}({in0}, svld1_gather_s{typnbits}index_{suf}(
                         {in0}, {in1}, {in2}), {oz0});'''. \
                         format(oz0=oz0, **fmtspec)

# -----------------------------------------------------------------------------
# scatter

def scatter(simd_ext, typ):
    le = max_len(simd_ext, typ)
    real_le = real_len(simd_ext, typ)

    if simd_ext in sve:
        emul = '''int i;
                  {typ} buf[{le}];
                  i{typnbits} offset_buf[{le}];
                  svst1_s{typnbits}({svtrue}, offset_buf, {in1});
                  svst1_{suf}({svtrue}, buf, {in2});
                  for (i = 0; i < {real_le}; i++) {{
                    {in0}[offset_buf[i]] = buf[i];
                  }}'''.format(le=le, real_le=real_le, **fmtspec)
    else:
        emul = '\n'.join(['{in0}[vgetq_lane_s{typnbits}({in1}, {i})] = ' \
                          'vgetq_lane_{suf}({in2}, {i});\n'. \
                          format(i=i, **fmtspec) for i in range(int(le))])

    if typ == 'f16':
        if simd_ext in sve:
            return emul
        return '''#ifdef NSIMD_ARM_FP16
                    {emul}
                  #else
                  '''.format(emul=emul) + \
                  '\n'.join(['{in0}[vgetq_lane_s16({in1}, {i})] = ' \
                             'nsimd_f32_to_f16(vgetq_lane_f32({in2}.v0, '
                             '{i}));\n'.format(i=i, **fmtspec) \
                             for i in range(4)]) + \
                  '\n'.join(['{in0}[vgetq_lane_s16({in1}, 4 + {i})] = ' \
                             'nsimd_f32_to_f16(vgetq_lane_f32({in2}.v1, '
                             '{i}));\n'.format(i=i, **fmtspec) \
                             for i in range(4)]) + \
               '''
                  #endif'''
    if simd_ext == 'neon128' and typ == 'f64':
        return '''i64 offset_buf[2];
                  vst1q_s64(offset_buf, {in1});
                  {in0}[offset_buf[0]] = {in2}.v0;
                  {in0}[offset_buf[1]] = {in2}.v1;'''.format(**fmtspec)
    if simd_ext in neon or typ in ['i8', 'u8', 'i16', 'u16']:
        return emul
    # getting here means SVE
    return 'svst1_scatter_s{typnbits}index_{suf}({svtrue}, {in0}, ' \
           '{in1}, {in2});'.format(le=le, **fmtspec)

# -----------------------------------------------------------------------------
# linear scatter

def scatter_linear(simd_ext, typ):
    if simd_ext in sve:
        if typ in ['i8', 'u8', 'i16', 'u16', 'f16']:
            le = max_len(simd_ext, typ)
            real_le = real_len(simd_ext, typ)
            return '''{typ} buf[{le}];
                      int i;
                      svst1_{suf}({svtrue}, buf, {in2});
                      for (i = 0; i < {real_le}; i++) {{
                        {in0}[i * {in1}] = buf[i];
                      }}'''.format(le=le, real_le=real_le, **fmtspec)
        else:
            return 'svst1_scatter_s{typnbits}index_{suf}({svtrue}, {in0}, ' \
                   'svindex_s{typnbits}(0, (i{typnbits}){in1}), {in2});'. \
                   format(**fmtspec)
    # getting here means neon128 and aarch64
    intrinsic = '\n'.join([
      '{in0}[{i} * {in1}] = vgetq_lane_{suf}({in2}, {i});'. \
      format(i=i, **fmtspec) for i in range(128 // int(fmtspec['typnbits']))])
    if typ == 'f16':
        return '''#ifdef NSIMD_ARM_FP16
                    {intrinsic}
                  #else
                    f32 buf[8];
                    int i;
                    vst1q_f32(buf, {in2}.v0);
                    vst1q_f32(buf + 4, {in2}.v1);
                    for (i = 0; i < 8; i++) {{
                      {in0}[i * {in1}] = nsimd_f32_to_f16(buf[i]);
                    }}
                  #endif'''.format(intrinsic=intrinsic, **fmtspec)
    if typ == 'f64' and simd_ext == 'neon128':
        return '''{in0}[0] = {in2}.v0;
                  {in0}[{in1}] = {in2}.v1;'''.format(**fmtspec)
    return intrinsic

# -----------------------------------------------------------------------------
# mask_scatter

def mask_scatter(simd_ext, typ):
    le = max_len(simd_ext, typ)
    real_le = real_len(simd_ext, typ)

    if simd_ext in sve:
        store = '''svst1_s{typnbits}({svtrue}, offset_buf, {in2});
                   svst1_u{typnbits}({svtrue}, mask, svsel_u{typnbits}(
                       {in0}, svdup_n_u{typnbits}((u{typnbits})1),
                              svdup_n_u{typnbits}((u{typnbits})0)));
                   svst1_{suf}({svtrue}, buf, {in3});'''.format(**fmtspec)
    else:
        store = '''vst1q_s{typnbits}(offset_buf, {in2});
                   vst1q_{suf}(buf, {in3});
                   vst1q_u{typnbits}(mask, {in0});'''.format(**fmtspec)

    emul = '''int i;
              {typ} buf[{le}];
              u{typnbits} mask[{le}];
              i{typnbits} offset_buf[{le}];
              {store}
              for (i = 0; i < {real_le}; i++) {{
                if (mask[i]) {{
                  {in1}[offset_buf[i]] = buf[i];
                }}
              }}'''.format(le=le, real_le=real_le, store=store, **fmtspec)
    if typ == 'f16':
        if simd_ext in sve:
            return emul
        return '''#ifdef NSIMD_ARM_FP16
                    {emul}
                  #else
                    int i;
                    f32 buf[{le}];
                    u32 mask[{le}];
                    i16 offset_buf[{le}];
                    vst1q_s16(offset_buf, {in2});
                    vst1q_f32(buf, {in3}.v0);
                    vst1q_f32(buf + {leo2}, {in3}.v1);
                    vst1q_u32(mask, {in0}.v0);
                    vst1q_u32(mask + {leo2}, {in0}.v1);
                    for (i = 0; i < {le}; i++) {{
                      if (mask[i]) {{
                        {in1}[offset_buf[i]] = nsimd_f32_to_f16(buf[i]);
                      }}
                    }}
                  #endif'''.format(emul=emul, le=le, leo2=le // 2, **fmtspec)
    if simd_ext == 'neon128' and typ == 'f64':
        return '''i64 offset_buf[2];
                  vst1q_s64(offset_buf, {in2});
                  if ({in0}.v0) {{
                    {in1}[offset_buf[0]] = {in3}.v0;
                  }}
                  if ({in0}.v1) {{
                    {in1}[offset_buf[1]] = {in3}.v1;
                  }}'''.format(**fmtspec)
    if simd_ext in neon or typ in ['i8', 'u8', 'i16', 'u16']:
        return emul
    # getting here means SVE
    return 'svst1_scatter_s{typnbits}index_{suf}({in0}, {in1}, ' \
           '{in2}, {in3});'.format(le=le, **fmtspec)


# -----------------------------------------------------------------------------
# get_impl function

def get_impl(opts, func, simd_ext, from_typ, to_typ):
    global fmtspec

    simd_ext2 = simd_ext if not simd_ext in fixed_sized_sve else 'sve'

    fmtspec = {
      'simd_ext': simd_ext,
      'simd_ext2': simd_ext2,
      'typ': from_typ,
      'from_typ': from_typ,
      'to_typ': to_typ,
      'suf': suf(from_typ),
      'in0': common.in0,
      'in1': common.in1,
      'in2': common.in2,
      'in3': common.in3,
      'in4': common.in4,
      'in5': common.in5,
      'typnbits': from_typ[1:],
      'svtrue': 'svptrue_b{}()'.format(from_typ[1:]),
      'svetyp': sve_typ(from_typ),
    }

    impls = {
        'loada': lambda: load1234(opts, simd_ext, from_typ, 1),
        'masko_loada1': lambda: maskoz_load('o', simd_ext, from_typ),
        'maskz_loada1': lambda: maskoz_load('z', simd_ext, from_typ),
        'load2a': lambda: load1234(opts, simd_ext, from_typ, 2),
        'load3a': lambda: load1234(opts, simd_ext, from_typ, 3),
        'load4a': lambda: load1234(opts, simd_ext, from_typ, 4),
        'loadu': lambda: load1234(opts, simd_ext, from_typ, 1),
        'masko_loadu1': lambda: maskoz_load('o', simd_ext, from_typ),
        'maskz_loadu1': lambda: maskoz_load('z', simd_ext, from_typ),
        'load2u': lambda: load1234(opts, simd_ext, from_typ, 2),
        'load3u': lambda: load1234(opts, simd_ext, from_typ, 3),
        'load4u': lambda: load1234(opts, simd_ext, from_typ, 4),
        'storea': lambda: store1234(opts, simd_ext, from_typ, 1),
        'mask_storea1': lambda: mask_store(simd_ext, from_typ),
        'store2a': lambda: store1234(opts, simd_ext, from_typ, 2),
        'store3a': lambda: store1234(opts, simd_ext, from_typ, 3),
        'store4a': lambda: store1234(opts, simd_ext, from_typ, 4),
        'storeu': lambda: store1234(opts, simd_ext, from_typ, 1),
        'mask_storeu1': lambda: mask_store(simd_ext, from_typ),
        'store2u': lambda: store1234(opts, simd_ext, from_typ, 2),
        'store3u': lambda: store1234(opts, simd_ext, from_typ, 3),
        'store4u': lambda: store1234(opts, simd_ext, from_typ, 4),
        'gather': lambda: gather(simd_ext, from_typ),
        'gather_linear': lambda: gather_linear(simd_ext, from_typ),
        'maskz_gather': lambda: maskoz_gather('z', simd_ext, from_typ),
        'masko_gather': lambda: maskoz_gather('o', simd_ext, from_typ),
        'scatter': lambda: scatter(simd_ext, from_typ),
        'scatter_linear': lambda: scatter_linear(simd_ext, from_typ),
        'mask_scatter': lambda: mask_scatter(simd_ext, from_typ),
        'andb': lambda: binop2("andb", simd_ext2, from_typ),
        'xorb': lambda: binop2("xorb", simd_ext2, from_typ),
        'orb': lambda: binop2("orb", simd_ext2, from_typ),
        'andl': lambda: lop2(opts, "andl", simd_ext2, from_typ),
        'xorl': lambda: lop2(opts, "xorl", simd_ext2, from_typ),
        'orl': lambda: lop2(opts, "orl", simd_ext2, from_typ),
        'notb': lambda: not1(simd_ext2, from_typ),
        'notl': lambda: lnot1(opts, simd_ext2, from_typ),
        'andnotb': lambda: binop2("andnotb", simd_ext2, from_typ),
        'andnotl': lambda: lop2(opts, "andnotl", simd_ext2, from_typ),
        'add': lambda: addsub("add", simd_ext2, from_typ),
        'sub': lambda: addsub("sub", simd_ext2, from_typ),
        'adds': lambda: adds(simd_ext2, from_typ),
        'subs': lambda: subs(simd_ext2, from_typ),
        'div': lambda: div2(simd_ext2, from_typ),
        'sqrt': lambda: sqrt1(simd_ext2, from_typ),
        'len': lambda: len1(simd_ext, from_typ),
        'mul': lambda: mul2(simd_ext2, from_typ),
        'shl': lambda: shl_shr("shl", simd_ext2, from_typ),
        'shr': lambda: shl_shr("shr", simd_ext2, from_typ),
        'shra': lambda: shra(simd_ext2, from_typ),
        'set1': lambda: set1(simd_ext2, from_typ),
        'set1l': lambda: lset1(simd_ext2, from_typ),
        'eq': lambda: cmp2(opts, "eq", simd_ext2, from_typ),
        'lt': lambda: cmp2(opts, "lt", simd_ext2, from_typ),
        'le': lambda: cmp2(opts, "le", simd_ext2, from_typ),
        'gt': lambda: cmp2(opts, "gt", simd_ext2, from_typ),
        'ge': lambda: cmp2(opts, "ge", simd_ext2, from_typ),
        'ne': lambda: neq2(opts, simd_ext2, from_typ),
        'if_else1': lambda: if_else3(opts, simd_ext2, from_typ),
        'min': lambda: minmax2("min", simd_ext2, from_typ),
        'max': lambda: minmax2("max", simd_ext2, from_typ),
        'loadla': lambda: loadl(True, simd_ext2, from_typ),
        'loadlu': lambda: loadl(False, simd_ext2, from_typ),
        'storela': lambda: storel(True, simd_ext2, from_typ),
        'storelu': lambda: storel(False, simd_ext2, from_typ),
        'abs': lambda: abs1(simd_ext2, from_typ),
        'fma': lambda: fmafnma3("fma", simd_ext2, from_typ),
        'fnma': lambda: fmafnma3("fnma", simd_ext2, from_typ),
        'fms': lambda: fmsfnms3("fms", simd_ext2, from_typ),
        'fnms': lambda: fmsfnms3("fnms", simd_ext2, from_typ),
        'ceil': lambda: round1("ceil", simd_ext2, from_typ),
        'floor': lambda: round1("floor", simd_ext2, from_typ),
        'trunc': lambda: round1("trunc", simd_ext2, from_typ),
        'round_to_even': lambda: round1("round_to_even", simd_ext2, from_typ),
        'all': lambda: allany1(opts, "all", simd_ext2, from_typ),
        'any': lambda: allany1(opts, "any", simd_ext2, from_typ),
        'reinterpret': lambda: reinterpret1(simd_ext2, from_typ, to_typ),
        'reinterpretl': lambda: reinterpretl1(simd_ext2, from_typ, to_typ),
        'cvt': lambda: convert1(simd_ext2, from_typ, to_typ),
        'rec11': lambda: recs1("rec11", simd_ext2, from_typ),
        'rec8': lambda: recs1("rec8", simd_ext2, from_typ),
        'rsqrt11': lambda: recs1("rsqrt11", simd_ext2, from_typ),
        'rsqrt8': lambda: recs1("rsqrt8", simd_ext2, from_typ),
        'rec': lambda: recs1("rec", simd_ext2, from_typ),
        'neg': lambda: neg1(simd_ext2, from_typ),
        'nbtrue': lambda: nbtrue1(opts, simd_ext2, from_typ),
        'reverse': lambda: reverse1(simd_ext2, from_typ),
        'addv': lambda: addv(simd_ext2, from_typ),
        'upcvt': lambda: upcvt1(simd_ext2, from_typ, to_typ),
        'downcvt': lambda: downcvt1(simd_ext2, from_typ, to_typ),
        'to_logical': lambda: to_logical1(opts, simd_ext2, from_typ),
        'to_mask': lambda: to_mask1(opts, simd_ext2, from_typ),
        'ziplo': lambda: zip_unzip_half("zip1", simd_ext2, from_typ),
        'ziphi': lambda: zip_unzip_half("zip2", simd_ext2, from_typ),
        'unziplo': lambda: zip_unzip_half("uzp1", simd_ext2, from_typ),
        'unziphi': lambda: zip_unzip_half("uzp2", simd_ext2, from_typ),
        'zip' : lambda: zip_unzip("zip", simd_ext2, from_typ),
        'unzip' : lambda: zip_unzip("uzp", simd_ext2, from_typ),
        'mask_for_loop_tail': lambda : mask_for_loop_tail(simd_ext, from_typ),
        'iota': lambda : iota(simd_ext2, from_typ)
    }
    if simd_ext not in get_simd_exts():
        raise ValueError('Unknown SIMD extension "{}"'.format(simd_ext))
    if not from_typ in common.types:
        raise ValueError('Unknown type "{}"'.format(from_typ))
    if not func in impls:
        return common.NOT_IMPLEMENTED
    else:
        return impls[func]()
