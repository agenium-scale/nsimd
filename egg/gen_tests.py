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

import os
import sys
import common
import operators
from datetime import date

# -----------------------------------------------------------------------------
# Get filename for test

def get_filename(opts, op, typ, lang):
    pp_lang = { 'c_base': 'C (base API)',
                'cxx_base' : 'C++ (base API)',
                'cxx_adv' : 'C++ (advanced API)' }
    tests_dir = os.path.join(opts.tests_dir, lang)
    common.mkdir_p(tests_dir)
    filename = os.path.join(tests_dir, '{}.{}.{}'.format(op.name, typ,
                 'c' if lang == 'c_base' else 'cpp'));
    if common.can_create_filename(opts, filename):
        return filename
    else:
        return None

# -----------------------------------------------------------------------------
# Get standard includes

def get_includes(lang):
    ret = '#include <nsimd/nsimd.h>\n'
    if lang == 'cxx_adv':
        ret += '#include <nsimd/cxx_adv_api.hpp>\n'
    if lang == 'c_base':
        ret += '''#include <stdlib.h>
                  #include <stdio.h>
                  #include <errno.h>
                  #include <string.h>'''
    else:
        ret += '''#include <cstdlib>
                  #include <cstdio>
                  #include <cerrno>
                  #include <cstring>'''
    return ret

# -----------------------------------------------------------------------------
# Function to compute number of common bits between two floatting points
# numbers

get_2th_power = \
'''
/* One could use ldexp, but it is not available in C89. Plus we really
   don't care about performences here. */
double get_2th_power(int a) {
  double ret = 1.0;
  if (a == 0) {
    return ret;
  }
  if (a < 0) {
    int i;
    for (i = 0; i < (-a); i++) {
      ret /= 2.0;
    }
    return ret;
  }
  /* a > 0 */ {
    int i;
    for (i = 0; i < a; i++) {
      ret *= 2.0;
    }
    return ret;
  }
}

/* ------------------------------------------------------------------------- */

'''

relative_distance_cpp = \
'''
double relative_distance(double a, double b) {
  double ma, mi;

  if (std::isnan(a) && std::isnan(b)) {
    return 0.0;
  }

  if (std::isnan(a) || std::isnan(b)) {
    return -1.;
  }

  if (std::isinf(a) && std::isinf(b) && ((a > 0 && b > 0) || (a<0&&b<0))) {
    return 0.0;
  }

  if (std::isinf(a) || std::isinf(b)) {
    return -1.;
  }

  a = (a > 0.0 ? a : -a);
  b = (b > 0.0 ? b : -b);
  ma = (a > b ? a : b);
  mi = (a < b ? a : b);

  if (ma == 0.0) {
    return 0.0;
  }

  return (ma - mi) / ma;
}

/* ------------------------------------------------------------------------- */
''' + get_2th_power

relative_distance_c = \
'''
double relative_distance(double a, double b) {
  double ma, mi;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
  if (isnan(a) && isnan(b)) {
    return 0.0;
  }

  if (isnan(a) || isnan(b)) {
    return -1.;
  }

  if (isinf(a) * isinf(b) > 0) {
    return 0.0;
  }

  if (isinf(a) || isinf(b)) {
    return -1.;
  }
#pragma GCC diagnostic pop

  a = (a > 0.0 ? a : -a);
  b = (b > 0.0 ? b : -b);
  ma = (a > b ? a : b);
  mi = (a < b ? a : b);

  if (ma == 0.0) {
    return 0.0;
  }

  return (ma - mi) / ma;
}

/* ------------------------------------------------------------------------- */
''' + get_2th_power

# -----------------------------------------------------------------------------
# Template for a lot of tests

template = \
'''{includes}

#define SIZE (2048 / {sizeof})

#define STATUS "test of {op_name} over {typ}"

#define CHECK(a) {{ \\
  errno = 0; \\
  if (!(a)) {{ \\
    fprintf(stderr, "ERROR: " #a ":%d: %s\\n", \\
            __LINE__, strerror(errno)); \\
    fflush(stderr); \\
    exit(EXIT_FAILURE); \\
  }} \\
}}

/* ------------------------------------------------------------------------- */
{extra_code}

int comp_function({typ} mpfr_out, {typ} nsimd_out)
{{
   {comp};
}}

int main(void) {{
  int vi, i, step;
  {typ} *vout0, *vout1;
  {vin_defi}

  CHECK(vout0 = ({typ}*)nsimd_aligned_alloc(SIZE * {sizeof}));
  CHECK(vout1 = ({typ}*)nsimd_aligned_alloc(SIZE * {sizeof}));

  step = vlen({typ});

  fprintf(stdout, STATUS "...\\n");
  fflush(stdout);

  /* Fill input vector(s) with random values */
  for (i = 0; i < SIZE; i++) {{
    {vin_rand}
  }}

  /* Fill output vector 0 with reference values */
  for (i = 0; i < SIZE; i += {cpu_step}) {{
    /* This is a call directly to the cpu API of nsimd
       to ensure that we call the scalar version of the
       function */
    {vout0_comp}
  }}

  /* Fill output vector 1 with computed values */
  for (i = 0; i < SIZE; i += step) {{
    {vout1_comp}
  }}

  /* Compare results */
  for (vi = 0; vi < SIZE; vi += step) {{
    for (i = vi; i < vi + step; i++) {{
      if (comp_function(vout0[i], vout1[i])) {{
        fprintf(stdout, STATUS "... FAIL\\n");
        fflush(stdout);
        return -1;
      }}
    }}
  }}

  fprintf(stdout, STATUS "... OK\\n");
  fflush(stdout);
  return 0;
}}'''

# -----------------------------------------------------------------------------
# Common to most of the tests

def get_content(op, typ, lang):
    cast = 'f32' if typ in ['f16', 'f32'] else 'f64'

    # By default we use emulation functions ("cpu" architecture) for testing
    # in which case increment is given by nsimd_cpu_len()
    cpu_step = 'nsimd_len_cpu_{}()'.format(typ)

    # For floatting points generate some non integer inputs
    if typ in common.iutypes:
        rand = '(1 << (rand() % 4))'
    else:
        if op.src:
          rand = '({cast})2 * ({cast})rand() / ({cast})RAND_MAX'. \
                 format(cast=cast)
        else:
          rand = '({cast})(1 << (rand() % 4)) / ({cast})(1 << (rand() % 4))'. \
                 format(cast=cast)

    # For signed types, make some positive and negative inputs
    if op.name not in ['sqrt', 'rsqrt11'] and typ in common.itypes:
        rand = '(2 * (rand() % 2) - 1) * {}'.format(rand)
    if op.name not in ['sqrt', 'rsqrt11'] and typ in common.ftypes:
        rand = '({})(2 * (rand() % 2) - 1) * {}'.format(cast, rand)

    # Depending on function parameters, generate specific input, ...
    if all(e == 'v' for e in op.params) or all(e == 'l' for e in op.params):
        logical = 'l' if op.params[0] == 'l' else ''
        if logical == 'l':
            rand = '(1 << (rand() % 2))' if typ != 'f16' \
                                         else '(float)(1 << (rand() % 2))'
        nargs = range(1, len(op.params))

        # Make vin_defi
        code = ['{} *vin{};'.format(typ, i) for i in nargs]
        code += ['CHECK(vin{} = ({}*)nsimd_aligned_alloc(SIZE * {}));'. \
                 format(i, typ, common.sizeof(typ)) for i in nargs]
        vin_defi = '\n'.join(code)
        if typ == 'f16':
            code = ['vin{}[i] = nsimd_f32_to_f16({});'. \
                    format(i, rand) for i in nargs]
        else:
            code = ['vin{}[i] = ({})({});'.format(i, typ, rand) for i in nargs]
        vin_rand = '\n'.join(code)

        # Make vout0_comp
        # We use MPFR on Linux to compare numerical results, but it is only on
        # Linux as MPFR does not play well on Windows. On Windows we compare
        # against the cpu implementation. When using MPFR, we set one element
        # at a time => cpu_step = '1'
        if op.tests_mpfr and sys.platform.startswith('linux'):
            cpu_step = '1'
            variables = ', '.join(['a{}'.format(i) for i in nargs])
            mpfr_inits = '\n'.join(['mpfr_init2(a{}, 64);'.format(i) \
                                   for i in nargs])
            if typ == 'f16':
                mpfr_set = '''mpfr_set_flt(a{i}, nsimd_u16_to_f32(
                                ((u16 *)vin{i})[i]), MPFR_RNDN);'''
                vout0_set = '''((u16 *)vout0)[i] = nsimd_f32_to_u16(
                                 mpfr_get_flt(c, MPFR_RNDN));'''
            elif typ == 'f32':
                mpfr_set = 'mpfr_set_flt(a{i}, vin{i}[i], MPFR_RNDN);'
                vout0_set = 'vout0[i] = mpfr_get_flt(c, MPFR_RNDN);'
            else:
                mpfr_set = 'mpfr_set_d(a{i}, vin{i}[i], MPFR_RNDN);'
                vout0_set = 'vout0[i] = mpfr_get_d(c, MPFR_RNDN);'
            mpfr_sets = '\n'.join([mpfr_set.format(i=j) for j in nargs])
            mpfr_clears = '\n'.join(['mpfr_clear(a{});'.format(i) \
                                     for i in nargs])
            vout0_comp = \
            '''mpfr_t c, {variables};
               mpfr_init2(c, 64);
               {mpfr_inits}
               {mpfr_sets}
               {mpfr_op_name}(c, {variables}, MPFR_RNDN);
               {vout0_set}
               mpfr_clear(c);
               {mpfr_clears}'''. \
               format(variables=variables, mpfr_sets=mpfr_sets,
                      mpfr_clears=mpfr_clears, vout0_set=vout0_set,
                      mpfr_op_name=op.tests_mpfr_name(), mpfr_inits=mpfr_inits)
        else:
            args = ', '.join(['va{}'.format(i) for i in nargs])
            code = ['nsimd_cpu_v{}{} {}, vc;'.format(logical, typ, args)]
            code += ['va{} = nsimd_load{}u_cpu_{}(&vin{}[i]);'. \
                     format(i, logical, typ, i) for i in nargs]
            code += ['vc = nsimd_{}_cpu_{}({});'.format(op.name, typ, args)]
            code += ['nsimd_store{}u_cpu_{}(&vout0[i], vc);'. \
                     format(logical, typ)]
            vout0_comp = '\n'.join(code)

        # Make vout1_comp
        args = ', '.join(['va{}'.format(i) for i in nargs])
        if lang == 'c_base':
            code = ['vec{}({}) {}, vc;'.format(logical, typ, args)]
            code += ['va{} = vload{}u(&vin{}[i], {});'. \
                     format(i, logical, i, typ) for i in nargs]
            code += ['vc = v{}({}, {});'.format(op.name, args, typ)]
            code += ['vstore{}u(&vout1[i], vc, {});'.format(logical, typ)]
            vout1_comp = '\n'.join(code)
        if lang == 'cxx_base':
            code = ['vec{}({}) {}, vc;'.format(logical, typ, args)]
            code += ['va{} = nsimd::load{}u(&vin{}[i], {}());'. \
                     format(i, logical, i, typ) for i in nargs]
            code += ['vc = nsimd::{}({}, {}());'.format(op.name, args, typ)]
            code += ['nsimd::store{}u(&vout1[i], vc, {}());'. \
                     format(logical, typ)]
            vout1_comp = '\n'.join(code)
        if lang == 'cxx_adv':
            code = ['nsimd::pack{}<{}> {}, vc;'.format(logical, typ, args)]
            code += ['''va{i} = nsimd::load{logical}u<
                                  nsimd::pack{logical}<{typ}> >(
                                      &vin{i}[i]);'''. \
                     format(i=i, logical=logical, typ=typ) for i in nargs]
            if op.cxx_operator:
                if len(op.params[1:]) == 1:
                    code += ['vc = {}va1;'. \
                             format(op.cxx_operator[8:])]
                if len(op.params[1:]) == 2:
                    code += ['vc = va1 {} va2;'. \
                             format(op.cxx_operator[8:])]
            else:
                code += ['vc = nsimd::{}({});'.format(op.name, args)]
            code += ['nsimd::store{}u(&vout1[i], vc);'.format(logical, typ)]
            vout1_comp = '\n'.join(code)
    elif op.params == ['l', 'v', 'v']:
        vin_defi = \
        '''{typ} *vin1, *vin2;
           CHECK(vin1 = ({typ}*)nsimd_aligned_alloc(SIZE * {sizeof}));
           CHECK(vin2 = ({typ}*)nsimd_aligned_alloc(SIZE * {sizeof}));'''. \
           format(typ=typ, sizeof=common.sizeof(typ))
        if typ == 'f16':
            vin_rand = '''vin1[i] = nsimd_f32_to_f16((float)(rand() % 4));
                          vin2[i] = nsimd_f32_to_f16((float)(rand() % 4));'''
        else:
            vin_rand = '''vin1[i] = ({typ})(rand() % 4);
                          vin2[i] = ({typ})(rand() % 4);'''.format(typ=typ)
        vout0_comp = '''nsimd_cpu_v{typ} va1, va2;
                        nsimd_cpu_vl{typ} vc;
                        va1 = nsimd_loadu_cpu_{typ}(&vin1[i]);
                        va2 = nsimd_loadu_cpu_{typ}(&vin2[i]);
                        vc = nsimd_{op_name}_cpu_{typ}(va1, va2);
                        nsimd_storelu_cpu_{typ}(&vout0[i], vc);'''. \
                        format(typ=typ, op_name=op.name)
        if lang == 'c_base':
            vout1_comp = '''vec({typ}) va1, va2;
                            vecl({typ}) vc;
                            va1 = vloadu(&vin1[i], {typ});
                            va2 = vloadu(&vin2[i], {typ});
                            vc = v{op_name}(va1, va2, {typ});
                            vstorelu(&vout1[i], vc, {typ});'''. \
                            format(typ=typ, op_name=op.name)
        if lang == 'cxx_base':
            vout1_comp = '''vec({typ}) va1, va2;
                            vecl({typ}) vc;
                            va1 = nsimd::loadu(&vin1[i], {typ}());
                            va2 = nsimd::loadu(&vin2[i], {typ}());
                            vc = nsimd::{op_name}(va1, va2, {typ}());
                            nsimd::storelu(&vout1[i], vc, {typ}());'''. \
                            format(typ=typ, op_name=op.name)
        if lang == 'cxx_adv':
            if op.cxx_operator:
                do_computation = 'vc = va1 {} va2;'. \
                                 format(op.cxx_operator[8:])
            else:
                do_computation = 'vc = nsimd::{}(va1, va2, {}());'. \
                                 format(op.name, typ)
            vout1_comp = '''nsimd::pack<{typ}> va1, va2;
                            nsimd::packl<{typ}> vc;
                            va1 = nsimd::loadu<nsimd::pack<{typ}> >(&vin1[i]);
                            va2 = nsimd::loadu<nsimd::pack<{typ}> >(&vin2[i]);
                            {do_computation}
                            nsimd::storelu(&vout1[i], vc);'''. \
                            format(typ=typ, op_name=op.name,
                                   do_computation=do_computation)
    elif op.params == ['v', 'v', 'p']:
        vin_defi = \
        '''{typ} *vin1;
           CHECK(vin1 = ({typ}*)nsimd_aligned_alloc(SIZE * {sizeof}));'''. \
           format(typ=typ, sizeof=common.sizeof(typ))
        vin_rand = 'vin1[i] = ({typ})(rand() % 4);'.format(typ=typ)
        vout0_comp = '''nsimd_cpu_v{typ} va1, vc;
                        va1 = nsimd_loadu_cpu_{typ}(&vin1[i]);
                        vc = nsimd_{op_name}_cpu_{typ}(va1, (i / step) % 7);
                        nsimd_storeu_cpu_{typ}(&vout0[i], vc);'''. \
                        format(typ=typ, op_name=op.name)
        if lang == 'c_base':
            vout1_comp = '''vec({typ}) va1, vc;
                            va1 = vloadu(&vin1[i], {typ});
                            vc = v{op_name}(va1, (i / step) % 7, {typ});
                            vstoreu(&vout1[i], vc, {typ});'''. \
                            format(typ=typ, op_name=op.name)
        if lang == 'cxx_base':
            vout1_comp = \
            '''vec({typ}) va1, vc;
               va1 = nsimd::loadu(&vin1[i], {typ}());
               vc = nsimd::{op_name}(va1, (i / step) % 7, {typ}());
               nsimd::storeu(&vout1[i], vc, {typ}());'''. \
               format(typ=typ, op_name=op.name)
        if lang == 'cxx_adv':
            if op.cxx_operator:
                do_computation = 'vc = va1 {} ((i / step) % 7);'. \
                                 format(op.cxx_operator[8:])
            else:
                do_computation = 'vc = nsimd::{}(va1, (i / step) % 7);'. \
                                 format(op.name)
            vout1_comp = \
            '''nsimd::pack<{typ}> va1, vc;
               va1 = nsimd::loadu<nsimd::pack<{typ}> >(&vin1[i]);
               {do_computation}
               nsimd::storeu(&vout1[i], vc);'''. \
               format(typ=typ, do_computation=do_computation)
    else:
        raise ValueError('No test available for operator "{}" on type "{}"'. \
                         format(op.name, typ))
    return { 'vin_defi': vin_defi, 'vin_rand': vin_rand, 'cpu_step': cpu_step,
             'vout0_comp': vout0_comp, 'vout1_comp': vout1_comp }

# -----------------------------------------------------------------------------
# Generate test in C, C++ (base API) and C++ (advanced API) for almost all
# tests

def gen_test(opts, op, typ, lang, ulps):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return

    content = get_content(op, typ, lang)

    if op.name in ['not', 'and', 'or', 'xor', 'andnot']:
        comp = 'return *({uT}*)&mpfr_out != *({uT}*)&nsimd_out'. \
               format(uT=common.bitfield_type[typ])
    else:
        if typ == 'f16':
            left = '(double)nsimd_f16_to_f32(mpfr_out)'
            right = '(double)nsimd_f16_to_f32(nsimd_out)'
        elif typ == 'f32':
            left = '(double)mpfr_out'
            right = '(double)nsimd_out'
        else:
            left = 'mpfr_out'
            right = 'nsimd_out'
        relative_distance = relative_distance_c if lang == 'c_base' \
                            else relative_distance_cpp
        if op.tests_ulps != None:
            comp = 'return relative_distance({}, {}) > get_2th_power(-{nbits})'. \
                   format(left, right, nbits=op.tests_ulps \
                          if typ != 'f16' else min(9, op.tests_ulps))
            extra_code = relative_distance
        elif op.src:
            if op.name in ulps:
                nbits = ulps[op.name][typ]["ulps"]
                nbits_dnz = ulps[op.name][typ]["ulps for denormalized output"]
                inf_error = ulps[op.name][typ]["Inf Error"]
                nan_error = ulps[op.name][typ]["NaN Error"]

                comp = '''#pragma GCC diagnostic push
                          #pragma GCC diagnostic ignored "-Wconversion"
                          #pragma GCC diagnostic ignored "-Wdouble-promotion"
                          '''
                if nan_error:
                    # Ignore error with NaN output, we know we will encounter some
                    comp += 'if ({isnan}((double){left})) return 0;\n'
                else:
                    # Return false if one is NaN and not the other
                    comp += 'if ({isnan}((double){left}) ^ isnan({rigth})) return 1;\n'

                if inf_error:
                    # Ignore error with infinite output, we know we will encounter some
                    comp += 'if ({isinf}((double){left})) return 0;\n'
                else:
                    # One is infinite and not the other
                    comp += 'if ({isinf}((double){left}) ^ {isinf}((double){rigth})) return 1;\n'
                    # Wrong sign for infinite
                    comp += 'if ({isinf}((double){left}) && {isinf}((double){rigth}) \
                                    && ({right}*{left} < 0)) \
                                        return 1;\n'

                comp += '''
                if ({isnormal}((double){left})) {{
                    return relative_distance({left}, {right}) > get_2th_power(-({nbits}));
                }} else {{
                    return relative_distance({left}, {right}) > get_2th_power(-({nbits_dnz}));
                }}
                #pragma GCC diagnostic pop
                '''

                if lang == 'c_base':
                    comp = comp.format(left = left,
                            right = right,
                            nbits = nbits,
                            nbits_dnz = nbits_dnz,
                            isnormal = 'isnormal',
                            isinf='isinf',
                            isnan='isnan')
                else:
                    comp = comp.format(left = left,
                            right = right,
                            nbits = nbits,
                            nbits_dnz = nbits_dnz,
                            isnormal = 'std::isnormal',
                            isinf='std::isinf',
                            isnan='std::isnan')

            else:
                nbits = {'f16': '10', 'f32': 21, 'f64': '48'}
                comp = 'return relative_distance({}, {}) > get_2th_power(-{nbits})'. \
                        format(left, right, nbits=nbits[typ])

            extra_code = relative_distance
        else:
            comp = 'return {} != {}'.format(left, right)
            extra_code = ''

    includes = get_includes(lang)
    if op.src or op.tests_ulps or op.tests_mpfr:
        if lang == 'c_base':
            includes = '''#define _POSIX_C_SOURCE 200112L

                          #include <math.h>
                          #include <float.h>
                          {}'''.format(includes)
        else:
            includes = '''#define _POSIX_C_SOURCE 200112L

                          #include <cmath>
                          #include <cfloat>
                          {}'''.format(includes)
        if op.tests_mpfr and sys.platform.startswith('linux'):
            includes = includes + '''
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wsign-conversion"
            #include <mpfr.h>
            #pragma GCC diagnostic pop
            '''

    with common.open_utf8(filename) as out:
        out.write(template.format( \
            includes=includes, sizeof=common.sizeof(typ), typ=typ,
            op_name=op.name, year=date.today().year, comp=comp,
            extra_code=extra_code, **content))
            #vin_defi=content['vin_defi'],
            #vin_rand=content['vin_rand'], vout0_comp=content['vout0_comp'],
            #vout1_comp=content['vout1_comp']))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for addv

def gen_addv(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return
    if lang == 'c_base':
        op_test = 'v{}(vloada(buf, {}), {})'.format(op.name, typ, typ)
        extra_code = relative_distance_c
    elif lang == 'cxx_base':
        op_test = 'nsimd::{}(nsimd::loada(buf, {}()), {}())'.format(op.name, typ, typ)
        extra_code = relative_distance_cpp
    else:
        op_test = 'nsimd::{}(nsimd::loada<nsimd::pack<{}>>(buf))'.format(op.name, typ)
        extra_code = relative_distance_cpp

    nbits = {'f16': '10', 'f32': '21', 'f64': '48'}
    head = '''#define _POSIX_C_SOURCE 200112L
              {includes}
              #include <float.h>
              #include <math.h>

              #define CHECK(a) {{ \\
                errno = 0; \\
                if (!(a)) {{ \\
                fprintf(stderr, "ERROR: " #a ":%d: %s\\n", \\
                        __LINE__, strerror(errno)); \\
                fflush(stderr); \\
                exit(EXIT_FAILURE); \\
                }} \\
              }}

              {extra_code}''' .format(year=date.today().year,
                                      includes=get_includes(lang),
                                      extra_code=extra_code)

    if typ == 'f16':
        # Variables initialization
        init = '''f16 res = nsimd_f32_to_f16(0.0f);
                  f32 ref = 0.0f;'''
        rand = '''nsimd_f32_to_f16((f32)(2 * (rand() % 2) - 1) *
                         (f32)(1 << (rand() % 4)) /
                           (f32)(1 << (rand() % 4)))'''
        init_statement = 'buf[i] = {};'.format(rand)
        ref_statement = 'ref += nsimd_u16_to_f32(((u16 *)buf)[i]);'
        test = '''if (relative_distance((double) ref,
                                        (double) nsimd_f16_to_f32(res)) >
                                          get_2th_power(-{nbits})) {{
                    return EXIT_FAILURE;
                  }}'''.format(nbits=nbits[typ])
    else:
        init = '''{typ} ref = ({typ})0;
                  {typ} res = ({typ})0;''' .format(typ=typ)
        rand = '''({typ})(2 * (rand() % 2) - 1) *
                      ({typ})(1 << (rand() % 4)) /
                        ({typ})(1 << (rand() % 4))'''.format(typ=typ)
        init_statement = 'buf[i] = {};'.format(rand)
        ref_statement = 'ref += buf[i];'
        test = '''if (relative_distance((double)ref,
                      (double)res) > get_2th_power(-{nbits})) {{
                    return EXIT_FAILURE;
                  }}'''.format(nbits=nbits[typ])
    with common.open_utf8(filename) as out:
        out.write(
            ''' \
            {head}

            int main(void) {{

            const int len = vlen({typ});
            {typ} *buf;
            int i;
            {init}

            fprintf(stdout, "test of {op_name} over {typ}...\\n");
            CHECK(buf = ({typ} *)nsimd_aligned_alloc(len * {sizeof}));

            for(i = 0; i < len; i++) {{
                {init_statement}
            }}

            for(i = 0; i < len; i++) {{
                {ref_statement}
            }}

            res = {op_test};

            {test}

            fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
            return EXIT_SUCCESS;
            }}
            '''.format(head=head, init=init, op_name=op.name, typ=typ,
                       sizeof=common.sizeof(typ),init_statement=init_statement,
                       ref_statement=ref_statement,op_test=op_test, test=test)
        )
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for all and any

def gen_all_any(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return
    if lang == 'c_base':
        op_test = 'v{}(vloadla(buf, {}), {})'.format(op.name, typ, typ)
    elif lang == 'cxx_base':
        op_test = 'nsimd::{}(nsimd::loadla(buf, {}()), {}())'. \
                  format(op.name, typ, typ)
    else:
        op_test = 'nsimd::{}(nsimd::loadla<nsimd::packl<{}> >(buf))'. \
                  format(op.name, typ)
    if typ == 'f16':
        scalar0 = 'nsimd_f32_to_f16(0)'
        scalar1 = 'nsimd_f32_to_f16(1)'
    else:
        scalar0 = '({})0'.format(typ)
        scalar1 = '({})1'.format(typ)
    with common.open_utf8(filename) as out:
        out.write(
        '''{includes}

           #define CHECK(a) {{ \\
             errno = 0; \\
             if (!(a)) {{ \\
               fprintf(stderr, "ERROR: " #a ":%d: %s\\n", \\
                       __LINE__, strerror(errno)); \\
               fflush(stderr); \\
               exit(EXIT_FAILURE); \\
             }} \\
           }}

           int main(void) {{
             int i;
             {typ} *buf;
             int len = vlen({typ});

             fprintf(stdout, "test of {op_name} over {typ}...\\n");
             CHECK(buf = ({typ}*)nsimd_aligned_alloc(len * {sizeof}));

             /* Test with all elements to true */
             for (i = 0; i < len; i++) {{
               buf[i] = {scalar1};
             }}
             if (!{op_test}) {{
               exit(EXIT_FAILURE);
             }}

             /* Test with all elements set to false */
             for (i = 0; i < len; i++) {{
               buf[i] = {scalar0};
             }}
             if ({op_test}) {{
               exit(EXIT_FAILURE);
             }}

             /* Test with only one element set to true */
             if (len > 1) {{
               buf[0] = {scalar1};
               if ({notl}{op_test}) {{
                 exit(EXIT_FAILURE);
               }}
             }}

             fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
             return EXIT_SUCCESS;
           }}'''.format(includes=get_includes(lang), op_name=op.name,
                        typ=typ, op_test=op_test, year=date.today().year,
                        notl='!' if op.name == 'any' else '', scalar0=scalar0,
                        scalar1=scalar1, sizeof=common.sizeof(typ)))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for load/store of degrees 2, 3 and 4

def gen_load_store(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return
    if op.name.startswith('load'):
        deg = op.name[4]
        align = op.name[5]
    elif op.name.startswith('store'):
        deg = op.name[5]
        align = op.name[6]
    variables = ', '.join(['v.v{}'.format(i) for i in range(0, int(deg))])
    if lang == 'c_base':
        load_store = \
        '''vecx{deg}({typ}) v = vload{deg}{align}(&vin[i], {typ});
           vstore{deg}{align}(&vout[i], {variables}, {typ});'''. \
           format(deg=deg, typ=typ, align=align, variables=variables)
    elif lang == 'cxx_base':
        load_store = \
        '''vecx{deg}({typ}) v = nsimd::load{deg}{align}(&vin[i], {typ}());
           nsimd::store{deg}{align}(&vout[i], {variables}, {typ}());'''. \
           format(deg=deg, typ=typ, align=align, variables=variables)
    else:
        load_store = \
        '''nsimd::packx{deg}<{typ}> v = nsimd::load{deg}{align}<
                                          nsimd::packx{deg}<{typ}> >(&vin[i]);
           nsimd::store{deg}{align}(&vout[i], {variables});'''. \
           format(deg=deg, typ=typ, align=align, variables=variables)
    if typ == 'f16':
        rand = '*((u16*)vin + i) = nsimd_f32_to_u16((float)(rand() % 10));'
        comp = '*((u16 *)vin + i) != *((u16 *)vout + i)'
    else:
        rand = 'vin[i] = ({})(rand() % 10);'.format(typ)
        comp = 'vin[i] != vout[i]'
    with common.open_utf8(filename) as out:
        out.write(
        '''{includes}

           #define SIZE (2048 / {sizeof})

           #define STATUS "test of {op_name} over {typ}"

           #define CHECK(a) {{ \\
             errno = 0; \\
             if (!(a)) {{ \\
               fprintf(stderr, "ERROR: " #a ":%d: %s\\n", \\
                       __LINE__, strerror(errno)); \\
               fflush(stderr); \\
               exit(EXIT_FAILURE); \\
             }} \\
           }}

           int main(void) {{
             int i, vi;
             {typ} *vin, *vout;
             int len = vlen({typ});
             int n = SIZE * {deg} * len;

             fprintf(stdout, "test of {op_name} over {typ}...\\n");
             CHECK(vin = ({typ}*)nsimd_aligned_alloc(n * {sizeof}));
             CHECK(vout = ({typ}*)nsimd_aligned_alloc(n * {sizeof}));

             /* Fill with random data */
             for (i = 0; i < n; i++) {{
               {rand}
             }}

             /* Load and put back data into vout */
             for (i = 0; i < n; i += {deg} * len) {{
               {load_store}
             }}

             /* Compare results */
             for (vi = 0; vi < SIZE; vi += len) {{
               for (i = vi; i < vi + len; i++) {{
                 if ({comp}) {{
                   fprintf(stdout, STATUS "... FAIL\\n");
                   fflush(stdout);
                   return -1;
                 }}
               }}
             }}

             fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
             return EXIT_SUCCESS;
           }}'''.format(includes=get_includes(lang), op_name=op.name,
                        typ=typ, rand=rand, year=date.today().year, deg=deg,
                        sizeof=common.sizeof(typ), load_store=load_store,
                        comp=comp))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for nbtrue

def gen_nbtrue(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return
    if lang == 'c_base':
        nbtrue = 'vnbtrue(vloadla(buf, {}), {})'. \
                 format(typ, typ, typ)
    elif lang == 'cxx_base':
        nbtrue = 'nsimd::nbtrue(nsimd::loadla(buf, {}()), {}())'. \
                 format(typ, typ)
    else:
        nbtrue = 'nsimd::nbtrue(nsimd::loadla<nsimd::packl<{}> >(buf))'. \
                 format(typ)
    if typ == 'f16':
        scalar0 = 'nsimd_f32_to_f16(0)'
        scalar1 = 'nsimd_f32_to_f16(1)'
    else:
        scalar0 = '({})0'.format(typ)
        scalar1 = '({})1'.format(typ)
    with common.open_utf8(filename) as out:
        out.write(
        '''{includes}

           #define CHECK(a) {{ \\
             errno = 0; \\
             if (!(a)) {{ \\
               fprintf(stderr, "ERROR: " #a ":%d: %s\\n", \\
                       __LINE__, strerror(errno)); \\
               fflush(stderr); \\
               exit(EXIT_FAILURE); \\
             }} \\
           }}

           int main(void) {{
             int i;
             {typ} *buf;
             int len = vlen({typ});

             fprintf(stdout, "test of {op_name} over {typ}...\\n");
             CHECK(buf = ({typ}*)nsimd_aligned_alloc(len * {sizeof}));

             /* Test with all elements to true */
             for (i = 0; i < len; i++) {{
               buf[i] = {scalar1};
             }}
             if ({nbtrue} != len) {{
               exit(EXIT_FAILURE);
             }}

             /* Test with all elements to false */
             for (i = 0; i < len; i++) {{
               buf[i] = {scalar0};
             }}
             if ({nbtrue} != 0) {{
               exit(EXIT_FAILURE);
             }}

             /* Test with only one element to true */
             buf[0] = {scalar1};
             if ({nbtrue} != 1) {{
               exit(EXIT_FAILURE);
             }}

             fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
             return EXIT_SUCCESS;
           }}'''.format(includes=get_includes(lang), op_name=op.name,
                        typ=typ, nbtrue=nbtrue, year=date.today().year,
                        notl='!' if op.name == 'any' else '', scalar0=scalar0,
                        scalar1=scalar1, sizeof=common.sizeof(typ)))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for reinterprets and converts

def gen_reinterpret_convert(opts, op, from_typ, to_typ, lang):
    filename = get_filename(opts, op, '{}_to_{}'.format(from_typ, to_typ),
                            lang)
    if filename == None:
        return
    logical = 'l' if op.name == 'reinterpretl' or op.name == 'to_mask' else ''
    if lang == 'c_base':
        if op.name == 'upcvt':
            comp = '''{{
                        vecx2({to_typ}) tmp =
                          vupcvt(vload{logical}a(in, {from_typ}),
                                                 {from_typ}, {to_typ});
                        vstore{logical}a(out, vdowncvt(
                            tmp.v0, tmp.v1, {to_typ}, {from_typ}),
                            {from_typ});
                      }}'''.format(op_name=op.name, from_typ=from_typ,
                                   to_typ=to_typ, logical=logical)
        elif op.name == 'to_mask':
            comp = '''vstorela(out, vto_logical(vto_mask(vloadla(in, {typ}),
                               {typ}), {typ}), {typ});'''.format(typ=from_typ)
        else:
            comp = '''vstore{logical}a(out, v{op_name}(v{op_name}(
                        vload{logical}a(in, {from_typ}), {from_typ}, {to_typ}),
                          {to_typ}, {from_typ}), {from_typ});'''. \
                          format(op_name=op.name, from_typ=from_typ,
                                 to_typ=to_typ, logical=logical)
    elif lang == 'cxx_base':
        if op.name == 'upcvt':
            comp = '''vecx2({to_typ}) tmp =
                        nsimd::upcvt(nsimd::load{logical}a(
                            in, {from_typ}()), {from_typ}(), {to_typ}());
                        nsimd::store{logical}a(out, nsimd::downcvt(
                            tmp.v0, tmp.v1, {to_typ}(), {from_typ}()),
                            {from_typ}());'''. \
                            format(op_name=op.name, from_typ=from_typ,
                            to_typ=to_typ, logical=logical)
        elif op.name == 'to_mask':
            comp = '''nsimd::storela(out, nsimd::to_logical(nsimd::to_mask(
                        nsimd::loadla(in, {typ}()), {typ}()), {typ}()),
                          {typ}());'''.format(typ=from_typ)
        else:
            comp = '''nsimd::store{logical}a(out, nsimd::{op_name}(
                        nsimd::{op_name}(nsimd::load{logical}a(
                          in, {from_typ}()), {from_typ}(), {to_typ}()),
                            {to_typ}(), {from_typ}()), {from_typ}());'''. \
                            format(op_name=op.name, from_typ=from_typ,
                                   to_typ=to_typ, logical=logical)
    else:
        if op.name == 'upcvt':
            comp = \
            '''nsimd::packx2<{to_typ}> tmp = nsimd::upcvt<
                 nsimd::pack{logical}x2<{to_typ}> >(nsimd::load{logical}a<
                   nsimd::pack{logical}<{from_typ}> >(in));
               nsimd::store{logical}a(out, nsimd::downcvt<
                 nsimd::pack{logical}<{from_typ}> >(tmp.v0, tmp.v1));'''. \
                 format(op_name=op.name, from_typ=from_typ,
                        to_typ=to_typ, logical=logical)
        elif op.name == 'to_mask':
            comp = '''nsimd::storela(out, nsimd::to_logical(nsimd::to_mask(
                        nsimd::loadla<nsimd::packl<{}> >(in))));'''. \
                        format(from_typ)
        else:
            comp = \
            '''nsimd::store{logical}a(out, nsimd::{op_name}<
                 nsimd::pack{logical}<{from_typ}> >(nsimd::{op_name}<
                   nsimd::pack{logical}<{to_typ}> >(nsimd::load{logical}a<
                     nsimd::pack{logical}<{from_typ}> >(in))));'''. \
                     format(op_name=op.name, from_typ=from_typ,
                            to_typ=to_typ, logical=logical)
    if logical == 'l':
        rand = '(rand() % 2)'
    else:
        if op.name == 'reinterpret' and to_typ == 'f16' and \
           from_typ in ['i16', 'u16']:
            rand = '(15360 /* no denormal */ | (1 << (rand() % 4)))'
        else:
            if to_typ in common.utypes or from_typ in common.utypes:
                rand = '(1 << (rand() % 4))'
            else:
                rand = '((2 * (rand() % 2) - 1) * (1 << (rand() % 4)))'
    if from_typ == 'f16':
        rand = 'nsimd_f32_to_f16((f32){});'.format(rand)
        neq_test = '(*(u16*)&in[j]) != (*(u16*)&out[j])'
    else:
        rand = '({}){}'.format(from_typ, rand)
        neq_test = 'in[j] != out[j]'
    with common.open_utf8(filename) as out:
        out.write(
        '''{includes}

           #define CHECK(a) {{ \\
             errno = 0; \\
             if (!(a)) {{ \\
               fprintf(stderr, "ERROR: " #a ":%d: %s\\n", \\
                       __LINE__, strerror(errno)); \\
               fflush(stderr); \\
               exit(EXIT_FAILURE); \\
             }} \\
           }}

           int main(void) {{
             int i, j;
             {from_typ} *in, *out;
             int len = vlen({from_typ});

             fprintf(stdout,
                     "test of {op_name} from {from_typ} to {to_typ}...\\n");
             CHECK(in = ({from_typ}*)nsimd_aligned_alloc(len * {sizeof}));
             CHECK(out = ({from_typ}*)nsimd_aligned_alloc(len * {sizeof}));

             for (i = 0; i < 100; i++) {{
               for (j = 0; j < len; j++) {{
                 in[j] = {rand};
               }}

               {comp}

               for (j = 0; j < len; j++) {{
                 if ({neq_test}) {{
                   exit(EXIT_FAILURE);
                 }}
               }}
             }}

             fprintf(stdout,
                     "test of {op_name} from {from_typ} to {to_typ}... OK\\n");
             return EXIT_SUCCESS;
           }}'''.format(includes=get_includes(lang), op_name=op.name,
                        to_typ=to_typ, from_typ=from_typ, comp=comp,
                        year=date.today().year, rand=rand, neq_test=neq_test,
                        sizeof=common.sizeof(from_typ)))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Shuffle

def gen_reverse(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return
    if lang == 'c_base':
        test_code = 'vstorea( out, vreverse( vloada( in, {typ} ), {typ} ), {typ} );'.format( typ=typ )
    elif lang == 'cxx_base':
        test_code = 'nsimd::storea( out, nsimd::reverse( nsimd::loada( in, {typ}() ), {typ}() ), {typ}() );'.format( typ=typ )
    elif lang == 'cxx_adv':
        test_code = 'nsimd::storea( out, nsimd::reverse( nsimd::loada<nsimd::pack<{typ}>>( in ) ) );'.format( typ=typ )

    if typ == 'f16':
        init = 'in[ i ] = nsimd_f32_to_f16((float)(i + 1));'
        comp = 'ok &= nsimd_f16_to_f32( out[len - 1 - i] ) == nsimd_f16_to_f32( in[i] );'
    else:
        init = 'in[ i ] = ({typ})(i + 1);'.format( typ=typ )
        comp = 'ok &= out[len - 1 - i] == in[i];'

    with common.open_utf8(filename) as out:
        out.write(
        '''{includes}

           #define CHECK(a) {{ \\
             errno = 0; \\
             if (!(a)) {{ \\
               fprintf(stderr, "ERROR: " #a ":%d: %s\\n", \\
                       __LINE__, strerror(errno)); \\
               fflush(stderr); \\
               exit(EXIT_FAILURE); \\
             }} \\
           }}

           int main(void) {{
             unsigned char i;
             int ok;
             {typ} * in;
             {typ} * out;

             int len = vlen({typ});

             fprintf(stdout, "test of {op_name} over {typ}...\\n");
             CHECK(in = ({typ}*)nsimd_aligned_alloc(len * {sizeof}));
             CHECK(out = ({typ}*)nsimd_aligned_alloc(len * {sizeof}));

             for( i = 0 ; i < len ; ++i )
             {{
                 {init}
             }}

             {test_code}

             ok = 1;

             for( i = 0 ; i < len ; ++i )
             {{
               {comp}
             }}

             /*fprintf( stdout, "%f %f %f %f\\n", in[ 0 ], out[ 0 ], in[ 1 ], out[ 1 ] );*/

             if( ok )
             {{
               fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
             }}
             else
             {{
               fprintf(stderr, "test of {op_name} over {typ}... FAIL\\n");
               exit(EXIT_FAILURE);
             }}

             nsimd_aligned_free( in );
             nsimd_aligned_free( out );

             return EXIT_SUCCESS;
           }}
        '''.format(includes=get_includes(lang), op_name=op.name,
                   typ=typ,test_code=test_code, year=date.today().year,sizeof=common.sizeof(typ),
                  init=init,comp=comp))

    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Unpack

def gen_unpack(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return
    if typ == 'f16':
      left = '(double)nsimd_f16_to_f32(mpfr_out)'
      right = '(double)nsimd_f16_to_f32(nsimd_out)'
    elif typ == 'f32':
      left = '(double)mpfr_out'
      right = '(double)nsimd_out'
    else:
      left = 'mpfr_out'
      right = 'nsimd_out'

    if lang == 'c_base':
      extra_code = relative_distance_c
      typ_nsimd = 'vec({typ})'.format(typ=typ)
      vout1_comp = '''vec({typ}) va1, va2, vc;
                      va1 = vloadu(&vin1[i], {typ});
                      va2 = vloadu(&vin2[i], {typ});
                      vc = v{op_name}(va1, va2, {typ});
                      vstoreu(&vout[i], vc, {typ});'''. \
                      format(typ=typ, op_name=op.name)
    if lang == 'cxx_base':
      extra_code = relative_distance_cpp
      typ_nsimd = 'vec({typ})'.format(typ=typ)
      vout1_comp = '''vec({typ}) va1, va2, vc;
                      va1 = nsimd::loadu(&vin1[i], {typ}());
                      va2 = nsimd::loadu(&vin2[i], {typ}());
                      vc = nsimd::{op_name}(va1, va2, {typ}());
                      nsimd::storeu(&vout[i], vc, {typ}());'''. \
                      format(typ=typ, op_name=op.name)
    if lang == 'cxx_adv':
      extra_code = relative_distance_cpp
      typ_nsimd = 'nsimd::pack<{typ}>'.format(typ=typ)
      vout1_comp = '''nsimd::pack<{typ}> va1, va2, vc;
                      va1 = nsimd::loadu<nsimd::pack<{typ}> >(&vin1[i]);
                      va2 = nsimd::loadu<nsimd::pack<{typ}> >(&vin2[i]);
                      vc = nsimd::{op_name}(va1, va2);
                      nsimd::storeu(&vout[i], vc);'''. \
                      format(typ=typ, op_name=op.name)

    op_test =  'step/(2*nb_lane)'
    if op.name in['ziphi', 'ziplo']:
        offset = 'int offset = {val};'.\
            format(val= '0' if op.name == 'ziplo' else 'vlen({typ}) / 2'.format(typ=typ))
    else:
        offset = ''

    if op.name in ['unziplo', 'unziphi']:
        if typ == 'f16':
            comp_unpack = '''\
            (nsimd_f16_to_f32(vout[i]) != nsimd_f16_to_f32(vin1[vi + 2 * j + {i}]))
            || (nsimd_f16_to_f32(vout[i + step / 2]) != nsimd_f16_to_f32(vin2[vi + 2 * j + {i}]))
            '''.format(i = '0' if op.name == 'unziplo' else '1')
        else:
             comp_unpack =  '''\
             (vout[i] != vin1[vi + 2 * j + {i}])
             || (vout[i + step / 2] != vin2[vi + 2 * j + {i}])
             '''.format(i = '0' if op.name == 'unziplo' else '1')
    else:
        if typ == 'f16':
            comp_unpack ='''(nsimd_f16_to_f32(vout[i]) != nsimd_f16_to_f32(vin1[j])) ||
            (nsimd_f16_to_f32(vout[i + 1]) != nsimd_f16_to_f32(vin2[j]))'''
        else:
            comp_unpack ='''(vout[i] != vin1[j]) ||
            (vout[i + 1] != vin2[j])'''

    nbits = {'f16': '10', 'f32': '21', 'f64': '48'}
    head = '''#define _POSIX_C_SOURCE 200112L

              {includes}
              #include <float.h>
              #include <math.h>

              #define SIZE (2048 / {sizeof})

              #define CHECK(a) {{ \\
                errno = 0; \\
                if (!(a)) {{ \\
                fprintf(stderr, "ERROR: " #a ":%d: %s\\n", \\
                        __LINE__, strerror(errno)); \\
                fflush(stderr); \\
                exit(EXIT_FAILURE); \\
                }} \\
              }}

              {extra_code}

              // {simd}
            ''' .format(year=date.today().year, typ=typ,
                          includes=get_includes(lang),
                          extra_code=extra_code,
                          comp_unpack=comp_unpack,
                          sizeof=common.sizeof(typ), simd= opts.simd)
    if typ == 'f16':
        rand = '''nsimd_f32_to_f16((f32)(2 * (rand() % 2) - 1) *
        (f32)(1 << (rand() % 4)) /
        (f32)(1 << (rand() % 4)))'''
    else:
        rand = '''({typ})(({typ})(2 * (rand() % 2) - 1) * ({typ})(1 << (rand() % 4))
        / ({typ})(1 << (rand() % 4)))'''.format(typ=typ)

    with common.open_utf8(filename) as out:
        out.write(
        '''{head}

           int main(void) {{
              int vi, i, j, step, nb_lane;
              {typ} *vin1, *vin2;
              {typ} *vout;

              CHECK(vin1 = ({typ} *)nsimd_aligned_alloc(SIZE * {sizeof}));
              CHECK(vin2 = ({typ} *)nsimd_aligned_alloc(SIZE * {sizeof}));
              CHECK(vout = ({typ} *)nsimd_aligned_alloc(SIZE * {sizeof}));

              step = vlen({typ});
              nb_lane = sizeof({typ_nsimd})/16;
              if (nb_lane == 0){{
                nb_lane = 1;
              }}

              fprintf(stdout, "test of {op_name} over {typ}...\\n");

              /* Fill input vector(s) with random */
              for (i = 0; i < SIZE; i++)
              {{
                vin1[i] = {rand};
                vin2[i] = {rand};
              }}

              /* Fill output vector with computed values */
              for (i = 0; i < SIZE; i += step)
              {{
                {vout1_comp}
              }}

              /* Compare results */
              if (step != 1) {{
                {offset}
                for (vi = 0; vi < SIZE; vi += step){{
                 j = {init_j};
                 for (i = vi; i < {cond}; {inc}) {{
                   if({comp_unpack}) {{
                     fprintf(stderr, "test of {op_name} over {typ}... FAIL\\n");
                     exit(EXIT_FAILURE);
                   }}
                   j++;
                  }}
                }}
              }}

              fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
              fflush(stdout);
              return EXIT_SUCCESS;
            }}
        '''.format(includes=get_includes(lang), op_name=op.name,
            typ=typ, year=date.today().year,sizeof=common.sizeof(typ),
            rand=rand, head=head, comp_unpack=comp_unpack,
            vout1_comp= vout1_comp, op_test=op_test, typ_nsimd=typ_nsimd,
            offset=offset,
            cond='vi + step' if op.name in['ziplo', 'ziphi'] else 'vi + step / 2',
            init_j='vi + offset' if op.name in['ziplo', 'ziphi'] else '0',
            inc='i += 2' if op.name in['ziphi', 'ziplo'] else 'i++',
            pos='0' if op.name in ['ziplo', 'unziplo', 'unziphi'] else op_test))

    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Entry point

def doit(opts):
    ulps = common.load_ulps_informations(opts)

    print ('-- Generating tests')
    for op_name, operator in operators.operators.items():
        ## Skip non-matching tests
        if opts.match and not opts.match.match(op_name):
            continue
        if op_name  in ['if_else1', 'loadu', 'loada', 'storeu', 'storea',
                        'len', 'loadlu', 'loadla', 'storelu', 'storela',
                        'set1', 'store2a', 'store2u', 'store3a', 'store3u',
                        'store4a', 'store4u', 'downcvt', 'to_logical']:
            continue
        for typ in operator.types:
            if operator.name in ['notb', 'andb', 'xorb', 'orb'] and \
               typ == 'f16':
                continue
            elif operator.name == 'nbtrue':
                gen_nbtrue(opts, operator, typ, 'c_base')
                gen_nbtrue(opts, operator, typ, 'cxx_base')
                gen_nbtrue(opts, operator, typ, 'cxx_adv')
            elif operator.name == 'addv':
                gen_addv(opts, operator, typ, 'c_base')
                gen_addv(opts, operator, typ, 'cxx_base')
                gen_addv(opts, operator, typ, 'cxx_adv')
            elif operator.name in ['all', 'any']:
                gen_all_any(opts, operator, typ, 'c_base')
                gen_all_any(opts, operator, typ, 'cxx_base')
                gen_all_any(opts, operator, typ, 'cxx_adv')
            elif operator.name in ['reinterpret', 'reinterpretl', 'cvt',
                                   'upcvt', 'to_mask']:
                for to_typ in common.get_output_types(typ, operator.output_to):
                    gen_reinterpret_convert(opts, operator, typ, to_typ,
                                            'c_base')
                    gen_reinterpret_convert(opts, operator, typ, to_typ,
                                            'cxx_base')
                    gen_reinterpret_convert(opts, operator, typ, to_typ,
                                            'cxx_adv')
            elif operator.name in ['load2a', 'load2u', 'load3a', 'load3u',
                                   'load4a', 'load4u']:
                gen_load_store(opts, operator, typ, 'c_base')
                gen_load_store(opts, operator, typ, 'cxx_base')
                gen_load_store(opts, operator, typ, 'cxx_adv')
            elif operator.name == 'reverse':
                gen_reverse(opts, operator, typ, 'c_base');
                gen_reverse(opts, operator, typ, 'cxx_base');
                gen_reverse(opts, operator, typ, 'cxx_adv');
            elif operator.name in ['ziplo', 'ziphi', 'unziplo', 'unziphi']:
                gen_unpack(opts, operator, typ, 'c_base')
                gen_unpack(opts, operator, typ, 'cxx_base')
                gen_unpack(opts, operator, typ, 'cxx_adv')
            else:
                gen_test(opts, operator, typ, 'c_base', ulps)
                gen_test(opts, operator, typ, 'cxx_base', ulps)
                gen_test(opts, operator, typ, 'cxx_adv', ulps)
