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

posix_c_source = \
'''#if !defined(_POSIX_C_SOURCE)
   #define _POSIX_C_SOURCE 200112L
   #elif _POSIX_C_SOURCE < 200112L
   #error "_POSIX_C_SOURCE defined by third-party but must be >= 200112L"
   #endif'''

# -----------------------------------------------------------------------------
# Get filename for test

def get_filename(opts, op, typ, lang, custom_name=''):
    pp_lang = {'c_base': 'C (base API)',
               'cxx_base' : 'C++ (base API)',
               'cxx_adv' : 'C++ (advanced API)'}
    tests_dir = os.path.join(opts.tests_dir, lang)
    common.mkdir_p(tests_dir)
    if not custom_name:
        filename = os.path.join(tests_dir, '{}.{}.{}'.format(op.name, typ,
                     'c' if lang == 'c_base' else 'cpp'))
    else:
        filename = os.path.join(tests_dir, '{}_{}.{}.{}'.format(op.name,
                     custom_name, typ, 'c' if lang == 'c_base' else 'cpp'))
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
  {typ} *vout_ref, *vout_nsimd;
  {vin_defi}

  CHECK(vout_ref = ({typ}*)nsimd_aligned_alloc(SIZE * {sizeof}));
  CHECK(vout_nsimd = ({typ}*)nsimd_aligned_alloc(SIZE * {sizeof}));

  step = vlen({typ});

  fprintf(stdout, STATUS "...\\n");
  fflush(stdout);

  /* Fill input vector(s) with random values */
  for (i = 0; i < SIZE; i++) {{
    {vin_rand}
  }}



  #ifdef NSIMD_DNZ_FLUSH_TO_ZERO
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wconversion"
  #pragma GCC diagnostic ignored "-Wdouble-promotion"
  for (i = 0; i < SIZE; i++) {{
    {denormalize_inputs}
  }}
  #pragma GCC diagnostic pop
  #endif

  /* Fill output vector 0 with reference values */
  for (i = 0; i < SIZE; i += {cpu_step}) {{
    /* This is a call directly to the cpu API of nsimd
       to ensure that we call the scalar version of the
       function */
    {vout_ref_comp}
  }}

  /* Fill output vector 1 with computed values */
  for (i = 0; i < SIZE; i += step) {{
    {vout_nsimd_comp}
  }}

  /* Compare results */
  for (vi = 0; vi < SIZE; vi += step) {{
    for (i = vi; i < vi + step; i++) {{
      if (comp_function(vout_ref[i], vout_nsimd[i])) {{
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

    nargs = range(1, len(op.params))

    if typ in common.ftypes:
        code = ['''if ({classify}({f32_conv}(vin{i}[i])) == FP_SUBNORMAL) {{
                     vin{i}[i] = {f16_conv}(0.f);
                }}'''. \
                format(i=i, f16_conv='nsimd_f32_to_f16' if typ=='f16' else '',
                f32_conv='nsimd_f16_to_f32' if typ=='f16' else '',
                classify='fpclassify' if lang=='c_base' \
                                      else 'std::fpclassify') for i in nargs]

        denormalize_inputs = '\n'.join(code)
    else:
        denormalize_inputs = ''

    # Depending on function parameters, generate specific input, ...
    if all(e == 'v' for e in op.params) or all(e == 'l' for e in op.params):
        logical = 'l' if op.params[0] == 'l' else ''

        # Make vin_defi
        code = ['{} *vin{};'.format(typ, i) for i in nargs]
        code += ['CHECK(vin{} = ({}*)nsimd_aligned_alloc(SIZE * {}));'.
                 format(i, typ, common.sizeof(typ)) for i in nargs]
        vin_defi = '\n'.join(code)
        if op.name in ['rec11', 'rec8', 'rsqrt11', 'rsqrt8']:
            if typ == 'f16':
                code = ['vin{}[i] = nsimd_f32_to_f16((float)rand() / ' \
                        '(float)INT_MAX);'.format(i) for i in nargs]
            else:
                code = ['vin{}[i] = ({})((float)rand() / (float)INT_MAX);'. \
                        format(i, typ) for i in nargs]
        else:
            code = ['vin{}[i] = rand{}();'.format(i, i) for i in nargs]
        vin_rand = '\n'.join(code)

        # lgamma doesn't work for negative input or for too big inputs.
        if op.name == 'lgamma' and typ == 'f64':
            vin_rand = 'vin1[i] = rand() % 64;'

        # Make vout_ref_comp
        # We use MPFR on Linux to compare numerical results, but it is only on
        # Linux as MPFR does not play well on Windows. On Windows we compare
        # against the cpu implementation. When using MPFR, we set one element
        # at a time => cpu_step = '1'
        if op.tests_mpfr and sys.platform.startswith('linux'):
            cpu_step = '1'
            variables = ', '.join(['a{}'.format(i) for i in nargs])
            mpfr_inits = '\n'.join(['mpfr_init2(a{}, 64);'.format(i)
                                    for i in nargs])
            if typ == 'f16':
                mpfr_set = '''mpfr_set_flt(a{i}, nsimd_u16_to_f32(
                                ((u16 *)vin{i})[i]), MPFR_RNDN);'''
                vout_ref_set = '''((u16 *)vout_ref)[i] = nsimd_f32_to_u16(
                                 mpfr_get_flt(c, MPFR_RNDN));'''
            elif typ == 'f32':
                mpfr_set = 'mpfr_set_flt(a{i}, vin{i}[i], MPFR_RNDN);'
                vout_ref_set = 'vout_ref[i] = mpfr_get_flt(c, MPFR_RNDN);'
            else:
                mpfr_set = 'mpfr_set_d(a{i}, vin{i}[i], MPFR_RNDN);'
                vout_ref_set = 'vout_ref[i] = ({})mpfr_get_d(c, MPFR_RNDN);'. \
                               format(typ)
            mpfr_sets = '\n'.join([mpfr_set.format(i=j) for j in nargs])
            mpfr_clears = '\n'.join(['mpfr_clear(a{});'.format(i)
                                     for i in nargs])
            vout_ref_comp = \
            '''mpfr_t c, {variables};
               mpfr_init2(c, 64);
               {mpfr_inits}
               {mpfr_sets}
               {mpfr_op_name}(c, {variables}, MPFR_RNDN);
               {vout_ref_set}
               mpfr_clear(c);
               {mpfr_clears}'''. \
               format(variables=variables, mpfr_sets=mpfr_sets,
                      mpfr_clears=mpfr_clears, vout_ref_set=vout_ref_set,
                      mpfr_op_name=op.tests_mpfr_name(), mpfr_inits=mpfr_inits)
        else:
            args = ', '.join(['va{}'.format(i) for i in nargs])
            code = ['nsimd_cpu_v{}{} {}, vc;'.format(logical, typ, args)]
            code += ['va{} = nsimd_load{}u_cpu_{}(&vin{}[i]);'.
                     format(i, logical, typ, i) for i in nargs]
            code += ['vc = nsimd_{}_cpu_{}({});'.format(op.name, typ, args)]
            code += ['nsimd_store{}u_cpu_{}(&vout_ref[i], vc);'. \
                     format(logical, typ)]
            vout_ref_comp = '\n'.join(code)

            if op.name[-2:] == '11' or op.name[-1:] == '8':
                vout_ref_comp += \
                '''/* Intel 11 bit precision intrinsics force denormalized output to 0. */
                   #ifdef NSIMD_X86
                   #pragma GCC diagnostic push
                   #pragma GCC diagnostic ignored "-Wconversion"
                   #pragma GCC diagnostic ignored "-Wdouble-promotion"
                   for (vi = i; vi < i+nsimd_len_cpu_{typ}(); ++vi) {{
                       if ({classify}({f32_conv}(vout_ref[vi])) ==
                           FP_SUBNORMAL) {{
                           vout_ref[vi] = {f16_conv}(0.f);
                       }}
                   }}
                   #pragma GCC diagnostic pop
                   #endif
                   '''.format(
                   typ=typ, f16_conv='nsimd_f32_to_f16' if typ=='f16' else '',
                   f32_conv='nsimd_f16_to_f32' if typ=='f16' else '',
                   classify='fpclassify' \
                   if lang=='c_base' else 'std::fpclassify')

        # Make vout_nsimd_comp
        args = ', '.join(['va{}'.format(i) for i in nargs])
        if lang == 'c_base':
            code = ['vec{}({}) {}, vc;'.format(logical, typ, args)]
            code += ['va{} = vload{}u(&vin{}[i], {});'.
                     format(i, logical, i, typ) for i in nargs]
            code += ['vc = v{}({}, {});'.format(op.name, args, typ)]
            code += ['vstore{}u(&vout_nsimd[i], vc, {});'.format(logical, typ)]
            vout_nsimd_comp = '\n'.join(code)
        if lang == 'cxx_base':
            code = ['vec{}({}) {}, vc;'.format(logical, typ, args)]
            code += ['va{} = nsimd::load{}u(&vin{}[i], {}());'.
                     format(i, logical, i, typ) for i in nargs]
            code += ['vc = nsimd::{}({}, {}());'.format(op.name, args, typ)]
            code += ['nsimd::store{}u(&vout_nsimd[i], vc, {}());'. \
                     format(logical, typ)]
            vout_nsimd_comp = '\n'.join(code)
        if lang == 'cxx_adv':
            code = ['nsimd::pack{}<{}> {}, vc;'.format(logical, typ, args)]
            code += ['''va{i} = nsimd::load{logical}u<
                                  nsimd::pack{logical}<{typ}> >(
                                      &vin{i}[i]);'''.
                     format(i=i, logical=logical, typ=typ) for i in nargs]
            if op.cxx_operator:
                if len(op.params[1:]) == 1:
                    code += ['vc = {}va1;'.
                             format(op.cxx_operator)]
                if len(op.params[1:]) == 2:
                    code += ['vc = va1 {} va2;'.
                             format(op.cxx_operator)]
            else:
                code += ['vc = nsimd::{}({});'.format(op.name, args)]
            code += ['nsimd::store{}u(&vout_nsimd[i], vc);'. \
                     format(logical, typ)]
            vout_nsimd_comp = '\n'.join(code)
    elif op.params == ['l', 'v', 'v']:
        vin_defi = \
            '''{typ} *vin1, *vin2;
           CHECK(vin1 = ({typ}*)nsimd_aligned_alloc(SIZE * {sizeof}));
           CHECK(vin2 = ({typ}*)nsimd_aligned_alloc(SIZE * {sizeof}));'''. \
           format(typ=typ, sizeof=common.sizeof(typ))
        code = ['vin{}[i] = rand{}();'.format(i,i) for i in nargs]
        vin_rand = '\n'.join(code)

        vout_ref_comp = '''nsimd_cpu_v{typ} va1, va2;
                        nsimd_cpu_vl{typ} vc;
                        va1 = nsimd_loadu_cpu_{typ}(&vin1[i]);
                        va2 = nsimd_loadu_cpu_{typ}(&vin2[i]);
                        vc = nsimd_{op_name}_cpu_{typ}(va1, va2);
                        nsimd_storelu_cpu_{typ}(&vout_ref[i], vc);'''. \
                        format(typ=typ, op_name=op.name)

        if lang == 'c_base':
            vout_nsimd_comp = '''vec({typ}) va1, va2;
                            vecl({typ}) vc;
                            va1 = vloadu(&vin1[i], {typ});
                            va2 = vloadu(&vin2[i], {typ});
                            vc = v{op_name}(va1, va2, {typ});
                            vstorelu(&vout_nsimd[i], vc, {typ});'''. \
                            format(typ=typ, op_name=op.name)
        if lang == 'cxx_base':
            vout_nsimd_comp = '''vec({typ}) va1, va2;
                            vecl({typ}) vc;
                            va1 = nsimd::loadu(&vin1[i], {typ}());
                            va2 = nsimd::loadu(&vin2[i], {typ}());
                            vc = nsimd::{op_name}(va1, va2, {typ}());
                            nsimd::storelu(&vout_nsimd[i], vc, {typ}());'''. \
                            format(typ=typ, op_name=op.name)
        if lang == 'cxx_adv':
            if op.cxx_operator:
                do_computation = 'vc = va1 {} va2;'. \
                                 format(op.cxx_operator)
            else:
                do_computation = 'vc = nsimd::{}(va1, va2, {}());'. \
                                 format(op.name, typ)
            vout_nsimd_comp = '''nsimd::pack<{typ}> va1, va2;
                            nsimd::packl<{typ}> vc;
                            va1 = nsimd::loadu<nsimd::pack<{typ}> >(&vin1[i]);
                            va2 = nsimd::loadu<nsimd::pack<{typ}> >(&vin2[i]);
                            {do_computation}
                            nsimd::storelu(&vout_nsimd[i], vc);'''. \
                            format(typ=typ, op_name=op.name,
                                   do_computation=do_computation)

    elif op.params == ['v', 'v', 'p']:
        vin_defi = \
        '''{typ} *vin1;
           CHECK(vin1 = ({typ}*)nsimd_aligned_alloc(SIZE * {sizeof}));'''. \
           format(typ=typ, sizeof=common.sizeof(typ))
        vin_rand = 'vin1[i] = rand1();'.format(typ=typ)
        vout_ref_comp = \
        '''nsimd_cpu_v{typ} va1, vc;
           va1 = nsimd_loadu_cpu_{typ}(&vin1[i]);
           vc = nsimd_{op_name}_cpu_{typ}(va1, (i / step) % {typnbytes});
           nsimd_storeu_cpu_{typ}(&vout_ref[i], vc);'''. \
           format(typ=typ, op_name=op.name, typnbytes=typ[1:])
        if lang == 'c_base':
            vout_nsimd_comp = \
            '''vec({typ}) va1, vc;
               va1 = vloadu(&vin1[i], {typ});
               vc = v{op_name}(va1, (i / step) % {typnbytes}, {typ});
               vstoreu(&vout_nsimd[i], vc, {typ});'''. \
               format(typ=typ, op_name=op.name, typnbytes=typ[1:])
        if lang == 'cxx_base':
            vout_nsimd_comp = \
            '''vec({typ}) va1, vc;
               va1 = nsimd::loadu(&vin1[i], {typ}());
               vc = nsimd::{op_name}(va1, (i / step) % {typnbytes}, {typ}());
               nsimd::storeu(&vout_nsimd[i], vc, {typ}());'''. \
                       format(typ=typ, op_name=op.name, typnbytes=typ[1:])
        if lang == 'cxx_adv':
            if op.cxx_operator:
                do_computation = 'vc = va1 {} ((i / step) % {typnbytes});'. \
                        format(op.cxx_operator, typnbytes=typ[1:])
            else:
                do_computation = \
                'vc = nsimd::{}(va1, (i / step) % {typnbytes});'. \
                format(op.name, typnbytes=typ[1:])
            vout_nsimd_comp = \
            '''nsimd::pack<{typ}> va1, vc;
               va1 = nsimd::loadu<nsimd::pack<{typ}> >(&vin1[i]);
               {do_computation}
               nsimd::storeu(&vout_nsimd[i], vc);'''. \
               format(typ=typ, do_computation=do_computation)
    else:
        raise ValueError('No test available for operator "{}" on type "{}"'.
                         format(op.name, typ))
    return { 'vin_defi': vin_defi, 'vin_rand': vin_rand, 'cpu_step': cpu_step,
             'vout_ref_comp': vout_ref_comp, 'vout_nsimd_comp': vout_nsimd_comp,
             'denormalize_inputs': denormalize_inputs }

# -----------------------------------------------------------------------------
# Generate test in C, C++ (base API) and C++ (advanced API) for almost all
# tests


def gen_test(opts, op, typ, lang, ulps):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return

    content = get_content(op, typ, lang)

    extra_code = op.domain.gen_rand(typ)

    if op.name in ['notb', 'andb', 'orb', 'xorb', 'andnotb']:
        comp = 'return nsimd_scalar_reinterpret_{uT}_{typ}(mpfr_out) != ' \
                      'nsimd_scalar_reinterpret_{uT}_{typ}(nsimd_out)'. \
               format(typ=typ, uT=common.bitfield_type[typ])
    elif op.name in ['max', 'min'] and typ in common.ftypes:
        if typ == 'f16':
            left = 'nsimd_f16_to_f32(mpfr_out)'
            right = 'nsimd_f16_to_f32(nsimd_out)'
        else:
            left = 'mpfr_out'
            right = 'nsimd_out'

        comp = '''#pragma GCC diagnostic push
                  #pragma GCC diagnostic ignored "-Wconversion"
                  #pragma GCC diagnostic ignored "-Wdouble-promotion"

                  // None of the architecture correctly manage NaN with the
                  // function min and max. According to IEEE754, min(a, NaN)
                  // should return a but every architecture returns NaN.
                  if({isnan}({right})) {{
                    return 0;
                  }}

                  // PPC doesn't correctly manage +Inf and -Inf in relation
                  // with NaN either (min(NaN, -Inf) returns -Inf).
                  #ifdef NSIMD_POWERPC
                  if({isinf}({right})) {{
                    return 0;
                  }}
                  #endif

                  return {left} != {right};
                  #pragma GCC diagnostic pop
                  '''.format(left=left, right=right,
                          uT=common.bitfield_type[typ],
                          isnan='isnan' if lang=='c_base' else 'std::isnan',
                          isinf='isinf' if lang=='c_base' else 'std::isinf')
    else:
        if typ == 'f16':
            left = 'nsimd_f16_to_f32(mpfr_out)'
            right = 'nsimd_f16_to_f32(nsimd_out)'
        elif typ == 'f32':
            left = 'mpfr_out'
            right = 'nsimd_out'
        else:
            left = 'mpfr_out'
            right = 'nsimd_out'
        relative_distance = relative_distance_c if lang == 'c_base' \
                            else relative_distance_cpp
        if op.tests_ulps and typ in common.ftypes:
            comp = 'return relative_distance((double){}, ' \
                   '(double){}) > get_2th_power(-{nbits})'. \
                   format(left, right, nbits=op.tests_ulps[typ])
            extra_code += relative_distance
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
                    # Ignore error with NaN output, we know we will encounter
                    # some
                    comp += 'if ({isnan}({left})) return 0;\n'
                else:
                    # Return false if one is NaN and not the other
                    comp += 'if ({isnan}({left}) ^ isnan({rigth})) return 1;\n'

                if inf_error:
                    # Ignore error with infinite output, we know we will
                    # encounter some
                    comp += 'if ({isinf}({left})) return 0;\n'
                else:
                    # One is infinite and not the other
                    comp += \
                    'if ({isinf}({left}) ^ {isinf}({rigth})) return 1;\n'
                    # Wrong sign for infinite
                    comp += 'if ({isinf}({left}) && {isinf}({rigth}) ' \
                                   '&& ({right}*{left} < 0)) ' \
                                       'return 1;\n'

                comp += '''
                if ({isnormal}({left})) {{
                    return relative_distance((double){left}, (double){right})
                             > get_2th_power(-({nbits}));
                }} else {{
                    return relative_distance((double){left}, (double){right})
                             > get_2th_power(-({nbits_dnz}));
                }}
                #pragma GCC diagnostic pop
                '''

                if lang == 'c_base':
                    comp = comp.format(left=left,
                                       right=right,
                                       nbits=nbits,
                                       nbits_dnz=nbits_dnz,
                                       isnormal='isnormal',
                                       isinf='isinf',
                                       isnan='isnan')
                else:
                    comp = comp.format(left=left,
                                       right=right,
                                       nbits=nbits,
                                       nbits_dnz=nbits_dnz,
                                       isnormal='std::isnormal',
                                       isinf='std::isinf',
                                       isnan='std::isnan')

            else:
                nbits = {'f16': '10', 'f32': 21, 'f64': '48'}
                comp = 'return relative_distance((double){}, (double){}) ' \
                       '> get_2th_power(-{nbits})'. \
                       format(left, right, nbits=nbits[typ])

            extra_code += relative_distance
        else:
            if typ in common.ftypes:
                comp = \
                '''#pragma GCC diagnostic push
                   #pragma GCC diagnostic ignored "-Wconversion"
                   #pragma GCC diagnostic ignored "-Wdouble-promotion"
                   return {left} != {right}
                        && (!{isnan}({left}) || !{isnan}({right}));
                   #pragma GCC diagnostic pop
                 '''.format(left=left, right=right,
                            isnan='isnan' if lang=='c_base' else 'std::isnan')
            else:
                comp = 'return {} != {};'.format(left, right)

            extra_code += ''

    includes = get_includes(lang)
    if op.src or op.tests_ulps or op.tests_mpfr:
        if lang == 'c_base':
            includes = '''{}

                          #include <math.h>
                          #include <float.h>
                          {}'''.format(posix_c_source, includes)
        else:
            includes = '''{}

                          #include <cmath>
                          #include <cfloat>
                          {}'''.format(posix_c_source, includes)
        if op.tests_mpfr and sys.platform.startswith('linux'):
            includes = includes + '''
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wsign-conversion"
            #include <mpfr.h>
            #pragma GCC diagnostic pop
            '''

    with common.open_utf8(opts, filename) as out:
        out.write(template.format( \
            includes=includes, sizeof=common.sizeof(typ), typ=typ,
            op_name=op.name, year=date.today().year, comp=comp,
            extra_code=extra_code, **content))
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
        op_test = 'nsimd::{}(nsimd::loada(buf, {}()), {}())'.format(
            op.name, typ, typ)
        extra_code = relative_distance_cpp
    else:
        op_test = 'nsimd::{}(nsimd::loada<nsimd::pack<{}> >(buf))'.format(
            op.name, typ)
        extra_code = relative_distance_cpp

    nbits = {'f16': '10', 'f32': '21', 'f64': '48'}
    head = '''{posix_c_source}
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
                                      posix_c_source=posix_c_source,
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
    elif typ[0] == 'f':
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
    else:
        init = '''{typ} ref = ({typ}) 0;
                  {typ} res = ({typ}) 0;'''.format(typ=typ)
        rand = '({})(rand() % 4)'.format(typ)
        init_statement = 'buf[i] = {rand};' .format(rand=rand)
        ref_statement = 'ref += buf[i];'
        test = '''if(ref != res) {{
                      return EXIT_FAILURE;
                  }}'''

    with common.open_utf8(opts, filename) as out:
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
                       sizeof=common.sizeof(typ), init_statement=init_statement,
                       ref_statement=ref_statement, op_test=op_test, test=test)
        )
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# General tests helpers for adds/subs

def aligned_alloc_error():
      return '''
      #define CHECK(a) \\
      {{ \\
        errno = 0; \\
        if (!(a)) \\
        {{ \\
          fprintf(stderr, \"ERROR: \" #a \":%d: %s\\n\", \\
                __LINE__, strerror(errno)); \\
          fflush(stderr); \\
          exit(EXIT_FAILURE); \\
        }} \\
      }}
      '''

def equal(typ):
      return '''
      int equal({typ} expected_result, {typ} computed_result)
      {{
        return expected_result == computed_result;
      }}
      '''.format(typ=typ)

def adds_subs_check_case():
      return '''
      #define CHECK_CASE(test_output, which_test) \\
      {{ \\
        if(0 == (test_output)) \\
        {{ \\
          fprintf(stdout, STATUS \" ... \" which_test \" check FAIL\\n\"); \\
          fflush(stdout); \\
          return -1; \\
        }} \\
      }}
      '''

def random_sign_flip():
      return '''
      int random_sign_flip(void)
      {{
          return 2 * (rand() % 2) - 1;
      }}
      '''

def zero_out_arrays(typ):
      return '''
      void zero_out_arrays({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        int ii = 0;
        for(ii = 0; ii < SIZE; ++ii)
        {{
           vin1[ii] = ({typ})0;
           vin2[ii] = ({typ})0;
           vout_expected[ii] = ({typ})0;
           vout_computed[ii] = ({typ})0;
        }}
      }}
      '''.format(typ=typ)

def compute_op_given_language(typ, op, language):
      if 'c_base' == language:
            return '''
              vec({typ}) va1, va2, vc;
              va1 = vloadu(&vin1[outer], {typ});
              va2 = vloadu(&vin2[outer], {typ});
              vc = v{op}(va1, va2, {typ});
              vstoreu(&vout_computed[outer], vc, {typ});
            '''.format(typ=typ, op=op)
      elif 'cxx_base' == language:
            return '''
              vec({typ}) va1, va2, vc;
              va1 = nsimd::loadu(&vin1[outer], {typ}());
              va2 = nsimd::loadu(&vin2[outer], {typ}());
              vc = nsimd::{op}(va1, va2, {typ}());
              nsimd::storeu(&vout_computed[outer], vc, {typ}());
            '''.format(typ=typ, op=op)
      else:
            return '''
                nsimd::pack<{typ}> va1, va2, vc;
                va1 = nsimd::loadu<nsimd::pack<{typ}> >(&vin1[outer]);
                va2 = nsimd::loadu<nsimd::pack<{typ}> >(&vin2[outer]);
                vc = nsimd::{op}(va1, va2);
                nsimd::storeu(&vout_computed[outer], vc);
            '''.format(typ=typ, op=op)

def compare_expected_vs_computed(typ, op, language):
      values_computation = compute_op_given_language(typ, op, language)
      return '''
      int compare_expected_vs_computed(const {typ}* vin1, const {typ}* vin2, const {typ}* vout_expected, {typ} vout_computed[])
      {{
          const int step = vlen({typ});
          int outer = 0;
          int inner = 0;

          for (outer = 0; outer < SIZE; outer += step) {{
          /* Fill vout_computed with computed values */
          {values_computation}
          /* Compare results */
          for (inner = outer; inner < outer + step; ++inner) {{
              if (! equal(vout_expected[inner], vout_computed[inner])) {{
                return 0;
              }}
            }}
          }}

          return 1;
      }}
      '''.format(typ=typ, values_computation=values_computation)

def test_signed_neither_overflow_nor_underflow(typ, min_, max_, operator, check):
      return '''
      int test_neither_overflow_nor_underflow({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        int ii = 0;
        while(ii < SIZE)
        {{
          {typ} a = ({typ})((random_sign_flip() * rand()) % {max_} % {min_});
          {typ} b = ({typ})((random_sign_flip() * rand()) % {max_} % {min_});
          if({check}(a, b))
          {{
            vin1[ii] = a;
            vin2[ii] = b;
            vout_expected[ii] = ({typ})(a {operator} b);
            ++ ii;
          }}
        }}
        assert(ii == SIZE);
        /*
        Test:
        if (neither overflow nor underflow) {{ vout_expected[ii] == a {operator} b; }}
        */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
      }}
      '''.format(typ=typ, min_=min_, max_=max_, operator=operator, check=check)

def test_signed_all_cases(typ, min_, max_, oper, oper_is_overflow, oper_is_underflow):
      return '''
      int test_all_cases({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        int ii = 0;
        for(ii = 0; ii < SIZE; ++ii)
        {{
          vin1[ii] = ({typ})((random_sign_flip() * rand()) % {max_} % {min_});
          vin2[ii] = ({typ})((random_sign_flip() * rand()) % {max_} % {min_});
          if({oper_is_overflow}(vin1[ii], vin2[ii]))
          {{
            vout_expected[ii] = {max_};
          }}
          else if({oper_is_underflow}(vin1[ii], vin2[ii]))
          {{
            vout_expected[ii] = {min_};
          }}
          else
          {{
            vout_expected[ii] = ({typ})(vin1[ii] {oper} vin2[ii]);
          }}
        }}
        /* Test all cases */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
      }}
      ''' .format(typ=typ, min_=min_, max_=max_,
                  oper=oper, oper_is_overflow=oper_is_overflow,
                  oper_is_underflow=oper_is_underflow)

# -----------------------------------------------------------------------------
# Tests helpers for adds - is overflow/underflow/neither overflow nor underflow

def adds_is_overflow(typ, max_):
      return '''
      int adds_is_overflow(const {typ} a, const {typ} b)
      {{
        return (a > 0) && (b > {max_} - a);
      }}
      '''.format(typ=typ, max_=max_)

def adds_signed_is_underflow(typ, min_):
      return '''
      int adds_signed_is_underflow(const {typ} a, const {typ} b)
      {{
        return (a < 0) && (b < {min_} - a);
      }}
      '''.format(typ=typ, min_=min_)

def adds_signed_is_neither_overflow_nor_underflow(typ):
      return '''
      int adds_signed_is_neither_overflow_nor_underflow(const {typ} a, const {typ} b)
      {{
        return ! adds_is_overflow(a, b) && ! adds_signed_is_underflow(a, b);
      }}
      '''.format(typ=typ)

# -----------------------------------------------------------------------------
# Tests helpers for adds with integer types

# test integer overflow
def test_adds_overflow(typ, max_):
      rand_ = '({typ})rand()'.format(typ=typ) if typ in common.utypes else 'rand()'
      return '''
      int test_overflow({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        /* if ((vin1[ii] > 0) && (vin2[ii] > {max_} - vin1[ii])) {{ overflow }} */
        int ii = 0;

        /* vin1[ii] > 0 */
        for(ii = 0; ii < SIZE; ++ii)
        {{
          {typ} rand_val = ({typ})({rand_} % {max_});
          vin1[ii] = (rand_val == 0 ? 1 : rand_val);
        }}

        /*
        vin2[ii] > {max_} - vin1[ii]
        vin2[ii] = {max_} - vin1[ii] + rand_val
        s.t.: 0 < rand_val <= vin1[ii]
        */
        for(ii = 0; ii < SIZE; ++ii)
        {{
            {typ} rand_val = ({typ})({rand_} % (vin1[ii] + 1));
            rand_val = (rand_val == 0 ? 1 : rand_val);
            vin2[ii] = ({typ})({max_} - vin1[ii] + rand_val);
            vout_expected[ii] = {max_};
        }}

        /*
        Test:
        if ((vin1[ii] > 0) && (vin2[ii] > {max_} - vin1[ii])) {{ vout_expected[ii] == {max_}; }}
        */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
     }}
      '''.format(typ=typ, max_=max_, rand_=rand_)

# -----------------------------------------------------------------------------
# Tests helpers for adds with signed integer types

# test signed underflow
def test_adds_signed_underflow(typ, min_):
      return '''
      int test_underflow({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        /* if ((vin1[ii] < 0) && (vin2[ii] < {min_} - vin1[ii])) {{ underflow }} */
        int ii = 0;

        /* vin1[ii] < 0 */
        for(ii = 0; ii < SIZE; ++ii)
        {{
            {typ} rand_val = ({typ})((- rand()) % {min_});
            vin1[ii] = (rand_val == 0 ? - 1 : rand_val);
        }}

        /*
        vin1[ii] < 0
        vin2[ii] < {min_} - vin1[ii]
        vin2[ii] = {min_} - vin1[ii] - rand_val
        s.t.: 0 < rand_val < - vin1[ii]
        */

        for(ii = 0; ii < SIZE; ++ii)
        {{
            {typ} rand_val = ({typ})((rand()) % (- vin1[ii]));
            rand_val = (rand_val == 0 ? 1 : rand_val);
            vin2[ii] = ({typ})({min_} - vin1[ii] - rand_val);
            vout_expected[ii] = {min_};
        }}

        /*
        Test:
        if ((vin1[ii] < 0) && (vin2[ii] < {min_} - vin1[ii])) {{ vout_expected[ii] == {min_}; }}
        */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
      }}
      '''.format(typ=typ, min_=min_)

# test signed neither overflow nor underflow
def test_adds_signed_neither_overflow_nor_underflow(typ, min_, max_):
      return \
        test_signed_neither_overflow_nor_underflow(typ, min_, max_,
         '+', 'adds_signed_is_neither_overflow_nor_underflow')

# test signed all cases
def test_adds_signed_all_cases(typ, min_, max_):
      return test_signed_all_cases(typ, min_, max_, '+', 'adds_is_overflow', 'adds_signed_is_underflow')

# all signed tests
def tests_adds_signed():
      return'''
      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_overflow(vin1, vin2, vout_expected,
                 vout_computed), "overflow");

      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_underflow(vin1, vin2, vout_expected,
                 vout_computed), "underflow");

      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_neither_overflow_nor_underflow(vin1, vin2, vout_expected, vout_computed),
      "neither underflow nor overflow");

      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_all_cases(vin1, vin2, vout_expected,
                 vout_computed), "all cases");
      '''

# -----------------------------------------------------------------------------
# Tests helper for adds with unsigned types

# test signed neither overflow nor underflow
def test_adds_unsigned_no_overflow(typ, max_):
      return '''
      int test_no_overflow({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        int ii = 0;
        while(ii < SIZE)
        {{
          {typ} a = ({typ})(({typ})rand() % {max_});
          {typ} b = ({typ})(({typ})rand() % {max_});
          if(! adds_is_overflow(a, b))
          {{
            vin1[ii] = a;
            vin2[ii] = b;
            vout_expected[ii] = ({typ})(a + b);
            ++ ii;
          }}
        }}
        assert(ii == SIZE);
        /*
        Test:
        if (not adds is overflow) {{ vout_expected[ii] == a + b; }}
        */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
      }}
      '''.format(typ=typ, max_=max_)

# test unsigned all cases
def test_adds_unsigned_all_cases(typ, max_):
      return '''
      int test_all_cases({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        int ii = 0;
        for(ii = 0; ii < SIZE; ++ii)
        {{
          vin1[ii] = ({typ})(({typ})rand() % {max_});
          vin2[ii] = ({typ})(({typ})rand() % {max_});
          if(adds_is_overflow(vin1[ii], vin2[ii]))
          {{
            vout_expected[ii] = {max_};
          }}
          else {{ vout_expected[ii] = ({typ})(vin1[ii] + vin2[ii]); }}
        }}
        /* Test all cases: */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
      }}
      '''.format(typ=typ, max_=max_)

# all unsigned tests
def tests_adds_unsigned():
      return'''
      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_overflow(vin1, vin2, vout_expected,
                 vout_computed), "overflow");

      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_no_overflow(vin1, vin2, vout_expected,
                 vout_computed), "no overflow");

      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_all_cases(vin1, vin2, vout_expected,
                 vout_computed), "all cases");
      '''

# ------------------------------------------------------------------------------
# Get adds tests given type

def get_adds_tests_cases_for_signed_types(typ, min_, max_):
      helpers = '''
            {test_adds_overflow}

            {test_adds_signed_underflow}

            {adds_is_overflow}

            {adds_signed_is_underflow}

            {adds_signed_is_neither_overflow_nor_underflow}

            {test_adds_signed_neither_overflow_nor_underflow}

            {test_adds_signed_all_cases}
          ''' .format(test_adds_overflow=test_adds_overflow(typ, max_),
                      test_adds_signed_underflow=test_adds_signed_underflow(
                          typ, min_),
                      adds_is_overflow=adds_is_overflow(typ, max_),
                      adds_signed_is_underflow=adds_signed_is_underflow(
                          typ, min_),
                      adds_signed_is_neither_overflow_nor_underflow=adds_signed_is_neither_overflow_nor_underflow(
                          typ),
                      test_adds_signed_neither_overflow_nor_underflow=test_adds_signed_neither_overflow_nor_underflow(
                          typ, min_=min_, max_=max_),
                      test_adds_signed_all_cases=test_adds_signed_all_cases(
                          typ, min_=min_, max_=max_)
                      )
      return {'helpers': helpers, 'tests': tests_adds_signed()}

def get_adds_tests_cases_for_unsigned_types(typ, max_):
      helpers = '''
          {test_adds_overflow}

          {adds_is_overflow}

          {test_adds_unsigned_no_overflow}

          {test_adds_unsigned_all_cases}
          ''' .format(test_adds_overflow=test_adds_overflow(typ, max_),
                      adds_is_overflow=adds_is_overflow(typ, max_),
                      test_adds_unsigned_no_overflow=test_adds_unsigned_no_overflow(
                          typ, max_),
                      test_adds_unsigned_all_cases=test_adds_unsigned_all_cases(typ, max_)
                      )
      return {'helpers': helpers, 'tests': tests_adds_unsigned()}

def get_adds_tests_cases_given_type(typ):
      if typ in common.iutypes:
            type_limits = common.limits[typ]
            min_ = type_limits['min']
            max_ = type_limits['max']

            if typ in common.itypes:
                  return get_adds_tests_cases_for_signed_types(typ=typ, min_=min_, max_=max_)

            if typ in common.utypes:
                  return get_adds_tests_cases_for_unsigned_types(typ=typ, max_=max_)
      else:
            msg = '{typ} not implemented'.format(typ=typ)
            raise TypeError(msg)

# -----------------------------------------------------------------------------
# gen_adds

def gen_adds(opts, op, typ, lang, ulps):

    # Do not test for floats since adds(floats) == add(floats)
    if typ in common.ftypes:
        return

    filename = get_filename(opts, op, typ, lang)

    if filename == None:
        return

    sizeof = common.sizeof(typ)

    head = '''
              {includes}
              #include <assert.h>

              #define SIZE (2048 / {sizeof})

              #define STATUS "test of {op_name} over {typ}"

              {aligned_alloc_error}

              {adds_subs_check_case}
            ''' .format(includes=get_includes(lang),
                        op_name=op.name,
                        typ=typ,
                        sizeof=sizeof,
                        aligned_alloc_error=aligned_alloc_error(),
                        adds_subs_check_case=adds_subs_check_case())

    with common.open_utf8(opts, filename) as out:
        out.write(
            ''' \
            {head}
            /* ------------------------------------------------------------------------- */

            {random_sign_flip}

            {zero_out_arrays}

            {equal}

            {compare_expected_vs_computed}

            {tests_helpers}

            int main(void)
            {{
              const int mem_aligned_size = SIZE * {sizeof};

              {typ} *vin1;
              {typ} *vin2;

              {typ} *vout_expected;
              {typ} *vout_computed;

              CHECK(vin1 = ({typ} *)nsimd_aligned_alloc(mem_aligned_size));
              CHECK(vin2 = ({typ} *)nsimd_aligned_alloc(mem_aligned_size));

              CHECK(vout_expected = ({typ} *)nsimd_aligned_alloc(mem_aligned_size));
              CHECK(vout_computed = ({typ} *)nsimd_aligned_alloc(mem_aligned_size));

              {tests}

              fprintf(stdout, STATUS "... OK\\n");
              fflush(stdout);
              return EXIT_SUCCESS;
            }}
        ''' .format(head=head,
                    compare_expected_vs_computed=compare_expected_vs_computed(typ, op.name, lang),
                    random_sign_flip='' if typ in common.utypes else random_sign_flip(),
                    zero_out_arrays=zero_out_arrays(typ),
                    equal=equal(typ),
                    tests_helpers=get_adds_tests_cases_given_type(typ)['helpers'],
                    tests=get_adds_tests_cases_given_type(typ)['tests'],
                    op_name = op.name,
                    typ=typ,
                    sizeof = sizeof)
        )

    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests helpers for subs - is overflow/underflow/neither overflow nor underflow

# subs signed

def subs_signed_is_overflow(typ, max_):
      return '''
      int subs_signed_is_overflow(const {typ} a, const {typ} b)
      {{
        return (b < 0) && (a > {max_} + b);
      }}
      '''.format(typ=typ, max_=max_)

def subs_signed_is_underflow(typ, min_):
      return '''
      int subs_signed_is_underflow(const {typ} a, const {typ} b)
      {{
        return (b > 0) && (a < {min_} + b);
      }}
      '''.format(typ=typ, min_=min_)

def subs_signed_is_neither_overflow_nor_underflow(typ):
      return '''
      int subs_signed_is_neither_overflow_nor_underflow(const {typ} a, const {typ} b)
      {{
        return ! subs_signed_is_overflow(a, b) && ! subs_signed_is_underflow(a, b);
      }}
      '''.format(typ=typ)

# subs unsigned

def subs_unsigned_is_underflow(typ):
      return '''
      int subs_unsigned_is_underflow(const {typ} a, const {typ} b)
      {{
        return a < b;
      }}
      '''.format(typ=typ)

# -----------------------------------------------------------------------------
# Tests helpers for subs with signed types

# test signed integer overflow
def test_subs_signed_overflow(typ, min_, max_):
      return '''
      int test_overflow({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        /* if ((vin2[ii] < 0) && (vin1[ii] > {max_} + vin2[ii])) {{ overflow }} */
        int ii = 0;

        /* vin2[ii] < 0 */
        for(ii = 0; ii < SIZE; ++ii)
        {{
          {typ} rand_val = ({typ})((- rand()) % {min_});
          vin2[ii] = (rand_val == 0 ? - 1 : rand_val);
        }}

        /*
        vin1[ii] - vin2[ii] > {max_}
        vin1[ii] > {max_} + vin2[ii]
        vin1[ii] = {max_} + vin2[ii] + rand_val
        s.t.: 0 < rand_val <= - vin2[ii]

        (- TYPE_MIN) overflows
        if vin2[ii] == -1 -->  rand() % -(vin2[ii] + 1) --> rand() % 0
        Therefore check if vin2[ii] == -1 --> if True --> set rand_val == 1
        */

        for(ii = 0; ii < SIZE; ++ii)
        {{
          {typ} rand_val = 0;
          if(-1 == vin2[ii]){{ rand_val = 1; }}
          else{{
            rand_val = ({typ})(rand() % -(vin2[ii] + 1));
            rand_val = (rand_val == 0 ? 1 : rand_val);
          }}
            vin1[ii] = ({typ})({max_} + vin2[ii] + rand_val);
            vout_expected[ii] = {max_};
        }}

        /*
        Test:
        if ((vin2[ii] < 0) && (vin1[ii] > {max_} + vin2[ii])) {{ vout_expected[ii] == {max_}; }}
        */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
     }}
      '''.format(typ=typ, min_=min_, max_=max_)

# test signed underflow
def test_subs_signed_underflow(typ, min_, max_):
      return '''
      int test_underflow({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        /* if ((vin2[ii] > 0) && (vin1[ii] < {min_} + vin2[ii])) {{ underflow }} */
        int ii = 0;

        /* vin2[ii] > 0 */
        for(ii = 0; ii < SIZE; ++ii)
        {{
            {typ} rand_val = ({typ})(rand() % {max_});
            vin2[ii] = (rand_val == 0 ? 1 : rand_val);
        }}

        /*
        vin1[ii] < {min_} + vin2[ii]
        vin1[ii] = {min_} + vin2[ii] - rand_val
        s.t.: 0 < rand_val < vin2[ii]
        */
        for(ii = 0; ii < SIZE; ++ii)
        {{
            {typ} rand_val = ({typ})(rand() % vin2[ii]);
            rand_val = (rand_val == 0 ? 1 : rand_val);
            vin1[ii] = ({typ})({min_} + vin2[ii] - rand_val);
            vout_expected[ii] = {min_};
        }}

        /*
        Test:
        if ((vin2[ii] > 0) && (vin1[ii] < {min_} + vin2[ii])) {{ vout_expected[ii] == {min_}; }}
        */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
      }}
      '''.format(typ=typ, min_=min_, max_=max_)

# test signed neither overflow nor underflow
def test_subs_signed_neither_overflow_nor_underflow(typ, min_, max_):
      return \
        test_signed_neither_overflow_nor_underflow(typ, min_, max_,
         '-', 'subs_signed_is_neither_overflow_nor_underflow')

# test signed all cases
def test_subs_signed_all_cases(typ, min_, max_):
      return test_signed_all_cases(typ, min_, max_, '-', 'subs_signed_is_overflow', 'subs_signed_is_underflow')

# all signed tests
def tests_subs_signed():
      return'''
      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_overflow(vin1, vin2, vout_expected,
                 vout_computed), "overflow");

      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_underflow(vin1, vin2, vout_expected,
                 vout_computed), "underflow");

      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_neither_overflow_nor_underflow(vin1, vin2, vout_expected, vout_computed),
      "neither underflow nor overflow");

      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_all_cases(vin1, vin2, vout_expected,
                 vout_computed), "all cases");
      '''

# -----------------------------------------------------------------------------
# Tests helpers for subs with unsigned types

# test unsigned underflow
def test_subs_unsigned_underflow(typ, min_, max_):
      return '''
      int test_underflow({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        /* if (vin1[ii] < vin2[ii]) {{ underflow }} */
        int ii = 0;

        /* vin1[ii] */
        for(ii = 0; ii < SIZE; ++ii){{ vin1[ii] = ({typ})(({typ})rand() % {max_}); }}

        /*
        vin1[ii] < vin2[ii]
        vin2[ii] = vin1[ii] + rand_val
        s.t.: 0 < rand_val < {max_} - vin1[ii]
        */
        for(ii = 0; ii < SIZE; ++ii)
        {{
            {typ} rand_val = ({typ})(({typ})rand() % ({max_} - vin1[ii]));
            rand_val = (rand_val == 0 ? 1 : rand_val);
            vin2[ii] = ({typ})(vin1[ii] + rand_val);
            vout_expected[ii] = ({typ}){min_};
        }}

        /*
        Test:
        if (vin1[ii] < vin2[ii]) {{ vout_expected[ii] == {min_}; }}
        */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
      }}
      '''.format(typ=typ, min_=min_, max_=max_)

# test unsigned no underflow
def test_subs_unsigned_no_underflow(typ, max_):
      return '''
      int test_no_underflow({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        /* if (vin1[ii] >= vin2[ii]) {{ no underflow }} */
        int ii = 0;

        /* vin1[ii] */
        for(ii = 0; ii < SIZE; ++ii){{ vin1[ii] = ({typ})(({typ})rand() % {max_}); }}

        /*
        vin1[ii] >= vin2[ii]
        vin2 = vin1 - rand_val
        s.t. 0 <= rand_val <= vin1
        */

        for(ii = 0; ii < SIZE; ++ii)
        {{
            {typ} rand_val = ({typ})(({typ})rand() % (vin1[ii] + 1));
            vin2[ii] = ({typ})(vin1[ii] - rand_val);
            vout_expected[ii] = ({typ})(vin1[ii] - vin2[ii]);
        }}

        /*
        Test:
        if (vin1[ii] >= vin2[ii]) {{ vout_expected[ii] == vin1[ii] - vin2[ii]; }}
        */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
      }}
      '''.format(typ=typ, max_=max_)

# test signed all cases
def test_subs_unsigned_all_cases(typ, min_, max_):
      return '''
      int test_all_cases({typ} vin1[], {typ} vin2[], {typ} vout_expected[], {typ} vout_computed[])
      {{
        int ii = 0;
        for(ii = 0; ii < SIZE; ++ii)
        {{
          vin1[ii] = ({typ})(({typ})rand() % {max_});
          vin2[ii] = ({typ})(({typ})rand() % {max_});
          if(subs_unsigned_is_underflow(vin1[ii], vin2[ii]))
          {{
            vout_expected[ii] = ({typ}){min_};
          }}
          else {{ vout_expected[ii] = ({typ})(vin1[ii] - vin2[ii]); }}
        }}
        /* Test all cases: */
        return compare_expected_vs_computed(vin1, vin2, vout_expected, vout_computed);
      }}
      '''.format(typ=typ, min_=min_, max_=max_)

# all unsigned tests
def tests_subs_unsigned():
      return'''
      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_underflow(vin1, vin2, vout_expected,
                 vout_computed), "underflow");

      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_no_underflow(vin1, vin2, vout_expected, vout_computed),
      "no underflow");

      zero_out_arrays(vin1, vin2, vout_expected, vout_computed);
      CHECK_CASE(test_all_cases(vin1, vin2, vout_expected,
                 vout_computed), "all cases");
      '''

# ------------------------------------------------------------------------------
# Get subs tests given type

def get_subs_tests_cases_for_signed_types(typ, min_, max_):
      helpers = '''
            {test_subs_signed_overflow}

            {test_subs_signed_underflow}

            {subs_signed_is_overflow}

            {subs_signed_is_underflow}

            {subs_signed_is_neither_overflow_nor_underflow}

            {test_subs_signed_neither_overflow_nor_underflow}

            {test_subs_signed_all_cases}
          ''' .format(test_subs_signed_overflow=test_subs_signed_overflow(typ, min_, max_),
                      test_subs_signed_underflow=test_subs_signed_underflow(typ, min_, max_),
                      subs_signed_is_overflow=subs_signed_is_overflow(typ, max_),
                      subs_signed_is_underflow=subs_signed_is_underflow(typ, min_),
                      subs_signed_is_neither_overflow_nor_underflow=subs_signed_is_neither_overflow_nor_underflow(typ),
                      test_subs_signed_neither_overflow_nor_underflow=test_subs_signed_neither_overflow_nor_underflow(
                          typ, min_=min_, max_=max_),
                      test_subs_signed_all_cases=test_subs_signed_all_cases(typ, min_=min_, max_=max_)
                      )
      return {'helpers': helpers, 'tests': tests_subs_signed()}

def get_subs_tests_cases_for_unsigned_types(typ, min_, max_):
      helpers = '''
          {test_subs_unsigned_underflow}

          {test_subs_unsigned_no_underflow}

          {subs_unsigned_is_underflow}

          {test_subs_unsigned_all_cases}
          ''' .format(test_subs_unsigned_underflow=test_subs_unsigned_underflow(typ, min_, max_),
                      test_subs_unsigned_no_underflow=test_subs_unsigned_no_underflow(typ, max_),
                      subs_unsigned_is_underflow=subs_unsigned_is_underflow(typ),
                      test_subs_unsigned_all_cases=test_subs_unsigned_all_cases(typ, min_, max_)
                      )
      return {'helpers': helpers, 'tests': tests_subs_unsigned()}

def get_subs_tests_cases_given_type(typ):
      if typ in common.iutypes:
            type_limits = common.limits[typ]
            min_ = type_limits['min']
            max_ = type_limits['max']

            if typ in common.itypes:
                  return get_subs_tests_cases_for_signed_types(typ=typ, min_=min_, max_=max_)

            if typ in common.utypes:
                  return get_subs_tests_cases_for_unsigned_types(typ=typ, min_=min_, max_=max_)
      else:
            msg = '{typ} not implemented'.format(typ=typ)
            raise TypeError(msg)

# -----------------------------------------------------------------------------
# gen_subs

def gen_subs(opts, op, typ, lang, ulps):

    # Do not test for floats since subs(floats) == sub(floats)
    if typ in common.ftypes:
          return

    filename = get_filename(opts, op, typ, lang)

    if filename == None:
        return

    sizeof = common.sizeof(typ)

    head = '''
              {includes}
              #include <assert.h>

              #define SIZE (2048 / {sizeof})

              #define STATUS "test of {op_name} over {typ}"

              {aligned_alloc_error}

              {adds_subs_check_case}
            ''' .format(includes=get_includes(lang),
                        op_name=op.name,
                        typ=typ,
                        sizeof=sizeof,
                        aligned_alloc_error=aligned_alloc_error(),
                        adds_subs_check_case=adds_subs_check_case())

    with common.open_utf8(opts, filename) as out:
        out.write(
            ''' \
            {head}
            /* ------------------------------------------------------------------------- */

            {random_sign_flip}

            {zero_out_arrays}

            {equal}

            {compare_expected_vs_computed}

            {tests_helpers}

            int main(void)
            {{
              const int mem_aligned_size = SIZE * {sizeof};

              {typ} *vin1;
              {typ} *vin2;

              {typ} *vout_expected;
              {typ} *vout_computed;

              CHECK(vin1 = ({typ} *)nsimd_aligned_alloc(mem_aligned_size));
              CHECK(vin2 = ({typ} *)nsimd_aligned_alloc(mem_aligned_size));

              CHECK(vout_expected = ({typ} *)nsimd_aligned_alloc(mem_aligned_size));
              CHECK(vout_computed = ({typ} *)nsimd_aligned_alloc(mem_aligned_size));

              {tests}

              fprintf(stdout, STATUS "... OK\\n");
              fflush(stdout);
              return EXIT_SUCCESS;
            }}
        ''' .format(head=head,
                    compare_expected_vs_computed=compare_expected_vs_computed(typ, op.name, lang),
                    random_sign_flip='' if typ in common.utypes else random_sign_flip(),
                    zero_out_arrays=zero_out_arrays(typ),
                    equal=equal(typ),
                    tests_helpers=get_subs_tests_cases_given_type(typ)['helpers'],
                    tests=get_subs_tests_cases_given_type(typ)['tests'],
                    op_name = op.name,
                    typ=typ,
                    sizeof = sizeof)
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
    with common.open_utf8(opts, filename) as out:
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
        comp = '*((u16*)vin + i) != *((u16 *)vout + i)'
    else:
        rand = 'vin[i] = ({})(rand() % 10);'.format(typ)
        comp = 'vin[i] != vout[i]'

    if align=='u':
        unalign = '+1'
    else:
        unalign = ''

    with common.open_utf8(opts, filename) as out:
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
             CHECK(vin = ({typ}*)nsimd_aligned_alloc(n * {sizeof} {unalign}) {unalign});
             CHECK(vout = ({typ}*)nsimd_aligned_alloc(n * {sizeof} {unalign}) {unalign});

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
                        comp=comp, unalign=unalign))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for gather/scatter

def gen_gather_scatter(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return

    ityp = 'i' + typ[1:]

    if lang == 'c_base':
        if op.name == 'gather_linear':
            gather_scatter = '''vscatter_linear(vout + 1, 2, vgather_linear(
                                    vin, 2, {typ}), {typ});'''.format(typ=typ)
        else:
            gather_scatter = \
                '''vec({ityp}) offsets = vmul(viota({ityp}), vset1(({ityp})2,
                                              {ityp}), {ityp});
                   vec({typ}) v = vgather(vin, offsets, {typ});
                   offsets = vadd(offsets, vset1(({ityp})1, {ityp}), {ityp});
                   vscatter(vout, offsets, v, {typ});'''. \
                   format(typ=typ, ityp=ityp)
    elif lang == 'cxx_base':
        if op.name == 'gather_linear':
            gather_scatter = '''nsimd::scatter_linear(vout + 1, 2,
                                  nsimd::gather_linear(
                                    vin, 2, {typ}()), {typ}());'''. \
                                    format(typ=typ)
        else:
            gather_scatter = \
            '''vec({ityp}) offsets = nsimd::mul(nsimd::iota({ityp}()),
                                     nsimd::set1(({ityp})2, {ityp}()),
                                     {ityp}());
               vec({typ}) v = nsimd::gather(vin, offsets, {typ}());
               offsets = nsimd::add(offsets, nsimd::set1(({ityp})1, {ityp}()),
                                    {ityp}());
               nsimd::scatter(vout, offsets, v, {typ}());'''. \
               format(typ=typ, ityp=ityp)
    else:
        if op.name == 'gather_linear':
            gather_scatter = '''nsimd::scatter_linear(vout + 1, 2,
                                  nsimd::gather_linear<nsimd::pack<{typ}> >(
                                      vin, 2));'''.format(typ=typ)
        else:
            gather_scatter = \
            '''typedef nsimd::pack<{typ}> pack;
               typedef nsimd::pack<{ityp}> ipack;
               ipack offsets = nsimd::mul(nsimd::iota<ipack>(),
                               nsimd::set1<ipack>(({ityp})2));
               pack v = nsimd::gather<pack>(vin, offsets);
               offsets = nsimd::add(offsets, nsimd::set1<ipack>(({ityp})1));
               nsimd::scatter(vout, offsets, v);'''. \
               format(typ=typ, ityp=ityp)

    if typ == 'f16':
        one = 'nsimd_f32_to_f16(1.0f)'
        zero = 'nsimd_f32_to_f16(0.0f)'
        comp = 'nsimd_f16_to_f32(vout[i]) != 0.0f'
    else:
        one = '({typ})1'.format(typ=typ)
        zero = '({typ})0'.format(typ=typ)
        comp = 'vout[i] != ({typ})0'.format(typ=typ)

    with common.open_utf8(opts, filename) as out:
        out.write(
           '''{includes}

           #define STATUS "test of {op_name} over {typ}"

           int main(void) {{
             int n = 2 * vlen({typ});
             int i;
             {typ} vin[2 * NSIMD_MAX_LEN({typ})];
             {typ} vout[2 * NSIMD_MAX_LEN({typ})];

             fprintf(stdout, "test of {op_name} over {typ}...\\n");

             /* Fill input and output with 0 1 0 1 0 1 ... */
             for (i = 0; i < n; i++) {{
               if ((i % 2) == 1) {{
                 vin[i] = {one};
                 vout[i] = {one};
               }} else {{
                 vin[i] = {zero};
                 vout[i] = {zero};
               }}
             }}

             /* We gather odd offsets elements from vin and put then at even */
             /* offsets. */
             {{
               {gather_scatter}
             }}

             /* Compare results */
             for (i = 0; i < n; i++) {{
               if ({comp}) {{
                 fprintf(stdout, STATUS "... FAIL\\n");
                 fflush(stdout);
                 return -1;
               }}
             }}

             fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
             return EXIT_SUCCESS;
           }}'''.format(includes=get_includes(lang), ityp=ityp, comp=comp,
                        typ=typ, year=date.today().year, op_name=op.name,
                        gather_scatter=gather_scatter, zero=zero, one=one))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for masked scatter

def gen_mask_scatter(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return

    ityp = 'i' + typ[1:]

    if typ == 'f16':
        two = 'nsimd_f32_to_f16(2.0f)'
        one = 'nsimd_f32_to_f16(1.0f)'
        zero = 'nsimd_f32_to_f16(0.0f)'
        comp_with_0 = 'nsimd_f16_to_f32(vout[2 * k]) != 0.0f'
        comp_with_1 = 'nsimd_f16_to_f32(vout[2 * k + 1]) != 1.0f'
        comp_with_2 = 'nsimd_f16_to_f32(vout[2 * k]) != 2.0f'
    else:
        two = '({typ})2'.format(typ=typ)
        one = '({typ})1'.format(typ=typ)
        zero = '({typ})0'.format(typ=typ)
        comp_with_0 = 'vout[2 * k] != ({typ})0'.format(typ=typ)
        comp_with_1 = 'vout[2 * k + 1] != ({typ})1'.format(typ=typ)
        comp_with_2 = 'vout[2 * k] != ({typ})2'.format(typ=typ)

    if lang == 'c_base':
        mask_scatter = \
            '''vec({ityp}) offsets = vmul(viota({ityp}), vset1(({ityp})2,
                                          {ityp}), {ityp});
               vecl({typ}) mask = vmask_for_loop_tail(0, i, {typ});
               vmask_scatter(mask, vout, offsets, vset1({two}, {typ}),
                             {typ});'''.format(two=two, typ=typ, ityp=ityp)
    elif lang == 'cxx_base':
        mask_scatter = \
            '''vec({ityp}) offsets = nsimd::mul(nsimd::iota({ityp}()),
                                     nsimd::set1(({ityp})2, {ityp}()),
                                     {ityp}());
               vecl({typ}) mask = nsimd::mask_for_loop_tail(0, i, {typ}());
               nsimd::mask_scatter(mask, vout, offsets, nsimd::set1(
                                   {two}, {typ}()), {typ}());'''. \
                                   format(two=two, typ=typ, ityp=ityp)
    else:
        mask_scatter = \
            '''typedef nsimd::pack<{typ}> pack;
               typedef nsimd::pack<{ityp}> ipack;
               typedef nsimd::packl<{typ}> packl;
               ipack offsets = nsimd::mul(nsimd::iota<ipack>(),
                               nsimd::set1<ipack>(({ityp})2));
               packl mask = nsimd::mask_for_loop_tail<packl>(0, i);
               nsimd::mask_scatter(mask, vout, offsets,
                                   nsimd::set1<pack>({two}));'''. \
                                   format(two=two, typ=typ, ityp=ityp)

    with common.open_utf8(opts, filename) as out:
        out.write(
           '''{includes}

           #define STATUS "test of {op_name} over {typ}"

           int main(void) {{
             int n = 2 * vlen({typ});
             int i, j, k;
             {typ} vout[2 * NSIMD_MAX_LEN({typ})];

             fprintf(stdout, "test of {op_name} over {typ}...\\n");

             for (i = 0; i < n / 2; i++) {{
               /* Fill output with 0 1 0 1 0 1 ... */
               for (j = 0; j < n; j++) {{
                 vout[j] = (j % 2 == 0 ? {zero} : {one});
               }}

               {{
                 {mask_scatter}
               }}

               /* Check results */
               for (k = 0; k < n / 2; k++) {{
                 if ({comp_with_1}) {{
                   goto error;
                 }}
               }}
               for (k = 0; k < i; k++) {{
                 if ({comp_with_2}) {{
                   goto error;
                 }}
               }}
               for (; k < n / 2; k++) {{
                 if ({comp_with_0}) {{
                   goto error;
                 }}
               }}
             }}

             fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
             fflush(stdout);
             return EXIT_SUCCESS;

           error:
             fprintf(stdout, STATUS "... FAIL\\n");
             fflush(stdout);
             return EXIT_FAILURE;
           }}'''.format(includes=get_includes(lang), ityp=ityp, two=two,
                        typ=typ, year=date.today().year, op_name=op.name,
                        mask_scatter=mask_scatter, zero=zero, one=one,
                        comp_with_0=comp_with_0, comp_with_2=comp_with_2,
                        comp_with_1=comp_with_1))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for masked gather

def gen_maskoz_gather(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return

    ityp = 'i' + typ[1:]

    if typ == 'f16':
        three = 'nsimd_f32_to_f16(3.0f)'
        two = 'nsimd_f32_to_f16(2.0f)'
        one = 'nsimd_f32_to_f16(1.0f)'
        zero = 'nsimd_f32_to_f16(0.0f)'
        comp_with_1 = 'nsimd_f16_to_f32(vout[k]) != 1.0f'
        if op.name == 'maskz_gather':
            comp_with_0_or_3 = 'nsimd_f16_to_f32(vout[k]) != 0.0f'
        else:
            comp_with_0_or_3 = 'nsimd_f16_to_f32(vout[k]) != 3.0f'
    else:
        three = '({typ})3'.format(typ=typ)
        two = '({typ})2'.format(typ=typ)
        one = '({typ})1'.format(typ=typ)
        zero = '({typ})0'.format(typ=typ)
        comp_with_1 = 'vout[k] != ({typ})1'.format(typ=typ)
        if op.name == 'maskz_gather':
            comp_with_0_or_3 = 'vout[k] != ({typ})0'.format(typ=typ)
        else:
            comp_with_0_or_3 = 'vout[k] != ({typ})3'.format(typ=typ)

    oz = 'o' if op.name == 'masko_gather' else 'z'

    if lang == 'c_base':
        ta = ', vset1({three}, {typ})'.format(three=three, typ=typ) \
             if op.name == 'masko_gather' else ''
        maskoz_gather = \
            '''vec({ityp}) offsets = vmul(viota({ityp}), vset1(({ityp})2,
                                          {ityp}), {ityp});
               vecl({typ}) mask = vmask_for_loop_tail(0, i, {typ});
               vstoreu(vout, vmask{oz}_gather(mask, vin, offsets{ta},
                       {typ}), {typ});'''. \
                       format(typ=typ, ityp=ityp, ta=ta, oz=oz)
    elif lang == 'cxx_base':
        ta = ', nsimd::set1({three}, {typ}())'.format(three=three, typ=typ) \
             if op.name == 'masko_gather' else ''
        maskoz_gather = \
            '''vec({ityp}) offsets = nsimd::mul(nsimd::iota({ityp}()),
                                     nsimd::set1(({ityp})2, {ityp}()),
                                     {ityp}());
               vecl({typ}) mask = nsimd::mask_for_loop_tail(0, i, {typ}());
               nsimd::storeu(vout, nsimd::mask{oz}_gather(
                   mask, vin, offsets{ta}, {typ}()), {typ}());'''. \
                   format(typ=typ, ityp=ityp, ta=ta, oz=oz)
    else:
        ta = ', nsimd::set1<nsimd::pack<{typ}> >({three})'. \
             format(three=three, typ=typ) if op.name == 'masko_gather' else ''
        maskoz_gather = \
            '''typedef nsimd::pack<{ityp}> ipack;
               typedef nsimd::packl<{typ}> packl;
               ipack offsets = nsimd::mul(nsimd::iota<ipack>(),
                               nsimd::set1<ipack>(({ityp})2));
               packl mask = nsimd::mask_for_loop_tail<packl>(0, i);
               nsimd::storeu(vout, nsimd::mask{oz}_gather(
                   mask, vin, offsets{ta}));'''. \
                   format(ta=ta, oz=oz, typ=typ, ityp=ityp)

    with common.open_utf8(opts, filename) as out:
        out.write(
           '''{includes}

           #define STATUS "test of {op_name} over {typ}"

           int main(void) {{
             int n = 2 * vlen({typ});
             int i, j, k;
             {typ} vin[2 * NSIMD_MAX_LEN({typ})];
             {typ} vout[NSIMD_MAX_LEN({typ})];

             fprintf(stdout, "test of {op_name} over {typ}...\\n");

             for (i = 0; i < n / 2; i++) {{
               /* Fill input with 1 0 1 0 1 0 ... */
               for (j = 0; j < n; j++) {{
                 vin[j] = (j % 2 == 1 ? {zero} : {one});
               }}

               /* Fill output with 2's ... */
               for (j = 0; j < n / 2; j++) {{
                 vout[j] = {two};
               }}

               {{
                 {maskoz_gather}
               }}

               /* Check results */
               for (k = 0; k < i; k++) {{
                 if ({comp_with_1}) {{
                   goto error;
                 }}
               }}
               for (; k < n / 2; k++) {{
                 if ({comp_with_0_or_3}) {{
                   goto error;
                 }}
               }}
             }}

             fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
             fflush(stdout);
             return EXIT_SUCCESS;

           error:
             fprintf(stdout, STATUS "... FAIL\\n");
             fflush(stdout);
             return EXIT_FAILURE;
           }}'''.format(includes=get_includes(lang), ityp=ityp, two=two,
                        typ=typ, year=date.today().year, op_name=op.name,
                        maskoz_gather=maskoz_gather, zero=zero, one=one,
                        comp_with_0_or_3=comp_with_0_or_3, three=three,
                        comp_with_1=comp_with_1))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for masked loads

def gen_mask_load(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return

    if typ == 'f16':
        fill_vin = 'vin[i] = nsimd_f32_to_f16((f32)i);'
        m1 = 'nsimd_f32_to_f16(-1.0f)'
        comp1 = 'nsimd_f16_to_f32(vout[j]) != (f32)j'
    else:
        fill_vin = 'vin[i] = ({typ})i;'.format(typ=typ)
        m1 = '({typ})-1'.format(typ=typ)
        comp1 = 'vout[j] != ({typ})j'.format(typ=typ)

    if op.name in ['masko_loada1', 'masko_loadu1']:
        if lang == 'c_base':
            test = \
            '''vecl({typ}) mask = vmask_for_loop_tail(0, i, {typ});
               vec({typ}) other = vset1({m1}, {typ});
               vstoreu(vout, v{op_name}(mask, vin, other, {typ}), {typ});'''. \
               format(typ=typ, op_name=op.name, m1=m1)
        elif lang == 'cxx_base':
            test = \
            '''vecl({typ}) mask = nsimd::mask_for_loop_tail(0, i, {typ}());
               vec({typ}) other = nsimd::set1({m1}, {typ}());
               nsimd::storeu(vout, nsimd::{op_name}(
                   mask, vin, other, {typ}()), {typ}());'''. \
                   format(typ=typ, op_name=op.name, m1=m1)
        elif lang == 'cxx_adv':
            test = \
            '''nsimd::packl<{typ}> mask =
                   nsimd::mask_for_loop_tail<nsimd::packl<{typ}> >(0, i);
               nsimd::pack<{typ}> other = nsimd::set1<nsimd::pack<{typ}> >(
                                              {m1});
               nsimd::storeu(vout, nsimd::{op_name}(mask, vin, other));'''. \
               format(typ=typ, op_name=op.name, m1=m1)
        comp2 = 'vout[j] != ({typ})-1'.format(typ=typ) if typ != 'f16' else \
                'nsimd_f16_to_f32(vout[j]) != -1.0f'
    else:
        if lang == 'c_base':
            test = \
            '''vecl({typ}) mask = vmask_for_loop_tail(0, i, {typ});
               vstoreu(vout, v{op_name}(mask, vin, {typ}), {typ});'''. \
               format(typ=typ, op_name=op.name, m1=m1)
        elif lang == 'cxx_base':
            test = \
            '''vecl({typ}) mask = nsimd::mask_for_loop_tail(0, i, {typ}());
               nsimd::storeu(vout, nsimd::{op_name}(
                   mask, vin, {typ}()), {typ}());'''. \
                   format(typ=typ, op_name=op.name, m1=m1)
        elif lang == 'cxx_adv':
            test = \
            '''nsimd::packl<{typ}> mask =
                   nsimd::mask_for_loop_tail<nsimd::packl<{typ}> >(0, i);
               nsimd::storeu(vout, nsimd::{op_name}(mask, vin));'''. \
               format(typ=typ, op_name=op.name, m1=m1)
        comp2 = 'vout[j] != ({typ})0'.format(typ=typ) if typ != 'f16' else \
                'nsimd_f16_to_f32(vout[j]) != -0.0f'

    if op.name in ['masko_loadu1', 'maskz_loadu1']:
        unalign = '\nvin += 1;'
    else:
        unalign = ''

    with common.open_utf8(opts, filename) as out:
        out.write(
           '''{includes}

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
             int i, j;
             {typ} *vin;
             {typ} vout[NSIMD_MAX_LEN({typ})];
             int len = vlen({typ});

             fprintf(stdout, "test of {op_name} over {typ}...\\n");

             CHECK(vin = ({typ}*)nsimd_aligned_alloc(2 * len));{unalign}

             /* Fill with data */
             for (i = 0; i < len; i++) {{
               {fill_vin}
             }}

             /* Load and put back data into vout */
             for (i = 0; i < len; i++) {{
               {test}

               for (j = 0; j < i; j++) {{
                 if ({comp1}) {{
                   fprintf(stdout, STATUS "... FAIL\\n");
                   fflush(stdout);
                   return -1;
                 }}
               }}
               for (; j < len; j++) {{
                 if ({comp2}) {{
                   fprintf(stdout, STATUS "... FAIL\\n");
                   fflush(stdout);
                   return -1;
                 }}
               }}
             }}

             fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
             return EXIT_SUCCESS;
           }}'''.format(includes=get_includes(lang), op_name=op.name,
                        typ=typ, year=date.today().year, test=test,
                        comp1=comp1, comp2=comp2, unalign=unalign,
                        fill_vin=fill_vin))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for masked stores

def gen_mask_store(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return

    if typ == 'f16':
        fill_vout = 'vout[i] = nsimd_f32_to_f16((f32)0);'
        one = 'nsimd_f32_to_f16(1.0f)'
        comp1 = 'nsimd_f16_to_f32(vout[j]) != (f32)1'
        comp2 = 'nsimd_f16_to_f32(vout[j]) != (f32)0'
    else:
        fill_vout = 'vout[i] = ({typ})0;'.format(typ=typ)
        one = '({typ})1'.format(typ=typ)
        comp1 = 'vout[j] != ({typ})1'.format(typ=typ)
        comp2 = 'vout[j] != ({typ})0'.format(typ=typ)

    if lang == 'c_base':
        test = \
        '''vecl({typ}) mask = vmask_for_loop_tail(0, i, {typ});
           v{op_name}(mask, vout, vset1({one}, {typ}), {typ});'''. \
           format(typ=typ, op_name=op.name, one=one)
    elif lang == 'cxx_base':
        test = \
        '''vecl({typ}) mask = nsimd::mask_for_loop_tail(0, i, {typ}());
           nsimd::{op_name}(mask, vout, nsimd::set1({one}, {typ}()),
                            {typ}());'''.format(typ=typ, op_name=op.name,
                                                one=one)
    elif lang == 'cxx_adv':
        test = \
        '''nsimd::packl<{typ}> mask =
               nsimd::mask_for_loop_tail<nsimd::packl<{typ}> >(0, i);
           nsimd::{op_name}(mask, vout,
                            nsimd::set1<nsimd::pack<{typ}> >({one}));'''. \
                            format(typ=typ, op_name=op.name, one=one)

    if op.name == 'mask_storeu1':
        unalign = '\nvout += 1;'
    else:
        unalign = ''

    with common.open_utf8(opts, filename) as out:
        out.write(
           '''{includes}

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
             int i, j;
             {typ} *vout;
             int len = vlen({typ});

             fprintf(stdout, "test of {op_name} over {typ}...\\n");

             CHECK(vout = ({typ}*)nsimd_aligned_alloc(2 * len));{unalign}

             /* Fill vout with zeors */
             for (i = 0; i < len; i++) {{
               {fill_vout}
             }}

             /* Store data into vout */
             for (i = 0; i < len; i++) {{
               {test}

               for (j = 0; j < i; j++) {{
                 if ({comp1}) {{
                   fprintf(stdout, STATUS "... FAIL\\n");
                   fflush(stdout);
                   return -1;
                 }}
               }}
               for (; j < len; j++) {{
                 if ({comp2}) {{
                   fprintf(stdout, STATUS "... FAIL\\n");
                   fflush(stdout);
                   return -1;
                 }}
               }}
             }}

             fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
             return EXIT_SUCCESS;
           }}'''.format(includes=get_includes(lang), op_name=op.name,
                        typ=typ, year=date.today().year, test=test,
                        comp1=comp1, comp2=comp2, unalign=unalign,
                        fill_vout=fill_vout))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests that load/store of degrees 2, 3 and 4 ravels vectors correctly

def gen_load_store_ravel(opts, op, typ, lang):
    # This test only the libs internal, not the API, so we only generate test
    # for c
    filename = get_filename(opts, op, typ, lang, 'ravel')
    if filename == None:
        return

    deg = op.name[4]
    align = op.name[5]

    if typ=='f16':
        convert_to='nsimd_f32_to_f16((f32)'
    else:
        convert_to='({typ})('.format(typ=typ)

    check = '\n'.join(['''
      comp = vset1({convert_to}{i}+1), {typ});
      err = err || vany(vne(v.v{i}, comp, {typ}), {typ});
      '''.format(typ=typ, i=i, convert_to=convert_to) \
      for i in range (0, int(deg))])

    with common.open_utf8(opts, filename) as out:
        out.write(
        '''{includes}

           #define SIZE (2048 / {sizeof})

           #define STATUS "test raveling of {op_name} over {typ}"

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
             fprintf(stdout, "test raveling of {op_name} over {typ}...\\n");

             {typ}* vin;
             {typ}* vout;
             int i;
             int len = vlen({typ});
             int n = {deg} * len;

             CHECK(vin = ({typ}*)nsimd_aligned_alloc(n * {sizeof}));
             CHECK(vout = ({typ}*)nsimd_aligned_alloc(n * {sizeof}));

             /* Fill in the vectors */
             for (i=0; i<n; ++i) {{
                 vin[i] = {convert_to}(i%{deg}) + 1);
             }}

             /* Load data and check that each vector is correctly filled */
             vecx{deg}({typ}) v = v{op_name}(vin, {typ});

             int err=0;
             vec({typ}) comp;

             {check}

             if (err) {{
               fprintf(stdout, STATUS "... FAIL\\n");
               fflush(stdout);
               return -1;
             }}

             fprintf(stdout, "Raveling of {op_name} over {typ}... OK\\n");
             return EXIT_SUCCESS;
           }}'''.format(includes=get_includes(lang), op_name=op.name,
                        typ=typ, year=date.today().year, deg=deg,
                        convert_to=convert_to,
                        sizeof=common.sizeof(typ), check=check))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Tests for iota

def gen_iota(opts, op, typ, lang):
    filename = get_filename(opts, op, typ, lang)
    if filename == None:
        return
    if lang == 'c_base':
        do_iota = 'vstoreu(buf, viota({typ}), {typ});'.format(typ=typ)
    elif lang == 'cxx_base':
        do_iota = 'nsimd::storeu(buf, nsimd::iota({typ}()), {typ}());'. \
                  format(typ=typ)
    else:
        do_iota = 'nsimd::storeu(buf, nsimd::iota<nsimd::pack<{typ}> >());'. \
                  format(typ=typ)

    if typ == 'f16':
        comp_i = 'nsimd_f16_to_f32(buf[i]) != (f32)i'
    else:
        comp_i = 'buf[i] != ({typ})i'.format(typ=typ)

    with common.open_utf8(opts, filename) as out:
        out.write(
            '''{includes}

           int main(void) {{
             int i;
             {typ} buf[NSIMD_MAX_LEN({typ})];
             int len = vlen({typ});

             fprintf(stdout, "test of {op_name} over {typ}...\\n");

             {do_iota}

             for (i = 0; i < len; i++) {{
               if ({comp_i}) {{
                 exit(EXIT_FAILURE);
               }}
             }}

             fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
             return EXIT_SUCCESS;
           }}'''.format(includes=get_includes(lang), op_name=op.name,
                        typ=typ, do_iota=do_iota, year=date.today().year,
                        comp_i=comp_i))
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
    with common.open_utf8(opts, filename) as out:
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
    with common.open_utf8(opts, filename) as out:
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
        test_code = \
        'vstorea(out, vreverse(vloada(in, {typ}), {typ}), {typ});'. \
        format(typ=typ)
    elif lang == 'cxx_base':
        test_code = \
        'nsimd::storea(out, nsimd::reverse(nsimd::loada(in, {typ}()), ' \
        '{typ}()), {typ}());'.format(typ=typ)
    elif lang == 'cxx_adv':
        test_code = \
        'nsimd::storea(out, nsimd::reverse(' \
        'nsimd::loada<nsimd::pack<{typ}> >(in)));'.format(typ=typ)
    if typ == 'f16':
        init = 'in[ i ] = nsimd_f32_to_f16((float)(i + 1));'
        comp = 'ok &= nsimd_f16_to_f32(out[len - 1 - i]) == ' \
               'nsimd_f16_to_f32(in[i]);'
    else:
        init = 'in[ i ] = ({typ})(i + 1);'.format(typ=typ)
        comp = 'ok &= out[len - 1 - i] == in[i];'

    with common.open_utf8(opts, filename) as out:
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
           }}'''.format(includes=get_includes(lang), op_name=op.name,
                        typ=typ, test_code=test_code, year=date.today().year,
                        sizeof=common.sizeof(typ), init=init, comp=comp))

    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Unpack half

def gen_unpack_half(opts, op, typ, lang):
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

    op_test =  'step/(2 * nb_lane)'
    if op.name in['ziphi', 'ziplo']:
        offset = 'int offset = {val};'.format(val= '0' \
                 if op.name == 'ziplo' else 'vlen({typ}) / 2'.format(typ=typ))
    else:
        offset = ''

    if op.name in ['unziplo', 'unziphi']:
        if typ == 'f16':
            comp_unpack = '''
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
    head = '''{posix_c_source}

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
                        posix_c_source=posix_c_source,
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

    with common.open_utf8(opts, filename) as out:
        out.write(
        '''{head}

           int main(void) {{
              int vi, i, j, step;
              {typ} *vin1, *vin2;
              {typ} *vout;

              CHECK(vin1 = ({typ} *)nsimd_aligned_alloc(SIZE * {sizeof}));
              CHECK(vin2 = ({typ} *)nsimd_aligned_alloc(SIZE * {sizeof}));
              CHECK(vout = ({typ} *)nsimd_aligned_alloc(SIZE * {sizeof}));

              step = vlen({typ});

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

# ------------------------------------------------------------------------------
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
        vout1_comp = '''vec({typ}) va1, va2;
        vecx2({typ}) vc;
        va1 = vloadu(&vin1[i], {typ});
        va2 = vloadu(&vin2[i], {typ});
        vc = v{op_name}(va1, va2, {typ});
        vstoreu(&vout[2 * i], vc.v0, {typ});
        vstoreu(&vout[2 * i + vlen({typ})], vc.v1, {typ});'''. \
            format(typ=typ, op_name=op.name)
    if lang == 'cxx_base':
        extra_code = relative_distance_cpp
        typ_nsimd = 'vec({typ})'.format(typ=typ)
        vout1_comp = '''vec({typ}) va1, va2;
        vecx2({typ}) vc;
        va1 = nsimd::loadu(&vin1[i], {typ}());
        va2 = nsimd::loadu(&vin2[i], {typ}());
        vc = nsimd::{op_name}(va1, va2, {typ}());
        nsimd::storeu(&vout[2 * i], vc.v0, {typ}());
        nsimd::storeu(&vout[2 * i + vlen({typ})], vc.v1, {typ}());'''. \
            format(typ=typ, op_name=op.name)
    if lang == 'cxx_adv':
        extra_code = relative_distance_cpp
        typ_nsimd = 'nsimd::pack<{typ}>'.format(typ=typ)
        vout1_comp = '''nsimd::pack<{typ}> va1, va2;
        nsimd::packx2<{typ}> vc;
        va1 = nsimd::loadu<nsimd::pack<{typ}> >(&vin1[i]);
        va2 = nsimd::loadu<nsimd::pack<{typ}> >(&vin2[i]);
        vc = nsimd::{op_name}(va1, va2);
        nsimd::storeu(&vout[2 * i], vc.v0);
        nsimd::storeu(&vout[2 * i + nsimd::len({typ}())], vc.v1);'''. \
            format(typ=typ, op_name=op.name)

    head = '''{posix_c_source}

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
                posix_c_source=posix_c_source,
                includes=get_includes(lang),
                extra_code=extra_code,
                sizeof=common.sizeof(typ), simd= opts.simd)

    if typ == 'f16':
        rand = '''nsimd_f32_to_f16((f32)(2 * (rand() % 2) - 1) *
        (f32)(1 << (rand() % 4)) /
        (f32)(1 << (rand() % 4)))'''
    else:
        rand = '''({typ})(({typ})(2 * (rand() % 2) - 1) * ({typ})(1 << (rand() % 4))
        / ({typ})(1 << (rand() % 4)))'''.format(typ=typ)

    if op.name == 'zip':
        scalar_code = '''\
        for(i = 0; i < step; i ++)
        {{
        out_ptr[2 * i] = vin1_ptr[i];
        out_ptr[2 * i + 1] = vin2_ptr[i];
        }}
        '''
    else:
        scalar_code = '''\
        for(i = 0; i < step / 2; i++)
        {{
        out_ptr[i] = vin1_ptr[2 * i];
        out_ptr[step / 2 + i] = vin2_ptr[2 * i];
        out_ptr[step + i] = vin1_ptr[2 * i + 1];
        out_ptr[step + step / 2 + i] = vin2_ptr[2 * i + 1];
        }}'''

    if typ == 'f16':
        comp = 'nsimd_f16_to_f32(vout[vi]) !=  nsimd_f16_to_f32(vout_ref[vi])'
    else:
        comp = 'vout[vi] != vout_ref[vi]'

    with common.open_utf8(opts, filename) as out:
        out.write(
        '''{head}

        int main(void){{
          int i, vi, step;
          {typ} *vin1, *vin2;
          {typ} *vout;
          {typ} *vout_ref;

          CHECK(vin1 = ({typ} *)nsimd_aligned_alloc(SIZE * {sizeof}));
          CHECK(vin2 = ({typ} *)nsimd_aligned_alloc(SIZE * {sizeof}));
          CHECK(vout = ({typ} *)nsimd_aligned_alloc(2 * SIZE * {sizeof}));
          CHECK(vout_ref = ({typ} *)nsimd_aligned_alloc(2 * SIZE * {sizeof}));

          step = vlen({typ});

          fprintf(stdout, "test of {op_name} over {typ}...\\n");

          /* Fill input vector(s) with random */
          for (i = 0; i < SIZE; i++)
          {{
            vin1[i] = {rand};
            vin2[i] = {rand};
          }}

          /* Compute a scalar reference version */
          for(vi = 0; vi < SIZE; vi += step)
          {{
            {typ} *out_ptr = vout_ref + 2 * vi;
            {typ} *vin1_ptr = vin1 + vi;
            {typ} *vin2_ptr = vin2 + vi;

            {scalar_code}
          }}

          /* Fill output vector with computed values */
          for (i = 0; i < SIZE; i += step)
          {{
            {vout1_comp}
          }}

          /* Compare results */
          for(vi = 0; vi < SIZE; vi++) {{
            if({comp}) {{
              fprintf(stderr, "test of {op_name} over {typ}... FAIL\\n");
              exit(EXIT_FAILURE);
            }}
          }}

          fprintf(stdout, "test of {op_name} over {typ}... OK\\n");
          fflush(stdout);
          return EXIT_SUCCESS;
        }}
        '''.format(includes=get_includes(lang), op_name=op.name,
                   typ=typ, year=date.today().year,sizeof=common.sizeof(typ),
                   rand=rand, head=head, scalar_code=scalar_code, comp=comp,
                   vout1_comp= vout1_comp, typ_nsimd=typ_nsimd))
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------
# Entry point

def doit(opts):
    ulps = common.load_ulps_informations(opts)
    common.myprint(opts, 'Generating tests')
    for op_name, operator in operators.operators.items():
        # Skip non-matching tests
        if opts.match and not opts.match.match(op_name):
            continue
        if op_name  in ['if_else1', 'loadu', 'loada', 'storeu', 'storea',
                        'len', 'loadlu', 'loadla', 'storelu', 'storela',
                        'set1', 'store2a', 'store2u', 'store3a', 'store3u',
                        'store4a', 'store4u', 'downcvt', 'to_logical',
                        'mask_for_loop_tail', 'set1l', 'scatter',
                        'scatter_linear']:
            continue
        for typ in operator.types:
            if operator.name in ['notb', 'andb', 'xorb', 'orb', 'andnotb'] and \
               typ == 'f16':
                continue
            elif operator.name == 'nbtrue':
                gen_nbtrue(opts, operator, typ, 'c_base')
                gen_nbtrue(opts, operator, typ, 'cxx_base')
                gen_nbtrue(opts, operator, typ, 'cxx_adv')
            elif operator.name == 'addv':
                if typ in common.ftypes:
                    gen_addv(opts, operator, typ, 'c_base')
                    gen_addv(opts, operator, typ, 'cxx_base')
                    gen_addv(opts, operator, typ, 'cxx_adv')
            elif operator.name == 'adds':
                gen_adds(opts, operator, typ, 'c_base', ulps)
                gen_adds(opts, operator, typ, 'cxx_base', ulps)
                gen_adds(opts, operator, typ, 'cxx_adv', ulps)
            elif operator.name == 'subs':
                gen_subs(opts, operator, typ, 'c_base', ulps)
                gen_subs(opts, operator, typ, 'cxx_base', ulps)
                gen_subs(opts, operator, typ, 'cxx_adv', ulps)
            elif operator.name in ['all', 'any']:
                gen_all_any(opts, operator, typ, 'c_base')
                gen_all_any(opts, operator, typ, 'cxx_base')
                gen_all_any(opts, operator, typ, 'cxx_adv')
            elif operator.name == 'iota':
                gen_iota(opts, operator, typ, 'c_base')
                gen_iota(opts, operator, typ, 'cxx_base')
                gen_iota(opts, operator, typ, 'cxx_adv')
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
                gen_load_store_ravel(opts, operator, typ, 'c_base')
            elif operator.name in ['gather', 'gather_linear']:
                gen_gather_scatter(opts, operator, typ, 'c_base')
                gen_gather_scatter(opts, operator, typ, 'cxx_base')
                gen_gather_scatter(opts, operator, typ, 'cxx_adv')
            elif operator.name == 'mask_scatter':
                gen_mask_scatter(opts, operator, typ, 'c_base')
                gen_mask_scatter(opts, operator, typ, 'cxx_base')
                gen_mask_scatter(opts, operator, typ, 'cxx_adv')
            elif operator.name in ['maskz_gather', 'masko_gather']:
                gen_maskoz_gather(opts, operator, typ, 'c_base')
                gen_maskoz_gather(opts, operator, typ, 'cxx_base')
                gen_maskoz_gather(opts, operator, typ, 'cxx_adv')
            elif operator.name in ['masko_loada1', 'masko_loadu1',
                                   'maskz_loada1', 'maskz_loadu1']:
                gen_mask_load(opts, operator, typ, 'c_base')
                gen_mask_load(opts, operator, typ, 'cxx_base')
                gen_mask_load(opts, operator, typ, 'cxx_adv')
            elif operator.name in ['mask_storea1', 'mask_storeu1']:
                gen_mask_store(opts, operator, typ, 'c_base')
                gen_mask_store(opts, operator, typ, 'cxx_base')
                gen_mask_store(opts, operator, typ, 'cxx_adv')
            elif operator.name == 'reverse':
                gen_reverse(opts, operator, typ, 'c_base');
                gen_reverse(opts, operator, typ, 'cxx_base');
                gen_reverse(opts, operator, typ, 'cxx_adv');
            elif operator.name in ['ziplo', 'ziphi',
                                   'unziplo', 'unziphi']:
                gen_unpack_half(opts, operator, typ, 'c_base')
                gen_unpack_half(opts, operator, typ, 'cxx_base')
                gen_unpack_half(opts, operator, typ, 'cxx_adv')
            elif operator.name in ['zip', 'unzip']:
                gen_unpack(opts, operator, typ, 'c_base')
                gen_unpack(opts, operator, typ, 'cxx_base')
                gen_unpack(opts, operator, typ, 'cxx_adv')
            else:
                gen_test(opts, operator, typ, 'c_base', ulps)
                gen_test(opts, operator, typ, 'cxx_base', ulps)
                gen_test(opts, operator, typ, 'cxx_adv', ulps)
