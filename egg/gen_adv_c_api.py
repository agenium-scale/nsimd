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
import os
import operators

# -----------------------------------------------------------------------------
# Construct C11 types

def get_c11_types(simd_ext):
    ret = ''
    for se in common.simds_deps[simd_ext]:
        ret += '\n\n'.join([
               '''typedef NSIMD_STRUCT nsimd_pack_{typ}_{se} {{
                    nsimd_{se}_v{typ} v;
                  }} nsimd_pack_{typ}_{se};

                  NSIMD_INLINE nsimd_pack_{typ}_{se}
                  nsimd_make_pack_{typ}_{se}(nsimd_{se}_v{typ} v) {{
                    return (nsimd_pack_{typ}_{se}){{ v }};
                  }}'''.format(typ=typ, se=se) for typ in common.types])
        ret += '\n\n'
        ret += '\n\n'.join([
               '''typedef NSIMD_STRUCT nsimd_packl_{typ}_{se} {{
                    nsimd_{se}_vl{typ} v;
                  }} nsimd_packl_{typ}_{se};

                  NSIMD_INLINE nsimd_packl_{typ}_{se}
                  nsimd_make_packl_{typ}_{se}(nsimd_{se}_vl{typ} v) {{
                    return (nsimd_packl_{typ}_{se}){{ v }};
                  }}'''.format(typ=typ, se=se) for typ in common.types])
        for deg in [2, 3, 4]:
            vs = ', '.join(['v{}'.format(i) for i in range(deg)])
            avs = ', '.join(['{{a0.v{}}}'.format(i) for i in range(deg)])
            ret += '\n\n'
            ret += '\n\n'.join([
                   '''typedef NSIMD_STRUCT nsimd_packx{deg}_{typ}_{se} {{
                        nsimd_pack_{typ}_{se} {vs};
                      }} nsimd_packx{deg}_{typ}_{se};

                      NSIMD_INLINE nsimd_packx{deg}_{typ}_{se}
                      nsimd_make_packx{deg}_{typ}_{se}
                      (nsimd_{se}_v{typ}x{deg} a0) {{
                        return (nsimd_packx{deg}_{typ}_{se}){{ {avs} }};
                      }}                      '''. \
                      format(typ=typ, se=se, vs=vs, deg=deg, avs=avs) \
                      for typ in common.types])

    ret += '\n\n'
    ret += '#define nsimd_make_pack(var, func) ' \
           '_Generic(var, \\\n'
    ret += '\n'.join([
           'nsimd_pack_{typ}_{se}: nsimd_make_pack_{typ}_{se}, \\'. \
           format(typ=typ, se=se) for typ in common.types \
                                  for se in common.simds_deps[simd_ext]])
    ret += '\n'
    ret += '\n'.join([
           'nsimd_packl_{typ}_{se}: nsimd_make_packl_{typ}_{se}, \\'. \
           format(typ=typ, se=se) for typ in common.types \
                                  for se in common.simds_deps[simd_ext]])
    ret += '\n'
    ret += '\n'.join([
           'nsimd_packx{d}_{typ}_{se}: nsimd_make_packx{d}_{typ}_{se}, \\'. \
           format(typ=typ, se=se, d=d) for typ in common.types \
                                       for d in [2, 3, 4] \
                                       for se in common.simds_deps[simd_ext]])
    ret += '\ndefault: nsimd_c11_type_unsupported)(func)'

    ret += '\n\n'
    ret += '\n'.join([
           'typedef nsimd_pack_{typ}_{simd_ext} nsimd_pack_{typ};'. \
            format(typ=typ, simd_ext=simd_ext) for typ in common.types])
    ret += '\n\n'
    ret += '\n'.join([
           'typedef nsimd_packl_{typ}_{simd_ext} nsimd_packl_{typ};'. \
            format(typ=typ, simd_ext=simd_ext) for typ in common.types])
    ret += '\n\n'
    ret += '\n'.join([
           'typedef nsimd_packx{d}_{typ}_{simd_ext} nsimd_packx{d}_{typ};'. \
            format(typ=typ, simd_ext=simd_ext, d=d) \
            for typ in common.types for d in [2, 3, 4]])

    ret += '\n\n'
    ret += '#define nsimd_c11_pack(var) _Generic((var), \\\n'
    ret += '\n'.join([
           'nsimd_packl_{typ}_{se}: ' \
           '((nsimd_pack_{typ}_{se} (*)())NULL)(), \\'. \
           format(typ=typ, se=se) for typ in common.types \
                                  for se in common.simds_deps[simd_ext]])
    ret += '\ndefault: NULL)'

    ret += '\n\n'
    ret += '#define nsimd_c11_packl(var) _Generic((var), \\\n'
    ret += '\n'.join([
           'nsimd_pack_{typ}_{se}: ' \
           '((nsimd_packl_{typ}_{se} (*)())NULL)(), \\'. \
           format(typ=typ, se=se) for typ in common.types \
                                  for se in common.simds_deps[simd_ext]])
    ret += '\ndefault: NULL)'

    ret += '\n\n'
    ret += '#define nsimd_c11_packx2(var) _Generic((var), \\\n'
    ret += '\n'.join([
           'nsimd_pack_{typ}_{se}: ' \
           '((nsimd_packx2_{typ}_{se} (*)())NULL)(), \\'. \
           format(typ=typ, se=se) for typ in common.types \
                                  for se in common.simds_deps[simd_ext]])
    ret += '\ndefault: NULL)'

    return ret

# -----------------------------------------------------------------------------
# Construct C11 overloads

def get_c11_overloads(op, simd_ext):
    if common.get_first_discriminating_type(op.params) == -1:
        # Only the len operator should go here
        assert op.name == 'len'
        ret = '\n\n'.join([
        '''#define NSIMD_C11_LEN_nsimd_pack_{typ}_{se}() \\
                   nsimd_len_{se}_{typ}()

           #define NSIMD_C11_LEN_nsimd_packl_{typ}_{se}() \\
                   nsimd_len_{se}_{typ}()

           #define NSIMD_C11_LEN_nsimd_packx2_{typ}_{se}() \\
                   (2 * nsimd_len_{se}_{typ}())

           #define NSIMD_C11_LEN_nsimd_packx3_{typ}_{se}() \\
                   (3 * nsimd_len_{se}_{typ}())

           #define NSIMD_C11_LEN_nsimd_packx4_{typ}_{se}() \\
                   (4 * nsimd_len_{se}_{typ}())'''.format(typ=typ, se=se) \
                   for typ in op.types for se in common.simds_deps[simd_ext]])

        ret += '\n\n'
        ret += '\n\n'.join([
        '''#define NSIMD_C11_LEN_nsimd_pack_{typ}() \\
                   nsimd_len_{simd_ext}_{typ}()

           #define NSIMD_C11_LEN_nsimd_packl_{typ}() \\
                   nsimd_len_{simd_ext}_{typ}()

           #define NSIMD_C11_LEN_nsimd_packx2_{typ}() \\
                   (2 * nsimd_len_{simd_ext}_{typ}())

           #define NSIMD_C11_LEN_nsimd_packx3_{typ}() \\
                   (3 * nsimd_len_{simd_ext}_{typ}())

           #define NSIMD_C11_LEN_nsimd_packx4_{typ}() \\
                   (4 * nsimd_len_{simd_ext}_{typ}())'''. \
                   format(typ=typ, simd_ext=simd_ext) for typ in common.types])
        ret += '\n\n'
        ret += '#define nsimd_len(type) \\\n' \
               'NSIMD_PP_CAT_2(NSIMD_C11_LEN_, type)()\n\n'
        return ret

    def get_c11_arg(param, name):
        if param in ['*', 'c*', 's', 'p']:
            return name
        elif param in ['v', 'l', 'vi']:
            return '({}).v'.format(name)

    args = op.params[1:]
    i0 = common.get_first_discriminating_type(args)
    if i0 == -1:
        if op.params[0] == 'v':
            pack = 'pack'
        elif op.params[0] == 'l':
            pack = 'packl'
        elif op.params[0] == 'vx2':
            pack = 'packx2'
        elif op.params[0] == 'vx3':
            pack = 'packx3'
        elif op.params[0] == 'vx4':
            pack = 'packx4'
        macro_args = ', '.join(['a{}'.format(i) for i in range(len(args))])
        ret = '\n\n'.join([
        '''#define NSIMD_C11_{OP_NAME}_nsimd_{pack}_{typ}_{se}({macro_args}) \\
                     nsimd_make_{pack}_{typ}_{se}( \\
                       nsimd_{op_name}_{se}_{typ}({macro_args}))'''. \
                       format(OP_NAME=op.name.upper(), se=se,
                              macro_args=macro_args,
                              op_name=op.name, typ=typ, pack=pack) \
                              for typ in op.types \
                              for se in common.simds_deps[simd_ext]])
        ret += '\n\n'
        ret += '\n\n'.join([
        '''#define NSIMD_C11_{OP_NAME}_nsimd_{pack}_{typ}({macro_args}) \\
                     nsimd_make_{pack}_{typ}_{simd_ext}( \\
                       nsimd_{op_name}_{simd_ext}_{typ}({macro_args}))'''. \
                       format(OP_NAME=op.name.upper(), simd_ext=simd_ext,
                              macro_args=macro_args, op_name=op.name, typ=typ,
                              pack=pack) for typ in op.types])
        ret += '\n\n'
        type_args = ', '.join(['type'] + \
                              ['a{}'.format(i) for i in range(len(args))])
        call_args = ', '.join([get_c11_arg(args[i], 'a{}'.format(i)) \
                               for i in range(len(args))])
        ret += '\n\n#define nsimd_{op_name}({type_args})' \
               ' NSIMD_PP_CAT_2(NSIMD_C11_{OP_NAME}_, type)({call_args})'. \
               format(op_name=op.name, OP_NAME=op.name.upper(),
                      call_args=call_args, type_args=type_args)
        return ret

    # Getting here means that i0 >= 0 i.e. that overloads can be determined
    # by argument i0 of the operator which is in ['v', 'l', 'vx2', 'vx3',
    # 'vx4']

    macro_args = ['a{}'.format(i) for i in range(len(args))]
    call_args = ', '.join([get_c11_arg(args[i], 'a{}'.format(i)) \
                           for i in range(len(args))])
    if not op.closed:
        macro_args = ['to_type'] + macro_args
    macro_args = ', '.join(macro_args)

    if op.params[0] in ['v', 'l', 'vx2', 'vx3', 'vx4']:
        if not op.closed:
            ret = '#define nsimd_{}({}) ' \
                  'nsimd_make_pack((((to_type (*)())NULL)()), ' \
                  '_Generic(({}), \\\n'. \
                  format(op.name, macro_args, 'a{}'.format(i0))
        else:
            if op.params[0] != args[i0]:
                if op.params[0] == 'v':
                    ctrl_expr = 'nsimd_c11_pack(a{})'.format(i0)
                elif op.params[0] == 'l':
                    ctrl_expr = 'nsimd_c11_packl(a{})'.format(i0)
                elif op.params[0] == 'vx2':
                    ctrl_expr = 'nsimd_c11_packx2(a{})'.format(i0)
            else:
                ctrl_expr = 'a{}'.format(i0)
            ret = '#define nsimd_{}({}) ' \
                  'nsimd_make_pack({}, _Generic(({}), \\\n'. \
                  format(op.name, macro_args, ctrl_expr, 'a{}'.format(i0))
    else:
        ret = '#define nsimd_{}({}) _Generic(({}), \\\n'. \
              format(op.name, macro_args, 'a{}'.format(i0))

    suf = { 'v': '', 'l': 'l', 'vx2': 'x2', 'vx3': 'x3', 'vx4': 'x4'}

    arg = args[i0]
    typ_fmt = 'nsimd_pack{}_{{}}_{{}}'.format(suf[arg])

    for se in common.simds_deps[simd_ext]:
        for typ in op.types:
            ret += typ_fmt.format(typ, se) + ': '
            if op.closed:
                ret += 'nsimd_{}_{}_{}, \\\n'.format(op.name, se, typ)
                continue
            ret += '_Generic(((to_type (*)())NULL)(), \\\n'
            for to_typ in common.get_output_types(typ, op.output_to):
                to_pack = 'nsimd_pack{}_{}_{}'. \
                          format(suf[op.params[0]], to_typ, se)
                ret += '  {}: nsimd_{}_{}_{}_{}, \\\n'. \
                       format(to_pack, op.name, se, to_typ, typ)
            ret += '  default: nsimd_c11_type_unsupported), \\\n'

    ret += 'default: nsimd_c11_type_unsupported)({})'.format(call_args)
    if op.params[0] in ['v', 'l', 'vx2', 'vx3', 'vx4']:
        ret += ')'
    return ret

# -----------------------------------------------------------------------------

def doit(opts):
    common.myprint(opts, 'Generating advanced C API (requires C11)')
    filename = os.path.join(opts.include_dir, 'c_adv_api_functions.h')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as out:
        out.write('''#ifndef NSIMD_C_ADV_API_FUNCTIONS_H
                     #define NSIMD_C_ADV_API_FUNCTIONS_H

                     #include <nsimd/nsimd.h>

                     ''')

        for simd_ext in common.simds:
            out.write('''{hbar}
                         {hbar}
                         {hbar}

                         /* {SIMD_EXT} */

                         {hbar}
                         {hbar}
                         {hbar}

                         #ifdef NSIMD_{SIMD_EXT}

                         {types}

                         '''.format(hbar=common.hbar,
                                    types=get_c11_types(simd_ext),
                                    SIMD_EXT=simd_ext.upper()))

            for op_name, operator in operators.operators.items():
                out.write('/* {} */\n\n{}\n\n'. \
                          format(op_name, get_c11_overloads(operator,
                                                            simd_ext)))

            out.write('\n\n#endif')

        out.write('\n\n{}\n\n#endif\n'.format(common.hbar))

