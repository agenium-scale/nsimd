# Use utf-8 encoding
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Agenium Scale
#
# permission is hereby granted, free of charge, to any person obtaining a copy
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

# -----------------------------------------------------------------------------

# What does this script?
# ----------------------
#
# This is only a python module that holds what is shared by `generate.py`,
# the `platform_*.py` files and all other python code in `egg`. If contains
# the list of supported types, functions, operators, and some useful helper
# functions such as the python equivalent of `mkdir -p`.

# -----------------------------------------------------------------------------
# Import section

import math
import os
import sys
import io
import collections
import platform
import string

# -----------------------------------------------------------------------------
# print

def myprint(opts, obj):
    if opts.list_files:
        return
    print('-- {}'.format(obj))

# -----------------------------------------------------------------------------
# check if file exists

def can_create_filename(opts, filename):
    if opts.list_files:
        print(filename)
        return False
    if opts.verbose:
        sys.stdout.write('-- {}: '.format(filename))
    if os.path.isfile(filename) and not opts.force:
        if opts.verbose:
            sys.stdout.write('skipping\n')
        return False
    elif opts.force:
        if opts.verbose:
            sys.stdout.write('creating (forced)\n')
        return True
    else:
        if opts.verbose:
            sys.stdout.write('creating (missing)\n')
        return True

# -----------------------------------------------------------------------------
# open with UTF8 encoding

def open_utf8(opts, filename):
    dummy, ext = os.path.splitext(filename)
    if ext.lower() in ['.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx',
                       '.hpp']:
        begin_comment = '/*'
        end_comment = '*/'
    elif ext.lower() in ['.md', '.htm', '.html']:
        begin_comment = '<!--'
        end_comment = '-->'
    else:
        begin_comment = None
    with io.open(filename, mode='w', encoding='utf-8') as fout:
        if begin_comment is not None:
            if opts.simple_license:
                fout.write('''{}

Copyright (c) 2020 Agenium Scale

{}

'''.format(begin_comment, end_comment))
            else:
                fout.write('''{}

Copyright (c) 2020 Agenium Scale

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

{}

'''.format(begin_comment, end_comment))

        fout.write('{} This file has been auto-generated {}\n\n'.\
            format(begin_comment, end_comment))

    return io.open(filename, mode='a', encoding='utf-8')

# -----------------------------------------------------------------------------
# clang-format

def clang_format(opts, filename, cuda=False):
    if opts.enable_clang_format:
        # We add a newline at EOF as required by compilers
        with io.open(filename, mode='a', encoding='utf-8') as fout:
            fout.write('\n')
        return
    # TODO: not sure if needed to implement a smarter call to clang-format
    if cuda:
        os.system('clang-format -style="{{ Standard: Cpp11 }}" -i {}'. \
                  format(filename))
    else:
        os.system('clang-format -style="{{ Standard: Cpp03 }}" -i {}'. \
                  format(filename))
    with open(filename, 'a') as fout:
        fout.write('\n')

# -----------------------------------------------------------------------------
# Not implemented response

NOT_IMPLEMENTED = 'abort();'

# -----------------------------------------------------------------------------
# C/C++ comment hbar

hbar = '/* ' + ('-' * 73) + ' */'

# -----------------------------------------------------------------------------
# Convert constants for operators

OUTPUT_TO_SAME_TYPE       = 0
OUTPUT_TO_SAME_SIZE_TYPES = 1
OUTPUT_TO_UP_TYPES        = 2
OUTPUT_TO_DOWN_TYPES      = 3

# -----------------------------------------------------------------------------
# SIMD type

x86_simds = [
    'sse2',
    'sse42',
    'avx',
    'avx2',
    'avx512_knl',
    'avx512_skylake',
]

arm_simds = [
    'neon128',
    'aarch64',
    'sve',
    'sve128',
    'sve256',
    'sve512',
    'sve1024',
    'sve2048'
]

ppc_simds = [
    'power7',
    'power8',
]

simds = ['cpu'] + x86_simds + arm_simds + ppc_simds

simds_deps = {
    'cpu': ['cpu'],
    'sse2': ['cpu', 'sse2'],
    'sse42': ['cpu', 'sse2', 'sse42'],
    'avx': ['cpu', 'sse2', 'sse42', 'avx'],
    'avx2': ['cpu', 'sse2', 'sse42', 'avx', 'avx2'],
    'fma4': [],
    'avx512_knl': ['cpu', 'sse2', 'sse42', 'avx', 'avx2', 'avx512_knl'],
    'avx512_skylake': ['cpu', 'sse2', 'sse42', 'avx', 'avx2', 'avx512_skylake'],
    'neon128': ['cpu', 'neon128'],
    'aarch64': ['cpu', 'aarch64'],
    'sve': ['cpu', 'aarch64', 'sve'],
    'sve128': ['cpu', 'aarch64', 'sve128'],
    'sve256': ['cpu', 'aarch64', 'sve256'],
    'sve512': ['cpu', 'aarch64', 'sve512'],
    'sve1024': ['cpu', 'aarch64', 'sve1024'],
    'sve2048': ['cpu', 'aarch64', 'sve2048'],
    'power7': ['cpu', 'power7'],
    'power8': ['cpu', 'power8']
}

ftypes = ['f64', 'f32', 'f16']
ftypes_no_f16 = ['f64', 'f32']
itypes = ['i64', 'i32', 'i16', 'i8']
utypes = ['u64', 'u32', 'u16', 'u8']
iutypes = itypes + utypes
types = ftypes + iutypes

def logical(typ):
    return 'l{}'.format(typ)

signed_type = {
    'i8': 'i8',
    'u8': 'i8',
    'i16': 'i16',
    'u16': 'i16',
    'i32': 'i32',
    'u32': 'i32',
    'i64': 'i64',
    'u64': 'i64',
    'f16': 'f16',
    'f32': 'f32',
    'f64': 'f64'
}

bitfield_type = {
    'i8': 'u8',
    'u8': 'u8',
    'i16': 'u16',
    'u16': 'u16',
    'i32': 'u32',
    'u32': 'u32',
    'i64': 'u64',
    'u64': 'u64',
    'f16': 'u16',
    'f32': 'u32',
    'f64': 'u64'
}

in0 = 'a0'
in1 = 'a1'
in2 = 'a2'
in3 = 'a3'
in4 = 'a4'
in5 = 'a5'

CPU_NBITS = 64

if CPU_NBITS != 64 and CPU_NBITS != 128:
    raise ValueError('CPU_NBITS must be 64 or 128')

def get_args(n):
    fmtspec = { 'in0': in0, 'in1': in1, 'in2': in2, 'in3': in3, 'in4': in4,
                'in5': in5 }
    return ', '.join(['{{in{}}}'.format(i).format(**fmtspec) \
                      for i in range(0, n)])

def get_simds_deps_from_opts(opts):
    simds = set()
    for simd1 in opts.simd:
        for simd2 in simds_deps[simd1]:
            simds.add(simd2)
    return simds

def bitsize(typ):
    if not (typ in types):
        raise ValueError('Unknown type "{}"'.format(typ))
    return int(typ[1:])

def sizeof(typ):
    return bitsize(typ) // 8

def ilog2(x):
    if x <= 0:
        return None
    for i in range(0, x):
        if 2 ** (i + 1) > x:
            return i

#def get_same_size_types(typ):
#    nbits = typ[1:]
#    if typ in ['i8' ,'u8']:
#        return ['i8', 'u8']
#    else:
#        return ['i' + nbits, 'u' + nbits, 'f' + nbits]

def get_output_types(from_typ, output_to):
    if output_to == OUTPUT_TO_SAME_TYPE:
        return [from_typ]
    else:
        nbits = from_typ[1:]
        if output_to == OUTPUT_TO_SAME_SIZE_TYPES:
            if from_typ in ['i8' ,'u8']:
                return ['i8', 'u8']
            else:
                return ['i' + nbits, 'u' + nbits, 'f' + nbits]
        elif output_to == OUTPUT_TO_UP_TYPES:
            if nbits == '64':
                raise ValueError('No uptype for ' + from_typ)
            else:
                n = str(int(nbits) * 2)
                return ['i' + n, 'u' + n, 'f' + n]
        elif output_to == OUTPUT_TO_DOWN_TYPES:
            n = str(int(nbits) // 2)
            if nbits == '8':
                raise ValueError('No downtype for ' + from_typ)
            elif nbits == '16':
                return ['i' + n, 'u' + n]
            else:
                return ['i' + n, 'u' + n, 'f' + n]
        else:
            raise ValueError('Invalid argument for "output_to": {}'. \
                             format(output_to))

# -----------------------------------------------------------------------------
# mkdir -p (avoid a dependency for just one function)

def mkdir_p(path):
    if os.path.isdir(path):
        return path
    head, tail = os.path.split(path)
    if head != '':
        mkdir_p(head)
    os.mkdir(path)
    return path

# -----------------------------------------------------------------------------
# Replacement of enumerate

def enum(l):
    ret = []
    for i in range(0, len(l)):
        ret.append([i, l[i]])
    return ret

# -----------------------------------------------------------------------------
# List of supported SIMD operators/functions

# v   = SIMD vector parameter
# vx2 = struct of 2 SIMD vector parameters
# vx3 = struct of 3 SIMD vector parameters
# vx4 = struct of 4 SIMD vector parameters
# l   = SIMD vector of logicals parameter
# s   = Scalar parameter
# *   = Pointer to scalar parameter
# c*  = Pointer to const scalar parameter
# _   = void (only for return type)
# p   = Parameter (int)


# -----------------------------------------------------------------------------
# Type generators

def get_one_type_generic(param, typ):
    if param == '_':
        return 'void'
    elif param == 'p':
        return 'int'
    elif param == 's':
        return typ
    elif param == '*':
        return '{}*'.format(typ)
    elif param == 'c*':
        return '{} const*'.format(typ)
    elif param == 'v':
        return 'v{}'.format(typ)
    elif param == 'vx2':
        return 'v{}x2'.format(typ)
    elif param == 'vx3':
        return 'v{}x3'.format(typ)
    elif param == 'vx4':
        return 'v{}x4'.format(typ)
    elif param == 'l':
        return 'vl{}'.format(typ)
    else:
        raise ValueError("Unknown param '{}'".format(param))

def get_one_type_specific(param, ext, typ):
    if param == '_':
        return 'void'
    elif param == 'p':
        return 'int'
    elif param == 's':
        return typ
    elif param == '*':
        return '{}*'.format(typ)
    elif param == 'c*':
        return '{} const*'.format(typ)
    elif param == 'v':
        return 'nsimd_{}_v{}'.format(ext, typ)
    elif param == 'vx2':
        return 'nsimd_{}_v{}x2'.format(ext, typ)
    elif param == 'vx3':
        return 'nsimd_{}_v{}x3'.format(ext, typ)
    elif param == 'vx4':
        return 'nsimd_{}_v{}x4'.format(ext, typ)
    elif param == 'l':
        return 'nsimd_{}_vl{}'.format(ext, typ)
    else:
        raise ValueError("Unknown param '{}'".format(param))

def get_one_type_pack(param, inout, N):
    if param == '_':
        return 'void'
    if param == 'p':
        return 'int'
    if param == '*':
        return 'T*'
    if param == 'c*':
        return 'T const*'
    if param == 's':
        return 'T'
    if param.startswith('v'):
        if inout == 0:
            return 'pack<T, {}, SimdExt> const&'.format(N)
        else:
            return 'pack<T, {}, SimdExt>'.format(N)
    if param == 'l':
        if inout == 0:
            return 'packl<T, {}, SimdExt> const&'.format(N)
        else:
            return 'packl<T, {}, SimdExt>'.format(N)
    raise ValueError("Unknown param '{}'".format(param))

def get_one_type_generic_adv_cxx(param, T, N):
    if param == '_':
        return 'void'
    elif param == 'p':
        return 'int'
    elif param == '*':
        return '{}*'.format(T)
    elif param == 'c*':
        return '{} const*'.format(T)
    elif param == 's':
        return T
    elif param == 'v':
        return 'pack<{}, {}, SimdExt>'.format(T, N)
    elif param == 'vx2':
        return 'packx2<{}, {}, SimdExt>'.format(T, N)
    elif param == 'vx3':
        return 'packx3<{}, {}, SimdExt>'.format(T, N)
    elif param == 'vx4':
        return 'packx4<{}, {}, SimdExt>'.format(T, N)
    elif param == 'l':
        return 'packl<{}, {}, SimdExt>'.format(T, N)
    else:
        raise ValueError('Unknown param: "{}"'.format(param))

def get_one_type_scalar(param, t):
    if param == '_':
        return 'void'
    elif param in ['p', 'l']:
        return 'int'
    elif param in ['s', 'v']:
        return t
    else:
        raise ValueError('Unknown param: "{}"'.format(param))

# -----------------------------------------------------------------------------
# Formats

def pprint_lines(what):
    return '\n'.join(what)

def pprint_commas(what):
    return ', '.join(what)

def pprint_includes(what):
    return pprint_lines('#include {}'.format(i) for i in what)

# -----------------------------------------------------------------------------
# Function parsing signatures

def parse_signature(signature):
    l = signature.split(' ');
    name = l[1]
    if len(l) > 2:
        params = [l[0]] + l[2:]
    else:
        params = [l[0]]

    return (name, params)

# -----------------------------------------------------------------------------
# Load platforms

def get_platforms(opts):
    ret = dict()
    path = opts.script_dir
    myprint(opts, 'Searching platforms in "{}"'.format(path))
    for mod_file in os.listdir(path):
        if mod_file[-3:] == '.py' and mod_file[0:9] == 'platform_':
            mod_name = mod_file[:-3]
            myprint(opts, 'Found new platform: {}'.format(mod_name[9:]))
            ret[mod_name[9:]] = __import__(mod_name)
    return ret

# -----------------------------------------------------------------------------
# Find modules

def get_modules(opts):
    if opts.modules_list != None:
        return opts.modules_list
    ret = dict()
    # We have one module by directory
    path = os.path.join(opts.script_dir, 'modules')
    myprint(opts, 'Searching modules in "{}"'.format(path))
    for module_dir in os.listdir(path):
        if (not os.path.isdir(os.path.join(path, module_dir))) or \
           module_dir == '.' or module_dir == '..' or \
           (not os.path.exists(os.path.join(path, module_dir, 'hatch.py'))):
            continue
        myprint(opts, 'Found new module: {}'.format(module_dir))
        mod = __import__('modules.{}.hatch'.format(module_dir))
        ret[module_dir] = mod
    opts.modules_list = ret
    return ret

# -----------------------------------------------------------------------------
# Ulps

import json

def load_ulps_informations(opts):
    path = opts.script_dir
    filename = os.path.join(path, "ulp.json")
    with open(filename) as data_file:
        data = json.load(data_file)

    ulps = dict()
    for info in data["ulps"]:
        type = info["type"]
        func = info["func"]
        if not func in ulps:
            ulps[func] = dict()
        ulps[func][type] = info

    return ulps

# -----------------------------------------------------------------------------
# Domain stuff

class MathSet(object):
    @property
    def mul(self):
        return self.mul_

    @property
    def add(self):
        return self.add_

    @property
    def variable(self):
        return self.variable_

    @property
    def natural(self):
        return self.natural_

    def parse_sum(self, fstring):
        npos = fstring.find('n')
        zpos = fstring.find('z')
        if npos>=0 or zpos >=0:
            if npos>=0:
                self.natural_ = True
            else:
                self.natural_ = False

            if fstring.find('-')>=0:
                self.mul_ = -self.mul_
        else:
            self.add_ = float(fstring)

    def __init__(self, fstring):
        npos = fstring.find('n')
        zpos = fstring.find('z')
        if npos < 0 and zpos < 0:
            self.mul_ = 1.
            self.add_ = float(fstring)
            self.variable_ = False
            self.natural_ = False
        else:
            self.variable_ = True
            self.mul_ = 1.
            self.add_ = 0.
            self.natural_ = False

            product = fstring.split('*')

            for part in product:
                npos = part.find('n')
                zpos = part.find('z')
                if npos >= 0 or zpos >= 0:
                    sumstring = part.split('+')
                    if len(sumstring) > 1:
                        self.parse_sum(sumstring[0][1:])
                        self.parse_sum(sumstring[1][:-1])
                    else:
                        self.parse_sum(sumstring[0])
                else:
                    if part == 'pi':
                        self.mul_ = math.pi * self.mul_
                    else:
                        self.mul_ = float(part) * self.mul_

    def __str__ (self):
        if self.variable:
            if self.natural_:
                var = 'n'
                set_ = 'ℕ'
            else:
                var = 'z'
                set_ = 'ℤ'

            if self.add_ != 0:
                add='({}+{:g})'.format(var,self.add_)
            else:
                add='{}'.format(var)

            if self.mul_ == math.pi:
                mul = 'π⋅'
            elif abs(self.mul_) == 1:
                mul = ''
            else:
                mul = '{:g}⋅'.format(abs(self.mul_))

            sign = ''
            if self.mul_ < 0:
                sign = '-'

            return '{var}∈{set_}:{sign}{mul}{add}'.format(var=var, set_=set_,
                    sign=sign, add=add, mul=mul)
        else:
            return '{:g}'.format(self.add)

# -----------------------------------------------------------------------------
# Class representing an interval, used to define function domains

class Interval(object):

    @property
    def left(self):
        return self.left_

    @property
    def right(self):
        return self.right_

    @property
    def open_left(self):
        return self.open_left_

    @property
    def open_right(self):
        return self.open_right_

    @property
    def removed(self):
        return self.removed_

    def code_for(self, value, typ):
        is_fp = typ == 'f32' or typ == 'f64'
        if value == float('-Inf'):
            if is_fp:
                return '-std::numeric_limits<{}>::infinity()'.format(typ)
            else:
                return 'std::numeric_limits<{}>::min()'.format(typ)
        elif value == float('Inf'):
            if is_fp:
                return 'std::numeric_limits<{}>::infinity()'.format(typ)
            else:
                return 'std::numeric_limits<{}>::max()'.format(typ)
        else:
            return value

    def code_left(self, typ):
        return self.code_for(self.left_, typ)

    def code_right(self, typ):
        return self.code_for(self.right_, typ)

    # Parse the part before the '-' in the interval
    # For instance, '(0,1)' in '(0,1)-{0.5}'
    def parse_first_part(self,fstring):
        real_ = True
        fstring = fstring
        if fstring[0] == 'R':
            self.open_left_ = True
            self.open_right_ = True
            self.left_ = float('-Inf')
            self.right_ = float('Inf')
            self.real_ = True
            return
        if fstring[0] == 'B':
            self.open_left_ = False
            self.open_right_ = False
            self.left_ = 0
            self.right_ = 1
            self.logical_ = True
            return
        if fstring[0] == 'N':
            self.open_left_ = True
            self.open_right_ = True
            self.left_ = float('0')
            self.right_ = float('Inf')
            self.natural_ = True
            return
        if fstring[0] == 'Z':
            self.open_left_ = True
            self.open_right_ = True
            self.left_ = float('-Inf')
            self.right_ = float('Inf')
            self.natural_ = True
            return
        elif fstring[0] == '(':
            self.open_left_ = True
        elif fstring[0] == '[':
            self.open_left_ = False
        else:
            raise ValueError('Error in format string : "{}"'.format(fstring))

        self.real_ = True

        length = len(fstring)

        if fstring[length-1] == ')':
            self.open_right_ = True
        elif fstring[length-1] == ']':
            self.open_right_ = False
        else:
            raise ValueError('Error in format string : "{}"'.format(fstring))

        numbers = fstring[1:length-1].split(',')

        if len(numbers) != 2:
            raise ValueError('Error in format string : "{}"'.format(fstring))

        self.left_ = float(numbers[0])
        self.right_ = float(numbers[1])

    def parse_second_part(self, fstring):
        for removed in fstring.split(','):
            self.removed.append(MathSet(removed))


    def __init__(self, fstring):
        self.left_ = -float('Inf')
        self.right_ = float('Inf')
        self.open_left_ = True
        self.open_right_ = True

        self.real_ = False
        self.natural_ = False
        self.logical_ = False

        self.removed_ = []

        split = fstring.find('\{')

        if split < 0:
            self.parse_first_part(fstring);
        else:
            first_part = fstring[0:split]
            scd_part = fstring[split+2:-1]

            self.parse_first_part(first_part);
            if split > 0:
                self.parse_second_part(scd_part)

    def __str__(self):
        ret = ''
        if self.real_:
            if self.open_left:
                open_left = '('
            else:
                open_left = '['

            if self.open_right:
                open_right = ')'
            else:
                open_right = ']'

            all_r = True
            if self.left == -float('inf'):
                left = '-∞'
            else:
                left = '{:g}'.format(self.left)
                all_r = False

            if self.right == float('inf'):
                right = '+∞'
            else:
                right = '{:g}'.format(self.right)
                all_r = False

            if all_r:
                ret = 'ℝ'
            else:
                ret = '{}{}, {}{}'.format(open_left, left, right, open_right)
        elif self.natural_:
            if self.left == -float('inf'):
                ret = 'ℤ'
            else:
                ret = 'ℕ'
        elif self.logical_:
            ret = '𝔹'
        else:
            raise ValueError ('Trying to print invalid interval')

        if self.removed_:
            ret += '∖\\{'
            comma = ''
            for removed in self.removed_:
                ret+=comma+str(removed)
                comma = ', '
            ret += '\\}'

        return ret

    def code(self, typ):
        left = self.code_left(typ)
        right = self.code_right(typ)
        if len(self.removed):
            excluded = []
            for r in self.removed:
                if r.variable:
                    ## TODO:
                    pass
                else:
                    excluded.append(r.add)
            if len(excluded):
                exclude = ' || '.join('r == {}({})'. \
                                      format(typ, e) for e in excluded)
            else:
                exclude = 'false'
            return '''
            while (true) {{
                {type} r = nsimd::benches::rand<{type}>({min}, {max});
                if ({exclude}) {{
                    continue;
                }} else {{
                    return r;
                }}
            }}
            '''.format(type=typ, exclude=exclude, min=left, max=right)
        else:
            return 'return nsimd::benches::rand<{type}>({min}, {max});'. \
                   format(type=typ, min=left, max=right)

# -----------------------------------------------------------------------------
# Class representing a function domain

class Domain(object):

    def __init__(self, str_list):
        self.intervals_ = []

        # Remove spaces in str_list
        str_list = str_list.replace(' ','')

        # 0 dimension
        if not str_list:
            return

        dimensions_string = str_list.split('x')

        for union_string in dimensions_string:
            interval_string = union_string.split('U')

            current = []
            for interval in interval_string:
                try:
                    current.append(Interval(interval))
                except ValueError as v:
                    raise ValueError( \
                          '{}\nEncountered while parsing domain {}'. \
                          format(v, str_list))

            self.intervals_.append(current)

    def __str__(self):
        ret = ''

        product = ''
        for union in self.intervals_:
            u = ''
            ret += product
            if len(union) > 1 and len(self.intervals_) > 1:
                ret += '('
            for interval in union:
                ret += '{}{}'.format(u, interval)
                u = ' ⋃ '
            if len(union) > 1 and len(self.intervals_) > 1:
                ret += ')'

            product = ' × '

        return '$' + ret + '$'

    @property
    def intervals(self):
        return self.intervals_

    @property
    def ndims(self):
        return len(self.intervals_)

    def code(self, prefix_fun_name, typ):
        code = ''
        for i, unions in enumerate(self.intervals):
            nunions = len(unions)
            if nunions == 1:
                nested_code = unions[0].code(typ)
            else:
                cases = []
                for j, union in enumerate(unions):
                    cases.append('case {n}: {{ {code} }};'. \
                                 format(n=j, code=union.code(typ)))
                nested_code = '''
                // Branch to one of the nested interval (union)
                switch (rand() % {nunions}) {{
                    {cases}
                    default:
                        // SHOULD NEVER HAPPEN! This removes compiler warning!
                        return {type}();
                }}
                '''.format(cases='\n'.join(cases), nunions=nunions, type=typ)
            code += '''
            {type} {prefix}{n}() {{
                {code}
            }}

            '''.format(type=typ, prefix=prefix_fun_name, n=i, code=nested_code)
        return code

    def gen_rand(self, typ):
        typlen = typ[1:]
        ret = ''

        if typ[0] in ('i', 'u'):
            #TODO: check that random number is in the function domain
            for u, union in enumerate(self.intervals):
                ret += \
                    '''{typ} rand{u}() {{
                            nsimd_nat i, r;
                            u8 *alias;
                            {typ} ret;
                            (void)i;
                            (void)alias;
                            (void)r;
                            '''.format(u=u+1, typ=typ)

                for i, interval in enumerate(union):
                    if interval.logical_:
                        ret += 'ret = (u8)(rand()) % 2;'
                    else:
                        if not interval.removed:
                            test='0'
                        else:
                            test = '||\n'. \
                                join(['ret == {}'.format(removed) \
                                for removed in interval.removed])

                        ret += \
                         '''do {{
                                alias = (u8*)(&ret);
                                for(i=0, r=rand(); i<(r%{it})+1; ++i) {{
                                    alias[i] = (u8)(rand() & 0xFF);
                                }}
                                for(;i<{it}; ++i) {{
                                    alias[i] = 0u;
                                }}
                            }} while ({test});
                            '''.format(test=test, it=int(typlen)//8)

                ret += 'return ret;}'
        elif typ in ftypes:
            #TODO: check that random number is in the function domain
            for u, union in enumerate(self.intervals):
                ret += \
                    '''{typ} rand{u}() {{
                            nsimd_nat i;
                            u8 *alias;
                            {typ} ret;
                            (void)i;
                            (void)alias;
                            '''.format(u=u+1, typ=typ)

                for i, interval in enumerate(union):
                    if interval.logical_:
                        if typ == 'f16':
                            ret += '''u16 tmp = ((u16)rand()%2);
                                      memcpy((void *)&ret, (void *)&tmp,
                                             sizeof(ret));'''
                        else:
                            ret += 'ret = ({})(rand()%2);'.format(typ)
                    else:
                        ret += \
                         '''alias = (u8*)(&ret);
                            for(i=0; i<{it}; ++i) {{
                                alias[i] = (u8)(rand() & 0xFF);
                            }}
                            '''.format(it=int(typlen)//8)

                ret += 'return ret;}'

        return ret


# -----------------------------------------------------------------------------
# Sleef

sleef_types = [
        'f32',
        'f64',
        ]

sleef_simds = [
        'sse2',
        'sse42',
        'avx',
        'fma4',
        'avx2',
        ]

def sleef_support_type(simd, typ):
    ## NEON128 only supports 32bit floating points
    if simd == 'neon128' and typ == 'f64':
        return False
    ## No f16 support + No integer supports
    return not (typ == 'f16' or typ in itypes or typ in utypes)

def sleef_name(name, simd, typ, ulp=None):
    ## Sleef mangling:
    '''
    1. Function name in math.h
    2. Data type of vector element
    3. Number of elements in a vector
    4. [Accuracy for typical input domain]
    5. Vector extension
    '''
    ## Filter
    if not sleef_support_type(simd, typ):
        return None
    ## Craft it
    ## 1.
    name = 'Sleef_' + name
    ## 2. + 3.
    types_cpu = {
            'f32': 'f',
            'f64': '',
            }
    types_128 = {
            'f32': 'f4',
            'f64': 'd2',
            }
    types_256 = {
            'f32': 'f8',
            'f64': 'd4',
            }
    types_512 = {
            'f32': 'f16',
            'f64': 'd8',
            }
    types_unknown = {
            'f32': 'fx',
            'f64': 'dx',
            }
    name += ({
        'cpu': types_cpu,
        'sse2': types_128,
        'sse42': types_128,
        'avx': types_256,
        'fma4': types_256,
        'avx2': types_256,
        'avx512_knl': types_512,
        'avx512_skylake': types_512,
        'neon128': types_128,
        'aarch64': types_128,
        'sve': types_unknown,
        'sve128': types_unknown,
        'sve256': types_unknown,
        'sve512': types_unknown,
        'sve1024': types_unknown,
        'sve2048': types_unknown,
        'power7': types_128,
        'power8': types_128
    })[simd][typ]
    ## 4. (We cannot really guess that...
    ##     Instead you have to add bench manually)
    if ulp is not None:
        name += '_u{}'.format(ulp)
    ## 5. (Translate or use `simd` directly)
    if simd != 'cpu':
        ## Careful of the extra _
        if ulp is None:
            name += '_'
        name += ({
            'sse42': 'sse4',
            'avx512_knl': 'avx512f',
            'avx512_skylake': 'avx512f',
            'neon128': '',
            'aarch64': 'advsimd',
        }).get(simd, simd)
    return name

# -----------------------------------------------------------------------------
# Integer limits per type using macros defined in <limits.h> or <climits>

limits = {
    'i8':   {'min': 'NSIMD_I8_MIN',     'max': 'NSIMD_I8_MAX'   },
    'i16':  {'min': 'NSIMD_I16_MIN',    'max': 'NSIMD_I16_MAX'  },
    'i32':  {'min': 'NSIMD_I32_MIN',    'max': 'NSIMD_I32_MAX'  },
    'i64':  {'min': 'NSIMD_I64_MIN',    'max': 'NSIMD_I64_MAX'  },
    'u8':   {'min': 'NSIMD_U8_MIN',     'max': 'NSIMD_U8_MAX'   },
    'u16':  {'min': 'NSIMD_U16_MIN',    'max': 'NSIMD_U16_MAX'  },
    'u32':  {'min': 'NSIMD_U32_MIN',    'max': 'NSIMD_U32_MAX'  },
    'u64':  {'min': 'NSIMD_U64_MIN',    'max': 'NSIMD_U64_MAX'  }
  }

# -----------------------------------------------------------------------------
# Misc

def ext_from_lang(lang):
    return 'c' if lang == 'c_base' else 'cpp'

def nsimd_category(category):
    return 'nsimd_' + category

# ------------------------------------------------------------------------------
# Doc common

doc_header = '''\
<!DOCTYPE html>

<html>
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>{}</title>
    <style type=\"text/css\">
      body {{
        /*margin:40px auto;*/
        margin:10px auto;
        /*max-width:650px;*/
        max-width:800px;
        /*line-height:1.6;*/
        line-height:1.4;
        /*font-size:18px;*/
        color:#444;
        padding:0 10px
      }}
      h1,h2,h3 {{
        line-height:1.2
      }}
      table,th, td {{
        border: 1px solid gray;
        border-collapse : collapse;
        padding: 1px 3px;
      }}
    </style>
    <!-- https://www.mathjax.org/#gettingstarted -->
    <script src=\"assets/polyfill.min.js\"></script>
    <script id=\"MathJax-script\" async src=\"assets/tex-mml-chtml.js\"></script>
    <!-- Highlight.js -->
    <link rel=\"stylesheet\" href= \"assets/highlight.js.default.min.css\">
    <script src=\"assets/highlight.min.js\"></script>
    <script src=\"assets/cpp.min.js\"></script>
    <script>hljs.initHighlightingOnLoad();</script>
  </head>
<body>

<div style="text-align: center; margin-bottom: 1em;">
  <img src=\"img/logo.svg\">
  <hr>
</div>
<div style="text-align: center; margin-bottom: 1em;">
  <b>NSIMD documentation</b>
</div>
<div style="text-align: center; margin-bottom: 1em;">
  <a href=\"index.html\">Index</a> |
  <a href=\"quick_start.html\">Quick Start</a> |
  <a href=\"tutorials.html\">Tutorials</a> |
  <a href=\"faq.html\">FAQ</a> |
  <a href=\"contribute.html\">Contribute</a> |
  <a href=\"overview.html\">API overview</a> |
  <a href=\"api.html\">API reference</a> |
  <a href=\"modules.html\">Modules</a>
  <hr>
</div>
{}
'''

doc_footer = '''\
  </body>
</html>
'''

def to_filename(op_name):
    valid = string.ascii_letters + string.digits
    ret = ''
    for c in op_name:
        ret += '-' if c not in valid else c
    return ret

def get_html_header(opts, title, filename):
    # check if filename is part of a module doc
    for mod in opts.modules_list:
        if filename.startswith('module_{}_'.format(mod)):
            links = eval('opts.modules_list[mod].{}.hatch.doc_menu()'. \
                         format(mod))
            name = eval('opts.modules_list[mod].{}.hatch.name()'.format(mod))
            html = '<div style="text-align: center; margin-bottom: 1em;">\n'
            html += '<b>{} module documentation</b>\n'.format(name)
            if len(links) > 0:
                html += '</div>\n'
                html += \
                '<div style="text-align: center; margin-bottom: 1em;">\n'
                html += ' | '.join(['<a href=\"module_{}_{}.html\">{}</a>'. \
                                    format(mod, href, label) \
                                    for label, href in links.items()])
            html += '\n<hr>\n</div>\n'
            return doc_header.format(title, html)
    return doc_header.format(title, '')

def get_html_footer():
    return doc_footer

def get_markdown_dir(opts):
    return os.path.join(opts.script_dir, '..', 'doc', 'markdown')

def get_markdown_api_file(opts, name, module=''):
    root = get_markdown_dir(opts)
    op_name = to_filename(name)
    if module == '':
        return os.path.join(root, 'api_{}.md'.format(op_name))
    else:
        return os.path.join(root, 'module_{}_api_{}.md'.format(module, op_name))

def get_markdown_file(opts, name, module=''):
    root =  get_markdown_dir(opts)
    op_name = to_filename(name)
    if module == '':
        return os.path.join(root, '{}.md'.format(op_name))
    else:
        return os.path.join(root, 'module_{}_{}.md'.format(module, op_name))

def get_html_dir(opts):
    return os.path.join(opts.script_dir, '..', 'doc', 'html')

def get_html_api_file(opts, name, module=''):
    root = get_html_dir(opts)
    op_name = to_filename(name)
    if module == '':
        return os.path.join(root, 'api_{}.html'.format(op_name))
    else:
        return os.path.join(root, 'module_{}_api_{}.html'.format(module, op_name))

def get_html_file(opts, name, module=''):
    root = get_html_dir(opts)
    op_name = to_filename(name)
    if module == '':
        return os.path.join(root, '{}.html'.format(op_name))
    else:
        return os.path.join(root, 'module_{}_{}.html'.format(module, op_name))

def gen_doc_html(opts, title):
    if not opts.list_files:
        # check if md2html exists
        md2html = 'md2html.exe' if platform.system() == 'Windows' \
                                else 'md2html'
        doc_dir = os.path.join(opts.script_dir, '..', 'doc')
        full_path_md2html = os.path.join(doc_dir, md2html)
        if not os.path.isfile(full_path_md2html):
            msg = 'Cannot generate HTML: {} not found. '.format(md2html)
            if platform.system() == 'Windows':
                msg += 'Run "nmake /F Makefile.win" in {}'.format(doc_dir)
            else:
                msg += 'Run "make -f Makefile.nix" in {}'.format(doc_dir)
            myprint(opts, msg)
            return

    # get all markdown files
    md_dir = get_markdown_dir(opts)
    html_dir = get_html_dir(opts)

    if not os.path.isdir(html_dir):
        mkdir_p(html_dir)

    doc_files = []
    for filename in os.listdir(md_dir):
        name =  os.path.basename(filename)
        if name.endswith('.md'):
            doc_files.append(os.path.splitext(name)[0])

    if opts.list_files:
        ## list gen files
        for filename in doc_files:
            input_name = os.path.join(md_dir, filename + '.md')
            output_name = os.path.join(html_dir, filename + '.html')
            print(output_name)
    else:
        ## gen html files
        footer = get_html_footer()
        tmp_file = os.path.join(doc_dir, 'tmp.html')
        for filename in doc_files:
            header = get_html_header(opts, title, filename)
            input_name = os.path.join(md_dir, filename + '.md')
            output_name = os.path.join(html_dir, filename + '.html')
            os.system('{} "{}" "{}"'.format(full_path_md2html, input_name,
                                            tmp_file))
            with open_utf8(opts, output_name) as fout:
                fout.write(header)
                with io.open(tmp_file, mode='r', encoding='utf-8') as fin:
                    fout.write(fin.read())
                fout.write(footer)
