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

# How nsimd works?
# ----------------
#
# nsimd.h includes the following:
#   - config.h         compiler detection, inline stuff and more and includes:
#     - pp.h           preprocessor stuff
#     - detect_simd.h  detect SIMD based on what is given by the user
#     - basic_types.h  defines basic arithmetic types
#
# Then each function in `include/nsimd` does an NSIMD_AUTO_INCLUDE which,
# based on what was detected in `detect_simd.h` includes the correct function
# from `include/nsimd/PLATFORM/SIMD_EXT`. The same holds for the advanced
# C++ API.
#
# In `src` lies all functions that do not need to be in headers for
# performance such as memory management functions, trigonometric functions,
# log-exp functions, ...
#
# What does this script?
# ----------------------
#
# This script generates code for each architecture, the base C/C++ APIs and
# the advanced C++ API. Each part to be generated is handled by a
# `gen_*.py` file. This script simply calls the `doit` function of each
# `gen_*.py` module. Names are self-explanatory.
#
# The list of supported architectures is determined by looking at the `egg`
# directory and listing all `platform_*.py` files. Each file must contain all
# SIMD extensions for a given architecture. For example the default (no SIMD) is
# given by `platform_cpu.py`. All the Intel SIMD extensions are given by
# `platform_x86.py`.
#
# Each module that implements a platform:
#   - must be named 'platform_[name for platform].py
#   - must export at least the following functions
#
#     * def get_type(simd_ext, typ)
#       Returns the "intrinsic" SIMD type corresponding to the given
#       arithmetic type. If typ or simd_ext is not known then a ValueError
#       exception must be raised.
#
#     * def get_additional_include(func, simd_ext, typ)
#       Returns additional include if need be for the implementation of func for
#       the given simd_ext and typ.
#
#     * def get_logical_type(simd_ext, typ)
#       Returns the "intrinsic" logical SIMD type corresponding to the given
#       arithmetic type. If typ or simd_ext is not known then a ValueError
#       exception must be raised.
#
#     * def get_nb_registers(simd_ext)
#       Returns the number of registers for this SIMD extension.
#
#     * def get_impl(func, simd_ext, from_typ, to_typ)
#       Returns the implementation (C code) for func on type typ for simd_ext.
#       If typ or simd_ext is not known then a ValueError exception must be
#       raised. Any func given satisfies `S func(T a0, T a1, ... T an)`.
#
#     * def has_compatible_SoA_types(simd_ext)
#       Returns True iff the given simd_ext has structure of arrays types
#       compatible with NSIMD i.e. whose members are v1, v2, ... Returns False
#       otherwise. If simd_ext is not known then a ValueError exception must be
#       raised.
#
#     * def get_SoA_type(simd_ext, typ, deg)
#       Returns the structure of arrays types for the given typ, simd_ext and
#       deg. If simd_ext is not known or does not name a type whose
#       corresponding SoA types are compatible with NSIMD then a ValueError
#       exception must be raised.
#
#     * def emulate_fp16(simd_ext)
#       Returns True iff the given SIMD extension has to emulate FP16's with
#       two FP32's.

# -----------------------------------------------------------------------------
# First thing we do is check whether python3 is used

import sys
if sys.version_info[0] < 3:
    print('Only Python 3 is supported')
    sys.exit(1)

# -----------------------------------------------------------------------------
# Imports

import argparse
import os
import re
import common
import gen_archis
import gen_base_apis
import gen_advanced_api
import gen_tests
import gen_benches
import gen_src
import gen_doc
import gen_friendly_but_not_optimized
import gen_ulps

# Dir of this script
script_dir = os.path.dirname(__file__)
if script_dir == '':
    script_dir = '.'

# -----------------------------------------------------------------------------
# Arguments parsing

def parse_args(args):
    def parse_simd(value):
        ## Split .simd now
        values = {
            'x86': common.x86_simds,
            'arm': common.arm_simds,
            'ppc': common.ppc_simds,
            'all': common.simds,
        }.get(value, value.split(','))
        ## Check that all simd are valid
        ret = []
        for simd in values:
            if simd not in common.simds:
                raise argparse.ArgumentTypeError(
                        "SIMD '{}' not found in {}".format(simd, common.simds))
            ret += common.simds_deps[simd]
        return list(set(ret))
    def parse_match(value):
        if value is None:
            return None
        else:
            return re.compile(value)
    parser = argparse.ArgumentParser(
                 description='This is NSIMD generation script.')
    parser.add_argument('--force', '-f', action='store_true',
        help='Generate all files even if they already exist')
    parser.add_argument('--archis', '-a', action='store_true',
        help='Generate code for all architectures')
    parser.add_argument('--all', '-A', action='store_true',
        help='Generate code for all architectures, C and C++ APIs')
    parser.add_argument('--base-apis', '-c', action='store_true',
        help='Generate the base C and C++ APIs')
    parser.add_argument('--cxx-api', '-C', action='store_true',
        help='Generate the "pack" C++ish API')
    parser.add_argument('--ulps', '-u', action='store_true',
        help='Generate code to compute precision on big functions')
    parser.add_argument('--ulps-dir', '-U', type=str,
        default=os.path.join(script_dir, '..', 'ulps'),
        help='Generate code to compute precision on big functions')
    parser.add_argument('--friendly-but-not-optimized', '-o',
        action='store_true',
        help='Generate friendly but not optimized overloads for C++')
    parser.add_argument('--tests', '-t', action='store_true',
        help='Generate tests in C and C++')
    parser.add_argument('--tests-fp', action='store_true',
        help='Generate tests in C and C++ for the fixed precision module')
    parser.add_argument('--benches', '-b', action='store_true',
        help='Generate benches in C and C++')
    parser.add_argument('--include-dir', '-i', type=str,
        default=os.path.join(script_dir, '..', 'include', 'nsimd'),
        help='Base directory for headers')
    parser.add_argument('--benches-dir', '-B', type=str,
        default=os.path.join(script_dir, '..', 'benches'),
        help='Base directory for benches')
    parser.add_argument('--tests-dir', '-T', type=str,
        default=os.path.join(script_dir, '..', 'tests'),
        help='Base directory for tests')
    parser.add_argument('--doc', '-d', action='store_true',
        help='Generate all documentation')
    parser.add_argument('--disable-clang-format', '-F', action='store_true',
        help='Disable Clang Format (mainly for speed on Windows)')
    parser.add_argument('--src', '-s', action='store_true',
        help='Generate all of the src function bindings')
    parser.add_argument('--src-dir', '-S', action='store_true',
        default=os.path.join(script_dir, '..', 'src'),
        help='Base directory for src')
    parser.add_argument('--simd', '-D',
        type=parse_simd,
        default='all',
        help='List of SIMD extensions (separated by a comma)')
    parser.add_argument('--match', '-m',
        type=parse_match,
        default=None,
        help='Regex used to filter generation on operator names')
    parser.add_argument('--verbose', '-v',
        action = 'store_true',
        default=None,
        help='Enable verbose mode')
    return parser.parse_args(args)

# -----------------------------------------------------------------------------
# Entry point

def main():
    opts = parse_args(sys.argv[1:])
    opts.script_dir = script_dir

    ## Gather all SIMD dependencies
    opts.simd = common.get_simds_deps_from_opts(opts)
    print('-- List of SIMD: {}'.format(', '.join(opts.simd)))
    if opts.archis == True or opts.all == True:
        gen_archis.doit(opts)
    if opts.base_apis == True or opts.all == True:
        gen_base_apis.doit(opts)
    if opts.cxx_api == True or opts.all == True:
        gen_advanced_api.doit(opts)
    if opts.ulps == True or opts.all == True:
        gen_ulps.doit(opts)
    if opts.tests == True or opts.all == True:
        gen_tests.doit(opts)
    if opts.benches == True or opts.all == True:
        gen_benches.doit(opts)
    if opts.src == True or opts.all == True:
        gen_src.doit(opts)
    if opts.friendly_but_not_optimized == True or opts.all == True:
        gen_friendly_but_not_optimized.doit(opts)
    if opts.doc == True or opts.all == True:
        gen_doc.doit(opts)

if __name__ == '__main__':
    main()
