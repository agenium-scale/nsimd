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

# What does this script?
# ----------------------
#
# This script generates code for each architecture, the base C/C++ APIs and
# the advanced C++ API. Each part to be generated is handled by a
# `gen_*.py` file. This script simply calls the `doit` function of each
# `gen_*.py` module. Names are self-explanatory.
#
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
import gen_adv_cxx_api
import gen_adv_c_api
import gen_tests
import gen_benches
import gen_src
import gen_doc
import gen_friendly_but_not_optimized
import gen_modules
import gen_scalar_utilities
import get_sleef_code

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
    # In pratice, we either generate all or all except benches and we never
    # change default directories for code generation. So we remove unused
    # options and regroup some into --library.
    parser = argparse.ArgumentParser(
                 description='This is NSIMD generation script.')
    parser.add_argument('--force', '-f', action='store_true',
        help='Generate all files even if they already exist')
    parser.add_argument('--list-files', '-L', action='store_true',
        default=False,
        help='List files that will be created by hatch.py')
    parser.add_argument('--all', '-A', action='store_true',
        help='Generate code for the library and its benches')
    parser.add_argument('--library', '-l', action='store_true',
        help='Generate code of the library (C and C++ APIs)')
    parser.add_argument('--sleef', '-s', action='store_true', default=False,
        help='Compile Sleef')
    parser.add_argument('--tests', '-t', action='store_true',
        help='Generate tests in C and C++')
    parser.add_argument('--benches', '-b', action='store_true',
        help='Generate benches in C and C++')
    parser.add_argument('--doc', '-d', action='store_true',
        help='Generate all documentation')
    parser.add_argument('--enable-clang-format', '-F', action='store_false',
        default=True,
        help='Disable Clang Format (mainly for speed on Windows)')
    parser.add_argument('--sve-emulate-bool', action='store_true',
        default=False,
        help='Use normal SVE vector to emulate predicates.')
    parser.add_argument('--simd', '-D', type=parse_simd, default='all',
        help='List of SIMD extensions (separated by a comma)')
    parser.add_argument('--match', '-m', type=parse_match, default=None,
        help='Regex used to filter generation on operator names')
    parser.add_argument('--verbose', '-v', action = 'store_true', default=None,
        help='Enable verbose mode')
    parser.add_argument('--simple-license', action='store_true', default=False,
        help='Put a simple copyright statement instead of the whole license')
    opts = parser.parse_args(args)
    # When -L has been chosen, we want to list all files and so we have to
    # turn to True other parameters
    if opts.list_files:
        opts.library = True
        opts.tests = True
        opts.benches = True
        opts.force = True
        opts.doc = True
    # We set variables here because all the code depends on them + we do want
    # to keep the possibility to change them in the future
    opts.archis = opts.library
    opts.base_apis = opts.library
    opts.adv_cxx_api = opts.library
    opts.adv_c_api = opts.library
    opts.friendly_but_not_optimized = opts.library
    opts.src = opts.library
    opts.scalar_utilities = opts.library
    opts.sleef_version = '3.5.1'
    opts.include_dir = os.path.join(script_dir, '..', 'include', 'nsimd')
    opts.benches_dir = os.path.join(script_dir, '..', 'benches')
    opts.tests_dir = os.path.join(script_dir, '..', 'tests')
    opts.src_dir = os.path.join(script_dir, '..', 'src')
    return opts

# -----------------------------------------------------------------------------
# Entry point

def main():
    opts = parse_args(sys.argv[1:])
    opts.script_dir = script_dir
    opts.modules_list = None
    opts.platforms_list = None

    ## Gather all SIMD dependencies
    opts.simd = common.get_simds_deps_from_opts(opts)
    common.myprint(opts, 'List of SIMD: {}'.format(', '.join(opts.simd)))
    if opts.archis == True or opts.all == True:
        gen_archis.doit(opts)
    if opts.base_apis == True or opts.all == True:
        gen_base_apis.doit(opts)
    if opts.adv_cxx_api == True or opts.all == True:
        gen_adv_cxx_api.doit(opts)
    if opts.adv_c_api == True or opts.all == True:
        gen_adv_c_api.doit(opts)
    if opts.tests == True or opts.all == True:
        gen_tests.doit(opts)
    if opts.benches == True or opts.all == True:
        gen_benches.doit(opts)
    if opts.src == True or opts.all == True:
        gen_src.doit(opts)
    if opts.sleef == True or opts.all == True:
        get_sleef_code.doit(opts)
    if opts.scalar_utilities == True or opts.all == True:
        gen_scalar_utilities.doit(opts)
    if opts.friendly_but_not_optimized == True or opts.all == True:
        gen_friendly_but_not_optimized.doit(opts)
    gen_modules.doit(opts) # this must be here after all NSIMD
    if opts.doc == True or opts.all == True:
        gen_doc.doit(opts)

if __name__ == '__main__':
    main()
