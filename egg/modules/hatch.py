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
import gen_tests

# Dir of this script
script_dir = os.path.dirname(__file__)
if script_dir == '':
    script_dir = '.'

# -----------------------------------------------------------------------------
# Arguments parsing

def parse_args(args):
    def parse_match(value):
        if value is None:
            return None
        else:
            return re.compile(value)
    parser = argparse.ArgumentParser(
                 description='This is NSIMD generation script.')
    parser.add_argument('--force', '-f', action='store_true',
        help='Generate all files even if they already exist')
    parser.add_argument('--all', '-A', action='store_true',
        help='Generate code for all architectures, C and C++ APIs')
    parser.add_argument('--friendly-but-not-optimized', '-o',
        action='store_true',
        help='Generate friendly but not optimized overloads for C++')
    parser.add_argument('--tests', '-t', action='store_true',
        help='Generate tests in C++')
    parser.add_argument('--include-dir', '-i', type=str,
        default=os.path.join(script_dir, '..', 'include', 'nsimd'),
        help='Base directory for headers')
    parser.add_argument('--tests-dir', '-T', type=str,
        default=os.path.join(script_dir, '..', 'tests/modules/fixed_point'),
        help='Base directory for tests')
    parser.add_argument('--doc', '-d', action='store_true',
        help='Generate all documentation')
    parser.add_argument('--disable-clang-format', '-F', action='store_true',
        help='Disable Clang Format (mainly for speed on Windows)')
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
    if opts.tests_fp == True or opts.all == True:
        pass
        #gen_tests_fp.doit(opts)
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
