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

import os
import operators
import common
import cuda
import gen_scalar_utilities

# -----------------------------------------------------------------------------

def gen_functions(opts):
    functions = ''

    # Write the code to file
    dirname = os.path.join(opts.include_dir, 'modules', 'spmd')
    common.mkdir_p(dirname)
    filename = os.path.join(dirname, 'functions.hpp')
    if not common.can_create_filename(opts, filename):
        return
    with common.open_utf8(opts, filename) as out:
        out.write('#ifndef NSIMD_MODULES_SPMD_FUNCTIONS_HPP\n')
        out.write('#define NSIMD_MODULES_SPMD_FUNCTIONS_HPP\n\n')
        out.write('namespace spmd {\n\n')
        out.write('{}\n\n'.format(common.hbar))
        out.write(functions)
        out.write('} // namespace spmd\n\n')
        out.write('#endif\n')
    common.clang_format(opts, filename)

# -----------------------------------------------------------------------------

def name():
    return 'Random123'

def desc():
    return '''Random123'''

def doc_menu():
    return {'Overview': 'overview', 'API reference': 'api'}

# -----------------------------------------------------------------------------

def doit(opts):
    common.myprint(opts, 'Generating module spmd')
    if opts.library:
        gen_functions(opts)
    if opts.tests:
        # TODO
        pass
    if opts.doc:
        # TODO
        pass

