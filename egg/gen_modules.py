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
import common

def doit(opts):
    # We have one module by directory
    path = os.path.join(opts.script_dir, 'modules')
    common.myprint(opts, 'Searching modules in "{}"'.format(path))
    for module_dir in os.listdir(path):
        if (not os.path.isdir(os.path.join(path, module_dir))) or \
           module_dir == '.' or module_dir == '..' or \
           (not os.path.exists(os.path.join(path, module_dir, 'hatch.py'))):
            continue
        common.myprint(opts, 'Found new module: {}'.format(module_dir))
        mod = __import__('modules.{}.hatch'.format(module_dir))
        exec('mod.{}.hatch.doit(opts)'.format(module_dir))
