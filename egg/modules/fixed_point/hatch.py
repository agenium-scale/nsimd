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
# Imports

import modules.fixed_point.gen_tests
import modules.fixed_point.gen_doc

# -----------------------------------------------------------------------------

def name():
    return 'Fixed-point arithmetic'

def desc():
    return '''This module provides vectorized fixed-point arithmetic through
a C++98 API. The programmer can choose the integral type and the place of the
coma for representing its fixed-point numbers. A number of operators are
also provided.'''

def doc_menu():
    return {'Overview': 'overview', 'API reference': 'api'}

# -----------------------------------------------------------------------------
# Entry point

def doit(opts):
    if opts.tests == True or opts.all == True:
        modules.fixed_point.gen_tests.doit(opts)
    if opts.doc == True or opts.all == True:
        modules.fixed_point.gen_doc.doit(opts)
