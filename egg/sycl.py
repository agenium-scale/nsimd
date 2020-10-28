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

import common

fmtspec = dict()

# -----------------------------------------------------------------------------

def get_impl(operator, totyp, typ):
    global fmtspec

    fmtspec = {
      'in0': common.in0,
      'in1': common.in1,
      'in2': common.in2,
      'typ': typ,
      'totyp': totyp,
      'typnbits': typ[1:]
    }

    if operator.name == 'trunc':
        cpp_code = ''
        if typ == 'f16':
            cpp_code = \
                '''f32 buf = nsimd_f16_to_f32({in0});
                return nsimd_f32_to_f16(buf >= 0.0f ?
                sycl::floor(buf) : sycl::ceil(buf));'''.format(**fmtspec)
        elif typ in ['f32', 'f64']:
            cpp_code = '''\
            return {in0} >= 0.0{f} ? sycl::floor({in0}) : sycl::ceil({in0});'''.\
            format(f='f' if typ == 'f32' else '', **fmtspec)
        return cpp_code

    if operator.name in ['min', 'max']:
        op = 'sycl::fmin' if operator.name == 'min' else 'sycl::fmax'
        if typ == 'f16':
            return '''f32 in0 = nsimd_f16_to_f32({in0});
                      f32 in1 = nsimd_f16_to_f32({in1});
                      return nsimd_f32_to_f16({op}(in0, in1));'''. \
                      format(op=op, **fmtspec)
        elif typ in ['f32', 'f64']:
            return 'return {op}({in0}, {in1});'. \
                   format(op=op, **fmtspec)
        
    if operator.name in ['floor', 'ceil', 'sqrt']:
        if typ == 'f16':
            return'return nsimd_f32_to_f16((f32)sycl::{op}((f32)nsimd_f16_to_f32({in0})));'.\
                format(op=operator.name, **fmtspec)
        elif typ in ['f32', 'f64']:
            return 'return sycl::{op}({in0});'.\
                format(op=operator.name, **fmtspec)
