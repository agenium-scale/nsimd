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
# NVIDIA doc on f16 can be found at
# https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__HALF.html

def get_impl_f16(operator, totyp, typ):
    if operator.name == 'round_to_even':
        return 'return __hrint({in0}, {in1});'
    elif operator.name in ['rec', 'rec8', 'rec11']:
        return 'return __hrcp({in0});'
    elif operator.name in ['rsqrt', 'rsqrt8', 'rsqrt11']:
        return 'return __hrsqrt({in0});'
    elif operator.name in ['fma', 'fms', 'fnma', 'fnms']:
        neg = '-' if operator.name in ['fnma, fnms'] else ''
        op = '-' if operator.name in ['fnms, fms'] else ''
        return 'return __hfma({neg}{{in0}}, {{in1}}, {op}{{in2}});'. \
               format(neg=neg, op=op)
    elif operator.name in ['min', 'max']:
        intr = '__hlt' if operator.name == 'min' else '__hgt'
        return '''if ({intr}) {{
                    return {{in0}};
                  }} else {{
                    return {{in1}};
                  }}'''.format(intr=intr)
    else:
        return 'return __h{}({{in0}}, {{in1}});'.format(operator.name)

# -----------------------------------------------------------------------------

def get_impl(operator, totyp, typ):
    if typ == 'f16':
        return get_impl_f16(operator, totyp, typ)
    func = {
        'orb': lambda: opbit('{in0} | {in1}', typ),
        'andb': lambda: opbit('{in0} & {in1}', typ),
        'andnotb': lambda: opbit('{in0} & (~{in1})', typ),
        'notb': lambda: opbit('~{in0}', typ),
        'xorb': lambda: opbit('{in0} ^ {in1}', typ),
        'add': lambda: opnum('{in0} + {in1}', typ),
        'sub': lambda: opnum('{in0} - {in1}', typ),
        'mul': lambda: opnum('{in0} * {in1}', typ),
        'div': lambda: opnum('{in0} / {in1}', typ),
        'neg': lambda: opnum('-{in0}', typ),
        'rec': lambda: opnum('1.0{f} / {{in0}};'.format(f=f), typ),
        'rec8': lambda: opnum('1.0{f} / {{in0}};'.format(f=f), typ),
        'rec11': lambda: opnum('1.0{f} / {{in0}};'.format(f=f), typ),
    }
    if operator.name in func:
        return func[operator.name]()
    elif operator.name in ['fma', 'fms', 'fnma', 'fnms']:
        neg = '-' if operator.name in ['fnma, fnms'] else ''
        op = '-' if operator.name in ['fnms, fms'] else ''
        return 'return fma{f}({neg}{{in0}}, {{in1}}, {op}{{in2}});'. \
               format(f='f' if typ == 'f32' else '', neg=neg, op=op)
    else:
        cuda_name = {
            'round_to_even': 'rint',
            'min': 'fmin',
            'max': 'fmax',
            'abs': 'fabs'
        }
        args = ', '.join(['{{in{}}}'.format(i) \
                          for i in range(len(operator.args))])
        return 'return {name}{f}({args});'. \
               format(name=cuda_func[operator.name] \
                      if operator.name in cuda_name else operator.name,
                      name=cuda_func,f='f' if typ == 'f32' else '', args=args)
