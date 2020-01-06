# Use utf-8 encoding
# -*- coding: utf-8 -*-

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
import sys
sys.path.append("..")
from common import *

## -----------------------------------------------------------------------------

fp_operators = []

DocMisc = 'Miscellaneous operators'
DocLoadStore = 'Loads and stores'
DocBasicArithmetic = 'Basic arithmetic operators'
DocComparison = 'Comparison operators'
DocLogicalOperators = 'Logical operators'
DocBitsOperators = 'Bits manipulation operators'

fp_categories = sorted([
    DocMisc,
    DocLoadStore,
    DocBasicArithmetic,
    DocComparison,
    DocLogicalOperators,
    DocBitsOperators])

class FpOperator(object):
    name = ''
    full_name = ''
    domain = Domain('R')
    desc = ''
    signatures = ''
    categories = []
    cxx_operator = ''
    
## -----------------------------------------------------------------------------
# Len

Len = FpOperator()
Len.name = 'len'
Len.full_name = 'Vector length'
Len.signatures = ['p len cT&', 'p len cv&']
Len.domain = Domain('')
Len.categories = [DocMisc]
Len.desc = 'Returns the number of elements contained in a vector.'
fp_operators.append(Len)

## -----------------------------------------------------------------------------
# Set1

Set1 = FpOperator()
Set1.name = 'set1'
Set1.full_name = 'Value broadcast'
Set1.signatures = ['T set1 s']
Set1.categories = [DocMisc]
Set1.desc = 'Returns a vector whose all elements are set to the given value.'
fp_operators.append(Set1)

## -----------------------------------------------------------------------------
# Loadu

Loadu = FpOperator()
Loadu.name = 'loadu'
Loadu.full_name = 'Vector unaligned load'
Loadu.signatures = ['v loadu s*']
Loadu.categories = [DocLoadStore]
Loadu.desc = 'Load data from unaligned memory.'
fp_operators.append(Loadu)

## -----------------------------------------------------------------------------
# Loada

Loada = FpOperator()
Loada.name = 'loada'
Loada.full_name = 'Vector aligned load'
Loada.signatures = ['v loada s*']
Loada.categories = [DocLoadStore]
Loada.desc = 'Load data from aligned memory.'
fp_operators.append(Loada)

## -----------------------------------------------------------------------------
# Loadlu

Loadlu = FpOperator()
Loadlu.name = 'loadlu'
Loadlu.full_name = 'Logical vector unaligned load'
Loadlu.signatures = ['vl loadlu s*']
Loadlu.categories = [DocLoadStore]
Loadlu.desc = 'Load logical data from unaligned memory.'
fp_operators.append(Loadlu)

## -----------------------------------------------------------------------------
# Loadla

Loadla = FpOperator()
Loadla.name = 'loadla'
Loadla.full_name = 'Logical vector aligned load'
Loadla.signatures = ['vl loadla s*']
Loadla.categories = [DocLoadStore]
Loadla.desc = 'Load logical data from aligned memory.'
fp_operators.append(Loadla)

## -----------------------------------------------------------------------------
# Storeu

Storeu = FpOperator()
Storeu.name = 'storeu'
Storeu.full_name = 'Vector unaligned store'
Storeu.signatures = ['V storeu s* T']
Storeu.categories = [DocLoadStore]
Storeu.desc = 'Store a vector in unaligned memory.'
fp_operators.append(Storeu)

## -----------------------------------------------------------------------------
# Storea

Storea = FpOperator()
Storea.name = 'storea'
Storea.full_name = 'Vector aligned store'
Storea.signatures = ['V storea s* T']
Storea.categories = [DocLoadStore]
Storea.desc = 'Store a vector in aligned memory.'
fp_operators.append(Storea)

## -----------------------------------------------------------------------------
# Storelu

Storelu = FpOperator()
Storelu.name = 'storelu'
Storelu.full_name = 'Logical vector unaligned store'
Storelu.signatures = ['V storelu s* T']
Storelu.categories = [DocLoadStore]
Storelu.desc = 'Store a logical vector in an unaligned memory.'
fp_operators.append(Storelu)

## -----------------------------------------------------------------------------
# Storela

Storela = FpOperator()
Storela.name = 'storela'
Storela.full_name = 'Logical vector aligned store'
Storela.signatures = ['V storela s* T']
Storela.categories = [DocLoadStore]
Storela.desc = 'Store a logical vector in an aligned memory.'
fp_operators.append(Storela)

## -----------------------------------------------------------------------------
# Add

Add = FpOperator()
Add.name = 'add'
Add.full_name = 'Addition of two vectors'
Add.signatures = ['v add cv& cv&']
Add.categories = [DocBasicArithmetic]
Add.desc = 'Adds two vectors.'
Add.cxx_operator = '+'
fp_operators.append(Add)

## -----------------------------------------------------------------------------
# Sub

Sub = FpOperator()
Sub.name = 'sub'
Sub.full_name = 'Substraction of two vectors'
Sub.signatures = ['v sub cv& cv&']
Sub.categories = [DocBasicArithmetic]
Sub.cxx_operator = '-'
Sub.desc = 'Substracts two vectors.'
fp_operators.append(Sub)

## -----------------------------------------------------------------------------
# Mul

Mul = FpOperator()
Mul.name = 'mul'
Mul.full_name = 'Multplication of two vectors'
Mul.signatures = ['v mul cv& cv&']
Mul.categories = [DocBasicArithmetic]
Mul.cxx_operator = '*'
Mul.desc = 'Multiplies two vectors.'
fp_operators.append(Mul)

## -----------------------------------------------------------------------------
# Div

Div = FpOperator()
Div.name = 'div'
Div.full_name = 'Division of two vectors'
Div.signatures = ['v div cv& cv&']
Div.categories = [DocBasicArithmetic]
Div.cxx_operator = '/'
Div.desc = 'Divides two vectors.'
fp_operators.append(Div)

## -----------------------------------------------------------------------------
# Fma

Fma = FpOperator()
Fma.name = 'fma'
Fma.full_name = 'Fused multiplication and accumulation emulation'
Fma.signatures = ['v fma cv& cv& cv&']
Fma.categories = [DocBasicArithmetic]
Fma.desc = 'Emulates the FMA operation with fixed-point arithmetic' \
    + 'for compatibility.\n' \
    + 'This function is just a wrapper that calls consecutively  an add then\n' \
    + 'a mul operation.'
fp_operators.append(Fma)

## -----------------------------------------------------------------------------
# Min

Min = FpOperator()
Min.name = 'min'
Min.full_name = 'Minimum value'
Min.signatures = ['v min cv& cv&']
Min.categories = [DocBasicArithmetic]
Min.desc = 'Returns a vector with the min values of the input vectors.'
fp_operators.append(Min)

## -----------------------------------------------------------------------------
# Max

Max = FpOperator()
Max.name = 'max'
Max.full_name = 'Maximum value'
Max.signatures = ['v max cv& cv&']
Max.categories = [DocBasicArithmetic]
Max.desc = 'Returns a vector with the max values of the input vectors.'
fp_operators.append(Max)

## -----------------------------------------------------------------------------
# Abs

Abs = FpOperator()
Abs.name = 'abs'
Abs.full_name = 'Absolute value'
Abs.signatures = ['v abs cv&']
Abs.categories = [DocBasicArithmetic]
Abs.desc = 'Absolute value of a fixed-point vector'
fp_operators.append(Abs)

## -----------------------------------------------------------------------------
# Rec

Rec = FpOperator()
Rec.name =  'rec'
Rec.full_name = 'Reciprocal'
Rec.signatures = ['v rec cv&']
Rec.categories = [DocBasicArithmetic]
Rec.desc = 'Reciprocal value of a fixed-point SIMD register.'
fp_operators.append(Rec)

## -----------------------------------------------------------------------------
# Eq

Eq = FpOperator()
Eq.name = 'eq'
Eq.full_name = 'Compare for equality'
Eq.signatures = ['vl eq cv& cv&']
Eq.categories = [DocComparison]
Eq.cxx_operator = '=='
Eq.desc = '''\
Peforms an equality test between two fixed-point registers, and returns
the results of the test in a logical register.
'''
fp_operators.append(Eq)

## -----------------------------------------------------------------------------
# Ne

Ne = FpOperator()
Ne.name = 'ne'
Ne.full_name = 'Compare for inequality'
Ne.signatures = ['vl ne cv& cv&']
Ne.categories = [DocComparison]
Ne.cxx_operator = '!='
Ne.desc = '''\
Performs an inequality test between two fixed-point registers, and returns
the results on the test in a logical register.
'''
fp_operators.append(Ne)

## -----------------------------------------------------------------------------
# Le

Le = FpOperator()
Le.name = 'le'
Le.full_name = 'Compare for lesser-or-equal-than'
Le.signatures = ['vl le cv& cv&']
Le.categories = [DocComparison]
Le.cxx_operator = '<='
Le.desc = '''\
Performs a lesser-or-equal comparison between two fixed-point registers, and returns
the results of the test in a logical vector.
'''
fp_operators.append(Le)

## -----------------------------------------------------------------------------
# Lt

Lt = FpOperator()
Lt.name = 'lt'
Lt.full_name = 'Compare for lesser-than'
Lt.signatures = ['vl lt cv& cv&']
Lt.categories = [DocComparison]
Lt.cxx_operator = '<'
Lt.desc = '''\
Performs a lesser-than comparison between two fixed-point registers, and returns 
the results of the test in a logical vector.
'''
fp_operators.append(Lt)

## -----------------------------------------------------------------------------
# e

Ge = FpOperator()
Ge.name = 'ge'
Ge.full_name = 'Compare for greater-or-equal-than'
Ge.signatures = ['vl ge cv& cv&']
Ge.categories = [DocComparison]
Ge.cxx_operator = '>='
Ge.desc = '''\
Performs a greater-or-equal-than comparison between two fixed-point registers, and returns 
the results of the test in a logical vector.
'''
fp_operators.append(Ge)

## -----------------------------------------------------------------------------
# Gt

Gt = FpOperator()
Gt.name = 'gt'
Gt.full_name = 'Compare for greater-than'
Gt.signatures = ['vl gt cv& cv&']
Gt.categories = [DocComparison]
Gt.cxx_operator = '>'
Gt.desc = '''\
Performs a greater-than comparison between two fixed-point registers, and returns 
the results of the test in a logical vector.
'''
fp_operators.append(Gt)

## -----------------------------------------------------------------------------
# IfElse1

IfElse = FpOperator()
IfElse.name = 'if_else1'
IfElse.full_name = 'Vector blending'
IfElse.signatures = ['vl if_else1 cv& cv&']
IfElse.categories = [DocMisc]
IfElse.desc = '''\
Blend the inputs using the vector of logical as a first argument. 
Elements of the second input is taken when the corresponding elements from the vector 
of logicals is true, otherwise elements of the second input are taken.
'''
fp_operators.append(IfElse)

## -----------------------------------------------------------------------------
# Andb

Andb = FpOperator()
Andb.name = 'andb'
Andb.full_name = 'Bitwise and'
Andb.signatures = ['v andb cv& cv&']
Andb.categories = [DocBitsOperators]
Andb.desc = 'Bitwise and between two fixed-point SIMD registers.'
fp_operators.append(Andb)

## -----------------------------------------------------------------------------
# Andnotb

Andnotb = FpOperator()
Andnotb.name = 'andnotb'
Andnotb.full_name = 'Bitwise and not'
Andnotb.signatures = ['v andnotb cv& cv&']
Andnotb.categories = [DocBitsOperators]
Andnotb.desc = 'Bitwise and not between two fixed-point SIMD registers.'
fp_operators.append(Andnotb)

## -----------------------------------------------------------------------------
# Norb

Notb = FpOperator()
Notb.name = 'notb'
Notb.full_name = 'Bitwise not'
Notb.signatures = ['v notb cv&']
Notb.categories = [DocBitsOperators]
Notb.desc = 'Not operator on a fixed-point SIMD register.'
fp_operators.append(Notb)

## -----------------------------------------------------------------------------
# Orb

Orb = FpOperator()
Orb.name = 'orb'
Orb.full_name = 'Bitwise or'
Orb.signatures = ['v orb cv& cv&']
Orb.categories = [DocBitsOperators]
Orb.desc = 'Bitwise or between two fixed-point SIMD registers.'
fp_operators.append(Orb)

## -----------------------------------------------------------------------------
# Xorb
Xorb = FpOperator()
Xorb.name = 'xorb'
Xorb.full_name = 'Bitwise xor'
Xorb.signatures = ['v xorb cv& cv&']
Xorb.categories = [DocBitsOperators]
Xorb.desc = 'Bitwise xor between two fixed-point SIMD registers.'
fp_operators.append(Xorb)

## -----------------------------------------------------------------------------
# Andl

Andl = FpOperator()
Andl.name = 'andl'
Andl.full_name = 'Bitwise logical and'
Andl.signatures = ['vl andl cvl& cvl&']
Andl.categories = [DocLogicalOperators]
Andl.desc = 'Bitwise and between two logical SIMD registers.'
fp_operators.append(Andl)

## -----------------------------------------------------------------------------
# Andnotl

Andnotl = FpOperator()
Andnotl.name = 'andnotl'
Andnotl.full_name = 'Bitwise and not'
Andnotl.signatures = ['vl andnotl cvl& cvl&']
Andnotl.categories = [DocLogicalOperators]
Andnotl.desc = 'Bitwise and not between two logical SIMD registers.'
fp_operators.append(Andnotl)

## -----------------------------------------------------------------------------
# Notl

Notl = FpOperator()
Notl.name = 'notl'
Notl.full_name = 'Bitwise not'
Notl.signatures = ['vl notb cvl&']
Notl.categories = [DocLogicalOperators]
Notl.desc = 'Not operator on a logical SIMD register.'
fp_operators.append(Notl)

## -----------------------------------------------------------------------------
# Orl

Orl = FpOperator()
Orl.name = 'orl'
Orl.full_name = 'Bitwise or'
Orl.signatures = ['vl orb cvl& cvl&']
Orl.categories = [DocLogicalOperators]
Orl.desc = 'Bitwise or between two logical SIMD registers.'
fp_operators.append(Orl)

## -----------------------------------------------------------------------------
# Xorl

Xorl = FpOperator()
Xorl.name = 'xorl'
Xorl.full_name = 'Bitwise xor'
Xorl.signatures = ['vl xorb cvl& cvl&']
Xorl.categories = [DocLogicalOperators]
Xorl.desc = 'Bitwise xor between two logical SIMD registers.'
fp_operators.append(Xorl)

## -----------------------------------------------------------------------------

fp_operators = sorted(fp_operators, key=lambda op: op.name)
