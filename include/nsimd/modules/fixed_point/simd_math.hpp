/*

Copyright (c) 2019 Agenium Scale

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

*/

#ifndef NSIMD_MODULES_FIXED_POINT_SIMD_MATH_HPP
#define NSIMD_MODULES_FIXED_POINT_SIMD_MATH_HPP

#include "fixed_point/simd.hpp"

// Load/Store
#include "fixed_point/function/simd/loadu.hpp"
#include "fixed_point/function/simd/loada.hpp"
#include "fixed_point/function/simd/loadlu.hpp"
#include "fixed_point/function/simd/loadla.hpp"
#include "fixed_point/function/simd/storeu.hpp"
#include "fixed_point/function/simd/storea.hpp"
#include "fixed_point/function/simd/storelu.hpp"
#include "fixed_point/function/simd/storela.hpp"

// Constants
#include "fixed_point/constants.hpp"

// Bitwise operations
#include "fixed_point/function/simd/andl.hpp"
#include "fixed_point/function/simd/andb.hpp"
#include "fixed_point/function/simd/andnotl.hpp"
#include "fixed_point/function/simd/andnotb.hpp"
#include "fixed_point/function/simd/notl.hpp"
#include "fixed_point/function/simd/notb.hpp"
#include "fixed_point/function/simd/orl.hpp"
#include "fixed_point/function/simd/orb.hpp"
#include "fixed_point/function/simd/xorl.hpp"
#include "fixed_point/function/simd/xorb.hpp"
#include "fixed_point/function/simd/if_else1.hpp"

// Comparisons
#include "fixed_point/function/simd/eq.hpp"
#include "fixed_point/function/simd/ne.hpp"
#include "fixed_point/function/simd/le.hpp"
#include "fixed_point/function/simd/lt.hpp"
#include "fixed_point/function/simd/ge.hpp"
#include "fixed_point/function/simd/gt.hpp"

// Basic arithmetic
#include "fixed_point/function/simd/add.hpp"
#include "fixed_point/function/simd/sub.hpp"
#include "fixed_point/function/simd/mul.hpp"
#include "fixed_point/function/simd/div.hpp"
#include "fixed_point/function/simd/fma.hpp"

// Math functions
#include "fixed_point/function/simd/rec.hpp"
//#include "fixed_point/function/simd/abs.hpp"
//#include "fixed_point/function/simd/sqrt.hpp"
//#include "fixed_point/function/simd/log.hpp"
//#include "fixed_point/function/simd/rsqrt.hpp"
//#include "fixed_point/function/simd/exp.hpp"
//#include "fixed_point/function/simd/ceil.hpp"
// #include "fixed_point/function/simd/floor.hpp"
//#include "fixed_point/function/simd/round.hpp"

// Trigonometric functions
// #include "fixed_point/function/simd/cos.hpp" 
// #include "fixed_point/function/simd/sin.hpp"
// #include "fixed_point/function/simd/tan.hpp"
//#include "fixed_point/function/simd/csc.hpp"
//#include "fixed_point/function/simd/sec.hpp"
//#include "fixed_point/function/simd/cot.hpp"
//#include "fixed_point/function/simd/asin.hpp"
//#include "fixed_point/function/simd/acos.hpp"
//#include "fixed_point/function/simd/atan.hpp"
//#include "fixed_point/function/simd/sinh.hpp"
//#include "fixed_point/function/simd/cosh.hpp"
//#include "fixed_point/function/simd/tanh.hpp"
//#include "fixed_point/function/simd/asinh.hpp"
//#include "fixed_point/function/simd/acosh.hpp"
//#include "fixed_point/function/simd/atanh.hpp"

#endif
