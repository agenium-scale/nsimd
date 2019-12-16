#!/usr/bin/env bash

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

# (cd src && bash nsimd.sh)

rm -rf nsimd
git clone https://github.com/agenium-scale/nsimd.git

cd nsimd
python3 egg/hatch.py -Af

mkdir build_cpu
cd build_cpu
cmake ..
make
cd ..

mkdir build_sse2
cd build_sse2
# Enable SSE2 support for nsimd
# SIMD possible values are:
# - SSE2, SSE42, AVX, AVX2, AVX512_KNL, AVX512_SKYLAKE
# - NEON128, AARCH64, SVE
cmake .. -DSIMD=SSE2
make
cd ..

mkdir build_avx_fma
cd build_avx_fma
# Enable AVX with FMA support for nsimd
cmake .. -DSIMD=AVX -DSIMD_OPTIONALS=FMA
make
cd ..
