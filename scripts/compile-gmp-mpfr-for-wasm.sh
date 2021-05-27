#!/bin/sh
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

cd `dirname $0`

set -e
#set -x

PREFIX="${PWD}/../_wasm-deps"
rm -rf ${PREFIX}
mkdir -p ${PREFIX}
J=`nproc || echo 10`

# -----------------------------------------------------------------------------
# GMP first

if [ "$1" != "" ]; then
  VER=$1
else
  VER=6.2.1
fi

URL=https://gmplib.org/download/gmp/gmp-${VER}.tar.xz

curl -L ${URL} -o gmp.tar.xz
tar xf gmp.tar.xz
(cd gmp-${VER} && \
 emconfigure ./configure --disable-assembly \
                         --host none \
                         --enable-cxx \
                         --prefix=${PREFIX} && \
 make -j${J} &&
 make install)

# -----------------------------------------------------------------------------
# MPFR first

if [ "$1" != "" ]; then
  VER=$1
else
  VER=4.1.0
fi

URL=https://www.mpfr.org/mpfr-current/mpfr-${VER}.tar.gz

curl -L ${URL} -o mpfr.tar.xz
tar xf mpfr.tar.xz
(cd mpfr-${VER} && \
 emconfigure ./configure --disable-assembly \
                         --host none \
                         --with-gmp=${PREFIX} \
                         --prefix=${PREFIX} && \
 make -j${J} &&
 make install)

# -----------------------------------------------------------------------------
# Echo nsconfig parameters to compile for WebAssembly

echo
echo
echo
echo
echo "+---------------------------------------------------------+"
echo
echo "Compilation of MPFR + GMP is ok."
echo "Invocation of nsconfig to compile for WebAssembly"
echo
echo "CPU emulation:"
echo
echo "../nstools/bin/nsconfig .. -Dsimd=cpu \\"
echo "    -Dmpfr=\"-I${PREFIX}/include -L${PREFIX}/lib -lmpfr -lgmp\""
echo
echo "WASM SIMD128:"
echo
echo "../nstools/bin/nsconfig .. -Dsimd=wasm_simd128 \\"
echo "    -Dmpfr=\"-I${PREFIX}/include -L${PREFIX}/lib -lmpfr -lgmp\""
echo
echo "+---------------------------------------------------------+"
echo
