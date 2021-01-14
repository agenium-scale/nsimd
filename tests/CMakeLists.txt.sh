# MIT License
#
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

set -e
set -x

BUF="`dirname $0`/.."
NSIMD_CMAKE=`realpath ${BUF}`

for simd_ext in "$@"; do

  # First case: find a specific component
  ROOT_DIR="${PWD}/nsimd_cmake_tests/${simd_ext}"
  rm -rf ${ROOT_DIR}
  mkdir -p ${ROOT_DIR}
  (cd ${ROOT_DIR} && \
   cmake ${NSIMD_CMAKE} \
         -Dsimd=${simd_ext} \
         -DCMAKE_INSTALL_PREFIX=${ROOT_DIR}/root && \
   make VERBOSE=1 && \
   make install)

done
