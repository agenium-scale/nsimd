#!/bin/sh
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

###############################################################################

cd `dirname $0`
set -x
set -e

###############################################################################
# Init

BUILD_SH="${PWD}/build.sh"
HATCH_PY="${PWD}/../egg/hatch.py"
BUILD_ROOT="${PWD}/.."

###############################################################################
# Generate NSIMD tests

python3 --version 1>/dev/null 2>/dev/null && \
  python3 "${HATCH_PY}" -tf || \
  python "${HATCH_PY}" -tf

###############################################################################
# Run setup

sh "${BUILD_SH}" "$@" || exit 1

###############################################################################
# Parse command line arguments (check has been done by build.sh)

SIMD_EXTS=`echo "${2}" | sed -e 's,/, ,g'`

if [ "${3}" == "" ]; then
  COMPILER_ARG="gcc"
else
  COMPILER_ARG="${4}"
fi
COMPILERS=`echo ${COMPILER_ARG} | sed 's,/, ,g'`

###############################################################################
# Build tests

for compiler in ${COMPILERS}; do
  for simd_ext in ${SIMD_EXTS}; do
    BUILD_DIR="${BUILD_ROOT}/build-${simd_ext}-${compiler}"
    if [ -e "${BUILD_ROOT}/targets.txt" ]; then
      GLOBS=`cat ${BUILD_ROOT}/targets.txt | tr '\n' '|' | sed 's/|$//g'`
      TARGETS=`(cd ${BUILD_DIR} && ninja -t targets) | sed 's/:.*//g' | \
                grep -E "(${GLOBS})" | tr '\n' ' '`
    else
      TARGETS="tests"
    fi
    (cd "${BUILD_DIR}" && ninja ${TARGETS})
  done
done
