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

SETUP_SH="${PWD}/setup.sh"
NSCONFIG="${PWD}/../nstools/bin/nsconfig"
HATCH_PY="${PWD}/../egg/hatch.py"
BUILD_ROOT="${PWD}/.."

###############################################################################
# Run setup

sh "${SETUP_SH}"

###############################################################################
# Generate NSIMD

python3 --version 1>/dev/null 2>/dev/null && \
  python3 "${HATCH_PY}" -lf || \
  python "${HATCH_PY}" -lf

###############################################################################
# Check/parse command line arguments

if [ "${1}" == "" ]; then
  echo "$0: usage: $0 for simd_ext1,...,simd_ext2 [with compiler]"
  exit 0
fi

if [ "${1}" != "for" ]; then
  echo "ERROR: expected 'for' as first argument"
  exit 1
fi

if [ "${2}" == "" ]; then
  echo "ERROR: no SIMD extension given after 'for'"
  exit 1
fi
SIMD_EXTS=`echo "${2}" | sed -e 's,/, ,g'`

if [ "${3}" == "" ]; then
  COMPILER_ARG="gcc"
elif [ "${3}" == "with" ]; then
  if [ "${4}" == "" ]; then
    echo "ERROR: no compiler given after 'with'"
    exit 1
  fi
  COMPILER_ARG="${4}"
else
  echo "ERROR: expected 'with' as fourth argument"
  exit 1
fi

COMPILERS=`echo ${COMPILER_ARG} | sed 's,/, ,g'`
for compiler in ${COMPILERS}; do
  (${compiler} --version 1>/dev/null 2>/dev/null)
  if [ "$?" != "0" ]; then
    echo "ERROR: compiler ${compiler} not found in PATH"
    exit 1
  fi
done

###############################################################################
# Build NSIMD : one build directory per SIMD extension per compiler

for compiler in ${COMPILERS}; do
  for simd_ext in ${SIMD_EXTS}; do
    BUILD_DIR="${BUILD_ROOT}/build-${simd_ext}-${compiler}"
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    (cd "${BUILD_DIR}" && \
        "${NSCONFIG}" .. -Dsimd=${simd_ext} -comp=${compiler})
    (cd "${BUILD_DIR}" && ninja)
  done
done
