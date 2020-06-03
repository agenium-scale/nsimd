@echo off

REM Copyright (c) 2020 Agenium Scale
REM
REM Permission is hereby granted, free of charge, to any person obtaining a copy
REM of this software and associated documentation files (the "Software"), to deal
REM in the Software without restriction, including without limitation the rights
REM to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
REM copies of the Software, and to permit persons to whom the Software is
REM furnished to do so, subject to the following conditions:
REM
REM The above copyright notice and this permission notice shall be included in all
REM copies or substantial portions of the Software.
REM
REM THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
REM IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
REM FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
REM AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
REM LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
REM OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
REM SOFTWARE.

REM ###########################################################################

pushd "%~dp0"

REM ###########################################################################
REM Init

set SETUP_SH="%CD%/setup.sh"
set NSCONFIG="%CD%/../nstools/bin/nsconfig"
set HATCH_PY="%CD%/../egg/hatch.py"
set BUILD_ROOT="%CD%/.."

###############################################################################
# Generate NSIMD

python3 "%HATCH_PY%" -lf

###############################################################################
# Check/parse command line arguments

goto end_usage

:usage
echo "$0: usage: $0 for simd_ext1,...,simd_ext2 [with compiler]"
exit /B 0
:end_usage

if "%1" == "" goto usage
if "%1" == "--help" goto usage

if "${1}" != "for" (
  echo "ERROR: expected 'for' as first argument"
  exit /B 1
)  :wq


SIMD_EXTS=`echo "${2}" | sed -e 's/,/ /g'`

if [ "${3}" == "" ]; then
  COMPILER_ARG="gcc"
elif [ "${3}" == "with" ]; then
  COMPILER_ARG="${4}"
else
  echo "ERROR: expected 'with' as fourth argument"
  exit 1
fi

COMPILERS=`echo ${COMPILER_ARG} | sed 's/,/ /g'`
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
