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
#set -x
set -e

###############################################################################
# Init

BUILD_SH="${PWD}/build.sh"
BUILD_TESTS_SH="${PWD}/build-tests.sh"
HATCH_PY="${PWD}/../egg/hatch.py"
BUILD_ROOT="${PWD}/.."
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
NOW=`date +%Y%m%d%H%M%S`
GIT_URL=`git remote get-url origin`
if [ ${1} != "" ]; then
  GIT_BRANCH=${1}
else
  GIT_BRANCH=`git rev-parse --abbrev-ref HEAD`
fi
DWD="/home/${USER}/${NOW}"
CI_OUT_SH="${PWD}/ci-output.sh"

# Format is as follows:
# Lines are separated by ';' and columns by ':'
# Column 1 = name of the worker to ssh to
# Column 2 = working directory
# Column 3 = human readable description
# Column 4 = command to execute to perform tests
read -r -d '' WORKERS <<EOF
camelot:${DWD}:sse2-gcc:bash script/build-tests.sh for sse2 with gcc &&
                        cd build-sse2-gcc && ../nstools/bin/nstest -j80;
EOF

###############################################################################
# Launch jobs

echo "-- NSIMD CI"
echo "-- "
echo "-- Initialization:"

LINES=`echo ${WORKERS} | sed 's/;/\n/g'`
for l in ${LINES}; do
  ADDR=`echo ${l} | cut -f1 -d:`
  echo "-- Found new worker: ${ADDR}"
  WORK_DIR=`echo ${l} | cut -f2 -d:`
  echo "--   Working directory will be: ${WORK_DIR}"
  echo "--   Cloning NSIMD into ${WORK_DIR}"
  ${SSH} ${ADDR} "git clone ${GIT_URL} ${WORK_DIR}"
  ${SSH} ${ADDR} "cd ${WORK_DIR} && git checkout ${GIT_BRANCH}"
  echo "--   Launching commands"
  CMD=`echo ${l} | cut -f4 -d:`
  ${SSH} ${ADDR} "cd ${WORK_DIR} ; ( ( ${CMD} ) |& ${CI_OUT_SH} ) </dev/null
                  >/dev/null >/dev/null &"
done

###############################################################################
# Monitor jobs

LINES=`echo ${WORKERS} | sed 's/;/\n/g'`
while true; do
  for l in ${LINES}; do
    ADDR=`echo ${l} | cut -f1 -d:`
    DESC=`echo ${l} | cut -f3 -d:`
    WORK_DIR=`echo ${l} | cut -f2 -d:`
    ONE_LINER=`${SSH} ${ADDR} "cd ${WORK_DIR} ; cat one-liner.txt"`
    echo "${ADDR}, ${DESC}"
    echo "    `echo ${ONE_LINER} | cut -c 1-50`"
    echo
  done
  sleep 1
done
