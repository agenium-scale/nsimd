#!/bin/bash
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

###############################################################################

cd `dirname $0`
set -x
set -e

###############################################################################
# Init

NSTOOLS_DIR="${PWD}/../nstools"

###############################################################################
# Build nsconfig (if not already built)

[ -d "${NSTOOLS_DIR}" ] || \
    ( cd "${PWD}/.." && \
      ( [ -d .git ] \
        && ( git clone `git remote get-url origin | sed s/nsimd/nstools/g` ) \
        || ( git clone "https://github.com/agenium-scale/nstools.git" ) ) )

if [ "${NSTOOLS_CHECKOUT_LAST_COMMIT}" == "" ]; then
  git -C "${NSTOOLS_DIR}" checkout v3.0
else
  git -C "${NSTOOLS_DIR}" checkout master
  git -C "${NSTOOLS_DIR}" pull
fi

( cd "${NSTOOLS_DIR}/nsconfig" && \
  make -B -j8 -f Makefile.nix nsconfig && \
  make -B -j8 -f Makefile.nix nstest )
