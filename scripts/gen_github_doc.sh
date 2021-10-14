#!/bin/sh
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

###############################################################################

cd `dirname $0`
set -x
set -e

###############################################################################
# Init

GH_PAGES_DIR="${PWD}/gh-pages"
HTML_DOC_DIR="${PWD}/../doc/html"
EGG_DIR="${PWD}/../egg"

###############################################################################
# Generates HTML documentation

rm -f "${HTML_DOC_DIR}/*.html"
python3 "${EGG_DIR}/hatch.py" -fd

###############################################################################
# Put all HTML files into the gh-pages branch of NSIMD

rm -rf "${GH_PAGES_DIR}"
git clone git@github.com:agenium-scale/nsimd.git "${GH_PAGES_DIR}"
git -C "${GH_PAGES_DIR}" checkout gh-pages
git -C "${GH_PAGES_DIR}" rm -f '*.html'
cp ${HTML_DOC_DIR}/*.html ${GH_PAGES_DIR}
mkdir -p ${GH_PAGES_DIR}/assets
cp ${HTML_DOC_DIR}/assets/*.js ${GH_PAGES_DIR}/assets
git -C "${GH_PAGES_DIR}" add '*.html'
git -C "${GH_PAGES_DIR}" add 'assets/*.js'
git -C "${GH_PAGES_DIR}" commit -m "Documentation auto-generated on `date`"
git -C "${GH_PAGES_DIR}" push
rm -rf "${GH_PAGES_DIR}"
