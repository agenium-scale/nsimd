#!/bin/sh

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
git -C "${GH_PAGES_DIR}" add '*.html'
git -C "${GH_PAGES_DIR}" commit -m "Documentation auto-generated on `date`"
git -C "${GH_PAGES_DIR}" push
rm -rf "${GH_PAGES_DIR}"
