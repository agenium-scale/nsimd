#!/bin/sh

## The top-level dir
ROOT_DIR="$( git rev-parse --show-toplevel )"

## Where all the deps are gonna be installed
INSTALL_DIR="${ROOT_DIR}/_install"

get() {
    URL="$1"
    DEST="$2"
    ## Shift to consume all remaining arguments for cmake
    shift; shift;
    ## Get the repo
    git clone ${URL} ${DEST}
    ## Prepare build
    cd ${DEST}
    mkdir -p build
    cd build
    ## Make sure to install in the INSTALL_DIR dir
    cmake .. -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" "$@"
    cmake --build . --target install 
}

## Prepare deps dir + Cleanup
rm -rf ${INSTALL_DIR}
mkdir -p _deps
mkdir -p _install/lib _install/include

## MIPP
git clone https://github.com/aff3ct/MIPP.git _deps/mipp
cp -rfv _deps/mipp/src/* ${INSTALL_DIR}/include

## Sleef
get https://github.com/shibatch/sleef.git _deps/sleef -DBUILD_TESTS=OFF -DBUILD_DFT=OFF

## Benchmark
get https://github.com/google/benchmark _deps/benchmark -DBENCHMARK_ENABLE_TESTING=OFF
