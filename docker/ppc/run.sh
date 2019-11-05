#!/bin/bash

set -euxo pipefail

# Download nsimd
mkdir src
cd src
git clone -b eschnett/vsx https://github.com/eschnett/nsimd.git
cd nsimd

# Patch to create static libraries and executables, and to run tests
# with qemu
perl -pi -e 's/SHARED/STATIC/' CMakeLists.txt
perl -pi -e 's/LIBRARY DESTINATION/ARCHIVE DESTINATION/' CMakeLists.txt
perl -pi -e 's/\$\{target\} \$\{libname\}/\$\{target\} \$\{libname\} "-static"/' CMakeLists.txt
perl -pi -e 's/add_test\(NAME \$\{target\} COMMAND/add_test\(NAME \$\{target\} COMMAND \/milton\/qemu\/bin\/qemu-ppc64le -cpu power9/' CMakeLists.txt

# Create source files
python3 egg/hatch.py --all --force --simd=vsx

# Configure
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=powerpc64le-linux-gnu-gcc-9 -DCMAKE_CXX_COMPILER=powerpc64le-linux-gnu-g++-9 -DCMAKE_C_FLAGS=-mcpu=power9 -DCMAKE_CXX_FLAGS=-mcpu=power9 -DSIMD=VSX -DCMAKE_INSTALL_PREFIX=/milton/nsimd ..

# Build library
make -j$(nproc)

# Build tests
make -j$(nproc) tests

# Run tests
ctest -V .

# Install library
make -j$(nproc) install
