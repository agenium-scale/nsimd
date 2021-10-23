#!/bin/bash

set -euxo pipefail

# Download nsimd
mkdir src
cd src
git clone https://github.com/agenium-scale/nsimd.git
cd nsimd

# Patch to create static libraries and executables
perl -pi -e 's/SHARED/STATIC/' CMakeLists.txt
perl -pi -e 's/LIBRARY DESTINATION/ARCHIVE DESTINATION/' CMakeLists.txt
perl -pi -e 's/\$\{target\} \$\{libname\}/\$\{target\} \$\{libname\} "-static"/' CMakeLists.txt

# Create source files
python3 egg/hatch.py --all --force --simd=neon128

# Configure
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc-9 -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++-9 -DSIMD=NEON128 -DCMAKE_INSTALL_PREFIX=/milton/nsimd ..

# Build library
make -j$(nproc)

# Build tests
make -j$(nproc) tests

# Run tests
ctest -V .

# Install library
make -j$(nproc) install