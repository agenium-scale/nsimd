#!/bin/bash
#
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

set -e
set -x

FIND_NSIMD_CMAKE="`dirname $0`/../scripts/FindNSIMD.cmake"
SIMD_EXTS="sse2"

for simd_ext in ${SIMD_EXTS}; do

  # First case: find a specific component
  ROOT_DIR="${PWD}/find_nsimd_cmake_tests/${simd_ext}"
  rm -rf ${ROOT_DIR}
  mkdir -p "${ROOT_DIR}/cmake"
  cp "${FIND_NSIMD_CMAKE}" "${ROOT_DIR}/cmake"
  mkdir -p "${ROOT_DIR}/root/include/nsimd"
  touch "${ROOT_DIR}/root/include/nsimd/nsimd.h"
  mkdir -p "${ROOT_DIR}/root/lib"
  touch "${ROOT_DIR}/root/lib/libnsimd_${simd_ext}.so"

  cat >"${ROOT_DIR}/CMakeLists.txt" <<-EOF
	cmake_minimum_required(VERSION 3.0.0)
	project(FIND_NSIMD_CMAKE_TESTS)
	set(CMAKE_MODULE_PATH "${ROOT_DIR}/cmake")
	set(CMAKE_PREFIX_PATH "${ROOT_DIR}/root")
	find_package(NSIMD COMPONENTS ${simd_ext})
	message(STATUS "FindNSIMD.cmake test : specific for ${simd_ext}")
	message(STATUS "NSIMD_FOUND = \${NSIMD_FOUND}")
	if (\${NSIMD_FOUND})
	  message(STATUS "NSIMD_INCLUDE_DIRS = \${NSIMD_INCLUDE_DIRS}")
	  message(STATUS "NSIMD_LIBRARY_DIRS = \${NSIMD_LIBRARY_DIRS}")
	  message(STATUS "NSIMD_LIBRARIES = \${NSIMD_LIBRARIES}")
	else()
	  message(FATAL_ERROR "error NSIMD_FOUND should be TRUE")
	endif()
	EOF
  (cd "${ROOT_DIR}" && mkdir -p build && cd build && cmake ..)

  # Second case: find a automatically a component
  ROOT_DIR="${PWD}/find_nsimd_cmake_tests/${simd_ext}-auto"
  rm -rf ${ROOT_DIR}
  mkdir -p "${ROOT_DIR}/cmake"
  cp "${FIND_NSIMD_CMAKE}" "${ROOT_DIR}/cmake"
  mkdir -p "${ROOT_DIR}/root/include/nsimd"
  touch "${ROOT_DIR}/root/include/nsimd/nsimd.h"
  mkdir -p "${ROOT_DIR}/root/lib"
  touch "${ROOT_DIR}/root/lib/libnsimd_${simd_ext}.so"

  cat >"${ROOT_DIR}/CMakeLists.txt" <<-EOF
	cmake_minimum_required(VERSION 3.0.0)
	project(FIND_NSIMD_CMAKE_TESTS)
	set(CMAKE_MODULE_PATH "${ROOT_DIR}/cmake")
	set(CMAKE_PREFIX_PATH "${ROOT_DIR}/root")
	find_package(NSIMD)
	message(STATUS "FindNSIMD.cmake test : automatic for ${simd_ext}")
	message(STATUS "NSIMD_FOUND = \${NSIMD_FOUND}")
	if (\${NSIMD_FOUND})
	  message(STATUS "NSIMD_INCLUDE_DIRS = \${NSIMD_INCLUDE_DIRS}")
	  message(STATUS "NSIMD_LIBRARY_DIRS = \${NSIMD_LIBRARY_DIRS}")
	  message(STATUS "NSIMD_LIBRARIES = \${NSIMD_LIBRARIES}")
	else()
	  message(FATAL_ERROR "error NSIMD_FOUND should be TRUE")
	endif()
	EOF
  (cd "${ROOT_DIR}" && mkdir -p build && cd build && cmake ..)

  # Third case: find a specific component
  ROOT_DIR="${PWD}/find_nsimd_cmake_tests/${simd_ext}-notfound"
  rm -rf ${ROOT_DIR}
  mkdir -p "${ROOT_DIR}/cmake"
  cp "${FIND_NSIMD_CMAKE}" "${ROOT_DIR}/cmake"
  mkdir -p "${ROOT_DIR}/root/include/nsimd"
  touch "${ROOT_DIR}/root/include/nsimd/nsimd.h"
  mkdir -p "${ROOT_DIR}/root/lib"
  touch "${ROOT_DIR}/root/lib/libnsimd_cpu.so"

  cat >"${ROOT_DIR}/CMakeLists.txt" <<-EOF
	cmake_minimum_required(VERSION 3.0.0)
	project(FIND_NSIMD_CMAKE_TESTS)
	set(CMAKE_MODULE_PATH "${ROOT_DIR}/cmake")
	set(CMAKE_PREFIX_PATH "${ROOT_DIR}/root")
	find_package(NSIMD COMPONENTS ${simd_ext})
	message(STATUS "FindNSIMD.cmake test : "
	               "notfound specific for ${simd_ext}")
	message(STATUS "NSIMD_FOUND = \${NSIMD_FOUND}")
	if (\${NSIMD_FOUND})
	  message(STATUS "NSIMD_INCLUDE_DIRS = \${NSIMD_INCLUDE_DIRS}")
	  message(STATUS "NSIMD_LIBRARY_DIRS = \${NSIMD_LIBRARY_DIRS}")
	  message(STATUS "NSIMD_LIBRARIES = \${NSIMD_LIBRARIES}")
	  message(FATAL_ERROR "error NSIMD_FOUND should be FALSE")
	else()
	  message(STATUS "NSIMD not found")
	endif()
	EOF
  (cd "${ROOT_DIR}" && mkdir -p build && cd build && cmake ..)

  # Fourth case: find a automatically a component
  ROOT_DIR="${PWD}/find_nsimd_cmake_tests/${simd_ext}-auto-notfound"
  rm -rf ${ROOT_DIR}
  mkdir -p "${ROOT_DIR}/cmake"
  cp "${FIND_NSIMD_CMAKE}" "${ROOT_DIR}/cmake"
  mkdir -p "${ROOT_DIR}/root/include/nsimd"
  touch "${ROOT_DIR}/root/include/nsimd/nsimd.h"
  mkdir -p "${ROOT_DIR}/root/lib"

  cat >"${ROOT_DIR}/CMakeLists.txt" <<-EOF
	cmake_minimum_required(VERSION 3.0.0)
	project(FIND_NSIMD_CMAKE_TESTS)
	set(CMAKE_MODULE_PATH "${ROOT_DIR}/cmake")
	set(CMAKE_PREFIX_PATH "${ROOT_DIR}/root")
	find_package(NSIMD)
	message(STATUS "FindNSIMD.cmake test : "
	               "notfound automatic for ${simd_ext}")
	message(STATUS "NSIMD_FOUND = \${NSIMD_FOUND}")
	if (\${NSIMD_FOUND})
	  message(STATUS "NSIMD_INCLUDE_DIRS = \${NSIMD_INCLUDE_DIRS}")
	  message(STATUS "NSIMD_LIBRARY_DIRS = \${NSIMD_LIBRARY_DIRS}")
	  message(STATUS "NSIMD_LIBRARIES = \${NSIMD_LIBRARIES}")
	  message(FATAL_ERROR "error NSIMD_FOUND should be FALSE")
	else()
	  message(STATUS "NSIMD not found")
	endif()
	EOF
  (cd "${ROOT_DIR}" && mkdir -p build && cd build && cmake ..)

done
