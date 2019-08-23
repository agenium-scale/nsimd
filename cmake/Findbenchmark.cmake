## Copyright (c) 2019 Agenium Scale
## 
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to deal
## in the Software without restriction, including without limitation the rights
## to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
## copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
## 
## The above copyright notice and this permission notice shall be included in all
## copies or substantial portions of the Software.
## 
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
## OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
## 
## ------------------------------------------------------------------------------------------------

## From: https://github.com/google/benchmark/issues/188

# Findbenchmark.cmake
# - Try to find benchmark
#
# The following variables are optionally searched for defaults
#  benchmark_ROOT_DIR:  Base directory where all benchmark components are found
#
# Once done this will define
#  benchmark_FOUND - System has benchmark
#  benchmark_INCLUDE_DIRS - The benchmark include directories
#  benchmark_LIBRARIES - The libraries needed to use benchmark

set(benchmark_ROOT_DIR "" CACHE PATH "Folder containing benchmark")

find_path(benchmark_INCLUDE_DIR "benchmark/benchmark.h"
  PATHS ${benchmark_ROOT_DIR} ${CMAKE_INCLUDE_PATH}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_path(benchmark_INCLUDE_DIR "benchmark/benchmark.h")

find_library(benchmark_LIBRARY NAMES "benchmark"
  PATHS ${benchmark_ROOT_DIR} ${CMAKE_LIBRARY_PATH}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH)
find_library(benchmark_LIBRARY NAMES "benchmark")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set benchmark_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(benchmark FOUND_VAR benchmark_FOUND
  REQUIRED_VARS benchmark_LIBRARY
  benchmark_INCLUDE_DIR)

if(benchmark_FOUND)
  set(benchmark_LIBRARIES ${benchmark_LIBRARY})
  set(benchmark_INCLUDE_DIRS ${benchmark_INCLUDE_DIR})
endif()

mark_as_advanced(benchmark_INCLUDE_DIR benchmark_LIBRARY)
