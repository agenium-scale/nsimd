# MIT License
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
#
#.rst:
# FindNSIMD
# ---------
#
# Find the NSIMD library, Agenium Scale's vectorization library.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``NSIMD_INCLUDE_DIRS``
#   where to find nsimd.h, etc.
# ``NSIMD_LIBRARY_DIRS``
#   where to find the library to link against to use NSIMD.
# ``NSIMD_LIBRARIES``
#   the library to link against to use NSIMD.
# ``NSIMD_FOUND``
#   If false, do not try to use NSIMD.

if (NOT NSIMD_FOUND AND NOT DEFINED NSIMD_LIBRARIES)
  list(LENGTH NSIMD_FIND_COMPONENTS l)
  if ("${l}" STREQUAL "0")
    find_library(NSIMD_LIBRARIES NAMES nsimd_cpu
                                       nsimd_sse2
                                       nsimd_sse42
                                       nsimd_avx
                                       nsimd_avx2
                                       nsimd_avx512_knl
                                       nsimd_avx512_skylake
                                       nsimd_neon128
                                       nsimd_aarch64
                                       nsimd_sve
                                       nsimd_sve128
                                       nsimd_sve256
                                       nsimd_sve512
                                       nsimd_sve1024
                                       nsimd_sve2048
                                       nsimd_cuda
                                       nsimd_rocm
				       nsimd_oneapi)
  elseif("${l}" STREQUAL "1")
    list(GET NSIMD_FIND_COMPONENTS 0 simd_ext)
    find_library(NSIMD_LIBRARIES NAMES nsimd_${simd_ext})
  else()
    if (NOT NSIMD_FIND_QUIETLY)
      message(FATAL_ERROR "cannot handle several components")
    endif()
  endif()
endif()

if (NOT NSIMD_FOUND AND NOT DEFINED NSIMD_INCLUDE_DIRS)
  find_path(NSIMD_INCLUDE_DIRS NAMES nsimd/nsimd.h)
endif()

if (NOT "${NSIMD_INCLUDE_DIRS}" STREQUAL "NSIMD_INCLUDE_DIRS-NOTFOUND" AND
    NOT "${NSIMD_LIBRARIES}" STREQUAL "NSIMD_LIBRARIES-NOTFOUND")
  get_filename_component(NSIMD_LIBRARY_DIRS ${NSIMD_LIBRARIES} DIRECTORY)
  if (NOT NSIMD_FIND_QUIETLY)
    message(STATUS "[include dir = ${NSIMD_INCLUDE_DIRS}]"
                   " [library = ${NSIMD_LIBRARIES}]")
  endif()
  set(NSIMD_FOUND TRUE)
else()
  if (NOT NSIMD_FIND_QUIETLY)
    if (NOT DEFINED NSIMD_INCLUDE_DIRS)
      set(msg "[cannot determine include dir]")
    else()
      set(msg "[include dir = ${NSIMD_INCLUDE_DIRS}]")
    endif()
    if (NOT DEFINED NSIMD_LIBRARIES)
      set(msg "${msg} [cannot determine library dir]")
    else()
      set(msg "${msg} [library = ${NSIMD_LIBRARIES}]")
    endif()
    if (NSIMD_FIND_REQUIRED)
      message(FATAL_ERROR "${msg}")
    else()
      message(STATUS "${msg}")
    endif()
  endif()
  set(NSIMD_FOUND FALSE)
endif()

