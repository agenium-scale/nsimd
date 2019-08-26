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

# FindMPFR.cmake
# - Try to find MPFR
#
# The following variables are optionally searched for defaults
#  MPFR_ROOT_DIR:  Base directory where all MPFR components are found
#
# Once done this will define
#  MPFR_FOUND - System has MPFR
#  MPFR_INCLUDE_DIRS - The MPFR include directories
#  MPFR_LIBRARIES - The libraries needed to use MPFR

set(MPFR_ROOT_DIR "" CACHE PATH "Folder containing MPFR")

find_path(MPFR_INCLUDE_DIR "mpfr.h"
  PATHS ${MPFR_ROOT_DIR} ${CMAKE_INCLUDE_PATH}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_path(MPFR_INCLUDE_DIR "mpfr.h")

find_library(MPFR_LIBRARY NAMES "mpfr"
  PATHS ${MPFR_ROOT_DIR} ${CMAKE_LIBRARY_PATH}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH)
find_library(MPFR_LIBRARY NAMES "mpfr")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set MPFR_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(MPFR FOUND_VAR MPFR_FOUND
  REQUIRED_VARS MPFR_LIBRARY
  MPFR_INCLUDE_DIR)

if(MPFR_FOUND)
  set(MPFR_LIBRARIES ${MPFR_LIBRARY})
  set(MPFR_INCLUDE_DIRS ${MPFR_INCLUDE_DIR})
endif()

mark_as_advanced(MPFR_INCLUDE_DIR MPFR_LIBRARY)
