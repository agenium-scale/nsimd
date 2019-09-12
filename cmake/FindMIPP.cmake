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

# FindMIPP.cmake
# - Try to find MIPP
#
# The following variables are optionally searched for defaults
#  MIPP_ROOT_DIR:  Base directory where all MIPP components are found
#
# Once done this will define
#  MIPP_FOUND - System has MIPP
#  MIPP_INCLUDE_DIRS - The MIPP include directories

set(MIPP_ROOT_DIR "" CACHE PATH "Folder containing MIPP")

find_path(MIPP_INCLUDE_DIR "mipp.h"
  PATHS ${MIPP_ROOT_DIR} ${CMAKE_INCLUDE_PATH}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_path(MIPP_INCLUDE_DIR "mipp.h")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set MIPP_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(MIPP FOUND_VAR MIPP_FOUND
  REQUIRED_VARS
  MIPP_INCLUDE_DIR)

if(MIPP_FOUND)
  set(MIPP_INCLUDE_DIRS ${MIPP_INCLUDE_DIR})
endif()

mark_as_advanced(MIPP_INCLUDE_DIR)
