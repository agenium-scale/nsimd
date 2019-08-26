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

# FindSleef.cmake
# - Try to find Sleef
#
# The following variables are optionally searched for defaults
#  Sleef_ROOT_DIR:  Base directory where all Sleef components are found
#
# Once done this will define
#  Sleef_FOUND - System has Sleef
#  Sleef_INCLUDE_DIRS - The Sleef include directories
#  Sleef_LIBRARIES - The libraries needed to use Sleef

set(Sleef_ROOT_DIR "" CACHE PATH "Folder containing Sleef")

find_path(Sleef_INCLUDE_DIR "sleef.h"
  PATHS ${Sleef_ROOT_DIR} ${CMAKE_INCLUDE_PATH}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_path(Sleef_INCLUDE_DIR "sleef.h")

find_library(Sleef_LIBRARY NAMES "sleef"
  PATHS ${Sleef_ROOT_DIR} ${CMAKE_LIBRARY_PATH}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH)
find_library(Sleef_LIBRARY NAMES "sleef")

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set Sleef_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Sleef FOUND_VAR Sleef_FOUND
  REQUIRED_VARS Sleef_LIBRARY
  Sleef_INCLUDE_DIR)

if(Sleef_FOUND)
  set(Sleef_LIBRARIES ${Sleef_LIBRARY})
  set(Sleef_INCLUDE_DIRS ${Sleef_INCLUDE_DIR})
endif()

mark_as_advanced(Sleef_INCLUDE_DIR Sleef_LIBRARY)
