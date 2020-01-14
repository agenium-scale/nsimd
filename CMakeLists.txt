# Copyright (c) 2019 Agenium Scale
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
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

## Setup project
## ----------------------------------------------------------------------------

project(nsimd LANGUAGES C CXX CUDA)
set(NSIMD_ROOT ${CMAKE_SOURCE_DIR})
if (NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

## Dependencies
## ----------------------------------------------------------------------------

## Tells cmake where to find Find*.cmake files
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")

## Tests dependencies
find_package(MPFR)

## Benchmark dependencies
if (ENABLE_BENCHMARK)
    ## Force it for benches
    set(CMAKE_BUILD_TYPE Release)
    find_package(benchmark REQUIRED)
    find_package(Sleef REQUIRED)
    find_package(MIPP REQUIRED)
    if (NOT MSVC)
        set(PTHREAD_LIBRARY pthread)
    endif()
    message(STATUS "Benchmarks are enabled")
else()
    message(STATUS "Benchmarks are disabled")
endif()

## Compiler flags
## ----------------------------------------------------------------------------

include(CompilerFlags)
remove_some_msvc_flags()
compiler_flags(C++14 flag_cpp14)
compiler_flags(O3    flag_O3)
compiler_flags(G     flag_g)
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_definitions(-D_SCL_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_WARNINGS)
endif()

## SIMD setup
## ----------------------------------------------------------------------------

include(SIMD)
if (NOT DEFINED SIMD)
    set(SIMD CPU)
endif()
if (NOT DEFINED SIMD_OPTIONAL)
    set(SIMD_OPTIONAL "")
endif()
get_simd_infos(SIMD_INFO "${SIMD}" "${SIMD_OPTIONALS}")
if (WIN32)
    set(libname "libnsimd_${SIMD_INFO_PLATFORM}")
else()
    set(libname "nsimd_${SIMD_INFO_PLATFORM}")
endif()

## Check that egg/hatch.py has been run
## ----------------------------------------------------------------------------

file(TO_NATIVE_PATH "${NSIMD_ROOT}/include/nsimd/functions.h"
     TEST_FILE)
if(NOT EXISTS "${TEST_FILE}")
    message(FATAL_ERROR
            "\nIt seems that you did not generate C/C++ files,\n"
            "Run 'python3 egg/hatch.py -Af' for full generation\n"
            "and then rerun CMake.")
endif()

## CMake helpers
## ----------------------------------------------------------------------------

## Make pretty target from path
function(make_target_path root src OUT)
    ## Remove root path
    string(REPLACE ${root}/ "" relative ${src})
    ## Replace / by .
    string(REPLACE "/" "." target ${relative})
    ## Returns
    set(${OUT} ${target} PARENT_SCOPE)
endfunction()

## Make pretty test targets
function(make_tests root pattern lang)
    ## Get all files
    file(GLOB_RECURSE srcs ${root}/${pattern})
    ## List of functions that requires MPFR (use | to chain them)
    set(requires_mpfr_pattern "sqrt")
    set(requires_mpfr_r "^.*\\.${requires_mpfr_pattern}\\..*$")
    ## Iterates over and create the target
    foreach(src ${srcs})
        ## Sanitize
        make_target_path(${root} ${src} test.target)
        ## Create test target name
        set(target tests.${test.target})
        ## Check if requires MPFR
        set(target.libs)
        if (${test.target} MATCHES ${requires_mpfr_r})
            ## If the test requires MPFR but no MPFR has been found, log it
            if (NOT MPFR_FOUND)
                message(STATUS "WARNING: Test ${test.target} requires MPFR, "
                               "but MPFR is missing")
                continue()
            endif()
            set(target.libs ${MPFR_LIBRARY})
        endif()
        ## Link with libmath on Linux
        set(libm)
        if (NOT WIN32)
            set(libm -lm)
        endif()
        ## Create test target
        set(target tests.${test.target})
        add_executable(${target} ${src})
        set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
        target_link_libraries(${target} ${libname} ${libm} ${target.libs})

        if (lang STREQUAL "C++")
          set_target_properties(${target} PROPERTIES COMPILE_FLAGS
              "${flag_g} ${flag_O3} ${SIMD_INFO_FLAGS} ${flag_cpp14}")
        elseif(lang STREQUAL "C")
          set_target_properties(${target} PROPERTIES COMPILE_FLAGS
              "${flag_g} ${flag_O3} ${SIMD_INFO_FLAGS}")
        elseif(lang STREQUAL "CUDA")
          set_target_properties(${target} PROPERTIES COMPILE_FLAGS
          "${flag_g} ${flag_O3} -std=c++03 -lineinfo")
        endif()

        ## Attach to the main tests target
        add_dependencies(tests ${target})
        ## Add it to CTest
        add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}>)
    endforeach()
endfunction()

## Make pretty bench targets
function(make_benches root pattern)
    ## Get all files
    file(GLOB_RECURSE srcs "${root}/${pattern}")
    ## Iterates over and create the target
    foreach(src ${srcs})
        ## Sanitize
        make_target_path(${root} ${src} bench.target)
        ## Create bench target
        set(target benches.${bench.target})
        add_executable(${target} ${src})
        set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
        target_link_libraries(${target} ${libname} ${benchmark_LIBRARY}
                              ${PTHREAD_LIBRARY} ${Sleef_LIBRARIES})
        target_include_directories(${target} PRIVATE ${benchmark_INCLUDE_DIRS}
                                   ${Sleef_INCLUDE_DIRS} ${MIPP_INCLUDE_DIRS})
        set_target_properties(${target} PROPERTIES COMPILE_FLAGS
                              "${flag_cpp14} ${flag_O3} ${SIMD_INFO_FLAGS}")
        ## Attach to the main benches target
        add_dependencies(benches ${target})
    endforeach()
endfunction()

## Setup library
## ----------------------------------------------------------------------------

include_directories(${NSIMD_ROOT}/include)
set(nsimd.src
    ${NSIMD_ROOT}/src/fp16.cpp
    ${NSIMD_ROOT}/src/memory.cpp
)
foreach(simd_ext ${SIMD_INFO_LIST})
    string(TOLOWER "${simd_ext}" low_simd_ext)
    set(nsimd.src "${nsimd.src};${NSIMD_ROOT}/src/api_${low_simd_ext}.cpp")
endforeach()
add_library(${libname} SHARED ${nsimd.src})
set_target_properties(${libname} PROPERTIES COMPILE_FLAGS
                      "${flag_cpp14} ${flag_O3} ${SIMD_INFO_FLAGS}")

## Setup source generation target
## ----------------------------------------------------------------------------

add_custom_target(gen)
add_custom_command(TARGET gen
    COMMAND ${PYTHON_EXECUTABLE} ${HATCH_PY} -Ccsa
    COMMAND cmake -E touch "${CMAKE_CURRENT_LIST_FILE}"
)

## Setup tests generation target
## ----------------------------------------------------------------------------

## Enable CTest
enable_testing()

add_custom_target(gen.tests)
add_custom_target(tests)
add_custom_command(TARGET gen.tests
    COMMAND ${PYTHON_EXECUTABLE} ${HATCH_PY} --simd=${SIMD_INFO_HATCH} -ta
    COMMAND cmake -E touch "${CMAKE_CURRENT_LIST_FILE}"
)

## Create targets
make_tests(${NSIMD_ROOT}/tests *.c "C")
make_tests(${NSIMD_ROOT}/tests *.cpp "C++")
make_tests(${NSIMD_ROOT}/tests *.cu "CUDA")

## Setup benchmarks generation target
## ----------------------------------------------------------------------------

if (ENABLE_BENCHMARK)
    include_directories(${benchmark_INCLUDE_DIRS})

    add_custom_target(gen.benches)
    add_custom_target(benches)
    add_custom_command(TARGET gen.benches
        COMMAND ${PYTHON_EXECUTABLE} ${HATCH_PY} --simd=${SIMD_INFO_HATCH} -ba
        COMMAND cmake -E touch "${CMAKE_CURRENT_LIST_FILE}"
    )

    ## Create targets
    make_benches(${NSIMD_ROOT}/benches "*.${SIMD_INFO_HATCH}.*.cpp")
endif()

## Setup doc generation target
## ----------------------------------------------------------------------------

add_custom_target(gen.doc)
add_custom_command(TARGET gen.doc
    COMMAND ${PYTHON_EXECUTABLE} ${HATCH_PY} -d
)

add_custom_target(doc)
add_custom_command(TARGET doc
    COMMAND ${CMAKE_COMMAND} -E copy ${NSIMD_ROOT}/DOC.md ${NSIMD_ROOT}/index.md
)
add_dependencies(doc gen.doc)

## Install
## ----------------------------------------------------------------------------

if (WIN32)
    install(TARGETS ${libname} RUNTIME DESTINATION lib)
else()
    install(TARGETS ${libname} LIBRARY DESTINATION lib)
endif()
install(DIRECTORY include/nsimd DESTINATION include)

## Clean
## ----------------------------------------------------------------------------

file(GLOB nsimd.api.src ${NSIMD_ROOT}/src/api_*)

add_custom_target(distclean)
add_custom_command(TARGET distclean
    COMMAND ${CMAKE_COMMAND} -E remove ${nsimd.api.src}
    COMMAND ${CMAKE_COMMAND} -E remove ${NSIMD_ROOT}/include/nsimd/functions.h
    COMMAND ${CMAKE_COMMAND} -E remove ${NSIMD_ROOT}/include/nsimd/cxx_adv_api_functions.hpp
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${NSIMD_ROOT}/include/nsimd/x86
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${NSIMD_ROOT}/include/nsimd/arm
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${NSIMD_ROOT}/include/nsimd/cpu
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${NSIMD_ROOT}/tests/c_base
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${NSIMD_ROOT}/tests/cxx_adv
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${NSIMD_ROOT}/tests/cxx_base
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${NSIMD_ROOT}/benches/cxx_adv
    COMMAND cmake -E touch "${CMAKE_CURRENT_LIST_FILE}"
)
