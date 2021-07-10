# MIT License
#
# Copyright (c) 2021 Agenium Scale
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

cmake_minimum_required(VERSION 3.0.2)
project(NSIMD VERSION 3.0 LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# First check that NSIMD code has been generated

if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/nsimd/functions.h")
  if (WIN32)
    execute_process(COMMAND
                    python ${CMAKE_CURRENT_SOURCE_DIR}\\egg\\hatch.py -lf)
  else()
    execute_process(COMMAND
                    python3 ${CMAKE_CURRENT_SOURCE_DIR}/egg/hatch.py -lf)
  endif()
endif()

# -----------------------------------------------------------------------------
# Compilations options

option(NSIMD_ARM32_IS_ARMEL "Set whether ARM32 is in fact armel or armhf" ON)

function(nsimd_get_compiler_argument simd_ext argument)
  if (MSVC)
    if (CMAKE_CL_64)
      set(mapping_sse2 "/DSSE2")
      set(mapping_sse42 "/DSSE42")
    else()  
      set(mapping_sse2 "/DSSE2;/arch:SSE2")
      set(mapping_sse42 "/DSSE42;/arch:SSE2")
    endif()
    set(mapping_avx "/DAVX;/arch:AVX")
    set(mapping_avx2 "/DAVX2;/arch:AVX2")
    set(mapping_avx512_knl "/DAVX512_KNL;/arch:AVX512")
    set(mapping_avx512_skylake "/DAVX512_SKYLAKE;/arch:AVX512")
    set(mapping_neon128 "/DNEON128;/arch:VFPv4")
    set(mapping_aarch64 "/DAARCH64")
    set(mapping_sve "/DSVE")
    set(mapping_sve128 "/DSVE128")
    set(mapping_sve256 "/DSVE256")
    set(mapping_sve512 "/DSVE512")
    set(mapping_sve1024 "/DSVE1024")
    set(mapping_sve2048 "/DSVE2048")
    set(mapping_cuda "/DCUDA")
    set(mapping_rocm "/DROCM")
  else()
    set(mapping_sse2 "-DSSE2;-msse2" )
    set(mapping_sse42 "-DSSE42;-msse4.2" )
    set(mapping_avx "-DAVX;-mavx;-mno-avx256-split-unaligned-load"
                    ";-mno-avx256-split-unaligned-store" )
    set(mapping_avx2 "-DAVX2;-mavx2;-mfma;-mno-avx256-split-unaligned-load"
                     ";-mno-avx256-split-unaligned-store" )
    set(mapping_avx512_knl "-DAVX512_KNL;-mavx512f;-mavx512pf;-mavx512er"
                           ";-mavx512cd")
    set(mapping_avx512_skylake "-DAVX512_SKYLAKE;-mavx512f;-mavx512dq"
                               ";-mavx512cd;-mavx512bw;-mavx512vl")
    if (NSIMD_ARM32_IS_ARMEL)
      set(mapping_neon128 "-DNEON128;-mfloat-abi=softfp;-mfpu=neon")
    else()
      set(mapping_neon128 "-DNEON128;-mfpu=neon")
    endif()
    set(mapping_aarch64 "-DAARCH64")
    set(mapping_sve "-DSVE;-march=armv8.2-a+sve")
    set(mapping_sve128 "-DSVE128;-march=armv8.2-a+sve;-msve-vector-bits=128")
    set(mapping_sve256 "-DSVE256;-march=armv8.2-a+sve;-msve-vector-bits=256")
    set(mapping_sve512 "-DSVE512;-march=armv8.2-a+sve;-msve-vector-bits=512")
    set(mapping_sve1024 "-DSVE1024;-march=armv8.2-a+sve"
                        ";-msve-vector-bits=1024")
    set(mapping_sve2048 "-DSVE2048 -march=armv8.2-a+sve"
                        ";-msve-vector-bits=2048")
    set(mapping_cuda "-DCUDA")
    set(mapping_rocm "-DROCM")
  endif()
  if (DEFINED mapping_${simd_ext})
    set(${argument} "${mapping_${simd_ext}}" PARENT_SCOPE)
  else()
    if (MSVC)
      set(${argument} "/DCPU" PARENT_SCOPE)
    else()
      set(${argument} "-DCPU" PARENT_SCOPE)
    endif()
  endif()
endfunction()

if (NOT DEFINED simd)
  set(simd "cpu")
endif()
nsimd_get_compiler_argument(${simd} NSIMD_COMPILATION_OPTIONS)

# -----------------------------------------------------------------------------
# Object file selection

set(NSIMD_OBJS "fp16;memory;api_cpu;rempitab;sleefsp;sleefdp")

if ("${simd}" STREQUAL "sse2")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_sse2;sleef_sse2_f32;sleef_sse2_f64")
elseif ("${simd}" STREQUAL "sse42")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_sse2;api_sse42;"
                 "sleef_sse2_f32;sleef_sse2_f64;"
                 "sleef_sse42_f32;sleef_sse42_f64")
elseif ("${simd}" STREQUAL "avx")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_sse2;api_sse42;api_avx;"
                 "sleef_sse2_f32;sleef_sse2_f64;"
                 "sleef_sse42_f32;sleef_sse42_f64;"
                 "sleef_avx_f32;sleef_avx_f64")
elseif ("${simd}" STREQUAL "avx2")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_sse2;api_sse42;api_avx;api_avx2;"
                 "sleef_sse2_f32;sleef_sse2_f64;"
                 "sleef_sse42_f32;sleef_sse42_f64;"
                 "sleef_avx_f32;sleef_avx_f64;"
                 "sleef_avx2_f32;sleef_avx2_f64")
elseif ("${simd}" STREQUAL "avx512_knl")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_sse2;api_sse42;api_avx;api_avx2"
                 "sleef_sse2_f32;sleef_sse2_f64;"
                 "sleef_sse42_f32;sleef_sse42_f64;"
                 "sleef_avx_f32;sleef_avx_f64;"
                 "sleef_avx2_f32;sleef_avx2_f64;"
                 "api_avx512_knl;sleef_avx512_knl_f32;sleef_avx512_knl_f64")
elseif ("${simd}" STREQUAL "avx512_skylake")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_sse2;api_sse42;api_avx;api_avx2;"
                 "api_avx512_skylake;sleef_avx512_skylake_f32;"
                 "sleef_sse2_f32;sleef_sse2_f64;"
                 "sleef_sse42_f32;sleef_sse42_f64;"
                 "sleef_avx_f32;sleef_avx_f64;"
                 "sleef_avx2_f32;sleef_avx2_f64;"
                 "sleef_avx512_skylake_f64")
elseif ("${simd}" STREQUAL "neon128")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_neon128;"
                 "sleef_neon128_f32;sleef_neon128_f64")
elseif ("${simd}" STREQUAL "aarch64")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_aarch64;"
                 "sleef_aarch64_f32;sleef_aarch64_f64")
elseif ("${simd}" STREQUAL "sve")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_aarch64;api_sve;"
                 "sleef_aarch64_f32;sleef_aarch64_f64;"
                 "sleef_sve_f32;sleef_sve_f64")
elseif ("${simd}" STREQUAL "sve128")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_aarch64;api_sve128;"
                 "sleef_aarch64_f32;sleef_aarch64_f64;"
                 "sleef_sve128_f32;sleef_sve128_f64")
elseif ("${simd}" STREQUAL "sve256")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_aarch64;api_sve256;"
                 "sleef_aarch64_f32;sleef_aarch64_f64;"
                 "sleef_sve256_f32;sleef_sve256_f64")
elseif ("${simd}" STREQUAL "sve512")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_aarch64;api_sve512;"
                 "sleef_aarch64_f32;sleef_aarch64_f64;"
                 "sleef_sve512_f32;sleef_sve512_f64")
elseif ("${simd}" STREQUAL "sve1024")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_aarch64;api_sve1024;"
                 "sleef_aarch64_f32;sleef_aarch64_f64;"
                 "sleef_sve1024_f32;sleef_sve1024_f64")
elseif ("${simd}" STREQUAL "sve2048")
  set(NSIMD_OBJS "${NSIMD_OBJS};api_aarch64;api_sve2048;"
                 "sleef_aarch64_f32;sleef_aarch64_f64;"
                 "sleef_sve2048_f32;sleef_sve2048_f64")
endif()

# -----------------------------------------------------------------------------
# Rules for building the library

set(NSIMD_LIB_DEPS "")
foreach(o ${NSIMD_OBJS})
  if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${o}.cpp")
    add_library(${o} OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/src/${o}.cpp")
  elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/${o}.c")
    add_library(${o} OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/src/${o}.c")
  elseif(o MATCHES "sleef_.*_f32")
    add_library(${o} OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/src/sleefsimdsp.c")
  elseif(o MATCHES "sleef_.*_f64")
    add_library(${o} OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/src/sleefsimddp.c")
  endif()
  if (MSVC)
    set(sleef_cflags "/DNDEBUG;/DDORENAME=1")
  else()
    set(sleef_cflags "-DNDEBUG;-DDORENAME=1")
  endif()
  set_property(TARGET ${o} PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${o} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  if (MSVC)
    target_compile_definitions(${o} PUBLIC "/D_CRT_SECURE_NO_WARNINGS")
  endif()
  set(buf "")
  if ("${o}" STREQUAL "api_sse2")
    nsimd_get_compiler_argument("sse2" buf)
  elseif ("${o}" STREQUAL "api_sse42")
    nsimd_get_compiler_argument("sse42" buf)
  elseif ("${o}" STREQUAL "api_avx")
    nsimd_get_compiler_argument("avx" buf)
  elseif ("${o}" STREQUAL "api_avx2")
    nsimd_get_compiler_argument("avx2" buf)
  elseif ("${o}" STREQUAL "api_avx512_knl")
    nsimd_get_compiler_argument("avx512_knl" buf)
  elseif ("${o}" STREQUAL "api_avx512_skylake")
    nsimd_get_compiler_argument("avx512_skylake" buf)
  elseif ("${o}" STREQUAL "api_neon128")
    nsimd_get_compiler_argument("neon128" buf)
  elseif ("${o}" STREQUAL "api_aarch64")
    nsimd_get_compiler_argument("aarch64" buf)
  elseif ("${o}" STREQUAL "api_sve")
    nsimd_get_compiler_argument("sve" buf)
  elseif ("${o}" STREQUAL "api_sve128")
    nsimd_get_compiler_argument("sve128" buf)
  elseif ("${o}" STREQUAL "api_sve256")
    nsimd_get_compiler_argument("sve256" buf)
  elseif ("${o}" STREQUAL "api_sve512")
    nsimd_get_compiler_argument("sve512" buf)
  elseif ("${o}" STREQUAL "api_sve1024")
    nsimd_get_compiler_argument("sve1024" buf)
  elseif ("${o}" STREQUAL "api_sve2048")
    nsimd_get_compiler_argument("sve2048" buf)
  elseif ("${o}" STREQUAL "api_cuda")
    nsimd_get_compiler_argument("cuda" buf)
  elseif ("${o}" STREQUAL "api_rocm")
    nsimd_get_compiler_argument("rocm" buf)
  elseif ("${o}" STREQUAL "api_cpu")
    nsimd_get_compiler_argument("cpu" buf)
  elseif ("${o}" STREQUAL "rempitab")
    list(APPEND buf "${sleef_cflags}")
  elseif ("${o}" STREQUAL "sleefsp")
    list(APPEND buf "${sleef_cflags}")
  elseif ("${o}" STREQUAL "sleefdp")
    list(APPEND buf "${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_sse2_")
    nsimd_get_compiler_argument("sse2" buf)
    list(APPEND buf "-DNSIMD_SSE2;-DENABLE_SSE2=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_sse42_")
    nsimd_get_compiler_argument("sse42" buf)
    list(APPEND buf "-DNSIMD_SSE42;-DENABLE_SSE4=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_avx_")
    nsimd_get_compiler_argument("avx" buf)
    list(APPEND buf "-DNSIMD_AVX;-DENABLE_AVX=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_avx2_")
    nsimd_get_compiler_argument("avx2" buf)
    list(APPEND buf "-DNSIMD_AVX2;-DENABLE_AVX2=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_avx512_knl_")
    nsimd_get_compiler_argument("avx512_knl" buf)
    list(APPEND buf "-DNSIMD_AVX512_KNL;-DENABLE_AVX512F=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_avx512_skylake_")
    nsimd_get_compiler_argument("avx512_skylake" buf)
    list(APPEND buf
         "-DNSIMD_AVX512_SKYLAKE;-DENABLE_AVX512F=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_neon128_")
    nsimd_get_compiler_argument("neon128" buf)
    list(APPEND buf "-DNSIMD_NEON128;-DENABLE_NEON32=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_aarch64_")
    nsimd_get_compiler_argument("aarch64" buf)
    list(APPEND buf "-DNSIMD_AARCH64;-DENABLE_ADVSIMD=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_sve_")
    nsimd_get_compiler_argument("sve" buf)
    list(APPEND buf "-DNSIMD_SVE;-DENABLE_SVE=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_sve128_")
    nsimd_get_compiler_argument("sve128" buf)
    list(APPEND buf "-DNSIMD_SVE128;-DENABLE_SVE=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_sve256_")
    nsimd_get_compiler_argument("sve256" buf)
    list(APPEND buf "-DNSIMD_SVE256;-DENABLE_SVE=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_sve512_")
    nsimd_get_compiler_argument("sve512" buf)
    list(APPEND buf "-DNSIMD_SVE512;-DENABLE_SVE=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_sve1024_")
    nsimd_get_compiler_argument("sve1024" buf)
    list(APPEND buf "-DNSIMD_SVE1024;-DENABLE_SVE=1;${sleef_cflags}")
  elseif ("${o}" MATCHES "sleef_sve2048_")
    nsimd_get_compiler_argument("sve2048" buf)
    list(APPEND buf "-DNSIMD_SVE2048;-DENABLE_SVE=1;${sleef_cflags}")
  else()
    set(buf "")
  endif()
  message(STATUS "DEBUG: ${o} --> ${buf}")
  if (NOT "${buf}" STREQUAL "")
    target_compile_options(${o} PUBLIC "${buf}")
  endif()
  list(APPEND NSIMD_LIB_DEPS "$<TARGET_OBJECTS:${o}>")
endforeach()

set(NSIMD_LIB_TARGET "nsimd_${simd}")
add_library(${NSIMD_LIB_TARGET} SHARED ${NSIMD_LIB_DEPS})

# -----------------------------------------------------------------------------
# Installation stuff

if (WIN32)
  install(TARGETS ${NSIMD_LIB_TARGET} RUNTIME DESTINATION lib
                                      ARCHIVE DESTINATION lib)
else()
  install(TARGETS ${NSIMD_LIB_TARGET} LIBRARY DESTINATION lib)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/nsimd
        DESTINATION include)
