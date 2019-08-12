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

function(compiler_flags in out)

    # Empty in => empty out
    if ("${in}" MATCHES "^$")
        set(${out} "" PARENT_SCOPE)
        return()
    endif()

    string(TOUPPER "${in}" flag)

    # Sanity check first
    set(known_flags CPU SSE2 SSE42 AVX AVX2 AVX512_KNL AVX512_SKYLAKE NEON128
                    AARCH64 SVE FMA FP16 C++14 O3 G)

    list(FIND known_flags "${flag}" i)
    if (i EQUAL -1)
        message(FATAL_ERROR "Unknown/unsupported compiler flags: '${flag}'")
    endif()

    # Compiler flags for MSVC
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")

        set(flags_for_CPU            "")
        if ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
            set(flags_for_SSE2       "/arch:SSE2 /DSSE2")
        else()
            set(flags_for_SSE2       "/DSSE2")
        endif()

        set(flags_for_SSE42          "")
        set(flags_for_AVX            "/arch:AVX /DAVX")
        set(flags_for_AVX2           "/arch:AVX2 /DAVX2")
        set(flags_for_AVX512_KNL     "/arch:AVX512 /DAVX512_KNL")
        set(flags_for_AVX512_SKYLAKE "/arch:AVX512 /DAVX512_SKYLAKE")
        if(MSVC_VERSION GREATER 1900)
            set(flags_for_C++14      "/std:c++14 /Zc:__cplusplus")
        elseif (MSVC_VERSION EQUAL 1900)
            set(flags_for_C++14      "/std:c++14")
        else()
            set(flags_for_C++14      "")
        endif()
        set(flags_for_NEON128        "/DNEON128")
        set(flags_for_AARCH64        "/DAARCH64")
        set(flags_for_SVE            "/DSVE") # Not supported by MSVC yet
        set(flags_for_FP16           "/DFP16")
        set(flags_for_FMA            "/DFMA")
        set(flags_for_O3             "/Ox")
        set(flags_for_G              "/Zi")

    # Compiler flags for GCC and Clang
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR 
            CMAKE_CXX_COMPILER_ID MATCHES "GNU")

        set(flags_for_CPU            "")
        set(flags_for_SSE2           "-msse2 -DSSE2")
        set(flags_for_SSE42          "-msse4.2 -DSSE42")
        set(flags_for_AVX            "-mavx -DAVX")
        set(flags_for_AVX2           "-mavx2 -DAVX2")
        set(flags_for_AVX512_KNL     "-mavx512f -mavx512pf -mavx512er -mavx512cd -DAVX512_KNL")
        set(flags_for_AVX512_SKYLAKE "-mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -DAVX512_SKYLAKE")
        set(flags_for_C++14          "-std=c++14")
        set(flags_for_NEON128        "-mfpu=neon -DNEON128")
        set(flags_for_AARCH64        "-DAARCH64")
        set(flags_for_SVE            "-march=armv8-a+sve -DSVE")
        if ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(arm|ARM)")
            if ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
                set(flags_for_FP16   "-DFP16")
                set(flags_for_FMA    "-DFMA")
            else()
                set(flags_for_FP16   "-march=native+fp16 -DFP16")
                set(flags_for_FMA    "-DFMA")
            endif()
        else()
            set(flags_for_FP16       "-mf16c -DFP16")
            set(flags_for_FMA        "-mfma -DFMA")
        endif()
        set(flags_for_O3             "-O3")
        set(flags_for_G              "-g")

    # Compiler flags for Intel
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")

        set(flags_for_CPU            "")
        set(flags_for_SSE2           "-msse2 -DSSE2")
        set(flags_for_SSE42          "-msse4.2 -DSSE42")
        set(flags_for_AVX            "-mavx -DAVX")
        set(flags_for_AVX2           "-mavx2 -DAVX2")
        set(flags_for_AVX512_KNL     "-mavx512f -mavx512pf -mavx512er -mavx512cd -march=skylake-avx512 -DAVX512_KNL")
        set(flags_for_AVX512_SKYLAKE "-mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl -march=skylake-avx512 -DAVX512_SKYLAKE")
        set(flags_for_C++14          "-std=c++14")
        set(flags_for_NEON128        "")
        set(flags_for_AARCH64        "")
        set(flags_for_SVE            "")
        set(flags_for_FP16           "-mf16c -DFP16")
        set(flags_for_FMA            "-mfma -DFMA")
        set(flags_for_O3             "-O3")
        set(flags_for_G              "-g")

    # Unsupported compiler
    else()

        message(FATAL_ERROR "Compiler not supported yet")

    endif()

    # Return compiler flags
    set(${out} "${flags_for_${flag}}" PARENT_SCOPE)

endfunction()

## ----------------------------------------------------------------------------

function(remove_some_msvc_flags)
    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        foreach(lang C CXX)
            foreach(mode RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
                set(flags "${CMAKE_${lang}_FLAGS_${mode}}")
                string(REPLACE "/RTC1" "" flags "${flags}")
                string(REPLACE "/Ob0" "" flags "${flags}")
                string(REPLACE "/Ob1" "" flags "${flags}")
                string(REPLACE "/Ob2" "" flags "${flags}")
                string(REPLACE "/O1" "" flags "${flags}")
                string(REPLACE "/O2" "" flags "${flags}")
                string(REPLACE "/Od" "" flags "${flags}")
                set(CMAKE_${lang}_FLAGS_${mode} "${flags}" PARENT_SCOPE)
            endforeach()
        endforeach()
    endif()
endfunction()
