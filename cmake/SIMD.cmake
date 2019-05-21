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

## Retrieve SIMD CXXFLAGS and command-line options for hatch.py
function(get_simd_infos out simd simd_optional)

    # Sanity checks first
    set(simds
        CPU
        SSE SSE2 SSE3 SSSE SSE41 SSE42
        AVX AVX2 AVX512_KNL AVX512_SKYLAKE
        NEON64 NEON128
        AARCH64
        SVE
    )
    set(simd_optionals
        FMA FP16
    )

    list(FIND simds "${simd}" i)
    if (i EQUAL -1)
        message(FATAL_ERROR "Unknown SIMD: '${simd}' must be one of: ${simds}")
    endif()

    if (NOT "${simd_optional}" MATCHES "^$")
        list(FIND simd_optionals ${simd_optional} i)
        if (i EQUAL -1)
            message(FATAL_ERROR
                    "Unknown SIMD: '${simd_optional}' must "
                    "be one of: ${simd_optionals}")
        endif()
    endif()

    # Return compilation flag
    compiler_flags("${simd}" comp_flags)
    compiler_flags("${simd_optional}" comp_flags_optional)
    set(${out}_FLAGS "${comp_flags} ${comp_flags_optional}" PARENT_SCOPE)

    # Return hatch.py command line options
    string(TOLOWER "${simd}" hatch_flag)
    set(${out}_HATCH "${hatch_flag}" PARENT_SCOPE)

    # Return list of enabled SIMD extensions (not needed for now)
    set(list_for_cpu            "CPU")

    set(list_for_sse2           "${list_for_cpu};SSE2")
    set(list_for_sse42          "${list_for_sse2};SSE42")
    set(list_for_avx            "${list_for_sse42};AVX")
    set(list_for_avx2           "${list_for_avx};AVX2")
    set(list_for_avx512_knl     "${list_for_avx2};AVX512_KNL")
    set(list_for_avx512_skylake "${list_for_avx2};AVX512_SKYLAKE")

    set(list_for_neon128        "${list_for_cpu};NEON128")
    set(list_for_aarch64        "${list_for_cpu};AARCH64")
    set(list_for_sve            "${list_for_aarch64};SVE")
    
    set(${out}_LIST "${list_for_${hatch_flag}}" PARENT_SCOPE)

    # Return platform
    if ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "(arm|ARM|aarch64)")
        if ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
            set(${out}_PLATFORM "armv7" PARENT_SCOPE)
        else()
            set(${out}_PLATFORM "aarch64" PARENT_SCOPE)
        endif()
    elseif ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "AMD64")
        set(${out}_PLATFORM "x86_64" PARENT_SCOPE)
    elseif ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "x86_64")
        set(${out}_PLATFORM "x86_64" PARENT_SCOPE)
    elseif ("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "Intel")
        if ("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
            set(${out}_PLATFORM "x86" PARENT_SCOPE)
        else()
            set(${out}_PLATFORM "x86_64" PARENT_SCOPE)
        endif()
    else()
        message(FATAL_ERROR
                "Unsupported platform: '${CMAKE_SYSTEM_PROCESSOR}'")
    endif()

endfunction()
