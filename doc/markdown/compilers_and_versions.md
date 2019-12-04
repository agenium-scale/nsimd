<!--

Copyright (c) 2019 Agenium Scale

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

-->

`nsimd` is tested with GCC, Clang and MSVC. As a C89 and a C++98 API are
provided, other compilers should work fine. Old compiler versions should work as
long as they support the targeted SIMD extension. For instance, `nsimd` can
compile on MSVC 2010 `SSE4.2` code.

`nsimd` requires a C or a C++ compiler and is actually daily tested on the
following compilers for the following hardware:

**Compiler**            | **Version** | **Architecture** | **Extensions**
----------------------- | ----------- | ---------------- | --------------
GCC                     | 8.3.0       | Intel            | `SSE2`, `SSE4.2`, `AVX`, `AVX2`, `AVX-512` (`KNL` and `SKYLAKE`)
Clang                   | 7.0.1       | Intel            | `SSE2`, `SSE4.2`, `AVX`, `AVX2`, `AVX-512` (`KNL` and `SKYLAKE`)
GCC                     | 8.3.0       | ARM              | `Aarch64`, `NEON` (`ARMv7`), `SVE`
Clang                   | 7.0.1       | ARM              | `Aarch64`, `NEON` (`ARMv7`), `SVE`
Microsoft Visual Studio | 2017        | Intel            | `SSE4.2`
Intel C++ Compiler      | 19.0.4.243  | Intel            | `SSE2`, `SSE4.2`, `AVX`, `AVX2`, `AVX-512` (`SKYLAKE`)

<!-- TODO  -->
<!--We recommend using a 64-bits compiler as this results in significantly better
performance. Also, `nsimd` performances are only provided when compiled in an
optimized code with assertions disabled.-->
