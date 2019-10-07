# Copyright (c) 2019 Agenium Scale

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

MD_TO_HTML_EXE=markdown2html

CC=clang
CXX=clang++

FLAGS_C=-Wall -Wextra -Wconversion -Wsign-conversion -O3 -lm
FLAGS_CXX=-Wall -Wextra -Wconversion -Wsign-conversion -O3

FLAGS_C89=-std=c89 -pedantic ${FLAGS_C}
FLAGS_C99=-std=c99 -pedantic ${FLAGS_C}
FLAGS_C11=-std=c11 -pedantic ${FLAGS_C}

FLAGS_CXX98=-std=c++98 -pedantic ${FLAGS_CXX}
FLAGS_CXX11=-std=c++11 -pedantic ${FLAGS_CXX}

FLAGS_SSE2=-msse2 -DSSE2
FLAGS_AVX=-mavx -DAVX

FLAGS_I=-I build/nsimd/include
FLAGS_L_CPU=-L build/nsimd/build_cpu -rpath build/nsimd/build_cpu -lnsimd_x86_64
FLAGS_L_SSE2=-L build/nsimd/build_sse2 -rpath build/nsimd/build_sse2 -lnsimd_x86_64
FLAGS_L_AVX_FMA=-L build/nsimd/build_avx_fma -rpath build/nsimd/build_avx_fma -lnsimd_x86_64


all: all_html all_c all_cpp nsimd


clean_code:
	rm -rf build

clean_html:
	rm -rf html

clean: clean_code clean_html


all_html: html/index.html \
          html/quick_start.html \
          html/faq.html \
          html/contribute.html \
          html/license.html \
          html/api.html \
          html/tutorials.html \
          html/tutorials_basic_pack.html \
          html/tutorials_basic_loop.html \
          html/tutorials_basic_memory_alignment.html \
          html/tutorials_basic_mathematical_functions.html \
          html/tutorials_intermediate_dot_product.html \
          html/tutorials_intermediate_if_else.html \
          html/img/640px-Pavlovsk_Railing_of_bridge_Yellow_palace_Winter.jpg \
          html/img/threshold_nsimd.jpg \
          html/tutorials_intermediate_neural_network.html \
          html/tutorials_intermediate_nbody.html \
          html/tutorials_advanced_runtime_simd.html \
          html/tutorials_advanced_aos_soa.html \
          html/tutorials_advanced_julia_set.html \
          html/tutorials_advanced_mandelbrot_set.html

html/%.html: doc/%.md doc/header.html doc/footer.html
	mkdir -p html
	${MD_TO_HTML_EXE} $< $@.raw.html
	cat doc/header.html $@.raw.html doc/footer.html > $@ && rm $@.raw.html

html/img/%.jpg: img/%.jpg
	mkdir -p html/img
	cp $< $@


nsimd: build/nsimd/build_cpu/libnsimd_x86_64.so

build/nsimd/build_cpu/libnsimd_x86_64.so: src/nsimd.sh
	mkdir -p build
	cd build && bash ../src/nsimd.sh


all_c: nsimd \
       build/hello_world_c89_sse2 \
       build/hello_world_c99_sse2 \
       build/hello_world_c11_sse2 \
       build/hello_world_c89_avx2 \
       build/hello_world_c99_avx2 \
       build/hello_world_c11_avx2

build/hello_world_c89_sse2: src/hello_world.c nsimd
	${CC} ${FLAGS_C89} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@
build/hello_world_c99_sse2: src/hello_world.c nsimd
	${CC} ${FLAGS_C99} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@
build/hello_world_c11_sse2: src/hello_world.c nsimd
	${CC} ${FLAGS_C11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@
build/hello_world_c89_avx2: src/hello_world.c nsimd
	${CC} ${FLAGS_C89} ${FLAGS_AVX} ${FLAGS_I} ${FLAGS_L_AVX_FMA} $< -o $@
build/hello_world_c99_avx2: src/hello_world.c nsimd
	${CC} ${FLAGS_C99} ${FLAGS_AVX} ${FLAGS_I} ${FLAGS_L_AVX_FMA} $< -o $@
build/hello_world_c11_avx2: src/hello_world.c nsimd
	${CC} ${FLAGS_C11} ${FLAGS_AVX} ${FLAGS_I} ${FLAGS_L_AVX_FMA} $< -o $@


all_cpp: nsimd \
         build/hello_world_cpp98_sse2 \
         build/hello_world_cpp11_sse2 \
         build/hello_world_cpp98_avx2 \
         build/hello_world_cpp11_avx2 \
         build/pack \
         build/loop \
         build/dot_product \
         build/threshold \
         build/neural_network \
         build/nbody \
         build/runtime_simd \
         build/aos_soa \
         build/mandelbrot_set \
         build/julia_set

build/hello_world_cpp98_sse2: src/hello_world.cpp nsimd
	${CXX} ${FLAGS_CXX98} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@
build/hello_world_cpp11_sse2: src/hello_world.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@
build/hello_world_cpp98_avx2: src/hello_world.cpp nsimd
	${CXX} ${FLAGS_CXX98} ${FLAGS_AVX} ${FLAGS_I} ${FLAGS_L_AVX_FMA} $< -o $@
build/hello_world_cpp11_avx2: src/hello_world.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_AVX} ${FLAGS_I} ${FLAGS_L_AVX_FMA} $< -o $@

build/pack: src/pack.cpp nsimd
	${CXX} ${FLAGS_CXX98} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@

build/loop: src/loop.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@

build/dot_product: src/dot_product.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@

build/sin_cos: src/sin_cos.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@

build/threshold: src/threshold.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@

build/neural_network: src/neural_network.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@

build/nbody: src/nbody.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@

build/runtime_simd: src/runtime_simd.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_AVX} ${FLAGS_I} ${FLAGS_L_AVX_FMA} $< -o $@

build/aos_soa: src/aos_soa.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@

build/mandelbrot_set: src/mandelbrot_set.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@

build/julia_set: src/julia_set.cpp nsimd
	${CXX} ${FLAGS_CXX11} ${FLAGS_SSE2} ${FLAGS_I} ${FLAGS_L_SSE2} $< -o $@
