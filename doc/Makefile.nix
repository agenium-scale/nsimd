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

NS2_ROOT  = ../nstools/ns2
CXX       = c++
CXX_FLAGS = -O2 -Wall -Wextra -pedantic -std=c++11

all: md2html what_is_wrapped

libns2.a: $(NS2_ROOT)/../.git/logs/HEAD Makefile.nix
	rm -rf libns2
	mkdir -p libns2
	cp $(NS2_ROOT)/lib/*.cpp libns2
	(cd libns2 && $(CXX) $(CXX_FLAGS) -I../$(NS2_ROOT)/include -c *.cpp)
	ar rcs $@ libns2/*.o
	rm -rf libns2

md2html: libns2.a md2html.cpp Makefile.nix
	$(CXX) $(CXX_FLAGS) md2html.cpp -I$(NS2_ROOT)/include -o $@ -L. -lns2

what_is_wrapped: libns2.a what_is_wrapped.cpp Makefile.nix
	$(CXX) $(CXX_FLAGS) what_is_wrapped.cpp -I$(NS2_ROOT)/include -o $@ \
	       -L. -lns2
