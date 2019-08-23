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

CURRENT_DIR := $(shell pwd)
CC := g++
CFLAGS := -std=c++11 -O3 -DNSIMD_AVX2 -mavx2			\
	   -Wl,-rpath,$(CURRENT_DIR)/../../../build/
IFLAGS := -I$(CURRENT_DIR)/../../../include/			\
	  -I$(CURRENT_DIR)/../../../include/nsimd/modules/
LFLAGS := -L$(CURRENT_DIR)/../../../build/ -lnsimd_x86_64

EXEC := $(patsubst %.cpp,bin/%,$(wildcard *.cpp))

all: clean $(EXEC)

bin/%: %.cpp
	$(CC) $(IFLAGS) $^ -o $@ $(CFLAGS) $(LFLAGS)

clean::
	rm -f bin/*
