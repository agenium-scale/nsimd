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

CC := clang -W -Wall -Wno-unused-parameter -pedantic
CXX := clang++ -W -Wall -Wno-unused-parameter -pedantic

IFLAGS := -I/home/adrien/Projets/nsimd/include \
	-I./include/
CFLAGS := -std=c99 -g -O3 -DNSIMD_AVX2 -mavx2 -mfma -ffast-math
CXXFLAGS := -std=c++14 -g -O3 -DNSIMD_AVX2 -ffast-math -ffp-contract=fast

SRC_PATH := ./src/modules/convolution
OBJ_PATH := ./obj
LIB_PATH := ./lib
TESTS_PATH := ./tests
BIN_PATH := ./bin
LDFLAGS := -Wl,-rpath,$(LIB_PATH) -L./$(LIB_PATH) -lnsimd_conv

SRCS := $(shell find $(SRC_PATH) -type f -name *.c)
TESTS := $(shell find $(TESTS_PATH) -type f -name *.c)
OBJS := $(patsubst $(SRC_PATH)/%,$(OBJ_PATH)/%,$(SRCS:.c=.o))
TESTS :=$(patsubst $(TESTS_PATH)/%,$(BIN_PATH)/%,$(TESTS:.c=))

DNN_IFLAGS := -I/home/adrien/intel/oneapi/dnnl/2021.1-beta08/cpu_gomp/include \
	-I/home/adrien/intel/oneapi/dnnl/2021.1-beta08/cpu_gomp/examples 
DNN_LFLAGS := -L/home/adrien/intel/oneapi/dnnl/2021.1-beta08/cpu_gomp/lib \
	-Wl,-rpath,/home/adrien/intel/oneapi/dnnl/2021.1-beta08/cpu_gomp/lib \
	-ldnnl

## -----------------------------------------------------------------------------

all: clean dir lib tests

all_tests: $(TESTS)

lib: $(LIB_PATH)/libnsimd_conv.so

bin/example: src/example.cpp
	$(CXX) $(CXXFLAGS) $(DNN_IFLAGS) -o $@ $< $(DNN_LFLAGS)

$(LIB_PATH)/libnsimd_conv.so: $(OBJS)
	$(CC) $(CFLAGS) $(IFLAGS) -shared -o $@ $(OBJS)

dir:
	mkdir -p $(BIN_PATH)
	mkdir -p $(LIB_PATH)
	mkdir -p $(OBJ_PATH)
	mkdir -p $(OBJ_PATH)/packing
	mkdir -p $(OBJ_PATH)/kernels

$(OBJ_PATH)/%.o: $(SRC_PATH)/%.c
	$(CC) $(CFLAGS) -fPIC -c -o $@ $(IFLAGS) $<

$(BIN_PATH)/%: $(TESTS_PATH)/%.c
	$(CC) $(CFLAGS) -o $@ $(IFLAGS) $< $(LDFLAGS)

clean:
	rm -rf $(BIN_PATH)/*
	rm -rf $(OBJ_PATH)/*.o


