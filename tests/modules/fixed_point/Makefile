CC := g++
CFLAGS := -std=c++11 -DNSIMD_AVX2 -mavx2
IFLAGS := -I/home/adrien/Projets/nsimd/include
IFLAGS += -I/home/adrien/Projets/nsimd/include/nsimd/modules
LFLAGS := -L/home/adrien/Projets/nsimd/build -lnsimd_x86_64
CFLAGS += -Wl,-rpath,/home/adrien/Projets/nsimd/build

all:	clean fixed_point

fixed_point: fixed_point.cpp
	$(CC)  $^ -o $@ $(CFLAGS) $(IFLAGS) $(LFLAGS)

clean::
	rm -f fixed_point
