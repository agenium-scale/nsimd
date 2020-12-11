# Memory functions

Although the purpose of NSIMD is not to provide a full memory container
library, it provides some helper functions to facilitate the end-user. The
functions below only deals with CPUs. If your needs concerns GPUs or memory
transfers between CPUs and GPUs see the [memory management
module](module_memory_management_overview.md).

## Memory functions available in C and C++

- `void *nsimd_aligned_alloc(nsimd_nat n);`  
  Returns a pointer to `n` bytes of aligned memory. It returns NULL is an
  error occurs.

- `void nsimd_aligned_free(void *ptr);`
  Frees the memory pointed to by `ptr`.

## Memory functions available in C++

- `void *nsimd::aligned_alloc(nsimd_nat n);`  
  Returns a pointer to `n` bytes of aligned memory. It returns NULL is an
  error occurs.

- `void nsimd::aligned_free(void *ptr);`  
  Frees the memory pointed to by `ptr`.

- `template <typename T> T *nsimd::aligned_alloc_for(nsimd_nat n);`  
  Returns a pointer to `n` `T`'s of aligned memory. It returns NULL is an
  error occurs.

- `template <typename T> void nsimd::aligned_free_for(void *ptr);`  
  Free memory pointed to by `ptr`.

## C++ allocators for `std::vector`'s

NSIMD provides C++ allocators so that memory used by C++ container such as
`std::vector`'s will be suitably aligned in memory.

- `template <typename T> class nsimd::allocator;`  
  The class for allocating aligned memory inside C++ containers.

Exemple:

```c++
#include <nsimd/nsimd.h>

int main() {
  int n = // number of float's to allocate
  std::vector<float, nsimd::allocator<float> > myvector(size_t(n));
  
  // In what follows ptr is a pointer suitably aligned for the current SIMD
  // targeted architecture.
  float *ptr;
  
  // C++98
  ptr = &myvector[0]; 

  // C++11 and above
  ptr = myvector.data(); 
}
```

As there is no portable way of having aligned scoped memory, one can use the
NSIMD allocators to emulate such memory.

```c++
#include <nsimd/nsimd.h>

template <typename T, int N> void test() {
  std::vector<T, nsimd::allocator<T> > mem(size_t(N));
  T *ptr;
  
  // C++98
  ptr = &mem[0]; // scoped aligned memory

  // C++11 and above
  ptr = mem.data(); // scoped aligned memory
}

int main() {
  test<float, 16>();
  test<double, 8>();
}
```

## C++ scoped memory allocation

NSIMD provides a struct helper for the user to allocate a chunk of memory and
don't care about its release. It uses C++ RAII.

```c++
namespace nsimd {

template <typename T> class scoped_aligned_mem_for {

  template <typename I> scoped_aligned_mem(I n);
  // Construct a struct an array of n T's.

  T *get();
  // Return the pointer to access memory.

};

}

int main() {
  // Allocates 1024 floats in memory. It will be freed when the function (or
  // the program) terminates.
  nsimd::scoped_aligned_mem_for<float> buffer(1024);
  return 0;
}
```
