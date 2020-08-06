# Memory functions

Although the purpose of NSIMD is not to provide a full memory container
library, it provides some helper functions to facilitate the end-user.

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

NSIMD
