# NSIMD pack and related functions

The advanced C++ API provides types that represents SIMD registers. These
types are struct that allows NSIMD to define infix operators.

## The Pack type

```c++
template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT pack {

  // Typedef to retrieve T
  typedef T value_type;

  // Typedef to retrieve SimdExt
  typedef SimdExt simd_ext;

  // Static member to retrive N
  static const int unroll = N;

  // Ctor that splats `s`, the resulting vector will be [s, s, s, ...]
  template <NSIMD_CONCEPT_VALUE_TYPE S> pack(S const &s);

  // Ctor that takes a SIMD vector of native type
  // ONLY AVAILABLE when N == 1
  pack(simd_vector v);
  
  // Retrieve the underlying native SIMD vector
  // ONLY AVAILABLE when N == 1
  simd_vector native_register() const;

};
```

Example:

```c++
#include <nsimd/nsimd-all.hpp>
#include <iostream>

int main() {
  nsimd::pack<float> v(2.0f);
  std::cout << v << '\n';

  vf32 nv = v.native_register();
  nv = nsimd::add(nv, nv, f32());
  std::cout << nsimd::pack<f32>(nv) << '\n';

  return 0;
}
```

## The Packl type
