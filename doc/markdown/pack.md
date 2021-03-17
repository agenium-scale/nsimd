# NSIMD pack and related functions

The advanced C++ API provides types that represents SIMD registers. These
types are struct that allows NSIMD to define infix operators. In this page
NSIMD concepts are reported in the documentation but you can think of them
as usual `typename`s.

## The Pack type

```c++
template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct pack {
  // Typedef to retrieve the native SIMD type
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;

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

### Infix operators available for packs

- `pack operator+(pack const &, pack const &);`
- `pack operator*(pack const &, pack const &);`
- `pack operator-(pack const &, pack const &);`
- `pack operator/(pack const &, pack const &);`
- `pack operator-(pack const &);`
- `pack operator|(pack const &, pack const &);`
- `pack operator^(pack const &, pack const &);`
- `pack operator&(pack const &, pack const &);`
- `pack operator~(pack const &);`
- `pack operator<<(pack const &, int);` (only available for integers)
- `pack operator>>(pack const &, int);` (only available for integers)

## The Packl type

```c++
template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct packl {
  // Typedef to retrieve the native SIMD type
  typedef typename simd_traits<T, SimdExt>::simd_vectorl simd_vectorl;

  // Typedef to retrieve T
  typedef T value_type;

  // Typedef to retrieve SimdExt
  typedef SimdExt simd_ext;

  // Static member to retrive N
  static const int unroll = N;

  // Ctor that splats `s`, the resulting vector will be [s, s, s, ...]
  template <NSIMD_CONCEPT_VALUE_TYPE S> packl(S const &s);

  // Ctor that takes a SIMD vector of native type
  // ONLY AVAILABLE when N == 1
  packl(simd_vectorl v);
  
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
  nsimd::packl<float> mask;

  mask = nsimd::eq(v, v);
  std::cout << v << '\n';

  mask = nsimd::neq(v, v);
  std::cout << v << '\n';

  return 0;
}
```

### Infix operators involving packls

- `packl operator&&(packl const &, packl const &);`
- `packl operator||(packl const &, packl const &);`
- `packl operator!(packl const &, packl const &);`
- `packl operator==(pack const &, pack const &);`
- `packl operator!=(pack const &, pack const &);`
- `packl operator<(pack const &, pack const &);`
- `packl operator<=(pack const &, pack const &);`
- `packl operator>(pack const &, pack const &);`
- `packl operator>=(pack const &, pack const &);`

## Packs for SoA/AoS

Types containing several SIMD vectors are also provided to help the user
manipulate arrays of structures. When working, let's say, on complex numbers,
loading them from memory with layout `RIRIRIRIRIRI...` can be done with the
`load2*` operators that will returns 2 SIMD vectors `RRRR` and `IIII` where
`R` stands for real part and `I` for imaginary part.

Similarily loading an RGB image from memory stored following the layout
`RGBRGBRGBRGB...` can be done with `load3*` to get 3 SIMD vectors `RRRR`,
`GGGG` and `BBBB`.

### Packx1

```c++
template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx1 {

  // Usual typedefs and static members
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;
  static const int soa_num_packs = 1;

  // Member v0 for reading and writing
  pack<T, N, SimdExt> v0;
};
```

### Packx2

```c++
template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx2 {

  // Usual typedefs and static members
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;
  static const int soa_num_packs = 2;

  // Members for reading and writing
  pack<T, N, SimdExt> v0;
  pack<T, N, SimdExt> v1;
};
```

### Packx3

```c++
template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx3 {

  // Usual typedefs and static members
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;
  static const int soa_num_packs = 3;

  // Members for reading and writing
  pack<T, N, SimdExt> v0;
  pack<T, N, SimdExt> v1;
  pack<T, N, SimdExt> v2;
};
```

### Packx4

```c++
template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx4 {

  // Usual typedefs and static members
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;
  static const int soa_num_packs = 4;

  // Members for reading and writing
  pack<T, N, SimdExt> v0;
  pack<T, N, SimdExt> v1;
  pack<T, N, SimdExt> v2;
  pack<T, N, SimdExt> v3;
};
```

### Functions involving packx2, packx3 and packx4

The following functions converts packxs into unrolled packs. The difference
between the `to_pack` and `to_pack_interleave` families of functions is in
the way they flatten (or deinterleave) the structure of SIMD vectors.

```c++
template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 2 * N, SimdExt> to_pack(const packx2<T, N, SimdExt> &);

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 3 * N, SimdExt> to_pack(const packx3<T, N, SimdExt> &);

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 4 * N, SimdExt> to_pack(const packx4<T, N, SimdExt> &);

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 2 * N, SimdExt> to_pack_interleave(const packx2<T, N, SimdExt> &);

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 3 * N, SimdExt> to_pack_interleave(const packx3<T, N, SimdExt> &);

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 4 * N, SimdExt> to_pack_interleave(const packx4<T, N, SimdExt> &);
```

The `to_pack` family of functions performs the following operations:

```
packx2<T, 3> = | v0 = [u0 u1 u2] | ---> [u0 u1 u2 w0 w1 w2] = pack<T, 6>
               | v1 = [w0 w1 w2] |
```

while the `to_pack_interleave` family of functions does the following:

```
packx2<T, 3> = | v0 = [u0 u1 u2] | ---> [u0 w0 v1 w1 v2 w2] = pack<T, 6>
               | v1 = [w0 w1 w2] |
```

