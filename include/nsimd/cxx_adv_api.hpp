/*

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

*/

#ifndef NSIMD_CXX_ADV_API_HPP
#define NSIMD_CXX_ADV_API_HPP

#include <nsimd/nsimd.h>
#include <ostream>

// ----------------------------------------------------------------------------

namespace nsimd {

// ----------------------------------------------------------------------------
// "mimic" static_assert in C++98

template <bool> struct nsimd_static_assert;
template <> struct nsimd_static_assert<true> {};

// ----------------------------------------------------------------------------
// Definition of pack

template <NSIMD_CONCEPT_VALUE_TYPE T, int N = 1,
          NSIMD_CONCEPT_SIMD_EXT SimdExt = NSIMD_SIMD>
NSIMD_STRUCT pack;

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT pack<T, 1, SimdExt> {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = 1;
  static const int soa_num_packs = 1;

  simd_vector car;

  // Default ctor
  pack() {}

  // Ctor that splats
  template <NSIMD_CONCEPT_VALUE_TYPE S> pack(S const &s) {
    car = set1(T(s), T(), SimdExt());
  }

  // Ctor taking a SIMD vector
  pack(simd_vector v) { car = v; }

  // Underlying native SIMD vector getter
  simd_vector native_register() const { return car; }

  friend std::ostream &operator<<(std::ostream &os, pack const &a0) {
    T buf[max_len_t<T>::value];
    storeu(buf, a0.car, T(), SimdExt());
    os << "{ ";
    int n = len(a0);
    for (int i = 0; i < n; i++) {
      os << buf[i];
      if (i < n - 1) {
        os << ", ";
      }
    }
    os << " }";
    return os;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT pack {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;
  static const int soa_num_packs = 1;

  simd_vector car;
  pack<T, N - 1, SimdExt> cdr;

  // Default ctor
  pack() {}

  // Ctor that splats
  template <NSIMD_CONCEPT_VALUE_TYPE S> pack(S const &s) : cdr(s) {
    car = set1(T(s), T(), SimdExt());
  }

  friend std::ostream &operator<<(std::ostream &os, pack const &a0) {
    os << pack<T, 1, SimdExt>(a0.car) << ", " << a0.cdr;
    return os;
  }
};

#if NSIMD_CXX >= 2020
template <typename T> struct is_pack_t : public std::false_type {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct is_pack_t<pack<T, N, SimdExt> > : public std::true_type {};

template <typename T> concept is_pack_c = is_pack_t<T>::value;
#define NSIMD_CONCEPT_PACK nsimd::is_pack_c
#else
#define NSIMD_CONCEPT_PACK typename
#endif

// ----------------------------------------------------------------------------
// Definition of logical

template <NSIMD_CONCEPT_VALUE_TYPE T, int N = 1,
          NSIMD_CONCEPT_SIMD_EXT SimdExt = NSIMD_SIMD>
NSIMD_STRUCT packl;

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packl<T, 1, SimdExt> {
  typedef typename simd_traits<T, SimdExt>::simd_vectorl simd_vectorl;
  simd_vectorl car;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = 1;

  // Default ctor
  packl() {}

  // Ctor taking a SIMD vector
  packl(simd_vectorl v) { car = v; }

  // Ctor that splats
  template <NSIMD_CONCEPT_VALUE_TYPE_OR_BOOL S> packl(S const &s) {
    car = set1l(int(s), T(), SimdExt());
  }

  // Underlying native SIMD vector getter
  simd_vectorl native_register() const { return car; }

  friend std::ostream &operator<<(std::ostream &os, packl const &a0) {
    T buf[max_len_t<T>::value];
    storelu(buf, a0.car, T(), SimdExt());
    os << "{ ";
    int n = len(a0);
    for (int i = 0; i < n; i++) {
      os << buf[i];
      if (i < n - 1) {
        os << ", ";
      }
    }
    os << " }";
    return os;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packl {
  typename simd_traits<T, SimdExt>::simd_vectorl car;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;

  packl<T, N - 1, SimdExt> cdr;

  // Default ctor
  packl() {}

  // Ctor that splats
  template <NSIMD_CONCEPT_VALUE_TYPE S> packl(S const &s) : cdr(s) {
    car = set1l(int(s), T(), SimdExt());
  }

  friend std::ostream &operator<<(std::ostream &os, packl const &a0) {
    os << packl<T, 1, SimdExt>(a0.car) << ", " << a0.cdr;
    return os;
  }
};

#if NSIMD_CXX >= 2020
template <typename T> struct is_packl_t : public std::false_type {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct is_packl_t<packl<T, N, SimdExt> > : public std::true_type {};

template <typename T> concept is_packl_c = is_packl_t<T>::value;
#define NSIMD_CONCEPT_PACKL nsimd::is_packl_c
#else
#define NSIMD_CONCEPT_PACKL typename
#endif

// ----------------------------------------------------------------------------
// Definition of SOA of degree 1

template <NSIMD_CONCEPT_VALUE_TYPE T, int N = 1,
          NSIMD_CONCEPT_SIMD_EXT SimdExt = NSIMD_SIMD>
NSIMD_STRUCT packx1;

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx1<T, 1, SimdExt> {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = 1;
  static const int soa_num_packs = 1;

  pack<T, 1, SimdExt> v0;

  void set_car(simd_vector v0_) {
    v0.car = v0_;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx1 {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;
  static const int soa_num_packs = 1;

  pack<T, N, SimdExt> v0;

  void set_car(simd_vector v0_) {
    v0.car = v0_;
  }

  void set_cdr(pack<T, N - 1, SimdExt> const &v0_) {
    v0.cdr = v0_;
  }
};

#if NSIMD_CXX >= 2020
template <typename T> struct is_packx1_t : public std::false_type {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct is_packx1_t<packx1<T, N, SimdExt> > : public std::true_type {};

template <typename T> concept is_packx1_c = is_packx1_t<T>::value;
#define NSIMD_CONCEPT_PACKX1 nsimd::is_packx1_c
#else
#define NSIMD_CONCEPT_PACKX1 typename
#endif

// ----------------------------------------------------------------------------
// Definition of SOA of degree 2

template <NSIMD_CONCEPT_VALUE_TYPE T, int N = 1,
          NSIMD_CONCEPT_SIMD_EXT SimdExt = NSIMD_SIMD>
NSIMD_STRUCT packx2;

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx2<T, 1, SimdExt> {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = 1;
  static const int soa_num_packs = 2;

  pack<T, 1, SimdExt> v0;
  pack<T, 1, SimdExt> v1;

  void set_car(simd_vector v0_, simd_vector v1_) {
    v0.car = v0_;
    v1.car = v1_;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx2 {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;
  static const int soa_num_packs = 2;

  pack<T, N, SimdExt> v0;
  pack<T, N, SimdExt> v1;

  void set_car(simd_vector v0_, simd_vector v1_) {
    v0.car = v0_;
    v1.car = v1_;
  }

  void set_cdr(pack<T, N - 1, SimdExt> const &v0_,
               pack<T, N - 1, SimdExt> const &v1_) {
    v0.cdr = v0_;
    v1.cdr = v1_;
  }
};

#if NSIMD_CXX >= 2020
template <typename T> struct is_packx2_t : public std::false_type {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct is_packx2_t<packx2<T, N, SimdExt> > : public std::true_type {};

template <typename T> concept is_packx2_c = is_packx2_t<T>::value;
#define NSIMD_CONCEPT_PACKX2 nsimd::is_packx2_c
#else
#define NSIMD_CONCEPT_PACKX2 typename
#endif

// ----------------------------------------------------------------------------
// Definition of SOA of degree 3

template <NSIMD_CONCEPT_VALUE_TYPE T, int N = 1,
          NSIMD_CONCEPT_SIMD_EXT SimdExt = NSIMD_SIMD>
NSIMD_STRUCT packx3;

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx3<T, 1, SimdExt> {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = 1;
  static const int soa_num_packs = 3;

  pack<T, 1, SimdExt> v0;
  pack<T, 1, SimdExt> v1;
  pack<T, 1, SimdExt> v2;

  void set_car(simd_vector v0_, simd_vector v1_, simd_vector v2_) {
    v0.car = v0_;
    v1.car = v1_;
    v2.car = v2_;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx3 {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;
  static const int soa_num_packs = 3;

  pack<T, N, SimdExt> v0;
  pack<T, N, SimdExt> v1;
  pack<T, N, SimdExt> v2;

  void set_car(simd_vector v0_, simd_vector v1_, simd_vector v2_) {
    v0.car = v0_;
    v1.car = v1_;
    v2.car = v2_;
  }

  void set_cdr(pack<T, N - 1, SimdExt> const &v0_,
               pack<T, N - 1, SimdExt> const &v1_,
               pack<T, N - 1, SimdExt> const &v2_) {
    v0.cdr = v0_;
    v1.cdr = v1_;
    v2.cdr = v2_;
  }
};

#if NSIMD_CXX >= 2020
template <typename T> struct is_packx3_t : public std::false_type {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct is_packx3_t<packx3<T, N, SimdExt> > : public std::true_type {};

template <typename T> concept is_packx3_c = is_packx3_t<T>::value;
#define NSIMD_CONCEPT_PACKX3 nsimd::is_packx3_c
#else
#define NSIMD_CONCEPT_PACKX3 typename
#endif

// ----------------------------------------------------------------------------
// Definition of SOA of degree 4

template <NSIMD_CONCEPT_VALUE_TYPE T, int N = 1,
          NSIMD_CONCEPT_SIMD_EXT SimdExt = NSIMD_SIMD>
NSIMD_STRUCT packx4;

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx4<T, 1, SimdExt> {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = 1;
  static const int soa_num_packs = 4;

  pack<T, 1, SimdExt> v0;
  pack<T, 1, SimdExt> v1;
  pack<T, 1, SimdExt> v2;
  pack<T, 1, SimdExt> v3;

  void set_car(simd_vector v0_, simd_vector v1_, simd_vector v2_,
               simd_vector v3_) {
    v0.car = v0_;
    v1.car = v1_;
    v2.car = v2_;
    v3.car = v3_;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_STRUCT packx4 {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;
  static const int soa_num_packs = 4;

  pack<T, N, SimdExt> v0;
  pack<T, N, SimdExt> v1;
  pack<T, N, SimdExt> v2;
  pack<T, N, SimdExt> v3;

  void set_car(simd_vector v0_, simd_vector v1_, simd_vector v2_,
               simd_vector v3_) {
    v0.car = v0_;
    v1.car = v1_;
    v2.car = v2_;
    v3.car = v3_;
  }

  void set_cdr(
      pack<T, N - 1, SimdExt> const &v0_, pack<T, N - 1, SimdExt> const &v1_,
      pack<T, N - 1, SimdExt> const &v2_, pack<T, N - 1, SimdExt> const &v3_) {
    v0.cdr = v0_;
    v1.cdr = v1_;
    v2.cdr = v2_;
    v3.cdr = v3_;
  }
};

#if NSIMD_CXX >= 2020
template <typename T> struct is_packx4_t : public std::false_type {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct is_packx4_t<packx4<T, N, SimdExt> > : public std::true_type {};

template <typename T> concept is_packx4_c = is_packx4_t<T>::value;
#define NSIMD_CONCEPT_PACKX4 nsimd::is_packx4_c
#else
#define NSIMD_CONCEPT_PACKX4 typename
#endif

// ----------------------------------------------------------------------------
// A C++20 concept

#if NSIMD_CXX >=2020
template <typename T>
concept any_pack_c = is_pack_c<T> || is_packl_c<T> || is_packx1_c<T> ||
                     is_packx2_c<T> || is_packx3_c<T> || is_packx4_c<T>;
#define NSIMD_CONCEPT_ANY_PACK nsimd::any_pack_c
#else
#define NSIMD_CONCEPT_ANY_PACK typename
#endif

// ----------------------------------------------------------------------------
// The len function cannot be auto-generated

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int len(pack<T, N, SimdExt> const &) {
  return N * len(T(), SimdExt());
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int len(packl<T, N, SimdExt> const &) {
  return N * len(T(), SimdExt());
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int len(packx1<T, N, SimdExt> const &) {
  return N * len(T(), SimdExt());
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int len(packx2<T, N, SimdExt> const &) {
  return 2 * N * len(T(), SimdExt());
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int len(packx3<T, N, SimdExt> const &) {
  return 3 * N * len(T(), SimdExt());
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int len(packx4<T, N, SimdExt> const &) {
  return 4 * N * len(T(), SimdExt());
}

// ----------------------------------------------------------------------------
// The addv function cannot be auto-generated

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
T addv(pack<T, 1, SimdExt> const &a0) {
  return addv(a0.car, T(), SimdExt());
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
T addv(pack<T, N, SimdExt> const &a0) {
  return addv(a0.car, T(), SimdExt()) + addv(a0.cdr);
}

// ----------------------------------------------------------------------------
// The all function cannot be auto-generated

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int all(packl<T, 1, SimdExt> const &a0) {
  return all(a0.car, T(), SimdExt());
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int all(packl<T, N, SimdExt> const &a0) {
  return all(a0.car, T(), SimdExt()) && all(a0.cdr);
}

// ----------------------------------------------------------------------------
// The any function cannot be auto-generated

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int any(packl<T, 1, SimdExt> const &a0) {
  return any(a0.car, T(), SimdExt());
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int any(packl<T, N, SimdExt> const &a0) {
  return any(a0.car, T(), SimdExt()) || any(a0.cdr);
}

// ----------------------------------------------------------------------------
// The nbtrue function cannot be auto-generated

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int nbtrue(packl<T, 1, SimdExt> const &a0) {
  return nbtrue(a0.car, T(), SimdExt());
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
int nbtrue(packl<T, N, SimdExt> const &a0) {
  return nbtrue(a0.car, T(), SimdExt()) + nbtrue(a0.cdr);
}

// ----------------------------------------------------------------------------
// Include functions that act on packs

} // namespace nsimd

#include <nsimd/cxx_adv_api_functions.hpp>

namespace nsimd {

// ----------------------------------------------------------------------------
// The if_else function cannot be auto-generated

template <NSIMD_CONCEPT_VALUE_TYPE L, NSIMD_CONCEPT_VALUE_TYPE T,
          NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_REQUIRES_SAME_SIZEOF(L, T)
pack<T, 1, SimdExt>
if_else(packl<L, 1, SimdExt> const &a0, pack<T, 1, SimdExt> const &a1,
        pack<T, 1, SimdExt> const &a2) {
  pack<T, 1, SimdExt> ret;
  ret.car = if_else(a0.car, a1.car, a2.car, L(), T(), SimdExt());
  return ret;
}

template <NSIMD_CONCEPT_VALUE_TYPE L, NSIMD_CONCEPT_VALUE_TYPE T, int N,
          NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_REQUIRES_SAME_SIZEOF(L, T)
pack<T, N, SimdExt>
if_else(packl<L, N, SimdExt> const &a0, pack<T, N, SimdExt> const &a1,
        pack<T, N, SimdExt> const &a2) {
  pack<T, N, SimdExt> ret;
  ret.car = if_else(a0.car, a1.car, a2.car, L(), T(), SimdExt());
  ret.cdr = if_else(a0.cdr, a1.cdr, a2.cdr);
  return ret;
}

// ----------------------------------------------------------------------------
// Mask loads and stores cannot be auto-generated

template <NSIMD_CONCEPT_VALUE_TYPE L, NSIMD_CONCEPT_VALUE_TYPE T, int N,
          NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_REQUIRES_SAME_SIZEOF(L, T)
void mask_storea(packl<L, N, SimdExt> const &a0, T *a1,
                 pack<T, N, SimdExt> const &a2) {
  mask_storea1(reinterpretl<packl<T, N, SimdExt> >(a0), a1, a2);
}

template <NSIMD_CONCEPT_VALUE_TYPE L, NSIMD_CONCEPT_VALUE_TYPE T, int N,
          NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_REQUIRES_SAME_SIZEOF(L, T)
void mask_storeu(packl<L, N, SimdExt> const &a0, T *a1,
                 pack<T, N, SimdExt> const &a2) {
  mask_storeu1(reinterpretl<packl<T, N, SimdExt> >(a0), a1, a2);
}

template <NSIMD_CONCEPT_VALUE_TYPE L, NSIMD_CONCEPT_VALUE_TYPE T, int N,
          NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_REQUIRES_SAME_SIZEOF(L, T)
pack<T, N, SimdExt> maskz_loada(packl<L, N, SimdExt> const &a0, const T *a1) {
  return maskz_loada1(reinterpretl<packl<T, N, SimdExt> >(a0), a1);
}

template <NSIMD_CONCEPT_VALUE_TYPE L, NSIMD_CONCEPT_VALUE_TYPE T, int N,
          NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_REQUIRES_SAME_SIZEOF(L, T)
pack<T, N, SimdExt> maskz_loadu(packl<L, N, SimdExt> const &a0, const T *a1) {
  return maskz_loadu1(reinterpretl<packl<T, N, SimdExt> >(a0), a1);
}

template <NSIMD_CONCEPT_VALUE_TYPE L, NSIMD_CONCEPT_VALUE_TYPE T, int N,
          NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_REQUIRES_SAME_SIZEOF(L, T)
pack<T, N, SimdExt> masko_loada(packl<L, N, SimdExt> const &a0, const T *a1,
                                pack<T, N, SimdExt> const &a2) {
  return masko_loada1(reinterpretl<packl<T, N, SimdExt> >(a0), a1, a2);
}

template <NSIMD_CONCEPT_VALUE_TYPE L, NSIMD_CONCEPT_VALUE_TYPE T, int N,
          NSIMD_CONCEPT_SIMD_EXT SimdExt>
NSIMD_REQUIRES_SAME_SIZEOF(L, T)
pack<T, N, SimdExt> masko_loadu(packl<L, N, SimdExt> const &a0, const T *a1,
                                pack<T, N, SimdExt> const &a2) {
  return masko_loadu1(reinterpretl<packl<T, N, SimdExt> >(a0), a1, a2);
}

// ----------------------------------------------------------------------------
// Loads/Stores templated on the alignment cannot be auto-generated

namespace detail {

template <NSIMD_CONCEPT_PACKL P> struct loadz_return_t {
  typedef nsimd::pack<typename P::value_type, P::unroll, typename P::simd_ext>
      type;
};

template <NSIMD_CONCEPT_ANY_PACK SimdVector, NSIMD_CONCEPT_ALIGNMENT Alignment>
struct load_helper {};

template <NSIMD_CONCEPT_ANY_PACK SimdVector>
struct load_helper<SimdVector, aligned> {
  typedef typename SimdVector::value_type T;
  typedef typename SimdVector::simd_ext simd_ext;
  static const int N = SimdVector::unroll;

  static SimdVector load(const T *a0) { return loada<SimdVector>(a0); }
  static SimdVector loadl(const T *a0) { return loadla<SimdVector>(a0); }
  static SimdVector load2(const T *a0) { return load2a<SimdVector>(a0); }
  static SimdVector load3(const T *a0) { return load3a<SimdVector>(a0); }
  static SimdVector load4(const T *a0) { return load4a<SimdVector>(a0); }

  static SimdVector maskz_load(packl<T, N, simd_ext> const &a0, const T *a1) {
    return maskz_loada(a0, a1);
  }

  static pack<T, N, simd_ext> masko_load(packl<T, N, simd_ext> const &a0,
                                         const T *a1,
                                         pack<T, N, simd_ext> const &a2) {
    return masko_loada(a0, a1, a2);
  }
};

template <typename SimdVector> struct load_helper<SimdVector, unaligned> {
  typedef typename SimdVector::value_type T;
  typedef typename SimdVector::simd_ext simd_ext;
  static const int N = SimdVector::unroll;

  static SimdVector load(const T *a0) { return loadu<SimdVector>(a0); }
  static SimdVector loadl(const T *a0) { return loadlu<SimdVector>(a0); }
  static SimdVector load2(const T *a0) { return load2u<SimdVector>(a0); }
  static SimdVector load3(const T *a0) { return load3u<SimdVector>(a0); }
  static SimdVector load4(const T *a0) { return load4u<SimdVector>(a0); }

  static SimdVector maskz_load(packl<T, N, simd_ext> const &a0, const T *a1) {
    return maskz_loadu(a0, a1);
  }

  static pack<T, N, simd_ext> masko_load(packl<T, N, simd_ext> const &a0,
                                         const T *a1,
                                         pack<T, N, simd_ext> const &a2) {
    return masko_loadu(a0, a1, a2);
  }
};

template <NSIMD_CONCEPT_ALIGNMENT Alignment> struct store_helper {};

#define NSIMD_T typename P::value_type

template <> struct store_helper<aligned> {
  template <NSIMD_CONCEPT_PACK P> static void store(NSIMD_T *a0, P const &a1) {
    storea(a0, a1);
  }

  template <NSIMD_CONCEPT_PACKL PL, NSIMD_CONCEPT_PACK P>
#if NSIMD_CXX >= 2020
  requires std::is_same_v<typename PL::value_type, typename P::value_type>
#endif
  static void mask_store(PL const &a0, NSIMD_T *a1, P const &a2) {
    mask_storea(a0, a1, a2);
  }

  template <NSIMD_CONCEPT_PACK P> static void storel(NSIMD_T *a0, P const &a1) {
    storela(a0, a1);
  }

  template <NSIMD_CONCEPT_PACK P>
  static void store2(NSIMD_T *a0, P const &a1, P const &a2) {
    store2a(a0, a1, a2);
  }

  template <NSIMD_CONCEPT_PACK P>
  static void store3(NSIMD_T *a0, P const &a1, P const &a2, P const &a3) {
    store3a(a0, a1, a2, a3);
  }

  template <NSIMD_CONCEPT_PACK P>
  static void store4(NSIMD_T *a0, P const &a1, P const &a2, P const &a3,
                     P const &a4) {
    store4a(a0, a1, a2, a3, a4);
  }
};

template <> struct store_helper<unaligned> {
  template <NSIMD_CONCEPT_PACK P> static void store(NSIMD_T *a0, P const &a1) {
    storeu(a0, a1);
  }

  template <NSIMD_CONCEPT_PACKL PL, NSIMD_CONCEPT_PACK P>
#if NSIMD_CXX >= 2020
  requires std::is_same_v<typename PL::value_type, typename P::value_type>
#endif
  static void mask_store(PL const &a0, NSIMD_T *a1, P const &a2) {
    mask_storeu(a0, a1, a2);
  }

  template <NSIMD_CONCEPT_PACK P> static void storel(NSIMD_T *a0, P const &a1) {
    storelu(a0, a1);
  }

  template <NSIMD_CONCEPT_PACK P>
  static void store2(NSIMD_T *a0, P const &a1, P const &a2) {
    store2u(a0, a1, a2);
  }

  template <NSIMD_CONCEPT_PACK P>
  static void store3(NSIMD_T *a0, P const &a1, P const &a2, P const &a3) {
    store3u(a0, a1, a2, a3);
  }

  template <NSIMD_CONCEPT_PACK P>
  static void store4(NSIMD_T *a0, P const &a1, P const &a2, P const &a3,
                     P const &a4) {
    store4u(a0, a1, a2, a3, a4);
  }
};

#undef NSIMD_T

} // namespace detail

template <NSIMD_CONCEPT_PACK SimdVector, NSIMD_CONCEPT_ALIGNMENT Alignment>
SimdVector load(const typename SimdVector::value_type *ptr) {
  return detail::load_helper<SimdVector, Alignment>::load(ptr);
}

template <NSIMD_CONCEPT_ALIGNMENT Alignment, NSIMD_CONCEPT_PACKL Packl>
pack<typename Packl::value_type, Packl::unroll, typename Packl::simd_ext>
maskz_load(Packl const &pl, const typename Packl::value_type *ptr) {
  return detail::load_helper<pack<typename Packl::value_type, Packl::unroll,
                                  typename Packl::simd_ext>,
                             Alignment>::maskz_load(pl, ptr);
}

template <NSIMD_CONCEPT_ALIGNMENT Alignment, NSIMD_CONCEPT_PACKL Packl,
          NSIMD_CONCEPT_PACK Pack>
Pack masko_load(Packl const &pl, const typename Pack::value_type *ptr,
                Pack const &p) {
  return detail::load_helper<Pack, Alignment>::masko_load(pl, ptr, p);
}

template <NSIMD_CONCEPT_PACK SimdVector, NSIMD_CONCEPT_ALIGNMENT Alignment>
SimdVector loadl(const typename SimdVector::value_type *ptr) {
  return detail::load_helper<SimdVector, Alignment>::loadl(ptr);
}

template <NSIMD_CONCEPT_PACKX2 SimdVector, NSIMD_CONCEPT_ALIGNMENT Alignment>
SimdVector load2(const typename SimdVector::value_type *ptr) {
  return detail::load_helper<SimdVector, Alignment>::load2(ptr);
}

template <NSIMD_CONCEPT_PACKX3 SimdVector, NSIMD_CONCEPT_ALIGNMENT Alignment>
SimdVector load3(const typename SimdVector::value_type *ptr) {
  return detail::load_helper<SimdVector, Alignment>::load3(ptr);
}

template <NSIMD_CONCEPT_PACKX4 SimdVector, NSIMD_CONCEPT_ALIGNMENT Alignment>
SimdVector load4(const typename SimdVector::value_type *ptr) {
  return detail::load_helper<SimdVector, Alignment>::load4(ptr);
}

template <NSIMD_CONCEPT_ALIGNMENT Alignment, NSIMD_CONCEPT_PACK Pack>
void store(typename Pack::value_type *ptr, Pack const &p) {
  detail::store_helper<Alignment>::store(ptr, p);
}

template <NSIMD_CONCEPT_ALIGNMENT Alignment, NSIMD_CONCEPT_PACKL Packl,
          NSIMD_CONCEPT_PACK Pack>
void mask_store(Packl const &pl, typename Pack::value_type *ptr,
                Pack const &p) {
  detail::store_helper<Alignment>::mask_store(pl, ptr, p);
}

template <NSIMD_CONCEPT_ALIGNMENT Alignment, NSIMD_CONCEPT_PACKL Packl>
void storel(typename Packl::value_type *ptr, Packl const &pl) {
  return detail::store_helper<Alignment>::storel(ptr, pl);
}

template <NSIMD_CONCEPT_ALIGNMENT Alignment, NSIMD_CONCEPT_PACK Pack>
void store2(typename Pack::value_type *ptr, Pack const &p1, Pack const &p2) {
  return detail::store_helper<Alignment>::store2(ptr, p1, p2);
}

template <NSIMD_CONCEPT_ALIGNMENT Alignment, NSIMD_CONCEPT_PACK Pack>
void store3(typename Pack::value_type *ptr, Pack const &p1, Pack const &p2,
            Pack const &p3) {
  return detail::store_helper<Alignment>::store3(ptr, p1, p2, p3);
}

template <NSIMD_CONCEPT_ALIGNMENT Alignment, NSIMD_CONCEPT_PACK Pack>
void store4(typename Pack::value_type *ptr, Pack const &p1, Pack const &p2,
            Pack const &p3, Pack const &p4) {
  return detail::store_helper<Alignment>::store4(ptr, p1, p2, p3, p4);
}

// ----------------------------------------------------------------------------

template <NSIMD_CONCEPT_VALUE_TYPE T> T native_register(T a) { return a; }

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
typename pack<T, 1, SimdExt>::simd_vector
native_register(pack<T, 1, SimdExt> const &a) {
  return a.car;
}

// ----------------------------------------------------------------------------
// get_pack

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt,
          template <typename, int, typename> class packx, int Ix>
struct get_pack_helper {};

// ----------------------------------------------------------------------------
// get_pack_helper - packx1

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt,
          int Ix>
struct get_pack_helper<T, N, SimdExt, packx1, Ix> {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct get_pack_helper<T, N, SimdExt, packx1, 0> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx1<T, N, SimdExt> &packx_) const {
    return packx_.v0;
  }
};

// ----------------------------------------------------------------------------
// get_pack_helper - packx2

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt,
          int Ix>
struct get_pack_helper<T, N, SimdExt, packx2, Ix> {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct get_pack_helper<T, N, SimdExt, packx2, 0> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx2<T, N, SimdExt> &packx_) const {
    return packx_.v0;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct get_pack_helper<T, N, SimdExt, packx2, 1> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx2<T, N, SimdExt> &packx_) const {
    return packx_.v1;
  }
};

// ----------------------------------------------------------------------------
// get_pack_helper - packx3

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt,
          int Ix>
struct get_pack_helper<T, N, SimdExt, packx3, Ix> {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct get_pack_helper<T, N, SimdExt, packx3, 0> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx3<T, N, SimdExt> &packx_) const {
    return packx_.v0;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct get_pack_helper<T, N, SimdExt, packx3, 1> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx3<T, N, SimdExt> &packx_) const {
    return packx_.v1;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct get_pack_helper<T, N, SimdExt, packx3, 2> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx3<T, N, SimdExt> &packx_) const {
    return packx_.v2;
  }
};

// ----------------------------------------------------------------------------
// get_pack_helper - packx4

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt,
          int Ix>
struct get_pack_helper<T, N, SimdExt, packx4, Ix> {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct get_pack_helper<T, N, SimdExt, packx4, 0> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx4<T, N, SimdExt> &packx_) const {
    return packx_.v0;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct get_pack_helper<T, N, SimdExt, packx4, 1> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx4<T, N, SimdExt> &packx_) const {
    return packx_.v1;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct get_pack_helper<T, N, SimdExt, packx4, 2> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx4<T, N, SimdExt> &packx_) const {
    return packx_.v2;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
struct get_pack_helper<T, N, SimdExt, packx4, 3> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx4<T, N, SimdExt> &packx_) const {
    return packx_.v3;
  }
};

// ----------------------------------------------------------------------------
// get_pack
// get_pack for packx[Y]<T, 1..N, SimdExt> with Y = 1

template <int Ix, NSIMD_CONCEPT_VALUE_TYPE T, int N,
          NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, N, SimdExt> get_pack(const pack<T, N, SimdExt> &pack_) {
  nsimd_static_assert<0 == Ix>();
  return pack_;
}

// ----------------------------------------------------------------------------
// get_pack
// get_pack for packx[Y]<T, 1..N, SimdExt> with Y in {2, 3, 4}

template <int Ix, NSIMD_CONCEPT_VALUE_TYPE T, int N,
          NSIMD_CONCEPT_SIMD_EXT SimdExt,
          template <typename, int, typename> class packx>
pack<T, N, SimdExt> get_pack(const packx<T, N, SimdExt> &packx_) {
  return get_pack_helper<T, N, SimdExt, packx, Ix>()(packx_);
}

// ----------------------------------------------------------------------------
// to_pack_trait

template <class _packx> struct to_pack_trait {};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt,
          template <typename, int, typename> class _packx>
struct to_pack_trait<_packx<T, N, SimdExt> > {
  typedef pack<T, _packx<T, N, SimdExt>::soa_num_packs * N, SimdExt>
      value_type;
};

// ----------------------------------------------------------------------------
// to_pack
// to_pack for packx[Y]<T, 1..N, SimdExt> with Y = 1

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 1, SimdExt> to_pack(const pack<T, 1, SimdExt> &pack_) {
  return pack_;
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, N, SimdExt> to_pack(const pack<T, N, SimdExt> &pack_) {
  return pack_;
}

// ----------------------------------------------------------------------------
// to_pack
// to_pack for packx[Y]<T, N = 1, SimdExt> with Y in {2, 3, 4}

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 1, SimdExt> to_pack(const packx1<T, 1, SimdExt> &packx_) {

  nsimd::pack<T, 1, SimdExt> pack_;
  pack_.car = packx_.v0.car;

  return pack_;
}

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 2, SimdExt> to_pack(const packx2<T, 1, SimdExt> &packx_) {

  nsimd::pack<T, 2, SimdExt> pack_;
  pack_.car = packx_.v0.car;
  pack_.cdr.car = packx_.v1.car;

  return pack_;
}

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 3, SimdExt> to_pack(const packx3<T, 1, SimdExt> &packx_) {

  nsimd::pack<T, 3, SimdExt> pack_;
  pack_.car = packx_.v0.car;
  pack_.cdr.car = packx_.v1.car;
  pack_.cdr.cdr.car = packx_.v2.car;
  return pack_;
}

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 4, SimdExt> to_pack(const packx4<T, 1, SimdExt> &packx_) {

  nsimd::pack<T, 4, SimdExt> pack_;
  pack_.car = packx_.v0.car;
  pack_.cdr.car = packx_.v1.car;
  pack_.cdr.cdr.car = packx_.v2.car;
  pack_.cdr.cdr.cdr.car = packx_.v3.car;

  return pack_;
}

// ----------------------------------------------------------------------------
// to_pack for packx[Y]<T, (N > 1), SimdExt> with Y in {2, 3, 4}

// Advance
template <NSIMD_CONCEPT_VALUE_TYPE T, int from_pack_init_N,
          int from_pack_unroll_ix, int to_pack_unroll_ix,
          int which_from_pack_ix, NSIMD_CONCEPT_SIMD_EXT SimdExt,
          template <typename, int, typename> class packx>
struct to_pack_recurs_helper {
  static pack<T, to_pack_unroll_ix, SimdExt>
  to_pack(const packx<T, from_pack_init_N, SimdExt> &from_packx,
          const pack<T, from_pack_unroll_ix, SimdExt> &from_pack) {
    pack<T, to_pack_unroll_ix, SimdExt> to_pack_;
    to_pack_.car = from_pack.car;
    to_pack_.cdr =
        to_pack_recurs_helper<T, from_pack_init_N, from_pack_unroll_ix - 1,
                              to_pack_unroll_ix - 1, which_from_pack_ix,
                              SimdExt, packx>::to_pack(from_packx,
                                                       from_pack.cdr);
    return to_pack_;
  }
};

// Base case
// Base case condition: to_pack_unroll_ix == 1
template <NSIMD_CONCEPT_VALUE_TYPE T, int from_pack_init_N,
          int which_from_pack_ix, NSIMD_CONCEPT_SIMD_EXT SimdExt,
          template <typename, int, typename> class packx>
struct to_pack_recurs_helper<T, from_pack_init_N, 1 /* from_pack_unroll_ix */,
                             1 /* to_pack_unroll_ix */, which_from_pack_ix,
                             SimdExt, packx> {
  static pack<T, 1, SimdExt>
  to_pack(const packx<T, from_pack_init_N, SimdExt> &from_packx,
          const pack<T, 1, SimdExt> &from_pack) {
    (void)from_packx;
    pack<T, 1, SimdExt> to_pack_;
    to_pack_.car = from_pack.car; // simd_vector
    return to_pack_;
  }
};

// Switch: from_packx[i] --> from_packx[i+1]
// Switch condition: from_pack_unroll_ix == 1 && to_pack_unroll_ix > 1
template <NSIMD_CONCEPT_VALUE_TYPE T, int from_pack_init_N, int to_pack_unroll_ix,
          int which_from_pack_ix, NSIMD_CONCEPT_SIMD_EXT SimdExt,
          template <typename, int, typename> class packx>
struct to_pack_recurs_helper<T, from_pack_init_N, 1 /* from_pack_unroll_ix */,
                             to_pack_unroll_ix, which_from_pack_ix, SimdExt,
                             packx> {
  static pack<T, to_pack_unroll_ix, SimdExt>
  to_pack(const packx<T, from_pack_init_N, SimdExt> &from_packx,
          const pack<T, 1, SimdExt> &from_pack) {

    pack<T, to_pack_unroll_ix, SimdExt> to_pack_;
    to_pack_.car = from_pack.car; // simd_vector

    // get next pack
    to_pack_.cdr = to_pack_recurs_helper<
        T, from_pack_init_N, from_pack_init_N, to_pack_unroll_ix - 1,
        which_from_pack_ix + 1, SimdExt,
        packx>::to_pack(from_packx,
                        get_pack<which_from_pack_ix + 1>(from_packx));
    return to_pack_;
  }
};

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt,
          template <typename, int, typename> class packx>
typename to_pack_trait<packx<T, N, SimdExt> >::value_type
to_pack(const packx<T, N, SimdExt> &from_packx) {
  static const int to_pack_unroll_ix = packx<T, N, SimdExt>::soa_num_packs * N;
  pack<T, to_pack_unroll_ix, SimdExt> to_pack_;
  to_pack_.car = from_packx.v0.car; // simd_vector
  to_pack_.cdr = to_pack_recurs_helper<
      T, N /* from_pack_init_N*/, N - 1 /* from_pack_unroll_ix */,
      to_pack_unroll_ix - 1 /* to_pack_unroll_ix */,
      0 /* which_from_pack_ix */, SimdExt, packx>::to_pack(from_packx,
                                                           from_packx.v0.cdr);
  return to_pack_;
}

// ----------------------------------------------------------------------------
// to_pack_interleave

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 1, SimdExt> to_pack_interleave(const pack<T, 1, SimdExt> &pack_) {
  return pack_;
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, N, SimdExt> to_pack_interleave(const pack<T, N, SimdExt> &pack_) {
  return pack_;
}

// ----------------------------------------------------------------------------

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 1, SimdExt> to_pack_interleave(const packx1<T, 1, SimdExt> &packx1_) {
  pack<T, 1, SimdExt> pack_1;
  pack_1.car = packx1_.v0.car;
  pack_1.cdr = packx1_.v0.cdr;
  return pack_1;
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, N, SimdExt>
to_pack_interleave(const packx1<T, N, SimdExt> &packx1_N) {
  pack<T, N, SimdExt> pack_1;
  pack_1.car = packx1_N.v0.car;
  pack_1.cdr = packx1_N.v0.cdr;
  return pack_1;
}

// ----------------------------------------------------------------------------

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 2, SimdExt> to_pack_interleave(const packx2<T, 1, SimdExt> &packx2_) {

  nsimd::pack<T, 2, SimdExt> pack_2;
  pack_2.car = packx2_.v0.car;
  pack_2.cdr.car = packx2_.v1.car;

  return pack_2;
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 2 * N, SimdExt>
to_pack_interleave(const packx2<T, N, SimdExt> &packx2_N) {

  pack<T, 2 * N, SimdExt> pack_2xN;
  pack_2xN.car = packx2_N.v0.car;
  pack_2xN.cdr.car = packx2_N.v1.car;

  packx2<T, N - 1, SimdExt> packx2_n_1;
  packx2_n_1.v0 = packx2_N.v0.cdr;
  packx2_n_1.v1 = packx2_N.v1.cdr;

  pack_2xN.cdr.cdr = to_pack_interleave(packx2_n_1);

  return pack_2xN;
}

// ----------------------------------------------------------------------------

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 3, SimdExt> to_pack_interleave(const packx3<T, 1, SimdExt> &packx3_) {

  nsimd::pack<T, 3, SimdExt> pack_3;
  pack_3.car = packx3_.v0.car;
  pack_3.cdr.car = packx3_.v1.car;
  pack_3.cdr.cdr.car = packx3_.v2.car;

  return pack_3;
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 3 * N, SimdExt>
to_pack_interleave(const packx3<T, N, SimdExt> &packx3_n) {

  pack<T, 3 * N, SimdExt> pack_3xn;
  pack_3xn.car = packx3_n.v0.car;
  pack_3xn.cdr.car = packx3_n.v1.car;
  pack_3xn.cdr.cdr.car = packx3_n.v2.car;

  packx3<T, N - 1, SimdExt> packx3_n_1;
  packx3_n_1.v0 = packx3_n.v0.cdr;
  packx3_n_1.v1 = packx3_n.v1.cdr;
  packx3_n_1.v2 = packx3_n.v2.cdr;

  pack_3xn.cdr.cdr.cdr = to_pack_interleave(packx3_n_1);

  return pack_3xn;
}

// ----------------------------------------------------------------------------

template <NSIMD_CONCEPT_VALUE_TYPE T, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 4, SimdExt> to_pack_interleave(const packx4<T, 1, SimdExt> &packx4_) {

  nsimd::pack<T, 4, SimdExt> pack_4;
  pack_4.car = packx4_.v0.car;
  pack_4.cdr.car = packx4_.v1.car;
  pack_4.cdr.cdr.car = packx4_.v2.car;
  pack_4.cdr.cdr.cdr.car = packx4_.v3.car;

  return pack_4;
}

template <NSIMD_CONCEPT_VALUE_TYPE T, int N, NSIMD_CONCEPT_SIMD_EXT SimdExt>
pack<T, 4 * N, SimdExt>
to_pack_interleave(const packx4<T, N, SimdExt> &packx4_n) {

  pack<T, 4 * N, SimdExt> pack_4xn;
  pack_4xn.car = packx4_n.v0.car;
  pack_4xn.cdr.car = packx4_n.v1.car;
  pack_4xn.cdr.cdr.car = packx4_n.v2.car;
  pack_4xn.cdr.cdr.cdr.car = packx4_n.v3.car;

  packx4<T, N - 1, SimdExt> packx4_n_1;
  packx4_n_1.v0 = packx4_n.v0.cdr;
  packx4_n_1.v1 = packx4_n.v1.cdr;
  packx4_n_1.v2 = packx4_n.v2.cdr;
  packx4_n_1.v3 = packx4_n.v3.cdr;

  pack_4xn.cdr.cdr.cdr.cdr = to_pack_interleave(packx4_n_1);

  return pack_4xn;
}

} // namespace nsimd

#endif
