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

namespace nsimd {

// ----------------------------------------------------------------------------
// For ARM SVE we need a special struct

#ifdef NSIMD_SVE
#define NSIMD_STRUCT __sizeless_struct
#else
#define NSIMD_STRUCT struct
#endif

// ----------------------------------------------------------------------------
// Definition of pack

template <typename T, int N = 1, typename SimdExt = NSIMD_SIMD>
NSIMD_STRUCT pack;

template <typename T, typename SimdExt> NSIMD_STRUCT pack<T, 1, SimdExt> {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = 1;

  simd_vector car;

  // Default ctor
  pack() {}

  // Ctor that splats
  template <typename S> pack(S const &s) { car = set1(T(s), T(), SimdExt()); }

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

template <typename T, int N, typename SimdExt> NSIMD_STRUCT pack {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;

  simd_vector car;
  pack<T, N - 1, SimdExt> cdr;

  // Default ctor
  pack() {}

  // Ctor that splats
  template <typename S> pack(S const &s) : cdr(s) {
    car = set1(T(s), T(), SimdExt());
  }

  friend std::ostream &operator<<(std::ostream &os, pack const &a0) {
    os << pack<T, 1, SimdExt>(a0.car) << ", " << a0.cdr;
    return os;
  }
};

// ----------------------------------------------------------------------------
// Definition of logical

template <typename T, int N = 1, typename SimdExt = NSIMD_SIMD>
NSIMD_STRUCT packl;

template <typename T, typename SimdExt> NSIMD_STRUCT packl<T, 1, SimdExt> {
  typedef typename simd_traits<T, SimdExt>::simd_vectorl simd_vectorl;
  simd_vectorl car;

  // Default ctor
  packl() {}

  // Ctor taking a SIMD vector
  packl(simd_vectorl v) { car = v; }

  // Underlying native SIMD vector getter
  simd_vectorl native_register() const { return car; }

  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = 1;
};

template <typename T, int N, typename SimdExt> NSIMD_STRUCT packl {
  typename simd_traits<T, SimdExt>::simd_vectorl car;
  packl<T, N - 1, SimdExt> cdr;

  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;
};

// ----------------------------------------------------------------------------
// Definition of SOA of degree 2

template <typename T, int N = 1, typename SimdExt = NSIMD_SIMD>
NSIMD_STRUCT packx2;

template <typename T, typename SimdExt> NSIMD_STRUCT packx2<T, 1, SimdExt> {
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

template <typename T, int N, typename SimdExt> NSIMD_STRUCT packx2 {
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

// ----------------------------------------------------------------------------
// Definition of SOA of degree 3

template <typename T, int N = 1, typename SimdExt = NSIMD_SIMD>
NSIMD_STRUCT packx3;

template <typename T, typename SimdExt> NSIMD_STRUCT packx3<T, 1, SimdExt> {
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

template <typename T, int N, typename SimdExt> NSIMD_STRUCT packx3 {
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

// ----------------------------------------------------------------------------
// Definition of SOA of degree 4

template <typename T, int N = 1, typename SimdExt = NSIMD_SIMD>
NSIMD_STRUCT packx4;

template <typename T, typename SimdExt> NSIMD_STRUCT packx4<T, 1, SimdExt> {
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

template <typename T, int N, typename SimdExt> NSIMD_STRUCT packx4 {
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

// ----------------------------------------------------------------------------
// The len function cannot be auto-generated

template <typename T, int N, typename SimdExt>
int len(pack<T, N, SimdExt> const &) {
  return N * len(T(), SimdExt());
}

template <typename T, int N, typename SimdExt>
int len(packl<T, N, SimdExt> const &) {
  return N * len(T(), SimdExt());
}

template <typename T, int N, typename SimdExt>
int len(packx2<T, N, SimdExt> const &) {
  return 2 * N * len(T(), SimdExt());
}

template <typename T, int N, typename SimdExt>
int len(packx3<T, N, SimdExt> const &) {
  return 3 * N * len(T(), SimdExt());
}

template <typename T, int N, typename SimdExt>
int len(packx4<T, N, SimdExt> const &) {
  return 4 * N * len(T(), SimdExt());
}

// ----------------------------------------------------------------------------
// The addv function cannot be auto-generated

template <typename T, typename SimdExt> T addv(pack<T, 1, SimdExt> const &a0) {
  return addv(a0.car, T(), SimdExt());
}

template <typename T, int N, typename SimdExt>
T addv(pack<T, N, SimdExt> const &a0) {
  return addv(a0.car, T(), SimdExt()) + addv(a0.cdr);
}

// ----------------------------------------------------------------------------
// The all function cannot be auto-generated

template <typename T, typename SimdExt>
int all(packl<T, 1, SimdExt> const &a0) {
  return all(a0.car, T(), SimdExt());
}

template <typename T, int N, typename SimdExt>
int all(packl<T, N, SimdExt> const &a0) {
  return all(a0.car, T(), SimdExt()) && all(a0.cdr);
}

// ----------------------------------------------------------------------------
// The any function cannot be auto-generated

template <typename T, typename SimdExt>
int any(packl<T, 1, SimdExt> const &a0) {
  return any(a0.car, T(), SimdExt());
}

template <typename T, int N, typename SimdExt>
int any(packl<T, N, SimdExt> const &a0) {
  return any(a0.car, T(), SimdExt()) || any(a0.cdr);
}

// ----------------------------------------------------------------------------
// The nbtrue function cannot be auto-generated

template <typename T, typename SimdExt>
int nbtrue(packl<T, 1, SimdExt> const &a0) {
  return nbtrue(a0.car, T(), SimdExt());
}

template <typename T, int N, typename SimdExt>
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

template <typename L, typename T, typename SimdExt>
pack<T, 1, SimdExt> if_else(packl<L, 1, SimdExt> const &a0,
                            pack<T, 1, SimdExt> const &a1,
                            pack<T, 1, SimdExt> const &a2) {
  pack<T, 1, SimdExt> ret;
  ret.car = if_else(a0.car, a1.car, a2.car, L(), T(), SimdExt());
  return ret;
}

template <typename L, typename T, int N, typename SimdExt>
pack<T, N, SimdExt> if_else(packl<L, N, SimdExt> const &a0,
                            pack<T, N, SimdExt> const &a1,
                            pack<T, N, SimdExt> const &a2) {
  pack<T, N, SimdExt> ret;
  ret.car = if_else(a0.car, a1.car, a2.car, L(), T(), SimdExt());
  ret.cdr = if_else(a0.cdr, a1.cdr, a2.cdr);
  return ret;
}

// ----------------------------------------------------------------------------
// Loads/Stores templated on the alignment cannot be auto-generated

namespace detail {

template <typename SimdVector, typename Alignment> struct load_helper {};

template <typename SimdVector> struct load_helper<SimdVector, aligned> {
  template <typename A0> static SimdVector load(A0 a0) {
    return loada<SimdVector, A0>(a0);
  }

  template <typename A0> static SimdVector loadl(A0 a0) {
    return loadla<SimdVector, A0>(a0);
  }

  template <typename A0> static SimdVector load2(A0 a0) {
    return load2a<SimdVector, A0>(a0);
  }

  template <typename A0> static SimdVector load3(A0 a0) {
    return load3a<SimdVector, A0>(a0);
  }

  template <typename A0> static SimdVector load4(A0 a0) {
    return load4a<SimdVector, A0>(a0);
  }
};

template <typename SimdVector> struct load_helper<SimdVector, unaligned> {
  template <typename A0> static SimdVector load(A0 a0) {
    return loadu<SimdVector, A0>(a0);
  }

  template <typename A0> static SimdVector loadl(A0 a0) {
    return loadlu<SimdVector, A0>(a0);
  }

  template <typename A0> static SimdVector load2(A0 a0) {
    return load2u<SimdVector, A0>(a0);
  }

  template <typename A0> static SimdVector load3(A0 a0) {
    return load3u<SimdVector, A0>(a0);
  }

  template <typename A0> static SimdVector load4(A0 a0) {
    return load4u<SimdVector, A0>(a0);
  }
};

template <typename SimdVector, typename Alignment> struct store_helper {};

template <typename SimdVector> struct store_helper<SimdVector, aligned> {
  template <typename A0, typename A1> static SimdVector store(A0 a0, A1 a1) {
    storea<SimdVector, A0, A1>(a0, a1);
  }

  template <typename A0, typename A1> static SimdVector storel(A0 a0, A1 a1) {
    storela<SimdVector, A0, A1>(a0, a1);
  }

  template <typename A0, typename A1, typename A2>
  static SimdVector store2(A0 a0, A1 a1, A2 a2) {
    store2a<SimdVector, A0, A1, A2>(a0, a1, a2);
  }

  template <typename A0, typename A1, typename A2, typename A3>
  static SimdVector store3(A0 a0, A1 a1, A2 a2, A3 a3) {
    store3a<SimdVector, A0, A1, A2, A3>(a0, a1, a2, a3);
  }

  template <typename A0, typename A1, typename A2, typename A3, typename A4>
  static SimdVector store4(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4) {
    store4a<SimdVector, A0>(a0, a1, a2, a3, a4);
  }
};

template <typename SimdVector> struct store_helper<SimdVector, unaligned> {
  template <typename A0, typename A1> static SimdVector store(A0 a0, A1 a1) {
    storeu<SimdVector, A0, A1>(a0, a1);
  }

  template <typename A0, typename A1> static SimdVector storel(A0 a0, A1 a1) {
    storelu<SimdVector, A0, A1>(a0, a1);
  }

  template <typename A0, typename A1, typename A2>
  static SimdVector store2(A0 a0, A1 a1, A2 a2) {
    store2u<SimdVector, A0, A1, A2>(a0, a1, a2);
  }

  template <typename A0, typename A1, typename A2, typename A3>
  static SimdVector store3(A0 a0, A1 a1, A2 a2, A3 a3) {
    store3u<SimdVector, A0, A1, A2, A3>(a0, a1, a2, a3);
  }

  template <typename A0, typename A1, typename A2, typename A3, typename A4>
  static SimdVector store4(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4) {
    store4u<SimdVector, A0>(a0, a1, a2, a3, a4);
  }
};

} // namespace detail

template <typename SimdVector, typename Alignment, typename A0>
SimdVector load(A0 a0) {
  return detail::load_helper<SimdVector, Alignment>::load(a0);
}

template <typename SimdVector, typename Alignment, typename A0>
SimdVector loadl(A0 a0) {
  return detail::load_helper<SimdVector, Alignment>::loadl(a0);
}

template <typename SimdVector, typename Alignment, typename A0>
SimdVector load2(A0 a0) {
  return detail::load_helper<SimdVector, Alignment>::load2(a0);
}

template <typename SimdVector, typename Alignment, typename A0>
SimdVector load3(A0 a0) {
  return detail::load_helper<SimdVector, Alignment>::load3(a0);
}

template <typename SimdVector, typename Alignment, typename A0>
SimdVector load4(A0 a0) {
  return detail::load_helper<SimdVector, Alignment>::load4(a0);
}

template <typename SimdVector, typename Alignment, typename A0, typename A1>
SimdVector store(A0 a0, A1 a1) {
  detail::store_helper<SimdVector, Alignment>::store(a0, a1);
}

template <typename SimdVector, typename Alignment, typename A0, typename A1>
SimdVector storel(A0 a0, A1 a1) {
  return detail::store_helper<SimdVector, Alignment>::storel(a0, a1);
}

template <typename SimdVector, typename Alignment, typename A0, typename A1,
          typename A2>
SimdVector store2(A0 a0, A1 a1, A2 a2) {
  return detail::store_helper<SimdVector, Alignment>::store2(a0, a1, a2);
}

template <typename SimdVector, typename Alignment, typename A0, typename A1,
          typename A2, typename A3>
SimdVector store3(A0 a0, A1 a1, A2 a2, A3 a3) {
  return detail::store_helper<SimdVector, Alignment>::store3(a0, a1, a2, a3);
}

template <typename SimdVector, typename Alignment, typename A0, typename A1,
          typename A2, typename A3, typename A4>
SimdVector store4(A0 a0, A1 a1, A2 a2, A3 a3, A4 a4) {
  return detail::store_helper<SimdVector, Alignment>::store4(a0, a1, a2, a3,
                                                             a4);
}

// ----------------------------------------------------------------------------

template <typename T> T native_register(T a) { return a; }

template <typename T, typename SimdExt>
typename pack<T, 1, SimdExt>::simd_vector
native_register(pack<T, 1, SimdExt> const &a) {
  return a.car;
}

// ----------------------------------------------------------------------------
// get_pack

template <typename T, int N, typename SimdExt,
          template <typename, int, typename> class packx, int Ix>
struct get_pack_helper {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx<T, N, SimdExt> &packx_) const {}
};

// ----------------------------------------------------------------------------
// get_pack_helper - packx2

template <typename T, int N, typename SimdExt, int Ix>
struct get_pack_helper<T, N, SimdExt, packx2, Ix> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx2<T, N, SimdExt> &packx_) const {
    static_assert(0 <= Ix && Ix < packx2<T, N, SimdExt>::soa_num_packs,
                  "ERROR - get_pack_helper<Ix>{}(const packx2<T, N, SimdExt> "
                  "&packx_) const - Ix not in valid range: 0 <= Ix < 2");
  }
};

template <typename T, int N, typename SimdExt>
struct get_pack_helper<T, N, SimdExt, packx2, 0> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx2<T, N, SimdExt> &packx_) const {
    return packx_.v0;
  }
};

template <typename T, int N, typename SimdExt>
struct get_pack_helper<T, N, SimdExt, packx2, 1> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx2<T, N, SimdExt> &packx_) const {
    return packx_.v1;
  }
};

// ----------------------------------------------------------------------------
// get_pack_helper - packx3

template <typename T, int N, typename SimdExt, int Ix>
struct get_pack_helper<T, N, SimdExt, packx3, Ix> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx3<T, N, SimdExt> &packx_) const {
    static_assert(0 <= Ix && Ix < packx3<T, N, SimdExt>::soa_num_packs,
                  "ERROR - get_pack_helper<Ix>{}(const packx3<T, N, SimdExt> "
                  "&packx_) const - Ix not in valid range: 0 <= Ix < 3");
  }
};

template <typename T, int N, typename SimdExt>
struct get_pack_helper<T, N, SimdExt, packx3, 0> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx3<T, N, SimdExt> &packx_) const {
    return packx_.v0;
  }
};

template <typename T, int N, typename SimdExt>
struct get_pack_helper<T, N, SimdExt, packx3, 1> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx3<T, N, SimdExt> &packx_) const {
    return packx_.v1;
  }
};

template <typename T, int N, typename SimdExt>
struct get_pack_helper<T, N, SimdExt, packx3, 2> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx3<T, N, SimdExt> &packx_) const {
    return packx_.v2;
  }
};

// ----------------------------------------------------------------------------
// get_pack_helper - packx4

template <typename T, int N, typename SimdExt, int Ix>
struct get_pack_helper<T, N, SimdExt, packx4, Ix> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx4<T, N, SimdExt> &packx_) const {
    static_assert(0 <= Ix && Ix < packx4<T, N, SimdExt>::soa_num_packs,
                  "ERROR - get_pack_helper<Ix>{}(const packx4<T, N, SimdExt> "
                  "&packx_) const - Ix not in valid range: 0 <= Ix < 4");
  }
};

template <typename T, int N, typename SimdExt>
struct get_pack_helper<T, N, SimdExt, packx4, 0> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx4<T, N, SimdExt> &packx_) const {
    return packx_.v0;
  }
};

template <typename T, int N, typename SimdExt>
struct get_pack_helper<T, N, SimdExt, packx4, 1> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx4<T, N, SimdExt> &packx_) const {
    return packx_.v1;
  }
};

template <typename T, int N, typename SimdExt>
struct get_pack_helper<T, N, SimdExt, packx4, 2> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx4<T, N, SimdExt> &packx_) const {
    return packx_.v2;
  }
};

template <typename T, int N, typename SimdExt>
struct get_pack_helper<T, N, SimdExt, packx4, 3> {
  const nsimd::pack<T, N, SimdExt> &
  operator()(const packx4<T, N, SimdExt> &packx_) const {
    return packx_.v3;
  }
};

// ----------------------------------------------------------------------------
// get_pack
// get_pack for packx[Y]<T, 1..N, SimdExt> with Y = 1

template <int Ix, typename T, int N, typename SimdExt>
pack<T, N, SimdExt> get_pack(const pack<T, N, SimdExt> &pack_) {
  return pack_;
}

// ----------------------------------------------------------------------------
// get_pack
// get_pack for packx[Y]<T, 1..N, SimdExt> with Y in {2, 3, 4}

template <int Ix, typename T, int N, typename SimdExt,
          template <typename, int, typename> class packx>
pack<T, N, SimdExt> get_pack(const packx<T, N, SimdExt> &packx_) {
  return get_pack_helper<T, N, SimdExt, packx, Ix>{}(packx_);
}

// ----------------------------------------------------------------------------
// to_pack

template <typename T, typename SimdExt = NSIMD_SIMD>
pack<T, 1, SimdExt> to_pack(const pack<T, 1, SimdExt> &pack_) {
  return pack_;
}

template <typename T, int N, typename SimdExt = NSIMD_SIMD>
pack<T, N, SimdExt> to_pack(const pack<T, N, SimdExt> &pack_) {
  return pack_;
}

// ----------------------------------------------------------------------------
// to_pack - packx2<N = 1, ...> to pack<N = 2, ...>

template <typename T, typename SimdExt = NSIMD_SIMD>
pack<T, 2, SimdExt> to_pack(const packx2<T, 1, SimdExt> &packx2_) {

  nsimd::pack<T, 2, SimdExt> pack_2;
  pack_2.car = packx2_.v0.car;
  pack_2.cdr.car = packx2_.v1.car;

  return pack_2;
}

// ----------------------------------------------------------------------------
// to_pack - packx2<N, ...> to pack<2 * N, ...>

// Advance
template <typename T, int init_N, int packx_unroll_ix, int to_pack_unroll_ix,
          int v_ix, typename SimdExt,
          template <typename, int, typename> class packx>
struct to_pack_recurs_helper {
  static pack<T, to_pack_unroll_ix, SimdExt>
  to_pack(const packx<T, init_N, SimdExt> &from_packx_initN,
          const pack<T, packx_unroll_ix, SimdExt> &from_pack) {
    pack<T, to_pack_unroll_ix, SimdExt> to_pack_;
    to_pack_.car = from_pack.car;
    to_pack_.cdr =
        to_pack_recurs_helper<T, init_N, packx_unroll_ix - 1,
                              to_pack_unroll_ix - 1, v_ix, SimdExt,
                              packx>::to_pack(from_packx_initN, from_pack.cdr);
    return to_pack_;
  }
};

// Base case
template <typename T, int init_N, int v_ix, typename SimdExt,
          template <typename, int, typename> class packx>
struct to_pack_recurs_helper<T, init_N, 1 /* == 1: base case condition */,
                             1 /* == 1: base case condition */, v_ix, SimdExt,
                             packx> {
  static pack<T, 1, SimdExt>
  to_pack(const packx<T, init_N, SimdExt> &from_packx_initN,
          const pack<T, 1, SimdExt> &from_pack) {
    (void)from_packx_initN;
    pack<T, 1, SimdExt> to_pack_;
    to_pack_.car = from_pack.car; // simd_vector
    return to_pack_;
  }
};

// Switch from v_[i] to v_[i+1]
template <typename T, int init_N, int to_pack_unroll_ix, int v_ix,
          typename SimdExt, template <typename, int, typename> class packx>
struct to_pack_recurs_helper<
    T, init_N,
    1 /* packx_unroll_ix == 1: switch from v_[i] to v_[i+1] condition */,
    to_pack_unroll_ix /* > 1: switch from v_[i] to v_[i+1] condition */, v_ix,
    SimdExt, packx> {
  static pack<T, to_pack_unroll_ix, SimdExt>
  to_pack(const packx<T, init_N, SimdExt> &from_packx_initN,
          const pack<T, 1, SimdExt> &from_pack) {

    pack<T, to_pack_unroll_ix, SimdExt> to_pack_;
    to_pack_.car = from_pack.car; // simd_vector

    // get next pack<T, init_N> with index v_ix
    to_pack_.cdr = to_pack_recurs_helper<
        T, init_N, init_N, to_pack_unroll_ix - 1, v_ix + 1, SimdExt,
        packx>::to_pack(from_packx_initN, get_pack<v_ix + 1>(from_packx_initN));
    return to_pack_;
  }
};

template <typename T, int packx_unroll_ix /* N */,
          int to_pack_unroll_ix = 2 * packx_unroll_ix, typename SimdExt,
          template <typename, int, typename> class packx>
pack<T, to_pack_unroll_ix, SimdExt>
to_pack(const packx<T, packx_unroll_ix, SimdExt> &from_packx_initN) {
  pack<T, to_pack_unroll_ix, SimdExt> to_pack_;
  to_pack_.car = from_packx_initN.v0.car; // simd_vector
  to_pack_.cdr =
      to_pack_recurs_helper<T, packx_unroll_ix, packx_unroll_ix - 1,
                            to_pack_unroll_ix - 1, 0 /* v_ix */, SimdExt,
                            packx>::to_pack(from_packx_initN,
                                            from_packx_initN.v0.cdr);
  return to_pack_;
}

// ----------------------------------------------------------------------------
// to_pack_interleave

template <typename T, typename SimdExt = NSIMD_SIMD>
pack<T, 1, SimdExt> to_pack_interleave(const pack<T, 1, SimdExt> &pack_) {
  return pack_;
}

template <typename T, int N, typename SimdExt = NSIMD_SIMD>
pack<T, N, SimdExt> to_pack_interleave(const pack<T, N, SimdExt> &pack_) {
  return pack_;
}

// ----------------------------------------------------------------------------

template <typename T, typename SimdExt = NSIMD_SIMD>
pack<T, 2, SimdExt> to_pack_interleave(const packx2<T, 1, SimdExt> &packx2_) {

  nsimd::pack<T, 2, SimdExt> pack_2;
  pack_2.car = packx2_.v0.car;
  pack_2.cdr.car = packx2_.v1.car;

  return pack_2;
}

template <typename T, int N, typename SimdExt = NSIMD_SIMD>
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

template <typename T, typename SimdExt = NSIMD_SIMD>
pack<T, 3, SimdExt> to_pack_interleave(const packx3<T, 1, SimdExt> &packx3_) {

  nsimd::pack<T, 3, SimdExt> pack_3;
  pack_3.car = packx3_.v0.car;
  pack_3.cdr.car = packx3_.v1.car;
  pack_3.cdr.cdr.car = packx3_.v2.car;

  return pack_3;
}

template <typename T, int N, typename SimdExt = NSIMD_SIMD>
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

template <typename T, typename SimdExt = NSIMD_SIMD>
pack<T, 4, SimdExt> to_pack_interleave(const packx4<T, 1, SimdExt> &packx4_) {

  nsimd::pack<T, 4, SimdExt> pack_4;
  pack_4.car = packx4_.v0.car;
  pack_4.cdr.car = packx4_.v1.car;
  pack_4.cdr.cdr.car = packx4_.v2.car;
  pack_4.cdr.cdr.cdr.car = packx4_.v3.car;

  return pack_4;
}

template <typename T, int N, typename SimdExt = NSIMD_SIMD>
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
