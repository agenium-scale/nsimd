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

template <typename T, typename SimdExt>
NSIMD_STRUCT pack<T, 1, SimdExt> {
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

template <typename T, int N, typename SimdExt>
NSIMD_STRUCT pack {
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

template <typename T, typename SimdExt>
NSIMD_STRUCT packl<T, 1, SimdExt> {
  typename simd_traits<T, SimdExt>::simd_vectorl car;

  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = 1;
};

template <typename T, int N, typename SimdExt>
NSIMD_STRUCT packl {
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

template <typename T, int N, typename SimdExt>
NSIMD_STRUCT packx4 {
  typedef typename simd_traits<T, SimdExt>::simd_vector simd_vector;
  typedef T value_type;
  typedef SimdExt simd_ext;
  static const int unroll = N;

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

  void set_cdr(pack<T, N - 1, SimdExt> const &v0_,
               pack<T, N - 1, SimdExt> const &v1_,
               pack<T, N - 1, SimdExt> const &v2_,
               pack<T, N - 1, SimdExt> const &v3_) {
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

template <typename T, typename SimdExt>
T addv(pack<T, 1, SimdExt> const &a0) {
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

} // namespace nsimd

#endif
