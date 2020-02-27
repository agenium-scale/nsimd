#ifndef TESTS_HELPERS_HPP
#define TESTS_HELPERS_HPP

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/nsimd.h>

#define NSIMD_LOG_DEBUG 0

#define LOG_TEST(test_name, T)                                                \
  do {                                                                        \
    if (NSIMD_LOG_DEBUG) {                                                    \
      fprintf(stdout, "%s%s%s%s%s", "\n--------- ", get_type_str(T()), ": ",  \
              test_name, "---------------\n\n");                              \
    }                                                                         \
  } while (0)

#define LOG_PACK(vout, len, pack_type)                                        \
  do {                                                                        \
    if (NSIMD_LOG_DEBUG) {                                                    \
      print(vout, len, pack_type);                                            \
    }                                                                         \
  } while (0)

#define SIZE (4096 / 8)

#define CHECK(a)                                                              \
  {                                                                           \
    errno = 0;                                                                \
    if (!(a)) {                                                               \
      fprintf(stderr, "ERROR: " #a ":%d: %s\n", __LINE__, strerror(errno));   \
      fflush(stderr);                                                         \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }

/* -------------------------------------------------------------------------
 */

template <typename T> int comp_function(const T expected, const T computed) {
  return expected != computed;
  ;
}

namespace fprintf_helper {
// silent the warning for implicit conversion from ‘float’ to ‘double’ when
// passing argument to fprintf
template <typename T> struct f64_if_f32_else_T { typedef T value_type; };
template <> struct f64_if_f32_else_T<f32> { typedef f64 value_type; };

const char *specifier(i8) { return "%hhu"; }
const char *specifier(u8) { return "%hhu"; }
const char *specifier(i16) { return "%hd"; }
const char *specifier(u16) { return "%hu"; }
const char *specifier(i32) { return "%d"; }
const char *specifier(u32) { return "%u"; }
const char *specifier(i64) { return "%ld"; }
const char *specifier(u64) { return "%lu"; }
const char *specifier(f32) { return "%f"; }
const char *specifier(f64) { return "%f"; }

} // namespace fprintf_helper

const char *get_type_str(i8) { return "i8"; }
const char *get_type_str(u8) { return "u8"; }
const char *get_type_str(i16) { return "i16"; }
const char *get_type_str(u16) { return "u16"; }
const char *get_type_str(i32) { return "i32"; }
const char *get_type_str(u32) { return "u32"; }
const char *get_type_str(i64) { return "i64"; }
const char *get_type_str(u64) { return "u64"; }
const char *get_type_str(f32) { return "f32"; }
const char *get_type_str(f64) { return "f64"; }

template <typename T>
void print(T *const arr, const int len, const char *msg) {
  fprintf(stdout, "%-24s: ", msg);
  char formatter[12];
  strcpy(formatter, "%s");
  strcat(formatter, fprintf_helper::specifier(T()));
  for (int ii = 0; ii < len; ++ii) {
    fprintf(
        stdout, formatter, 0 == ii ? "{" : ", ",
        (typename fprintf_helper::f64_if_f32_else_T<T>::value_type)arr[ii]);
  }
  fprintf(stdout, "%s", "}\n");
  fflush(stdout);
}

template <typename T>
void init_array(T *const vout_expected, T *const vout_computed,
                const int len) {
  for (int ii = 0; ii < len; ++ii) {
    vout_expected[ii] = (T)-1;
    vout_computed[ii] = (T)1;
  }
}

/* ----------------------------- storea ------------------------------
 */
// tests helper
// storea
// storea for all packx[Y]<1 .. N> Y in {1, 2, 3, 4}

// struct storea_recurs_helper for packx[Y]<1 .. N> y in {2, 3, 4}

// General definition
template <typename T, int N, typename SimdExt,
          template <typename, int, typename> class pack_t, int VIx,
          bool EndRecurs>
struct storea_recurs_helper {};

// Recursive case
template <typename T, int N, typename SimdExt,
          template <typename, int, typename> class pack_t, int VIx>
struct storea_recurs_helper<T, N, SimdExt, pack_t, VIx, false> {
  void operator()(T *const begin, const pack_t<T, N, SimdExt> &pack_) const {
    nsimd::storea(begin, nsimd::get_pack<VIx>(pack_));
    storea_recurs_helper<T, N, SimdExt, pack_t, VIx + 1,
                         VIx + 1 == pack_t<T, N, SimdExt>::soa_num_packs>()(
        begin + nsimd::len(nsimd::pack<T, N, SimdExt>()), pack_);
  }
};

// Base case
template <typename T, int N, typename SimdExt,
          template <typename, int, typename> class pack_t, int VIx>
struct storea_recurs_helper<T, N, SimdExt, pack_t, VIx, true> {
  void operator()(T *const begin, const pack_t<T, N, SimdExt> &pack_) const {
    (void)begin;
    (void)pack_;
  }
};

// storea function for packx[Y]<1 .. N> y in {2, 3, 4}
template <typename T, int N, typename SimdExt,
          template <typename, int, typename> class pack_t>
void storea__(T *const begin, const pack_t<T, N, SimdExt> &pack_) {
  storea_recurs_helper<T, N, SimdExt, pack_t, 0,
                       0 == pack_t<T, N, SimdExt>::soa_num_packs>()(begin,
                                                                    pack_);
}

// storea for pack<1 .. N>
template <typename T, int N, typename SimdExt>
void storea__(T *const begin, const nsimd::pack<T, N, SimdExt> &pack_) {
  nsimd::storea(begin, pack_);
}

/* ---------------------- check_array ------------------------------- */

template <typename T>
bool check_array(const T *const vout_expected, const T *const vout_computed,
                 const int len_) {
  for (int ii = 0; ii < len_; ++ii) {
    if (comp_function(vout_expected[ii], vout_computed[ii])) {
      fprintf(stdout, STATUS "... FAIL\n");
      fflush(stdout);
      return 0;
    }
  }
  return 1;
}

/* ---------------------- check_pack_content ------------------------ */

template <typename T, int N_From, int N_To, typename SimdExt,
          template <typename, int, typename> class PackFrom,
          template <typename, int, typename> class PackTo>
bool check_pack_content(const PackFrom<T, N_From, SimdExt> &pack_from,
                        const PackTo<T, N_To, SimdExt> &pack_to,
                        const char *from_type, const char *to_type,
                        T *const vout_expected, T *const vout_computed) {

  if (nsimd::len(pack_from) != nsimd::len(pack_to)) {
    return 0;
  }
  const int len_ = nsimd::len(pack_to);
  init_array(vout_expected, vout_computed, len_);

  storea__(vout_expected, pack_from);
  LOG_PACK(vout_expected, nsimd::len(pack_from), from_type);

  nsimd::storea(vout_computed, pack_to);
  LOG_PACK(vout_computed, nsimd::len(pack_to), to_type);

  if (!check_array(vout_expected, vout_computed, len_)) {
    return 0;
  }

  return 1;
}

#endif