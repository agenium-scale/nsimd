#define STATUS "to_pack"

#include "tests_helpers.hpp"

template <typename T>
bool to_pack_from_pack_1_N_1(T *const vout_expected, T *const vout_computed) {

  LOG_TEST("to_pack_from_pack_1_N_1", T);

  nsimd::pack<T, 1> pack_from(42);

  nsimd::pack<T, 1> pack_to = nsimd::to_pack(pack_from);
  return check_pack_content(pack_from, pack_to, "nsimd::pack<T, 1>",
                            "nsimd::pack<T, 1>", vout_expected, vout_computed);
}

template <typename T>
bool to_pack_from_packx2_N_1(T *const vout_expected, T *const vout_computed) {

  LOG_TEST("to_pack_from_packx2_N_1", T);

  nsimd::pack<T, 1> v0(42);
  nsimd::pack<T, 1> v1(24);

  nsimd::packx2<T, 1> pack_from;
  pack_from.v0 = v0;
  pack_from.v1 = v1;

  nsimd::pack<T, 2> pack_to = nsimd::to_pack(pack_from);
  return check_pack_content(pack_from, pack_to, "nsimd::packx2<T, 1>",
                            "nsimd::pack<T, 2>", vout_expected, vout_computed);
}

template <typename T>
bool to_pack_from_packx3_N_1(T *const vout_expected, T *const vout_computed) {

  LOG_TEST("to_pack_from_packx3_N_1", T);

  nsimd::pack<T, 1> v0(42);
  nsimd::pack<T, 1> v1(24);
  nsimd::pack<T, 1> v2(66);

  nsimd::packx3<T, 1> pack_from;
  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;

  nsimd::pack<T, 3> pack_to = nsimd::to_pack(pack_from);
  return check_pack_content(pack_from, pack_to, "nsimd::packx3<T, 1>",
                            "nsimd::pack<T, 3>", vout_expected, vout_computed);
}

template <typename T>
bool to_pack_from_packx2_N_2(T *const vout_expected, T *const vout_computed) {

  LOG_TEST("to_pack_from_packx2_N_2", T);

  nsimd::pack<T, 2> v0(42);
  nsimd::pack<T, 2> v1(24);

  nsimd::packx2<T, 2> pack_from;
  pack_from.v0 = v0;
  pack_from.v1 = v1;

  nsimd::pack<T, 4> pack_to = nsimd::to_pack(pack_from);
  return check_pack_content(pack_from, pack_to, "nsimd::packx2<T, 2>",
                            "nsimd::pack<T, 4>", vout_expected, vout_computed);
}

template <typename T>
bool to_pack_from_packx2_N_3(T *const vout_expected, T *const vout_computed) {

  LOG_TEST("to_pack_from_packx2_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);

  nsimd::packx2<T, 3> pack_from;

  pack_from.v0 = v0;
  pack_from.v1 = v1;

  nsimd::pack<T, 6> pack_to = nsimd::to_pack(pack_from);

  return check_pack_content(pack_from, pack_to, "nsimd::packx2<T, 3>",
                            "nsimd::pack<T, 6>", vout_expected, vout_computed);
}

template <typename T>
bool to_pack_from_packx3_N_2(T *const vout_expected, T *const vout_computed) {

  LOG_TEST("to_pack_from_packx3_N_2", T);

  nsimd::pack<T, 2> v0(42);
  nsimd::pack<T, 2> v1(24);
  nsimd::pack<T, 2> v2(66);

  nsimd::packx3<T, 2> pack_from;

  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;

  nsimd::pack<T, 6> pack_to = nsimd::to_pack(pack_from);

  return check_pack_content(pack_from, pack_to, "nsimd::packx3<T, 2>",
                            "nsimd::pack<T, 6>", vout_expected, vout_computed);
}

template <typename T>
bool to_pack_from_packx3_N_3(T *const vout_expected, T *const vout_computed) {

  LOG_TEST("to_pack_from_packx3_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);
  nsimd::pack<T, 3> v2(66);

  nsimd::packx3<T, 3> pack_from;

  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;

  nsimd::pack<T, 9> pack_to = nsimd::to_pack(pack_from);

  return check_pack_content(pack_from, pack_to, "nsimd::packx3<T, 3>",
                            "nsimd::pack<T, 9>", vout_expected, vout_computed);
}

template <typename T>
bool to_pack_from_packx4_N_1(T *const vout_expected, T *const vout_computed) {

  LOG_TEST("to_pack_from_packx4_N_1", T);

  nsimd::pack<T, 1> v0(42);
  nsimd::pack<T, 1> v1(24);
  nsimd::pack<T, 1> v2(66);
  nsimd::pack<T, 1> v3(132);

  nsimd::packx4<T, 1> pack_from;

  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;
  pack_from.v3 = v3;

  nsimd::pack<T, 4> pack_to = nsimd::to_pack(pack_from);

  return check_pack_content(pack_from, pack_to, "nsimd::packx4<T, 1>",
                            "nsimd::pack<T, 4>", vout_expected, vout_computed);
}

template <typename T>
bool to_pack_from_packx4_N_2(T *const vout_expected, T *const vout_computed) {

  LOG_TEST("to_pack_from_packx4_N_2", T);

  nsimd::pack<T, 2> v0(42);
  nsimd::pack<T, 2> v1(24);
  nsimd::pack<T, 2> v2(66);
  nsimd::pack<T, 2> v3(132);

  nsimd::packx4<T, 2> pack_from;

  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;
  pack_from.v3 = v3;

  nsimd::pack<T, 8> pack_to = nsimd::to_pack(pack_from);

  return check_pack_content(pack_from, pack_to, "nsimd::packx4<T, 2>",
                            "nsimd::pack<T, 8>", vout_expected, vout_computed);
}

template <typename T>
bool to_pack_from_packx4_N_3(T *const vout_expected, T *const vout_computed) {

  LOG_TEST("to_pack_from_packx4_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);
  nsimd::pack<T, 3> v2(66);
  nsimd::pack<T, 3> v3(132);

  nsimd::packx4<T, 3> pack_from;

  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;
  pack_from.v3 = v3;

  nsimd::pack<T, 12> pack_to = nsimd::to_pack(pack_from);

  return check_pack_content(pack_from, pack_to, "nsimd::packx4<T, 3>",
                            "nsimd::pack<T, 12>", vout_expected, vout_computed);
}

template <typename T> bool test_all() {

  const int mem_aligned_size = SIZE * 8;

  T *vout_expected;
  T *vout_computed;

  CHECK(vout_expected = (T *)nsimd_aligned_alloc(mem_aligned_size));
  CHECK(vout_computed = (T *)nsimd_aligned_alloc(mem_aligned_size));

  if (!to_pack_from_pack_1_N_1(vout_expected, vout_computed)) {
    return 0;
  }

  if (!to_pack_from_packx2_N_1(vout_expected, vout_computed)) {
    return 0;
  }

  if (!to_pack_from_packx2_N_2(vout_expected, vout_computed)) {
    return 0;
  }

  if (!to_pack_from_packx2_N_3(vout_expected, vout_computed)) {
    return 0;
  }

  if (!to_pack_from_packx3_N_1(vout_expected, vout_computed)) {
    return 0;
  }

  if (!to_pack_from_packx3_N_2(vout_expected, vout_computed)) {
    return 0;
  }

  if (!to_pack_from_packx3_N_3(vout_expected, vout_computed)) {
    return 0;
  }

  if (!to_pack_from_packx4_N_1(vout_expected, vout_computed)) {
    return 0;
  }

  if (!to_pack_from_packx4_N_2(vout_expected, vout_computed)) {
    return 0;
  }

  if (!to_pack_from_packx4_N_3(vout_expected, vout_computed)) {
    return 0;
  }

  nsimd_aligned_free(vout_expected);
  nsimd_aligned_free(vout_computed);

  return 1;
}

int main(void) {

  if (!test_all<i8>() || !test_all<u8>() || !test_all<i16>() ||
      !test_all<u16>() || !test_all<i32>() || !test_all<u32>() ||
      !test_all<i64>() || !test_all<u64>() || !test_all<f32>() ||
      !test_all<f64>()) {
    return -1;
  }

  fprintf(stdout, "\n" STATUS "... OK\n\n");
  fflush(stdout);
  return 0;
}