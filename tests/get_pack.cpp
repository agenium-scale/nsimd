#define STATUS "test of get_pack over all types"

#include "tests_helpers.hpp"

template <typename T> bool get_pack_from_pack_N_1() {

  LOG_TEST("get_pack_from_pack_N_1", T);

  nsimd::pack<T, 1> pack_1(42);
  nsimd::pack<T, 1> v0_get = nsimd::get_pack<0>(pack_1);

  nsimd_scoped_aligned_mem<T> vout;

  return check_pack_content(pack_1, v0_get, "nsimd::pack<T, 1>",
                            "nsimd::pack<T, 1>", vout.expected, vout.computed);
}

template <typename T> bool get_pack_from_packx2_N_3() {

  LOG_TEST("get_pack_from_packx2_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);

  nsimd::packx2<T, 3> packx2_3;
  packx2_3.v0 = v0;
  packx2_3.v1 = v1;

  nsimd_scoped_aligned_mem<T, 3 * NSIMD_MAX_REGISTER_SIZE_BYTES> vout;

  nsimd::pack<T, 3> v0_get = nsimd::get_pack<0>(packx2_3);
  if (!check_pack_content(v0, v0_get, "nsimd::packx2<T, 3>.v0",
                          "nsimd::pack<T, 3>", vout.expected, vout.computed)) {
    return false;
  }

  nsimd::pack<T, 3> v1_get = nsimd::get_pack<1>(packx2_3);
  return check_pack_content(v1, v1_get, "nsimd::packx2<T, 3>.v1",
                            "nsimd::pack<T, 3>", vout.expected, vout.computed);
}

template <typename T> bool get_pack_from_packx3_N_3() {

  LOG_TEST("get_pack_from_packx3_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);
  nsimd::pack<T, 3> v2(66);

  nsimd::packx3<T, 3> packx3_3;

  packx3_3.v0 = v0;
  packx3_3.v1 = v1;
  packx3_3.v2 = v2;

  nsimd_scoped_aligned_mem<T, 3 * NSIMD_MAX_REGISTER_SIZE_BYTES> vout;

  nsimd::pack<T, 3> v0_get = nsimd::get_pack<0>(packx3_3);
  if (!check_pack_content(v0, v0_get, "nsimd::packx3<T, 3>.v0",
                          "nsimd::pack<T, 3>", vout.expected, vout.computed)) {
    return false;
  }

  nsimd::pack<T, 3> v1_get = nsimd::get_pack<1>(packx3_3);
  if (!check_pack_content(v1, v1_get, "nsimd::packx3<T, 3>.v1",
                          "nsimd::pack<T, 3>", vout.expected, vout.computed)) {
    return false;
  }

  nsimd::pack<T, 3> v2_get = nsimd::get_pack<2>(packx3_3);
  return check_pack_content(v2, v2_get, "nsimd::packx3<T, 3>.v2",
                            "nsimd::pack<T, 3>", vout.expected, vout.computed);
}

template <typename T> bool get_pack_from_packx4_N_3() {

  LOG_TEST("get_pack_from_packx4_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);
  nsimd::pack<T, 3> v2(66);
  nsimd::pack<T, 3> v3(90);

  nsimd::packx4<T, 3> packx4_3;

  packx4_3.v0 = v0;
  packx4_3.v1 = v1;
  packx4_3.v2 = v2;
  packx4_3.v3 = v3;

  nsimd_scoped_aligned_mem<T, 3 * NSIMD_MAX_REGISTER_SIZE_BYTES> vout;

  nsimd::pack<T, 3> v0_get = nsimd::get_pack<0>(packx4_3);
  if (!check_pack_content(v0, v0_get, "nsimd::packx4<T, 3>.v0",
                          "nsimd::pack<T, 3>", vout.expected, vout.computed)) {
    return false;
  }

  nsimd::pack<T, 3> v1_get = nsimd::get_pack<1>(packx4_3);
  if (!check_pack_content(v1, v1_get, "nsimd::packx4<T, 3>.v1",
                          "nsimd::pack<T, 3>", vout.expected, vout.computed)) {
    return false;
  }

  nsimd::pack<T, 3> v2_get = nsimd::get_pack<2>(packx4_3);
  if (!check_pack_content(v2, v2_get, "nsimd::packx4<T, 3>.v2",
                          "nsimd::pack<T, 3>", vout.expected, vout.computed)) {
    return false;
  }

  nsimd::pack<T, 3> v3_get = nsimd::get_pack<3>(packx4_3);
  return check_pack_content(v3, v3_get, "nsimd::packx4<T, 3>.v3",
                            "nsimd::pack<T, 3>", vout.expected, vout.computed);
}

template <typename T> bool test_all() {
  if (!get_pack_from_pack_N_1<T>()) {
    return 0;
  }
  if (!get_pack_from_packx2_N_3<T>()) {
    return 0;
  }
  if (!get_pack_from_packx3_N_3<T>()) {
    return 0;
  }
  if (!get_pack_from_packx4_N_3<T>()) {
    return 0;
  }
  return 1;
}

int main(void) {

  if (!test_all<i8>() || !test_all<u8>() || !test_all<i16>() ||
      !test_all<u16>() || !test_all<i32>() || !test_all<u32>() ||
      !test_all<i64>() || !test_all<u64>() || !test_all<f32>() ||
      !test_all<f64>()) {
    return -1;
  }

  fprintf(stdout, STATUS "... OK\n");
  fflush(stdout);
  return 0;
}
