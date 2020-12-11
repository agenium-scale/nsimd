/*

Copyright (c) 2020 Agenium Scale

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

#define STATUS "test of get_pack over all types"

#include "tests_helpers.hpp"

// ----------------------------------------------------------------------------
// Little helper for scope memory

// ----------------------------------------------------------------------------

template <typename T> bool get_pack_from_pack_N_1() {

  LOG_TEST_DEBUG("get_pack_from_pack_N_1", T);

  nsimd::pack<T, 1> pack_1(42);
  nsimd::pack<T, 1> v0_get = nsimd::get_pack<0>(pack_1);

  nsimd::scoped_aligned_mem_for<T> expected(NSIMD_MAX_LEN_BIT / 8);
  nsimd::scoped_aligned_mem_for<T> computed(NSIMD_MAX_LEN_BIT / 8);

  return nsimd_tests::check_pack_expected_vs_computed(
      pack_1, v0_get, "nsimd::pack<T, 1>", "nsimd::pack<T, 1>", expected.get(),
      computed.get());
}

// ----------------------------------------------------------------------------

template <typename T> bool get_pack_from_packx2_N_3() {

  LOG_TEST_DEBUG("get_pack_from_packx2_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);

  nsimd::packx2<T, 3> packx2_3;
  packx2_3.v0 = v0;
  packx2_3.v1 = v1;

  nsimd::scoped_aligned_mem_for<T> expected(3 * NSIMD_MAX_REGISTER_SIZE_BYTES);
  nsimd::scoped_aligned_mem_for<T> computed(3 * NSIMD_MAX_REGISTER_SIZE_BYTES);

  nsimd::pack<T, 3> v0_get = nsimd::get_pack<0>(packx2_3);
  if (!nsimd_tests::check_pack_expected_vs_computed(
          v0, v0_get, "nsimd::packx2<T, 3>.v0", "nsimd::pack<T, 3>",
          expected.get(), computed.get())) {
    return false;
  }

  nsimd::pack<T, 3> v1_get = nsimd::get_pack<1>(packx2_3);
  return nsimd_tests::check_pack_expected_vs_computed(
      v1, v1_get, "nsimd::packx2<T, 3>.v1", "nsimd::pack<T, 3>",
      expected.get(), computed.get());
}

// ----------------------------------------------------------------------------

template <typename T> bool get_pack_from_packx3_N_3() {

  LOG_TEST_DEBUG("get_pack_from_packx3_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);
  nsimd::pack<T, 3> v2(66);

  nsimd::packx3<T, 3> packx3_3;

  packx3_3.v0 = v0;
  packx3_3.v1 = v1;
  packx3_3.v2 = v2;

  nsimd::scoped_aligned_mem_for<T> expected(3 * NSIMD_MAX_REGISTER_SIZE_BYTES);
  nsimd::scoped_aligned_mem_for<T> computed(3 * NSIMD_MAX_REGISTER_SIZE_BYTES);

  nsimd::pack<T, 3> v0_get = nsimd::get_pack<0>(packx3_3);
  if (!nsimd_tests::check_pack_expected_vs_computed(
          v0, v0_get, "nsimd::packx3<T, 3>.v0", "nsimd::pack<T, 3>",
          expected.get(), computed.get())) {
    return false;
  }

  nsimd::pack<T, 3> v1_get = nsimd::get_pack<1>(packx3_3);
  if (!nsimd_tests::check_pack_expected_vs_computed(
          v1, v1_get, "nsimd::packx3<T, 3>.v1", "nsimd::pack<T, 3>",
          expected.get(), computed.get())) {
    return false;
  }

  nsimd::pack<T, 3> v2_get = nsimd::get_pack<2>(packx3_3);
  return nsimd_tests::check_pack_expected_vs_computed(
      v2, v2_get, "nsimd::packx3<T, 3>.v2", "nsimd::pack<T, 3>",
      expected.get(), computed.get());
}

// ----------------------------------------------------------------------------

template <typename T> bool get_pack_from_packx4_N_3() {

  LOG_TEST_DEBUG("get_pack_from_packx4_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);
  nsimd::pack<T, 3> v2(66);
  nsimd::pack<T, 3> v3(90);

  nsimd::packx4<T, 3> packx4_3;

  packx4_3.v0 = v0;
  packx4_3.v1 = v1;
  packx4_3.v2 = v2;
  packx4_3.v3 = v3;

  nsimd::scoped_aligned_mem_for<T> expected(3 * NSIMD_MAX_REGISTER_SIZE_BYTES);
  nsimd::scoped_aligned_mem_for<T> computed(3 * NSIMD_MAX_REGISTER_SIZE_BYTES);

  nsimd::pack<T, 3> v0_get = nsimd::get_pack<0>(packx4_3);
  if (!nsimd_tests::check_pack_expected_vs_computed(
          v0, v0_get, "nsimd::packx4<T, 3>.v0", "nsimd::pack<T, 3>",
          expected.get(), computed.get())) {
    return false;
  }

  nsimd::pack<T, 3> v1_get = nsimd::get_pack<1>(packx4_3);
  if (!nsimd_tests::check_pack_expected_vs_computed(
          v1, v1_get, "nsimd::packx4<T, 3>.v1", "nsimd::pack<T, 3>",
          expected.get(), computed.get())) {
    return false;
  }

  nsimd::pack<T, 3> v2_get = nsimd::get_pack<2>(packx4_3);
  if (!nsimd_tests::check_pack_expected_vs_computed(
          v2, v2_get, "nsimd::packx4<T, 3>.v2", "nsimd::pack<T, 3>",
          expected.get(), computed.get())) {
    return false;
  }

  nsimd::pack<T, 3> v3_get = nsimd::get_pack<3>(packx4_3);
  return nsimd_tests::check_pack_expected_vs_computed(
      v3, v3_get, "nsimd::packx4<T, 3>.v3", "nsimd::pack<T, 3>",
      expected.get(), computed.get());
}

// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------

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
