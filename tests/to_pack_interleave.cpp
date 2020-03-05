#define STATUS "test of to_pack_interleave over all types"

#include "tests_helpers.hpp"

template <typename T> bool to_pack_interleave_from_pack_1_N_1() {

  LOG_TEST_DEBUG("to_pack_interleave_from_pack_1_N_1", T);

  nsimd::pack<T, 1> pack_from(42);
  nsimd::pack<T, 1> pack_to = nsimd::to_pack_interleave(pack_from);

  nsimd::scoped_aligned_mem<T> expected;
  nsimd::scoped_aligned_mem<T> computed;

  return nsimd_tests::check_pack_expected_vs_computed(
      pack_from, pack_to, "nsimd::pack<T, 1>", "nsimd::pack<T, 1>",
      expected.get(), computed.get());
}

template <typename T> bool to_pack_interleave_from_packx2_N_1() {

  LOG_TEST_DEBUG("to_pack_interleave_from_packx2_N_1", T);

  nsimd::pack<T, 1> v0(42);
  nsimd::pack<T, 1> v1(24);

  nsimd::packx2<T, 1> pack_from;
  pack_from.v0 = v0;
  pack_from.v1 = v1;

  nsimd::scoped_aligned_mem<T, 2 * NSIMD_MAX_REGISTER_SIZE_BYTES> expected;
  nsimd::scoped_aligned_mem<T, 2 * NSIMD_MAX_REGISTER_SIZE_BYTES> computed;

  const int len_ = nsimd::len(nsimd::packx2<T, 1>());
  nsimd_tests::init_arrays(expected.get(), computed.get(), len_);

  T *begin = expected.get();
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.car));
  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.car));

  LOG_MEMORY_CONTENT_DEBUG(expected.get(), len_, "nsimd::packx2<T, 1>");

  nsimd::pack<T, 2> pack_to = nsimd::to_pack_interleave(pack_from);
  nsimd::storea(computed.get(), pack_to);

  LOG_MEMORY_CONTENT_DEBUG(computed.get(), len_, "nsimd::pack<T, 2>");

  return nsimd_tests::check_arrays(expected.get(), computed.get(), len_);
}

template <typename T> bool to_pack_interleave_from_packx2_N_2() {

  LOG_TEST_DEBUG("to_pack_interleave_from_packx2_N_2", T);

  nsimd::pack<T, 2> v0(42);
  nsimd::pack<T, 2> v1(24);

  nsimd::packx2<T, 2> pack_from;
  pack_from.v0 = v0;
  pack_from.v1 = v1;

  nsimd::scoped_aligned_mem<T, 4 * NSIMD_MAX_REGISTER_SIZE_BYTES> expected;
  nsimd::scoped_aligned_mem<T, 4 * NSIMD_MAX_REGISTER_SIZE_BYTES> computed;

  const int len_ = nsimd::len(nsimd::packx2<T, 2>());
  nsimd_tests::init_arrays(expected.get(), computed.get(), len_);

  T *begin = expected.get();
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.cdr.car));

  LOG_MEMORY_CONTENT_DEBUG(expected.get(), len_, "nsimd::packx2<T, 2>");

  nsimd::pack<T, 4> pack_to = nsimd::to_pack_interleave(pack_from);
  nsimd::storea(computed.get(), pack_to);

  LOG_MEMORY_CONTENT_DEBUG(computed.get(), len_, "nsimd::pack<T, 4>");

  return nsimd_tests::check_arrays(expected.get(), computed.get(), len_);
}

template <typename T> bool to_pack_interleave_from_packx3_N_2() {

  LOG_TEST_DEBUG("to_pack_interleave_from_packx3_N_2", T);

  nsimd::pack<T, 2> v0(42);
  nsimd::pack<T, 2> v1(24);
  nsimd::pack<T, 2> v2(66);

  nsimd::packx3<T, 2> pack_from;
  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;

  nsimd::scoped_aligned_mem<T, 6 * NSIMD_MAX_REGISTER_SIZE_BYTES> expected;
  nsimd::scoped_aligned_mem<T, 6 * NSIMD_MAX_REGISTER_SIZE_BYTES> computed;

  const int len_ = nsimd::len(nsimd::packx3<T, 2>());
  nsimd_tests::init_arrays(expected.get(), computed.get(), len_);

  T *begin = expected.get();
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.cdr.car));

  LOG_MEMORY_CONTENT_DEBUG(expected.get(), len_, "nsimd::packx3<T, 2>");

  nsimd::pack<T, 6> pack_to = nsimd::to_pack_interleave(pack_from);
  nsimd::storea(computed.get(), pack_to);

  LOG_MEMORY_CONTENT_DEBUG(computed.get(), len_, "nsimd::pack<T, 6>");

  return nsimd_tests::check_arrays(expected.get(), computed.get(), len_);
}

template <typename T> bool to_pack_interleave_from_packx3_N_3() {

  LOG_TEST_DEBUG("to_pack_interleave_from_packx3_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);
  nsimd::pack<T, 3> v2(66);

  nsimd::packx3<T, 3> pack_from;
  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;

  nsimd::scoped_aligned_mem<T, 9 * NSIMD_MAX_REGISTER_SIZE_BYTES> expected;
  nsimd::scoped_aligned_mem<T, 9 * NSIMD_MAX_REGISTER_SIZE_BYTES> computed;

  const int len_ = nsimd::len(nsimd::packx3<T, 3>());
  nsimd_tests::init_arrays(expected.get(), computed.get(), len_);

  T *begin = expected.get();
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.cdr.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.cdr.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.cdr.cdr.car));

  LOG_MEMORY_CONTENT_DEBUG(expected.get(), len_, "nsimd::packx3<T, 3>");

  nsimd::pack<T, 9> pack_to = nsimd::to_pack_interleave(pack_from);
  nsimd::storea(computed.get(), pack_to);

  LOG_MEMORY_CONTENT_DEBUG(computed.get(), len_, "nsimd::pack<T, 9>");

  return nsimd_tests::check_arrays(expected.get(), computed.get(), len_);
}

template <typename T> bool to_pack_interleave_from_packx4_N_1() {

  LOG_TEST_DEBUG("to_pack_interleave_from_packx4_N_1", T);

  nsimd::pack<T, 1> v0(42);
  nsimd::pack<T, 1> v1(24);
  nsimd::pack<T, 1> v2(66);
  nsimd::pack<T, 1> v3(132);

  nsimd::packx4<T, 1> pack_from;
  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;
  pack_from.v3 = v3;

  nsimd::scoped_aligned_mem<T, 4 * NSIMD_MAX_REGISTER_SIZE_BYTES> expected;
  nsimd::scoped_aligned_mem<T, 4 * NSIMD_MAX_REGISTER_SIZE_BYTES> computed;

  const int len_ = nsimd::len(nsimd::packx4<T, 1>());
  nsimd_tests::init_arrays(expected.get(), computed.get(), len_);

  T *begin = expected.get();
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v3.car));

  LOG_MEMORY_CONTENT_DEBUG(expected.get(), len_, "nsimd::packx4<T, 1>");

  nsimd::pack<T, 4> pack_to = nsimd::to_pack_interleave(pack_from);
  nsimd::storea(computed.get(), pack_to);

  LOG_MEMORY_CONTENT_DEBUG(computed.get(), len_, "nsimd::pack<T, 4>");

  return nsimd_tests::check_arrays(expected.get(), computed.get(), len_);
}

template <typename T> bool to_pack_interleave_from_packx4_N_2() {

  LOG_TEST_DEBUG("to_pack_interleave_from_packx4_N_2", T);

  nsimd::pack<T, 2> v0(42);
  nsimd::pack<T, 2> v1(24);
  nsimd::pack<T, 2> v2(66);
  nsimd::pack<T, 2> v3(132);

  nsimd::packx4<T, 2> pack_from;
  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;
  pack_from.v3 = v3;

  nsimd::scoped_aligned_mem<T, 8 * NSIMD_MAX_REGISTER_SIZE_BYTES> expected;
  nsimd::scoped_aligned_mem<T, 8 * NSIMD_MAX_REGISTER_SIZE_BYTES> computed;

  const int len_ = nsimd::len(nsimd::packx4<T, 2>());
  nsimd_tests::init_arrays(expected.get(), computed.get(), len_);

  T *begin = expected.get();
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v3.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v3.cdr.car));

  LOG_MEMORY_CONTENT_DEBUG(expected.get(), len_, "nsimd::packx4<T, 2>");

  nsimd::pack<T, 8> pack_to = nsimd::to_pack_interleave(pack_from);
  nsimd::storea(computed.get(), pack_to);

  LOG_MEMORY_CONTENT_DEBUG(computed.get(), len_, "nsimd::pack<T, 8>");

  return nsimd_tests::check_arrays(expected.get(), computed.get(), len_);
}

template <typename T> bool to_pack_interleave_from_packx4_N_3() {

  LOG_TEST_DEBUG("to_pack_interleave_from_packx4_N_3", T);

  nsimd::pack<T, 3> v0(42);
  nsimd::pack<T, 3> v1(24);
  nsimd::pack<T, 3> v2(66);
  nsimd::pack<T, 3> v3(132);

  nsimd::packx4<T, 3> pack_from;
  pack_from.v0 = v0;
  pack_from.v1 = v1;
  pack_from.v2 = v2;
  pack_from.v3 = v3;

  nsimd::scoped_aligned_mem<T, 12 * NSIMD_MAX_REGISTER_SIZE_BYTES> expected;
  nsimd::scoped_aligned_mem<T, 12 * NSIMD_MAX_REGISTER_SIZE_BYTES> computed;

  const int len_ = nsimd::len(nsimd::packx4<T, 3>());
  nsimd_tests::init_arrays(expected.get(), computed.get(), len_);

  T *begin = expected.get();
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v3.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v3.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v0.cdr.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v1.cdr.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v2.cdr.cdr.car));

  begin += nsimd::len(nsimd::pack<T, 1>());
  nsimd::storea(begin, nsimd::pack<T, 1>(pack_from.v3.cdr.cdr.car));

  LOG_MEMORY_CONTENT_DEBUG(expected.get(), len_, "nsimd::packx4<T, 3>");

  nsimd::pack<T, 12> pack_to = nsimd::to_pack_interleave(pack_from);
  nsimd::storea(computed.get(), pack_to);

  LOG_MEMORY_CONTENT_DEBUG(computed.get(), len_, "nsimd::pack<T, 12>");

  return nsimd_tests::check_arrays(expected.get(), computed.get(), len_);
}

template <typename T> bool test_all() {
  if (!to_pack_interleave_from_pack_1_N_1<T>()) {
    return 0;
  }
  if (!to_pack_interleave_from_packx2_N_1<T>()) {
    return 0;
  }
  if (!to_pack_interleave_from_packx2_N_2<T>()) {
    return 0;
  }
  if (!to_pack_interleave_from_packx3_N_2<T>()) {
    return 0;
  }
  if (!to_pack_interleave_from_packx3_N_3<T>()) {
    return 0;
  }
  if (!to_pack_interleave_from_packx4_N_1<T>()) {
    return 0;
  }
  if (!to_pack_interleave_from_packx4_N_2<T>()) {
    return 0;
  }
  if (!to_pack_interleave_from_packx4_N_3<T>()) {
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
