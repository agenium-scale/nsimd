#define STATUS "test of scoped_aligned_mem over all types"

#include "tests_helpers.hpp"

template <typename T, typename AlignedScopedMem> bool test_release() {

  LOG_TEST_DEBUG("test_release", T);

  T *iter = NULL;
  nat num_elems = 0;

  {
    AlignedScopedMem v0;
    T *const iter_v0 = v0.get();

    for (int ii = 0; ii < v0.num_elems; ++ii) {
      iter_v0[ii] = (T)ii;
    }

    iter = v0.release();
    if (NULL != v0.get()) {
      return TEST_NSIMD_FALSE;
    }
    num_elems = v0.num_elems;
  }

  for (int ii = 0; ii < num_elems; ++ii) {
    if (nsimd_tests::expected_not_equal_computed((T)ii, iter[ii])) {
      LOG_MEMORY_CONTENT_DEBUG(iter, num_elems, "aligned memory c array");
      AlignedScopedMem::free(&iter);
      return TEST_NSIMD_FALSE;
    }
  }

  AlignedScopedMem::free(&iter);
  return TEST_NSIMD_TRUE;
}

template <typename T> bool test_release_all() {
  if (!test_release<T, nsimd::scoped_aligned_mem<T> >()) {
    return TEST_NSIMD_FALSE;
  }
  if (!test_release<T, nsimd::scoped_aligned_mem<
                           T, 2 * NSIMD_MAX_REGISTER_SIZE_BYTES> >()) {
    return TEST_NSIMD_FALSE;
  }
  if (!test_release<T, nsimd::scoped_aligned_mem<
                           T, 3 * NSIMD_MAX_REGISTER_SIZE_BYTES> >()) {
    return TEST_NSIMD_FALSE;
  }
  if (!test_release<T, nsimd::scoped_aligned_mem_for<T, 4> >()) {
    return TEST_NSIMD_FALSE;
  }
  if (!test_release<T, nsimd::scoped_aligned_mem_for<T, 8> >()) {
    return TEST_NSIMD_FALSE;
  }
  return TEST_NSIMD_TRUE;
}

template <typename T, typename AlignedScopedMem> bool test_reset() {

  LOG_TEST_DEBUG("test_reset", T);

  AlignedScopedMem v0;

  {
    AlignedScopedMem v1;
    T *const iter_v1 = v1.get();
    for (int ii = 0; ii < v1.num_elems; ++ii) {
      iter_v1[ii] = (T)ii;
    }
    v0.reset(v1.release());
    if (NULL != v1.get()) {
      return TEST_NSIMD_FALSE;
    }
  }

  T *const iter_v0 = v0.get();

  for (int ii = 0; ii < v0.num_elems; ++ii) {
    if (nsimd_tests::expected_not_equal_computed((T)ii, iter_v0[ii])) {
      LOG_MEMORY_CONTENT_DEBUG(iter_v0, v0.num_elems,
                               "aligned memory c array");
      return TEST_NSIMD_FALSE;
    }
  }

  return TEST_NSIMD_TRUE;
}

template <typename T> bool test_reset_all() {
  if (!test_reset<T, nsimd::scoped_aligned_mem<T> >()) {
    return TEST_NSIMD_FALSE;
  }
  if (!test_reset<T, nsimd::scoped_aligned_mem<
                         T, 2 * NSIMD_MAX_REGISTER_SIZE_BYTES> >()) {
    return TEST_NSIMD_FALSE;
  }
  if (!test_reset<T, nsimd::scoped_aligned_mem<
                         T, 3 * NSIMD_MAX_REGISTER_SIZE_BYTES> >()) {
    return TEST_NSIMD_FALSE;
  }
  if (!test_reset<T, nsimd::scoped_aligned_mem_for<T, 4> >()) {
    return TEST_NSIMD_FALSE;
  }
  if (!test_reset<T, nsimd::scoped_aligned_mem_for<T, 8> >()) {
    return TEST_NSIMD_FALSE;
  }
  return TEST_NSIMD_TRUE;
}

template <typename T> bool test_all() {
  if (!test_release_all<T>()) {
    return TEST_NSIMD_FALSE;
  }
  if (!test_reset_all<T>()) {
    return TEST_NSIMD_FALSE;
  }
  return TEST_NSIMD_TRUE;
}

int main(void) {

  if (!test_all<i8>() || !test_all<u8>() || !test_all<i16>() ||
      !test_all<u16>() || !test_all<i32>() || !test_all<u32>() ||
      !test_all<i64>() || !test_all<u64>() || !test_all<f32>() ||
      !test_all<f64>()) {
    fprintf(stdout, STATUS "... FAIL\n");
    fflush(stdout);
    return TEST_NSIMD_ERROR;
  }

  fprintf(stdout, STATUS "... OK\n");
  fflush(stdout);
  return 0;
}
