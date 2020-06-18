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

#ifndef NSIMD_MODULES_TET1D_HPP
#define NSIMD_MODULES_TET1D_HPP

#include <nsimd/nsimd-all.hpp>

#include <cassert>
#include <vector>
#include <cstring>

namespace tet1d {

// ----------------------------------------------------------------------------
// general definitions

struct none_t {};

template <typename Op, typename Left, typename Right, typename Extra>
struct node {};

const nsimd::nat end = nsimd::nat(-1);

// ----------------------------------------------------------------------------
// Error management
#if defined(NSIMD_CUDA)
#define nsimd_cuda_assert(ans) tet1d::gpuCheck((ans), __FILE__, __LINE__)
inline void gpuCheck(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "NSIMD Internal error:\n\ttet1d Error: %s %s %d\n",
        cudaGetErrorString(code), file, line);
    exit(code);
  }
}
#endif

// ----------------------------------------------------------------------------
// supported kernels

#if defined(NSIMD_CUDA)

// CUDA component wise kernel
template <typename T, typename Expr>
__global__ void gpu_kernel_component_wise(T *dst, Expr const expr,
                                          nsimd::nat n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    dst[i] = expr.gpu_get(i);
  }
}

// CUDA component wise kernel with masked output
template <typename T, typename Mask, typename Expr>
__global__ void gpu_kernel_component_wise_mask(T *dst, Mask const mask,
                                               Expr const expr,
                                               nsimd::nat n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n && mask.gpu_get(i)) {
    dst[i] = expr.gpu_get(i);
  }
}

#elif defined(NSIMD_ROCM)

// ROCM component wise kernel
template <typename T, typename Expr>
__global__ void gpu_kernel_component_wise(T *dst, Expr const expr,
                                          nsimd::nat n) {
  int i = int(hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x);
  if (i < n) {
    dst[i] = expr.gpu_get(i);
  }
}

// ROCM component wise kernel with masked output
template <typename T, typename Mask, typename Expr>
__global__ void gpu_kernel_component_wise_mask(T *dst, Mask const mask,
                                               Expr const expr,
                                               nsimd::nat n) {
  int i = int(hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x);
  if (i < n && mask.gpu_get(i)) {
    dst[i] = expr.gpu_get(i);
  }
}

#else

// CPU component wise kernel
template <typename Pack, typename T, typename Expr>
void cpu_kernel_component_wise(T *dst, Expr const &expr, nsimd::nat n) {
  nsimd::nat i;
  int len = nsimd::len(Pack());
  for (i = 0; i + len < n; i += len) {
    nsimd::storeu(&dst[i], expr.template simd_get<Pack>(i));
  }
  for (; i < n; i++) {
    dst[i] = expr.scalar_get(i);
  }
}

// CPU component wise kernel with masked output
template <typename Pack, typename T, typename Mask, typename Expr>
void cpu_kernel_component_wise_mask(T *dst, Mask const &mask, Expr const &expr,
                                    nsimd::nat n) {
  nsimd::nat i;
  int len = nsimd::len(Pack());
  for (i = 0; i + len < n; i += len) {
    nsimd::storeu(&dst[i], nsimd::if_else(mask.template simd_get<Pack>(i),
                                          expr.template simd_get<Pack>(i),
                                          nsimd::loadu<Pack>(&dst[i])));
  }
  for (; i < n; i++) {
    if (mask.scalar_get(i)) {
      dst[i] = expr.scalar_get(i);
    }
  }
}

#endif

// ----------------------------------------------------------------------------
// helper for computing sizes of 1D vectors

nsimd::nat compute_size(nsimd::nat sz1, nsimd::nat sz2) {
  assert(sz1 >= 0 || sz2 >= 0);
  assert((sz1 < 0 && sz2 >= 0) || (sz1 >= 0 && sz2 < 0) || (sz1 == sz2));
  if (sz1 < 0) {
    return sz2;
  } else {
    return sz1;
  }
}

nsimd::nat compute_size(nsimd::nat sz1, nsimd::nat sz2, nsimd::nat sz3) {
  return compute_size(compute_size(sz1, sz2), sz3);
}

// ----------------------------------------------------------------------------
// meta for building a pack from another ignoring the base type

template <typename T, typename Pack> struct to_pack_t {
  static const int unroll = Pack::unroll;
  typedef typename Pack::simd_ext simd_ext;
  typedef nsimd::pack<T, unroll, simd_ext> type;
};

template <typename T, int Unroll, typename SimdExt, typename Pack>
struct to_pack_t<nsimd::pack<T, Unroll, SimdExt>, Pack> {
  static const int unroll = Pack::unroll;
  typedef typename Pack::simd_ext simd_ext;
  typedef nsimd::pack<T, unroll, simd_ext> type;
};

template <typename T, typename Pack> struct to_packl_t {
  static const int unroll = Pack::unroll;
  typedef typename Pack::simd_ext simd_ext;
  typedef nsimd::packl<T, unroll, simd_ext> type;
};

template <typename T, int Unroll, typename SimdExt, typename Pack>
struct to_packl_t<nsimd::pack<T, Unroll, SimdExt>, Pack> {
  static const int unroll = Pack::unroll;
  typedef typename Pack::simd_ext simd_ext;
  typedef nsimd::packl<T, unroll, simd_ext> type;
};

// ----------------------------------------------------------------------------
// scalar node

struct scalar_t {};

template <typename T> struct node<scalar_t, none_t, none_t, T> {
  typedef T in_type;
  typedef T out_type;
  T value;

#if defined(NSIMD_CUDA) || defined(NSIMD_ROCM)
  __device__ T gpu_get(nsimd::nat) const { return value; }
#else
  T scalar_get(nsimd::nat) const { return value; }
  template <typename Pack>
  typename to_pack_t<T, Pack>::type simd_get(nsimd::nat) const {
    typedef typename to_pack_t<T, Pack>::type pack;
    return pack(value);
  }
#endif

  nsimd::nat size() const { return -1; }
};

// ----------------------------------------------------------------------------
// build a node from a scalar and a node

template <typename T> struct to_node_t {
  typedef node<scalar_t, none_t, none_t, T> type;

  static type impl(T n) {
    type ret;
    ret.value = n;
    return ret;
  }
};

template <typename Op, typename Left, typename Right, typename Extra>
struct to_node_t<node<Op, Left, Right, Extra> > {
  typedef node<Op, Left, Right, Extra> type;

  static type impl(type node) { return node; }
};

template <typename T> typename to_node_t<T>::type to_node(T n) {
  return to_node_t<T>::impl(n);
}

// ----------------------------------------------------------------------------
// convert literal to one NSIMD base type

template <typename T> struct literal_to {
  template <typename S> static T impl(S a) { return T(a); }
};

template <> struct literal_to<f16> {
  template <typename S> static f16 impl(S a) {
    return nsimd_f32_to_f16(f32(a));
  }
};

// ----------------------------------------------------------------------------
// input node

struct in_t {};

#define TET1D_IN(T) tet1d::node<tet1d::in_t, tet1d::none_t, tet1d::none_t, T>

template <typename T> struct node<in_t, none_t, none_t, T> {
  const T *data;
  nsimd::nat sz;
  typedef T in_type;
  typedef T out_type;

#if defined(NSIMD_CUDA) || defined(NSIMD_ROCM)
  __device__ T gpu_get(nsimd::nat i) const { return data[i]; }
#else
  T scalar_get(nsimd::nat i) const { return data[i]; }
  template <typename Pack>
  typename to_pack_t<T, Pack>::type simd_get(nsimd::nat i) const {
    typedef typename to_pack_t<T, Pack>::type pack;
    return nsimd::loadu<pack>(&data[i]);
  }
#endif

  nsimd::nat size() const { return sz; }

  template <typename I0, typename I1>
  node<in_t, none_t, none_t, T> operator()(I0 i0_, I1 i1_) const {
    node<in_t, none_t, none_t, T> ret;
    nsimd::nat i0 = nsimd::nat(i0_);
    nsimd::nat i1 = nsimd::nat(i1_);
    i0 = i0 >= 0 ? i0 : sz + i0;
    i1 = i1 >= 0 ? i1 : sz + i1;
    assert(0 <= i0 && i0 < i1 && i1 < sz);
    ret.data = &data[i0];
    ret.sz = i1 - i0 + 1;
    return ret;
  }
};

// return an input node from a pointer
template <typename T, typename I>
inline node<in_t, none_t, none_t, T> in(const T *data, I sz) {
  node<in_t, none_t, none_t, T> ret;
  ret.data = data;
  ret.sz = nsimd::nat(sz);
  return ret;
}

// ----------------------------------------------------------------------------
// output with condition node: I(I > 50) = ...

struct mask_out_t {};

template <typename Mask, typename Pack>
struct node<mask_out_t, Mask, none_t, Pack> {
  typedef typename Pack::value_type T;
  T *data;
  nsimd::nat threads_per_block;
  void *stream;
  Mask mask;

  template <typename Op, typename Left, typename Right, typename Extra>
  node<mask_out_t, Mask, none_t, Pack>
  operator=(node<Op, Left, Right, Extra> const &expr) {
#if defined(NSIMD_CUDA) || defined(NSIMD_ROCM)
    nsimd::nat expr_size = compute_size(mask.size(), expr.size());
    nsimd::nat nt = threads_per_block < 0 ? 128 : threads_per_block;
    nsimd::nat nb = (expr_size + nt - 1) / nt; // div rounded up
    assert(nt > 0 && nt <= UINT_MAX);
    assert(nb > 0 && nb <= UINT_MAX);
#if defined(NSIMD_CUDA)
    cudaStream_t s = (stream == NULL ? NULL : *(cudaStream_t *)stream);

    // clang-format off
    gpu_kernel_component_wise_mask<<<(unsigned int)(nb), (unsigned int)(nt),
                                     0, s>>>
                                     (data, mask, expr, expr_size);
    // clang-format on
#elif defined(NSIMD_ROCM)
    hipStream_t s = stream == NULL ? NULL : *(hipStream_t *)stream;
    hipLaunchKernelGGL(gpu_kernel_component_wise_mask, (unsigned int)(nb),
                       (unsigned int)(nt), 0, s, data, mask, expr,
                       expr_size);
#endif
#else
    cpu_kernel_component_wise_mask<Pack>(
        data, mask, expr, compute_size(mask.size(), expr.size()));
#endif
    return *this;
  }

  template <typename S> node<mask_out_t, Mask, none_t, Pack> operator=(S a) {
    return operator=(to_node(literal_to<T>::impl(a)));
  }
};

// ----------------------------------------------------------------------------
// output node

struct out_t {};

#define TET1D_OUT(T)                                                          \
  tet1d::node<tet1d::out_t, tet1d::none_t, tet1d::none_t, nsimd::pack<T> >

#define TET1D_OUT_EX(T, N, SimdExt)                                           \
  tet1d::node<tet1d::out_t, tet1d::none_t, tet1d::none_t,                     \
              nsimd::pack<T, N, SimdExt> >

template <typename Pack> struct node<out_t, none_t, none_t, Pack> {
  typedef typename Pack::value_type T;
  T *data;
  nsimd::nat threads_per_block;
  void *stream;

  template <typename Mask>
  node<mask_out_t, Mask, none_t, Pack> operator()(Mask mask) const {
    node<mask_out_t, Mask, none_t, Pack> ret;
    ret.data = data;
    ret.mask = mask;
    ret.threads_per_block = threads_per_block;
    ret.stream = stream;
    return ret;
  }

  template <typename Op, typename Left, typename Right, typename Extra>
  node<out_t, none_t, none_t, Pack>
  operator=(node<Op, Left, Right, Extra> const &expr) {
#if defined(NSIMD_CUDA) || defined(NSIMD_ROCM)
    nsimd::nat nt = threads_per_block < 0 ? 128 : threads_per_block;
    nsimd::nat nb = (expr.size() + nt - 1) / nt; // div rounded up
    assert(nt > 0 && nt <= UINT_MAX);
    assert(nb > 0 && nb <= UINT_MAX);
#if defined(NSIMD_CUDA)
    cudaStream_t s = stream == NULL ? NULL : *(cudaStream_t *)stream;

    // clang-format off
    gpu_kernel_component_wise<<<(unsigned int)(nb), (unsigned int)(nt),
                                0, s>>>(data, expr, expr.size());
    // clang-format on

#elif defined(NSIMD_ROCM)
    hipStream_t s = stream == NULL ? NULL : *(hipStream_t *)stream;
    hipLaunchKernelGGL(
        (gpu_kernel_component_wise<T, node<Op, Left, Right, Extra> >),
        (unsigned int)(nb), (unsigned int)(nt), 0, s, data, expr, expr.size());
#endif
#else
    cpu_kernel_component_wise<Pack>(data, expr, expr.size());
#endif
    return *this;
  }
};

// return an output node from a pointer
template <typename T>
node<out_t, none_t, none_t, nsimd::pack<T> > out(T *data) {
  node<out_t, none_t, none_t, nsimd::pack<T> > ret;
  ret.data = data;
  ret.threads_per_block = 128;
  ret.stream = NULL;
  return ret;
}

template <typename T, typename Pack>
node<out_t, none_t, none_t, Pack> out(T *data, int threads_per_block,
                                      void *stream) {
  node<out_t, none_t, none_t, Pack> ret;
  ret.data = data;
  ret.threads_per_block = threads_per_block;
  ret.stream = stream;
  return ret;
}

// ----------------------------------------------------------------------------

} // namespace tet1d

#include <nsimd/modules/tet1d/functions.hpp>

#endif
