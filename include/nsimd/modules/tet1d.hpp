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
#include <stdexcept>
#include <utility>
#include <vector>

namespace tet1d {

// Eval

enum EvalMode { Scalar, Simd, Cuda, Hip };

// in

template <typename T> struct in_t {
  typedef typename T::value_type value_type;
  typedef value_type result_type;
  typedef nsimd::pack<value_type> simd_pack_type;

  T const &data;

  in_t(T const &data_) : data(data_) {
#ifdef NSIMD_IS_NVCC
    update_p_data_cuda();
#endif
  }

  int size() const { return int(data.size()); }

  bool is_scalar() const { return false; }

  value_type operator()(int const i) const { return data[i]; }

#ifdef NSIMD_IS_NVCC
  value_type const *p_data_cuda;
  void update_p_data_cuda() { p_data_cuda = data.data(); }
  __device__ value_type cuda_get(int const i) const {
    return *(p_data_cuda + i);
  }
#endif
};

template <typename T> struct in_scalar_t {
  typedef T value_type;
  typedef value_type result_type;
  typedef nsimd::pack<value_type> simd_pack_type;

  T data; // Copy

  in_scalar_t(T const data_) : data(data_) {}

  int size() const { return 1; }

  bool is_scalar() const { return true; }
};

template <typename T, typename A>
tet1d::in_t<std::vector<T, A> > in(std::vector<T, A> const &data_) {
  return tet1d::in_t<std::vector<T, A> >(data_);
}

template <typename T> tet1d::in_scalar_t<T> in(T const &data_) {
  return tet1d::in_scalar_t<T>(data_);
}

// out

template <typename T> struct out_t;

template <typename T, typename Node>
tet1d::out_t<T> &eval(tet1d::out_t<T> &out, Node const &node,
                      tet1d::EvalMode const eval_mode);

template <typename T> struct out_t {
  typedef typename T::value_type value_type;
  typedef value_type result_type;
  typedef nsimd::pack<value_type> simd_pack_type;

  T &data;

  tet1d::EvalMode eval_mode;

  out_t(T &data_, tet1d::EvalMode const eval_mode_ = tet1d::Simd)
      : data(data_), eval_mode(eval_mode_) {
#ifdef NSIMD_IS_NVCC
    update_p_data_cuda();
#endif
  }

  void resize(size_t const size) { data.resize(size); }

  int size() const { return int(data.size()); }

  bool is_scalar() const { return false; }

  value_type &operator()(int const i) { return data[i]; }
  value_type operator()(int const i) const { return data[i]; }

#ifdef NSIMD_IS_NVCC
  value_type *p_data_cuda;
  void update_p_data_cuda() { p_data_cuda = data.data(); }
  __device__ value_type &cuda_get(int const i) { return *(p_data_cuda + i); }
  __device__ value_type cuda_get(int const i) const {
    return *(p_data_cuda + i);
  }
#endif

  template <typename Node> tet1d::out_t<T> &operator=(Node const &node) {
    return tet1d::eval(*this, node, eval_mode);
  }
};

template <typename T> struct out_scalar_t {
  typedef T value_type;

  T &data;

  tet1d::EvalMode eval_mode;

  out_scalar_t(T &data_, tet1d::EvalMode const eval_mode_ = tet1d::Simd)
      : data(data_), eval_mode(eval_mode_) {}

  int size() const { return 1; }

  bool is_scalar() const { return true; }

  template <typename Node> tet1d::out_t<T> &operator=(Node const &node) {
    return tet1d::eval(*this, node, eval_mode);
  }
};

template <typename T, typename A>
tet1d::out_t<std::vector<T, A> >
out(std::vector<T, A> &data_, tet1d::EvalMode const eval_mode_ = tet1d::Simd) {
  return tet1d::out_t<std::vector<T, A> >(data_, eval_mode_);
}

template <typename T>
tet1d::out_scalar_t<T> out(T &data_,
                           tet1d::EvalMode const eval_mode_ = tet1d::Simd) {
  return tet1d::out_scalar_t<T>(data_, eval_mode_);
}

// Common type

template <typename A0, typename A1> struct common_type {};

template <typename T> struct common_type<T, T> { typedef T result_type; };

// TODO
// template <> struct common_type<float, double> { typedef double result_type;
// }; template <> struct common_type<double, float> { typedef double
// result_type; };

// Operation result type

template <typename Op, typename A> struct op1_result_type_t {};

template <typename Op, typename A0, typename A1> struct op2_result_type_t {};

template <typename Op, typename A0, typename A1, typename A2>
struct op3_result_type_t {};

// Node

template <typename Op, typename A> struct op1_t {
  typedef typename tet1d::op1_result_type_t<
      Op, typename A::result_type>::result_type result_type;
  typedef nsimd::pack<result_type> simd_pack_type;

  Op op;
  A a;

  op1_t(Op op_, A a_) : op(op_), a(a_) {}

  int size() const { return a.size(); }

  bool is_scalar() const { return a.is_scalar(); }
};

template <typename Op, typename A0, typename A1> struct op2_t {
  typedef
      typename tet1d::op2_result_type_t<Op, typename A0::result_type,
                                        typename A1::result_type>::result_type
          result_type;
  typedef nsimd::pack<result_type> simd_pack_type;

  Op op;
  A0 a0;
  A1 a1;

  op2_t(Op op_, A0 a0_, A1 a1_) : op(op_), a0(a0_), a1(a1_) {
    char const *const error_message = "TODO";
    assert((a0.is_scalar() || a1.is_scalar() || a0.size() == a1.size()) &&
           error_message);
  }

  int size() const { return a0.is_scalar() ? a1.size() : a0.size(); }

  bool is_scalar() const { return a0.is_scalar() && a1.is_scalar(); }
};

template <typename Op, typename A0, typename A1, typename A2> struct op3_t {
  typedef typename tet1d::op3_result_type_t<
      Op, typename A0::result_type, typename A1::result_type,
      typename A2::result_type>::result_type result_type;
  typedef nsimd::pack<result_type> simd_pack_type;

  Op op;
  A0 a0;
  A1 a1;
  A2 a2;

  op3_t(Op op_, A0 a0_, A1 a1_, A2 a2_) : op(op_), a0(a0_), a1(a1_), a2(a2_) {
    char const *const error_message = "TODO";
    assert((a0.is_scalar() || a1.is_scalar() || a0.size() == a1.size()) &&
           (a0.is_scalar() || a2.is_scalar() || a0.size() == a2.size()) &&
           (a1.is_scalar() || a2.is_scalar() || a1.size() == a2.size()) &&
           error_message);
  }

  int size() const {
    if (a0.is_scalar()) {
      if (a1.is_scalar()) {
        return a2.size();
      } else {
        return a1.size();
      }
    } else {
      return a0.size();
    }
  }

  bool is_scalar() const { return a0.is_scalar() && a1.is_scalar(); }
};

} // namespace tet1d

#include "tet1d/functions.hpp"

namespace tet1d {

// Eval scalar

template <typename T>
typename tet1d::in_t<T>::value_type eval_scalar_i(tet1d::in_t<T> const &in,
                                                  int const i) {
  return in(i);
}

template <typename T>
T eval_scalar_i(tet1d::in_scalar_t<T> const &in,
                int const i
) {
  return in.data;
}

template <typename T>
typename tet1d::out_t<T>::value_type &eval_scalar_i(tet1d::out_t<T> &out,
                                                    int const i) {
  return out(i);
}

template <typename T>
typename tet1d::out_t<T>::value_type eval_scalar_i(tet1d::out_t<T> const &out,
                                                   int const i) {
  return out(i);
}

template <typename Op, typename A>
typename op1_t<Op, A>::result_type eval_scalar_i(op1_t<Op, A> const &node,
                                                 int const i);

template <typename Op, typename A0, typename A1>
typename op2_t<Op, A0, A1>::result_type
eval_scalar_i(op2_t<Op, A0, A1> const &node, int const i);

template <typename Op, typename A0, typename A1, typename A2>
typename op3_t<Op, A0, A1, A2>::result_type
eval_scalar_i(op3_t<Op, A0, A1, A2> const &node, int const i);

template <typename Op, typename A>
typename op1_t<Op, A>::result_type eval_scalar_i(op1_t<Op, A> const &node,
                                                 int const i) {
  return node.op.eval_scalar(tet1d::eval_scalar_i(node.a, i));
}

template <typename Op, typename A0, typename A1>
typename op2_t<Op, A0, A1>::result_type
eval_scalar_i(op2_t<Op, A0, A1> const &node, int const i) {
  return node.op.eval_scalar(tet1d::eval_scalar_i(node.a0, i),
                             tet1d::eval_scalar_i(node.a1, i));
}

template <typename Op, typename A0, typename A1, typename A2>
typename op3_t<Op, A0, A1, A2>::result_type
eval_scalar_i(op3_t<Op, A0, A1, A2> const &node, int const i) {
  return node.op.eval_scalar(tet1d::eval_scalar_i(node.a0, i),
                             tet1d::eval_scalar_i(node.a1, i),
                             tet1d::eval_scalar_i(node.a2, i));
}

// Eval simd

template <typename T>
typename tet1d::in_t<T>::simd_pack_type eval_simd_i(tet1d::in_t<T> const &in,
                                                    int const i) {
  typedef typename tet1d::in_t<T>::value_type value_type;
  return nsimd::loadu<nsimd::pack<value_type> >(&in.data[i]);
}

template <typename T>
typename tet1d::in_scalar_t<T>::simd_pack_type
eval_simd_i(tet1d::in_scalar_t<T> const &in,
            int const // i
) {
  return nsimd::pack<T>(in.data);
}

template <typename Op, typename A>
typename op1_t<Op, A>::simd_pack_type eval_simd_i(op1_t<Op, A> const &node,
                                                  int const i);

template <typename Op, typename A0, typename A1>
typename op2_t<Op, A0, A1>::simd_pack_type
eval_simd_i(op2_t<Op, A0, A1> const &node, int const i);

template <typename Op, typename A0, typename A1, typename A2>
typename op3_t<Op, A0, A1, A2>::simd_pack_type
eval_simd_i(op3_t<Op, A0, A1, A2> const &node, int const i);

template <typename Op, typename A>
typename op1_t<Op, A>::simd_pack_type eval_simd_i(op1_t<Op, A> const &node,
                                                  int const i) {
  return node.op.eval_simd_i(tet1d::eval_simd_i(node.a, i));
}

template <typename Op, typename A0, typename A1>
typename op2_t<Op, A0, A1>::simd_pack_type
eval_simd_i(op2_t<Op, A0, A1> const &node, int const i) {
  return node.op.eval_simd_i(tet1d::eval_simd_i(node.a0, i),
                             tet1d::eval_simd_i(node.a1, i));
}

template <typename Op, typename A0, typename A1, typename A2>
typename op3_t<Op, A0, A1, A2>::simd_pack_type
eval_simd_i(op3_t<Op, A0, A1, A2> const &node, int const i) {
  return node.op.eval_simd_i(tet1d::eval_simd_i(node.a0, i),
                             tet1d::eval_simd_i(node.a1, i),
                             tet1d::eval_simd_i(node.a2, i));
}

// Eval CUDA

#ifdef NSIMD_IS_NVCC

template <typename T>
__device__ typename tet1d::in_t<T>::value_type
eval_cuda_i(tet1d::in_t<T> const &in, int const i) {
  return in.cuda_get(i);
}

template <typename T>
__device__ T eval_cuda_i(tet1d::in_scalar_t<T> const &in, int const // i
) {
  return in.data;
}

template <typename T>
__device__ typename tet1d::out_t<T>::value_type &
eval_cuda_i(tet1d::out_t<T> &out, int const i) {
  return out.cuda_get(i);
}

template <typename T>
__device__ typename tet1d::out_t<T>::value_type
eval_cuda_i(tet1d::out_t<T> const &out, int const i) {
  return out.cuda_get(i);
}

template <typename Op, typename A>
__device__ typename op1_t<Op, A>::result_type
eval_cuda_i(op1_t<Op, A> const &node, int const i);

template <typename Op, typename A0, typename A1>
__device__ typename op2_t<Op, A0, A1>::result_type
eval_cuda_i(op2_t<Op, A0, A1> const &node, int const i);

template <typename Op, typename A0, typename A1, typename A2>
__device__ typename op3_t<Op, A0, A1, A2>::result_type
eval_cuda_i(op3_t<Op, A0, A1, A2> const &node, int const i);

template <typename Op, typename A>
__device__ typename op1_t<Op, A>::result_type
eval_cuda_i(op1_t<Op, A> const &node, int const i) {
  return node.op.eval_cuda_i(tet1d::eval_cuda_i(node.a, i));
}

template <typename Op, typename A0, typename A1>
__device__ typename op2_t<Op, A0, A1>::result_type
eval_cuda_i(op2_t<Op, A0, A1> const &node, int const i) {
  return node.op.eval_cuda_i(tet1d::eval_cuda_i(node.a0, i),
                             tet1d::eval_cuda_i(node.a1, i));
}

template <typename Op, typename A0, typename A1, typename A2>
__device__ typename op3_t<Op, A0, A1, A2>::result_type
eval_cuda_i(op3_t<Op, A0, A1, A2> const &node, int const i) {
  return node.op.eval_cuda_i(tet1d::eval_cuda_i(node.a0, i),
                             tet1d::eval_cuda_i(node.a1, i),
                             tet1d::eval_cuda_i(node.a2, i));
}

#endif

// Eval

template <typename T, typename Node>
tet1d::out_t<T> &eval_scalar(tet1d::out_t<T> &out, Node const &node) {
  out.resize(node.size());

  for (int i = 0; i < node.size(); ++i) {
    out(i) = tet1d::eval_scalar_i(node, i);
  }

  return out;
}

template <typename T, typename Node>
tet1d::out_t<T> &eval_simd(tet1d::out_t<T> &out, Node const &node) {
  out.resize(node.size());

  int i = 0;
  // simd
  typedef typename tet1d::out_t<T>::value_type value_type;
  typedef typename tet1d::out_t<T>::simd_pack_type simd_pack_type;

  size_t len = size_t(nsimd::len(value_type()));
  for (; i + len < node.size(); i += len) {
    simd_pack_type r = tet1d::eval_simd_i(node, i);
    nsimd::storeu(&(out.data[i]), r);
  }
  // scalar
  for (; i < node.size(); ++i) {
    out(i) = tet1d::eval_scalar_i(node, i);
  }

  return out;
}

#ifdef NSIMD_IS_NVCC
template <typename T, typename Node>
__global__ void eval_cuda_kernel(tet1d::out_t<T> out // Copy
                                 ,
                                 Node const node // Copy
) {
  size_t const i = blockIdx.x;
  out.cuda_get(i) = tet1d::eval_cuda_i(node, i);
}

template <typename T, typename Node>
tet1d::out_t<T> &eval_cuda(tet1d::out_t<T> &out, Node const &node) {
  out.resize(node.size());
  out.update_p_data_cuda();
  // clang-format off
  eval_cuda_kernel<<<node.size(), 1>>>(out, node);
  // clang-format on
  cudaDeviceSynchronize();
  return out;
}
#endif

template <typename T, typename Node>
tet1d::out_t<T> &eval(tet1d::out_t<T> &out, Node const &node,
                      tet1d::EvalMode const eval_mode) {
  switch (eval_mode) {
  case tet1d::Scalar:
    return tet1d::eval_scalar(out, node);
  case tet1d::Simd:
    return tet1d::eval_simd(out, node);
  case tet1d::Cuda:
#ifdef NSIMD_IS_NVCC
    return tet1d::eval_cuda(out, node);
#else
    throw std::runtime_error("Compiler is not nvcc");
#endif
  case tet1d::Hip:
    throw std::runtime_error("Not implemented");
  }
  return out; // GCC warning
}

} // namespace tet1d

#endif
