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
#include <utility>

namespace tet1d {

// void_t

struct void_t { };

// in

template <typename T> struct in_t {
  typedef typename T::value_type value_type;
  typedef value_type result_type;
  typedef nsimd::pack<value_type> simd_pack_type;

  T const &data;

  in_t(T const &data_) : data(data_) {}

  int size() const { return int(data.size()); }

  value_type operator()(int const i) const { return data[i]; }
};

template <typename T> tet1d::in_t<T> in(T &data_) {
  return tet1d::in_t<T>(data_);
}

// out

template <typename T> struct out_t;

template <typename T, typename Node>
tet1d::out_t<T> &eval(tet1d::out_t<T> &out, Node const &node);

template <typename T> struct out_t {
  typedef typename T::value_type value_type;
  typedef value_type result_type;
  typedef nsimd::pack<value_type> simd_pack_type;

  T &data;

  out_t(T &data_) : data(data_) {}

  void resize(size_t const size) { data.resize(size); }

  int size() const { return int(data.size()); }

  value_type &operator()(int const i) { return data[i]; }
  value_type operator()(int const i) const { return data[i]; }

  template <typename Node> tet1d::out_t<T> &operator=(Node const &node) {
    return tet1d::eval(*this, node);
  }
};

template <typename T> tet1d::out_t<T> out(T &data_) {
  return tet1d::out_t<T>(data_);
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
    assert(a0.size() == a1.size() && "TODO");
  }

  int size() const { return a0.size(); }
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
    assert(a0.size() == a1.size() && a0.size() == a2.size() && "TODO");
  }

  int size() const { return a0.size(); }
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
typename tet1d::in_t<T>::simd_pack_type eval_simd_i(tet1d::in_t<T> const &in, int const i) {
  typedef typename tet1d::in_t<T>::value_type value_type;
  return nsimd::loadu<nsimd::pack<value_type> >(&in.data[i]);
}

template <typename Op, typename A>
typename op1_t<Op, A>::simd_pack_type eval_simd_i(op1_t<Op, A> const &node,
                                                 int const i);

template <typename Op, typename A0, typename A1>
typename op2_t<Op, A0, A1>::simd_pack_type eval_simd_i(op2_t<Op, A0, A1> const &node, int const i);

template <typename Op, typename A0, typename A1, typename A2>
typename op3_t<Op, A0, A1, A2>::simd_pack_type eval_simd_i(op3_t<Op, A0, A1, A2> const &node, int const i);

template <typename Op, typename A>
typename op1_t<Op, A>::simd_pack_type eval_simd_i(op1_t<Op, A> const &node,
                                                 int const i) {
  return node.op.eval_simd_i(tet1d::eval_simd_i(node.a, i));
}

template <typename Op, typename A0, typename A1>
typename op2_t<Op, A0, A1>::simd_pack_type eval_simd_i(op2_t<Op, A0, A1> const &node, int const i) {
  return node.op.eval_simd_i(tet1d::eval_simd_i(node.a0, i),
                             tet1d::eval_simd_i(node.a1, i));
}

template <typename Op, typename A0, typename A1, typename A2>
typename op3_t<Op, A0, A1, A2>::simd_pack_type eval_simd_i(op3_t<Op, A0, A1, A2> const &node, int const i) {
  return node.op.eval_simd_i(tet1d::eval_simd_i(node.a0, i),
                             tet1d::eval_simd_i(node.a1, i),
                             tet1d::eval_simd_i(node.a2, i));
}

// Eval

template <typename T, typename Node>
tet1d::out_t<T> &eval(tet1d::out_t<T> &out, Node const &node) {
  out.resize(node.size());

  bool const simd = true;

  int i = 0;
  // simd
  if (simd) {
    typedef typename tet1d::out_t<T>::value_type value_type;
    typedef typename tet1d::out_t<T>::simd_pack_type simd_pack_type;

    size_t len = size_t(nsimd::len(value_type()));
    for (; i + len < node.size(); i += len) {
      simd_pack_type r = tet1d::eval_simd_i(node, i);
      nsimd::storeu(&(out.data[i]), r);
    }
  }
  // scalar
  for (; i < node.size(); ++i) {
    out(i) = tet1d::eval_scalar_i(node, i);
  }

  return out;
}

} // namespace tet1d

#include <cassert>
#include <exception>
#include <nsimd/cxx_adv_api.hpp>
#include <nsimd/nsimd.h>
#include <stdexcept>
#include <vector>

namespace nsimd {

// ----------------------------------------------------------------------------
// Data placement

const int on_both = -1;
const int on_host = 0;
const int on_device = 1;

// ----------------------------------------------------------------------------
// Overview

// TODO: explain how it works

// ----------------------------------------------------------------------------
// A dead-end

struct none_ {};

// ----------------------------------------------------------------------------
// Nodes used for constructing expression template and for CUDA evaluation
// The return_type of these nodes are basic types (float, double, ...)

// A node with 3 arguments
template <typename O, typename A0, typename A1, typename A2> struct n0 {
  A0 a0;
  A1 a1;
  A2 a2;

  //   typedef typename O<typename A0::return_type, typename A1::return_type,
  //   typename A2::return_type>::return_type return_type;

  int get_data_placement() {
    if (a0.get_data_placement() == a1.get_data_placement() &&
        a1.get_data_placement() == a2.get_data_placement()) {
      return a0.get_data_placement();
    }
    return on_both;
  }

  nat get_n() {
    if (a0.get_n() == a1.get_n() && a1.get_n() == a2.get_n()) {
      return a0.get_n();
    }
    return -1;
  }

#ifdef NSIMD_IS_NVCC
  __device__ static return_type cuda_eval(nat i) {
    return O::cuda_compute(a0.eval(i), a1.eval(i), a2.eval(i));
  }
#endif
};

// A node with 2 arguments
template <typename O, typename A0, typename A1> struct n0<O, A0, A1, none_> {
  A0 a0;
  A1 a1;

  //   typedef typename O<typename A0::return_type, typename
  //   A1::return_type>::return_type return_type;

  int get_data_placement() {
    if (a0.get_data_placement() == a1.get_data_placement()) {
      return a0.get_data_placement();
    }
    return on_both;
  }

  nat get_n() {
    if (a0.get_n() == a1.get_n()) {
      return a0.get_n();
    }
    return -1;
  }

#ifdef NSIMD_IS_NVCC
  __device__ static return_type cuda_eval(nat i) {
    return O::cuda_compute(a0.eval(i), a1.eval(i));
  }
#endif
};

// A node with 1 argument
template <typename O, typename A0> struct n0<O, A0, none_, none_> {
  A0 a0;

  //   typedef typename O<typename A0::return_type>::return_type return_type;

  int get_data_placement() { return a0.get_data_placement(); }

  nat get_n() { return a0.get_n(); }

#ifdef NSIMD_IS_NVCC
  __device__ static return_type cuda_eval(nat i) {
    return O::cuda_compute(a0.eval(i));
  }
#endif
};

// ----------------------------------------------------------------------------
// Special nodes are here

// A pointer to some data
template <typename T> struct view_ {};

template <typename T> struct n0<view_<T>, none_, none_, none_> {
  const T *ptr_;
  nat n_;
  int data_placement_;
  typedef T return_type;

  int get_data_placement() { return data_placement_; }

  nat get_n() { return n_; }

#ifdef NSIMD_IS_NVCC
  __device__ T cuda_eval(nat i) const {
    assert(i >= 0 && i < n_);
    return ptr_[i];
  }
#endif
};

// View creator from std::vector
template <typename T>
n0<view_<T>, none_, none_, none_> view(std::vector<T> const &src) {
  n0<view_<T>, none_, none_, none_> ret;
  ret.ptr = src.data();
  ret.n = (nat)src.size();
  ret.data_placement = on_host;
  return ret;
}

// TODO: add creators that eat thrust vectors
#ifdef NSIMD_IS_NVCC
// TODO
#endif

// View creator from a pointer type
template <typename T>
n0<view_<T>, none_, none_, none_> view(T const *src, nat n,
                                       int data_placement) {
  assert(data_placement == on_host || data_placement == on_device);
  n0<view_<T>, none_, none_, none_> ret;
  ret.ptr = src;
  ret.n = n;
  ret.data_placement = data_placement;
  return ret;
}

// ----------------------------------------------------------------------------
// Nodes used for SIMD evaluation: no need for data_placement,
// we are on the host
// The return_type of these nodes are basic types (float, double, ...)

// A node with 3 arguments
template <int N, typename SimdExt, typename O, typename A0, typename A1,
          typename A2>
struct n1 {
  A0 a0;
  A1 a1;
  A2 a2;

  //   typedef typename O<A0::return_type, A1::return_type,
  //   A2::return_type>::return_type return_type;

  //   static pack<return_type, N, SimdExt> eval(nat i) {
  //     return O::compute(a0.eval(i), a1.eval(i), a2.eval(i));
  //   }
};

// A node with 2 arguments
template <int N, typename SimdExt, typename O, typename A0, typename A1>
struct n1<N, SimdExt, O, A0, A1, none_> {
  A0 a0;
  A1 a1;

  //   typedef typename O<A0::return_type, A1::return_type>::return_type
  //   return_type;

  //   static pack<return_type, N, SimdExt> eval(nat i) {
  //     return O::compute(a0.eval(i), a1.eval(i));
  //   }
};

// A node with 1 argument
template <int N, typename SimdVector, typename O, typename A0>
struct n1<N, SimdVector, O, A0, none_, none_> {
  A0 a0;

  //   typedef typename O<A0::return_type>::return_type return_type;

  //   static pack<return_type, N, SimdExt> eval(nat i) {
  //     return O::compute(a0.eval(i));
  //   }
};

// ----------------------------------------------------------------------------
// Transform an n0-tree to an n1-tree

// template <int N, typename SimdExt, typename Tree> struct n0_to_n1 {};
//
// template <N, SimdExt, typename O, typename A0, typename A1, typename A2>
// struct n0_to_n1< N, SimdExt, n0<O, A0, A1, A2> > {
//   typedef n1<N, SimdExt, O, A0, A1, A2> return_type;
//   return_type doit(n0<O, A0, A1, A2> const &node) {
//     return_type ret;
//     ret.a0 = n0_to_n1<A0>::doit(node.a0);
//     ret.a1 = n0_to_n1<A1>::doit(node.a1);
//     ret.a2 = n0_to_n1<A2>::doit(node.a2);
//     return ret;
//   }
// };
//
// template <N, SimdExt, typename O, typename A0, typename A1>
// struct n0_to_n1< N, SimdExt, n0<O, A0, A1, none_> > {
//   typedef n1<N, SimdExt, O, A0, A1, none_> return_type;
//   return_type doit(n0<O, A0, A1, none_> const &node) {
//     return_type ret;
//     ret.a0 = n0_to_n1<A0>::doit(node.a0);
//     ret.a1 = n0_to_n1<A1>::doit(node.a1);
//     return ret;
//   }
// };
//
// template <N, SimdExt, typename O, typename A0>
// struct n0_to_n1< N, SimdExt, n0<O, A0, none_, none_> > {
//   typedef n1<N, SimdExt, O, A0, none_, none_> return_type;
//   return_type doit(n0<O, A0, none_, none_> const &node) {
//     return_type ret;
//     ret.a0 = n0_to_n1<A0>::doit(node.a0);
//     return ret;
//   }
// };

// TODO: transform views

// ----------------------------------------------------------------------------
// struct containaing informations for eval on how to launch CUDA kernel

template <int N = 1, typename SimdExt = NSIMD_SIMD> struct eval_infos {
  int grid;
  int block;
#ifdef NSIMD_IS_NVCC
  cudaStream_t stream;
#endif
  typedef SimdExt simd_ext;
  static const int unroll = N;
};

// ----------------------------------------------------------------------------
// CUDA kernel for eval, expect n0-tree

#ifdef NSIMD_IS_NVCC
template <typename T, typename Tree>
__kernel__ void eval_kernel(Tree const &tree, T *dst, nat n) {
  nat i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = (T)tree.cuda_eval(i);
  }
}
#endif

// ----------------------------------------------------------------------------
// Eval helper function, expect n1-tree

// template <int N, typename SimdExt, typename Tree, typename T>
// void eval2(Tree const& tree, T *dst, nat n) {
//   // Getting here means that checks on tree have been done and that all is
//   ok
//   // First construct the tree for unroll of N and SIMD extension SimdExt
//   typedef pack<T, N, SimdExt> dst_pack;
//   typedef typename n0_to_n1<N, SimdExt, Tree> tree_cvt;
//   typename tree_cvt::return_type simd_tree = tree_cvt::doit(tree);
//   typedef pack<Tree::return_type, N, SimdExt> tree_pack;
//   int step = len<dst_pack>();
//   int i;
//   for (i = 0; i < n; i += step) {
//     storeu(&dst[i], cvt<tree_pack, dst_pack>(simd_tree.eval(i)));
//   }
//   // Then scalars for the tail
//   eval2<1, cpu, Tree, T>(tree, &dst[i], n - i);
// }

// ----------------------------------------------------------------------------
// Eval function

// template <typename Tree, typename T, typename EvalInfos>
// void eval(Tree const &tree, T *dst, EvalInfos *eval_infos = NULL) {
//   int data_placement = tree.get_data_placement();
//   if (data_placement == on_both) {
//     throw std::invalid_argument("inputs are not at the same place");
//   }
//   nat n = tree.get_n();
//   if (n < 0) {
//     throw std::invalid_argument("inputs have not the same length");
//   }
// #ifdef NSIMD_IS_NVCC
//   if (data_placement == on_device) {
//     if (eval_infos == NULL) {
//       eval_kernel<<<(n + 127) / 128, 128>>>(tree, dst, n);
//     } else {
//       eval_kernel<<<eval_infos->grid, eval_infos->block,
//       eval_infos->stream>>>(
//           tree, dst, n);
//     }
//   } else {
// #endif
//     if (eval_infos == NULL) {
//       eval2<1, NSIMD_SIMD, Tree, T>(tree, T, n);
//     } else {
//       eval2<EvalInfos::unroll, typename EvalInfos::simd_ext, Tree, T>(tree,
//       T,
//                                                                       n);
//     }
// #ifdef NSIMD_IS_NVCC
//   }
// #endif
// }

// ----------------------------------------------------------------------------
// CUDA kernel for eval_if, expect n0-tree

#ifdef NSIMD_IS_NVCC
template <typename T, typename CondTree, typename CompTree>
__kernel__ void eval_if_kernel(CondTree const &cond_tree,
                               CompTree const &comp_tree, T *dst, nat n) {
  nat i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    if (cond_tree.cuda_eval(i)) {
      dst[i] = (T)comp_tree.cuda_eval(i);
    }
  }
}
#endif

// ----------------------------------------------------------------------------
// Eval_if helper function, expect n1-tree

// template <int N, typename SimdExt, typename CondTree, typename CompTree,
//           typename T>
// void eval_if2(CondTree const &cond_tree, CompTree const &comp_tree, T *dst,
//               nat n) {
//   // Getting here means that checks on trees have been done and that all is
//   ok
//
//   typedef pack<T, N, SimdExt> dst_pack;
//
//   typedef typename n0_to_n1<N, SimdExt, CondTree> cond_tree_cvt;
//   typename cond_tree_cvt::return_type simd_cond_tree =
//       cond_tree_cvt::doit(cond_tree);
//   typedef packl<Tree::return_type, N, SimdExt> cond_tree_pack;
//
//   typedef typename n0_to_n1<N, SimdExt, CompTree> comp_tree_cvt;
//   typename comp_tree_cvt::return_type simd_comp_tree =
//       comp_tree_cvt::doit(comp_tree);
//   typedef pack<Tree::return_type, N, SimdExt> comp_tree_pack;
//
//   int step = len<dst_pack>();
//   int i;
//   for (i = 0; i < n; i += step) {
//     storeu(&dst[i],
//            if_else(simd_cond_tree.eval(i),
//                    cvt<comp_tree_pack, dst_pack>(simd_comp_tree.eval(i)),
//                    loadu<dst_pack>(&dst[i])));
//   }
//
//   // Then scalars for the tail
//   eval_if2<1, cpu, CondTree, CompTree, T>(cond_tree, comp_tree, &dst[i], n -
//   i);
// }

// ----------------------------------------------------------------------------
// Eval_if function

// template <typename CondTree, typename CompTree, typename T, typename
// EvalInfos> void eval_if(CondTree const &cond_tree, CompTree const
// &comp_tree, T *dst,
//              EvalInfos *eval_infos = NULL) {
//   int cond_data_placement = cond_tree.get_data_placement();
//   if (cond_data_placement == on_both) {
//     throw std::invalid_argument(
//         "inputs for condition are not at the same place");
//   }
//   nat cond_n = cond_tree.get_n();
//   if (n < 0) {
//     throw std::invalid_argument(
//         "inputs for condition have not the same length");
//   }
//   int comp_data_placement = comp_tree.get_data_placement();
//   if (comp_data_placement == on_both) {
//     throw std::invalid_argument(
//         "inputs for computation are not at the same place");
//   }
//   nat comp_n = comp_tree.get_n();
//   if (n < 0) {
//     throw std::invalid_argument(
//         "inputs for computation have not the same length");
//   }
//   if (cond_data_placement != comp_data_placement) {
//     throw std::invalid_argument("inputs for condition and inputs for "
//                                 "computation have not the same length");
//   }
//   if (cond_n != comp_n) {
//     throw std::invalid_argument("inputs for condition and inputs for "
//                                 "computation have not the same length");
//   }
// #ifdef NSIMD_IS_NVCC
//   if (cond_data_placement == on_device) {
//     if (eval_infos == NULL) {
//       eval_if_kernel<<<(n + 127) / 128, 128>>>(cond_tree, comp_tree, dst,
//       n);
//     } else {
//       eval_if_kernel<<<eval_infos->grid, eval_infos->block,
//                        eval_infos->stream>>>(cond_tree, comp_tree, dst, n);
//     }
//   } else {
// #endif
//     if (eval_infos == NULL) {
//       eval_if2<1, NSIMD_SIMD, CondTree, CompTree, T>(cond_tree, comp_tree,
//       T,
//                                                      n);
//     } else {
//       eval_if2<EvalInfos::unroll, typename EvalInfos::simd_ext, CondTree,
//                CompTree, T>(cond_tree, comp_tree, T, n);
//     }
// #ifdef NSIMD_IS_NVCC
//   }
// #endif
// }

// ----------------------------------------------------------------------------

} // namespace nsimd

#endif
