#ifndef CONCURRENT_OPS_HPP
#define CONCURRENT_OPS_HPP

#include "concurrent/common.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/var.hpp"

#include "concurrent/operations/activation.hpp"
#include "concurrent/operations/binary.hpp"
#include "concurrent/operations/eltwise.hpp"
#include "concurrent/operations/other.hpp"
#include "concurrent/operations/solver.hpp"

namespace concurrent {
// namespace ops {

template <typename R> class var;

namespace ops {
  template <typename R> struct activation;
  template <typename R> struct binary;
  template <typename R> struct eltwise;
  template <typename R> struct other;
  template <typename R> struct solver;
}

template <typename R>
struct operation :  ops::activation<R>,
                    ops::binary<R>,
                    ops::eltwise<R>,
                    ops::other<R>,
                    ops::solver<R> {
  
  static var<R>  add(const var<R>& x, const var<R>& y) { return ops::binary<R>::add(x, y); }
  static var<R>  mul(const var<R>& x, const var<R>& y) { return ops::binary<R>::eltmul(x, y); }
  static var<R>  sub(const var<R>& x, const var<R>& y) { return ops::binary<R>::sub(x, y); }
  static var<R>  div(const var<R>& x, const var<R>& y) { return ops::binary<R>::eltdivide(x, y); }

  static var<R>  add(const var<R>& x, R alpha) { return ops::binary<R>::add(x, alpha); }
  static var<R>  mul(const var<R>& x, R alpha) { return ops::binary<R>::eltmul(x, alpha); }
  static var<R>  sub(const var<R>& x, R alpha) { return ops::binary<R>::sub(x, alpha); }
  static var<R>  div(const var<R>& x, R alpha) { return ops::binary<R>::eltdivide(x, alpha); }

};

// }  // namespace ops
}  // namespace concurrent

#endif  // CONCURRENT_OPS_HPP
