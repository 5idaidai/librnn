#ifndef CONCURRENT_OPS_BINARY_HPP
#define CONCURRENT_OPS_BINARY_HPP

#include <cmath>

#include "concurrent/common.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/math.hpp"
#include "concurrent/var.hpp"

namespace concurrent {
template <typename R> class var;

namespace ops {

template <typename T>
struct binary {
  static var<T>       add(const var<T>& m1, const var<T>& m2);
  static var<T>    eltmul(const var<T>& m1, const var<T>& m2);
  static var<T>       sub(const var<T>& m1, const var<T>& m2);
  static var<T> eltdivide(const var<T>& m1, const var<T>& m2);
  static var<T>    matmul(const var<T>& A,  const var<T>& B, bool transposeA = false, bool transposeB = false);
};

}  // namespace ops
}  // namespace concurrent

#endif  // CONCURRENT_OPS_BINARY_HPP
