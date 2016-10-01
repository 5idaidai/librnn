#ifndef CONCURRENT_OPS_OTHER_HPP
#define CONCURRENT_OPS_OTHER_HPP

#include "concurrent/common.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/var.hpp"

namespace concurrent {
template <typename R> class var;

namespace ops {

template <typename T>
struct other {

  static void grad(var<T>* other);
  static void grad(var<T>* other, T value);

  // http://arunmallya.github.io/writeups/nn/backprop.html
  static var<T> softmax(var<T>& input);

};

}  // namespace ops
}  // namespace concurrent

#endif  // CONCURRENT_OPS_OTHER_HPP
