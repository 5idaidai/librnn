#ifndef CONCURRENT_OPS_ELTWISE_HPP
#define CONCURRENT_OPS_ELTWISE_HPP

#include <cmath>

#include "concurrent/common.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/var.hpp"

namespace concurrent {
template <typename R> class var;

namespace ops {

template <typename T>
struct eltwise {
  
  static var<T>       add(const var<T>& input, T alpha);
  static var<T>    eltmul(const var<T>& input, T alpha);
  static var<T> eltdivide(const var<T>& input, T alpha);
  static var<T>    square(const var<T>& input);
  static var<T>       exp(const var<T>& input);
  static var<T>   elt_inv(const var<T>& input);

  // static var<T>       pow(const var<T>& input, T alpha);
  // static var<T>    eltmax(const var<T>& input, T alpha);
  // static var<T>    square(const var<T>& input);
  // static var<T>       log(const var<T>& input);
  // static var<T>       abs(const var<T>& input);
  // static var<T>       pow(const var<T>& input, T alpha);
  // static var<T>      sqrt(const var<T>& input);

};

}  // namespace ops
}  // namespace concurrent

#endif  // CONCURRENT_OPS_ELTWISE_HPP
