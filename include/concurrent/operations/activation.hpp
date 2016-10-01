#ifndef CONCURRENT_OPS_ACTIVATION_HPP
#define CONCURRENT_OPS_ACTIVATION_HPP

#include <cmath>

#include "concurrent/common.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/var.hpp"

namespace concurrent {
template <typename R> class var;

namespace ops {

template <typename T>
struct activation {

  // https://www.reddit.com/r/MachineLearning/comments/4wr58u/activation_functions/
  static var<T> hard_sigmoid(const var<T>& input);
  static var<T>    hard_tanh(const var<T>& input);

  static var<T>  sigmoid(const var<T>& input);
  static var<T>     tanh(const var<T>& input);

  static var<T>     relu(const var<T>& input);
  static var<T>    relu6(const var<T>& input);
  static var<T>    prelu(const var<T>& input, float alpha); // http://arxiv.org/abs/1502.01852
  static var<T>      elu(const var<T>& input, float alpha); // http://arxiv.org/abs/1511.07289
  static var<T> softplus(const var<T>& input);
  static var<T> softsign(const var<T>& input);
  static var<T>   maxout(const var<T>& input);

};

}  // namespace ops
}  // namespace concurrent

#endif  // CONCURRENT_OPS_ACTIVATION_HPP
