#ifndef CONCURRENT_OPS_POOLING_HPP
#define CONCURRENT_OPS_POOLING_HPP

#include <cmath>

#include "concurrent/common.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/var.hpp"

namespace concurrent {
template <typename R> class var;

namespace ops {

template <typename T>
struct pooling {

  static var<T> avg_pool(const var<T>& input, int window_h, int window_w, int stride_h, int stride_w);
  static var<T> avg_pool(const var<T>& input, int window_size, int stride) {
    return avg_pool(input, window_size, window_size, stride, stride);
  }
  
  static var<T> max_pool(const var<T>& input, int window_h, int window_w, int stride_h, int stride_w);
  static var<T> max_pool(const var<T>& input, int window_size, int stride) {
    return max_pool(input, window_size, window_size, stride, stride);
  }

};

}  // namespace ops
}  // namespace concurrent

#endif  // CONCURRENT_OPS_POOLING_HPP
