#include "concurrent/operations/pooling.hpp"

namespace concurrent {
namespace ops {

template <typename T>
var<T> pooling<T>::max_pool(const var<T>& input, int window_h, int window_w, int stride_h, int stride_w) {
  NOT_IMPLEMENTED
}

template <typename T>
var<T> pooling<T>::avg_pool(const var<T>& input, int window_h, int window_w, int stride_h, int stride_w) {
  NOT_IMPLEMENTED
}

template struct pooling<float>;
template struct pooling<double>;

}  // namespace ops
}  // namespace concurrent
