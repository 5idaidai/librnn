#include "concurrent/operations/other.hpp"

namespace concurrent {
namespace ops {

template <typename T>
void other<T>::grad(var<T>* input) {
  T* input_diff = input->mutable_cpu_diff();
  for (int i = 0; i < input->count(); i++) {
    input_diff[i] = 1.0;
  }
}

template <typename T>
void other<T>::grad(var<T>* input, T value) {
  T* input_diff = input->mutable_cpu_diff();
  for (int i = 0; i < input->count(); i++) {
    input_diff[i] += value;
  }
}

template <typename T>
var<T> other<T>::softmax(var<T>& input) {
  auto output = var<T>::zeros_like(input);
  T sum = 0.0;
  const T* input_data = input.cpu_data();
  for (int i = 0; i < input.count(); ++i) {
    sum += std::exp(input_data[i]);
  }

  T* output_data = output.mutable_cpu_data();
  for (int i = 0; i < input.count(); ++i) {
    output_data[i] = std::exp(input_data[i])/sum;
  }

  return output;
}

template struct other<float>;
template struct other<double>;

}  // namespace ops
}  // namespace concurrent
