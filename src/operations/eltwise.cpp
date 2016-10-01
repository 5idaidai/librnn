#include "concurrent/operations/eltwise.hpp"

namespace concurrent {
namespace ops {

template <typename T>
var<T> eltwise<T>::add(const var<T>& input, T alpha) {
  auto output = var<T>::zeros_like(input);
  
  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (auto i = 0; i < output.count(); i++) {
    output_data[i] = input_data[i] + alpha;
  }

  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output]() mutable {
      const T* output_diff = output.cpu_diff();
      T* input_diff = input.mutable_cpu_diff();
      for (auto i = 0; i < input.count(); i++) {
        input_diff[i] += output_diff[i];
      }
    });
  }
  return output;
}

template <typename T>
var<T> eltwise<T>::square(const var<T>& input) {
  auto output = var<T>::zeros_like(input);

  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (auto i = 0; i < input.count(); i++) {
    output_data[i] = std::pow(input_data[i], 2);
  }
  
  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output]() mutable {
      const T* output_diff = output.cpu_diff();
      const T* input_data = input.cpu_data();
      T* input_diff = input.mutable_cpu_diff();
      for (auto i = 0; i < input.count(); i++) {
        input_diff[i] += output_diff[i] * input_data[i] * static_cast<T>(2.0);
      }
    });
  }
  return output;
}

template <typename T>
var<T> eltwise<T>::eltmul(const var<T>& input, T alpha) {
  auto output = var<T>::zeros_like(input);

  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (auto i = 0; i < output.count(); i++) {
    output_data[i] = input_data[i] * alpha;
  }

  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output, alpha]() mutable {
      const T* output_diff = output.cpu_diff();
      T* input_diff = input.mutable_cpu_diff();
      for (auto i = 0; i < input.count(); i++) {
        input_diff[i] += alpha * output_diff[i];
      }
    });
  }

  return output;
}

template <typename T>
var<T> eltwise<T>::eltdivide(const var<T>& input, T alpha) {
  auto output =  var<T>::zeros_like(input);

  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (auto i = 0; i < output.count(); i++) {
    output_data[i] = input_data[i] / alpha;
  }

  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output, alpha]() mutable {
      T* input_diff = input.mutable_cpu_diff();
      const T* output_diff = output.cpu_diff();
      for (auto i = 0; i < input.count(); i++) {
        input_diff[i] += (1.0 / alpha) * output_diff[i];
      }
    });
  }
  return output;
}

template <typename T>
var<T> eltwise<T>::elt_inv(const var<T>& input) {
  auto output =  var<T>::zeros_like(input);

  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (auto i = 0; i < input.count(); i++) {
    output_data[i] = static_cast<T>(1.0) / input_data[i];
  }

  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output]() mutable {
      const T* output_diff = output.cpu_diff();
      const T* output_data = output.cpu_data();
      T* input_diff = input.mutable_cpu_diff();
      for (auto i = 0; i < input.count(); i++) {
        input_diff[i] -= std::pow(output_data[i], 2) * output_diff[i];
      }
    });
  }

  return output;
}

template <typename T>
var<T> eltwise<T>::exp(const var<T>& input) {
  auto output =  var<T>::zeros_like(input);

  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (auto i = 0; i < input.count(); i++) {
    output_data[i] = std::exp(input_data[i]);
  }

  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output]() mutable {
      for (auto i = 0; i < input.count(); i++) {
        T* input_diff = input.mutable_cpu_diff();
        const T* output_data = output.cpu_data();
        const T* output_diff = output.cpu_diff();
        input_diff[i] += output_data[i] * output_diff[i];
      }
    });
  }
  return output;
}

template struct eltwise<float>;
template struct eltwise<double>;

}  // namespace ops
}  // namespace concurrent
