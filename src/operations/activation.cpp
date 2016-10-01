#include <cmath>

#include "concurrent/operations/activation.hpp"

namespace concurrent {
namespace ops {

template <typename T>
var<T> activation<T>::hard_sigmoid(const var<T>& input) {
  auto output = var<T>::zeros_like(input);
  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (int i = 0; i < output.count(); i++) {
    output_data[i] = std::max(0.0, std::min(1.0, 0.2 * input_data[i] + 0.5));
  }
  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output]() mutable {
      T sig_x;
      T* input_diff = input.mutable_cpu_diff();
      const T* output_data = output.cpu_data();
      const T* output_diff = output.cpu_diff();
      const int count = output.count();

      for (auto i = 0; i < input.count(); i++) {
        if (output_data[i] == 1 || output_data[i] == 0) {
          input_diff[i] += 0;
        } else {
          input_diff[i] += 0.2;
        }
      }

    });
  }
  return output;
}

template <typename T>
var<T> activation<T>::hard_tanh(const var<T>& input) {
  auto output = var<T>::zeros_like(input);
  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (int i = 0; i < output.count(); i++) {
    if (input_data[i] < -1) {
      output_data[i] = -1;
    } else if (input_data[i] > 1) {
      output_data[i] = 1;
    } else {
      output_data[i] = input_data[i];
    }
  }
  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output]() mutable {
      T sig_x;
      T* input_diff = input.mutable_cpu_diff();
      const T* output_data = output.cpu_data();
      const T* output_diff = output.cpu_diff();
      const int count = output.count();

      for (auto i = 0; i < input.count(); i++) {
        if (output_data[i] <= 1 && output_data[i] >= -1) {
          input_diff[i] += 1;
        } else {
          input_diff[i] += 0;
        }
      }

    });
  }
  return output;
}

template <typename T>
var<T> activation<T>::sigmoid(const var<T>& input) {
  auto output = var<T>::zeros_like(input);
  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (int i = 0; i < output.count(); i++) {
    output_data[i] = 1.0 / (1.0 + std::exp(-input_data[i]));
  }
  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output]() mutable {
      T sig_x;
      T* input_diff = input.mutable_cpu_diff();
      const T* output_data = output.cpu_data();
      const T* output_diff = output.cpu_diff();
      const int count = output.count();

      for (auto i = 0; i < count; i++) {
        sig_x = output_data[i];
        input_diff[i] += output_diff[i] * sig_x * (1 - sig_x);
      }
    });
  }
  return output;
}

template <typename T>
var<T> activation<T>::tanh(const var<T>& input) {
  auto output = var<T>::zeros_like(input);
  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (int i = 0; i < output.count(); i++) {
    output_data[i] = (2.0 / (1.0 + std::exp(-2 * input_data[i]))) - 1.0;
  }
  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output]() mutable {
      T tanh_x;
      T* input_diff = input.mutable_cpu_diff();
      const T* output_data = output.cpu_data();
      const T* output_diff = output.cpu_diff();
      for (auto i = 0; i < input.count(); i++) {
        tanh_x = output_data[i];
        input_diff[i] += output_diff[i] * (1 - tanh_x * tanh_x);
      }
    });
  }
  return output;
}

template <typename T>
var<T> activation<T>::relu(const var<T>& input) {
  auto output = var<T>::zeros_like(input);
  const T* input_data = input.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (int i = 0; i < input.count(); i++) {
    // Fast implementation for GPUs from:
    // https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/nnet.py#L2223
    output_data[i] = 0.5 * (input_data[i] + std::abs(input_data[i]));
  }
  if (graph::backprop_enabled()) {
    graph::emplace_back([input, output]() mutable {
      T* input_diff = input.mutable_cpu_diff();
      const T* output_data = output.cpu_data();
      const T* output_diff = output.cpu_diff();
      for (auto i = 0; i < input.count(); i++) {
        if (output_data[i] >= 0) {
          input_diff[i] += output_diff[i];
        } else {
          input_diff[i] += 0.0;
        }
      }
    });
  }
  return output;
}

template <typename T>
var<T> activation<T>::relu6(const var<T>& input) {
  auto output = var<T>::zeros_like(input);
  for (int i = 0; i < input.count(); i++) {
    // output.blob[i] = std::min(0.5 * (input.blob[i] + std::abs(input.blob[i])), 6.0);
  }
  if (graph::backprop_enabled()) {
    graph::emplace_back([&input, &output]() mutable {
      for (auto i = 0; i < input.count(); i++) {
        // if (output.blob[i] <= 0) {
        //   input.diff_[i] += 0.0 * output.diff_[i];
        // } else if (output.blob[i] >= 6) {
        //   input.diff_[i] += 0.0 * output.diff_[i];
        // } else {
        //   input.diff_[i] += 1.0 * output.diff_[i];
        // }
      }
    });
  }
  return output;
}

// Todo: faster implementation like relu
template <typename T>
var<T> activation<T>::prelu(const var<T>& input, float alpha) {
  auto output = var<T>::zeros_like(input);
  for (int i = 0; i < input.count(); i++) {
    // if (input.blob[i] >= 0) {
    //   output.blob[i] = input.blob[i];
    // } else {
    //   output.blob[i] = input.blob[i] * alpha;
    // }
  }
  if (graph::backprop_enabled()) {
    graph::emplace_back([&input, &output, alpha]() mutable {
      for (auto i = 0; i < input.count(); i++) {
        // if (output.blob[i] >= 0) {
        //   input.diff_[i] += 1.0 * output.diff_[i];
        // } else {
        //   input.diff_[i] += alpha * output.diff_[i];
        // }
      }
    });
  }
  return output;
}

// Todo: faster implementation like relu
template <typename T>
var<T> activation<T>::elu(const var<T>& input, float alpha) {
  auto output = var<T>::zeros_like(input);
  for (int i = 0; i < input.count(); i++) {
    // if (input.blob[i] >= 0) {
    //   output.blob[i] = input.blob[i];
    // } else {
    //   output.blob[i] = alpha * (std::exp(input.blob[i]) - 1);
    // }
  }
  if (graph::backprop_enabled()) {
    graph::emplace_back([&input, &output, alpha]() mutable {
      for (auto i = 0; i < input.count(); i++) {
        // if (output.blob[i] >= 0) {
        //   input.diff_[i] += output.diff_[i];
        // } else {
        //   input.diff_[i] += output.diff_[i] * (output.blob[i] + alpha);
        // }
      }
    });
  }
  return output;
}

template <typename T>
var<T> activation<T>::softplus(const var<T>& input) {
  auto output = var<T>::zeros_like(input);
  for (int i = 0; i < input.count(); i++) {
    // output.blob[i] = std::log1p(std::exp(input.blob[i]));
  }
  if (graph::backprop_enabled()) {
    graph::emplace_back([&input, &output]() mutable {
      for (auto i = 0; i < input.count(); i++) {
        // input.diff_[i] += (1 / (1 + std::exp(-input.blob[i]))) * output.diff_[i];
      }
    });
  }
  return output;
}

template <typename T>
var<T> activation<T>::softsign(const var<T>& input) {
  auto output = var<T>::zeros_like(input);
  for (int i = 0; i < input.count(); i++) {
    // output.blob[i] = input.blob[i] / (1 + std::abs(input.blob[i]));
  }
  if (graph::backprop_enabled()) {
    graph::emplace_back([&input, &output]() mutable {
      for (auto i = 0; i < input.count(); i++) {
        // input.diff_[i] += (1 / (1 + pow(std::abs(input.blob[i]), 2))) * output.diff_[i];
      }
    });
  }
  return output;
}

template struct activation<float>;
template struct activation<double>;

}  // namespace ops
}  // namespace concurrent
