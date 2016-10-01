#include "concurrent/operations/solver.hpp"

namespace concurrent {
namespace ops {

template <typename T>
void solver<T>::clip_and_regularize(var<T> param, T clip_abs, T clip_norm, T regc) {
  NOT_IMPLEMENTED
}

template <typename T>
void solver<T>::regularize(var<T> param, T regc) {
  NOT_IMPLEMENTED
}

template <typename T>
void solver<T>::normalize(var<T> param, T norm_threshold) {
  NOT_IMPLEMENTED
}

template <typename T>
void solver<T>::sgd_update(var<T>& param, T step_size) {
  for (int i = 0; i < param.count(); ++i) {
    T* param_data = param.mutable_cpu_data();
    T* param_diff = param.mutable_cpu_diff();
    param_data[i] -= step_size * param_diff[i];
  }
}

template <typename T>
void solver<T>::adagrad_update(var<T>& param, var<T>& cache, T step_size, T smooth_eps) {
  // CLIP [-5, 5]
  // clip the gradient to prevent explosions:
  float clip_amount = 5;
  T* param_diff = param.mutable_cpu_diff();
  for (int i=0; i<param.count(); ++i) {
    if (param_diff[i] < -clip_amount) {
      param_diff[i] = -clip_amount;
    } else if (param_diff[i] > clip_amount) {
      param_diff[i] = clip_amount;
    } else {
    }
  }

  T* mem_data = cache.mutable_cpu_data();
  for (int i = 0; i < cache.count(); ++i) {
    mem_data[i] += param_diff[i] * param_diff[i];
  }

  // update gradient using RMSprop rule
  T* param_data = param.mutable_cpu_data();
  for (int i = 0; i < param.count(); ++i) {
    param_data[i] += (-step_size * param_diff[i]) / std::sqrt(mem_data[i] + smooth_eps);
  }
}

template <typename T>
void solver<T>::rmsprop_update(var<T> param, var<T>& cache, T decay_rate, T step_size, T smooth_eps) {
  NOT_IMPLEMENTED
}

template <typename T>
void solver<T>::rmsprop_momentum_update(var<T> param, var<T>& n_cache, var<T>& g_cache, var<T>& momentum_cache, T decay_rate, T momentum, T step_size, T smooth_eps) {
  NOT_IMPLEMENTED
}

template <typename T>
void solver<T>::adadelta_update(var<T> param, var<T>& gsum, var<T>& xsum, T rho, T smooth_eps) {
  NOT_IMPLEMENTED
}

template <typename T>
void solver<T>::adam_update(var<T> param, var<T>& m, var<T>& v, T b1, T b2, T smooth_eps, T step_size, unsigned long long epoch) {
  NOT_IMPLEMENTED
}

template struct solver<float>;
template struct solver<double>;

}  // namespace ops
}  // namespace concurrent
