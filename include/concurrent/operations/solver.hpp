#ifndef CONCURRENT_OPS_SOLVER_HPP
#define CONCURRENT_OPS_SOLVER_HPP

#include "concurrent/common.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/math.hpp"
#include "concurrent/var.hpp"

namespace concurrent {
template <typename R> class var;

namespace ops {

template <typename T>
struct solver {

  static void     clip_and_regularize(var<T> param, T clip_abs, T clip_norm, T regc);
  static void              regularize(var<T> param, T regc);
  static void               normalize(var<T> param, T norm_threshold);

  static void              sgd_update(var<T>& param, T step_size);
  static void          adagrad_update(var<T>& param, var<T>& cache, T step_size, T smooth_eps);
  
  static void          rmsprop_update(var<T> param, var<T>& cache, T decay_rate, T step_size, T smooth_eps);
  static void rmsprop_momentum_update(var<T> param, var<T>& n_cache, var<T>& g_cache, var<T>& momentum_cache, T decay_rate, T momentum, T step_size, T smooth_eps);
  static void         adadelta_update(var<T> param, var<T>& gsum, var<T>& xsum, T rho, T smooth_eps);
  static void             adam_update(var<T> param, var<T>& m, var<T>& v, T b1, T b2, T smooth_eps, T step_size, unsigned long long epoch);

};

}  // namespace ops
}  // namespace concurrent

#endif  // CONCURRENT_OPS_SOLVER_HPP
