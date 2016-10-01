#ifndef CONCURRENT_MATH_HPP
#define CONCURRENT_MATH_HPP

#include "concurrent/common.hpp"
#ifdef __APPLE__
  #include <Accelerate/Accelerate.h>
#elif __linux__
  #include <cblas.h>
#endif

namespace concurrent {

template <typename Dtype>
void cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void rnn_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

template <typename Dtype>
void rnn_copy(const int N, const Dtype *X, Dtype *Y);

inline void rnn_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(librnn/alt_fn)
}

template <typename Dtype>
Dtype rnn_cpu_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
Dtype rnn_cpu_strided_dot(const int n, const Dtype* x, const int incx,
    const Dtype* y, const int incy);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype rnn_cpu_asum(const int n, const Dtype* x);

} // namespace concurrent

#endif // CONCURRENT_MATH_HPP
