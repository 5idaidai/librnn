#include "concurrent/math.hpp"
#include "concurrent/common.hpp"

namespace concurrent {

template<>
void cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  // int lda = M;
  // int ldb = K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template<>
void cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
}

template <>
void rnn_axpy<float>(const int N, const float alpha, const float* X, float* Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template <>
void rnn_axpy<double>(const int N, const double alpha, const double* X, double* Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

template <typename Dtype>
void rnn_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (concurrent::librnn::mode() == concurrent::librnn::GPU) {
// #ifndef CPU_ONLY
//       // NOLINT_NEXT_LINE(librnn/alt_fn)
//       CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
// #else
//       NO_GPU;
// #endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(librnn/alt_fn)
    }
  }
}

template <>
float rnn_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double rnn_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
float rnn_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double rnn_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype rnn_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return rnn_cpu_strided_dot(n, x, 1, y, 1);
}

template
float rnn_cpu_dot<float>(const int n, const float* x, const float* y);

template
double rnn_cpu_dot<double>(const int n, const double* x, const double* y);

template void rnn_copy<int>(const int N, const int* X, int* Y);
template void rnn_copy<unsigned int>(const int N, const unsigned int* X, unsigned int* Y);
template void rnn_copy<float>(const int N, const float* X, float* Y);
template void rnn_copy<double>(const int N, const double* X, double* Y);

} // namespace concurrent
