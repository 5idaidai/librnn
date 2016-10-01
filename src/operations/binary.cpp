#include "concurrent/operations/binary.hpp"

namespace concurrent {
namespace ops {

template <typename T>
var<T> binary<T>::add(const var<T>& m1, const var<T>& m2) {
  auto output = var<T>::zeros_like(m1);
  
  const T* m1_data = m1.cpu_data();
  const T* m2_data = m2.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (auto i = 0; i < output.count(); i++) {
    output_data[i] = m1_data[i] + m2_data[i];
  }

  if (graph::backprop_enabled()) {
    graph::emplace_back([m1, m2, output]() mutable {
      T* m1_diff = m1.mutable_cpu_diff();
      T* m2_diff = m2.mutable_cpu_diff();
      const T* output_diff = output.cpu_diff();
      for (auto i = 0; i < output.count(); i++) {
        m1_diff[i] += output_diff[i];
        m2_diff[i] += output_diff[i];
      }
    });
  }
  return output;
}

template <typename T>
var<T> binary<T>::eltmul(const var<T>& m1, const var<T>& m2) {
  auto output = var<T>::zeros_like(m1);
  
  const T* m1_data = m1.cpu_data();
  const T* m2_data = m2.cpu_data();
  T* output_data = output.mutable_cpu_data();
  const int count = output.count();
  for (auto i = 0; i < output.count(); i++) {
    output_data[i] = m1_data[i] * m2_data[i];
  }

  if (graph::backprop_enabled()) {
    graph::emplace_back([m1, m2, output]() mutable {
      T* m1_diff = m1.mutable_cpu_diff();
      T* m2_diff = m2.mutable_cpu_diff();
      const T* m1_data = m1.cpu_data();
      const T* m2_data = m2.cpu_data();
      const T* output_diff = output.cpu_diff();

      for (auto i = 0; i < output.count(); i++) {
        m1_diff[i] += m2_data[i] * output_diff[i];
        m2_diff[i] += m1_data[i] * output_diff[i];
      }
    });
  }
  return output;
}

template <typename T>
var<T> binary<T>::sub(const var<T>& m1, const var<T>& m2) {
  auto output = var<T>::zeros_like(m1);
  
  const T* m1_data = m1.cpu_data();
  const T* m2_data = m2.cpu_data();
  T* output_data = output.mutable_cpu_data();
  for (auto i = 0; i < output.count(); i++) {
    output_data[i] = m1_data[i] - m2_data[i];
  }

  if (graph::backprop_enabled()) {
    graph::emplace_back([m1, m2, output]() mutable {
      T* m1_diff = m1.mutable_cpu_diff();
      T* m2_diff = m2.mutable_cpu_diff();
      const T* output_diff = output.cpu_diff();
      
      for (auto i = 0; i < output.count(); i++) {
        m1_diff[i] += output_diff[i];
        m2_diff[i] -= output_diff[i];
      }
    });
  }
  return output;
}

template <typename T>
var<T> binary<T>::eltdivide(const var<T>& m1, const var<T>& m2) {
  auto output = var<T>::zeros_like(m1);
  
  const T* m1_data = m1.cpu_data();
  const T* m2_data = m2.cpu_data();
  T* output_data = output.mutable_cpu_data();
  const int count = output.count();
  for (auto i = 0; i < output.count(); i++) {
    output_data[i] = m1_data[i] / m2_data[i];
  }

  if (graph::backprop_enabled()) {
    graph::emplace_back([m1, m2, output]() mutable {
      T* m1_diff = m1.mutable_cpu_diff();
      T* m2_diff = m2.mutable_cpu_diff();
      const T* m1_data = m1.cpu_data();
      const T* m2_data = m2.cpu_data();
      const T* output_diff = output.cpu_diff();

      for (auto i = 0; i < output.count(); i++) {
        m1_diff[i] += output_diff[i] / m2_data[i];
        m2_diff[i] -= output_diff[i] * (m1_data[i] * std::pow(output_diff[i], 2));
      }
    });
  }
  return output;
}

template <typename T>
var<T> binary<T>::matmul(const var<T>& A, const var<T>& B, bool transposeA, bool transposeB) {
  // forward pass
  // A = M x K
  // B = K x N
  // C = M x N

  // M = number of rows         in matrices A and C
  // K = number of columns/rows in matrices A/B
  // N = number of columns      in matrices B and C

  int M = A.height();
  int N = B.width();
  int K = A.width();

  // if (A.width() != B.height()) { throw; }

  if (transposeA) {
    M = A.width();
    K = A.height();
  }

  if (transposeB) {
    N = B.height();
  }

  auto output = var<T>(1, 1, M, N);

  int lda = (transposeA) ? K : M;
  int ldb = (transposeB) ? N : K;
  
  const T* a_data = A.cpu_data();
  const T* b_data = B.cpu_data();
  T* c_data = output.mutable_cpu_data();

  const T alpha = 1.0;
  const T beta = 1.0;
  if (transposeA && transposeB) {
    cpu_gemm<T>(CblasTrans, CblasTrans, M, N, K, alpha, a_data, b_data, beta, c_data);
  } else if (transposeA && !transposeB) {
    cpu_gemm<T>(CblasTrans, CblasNoTrans, M, N, K, alpha, a_data, b_data, beta, c_data);
  } else if (!transposeA && transposeB) {
    cpu_gemm<T>(CblasNoTrans, CblasTrans, M, N, K, alpha, a_data, b_data, beta, c_data);
  } else {
    cpu_gemm<T>(CblasNoTrans, CblasNoTrans, M, N, K, alpha, a_data, b_data, beta, c_data);
  }

  if (graph::backprop_enabled()) {
    // dD = np.random.randn(*D.shape) # same shape as D
    // dA = dC.dot(B.T) 
    // dB = A.T.dot(dC)
    graph::emplace_back([A, B, output]() mutable {
      T* A_diff = A.mutable_cpu_diff();
      T* B_diff = B.mutable_cpu_diff();
      const T* A_data = A.cpu_data();
      const T* B_data = B.cpu_data();
      const T* output_diff = output.cpu_diff();
      // const int count = A.count();
      const T alpha = 1.0;
      const T beta = 1.0;


      int M = output.height();
      int K = B.width();
      int N = B.height();
      cpu_gemm<T>(CblasNoTrans, CblasTrans, M, N, K, alpha, output_diff, B_data, beta, A_diff);

      M = A.width();
      K = output.height();
      N = output.width();
      cpu_gemm<T>(CblasTrans, CblasNoTrans, M, N, K, alpha, A_data, output_diff, beta, B_diff);
    });
  }
  return output;
}

template struct binary<float>;
template struct binary<double>;

}  // namespace ops
}  // namespace concurrent
