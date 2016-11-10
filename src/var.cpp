#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "concurrent/var.hpp"

namespace concurrent {

template <typename Real>
void var<Real>::reshape(const int num, const int channels, const int height, const int width) {
  std::vector<int> shape(4);
  shape[0] = num;
  shape[1] = channels;
  shape[2] = height;
  shape[3] = width;
  reshape(shape);
}

template <typename Real>
void var<Real>::reshape(const std::vector<int>& shape) {
  count_ = 1;
  shape_.resize(shape.size());

  // TODO: add check if shape param is zero
  for (int i = 0; i < shape.size(); ++i) {
    count_ *= shape[i];
    shape_[i] = shape[i];
  }

  synced_data_ = std::make_shared<concurrent::synced_memory>(count_ * sizeof(Real));
  synced_diff_ = std::make_shared<concurrent::synced_memory>(count_ * sizeof(Real));
}

template <typename Real>
var<Real>::var(const int num, const int channels, const int height, const int width) {
  reshape(num, channels, height, width);
}

template <typename Real>
var<Real>::var(Real num) : var({1,1,1,1}, concurrent::initializer::constant<Real>(num)) {}

template <typename Real>
var<Real>::var(const std::vector<int>& shape) {
  reshape(shape);
}

// destructor
template <typename Real>
var<Real>::~var() {}

// // copy constructor
// template <typename Real>
// var<Real>::var(const var<Real>& other) {
//   count_       = other.count_;
//   shape_       = other.shape_;
//   synced_data_ = other.synced_data_;
//   synced_diff_ = other.synced_diff_;
// }

// // copy assignment
// template <typename Real>
// var<Real>& var<Real>::operator=(const var<Real>& other) {
//   if (&other == this) { return *this; } // check for self-assignment
//   count_       = other.count_;
//   shape_       = other.shape_;
//   synced_data_ = other.synced_data_;
//   synced_diff_ = other.synced_diff_;
//   return *this;
// }

// // move constructor
// template <typename Real>
// var<Real>::var(var<Real>&& other) {
//   shape_       = other.shape_;
//   count_       = other.count_;
//   synced_data_ = other.synced_data_;
//   synced_diff_ = other.synced_diff_;
// }

// // move assignment
// template <typename Real>
// var<Real>& var<Real>::operator=(var<Real>&& other) {
//   count_       = other.count_;
//   shape_       = other.shape_;
//   synced_data_ = other.synced_data_;
//   synced_diff_ = other.synced_diff_;
//   return *this;
// }

template <typename Real>
Real& var<Real>::operator[](const int index) {
  // TODO: should return row slice
  // currently does same thing as `at(index, false)`
  Real* data_blob = this->mutable_cpu_data();
  return data_blob[index];
}

template <typename Real>
Real& var<Real>::at(const int index, bool diff) {
  Real* data_blob = diff ? this->mutable_cpu_diff() : this->mutable_cpu_data();
  return data_blob[index];
}

template <typename Real>
Real var<Real>::diff_at(const int index) {
  const Real* diff_blob = this->cpu_diff();
  return diff_blob[index];
}

template <typename Real>
var<Real> var<Real>::operator-() {
  return (*this) * static_cast<Real>(-1.0);
}

template <typename Real>
var<Real> var<Real>::operator%(const var<Real>& rhs) const {
  return concurrent::operation<Real>::matmul(*this, rhs);
}

template <typename Real>
var<Real> var<Real>::operator+(const var<Real>& rhs) const {
  return concurrent::operation<Real>::add(*this, rhs);
}

template <typename Real>
var<Real> var<Real>::operator*(const var<Real>& rhs) const {
  return concurrent::operation<Real>::mul(*this, rhs);
}

template <typename Real>
var<Real> var<Real>::operator-(const var<Real>& rhs) const {
  return concurrent::operation<Real>::sub(*this, rhs);
}

template <typename Real>
var<Real> var<Real>::operator/(const var<Real>& rhs) const {
  return concurrent::operation<Real>::div(*this, rhs);
}

template <typename Real>
var<Real> var<Real>::operator+(Real alpha) const {
  return concurrent::operation<Real>::add(*this, alpha);
}

template <typename Real>
var<Real> var<Real>::operator*(Real alpha) const {
  return concurrent::operation<Real>::mul(*this, alpha);
}

// template <typename Real>
// var<Real> var<Real>::operator-(Real alpha) const {
//   return concurrent::operation<Real>::sub(*this, alpha);
// }

// template <typename Real>
// var<Real> var<Real>::operator/(Real alpha) const {
//   return concurrent::operation<Real>::div(*this, alpha);
// }

// template <typename Real>
// var<Real>& var<Real>::operator+=(var<Real>& rhs) {
//   Real* this_data = this->mutable_cpu_data();
//   const Real* rhs_data = rhs.cpu_data();
//   for (int i = 0; i < this->count(); ++i) {
//     this_data[i] += rhs_data[i];
//   }
// }

// template <typename Real>
// var<Real>& var<Real>::operator*=(var<Real>& rhs) {
//   Real* this_data = this->mutable_cpu_data();
//   const Real* rhs_data = rhs.cpu_data();
//   for (int i = 0; i < this->count(); ++i) {
//     this_data[i] *= rhs_data[i];
//   }
// }

// template <typename Real>
// var<Real>& var<Real>::operator-=(var<Real>& rhs) {
//   Real* this_data = this->mutable_cpu_data();
//   const Real* rhs_data = rhs.cpu_data();
//   for (int i = 0; i < this->count(); ++i) {
//     this_data[i] -= rhs_data[i];
//   }
// }

// template <typename Real>
// var<Real>& var<Real>::operator/=(var<Real>& rhs) {
//   Real* this_data = this->mutable_cpu_data();
//   const Real* rhs_data = rhs.cpu_data();
//   for (int i = 0; i < this->count(); ++i) {
//     this_data[i] /= rhs_data[i];
//   }
// }

// template <typename Real>
// var<Real>& var<Real>::operator+=(Real alpha) {
//   Real* this_data = this->mutable_cpu_data();
//   for (int i = 0; i < this->count(); ++i) {
//     this_data[i] += alpha;
//   }
// }

// template <typename Real>
// var<Real>& var<Real>::operator*=(Real alpha) {
//   Real* this_data = this->mutable_cpu_data();
//   for (int i = 0; i < this->count(); ++i) {
//     this_data[i] *= alpha;
//   }
// }

// template <typename Real>
// var<Real>& var<Real>::operator-=(Real alpha) {
//   Real* this_data = this->mutable_cpu_data();
//   for (int i = 0; i < this->count(); ++i) {
//     this_data[i] -= alpha;
//   }
// }

// template <typename Real>
// var<Real>& var<Real>::operator/=(Real alpha) {
//   Real* this_data = this->mutable_cpu_data();
//   for (int i = 0; i < this->count(); ++i) {
//     this_data[i] /= alpha;
//   }
// }

template <typename Real>
void var<Real>::print() {
  std::cout.precision(17);
  int rows = height();
  int cols = width();
  const Real* c_data = this->cpu_data();
  std::cout << "\n[ " << rows << "x" << cols << " [\n";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << c_data[cols * i + j] << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << "]]\n";
}

template <typename Real>
void var<Real>::print_diff() {
  int rows = height();
  int cols = width();
  const Real* c_data = this->cpu_diff();
  std::cout << "\n[ " << rows << "x" << cols << " [\n";
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << c_data[cols * i + j] << "  ";
    }
    std::cout << std::endl;
  }
  std::cout << "]]\n";
}

template <typename Real>
void var<Real>::print_shape() const {
  std::string message = "(";
  message += std::to_string(num());
  message += "x";
  message += std::to_string(channels());
  message += "x";
  message += std::to_string(height());
  message += "x";
  message += std::to_string(width());
  message += ")";
  std::cout << message << std::endl;
}

template <typename Real>
void var<Real>::update() {
  // perform computation on CPU
  // rnn_axpy<Real>(count_,
  //                   Real(-1),
  //                   static_cast<const Real*>(synced_diff_->cpu_data()),
  //                   static_cast<Real*>(synced_data_->mutable_cpu_data()));
}

template<typename Real>
var<Real> var<Real>::copy(const var<Real>& input) {
  auto output = zeros_like(input);
        Real* output_data = output.mutable_cpu_data();
        Real* output_diff = output.mutable_cpu_diff();
  const Real*  input_data =  input.cpu_data();
  const Real*  input_diff =  input.cpu_diff();

  for (int i = 0; i < output.count(); ++i) {
    output_data[i] = input_data[i];
    output_diff[i] = input_diff[i];
  }
  return output;
}

template<typename Real>
void var<Real>::deep_copy(const var<Real>& input) {
        Real*  this_data =  this->mutable_cpu_data();
        Real*  this_diff =  this->mutable_cpu_diff();
  const Real* input_data = input.cpu_data();
  const Real* input_diff = input.cpu_diff();

  for (int i = 0; i < this->count(); ++i) {
    this_data[i] = input_data[i];
    this_diff[i] = input_diff[i];
  }
}

template<typename Real>
void var<Real>::plus_eq(const var<Real>& input) {
        Real*  this_data =  this->mutable_cpu_data();
        // Real*  this_diff =  this->mutable_cpu_diff();
  const Real* input_data = input.cpu_data();
  // const Real* input_diff = input.cpu_diff();

  for (int i = 0; i < this->count(); ++i) {
    this_data[i] += input_data[i];
    // this_diff[i] += input_diff[i];
  }
}

template<typename Real>
void var<Real>::copy_data_to_data(const var<Real>& input) {
  const Real*  input_data = input.cpu_data();
        Real*   this_data = this->mutable_cpu_data();

  for (int i =0; i < this->count(); ++i) {
    this_data[i] = input_data[i];
  }
}

template<typename Real>
void var<Real>::copy_data_to_diff(const var<Real>& input) {
  const Real*  input_data = input.cpu_data();
        Real*   this_diff = this->mutable_cpu_diff();

  for (int i =0; i < this->count(); ++i) {
    this_diff[i] = input_data[i];
  }
}

template<typename Real>
void var<Real>::copy_diff_to_data(const var<Real>& input) {
  const Real*  input_diff = input.cpu_diff();
        Real*   this_data = this->mutable_cpu_data();

  for (int i =0; i < this->count(); ++i) {
    this_data[i] = input_diff[i];
  }
}

template<typename Real>
void var<Real>::copy_diff_to_diff(const var<Real>& input) {
  const Real*  input_diff = input.cpu_diff();
        Real*   this_diff = this->mutable_cpu_diff();

  for (int i =0; i < this->count(); ++i) {
    this_diff[i] = input_diff[i];
  }
}

template<typename Real>
var<Real> var<Real>::zeros_like(const var<Real>& input) {
  return var<Real>({input.num(),
                    input.channels(),
                    input.height(),
                    input.width()},
                    concurrent::initializer::constant<Real>(0.0));
}

template <typename Real>
const Real* var<Real>::cpu_data() const {
  return (const Real*)synced_data_->cpu_data();
}

template <typename Real>
const Real* var<Real>::cpu_diff() const {
  return (const Real*)synced_diff_->cpu_data();
}

template <typename Real>
Real* var<Real>::mutable_cpu_data() const {
  return static_cast<Real*>(synced_data_->mut_data());
}

template <typename Real>
Real* var<Real>::mutable_cpu_diff() const {
  return static_cast<Real*>(synced_diff_->mut_data());
}

template <typename Real>
Real var<Real>::asum_data() const {
  if (!synced_data_) { return 0; }
  return rnn_cpu_asum(count_, cpu_data());
  // return 0;
}

template <typename Real>
Real var<Real>::asum_diff() const {
  if (!synced_data_) { return 0; }
  return rnn_cpu_asum(count_, cpu_diff());
  // return 0;
}

template <typename Real>
Real var<Real>::sumsq_data() const {
  Real sumsq;
  const Real* data;
  if (!synced_data_) { return 0; }
  data = cpu_data();
  sumsq = rnn_cpu_dot(count_, data, data);
  return sumsq;
}

template <typename Real>
Real var<Real>::sumsq_diff() const {
  Real sumsq;
  const Real* diff;
  if (!synced_diff_) { return 0; }
  diff = cpu_diff();
  sumsq = rnn_cpu_dot(count_, diff, diff);
  return sumsq;
}

template <typename Real>
void var<Real>::zeros() {
  Real* data = this->mutable_cpu_data();
  for (int i = 0; i < count_; ++i) {
    data[i] = 0.0;
  }
}

template <typename Real>
void var<Real>::zeros_diff() {
  Real* diff = this->mutable_cpu_diff();
  for (int i = 0; i < count_; ++i) {
    diff[i] = 0.0;
  }
}

template <typename Real>
void var<Real>::grad() {
  concurrent::operation<Real>::grad(this);
}

template <typename Real>
void var<Real>::grad(Real value) {
  concurrent::operation<Real>::grad(this, value);
}

// INSTANTIATE_CLASS(var);
template class var<float>;
template class var<double>;

}  // namespace concurrent
