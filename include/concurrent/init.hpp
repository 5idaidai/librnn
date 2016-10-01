#ifndef CONCURRENT_INIT_HPP
#define CONCURRENT_INIT_HPP

#include <random>
// #include <algorithm>
#include "concurrent/var.hpp"
#include "concurrent/common.hpp"

namespace concurrent {

template <typename R> class var;

namespace initializer {

// class filler {
//  public:
//   virtual void fill(int weight, int fan_in, int fan_out) = 0;
// };

// class scalable : public filler {
// public:
//     scalable(Real value) : scale_(value) {}

//     void scale(Real value) {
//         scale_ = value;
//     }
// protected:
//     Real scale_;
// };

template <typename Real>
class scalable {
 public:
  scalable(Real value) : scale_(value) {}
  void scale(Real value) { scale_ = value; }
  virtual void fill(Real* weight, int size, int fan_in, int fan_out) = 0;
  virtual void fill(var<Real>& input) = 0;

 protected:
  Real scale_;
};

template <typename Real>
class xavier : public scalable<Real> {
 public:
  // Note: double(6) was used in the original code
  xavier() : scalable<Real>(Real(6)) {}
  explicit xavier(Real value) : scalable<Real>(value) {}

  void fill(Real* weight, int size, int fan_in, int fan_out) override {
    const Real weight_base = std::sqrt(this->scale_ / (fan_in + fan_out));
    // uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
    std::uniform_real_distribution<Real> dst(-weight_base, weight_base);
    std::mt19937 gen_(1);
    for (int i = 0; i < size; i++) {
      weight[i] = dst(gen_);
      // *it = dst(gen_);
    }
  }
};

template <typename Real>
class gaussian : public scalable<Real> {
 public:
  gaussian() : scalable<Real>(Real(1)) {}
  explicit gaussian(Real sigma) : scalable<Real>(sigma) {}

  void fill(Real* weight, int size, int fan_in, int fan_out) override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d;
    for (int i = 0; i < size; i++) {
      weight[i] = d(gen);
    }
    // gaussian_rand(weight->begin(), weight->end(), Real(0), scale_);
  }

  void fill(var<Real>& input) override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d;
    const int count = input.count();
    Real* input_data = input.mutable_cpu_data();
    for (int i = 0; i < count; i++) {
      input_data[i] = d(gen) * this->scale_;
    }
  }
};

template <typename Real>
class randn : public scalable<Real> {
 public:
  randn() : scalable<Real>(Real(1)) {}
  explicit randn(Real sigma) : scalable<Real>(sigma) {}

  void fill(Real* weight, int size, int fan_in, int fan_out) override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d;
    for (int i = 0; i < size; i++) {
      weight[i] = d(gen);
    }
    // gaussian_rand(weight->begin(), weight->end(), Real(0), scale_);
  }

  void fill(var<Real>& input) override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d;
    const int count = input.count();
    Real* input_data = input.mutable_cpu_data();
    for (int i = 0; i < count; i++) {
      input_data[i] = d(gen) * this->scale_;
    }
  }
};


template <typename Real>
class constant : public scalable<Real> {
 public:
  constant() : scalable<Real>(Real(7.0)) {}
  explicit constant(Real value) : scalable<Real>(value) {}

  void fill(Real* weight, int size, int fan_in, int fan_out) override {
    for (int i = 0; i < size; i++) {
      weight[i] = this->scale_;
    }
  }

  void fill(var<Real>& input) override {
    const int count = input.count();
    Real* input_data = input.mutable_cpu_data();
    for (int i = 0; i < count; i++) {
      input_data[i] = this->scale_;
    }
  }
};

// template <typename Real>
// class he : public scalable<Real> {
//  public:
//   he() : scalable<Real>(Real(2)) {}
//   explicit he(Real value) : scalable<Real>(value) {}

//   void fill(Real* weight, int fan_in, int fan_out) override {
//     const Real weight_base = std::sqrt(this->scale_ / fan_in);
//     // uniform_rand(weight->begin(), weight->end(), -weight_base, weight_base);
//     std::uniform_real_distribution<Real> dst(-weight_base, weight_base);
//     std::mt19937 gen_(1);
//     for (auto it = weight->begin(); it != weight->end(); ++it) {
//       *it = dst(gen_);
//     }
//   }
// };

} // namespace initializer
} // namespace concurrent

#endif  // CONCURRENT_INIT_HPP