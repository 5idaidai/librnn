#ifndef CONCURRENT_VAR_HPP
#define CONCURRENT_VAR_HPP

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <vector>

#include "concurrent/common.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/init.hpp"
#include "concurrent/operations/ops.hpp"
#include "concurrent/syncedmem.hpp"

namespace concurrent {

template <typename Real>
class var {
 public:
  var() { var(1, 1, 1, 1); }
  var(const int num, const int channels, const int height, const int width);
  explicit var(const std::vector<int>& shape);
  var(Real num);

  void reshape(const std::vector<int>& shape);
  void reshape(const int num, const int channels, const int height, const int width);

  template <typename Initializer>
  var(std::initializer_list<int> initlist, Initializer filler) {
    std::vector<int> shape;
    for (auto x : initlist) {
      shape.push_back(x);
    }
    reshape(shape);
    filler.fill(*this);
  }

  ~var();
  // var(const var<Real>& other);                   // copy constructor
  // var<Real>& operator=(const var<Real>& other);  // copy assignment
  // var(var<Real>&& other);                        // move constructor
  // var<Real>& operator=(var<Real>&& other);       // move assignment

  Real& operator[](const int index);
  Real& at(const int index, bool diff = false);
  Real  diff_at(const int index);

  // arithmetic op overloads (const after func?)
  var<Real>  operator-();
  var<Real>  operator%(const var<Real>& rhs) const;  // matrix multiplication
  var<Real>  operator+(const var<Real>& rhs) const;  // element-wise add
  var<Real>  operator*(const var<Real>& rhs) const;  // element-wise multiply
  var<Real>  operator-(const var<Real>& rhs) const;  // element-wise subtract
  var<Real>  operator/(const var<Real>& rhs) const;  // element-wise divide

  var<Real>  operator+(Real alpha) const;
  var<Real>  operator-(Real alpha) const;
  var<Real>  operator*(Real alpha) const;
  var<Real>  operator/(Real alpha) const;

  // var<Real>&  operator+(Real alpha);
  // var<Real>&  operator-(Real alpha);
  // var<Real>&  operator*(Real alpha);
  // var<Real>&  operator/(Real alpha);

  var<Real>& operator+=(const var<Real>& rhs);
  var<Real>& operator*=(const var<Real>& rhs);
  var<Real>& operator-=(const var<Real>& rhs);
  var<Real>& operator/=(const var<Real>& rhs);

  static var<Real>       copy(const var<Real>& input);
  static var<Real> zeros_like(const var<Real>& input);

  void         deep_copy(const var<Real>& input);
  void           plus_eq(const var<Real>& input);
  void copy_data_to_data(const var<Real>& input);
  void copy_data_to_diff(const var<Real>& input);
  void copy_diff_to_data(const var<Real>& input);
  void copy_diff_to_diff(const var<Real>& input);

  const Real* cpu_data()   const;
  const Real* cpu_diff()   const;
  Real* mutable_cpu_data() const;
  Real* mutable_cpu_diff() const;

  Real asum_data()  const;  // Compute the sum of absolute values (L1 norm) of the data.
  Real asum_diff()  const;  // Compute the sum of absolute values (L1 norm) of the diff.
  Real sumsq_data() const;  // Compute the sum of squares (L2 norm squared) of the data.
  Real sumsq_diff() const;  // Compute the sum of squares (L2 norm squared) of the diff.

  void update();
  void zeros();
  void zeros_diff();
  void print();
  void print_diff();
  void print_shape() const;
  void grad();
  void grad(Real value);

  inline int shape(int idx) const { return shape_[idx]; }
  inline int num()          const { return shape_[0]; }
  inline int channels()     const { return shape_[1]; }
  inline int height()       const { return shape_[2]; }
  inline int width()        const { return shape_[3]; }
  inline int count()        const { return count_; }
  bool is_grad_nan() { return true; }

 protected:
  int count_;
  std::vector<int> shape_;
  std::shared_ptr<concurrent::synced_memory> synced_data_;
  std::shared_ptr<concurrent::synced_memory> synced_diff_;
};

}  // namespace concurrent

#endif  // CONCURRENT_VAR_HPP
