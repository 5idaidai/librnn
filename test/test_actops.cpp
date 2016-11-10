#include <vector>

#include "gtest/gtest.h"

#include "concurrent/graph.hpp"
#include "concurrent/operations/ops.hpp"
#include "concurrent/var.hpp"

TEST(activationOpsTest, mult) {
  using var = concurrent::var<float>;

  auto input = var({1, 1, 2, 3}, concurrent::initializer::constant<float>(0.0));
  auto input_data = input.mutable_cpu_data();
  input_data[0] = -5;
  input_data[1] =  0.25;
  input_data[2] =  0;
  input_data[3] =  0.75;
  input_data[4] =  1.5;
  input_data[5] =  0.33;
  // { {-5, 0.25, 0}, {0.75, 1.5, 0.333} };
  
  var output = concurrent::operation<float>::sigmoid(input);
  std::vector<float> gradient = {0.0066480566707901546, 0.24613408273759835, 0.25, 0.21789499376181404, 0.14914645207033286, 0.24319554120272233};

  output.grad();
  concurrent::graph::backward();

  for (int i = 0; i < 2*2; i++) {
    EXPECT_FLOAT_EQ(input.diff_at(i), gradient[i]);
  }
}

TEST(activationOpsTest, twothiryone) {
  using var = concurrent::var<float>;
  using ops = concurrent::operation<float>;
  var w0(2);
  var x0(-1);
  var w1(-3);
  var x1(-2);
  var w2(-3);

  var result  = ops::elt_inv(ops::exp(-(w0 * x0 + w1 * x1 + w2)) + 1);
  result.grad();
  concurrent::graph::backward();

  EXPECT_FLOAT_EQ(w0.diff_at(0), -0.196612);
  EXPECT_FLOAT_EQ(x0.diff_at(0),  0.393224);
  EXPECT_FLOAT_EQ(w1.diff_at(0), -0.393224);
  EXPECT_FLOAT_EQ(x1.diff_at(0), -0.589836);
  
  EXPECT_FLOAT_EQ(result.diff_at(0), 1.0);
}
