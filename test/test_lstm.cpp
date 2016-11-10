#include "gtest/gtest.h"
#include "concurrent/layers/lstm.hpp"

TEST(lstm, f) {
    concurrent::lstm<float> l(1, 1, 1);

    EXPECT_EQ("lstm", l.layer_type());
}
