#ifndef CONCURRENT_LAYERS_BIDIRECTIONAL_HPP
#define CONCURRENT_LAYERS_BIDIRECTIONAL_HPP

#include <vector>

#include "concurrent/common.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/math.hpp"
#include "concurrent/operations/ops.hpp"
#include "concurrent/var.hpp"

namespace concurrent {
    
template <typename T>
class bidirectional {
 public:
  bidirectional();
  bidirectional(std::vector<var<T>*>& inputs, std::vector<var<T>*>& labels);
  ~bidirectional();

  std::string layer_type() { 
    std::string message = "bidirectional<";
    // message += model.layer_type();
    message += ">";
    return message;
  }

};

} // namespace concurrent

#endif // CONCURRENT_LAYERS_BIDIRECTIONAL_HPP
