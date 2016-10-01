#include "concurrent/parameter.hpp"

namespace concurrent {

parameter param;

void echo() { param.print(); }

void parameter::print() {
  std::cout << "float    dropout_prob    = "  << dropout_prob  << std::endl;
  std::cout << "float    grad_momentum   = "  << grad_momentum << std::endl;
  std::cout << "float    init_scale      = "  << init_scale    << std::endl;
  std::cout << "float    keep_prob       = "  << keep_prob     << std::endl;
  std::cout << "float    learning_rate   = "  << learning_rate << std::endl;
  std::cout << "float    lr_decay        = "  << lr_decay      << std::endl;
  std::cout << "  int    batch_size      = "  << batch_size    << std::endl;
  std::cout << "  int    hidden_size     = "  << hidden_size   << std::endl;
  std::cout << "  int    max_epoch       = "  << max_epoch     << std::endl;
  std::cout << "  int    max_grad_norm   = "  << max_grad_norm << std::endl;
  std::cout << "  int    max_max_epoch   = "  << max_max_epoch << std::endl;
  std::cout << "  int    num_layers      = "  << num_layers    << std::endl;
  std::cout << "  int    num_steps       = "  << num_steps     << std::endl;
  std::cout << "  int    vocab_size      = "  << vocab_size    << std::endl;
}  

}  // namespace concurrent
