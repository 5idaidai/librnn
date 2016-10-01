#ifndef CONCURRENT_PARAMETER_HPP
#define CONCURRENT_PARAMETER_HPP

#include <iostream>

#include "concurrent/common.hpp"

namespace concurrent {

void echo();

class parameter {
 public:
  float dropout_prob;   // - keep_prob - the probability of keeping weights in the dropout layer
  float grad_momentum;  
  float init_scale;     // - init_scale - the initial scale of the weights
  float keep_prob;      //
  float learning_rate;  // - learning_rate - the initial value of the learning rate
  float lr_decay;       // - lr_decay - the decay of the learning rate for each epoch after "max_epoch"
  int batch_size;       // - batch_size - the batch size
  int hidden_size;      // - hidden_size - the number of LSTM units
  int max_epoch;        // - max_epoch - the number of epochs trained with the initial learning rate
  int max_grad_norm;    // - max_grad_norm - the maximum permissible norm of the gradient
  int max_max_epoch;    // - max_max_epoch - the total number of epochs for training
  int num_layers;       // - num_layers - the number of LSTM layers
  int num_steps;        // - num_steps - the number of unrolled steps of LSTM
  int vocab_size;       

  int seq_length;
  int data_size;

  void print();
};

extern parameter param;

}  // namespace concurrent

#endif  // CONCURRENT_PARAMETER_HPP
