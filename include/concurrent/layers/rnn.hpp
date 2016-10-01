#ifndef CONCURRENT_LAYERS_RNN_HPP
#define CONCURRENT_LAYERS_RNN_HPP

#include <vector>
#include <unordered_map>

#include "concurrent/common.hpp"
#include "concurrent/parameter.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/math.hpp"
#include "concurrent/operations/ops.hpp"
#include "concurrent/var.hpp"

namespace concurrent {
    
template <typename T>
class rnn {
 public:
  rnn();
  rnn(int param_hidden_size, int param_data_size, int param_seq_length);
  ~rnn();

  T get_loss();
  
  std::vector<int> sample(int seed_ix, int n);
  int              choice(var<T>& c);

  void session_run(std::vector<int> in, std::vector<int> out);
  void forward();
  void backward();
  void update();
  void calc_loss();

  std::string layer_type() { return "rnn"; }

 private:
  // int input_size;
  int hidden_size;
  int data_size;
  // int output_size;
  // int seq_length
  int timesteps = 2;

  T loss = 0.0;
  T smooth_loss;
  const float learning_rate = 1e-1;

  std::unordered_map<int, var<T>> xs;
  std::unordered_map<int, var<T>> hs;
  std::unordered_map<int, var<T>> ys;
  std::unordered_map<int, var<T>> ps;

  // model parameters
  var<T> Wxh;
  var<T> Whh;
  var<T> Why;
  var<T> bh;
  var<T> by;

  // memory variables for Adagrad
  var<T> mWxh;
  var<T> mWhh;
  var<T> mWhy;
  var<T> mbh;
  var<T> mby;

  var<T> hprev;
  
  std::vector<int> inputs;
  std::vector<int> targets;

private:

};

} // namespace concurrent

#endif // CONCURRENT_LAYERS_RNN_HPP
