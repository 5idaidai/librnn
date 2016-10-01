#ifndef CONCURRENT_LAYERS_LSTM_HPP
#define CONCURRENT_LAYERS_LSTM_HPP

#include <vector>
#include <unordered_map>

#include "concurrent/common.hpp"
#include "concurrent/graph.hpp"
#include "concurrent/math.hpp"
#include "concurrent/operations/ops.hpp"
#include "concurrent/var.hpp"

namespace concurrent {
    
template <typename T>
class lstm {
 public:
  lstm();
  lstm(std::vector<var<T>*> inputs, std::vector<var<T>*> labels);
  lstm(int param_hidden_size, int param_data_size, int param_seq_length);
  ~lstm();

  T get_loss();
  void session_run(std::vector<int> in, std::vector<int> out);
  void forward();
  void backward();
  void update();
  void calc_loss();

  std::string layer_type() { return "lstm"; }

 private:
  // int input_size;
  // int hidden_size;
  // int output_size;

  void del_help(std::vector<var<T>*>& vec);

  void verify_data(var<T>& blob, T value);
  void verify_diff(var<T>& blob, T value);

  // var<T>* state_func(int i);
  // var<T>*   out_func(int i);
  int hidden_size;
  int data_size;
  int timesteps = 2;
  
  std::vector<var<T>*> data_;
  std::vector<var<T>*> label_;

  var<T> w_a;
  var<T> w_i;
  var<T> w_f;
  var<T> w_o;

  var<T> u_a;
  var<T> u_i;
  var<T> u_f;
  var<T> u_o;

  var<T> b_a;
  var<T> b_i;
  var<T> b_f;
  var<T> b_o;

  // std::vector<var<T>*> activation;
  // std::vector<var<T>*> input;
  // std::vector<var<T>*> forget;
  // std::vector<var<T>*> output;

  std::unordered_map<int, var<T>> activation;
  std::unordered_map<int, var<T>> input;
  std::unordered_map<int, var<T>> forget;
  std::unordered_map<int, var<T>> output;

  // std::vector<var<T>*> state;
  // std::vector<var<T>*> out;

  std::unordered_map<int, var<T>> state;
  std::unordered_map<int, var<T>> out;

private:

};

} // namespace concurrent

#endif // CONCURRENT_LAYERS_LSTM_HPP
