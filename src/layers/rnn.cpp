#include "concurrent/layers/rnn.hpp"

namespace concurrent {

namespace tf = concurrent;

template <typename T>
rnn<T>::rnn() {
  std::cout << "rnn constructor" << std::endl;
}

template <typename T>
rnn<T>::~rnn() {
}

template <typename T>
rnn<T>::rnn(int param_hidden_size, int param_data_size, int param_seq_length) {
  // using init = concurrent::initializer;
  hidden_size = param_hidden_size;
  data_size = param_data_size;
  timesteps = param_seq_length;
  hprev = var<T>({1, 1, hidden_size, 1}, tf::initializer::constant<T>(0.0));
  smooth_loss = -std::log(1.0 / data_size) * timesteps;
  // model parameters
  Wxh = var<T>({1, 1, hidden_size, data_size},   tf::initializer::randn<T>(0.01)); // input to hidden
  Whh = var<T>({1, 1, hidden_size, hidden_size}, tf::initializer::randn<T>(0.01)); // hidden to hidden
  Why = var<T>({1, 1, data_size,  hidden_size},  tf::initializer::randn<T>(0.01)); // hidden to output
  bh  = var<T>({1, 1, hidden_size, 1},           tf::initializer::constant<T>(0)); // hidden bias
  by  = var<T>({1, 1, data_size,  1},            tf::initializer::constant<T>(0)); // output bias

  // memory variables for Adagrad
  mWxh = var<T>::zeros_like(Wxh);
  mWhh = var<T>::zeros_like(Whh);
  mWhy = var<T>::zeros_like(Why);
  mbh  = var<T>::zeros_like(bh);
  mby  = var<T>::zeros_like(by);
}

template <typename T>
void rnn<T>::session_run(std::vector<int> in, std::vector<int> out) {
  inputs = in;
  targets = out;
}

template <typename T>
T rnn<T>::get_loss() {
  return loss;
}

template <typename T>
std::vector<int> rnn<T>::sample(int seed_ix, int n) {
  // sample a sequence of integers from the model 
  // h is memory state, seed_ix is seed letter for first time step
  using ops = concurrent::operation<T>;
  concurrent::graph::_set_backprop_enabled(false);
  int vocab_size = data_size;

  std::unordered_map<int, var<T>> h;
  // std::unordered_map<int, var<T>*> y;
  // std::unordered_map<int, var<T>*> p;

  h[-1]   = tf::var<T>::copy(hprev);
  auto x = var<T>({1, 1,  vocab_size, 1}, tf::initializer::constant<T>(0.0));
  auto y = var<T>({1, 1,  vocab_size, 1}, tf::initializer::constant<T>(0.0));
  auto p = var<T>({1, 1,  vocab_size, 1}, tf::initializer::constant<T>(0.0));
  x.at(seed_ix) = 1.0;
  std::vector<int> ixes;

  for (int t = 0; t < n; t++) {
    h[t] = ops::tanh(Wxh%x + Whh%h[t-1] + bh);
    y = Why%h[t] + by;
    p = ops::softmax(y);
    int ix = choice(p);
    x.zeros();
    x.at(ix) = 1.0;
    ixes.push_back(ix);
  }

  concurrent::graph::_set_backprop_enabled(true);
  //delete h[-1];
  //delete x;
  return ixes;
}

template <typename T>
int rnn<T>::choice(var<T>& c) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> d(0.0, 1.0);
  float val = d(gen);
  float sum = 0.0;
  const T* c_data = c.cpu_data();
  for (auto i = 0; i < c.count(); i++) {
    sum += c_data[i];
    if (val <= sum) { return i; }
  }
  return c.count() - 1;
}

template <typename T>
void rnn<T>::forward() {
  
  using ops = concurrent::operation<T>;
  hs[-1] = tf::var<T>::copy(hprev);
  loss = 0.0;
  // forward pass
  for (int t = 0; t < inputs.size(); ++t) {
    xs[t] = var<T>({1, 1, data_size, 1}, concurrent::initializer::constant<T>(0.0));     // encode in 1-of-k representation
    xs[t].at(inputs[t]) = 1.0;
    hs[t] = ops::tanh(Wxh % xs[t] + Whh % hs[t-1] + bh); // hidden state
    ys[t] = Why % hs[t] + by; // unnormalized log probabilities for next chars
    ps[t] = ops::softmax(ys[t]);                   // probabilities for next chars
    loss += -std::log(ps[t].at(targets[t]));      // softmax (cross-entropy loss)
    // PRECISE(loss);
  }
}

template <typename T>
void rnn<T>::backward() {
  for (int t = inputs.size() - 1; t >= 0; --t) {
    ys[t].copy_data_to_diff(ps[t]);
    ys[t].at(targets[t], true) -= 1;
  }
  Wxh.zeros_diff();
  Whh.zeros_diff();
  Why.zeros_diff();
  bh.zeros_diff();
  by.zeros_diff();
  
  concurrent::graph::backward();
  hprev.deep_copy(hs[inputs.size() - 1]);
}

template <typename T>
void rnn<T>::update() {
  using ops = concurrent::operation<T>;

  // perform parameter update with Adagrad
  ops::adagrad_update(Wxh, mWxh, learning_rate, 1e-8);
  ops::adagrad_update(Whh, mWhh, learning_rate, 1e-8);
  ops::adagrad_update(Why, mWhy, learning_rate, 1e-8);
  ops::adagrad_update(bh,  mbh,  learning_rate, 1e-8);
  ops::adagrad_update(by,  mby,  learning_rate, 1e-8);

  // ops::sgd_update(Wxh, learning_rate);
  // ops::sgd_update(Whh, learning_rate);
  // ops::sgd_update(Why, learning_rate);
  // ops::sgd_update(bh,  learning_rate);
  // ops::sgd_update(by,  learning_rate);
}

template <typename T>
void rnn<T>::calc_loss() {
  smooth_loss = smooth_loss * 0.999 + loss * 0.001;
  for (int i = timesteps - 1; i >= 0; --i) {
    // const T* label_l = label_[i]->cpu_data();
    // const T*   out_l =    out[i]->cpu_data();
    // T*    diff_out_l =    out[i]->mutable_cpu_diff();
    // diff_out_l[0] += out_l[0] - label_l[0];
  }
  return;
}

template class rnn<float>;
template class rnn<double>;

}
