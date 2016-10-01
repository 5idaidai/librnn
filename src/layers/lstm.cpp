#include "concurrent/layers/lstm.hpp"

namespace concurrent {

namespace tf = concurrent;

template <typename T>
lstm<T>::lstm() {
  std::cout << "lstm constructor" << std::endl;
}

template <typename T>
void lstm<T>::del_help(std::vector<var<T>*>& vec) {
  for (auto x : vec) {
    delete x;
    x = nullptr;
  }
}

template <typename T>
lstm<T>::lstm(int param_hidden_size, int param_data_size, int param_seq_length) {
  // using init = concurrent::initializer;
  hidden_size = param_hidden_size;
  data_size = param_data_size;
  timesteps = param_seq_length;
}

template <typename T>
lstm<T>::~lstm() {
  DBGLINE();
  del_help(data_);  //
  del_help(label_);
  // del_help(w_a);    //
  // del_help(w_i);
  // del_help(w_f);
  // del_help(w_o);
  // del_help(u_a);    //
  // del_help(u_i);
  // del_help(u_f);
  // del_help(u_o);
  // del_help(b_a);    //
  // del_help(b_i);
  // del_help(b_f);
  // del_help(b_o);
  // delete out[-1];
  // delete state[-1];

  // del_help(activation);  //
  // del_help(input);
  // del_help(forget);
  // del_help(output);
  // del_help(state);  //
  // del_help(out);
  DBGLINE();
}

template <typename T>
lstm<T>::lstm(std::vector<var<T>*> inputs, std::vector<var<T>*> labels) {
  data_ = inputs;
  label_ = labels;
  for (int i = 0; i < timesteps; i++) {
    // activation[i] = new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0));
         // input[i] = new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0));
        // forget[i] = new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0));
        // output[i] = new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0));
         // state.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));
           // out.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));
  }
  // state.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));
    // out.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));

  w_a = var<T>({1, 1, 1, 2}, tf::initializer::constant<T>(0.0));
  w_i = var<T>({1, 1, 1, 2}, tf::initializer::constant<T>(0.0));
  w_f = var<T>({1, 1, 1, 2}, tf::initializer::constant<T>(0.0));
  w_o = var<T>({1, 1, 1, 2}, tf::initializer::constant<T>(0.0));
  
  T* data_data  = nullptr;
  data_data     = w_a.mutable_cpu_data();
  data_data[0]  = 0.45;
  data_data[1]  = 0.25;

  data_data     = w_i.mutable_cpu_data();
  data_data[0]  = 0.95;
  data_data[1]  = 0.8;

  data_data     = w_f.mutable_cpu_data();
  data_data[0]  = 0.7;
  data_data[1]  = 0.45;

  data_data     = w_o.mutable_cpu_data();
  data_data[0]  = 0.6;
  data_data[1]  = 0.4;
  
  u_a = var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.15));
  u_i = var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.80));
  u_f = var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.10));
  u_o = var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.25));

  b_a = var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.20));
  b_i = var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.65));
  b_f = var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.15));
  b_o = var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.10));
  DBGLINE();
}

template <typename T>
void lstm<T>::verify_data(var<T>& blob, T value) {
  const T* data = blob.cpu_data();
  concurrent::EXPECT_REAL_EQ(data[0],  value);
}

template <typename T>
void lstm<T>::verify_diff(var<T>& blob, T value) {
  const T* diff = blob.cpu_diff();
  // std::cout.precision(17);std::cout << diff;
  concurrent::EXPECT_REAL_EQ(diff[0],  value);
}

template <typename T>
void lstm<T>::forward() {
  using ops = concurrent::operation<T>;
  out[-1] = concurrent::var<T>::zeros_like(u_a);
  state[-1] = var<T>({1, 1, 1, 1}, concurrent::initializer::constant<T>(0.0));

  for (int i = 0; i < timesteps; ++i) {
    activation[i] = ops::tanh(w_a % (*data_.at(i)) + u_a * out[i-1] + b_a);
    input[i]   = ops::sigmoid(w_i % (*data_.at(i)) + u_i * out[i-1] + b_i);
    forget[i]  = ops::sigmoid(w_f % (*data_.at(i)) + u_f * out[i-1] + b_f);
    output[i]  = ops::sigmoid(w_o % (*data_.at(i)) + u_o * out[i-1] + b_o);
    
    state[i]   = activation[i] * input[i] + forget[i] * state[i-1];
    out[i]     = ops::tanh(state[i]) * output.at(i);
  }
}

template <typename T>
void lstm<T>::backward() {
  out[1].grad(0.0);
  concurrent::graph::backward();

}

template <typename T>
void lstm<T>::update() {
  using tf = concurrent::operation<T>;
  tf::sgd_update(w_a, 0.1);
  tf::sgd_update(w_i, 0.1);
  tf::sgd_update(w_f, 0.1);
  tf::sgd_update(w_o, 0.1);
  tf::sgd_update(u_a, 0.1);
  tf::sgd_update(u_i, 0.1);
  tf::sgd_update(u_f, 0.1);
  tf::sgd_update(u_o, 0.1);
  tf::sgd_update(b_a, 0.1);
  tf::sgd_update(b_i, 0.1);
  tf::sgd_update(b_f, 0.1);
  tf::sgd_update(b_o, 0.1);
}

template <typename T>
void lstm<T>::calc_loss() {
  for (int i = timesteps - 1; i >= 0; --i) {
    const T* label_l = label_[i]->cpu_data();
    const T*   out_l =    out[i].cpu_data();
    T*    diff_out_l =    out[i].mutable_cpu_diff();
    diff_out_l[0] += out_l[0] - label_l[0];
  }
  return;
}

// template <typename T>
// var<T>* lstm<T>::state_func(int i) {
//   if (i == -1) {
//     auto toRet = new var<T>({forget[0]->get_num(), forget[0]->get_channels(), forget[0]->get_height(), forget[0]->get_width()}, tf::initializer::constant<T>(0.0));
//     return toRet;
//   } else if (i >= 0) {
//     return state[i];
//   } else {
//     std::cout << B_RED << "A index of " << i << " was used for state[i]!" << END_COLOR << std::endl;
//     return nullptr;
//   }
// }

// template <typename T>
// var<T>* lstm<T>::out_func(int i) {
//   if (i == -1) {
//     auto toRet = new var<T>({u_a[0]->get_num(), u_a[0]->get_channels(), u_a[0]->get_height(), u_a[0]->get_width()}, tf::initializer::constant<T>(0.0));
//     return toRet;
//   } else if (i >= 0) {
//     return out[i];
//   } else {
//     std::cout << B_RED << "A index of " << i << " was used for out[i]!" << END_COLOR << std::endl;
//     return nullptr;
//   }
// }

template class lstm<float>;
template class lstm<double>;

}
