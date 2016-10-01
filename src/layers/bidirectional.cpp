#include "concurrent/layers/bidirectional.hpp"

namespace concurrent {

namespace tf = concurrent;

// template <typename T>
// lstm<T>::lstm() {
//   std::cout << "lstm constructor" << std::endl;
// }

// template <typename T>
// void lstm<T>::del_help(std::vector<var<T>*>& vec) {
//   for (auto x : vec) {
//     delete x;
//   }
// }

// template <typename T>
// lstm<T>::~lstm() {
//   del_help(data_);  //
//   del_help(label_);
//   del_help(w_a);    //
//   del_help(w_i);
//   del_help(w_f);
//   del_help(w_o);
//   del_help(u_a);    //
//   del_help(u_i);
//   del_help(u_f);
//   del_help(u_o);
//   del_help(b_a);    //
//   del_help(b_i);
//   del_help(b_f);
//   del_help(b_o);
//   del_help(activation);  //
//   del_help(input);
//   del_help(forget);
//   del_help(output);
//   del_help(state);  //
//   del_help(out);
// }

// template <typename T>
// lstm<T>::lstm(std::vector<var<T>*>& inputs, std::vector<var<T>*>& labels) {
//   data_ = inputs;
//   label_ = labels;

//   for (int i = 0; i < timesteps; i++) {
//     activation.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));
//          input.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));
//         forget.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));
//         output.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));
//          state.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));
//            out.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));
//   }

//   state.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));
//     out.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.0)));

//   w_a.push_back(new var<T>({1, 1, 1, 2}, tf::initializer::constant<T>(0.0)));
//   w_i.push_back(new var<T>({1, 1, 1, 2}, tf::initializer::constant<T>(0.0)));
//   w_f.push_back(new var<T>({1, 1, 1, 2}, tf::initializer::constant<T>(0.0)));
//   w_o.push_back(new var<T>({1, 1, 1, 2}, tf::initializer::constant<T>(0.0)));
  
//   T* data_data  = nullptr;
//   data_data     = w_a.at(0)->mutable_cpu_data();
//   data_data[0]  = 0.45;
//   data_data[1]  = 0.25;

//   data_data     = w_i.at(0)->mutable_cpu_data();
//   data_data[0]  = 0.95;
//   data_data[1]  = 0.8;

//   data_data     = w_f.at(0)->mutable_cpu_data();
//   data_data[0]  = 0.7;
//   data_data[1]  = 0.45;

//   data_data     = w_o.at(0)->mutable_cpu_data();
//   data_data[0]  = 0.6;
//   data_data[1]  = 0.4;
  
//   u_a.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.15)));
//   u_i.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.8)));
//   u_f.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.1)));
//   u_o.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.25)));

//   b_a.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.2)));
//   b_i.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.65)));
//   b_f.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.15)));
//   b_o.push_back(new var<T>({1, 1, 1, 1}, tf::initializer::constant<T>(0.1)));
// }

// template <typename T>
// void lstm<T>::verify_data(var<T>* blob, T value) {
//   const T* data = blob->cpu_data();
//   concurrent::EXPECT_REAL_EQ(data[0],  value);
// }

// template <typename T>
// void lstm<T>::verify_diff(var<T>* blob, T value) {
//   const T* diff = blob->cpu_diff();
//   // std::cout.precision(17);std::cout << diff;
//   concurrent::EXPECT_REAL_EQ(diff[0],  value);
// }

// template <typename T>
// void lstm<T>::forward() {
//   using tf = concurrent::operation<T>;
//   // var<T> a =    concurrent::operation<T>::tanh(wa*x0 + Ua*out_m1 + ba);
//   // var<T> i = concurrent::operation<T>::sigmoid(wi*x0 + Ui*out_m1 + bi);
//   // var<T> f = concurrent::operation<T>::sigmoid(wf*x0 + Uf*out_m1 + bf);
//   // var<T> o = concurrent::operation<T>::sigmoid(wo*x0 + Uo*out_m1 + bo);
//   for (int i = 0; i < timesteps; ++i) {
//     activation[i] = tf::tanh(tf::add(tf::add(tf::matmul(w_a.at(0), data_.at(i)), tf::mul(u_a.at(0), out_func(i-1))), b_a.at(0)));
//     input[i]   = tf::sigmoid(tf::add(tf::add(tf::matmul(w_i.at(0), data_.at(i)), tf::mul(u_i.at(0), out_func(i-1))), b_i.at(0)));
//     forget[i]  = tf::sigmoid(tf::add(tf::add(tf::matmul(w_f.at(0), data_.at(i)), tf::mul(u_f.at(0), out_func(i-1))), b_f.at(0)));
//     output[i]  = tf::sigmoid(tf::add(tf::add(tf::matmul(w_o.at(0), data_.at(i)), tf::mul(u_o.at(0), out_func(i-1))), b_o.at(0)));
    
//     state[i]   = tf::add(tf::mul(activation[i], input[i]), tf::mul(forget[i], state_func(i-1)));
//     out[i]     = tf::mul(tf::tanh(state[i]), output.at(i));
//   }

//   DBGLINE();
//   verify_data(activation[0], 0.817754078);
//   verify_data(     input[0], 0.9608342772);
//   verify_data(    forget[0], 0.8519528019);
//   verify_data(    output[0], 0.8175744762);
//   verify_data(     state[0], 0.7857261485);
//   verify_data(       out[0], 0.5363133979);

//   DBGLINE();
//   verify_data(activation[1], 0.8498040223);
//   verify_data(     input[1], 0.9811839683);
//   verify_data(    forget[1], 0.8703019699);
//   verify_data(    output[1], 0.8499333428);
//   verify_data(     state[1], 1.5176330977);
//   verify_data(       out[1], 0.7719811058);
// }

// template <typename T>
// void lstm<T>::backward() {
//   out[1]->grad(0.0);
//   concurrent::graph::backward();
  
//   DBGLINE();
//   verify_diff(       out[1], -0.4780188942);
//   verify_diff(     state[1], -0.07110771472);
//   verify_diff(activation[1], -0.06976974971);
//   verify_diff(     input[1], -0.06042762199);
//   verify_diff(    forget[1], -0.05587119082);
//   verify_diff(    output[1], -0.4341770536);

//   DBGLINE();
//   verify_diff(       out[0],  0.01803815923);
//   verify_diff(     state[0], -0.05348367654);
//   verify_diff(activation[0], -0.05138894969);
//   verify_diff(     input[0], -0.0437364946);
//   verify_diff(    forget[0],  0.0);
//   verify_diff(    output[0],  0.01183269139);
// }

// template <typename T>
// void lstm<T>::update() {
//   using tf = concurrent::operation<T>;
//   tf::sgd_update(w_a[0], 0.1);
//   tf::sgd_update(w_i[0], 0.1);
//   tf::sgd_update(w_f[0], 0.1);
//   tf::sgd_update(w_o[0], 0.1);
//   tf::sgd_update(u_a[0], 0.1);
//   tf::sgd_update(u_i[0], 0.1);
//   tf::sgd_update(u_f[0], 0.1);
//   tf::sgd_update(u_o[0], 0.1);
//   tf::sgd_update(b_a[0], 0.1);
//   tf::sgd_update(b_i[0], 0.1);
//   tf::sgd_update(b_f[0], 0.1);
//   tf::sgd_update(b_o[0], 0.1);

//   DBGLINE();
//   verify_data(w_a[0], 0.45267161726951599);//->print();
//   verify_data(w_i[0], 0.95022034645080566);//->print();
//   verify_data(w_f[0], 0.70031529664993286);//->print();
//   verify_data(w_o[0], 0.60259240865707397);//->print();
//   verify_data(u_a[0], 0.15103961527347565);//->print();
//   verify_data(u_i[0], 0.80005985498428345);//->print();
//   verify_data(u_f[0], 0.10033822804689407);//->print();
//   verify_data(u_o[0], 0.25296998023986816);//->print();
//   verify_data(b_a[0], 0.20364084839820862);//->print();
//   verify_data(b_i[0], 0.65027612447738647);//->print();
//   verify_data(b_f[0], 0.1506306529045105);//->print();
//   verify_data(b_o[0], 0.10536130517721176);//->print();
// }

// template <typename T>
// void lstm<T>::calc_loss() {
//   for (int i = timesteps - 1; i >= 0; --i) {
//     const T* label_l = label_[i]->cpu_data();
//     const T*   out_l =    out[i]->cpu_data();
//     T*    diff_out_l =    out[i]->mutable_cpu_diff();
//     diff_out_l[0] += out_l[0] - label_l[0];
//   }
//   return;
// }

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

// template class lstm<float>;
// template class lstm<double>;

}
