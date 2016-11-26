#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <concurrent/core.hpp>

namespace tf = concurrent;
using ops    = concurrent::operation<float>;
using var    = concurrent::var<float>;
void lstm();
void char_rnn();
std::vector<int> sample(var& h, int seed_ix, int n);
int              choice(const var& c);

const int hidden_size = 100; // size of hidden layer of neurons
const int VSIZE = 80;

auto hprev =  var({1, 1, hidden_size, 1}, tf::initializer::constant<float>(0.0));
// model parameters
auto Wxh = var({1, 1, hidden_size, VSIZE},  tf::initializer::randn<float>(0.01));  // input to hidden
auto Whh = var({1, 1, hidden_size, hidden_size}, tf::initializer::randn<float>(0.01));  // hidden to hidden
auto Why = var({1, 1, VSIZE,  hidden_size}, tf::initializer::randn<float>(0.01));  // hidden to output
auto bh  = var({1, 1, hidden_size, 1}, tf::initializer::constant<float>(0.0));  // hidden bias
auto by  = var({1, 1, VSIZE,  1}, tf::initializer::constant<float>(0.0));  // output bias

// memory variables for Adagrad
auto mWxh = var({1, 1, hidden_size, VSIZE},  tf::initializer::constant<float>(0.0));  // input to hidden
auto mWhh = var({1, 1, hidden_size, hidden_size}, tf::initializer::constant<float>(0.0));  // hidden to hidden
auto mWhy = var({1, 1, VSIZE,  hidden_size}, tf::initializer::constant<float>(0.0));  // hidden to output
auto mbh  = var({1, 1, hidden_size, 1}, tf::initializer::constant<float>(0.0));  // hidden bias
auto mby  = var({1, 1, VSIZE,  1}, tf::initializer::constant<float>(0.0));  // output bias

std::vector<int> sample(var& hprev, int seed_ix, int n) {
  // sample a sequence of integers from the model 
  // h is memory state, seed_ix is seed letter for first time step
  concurrent::graph::_set_backprop_enabled(false);
  int vocab_size = VSIZE;

  std::unordered_map<int, var> h;
  // std::unordered_map<int, var*> y;
  // std::unordered_map<int, var*> p;

  h[-1]   = tf::var<float>::copy(hprev);
  auto x = var({1, 1,  vocab_size, 1}, tf::initializer::constant<float>(0.0));
  auto y = var({1, 1,  vocab_size, 1}, tf::initializer::constant<float>(0.0));
  auto p = var({1, 1,  vocab_size, 1}, tf::initializer::constant<float>(0.0));
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

  // concurrent::graph::_set_backprop_enabled(true);
  // delete h[-1];
  // delete x;
  return ixes;
}

int choice(const var& c) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> d(0.0, 1.0);
  float val = d(gen);
  float sum = 0.0;
  const float* c_data = c.cpu_data();
  for (auto i = 0; i < c.count(); i++) {
    sum += c_data[i];
    if (val <= sum) { return i; }
  }
  return c.count() - 1;
}

std::string predictSentence(int model, bool samplei, float temperature, int max_chars) {
  // if(typeof samplei === 'undefined') { samplei = false; }
  // if(typeof temperature === 'undefined') { temperature = 1.0; }

  // var G = new R.Graph(false);
  // var s = '';
  // var prev = {};
  std::string s = "";
  int max_chars_gen = max_chars; // ### TEMP ### //
  for (int i = 0; i < max_chars_gen; ++i) {
    // // RNN tick
    // var ix = s.length === 0 ? 0 : letterToIndex[s[s.length-1]];
    // var lh = forwardIndex(G, model, ix, prev);
    // prev = lh;

    // // sample predicted letter
    // logprobs = lh.o;
    // if(temperature !== 1.0 && samplei) {
    //   // scale log probabilities by temperature and renormalize
    //   // if temperature is high, logprobs will go towards zero
    //   // and the softmax outputs will be more diffuse. if temperature is
    //   // very low, the softmax outputs will be more peaky
    //   for(var q=0,nq=logprobs.w.length;q<nq;q++) {
    //     logprobs.w[q] /= temperature;
    //   }
    // }

    // probs = R.softmax(logprobs);
    // if(samplei) {
    //   var ix = R.samplei(probs.w);
    // } else {
    //   var ix = R.maxi(probs.w);
    // }

    // if(ix === 0) break; // END token predicted, break out
    // if(s.length > max_chars_gen) { break; } // something is wrong

    // var letter = indexToLetter[ix];
    // s += letter;
  }
  return s;
}

int main(int argc, char* argv[]) {
  // char_rnn();
  lstm();
  // using ops = concurrent::operation<float>;
  // timer::timer t;
  // std::unordered_map<int, var> bench;
  // std::unordered_map<int, var> results;
  // for (int i = 0; i < 23; ++i) {
  //   bench[i] = var({1, 1, 1000, 1000},  tf::initializer::randn<float>(0.01));
  // }
  // DBG(t.elapsed());
  // t.restart();
  // for (int i = 0; i < 20; ++i) {
  //   results[i] = bench[i] + bench[i+1] + bench[i+2];
  // }
  // DBG(t.elapsed());
  return 0;
}

void lstm() {
  // hyperparameters
  tf::param.hidden_size   = 100; // size of hidden layer of neurons
  tf::param.seq_length    = 53;  // number of steps to unroll the RNN for
  // tf::param.vocab_size    = 80;
  tf::param.max_epoch     = 2000;
  // tf::param.data_size     = 1808010;
  tf::param.learning_rate = 0.1;
  int seq_length = tf::param.seq_length;

  // data I/O
  std::ifstream data_file("data/ptb.valid.txt");
  // std::ifstream data_file("data/news.txt");
  // std::ifstream data_file("data/alpha.txt");
  std::stringstream buffer;
  buffer << data_file.rdbuf();
  data_file.close();
  auto data = buffer.str();
  std::set<char> chars_set;

  for (auto const& x : data) { chars_set.insert(x); }
  std::vector<char> chars(chars_set.begin(), chars_set.end());

  const int data_size  = data.size();  // 1808033, 1808010
  const int vocab_size = chars.size(); // 84, 80
  // CHECK_EQ(data_size, 1808010);
  // CHECK_EQ(vocab_size, 80);
  tf::param.data_size     = data_size;
  tf::param.vocab_size    = vocab_size;

  std::cout << "data has " << data_size << " characters, " << vocab_size << " unique.\n";

  std::map<char, int> char_to_ix;
  std::map<int, char> ix_to_char;

  int i = 0;
  for (auto it = chars.begin(); it != chars.end(); it++) {
    char_to_ix[*it] = i;
    ix_to_char[i] = *it;
    i++;
  }

  int p = 0;

  // loss at iteration 0
  float smooth_loss = -std::log(1.0 / vocab_size) * seq_length; // 110.770419971
  float loss;

  // auto hprev = new var({1, 1, hidden_size, 1}, tf::initializer::constant<float>(0.0));

  std::vector<int> inputs;
  std::vector<int> targets;

  timer::timer t;
  float costs = 0;
  tf::rnn<float> ptb(100, 80, 25);
  int mod = 100;
  for (auto n = 0; n < tf::param.max_epoch; ++n) {
    for (int p_idx = p; p_idx < p + tf::param.seq_length; p_idx++) {
      inputs.push_back(char_to_ix[data[p_idx]]);
      targets.push_back(char_to_ix[data[p_idx+1]]);
    }

    // if (n % mod == 0) {
    //   auto sample_ix = ptb.sample(inputs[0], 300);
    //   // std::string txt = predictSentence(0, true, 0.0, 300);
    //   std::string txt = "";
    //   for (int i = 0; i < sample_ix.size(); i++) { txt += ix_to_char[sample_ix[i]]; }
    //   std::cout << "----\n" << txt << "\n----";
    // }

    // ptb.session_rn
    ptb.session_run(inputs, targets);
    ptb.forward();
    loss = ptb.get_loss();
    costs += loss;
    ptb.backward();
    ptb.update();

    p += tf::param.seq_length; // move data pointer

    smooth_loss = smooth_loss * 0.999 + loss * 0.001;
    if (n % mod == 0) {
      std::string s_perplexity = "[perplexity: " + std::to_string(std::exp(costs/p)) + "]";
      std::string s_cps        = "[chars/sec = " + std::to_string((mod*tf::param.seq_length)/(t.elapsed())) + "]";
      std::string s_iteration  = "[iteration: "  + std::to_string(n)           + "]";
      std::string s_loss       = "[loss = "      + std::to_string(smooth_loss) + "]";
      std::cout << s_iteration << " " << s_loss << " " << s_perplexity << " " << s_cps << std::endl; // print progress
      t.restart();
    }

    // anneal learning rate
    // if (n >= 2000 && n%1000 == 0) {
    //   tf::param.learning_rate *= 0.95;
    // }

    inputs.clear();
    targets.clear();
  }

  return;
}

void char_rnn() {
  // data I/O
  std::ifstream data_file("data/news.txt");
  std::stringstream buffer;
  buffer << data_file.rdbuf();
  data_file.close();
  auto data = buffer.str();
  std::set<char> chars_set;

  for (auto const& x : data) { chars_set.insert(x); }
  std::vector<char> chars(chars_set.begin(), chars_set.end());

  const int data_size  = data.size();  // 1808033, 1808010
  const int vocab_size = chars.size(); // 84, 80
  CHECK_EQ(data_size, 1808010);
  CHECK_EQ(vocab_size, 80);

  std::cout << "data has " << data_size << " characters, " << vocab_size << " unique.\n";

  std::map<char, int> char_to_ix;
  std::map<int, char> ix_to_char;

  int i = 0;
  for (auto it = chars.begin(); it != chars.end(); it++) {
    char_to_ix[*it] = i;
    ix_to_char[i] = *it;
    i++;
  }

  // hyperparameters
  // const int hidden_size = 100; // size of hidden layer of neurons
  const int seq_length = 25;   // number of steps to unroll the RNN for
  const float learning_rate = 1e-1;

  int n = 0;
  int p = 0;

  // loss at iteration 0
  float smooth_loss = -std::log(1.0 / vocab_size) * seq_length; // 110.770419971
  float loss;

  // auto hprev = new var({1, 1, hidden_size, 1}, tf::initializer::constant<float>(0.0));

  std::vector<int> inputs;
  std::vector<int> targets;

  // while (n < 50) {
  while (true) {
    // prepare inputs (we're sweeping from left to right in steps seq_length long)
    if ( (p + seq_length + 1 >= data_size) || (n == 0) ) {
      hprev.zeros(); // reset RNN memory
      p = 0;          // go from start of data
    }

    for (int p_idx = p; p_idx < p + seq_length; p_idx++) {
      inputs.push_back(char_to_ix[data[p_idx]]);
      targets.push_back(char_to_ix[data[p_idx+1]]);
    }
    // sample from the moodel now and then
    if (n % 100 == 0) {
     std::vector<int> sample_ix = sample(hprev, inputs[0], 300); // 200
     std::string txt = "";
     for (int i = 0; i < sample_ix.size(); i++) { txt += ix_to_char[sample_ix[i]]; }
     std::cout << "----\n" << txt << "\n----";
    }

    // forward seq_length characters through the net and fetch gradient
    // loss = lossFun(inputs, targets, hprev);
    // loss = gradCheck(inputs, targets, hprev);
    // smooth_loss = smooth_loss * 0.999 + loss * 0.001;
    if (n % 100 == 0) {
      std::string s_iteration  = "[iteration: " + std::to_string(n)           + "]";
      std::string s_loss       = "[loss = "     + std::to_string(smooth_loss) + "]";
      std::cout << s_iteration << " " << s_loss << std::endl; // print progress
    }

    // perform parameter update with Adagrad
    ops::adagrad_update(Wxh, mWxh, learning_rate, 1e-8);
    ops::adagrad_update(Whh, mWhh, learning_rate, 1e-8);
    ops::adagrad_update(Why, mWhy, learning_rate, 1e-8);
    ops::adagrad_update(bh,  mbh,  learning_rate, 1e-8);
    ops::adagrad_update(by,  mby,  learning_rate, 1e-8);

    inputs.clear();
    targets.clear();
    p += seq_length; // move data pointer
    n += 1;          // iteration counter
    // concurrent::graph::reset_cache();
  }
}

