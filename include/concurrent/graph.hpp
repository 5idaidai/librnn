#ifndef CONCURRENT_GRAPH_HPP
#define CONCURRENT_GRAPH_HPP

#include <functional>
#include <vector>
#include <unordered_map>

#include "concurrent/common.hpp"
#include "concurrent/var.hpp"

namespace concurrent {
template <typename R> class var;
// class any;

namespace graph {

void emplace_back(std::function<void()>&& f);
void push_del(void* f);
void backward();
void clear();
void reset_cache();
bool backprop_enabled();
void _set_backprop_enabled(bool value);
size_t size();

// template <typename T>
class Tape {
 public:
  // std::unordered_map<std::string, concurrent::var<float>*> model_cache;
  ~Tape();
  void backward();
  std::vector<void*> cache_del;
  std::vector<std::function<void()>> backprop;
  // type::any a("test");
  // void push_del(concurrent::var<T>* f);
  void del_cache();
};

// // template <typename Real>
// class graph {
//  public:
//   std::vector<std::function<void()>> backprop;


//   graph();
//   ~graph();

//   // graph(const graph<Real>& other);                   // copy constructor
//   // graph<Real>& operator=(const graph<Real>& other);  // copy assignment
//   // graph(const graph<Real>&& other);                  // move constructor
//   // graph<Real>& operator=(graph<Real>&& other);       // move assignment

//   void backward();
//   void emplace_back(std::function<void()>&& f);

//  protected:
// };

// template class Tape<float>;
// template class Tape<double>;

}  // namespace graph
}  // namespace concurrent

#endif  // CONCURRENT_GRAPH_HPP
