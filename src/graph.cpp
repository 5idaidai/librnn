#include "concurrent/graph.hpp"

namespace concurrent {
namespace graph {

bool _backprop_enabled = true;
// Tape<float> tape;
Tape tape;

void emplace_back(std::function<void()>&& f) { tape.backprop.emplace_back(f);  }
void push_del(void* f)                       { tape.cache_del.emplace_back(f); }
void backward()                              { tape.backward();                }
void clear()                                 { tape.backprop.clear();          }
void reset_cache()                           { tape.del_cache();               }
bool backprop_enabled()                      { return _backprop_enabled;       }
void _set_backprop_enabled(bool value)       { _backprop_enabled = value;      }
size_t size()                                { return tape.backprop.size();    }

// Tape
// template <typename T>
// void Tape<T>::backward () {
void Tape::backward() {
  // std::cout << " Backwards happening" << std::endl;
  for (auto f : concurrent::adaptors::reverse(backprop)) {
    f();
  }
  backprop.clear();
}

// template <typename T>
// void Tape<T>::push_del(concurrent::var<T>* f) { 
//   cache_del.emplace_back(f);
// }
Tape::~Tape() {
  // DBGLINE();
  del_cache();
  // DBGLINE();
  // DBGLINE();
  // for (int i = 0; i < cache_del.size(); ++i) {
  //   if (cache_del[i]) {
  //     // std::cout << "Deleting a " << static_cast<concurrent::var<float>*>(cache_del[i])->name << std::endl;
  //     // if (cache_del[i]->name = "float") {
  //     delete static_cast<concurrent::var<float>*>(cache_del[i]);
  //     // }
  //   }
  // }
}

// template <typename T>
void Tape::del_cache() {
  for (int i = 0; i < cache_del.size(); ++i) {
    if (static_cast<concurrent::var<float>*>(cache_del[i])) {
      // DBGLINE();
      delete static_cast<concurrent::var<float>*>(cache_del[i]);
      cache_del[i] = nullptr;
    }
  }
  cache_del.clear();
}

// template class Tape<float>;
// template class Tape<double>;

}  // namespace graph
}  // namespace concurrent
