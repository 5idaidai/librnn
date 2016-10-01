#ifndef CONCURRENT_COMMON_HPP_
#define CONCURRENT_COMMON_HPP_

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
// #include "librnn/util/device_alternate.hpp"

#define B_RED     "\033[1;31m"
#define B_GREEN   "\033[1;32m"
#define B_YELLOW  "\033[1;33m"
#define B_BLUE    "\033[1;34m"
#define B_PURPLE  "\033[1;35m"
#define END_COLOR "\033[0m"

#define PRECISE(x) std::cout << #x << " = " << std::setprecision(20) << x << std::setprecision(6) << std::endl;
#define DBG(x)    std::cout << "DBG: " << #x << " = " << x << "\n";
#define INFO(x)   std::cout << "DBG: " << #x << " = \n";
#define LOGGER(x) std::cout << "LOG: " << x << " : " << __LINE__ << "\n";
#define DBGLINE() std::cout << B_PURPLE << "DBG: " << __FILE__ << " -- "<< __LINE__ << END_COLOR << "\n";

// Convert macro to string
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  template class classname<float>;   \
  template class classname<double>

#define LOG_FATAL(x)   std::cout << "[" << __FILE__ << "] [" << __LINE__ << "] [" << __func__ << "\033[1;31mbold red text\033[0m\n";
// #define LOG_INFO    std::cout << "\033[1;31mbold red text\033[0m\n";
// #define LOG_WARNING std::cout << "\033[1;31mbold red text\033[0m\n";
// #define LOG_ERROR   std::cout << "\033[1;31mbold red text\033[0m\n";

#define LOG(FATAL) std::cout << "FATAL"

#define CHECK_OP(val1, val2, op)                                   \
        if (!(val1 op val2))                                       \
        std::cout << "[" << __FILE__ << "] [" << __LINE__ << "] "  \
                  << "Check failed: " #val1 " " #op " " #val2 "\n" \

#define CHECK(condition)                                              \
  if (!(condition))                                                   \
    std::cout << B_RED << "[" << __FILE__ << "] [" << __LINE__ << "]" \
              << " Check failed: " << END_COLOR << #condition "##\n"  \

#define CHECK_EQ(val1, val2) CHECK_OP(val1, val2, ==)
#define CHECK_NE(val1, val2) CHECK_OP(val1, val2, !=)
#define CHECK_LE(val1, val2) CHECK_OP(val1, val2, <=)
#define CHECK_LT(val1, val2) CHECK_OP(val1, val2, <)
#define CHECK_GE(val1, val2) CHECK_OP(val1, val2, >=)
#define CHECK_GT(val1, val2) CHECK_OP(val1, val2, >)

// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
// #define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"
// #define NOT_IMPLEMENTED LOG_FATAL("x") << "Not Implemented Yet"
#define NOT_IMPLEMENTED std::cout << "NOT_IMPLEMENTED" << std::endl;

namespace concurrent {

// template <typename T>
void EXPECT_REAL_EQ(float x, float y);
// static void EXPECT_REAL_EQ(double x, double y);

// http://stackoverflow.com/questions/8542591/c11-reverse-range-based-for-loop
namespace adaptors {
  
  template <typename T>
  struct reversion_wrapper { T& iterable; };

  template <typename T>
  auto begin(reversion_wrapper<T> w) { return std::rbegin(w.iterable); }

  template <typename T>
  auto end(reversion_wrapper<T> w) { return std::rend(w.iterable); }

  template <typename T>
  reversion_wrapper<T> reverse(T&& iterable) { return { iterable }; }

}  // namespace adaptors

// Common functions and classes from std that librnn often uses.
using std::string;

// A singleton class to hold common librnn stuff, such as the handler that
// librnn is going to use for cublas, curand, etc.
class librnn {
 public:
  ~librnn();

  // Thread local context for librnn. Moved to common.cpp instead of
  // including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010)
  // on OSX. Also fails on Linux with CUDA 7.0.18.
  static librnn& Get();

  enum Brew { CPU, GPU };

  // This random number generator facade hides boost and CUDA rng
  // implementation from one another (for cross-platform compatibility).
  class RNG {
   public:
    RNG();
    explicit RNG(unsigned int seed);
    explicit RNG(const RNG&);
    RNG& operator=(const RNG&);
    void* generator();
   private:
    class Generator;
    std::shared_ptr<Generator> generator_;
  };

  // Getters for boost rng, curand, and cublas handles
  inline static RNG& rng_stream() {
    if (!Get().random_generator_) {
      Get().random_generator_.reset(new RNG());
    }
    return *(Get().random_generator_);
  }
// #ifndef CPU_ONLY
//   inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
//   inline static curandGenerator_t curand_generator() { return Get().curand_generator_; }
// #endif

  // Returns the mode: running on CPU or GPU.
  inline static Brew mode() { return Get().mode_; }
  // The setters for the variables
  // Sets the mode. It is recommended that you don't change the mode halfway
  // into the program since that may cause allocation of pinned memory being
  // freed in a non-pinned way, which may cause problems - I haven't verified
  // it personally but better to note it here in the header file.
  inline static void set_mode(Brew mode) { Get().mode_ = mode; }
  // Sets the random seed of both boost and curand
  static void set_random_seed(const unsigned int seed);
  // Sets the device. Since we have cublas and curand stuff, set device also
  // requires us to reset those values.
  static void SetDevice(const int device_id);
  // Prints the current GPU status.
  static void DeviceQuery();
  // Check if specified device is available
  static bool CheckDevice(const int device_id);
  // Search from start_id to the highest possible device ordinal,
  // return the ordinal of the first available device.
  static int FindDevice(const int start_id = 0);
  // Parallel training info
  inline static int solver_count() { return Get().solver_count_; }
  inline static void set_solver_count(int val) { Get().solver_count_ = val; }
  inline static bool root_solver() { return Get().root_solver_; }
  inline static void set_root_solver(bool val) { Get().root_solver_ = val; }

 protected:
// #ifndef CPU_ONLY
//   cublasHandle_t cublas_handle_;
//   curandGenerator_t curand_generator_;
// #endif
  std::shared_ptr<RNG> random_generator_;

  Brew mode_;
  int solver_count_;
  bool root_solver_;

 private:
  // The private constructor to avoid duplicate instantiation.
  librnn();

  DISABLE_COPY_AND_ASSIGN(librnn);
};

}  // namespace concurrent

#endif  // CONCURRENT_COMMON_HPP_
