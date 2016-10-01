#ifndef CONCURRENT_TIMER_HPP
#define CONCURRENT_TIMER_HPP

#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

namespace timer {

class timer {
  public:
    timer() : startTime(Clock::now()) {}
    
    // startTime = current clock time
    void restart() {
      startTime = Clock::now();
    }
    
    // Returns elapsed time since timer began
    float elapsed() const {
      return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now()-startTime).count()/1000.0;
    }

  private:
    std::chrono::high_resolution_clock::time_point startTime;
}; // timer

} // namespace timer

#endif  // CONCURRENT_TIMER_HPP
