#include <cmath>
#include <cstdio>
#include <ctime>
#include <iomanip>

#include "concurrent/common.hpp"

namespace concurrent {

typedef std::mt19937 rng_t;

void EXPECT_REAL_EQ(float x, float y) {
  float maxXYOne = std::max({std::fabs(x) , std::fabs(y), 1.0f});
  // maxXYOne = std::max(maxXYOne, 1.0f);
  if (std::fabs(x - y) <= std::numeric_limits<float>::epsilon()*maxXYOne) {
    // all good
    std::cout << B_GREEN << std::setw(4) << x << " equals " << std::setw(4) << y << END_COLOR << std::endl;
  } else {
    std::cout << B_RED << std::setw(4) << x << " does not equal " << std::setw(4) << y << END_COLOR << std::endl;
  }
}

// void EXPECT_REAL_EQ(double x, double y) {
//   double maxXYOne = std::max({std::fabs(x) , std::fabs(y), 1.0});
//   // maxXYOne = std::max(maxXYOne, 1.0f);
//   if (std::fabs(x - y) <= std::numeric_limits<double>::epsilon()*maxXYOne) {
//     // all good
//   } else {
//     std::cout << B_RED << x << " does not equal " << y << END_COLOR << std::endl;
//   }
// }
// Make sure each thread can have different values.
// static boost::thread_specific_ptr<librnn> thread_instance_;

librnn& librnn::Get() {
  // if (!thread_instance_.get()) {
  //   thread_instance_.reset(new librnn());
  // }
  // return *(thread_instance_.get());
  auto x = new concurrent::librnn();
  return *x;
}

// random seeding
int64_t cluster_seedgen(void) {
  int64_t s, seed, pid;
  FILE* f = fopen("/dev/urandom", "rb");
  if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
    fclose(f);
    return seed;
  }

  LOG(INFO) << "System entropy source not available, using fallback algorithm to generate seed instead.";
  if (f)
    fclose(f);

  // pid = getpid();
  pid = 5; // random
  s = time(NULL);
  seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


#ifdef CPU_ONLY  // CPU-only librnn.

librnn::librnn() : random_generator_(), mode_(librnn::CPU), solver_count_(1), root_solver_(true) { }

librnn::~librnn() { }

void librnn::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void librnn::SetDevice(const int device_id) {
  NO_GPU;
}

void librnn::DeviceQuery() {
  NO_GPU;
}

bool librnn::CheckDevice(const int device_id) {
  NO_GPU;
  return false;
}

int librnn::FindDevice(const int start_id) {
  NO_GPU;
  return -1;
}

class librnn::RNG::Generator {
 public:
  Generator() : rng_(new concurrent::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new concurrent::rng_t(seed)) {}
  concurrent::rng_t* rng() { return rng_.get(); }
 private:
  std::shared_ptr<concurrent::rng_t> rng_;
};

librnn::RNG::RNG() : generator_(new Generator()) { }

librnn::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

librnn::RNG& librnn::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* librnn::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU librnn.

// librnn::librnn() : cublas_handle_(NULL), curand_generator_(NULL), random_generator_(), mode_(librnn::CPU), solver_count_(1), root_solver_(true) {
librnn::librnn() : random_generator_(), mode_(librnn::CPU), solver_count_(1), root_solver_(true) {
  // Try to create a cublas handler, and report an error if failed (but we will
  // keep the program running as one might just want to run CPU code).
  // if (cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS) {
  //   LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
  // }
  // Try to create a curand handler.
  // if (curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
  //     != CURAND_STATUS_SUCCESS ||
  //     curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
  //     != CURAND_STATUS_SUCCESS) {
  //   LOG(ERROR) << "Cannot create Curand generator. Curand won't be available.";
  // }
}

librnn::~librnn() {
  // if (cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
  // if (curand_generator_) {
  //   CURAND_CHECK(curandDestroyGenerator(curand_generator_));
  // }
}

void librnn::set_random_seed(const unsigned int seed) {
  // Curand seed
  // static bool g_curand_availability_logged = false;
  // if (Get().curand_generator_) {
  //   CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(),
  //       seed));
  //   CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
  // } else {
  //   if (!g_curand_availability_logged) {
  //       LOG(ERROR) <<
  //           "Curand not available. Skipping setting the curand seed.";
  //       g_curand_availability_logged = true;
  //   }
  // }
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void librnn::SetDevice(const int device_id) {
  int current_device = 0;
  // CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  // CUDA_CHECK(cudaSetDevice(device_id));
  // if (Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
  // if (Get().curand_generator_) {
  //   CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
  // }
  // CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_));
  // CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
  //     CURAND_RNG_PSEUDO_DEFAULT));
  // CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
  //     cluster_seedgen()));
}

void librnn::DeviceQuery() {
  // cudaDeviceProp prop;
  // int device;
  // if (cudaSuccess != cudaGetDevice(&device)) {
  //   printf("No cuda device present.\n");
  //   return;
  // }
  // CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  // LOG(INFO) << "Device id:                     " << device;
  // LOG(INFO) << "Major revision number:         " << prop.major;
  // LOG(INFO) << "Minor revision number:         " << prop.minor;
  // LOG(INFO) << "Name:                          " << prop.name;
  // LOG(INFO) << "Total global memory:           " << prop.totalGlobalMem;
  // LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
  // LOG(INFO) << "Total registers per block:     " << prop.regsPerBlock;
  // LOG(INFO) << "Warp size:                     " << prop.warpSize;
  // LOG(INFO) << "Maximum memory pitch:          " << prop.memPitch;
  // LOG(INFO) << "Maximum threads per block:     " << prop.maxThreadsPerBlock;
  // LOG(INFO) << "Maximum dimension of block:    " << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2];
  // LOG(INFO) << "Maximum dimension of grid:     " << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2];
  // LOG(INFO) << "Clock rate:                    " << prop.clockRate;
  // LOG(INFO) << "Total constant memory:         " << prop.totalConstMem;
  // LOG(INFO) << "Texture alignment:             " << prop.textureAlignment;
  // LOG(INFO) << "Concurrent copy and execution: " << (prop.deviceOverlap ? "Yes" : "No");
  // LOG(INFO) << "Number of multiprocessors:     " << prop.multiProcessorCount;
  // LOG(INFO) << "Kernel execution timeout:      " << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
  // return;
}

bool librnn::CheckDevice(const int device_id) {
  auto x = device_id;
  // This function checks the availability of GPU #device_id.
  // It attempts to create a context on the device by calling cudaFree(0).
  // cudaSetDevice() alone is not sufficient to check the availability.
  // It lazily records device_id, however, does not initialize a
  // context. So it does not know if the host thread has the permission to use
  // the device or not.
  //
  // In a shared environment where the devices are set to EXCLUSIVE_PROCESS
  // or EXCLUSIVE_THREAD mode, cudaSetDevice() returns cudaSuccess
  // even if the device is exclusively occupied by another process or thread.
  // Cuda operations that initialize the context are needed to check
  // the permission. cudaFree(0) is one of those with no side effect,
  // except the context initialization.
  bool r = false;
  // bool r = ((cudaSuccess == cudaSetDevice(device_id)) && (cudaSuccess == cudaFree(0)));
  // // reset any error that may have occurred.
  // cudaGetLastError();
  return r;
}

int librnn::FindDevice(const int start_id) {
  auto x = start_id;
  // This function finds the first available device by checking devices with
  // ordinal from start_id to the highest available value. In the
  // EXCLUSIVE_PROCESS or EXCLUSIVE_THREAD mode, if it succeeds, it also
  // claims the device due to the initialization of the context.
  // int count = 0;
  // CUDA_CHECK(cudaGetDeviceCount(&count));
  // for (int i = start_id; i < count; i++) {
  //   if (CheckDevice(i)) return i;
  // }
  return -1;
}

class librnn::RNG::Generator {
 public:
  Generator() : rng_(new concurrent::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new concurrent::rng_t(seed)) {}
  concurrent::rng_t* rng() { return rng_.get(); }
 private:
  std::shared_ptr<concurrent::rng_t> rng_;
};

librnn::RNG::RNG() : generator_(new Generator()) { }

librnn::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) { }

librnn::RNG& librnn::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* librnn::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#endif  // CPU_ONLY

}  // namespace concurrent
