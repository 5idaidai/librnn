#include "concurrent/common.hpp"
#include "concurrent/syncedmem.hpp"
#include "concurrent/math.hpp"

namespace concurrent {

synced_memory::~synced_memory() {
  if (cpu_ptr_ && own_cpu_data_) {
    RNNFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }

// #ifndef CPU_ONLY
//   if (gpu_ptr_ && own_gpu_data_) {
//     int initial_device;
//     cudaGetDevice(&initial_device);
//     if (gpu_device_ != -1) {
//       CUDA_CHECK(cudaSetDevice(gpu_device_));
//     }
//     CUDA_CHECK(cudaFree(gpu_ptr_));
//     cudaSetDevice(initial_device);
//   }
// #endif  // CPU_ONLY
}

inline void synced_memory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    RNNMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    rnn_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
// #ifndef CPU_ONLY
//     if (cpu_ptr_ == NULL) {
//       RNNMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
//       own_cpu_data_ = true;
//     }
//     rnn_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
//     head_ = SYNCED;
// #else
//     NO_GPU;
// #endif
//     break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

// inline void synced_memory::to_gpu() {
// #ifndef CPU_ONLY
//   switch (head_) {
//   case UNINITIALIZED:
//     CUDA_CHECK(cudaGetDevice(&gpu_device_));
//     CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
//     rnn_gpu_memset(size_, 0, gpu_ptr_);
//     head_ = HEAD_AT_GPU;
//     own_gpu_data_ = true;
//     break;
//   case HEAD_AT_CPU:
//     if (gpu_ptr_ == NULL) {
//       CUDA_CHECK(cudaGetDevice(&gpu_device_));
//       CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
//       own_gpu_data_ = true;
//     }
//     rnn_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
//     head_ = SYNCED;
//     break;
//   case HEAD_AT_GPU:
//   case SYNCED:
//     break;
//   }
// #else
//   NO_GPU;
// #endif
// }

const void* synced_memory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void synced_memory::set_cpu_data(void* data) {
  auto x = data;
  // CHECK(data);
  // if (own_cpu_data_) {
  //   RNNFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  // }
  // cpu_ptr_ = data;
  // head_ = HEAD_AT_CPU;
  // own_cpu_data_ = false;
}

// const void* synced_memory::gpu_data() {
// #ifndef CPU_ONLY
//   to_gpu();
//   return (const void*)gpu_ptr_;
// #else
//   NO_GPU;
//   return NULL;
// #endif
// }

// void synced_memory::set_gpu_data(void* data) {
// #ifndef CPU_ONLY
//   CHECK(data);
//   if (own_gpu_data_) {
//     int initial_device;
//     cudaGetDevice(&initial_device);
//     if (gpu_device_ != -1) {
//       CUDA_CHECK(cudaSetDevice(gpu_device_));
//     }
//     CUDA_CHECK(cudaFree(gpu_ptr_));
//     cudaSetDevice(initial_device);
//   }
//   gpu_ptr_ = data;
//   head_ = HEAD_AT_GPU;
//   own_gpu_data_ = false;
// #else
//   NO_GPU;
// #endif
// }

void* synced_memory::mut_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

// void* synced_memory::mutable_gpu_data() {
// #ifndef CPU_ONLY
//   to_gpu();
//   head_ = HEAD_AT_GPU;
//   return gpu_ptr_;
// #else
//   NO_GPU;
//   return NULL;
// #endif
// }

// #ifndef CPU_ONLY
// void synced_memory::async_gpu_push(const cudaStream_t& stream) {
//   CHECK(head_ == HEAD_AT_CPU);
//   if (gpu_ptr_ == NULL) {
//     CUDA_CHECK(cudaGetDevice(&gpu_device_));
//     CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
//     own_gpu_data_ = true;
//   }
//   const cudaMemcpyKind put = cudaMemcpyHostToDevice;
//   CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
//   // Assume caller will synchronize on the stream before use
//   head_ = SYNCED;
// }
// #endif

}  // namespace concurrent

