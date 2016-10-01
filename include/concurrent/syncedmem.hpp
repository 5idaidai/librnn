#ifndef CONCURRENT_SYNCEDMEM_HPP_
#define CONCURRENT_SYNCEDMEM_HPP_

// synced_mem is mostly from Caffe.
// TODO: refactor for librnn
// COPYRIGHT

// All contributions by the University of California:
// Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
// All rights reserved.

// All other contributions:
// Copyright (c) 2014, 2015, the respective contributors
// All rights reserved.

// Caffe uses a shared copyright model: each contributor holds copyright over
// their contributions to Caffe. The project versioning records all such
// contribution and copyright details. If a contributor wants to further mark
// their specific copyright on a particular contribution, they should indicate
// their copyright solely in the commit message of the change when it is
// committed.

// LICENSE

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met: 

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer. 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution. 

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// CONTRIBUTION AGREEMENT

// By contributing to the BVLC/caffe repository through pull-request, comment,
// or otherwise, the contributor releases their content to the
// license and copyright terms herein.

#include <cstdlib>

#include "concurrent/common.hpp"

namespace concurrent {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void RNNMallocHost(void** ptr, size_t size, bool* use_cuda) {
// #ifndef CPU_ONLY
//   if (librnn::mode() == librnn::GPU) {
//     CUDA_CHECK(cudaMallocHost(ptr, size));
//     *use_cuda = true;
//     return;
//   }
// #endif
  *ptr = malloc(size);
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void RNNFreeHost(void* ptr, bool use_cuda) {
// #ifndef CPU_ONLY
//   if (use_cuda) {
//     CUDA_CHECK(cudaFreeHost(ptr));
//     return;
//   }
// #endif
  free(ptr);
}

// Manages memory allocation and synchronization between the host (CPU) and device (GPU).
// http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
class synced_memory {
 public:
  synced_memory()
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {
          // std::cout << "mem" << std::endl;
        }
  explicit synced_memory(size_t size)
      : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
        own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false),
        gpu_device_(-1) {
          // std::cout << "explicit mem" << std::endl;
          // DBG(size_);
        }
  ~synced_memory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  // const void* gpu_data();
  // void set_gpu_data(void* data);
  void* mut_data();
  // void* mutable_gpu_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
  SyncedHead head() { return head_; }
  size_t size() { return size_; }

 // private:
  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;
  void* gpu_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  int gpu_device_;

  synced_memory& operator=(const synced_memory&) = delete;
  synced_memory(const std::shared_ptr<synced_memory>& other) {
    cpu_ptr_ = other->cpu_ptr_;
    gpu_ptr_ = other->gpu_ptr_;
    size_ = other->size_;
    head_ = other->head_;
    own_cpu_data_ = other->own_cpu_data_;
    cpu_malloc_use_cuda_ = other->cpu_malloc_use_cuda_;
    own_gpu_data_ = other->own_gpu_data_;
    gpu_device_ = other->gpu_device_;
  }
};  // class synced_memory

}  // namespace concurrent

#endif  // CONCURRENT_SYNCEDMEM_HPP_
