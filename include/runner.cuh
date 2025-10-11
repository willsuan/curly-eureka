#pragma once
#include <array>
#include <vector>
#include <cuda_runtime.h>
#include "jacobian_kernels.cuh"

// Build device arrays of pointers from host arrays

template <typename T>
static inline T** make_device_ptr_array(const std::vector<T*>& host_ptrs){
  T** d=nullptr; cudaMalloc(&d, host_ptrs.size()*sizeof(T*));
  cudaMemcpy(d, host_ptrs.data(), host_ptrs.size()*sizeof(T*), cudaMemcpyHostToDevice);
  return d;
}

template <typename T, size_t N>
static inline T** make_device_ptr_array(const std::array<T*,N>& arr){
  T** d=nullptr; cudaMalloc(&d, N*sizeof(T*));
  cudaMemcpy(d, arr.data(), N*sizeof(T*), cudaMemcpyHostToDevice);
  return d;
}

// Generic Runner for arbitrary K, M, T, Functor

template <int K, int M, typename T, typename Fun>
struct Runner {
  static void run(const std::array<T*,K>& inputs,
                  const std::array<T*,M>& outF,
                  const std::array<T*,M*K>& outJ,
                  int N, cudaStream_t s){
    T** d_inputs = make_device_ptr_array(inputs);
    T** d_outF   = make_device_ptr_array(outF);
    T** d_outJ   = make_device_ptr_array(outJ);
    dim3 blk(256), grd((N+255)/256);
    Fun F{};
    eval_jacobian_KM<K,M,T,Fun><<<grd, blk, 0, s>>>(
      const_cast<const T* const*>(d_inputs), d_outF, d_outJ, N, F);
    cudaFree(d_inputs); cudaFree(d_outF); cudaFree(d_outJ);
  }
};
