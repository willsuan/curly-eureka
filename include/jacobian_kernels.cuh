#pragma once
#include "hyperdual.cuh"

// Generic KxM Jacobian kernel template

template<int K, int M, typename T, typename F>
__global__ void eval_jacobian_KM(const T* const* __restrict__ inputs,
                                 T* const* __restrict__ outF,
                                 T* const* __restrict__ outJ,
                                 int n, F f)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  MD<K,T> x[K];
  #pragma unroll
  for(int j=0;j<K;++j){ x[j] = make_seed<K,T>( inputs[j][i], j ); }
  MD<K,T> y[M];
  f.template operator()<K,T>(x, y);
  #pragma unroll
  for(int m=0;m<M;++m){ outF[m][i] = y[m].r; }
  #pragma unroll
  for(int m=0;m<M;++m){
    #pragma unroll
    for(int j=0;j<K;++j){ outJ[m*K + j][i] = y[m].e[j]; }
  }
}

// Example functor producing M=2 outputs from K=2 inputs
struct Fun2From2 {
  template<int K, typename T>
  __device__ void operator()(const MD<K,T>* x, MD<K,T>* y) const {
    y[0] = f_xy_md<K,T>(x[0], x[1]);
    y[1] = md_tanh<K,T>(x[0]) + md_hypot<K,T>(x[0], x[1]);
  }
};
