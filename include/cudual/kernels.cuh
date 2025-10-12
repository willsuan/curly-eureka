// include/cudual/kernels.cuh
#pragma once
#include <array>
#include "cudual/cudual.cuh"
namespace cudadual {
// Gradient
template <typename T, int N, class F>
__global__ void gradient_kernel(F f, const T* __restrict__ X, int batch, T* __restrict__ f_out, T* __restrict__ grad_out){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch) return;
  T xh[N]; #pragma unroll
  for (int i=0;i<N;++i) xh[i] = X[idx*N + i];
  std::array< MultiDual<T,N>, N > x; #pragma unroll
  for (int i=0;i<N;++i) x[i] = make_variable<T,N>(xh[i], i);
  MultiDual<T,N> y = f(x); if (f_out) f_out[idx] = y.f; #pragma unroll
  for (int i=0;i<N;++i) grad_out[idx*N + i] = y.d[i];
}
// Hessian via hyper-dual
template <typename T, int N, class F>
__global__ void hessian_kernel(F f, const T* __restrict__ X, int batch, T* __restrict__ hess_out){
  int b = blockIdx.x; int i = blockIdx.y; int j = threadIdx.x; if (b >= batch || i>=N || j>=N) return; if (j < i) return;
  T xh[N]; #pragma unroll
  for (int k=0;k<N;++k) xh[k] = X[b*N + k];
  std::array< HyperDual<T>, N > x; #pragma unroll
  for (int k=0;k<N;++k) x[k] = make_hyper<T>(xh[k], k==i, k==j);
  HyperDual<T> y = f(x); T h = y.e12; hess_out[b*N*N + i*N + j] = h; hess_out[b*N*N + j*N + i] = h;
}
// Grad + Hess via MultiDual2
template <typename T, int N, class F>
__global__ void grad_hess_kernel(F f, const T* __restrict__ X, int batch, T* __restrict__ f_out, T* __restrict__ grad_out, T* __restrict__ hess_out){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch) return;
  T xh[N]; #pragma unroll
  for (int i=0;i<N;++i) xh[i] = X[idx*N + i];
  std::array< MultiDual2<T,N>, N > x; #pragma unroll
  for (int i=0;i<N;++i) x[i] = make_variable2<T,N>(xh[i], i);
  MultiDual2<T,N> y = f(x); if (f_out) f_out[idx] = y.f; if (grad_out) for (int i=0;i<N;++i) grad_out[idx*N + i] = y.g[i];
  for (int i=0;i<N;++i) for (int j=0;j<N;++j) hess_out[idx*N*N + i*N + j] = y.H[i*N + j];
}
// Jacobian
template <typename T, int N, int M, class F>
__global__ void jacobian_kernel(F f, const T* __restrict__ X, int batch, T* __restrict__ y_out, T* __restrict__ J_out){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch) return;
  T xh[N]; #pragma unroll
  for (int i=0;i<N;++i) xh[i] = X[idx*N + i];
  std::array< MultiDual<T,N>, N > x; #pragma unroll
  for (int i=0;i<N;++i) x[i] = make_variable<T,N>(xh[i], i);
  auto y = f(x); for (int m=0;m<M;++m){ if (y_out) y_out[idx*M + m] = y[m].f; for (int n=0;n<N;++n) J_out[idx*(M*N) + m*N + n] = y[m].d[n]; }
}
// Jacobian + Hessians
template <typename T, int N, int M, class F>
__global__ void jacobian_hessians_kernel(F f, const T* __restrict__ X, int batch, T* __restrict__ y_out, T* __restrict__ J_out, T* __restrict__ H_out){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch) return;
  T xh[N]; #pragma unroll
  for (int i=0;i<N;++i) xh[i] = X[idx*N + i];
  std::array< MultiDual2<T,N>, N > x; #pragma unroll
  for (int i=0;i<N;++i) x[i] = make_variable2<T,N>(xh[i], i);
  auto y = f(x);
  for (int m=0;m<M;++m){ if (y_out) y_out[idx*M + m] = y[m].f; if (J_out) for (int n=0;n<N;++n) J_out[idx*(M*N) + m*N + n] = y[m].g[n];
    for (int i=0;i<N;++i) for (int j=0;j<N;++j) H_out[idx*(M*N*N) + m*(N*N) + i*N + j] = y[m].H[i*N + j]; }
}
// Host wrappers
template <typename T, int N, class F>
inline void launch_gradient(F f, const T* dX, int batch, T* df, T* dgrad, cudaStream_t stream=nullptr){
  int threads=256, blocks=(batch+threads-1)/threads;
  gradient_kernel<T,N><<<blocks,threads,0,stream>>>(f,dX,batch,df,dgrad);
}
template <typename T, int N, class F>
inline void launch_hessian(F f, const T* dX, int batch, T* dH, cudaStream_t stream=nullptr){
  dim3 grid(batch,N); dim3 block(N);
  hessian_kernel<T,N><<<grid,block,0,stream>>>(f,dX,batch,dH);
}
template <typename T, int N, class F>
inline void launch_grad_hess(F f, const T* dX, int batch, T* df, T* dgrad, T* dH, cudaStream_t stream=nullptr){
  int threads=256, blocks=(batch+threads-1)/threads;
  grad_hess_kernel<T,N><<<blocks,threads,0,stream>>>(f,dX,batch,df,dgrad,dH);
}
template <typename T, int N, int M, class F>
inline void launch_jacobian(F f, const T* dX, int batch, T* dy, T* dJ, cudaStream_t stream=nullptr){
  int threads=256, blocks=(batch+threads-1)/threads;
  jacobian_kernel<T,N,M><<<blocks,threads,0,stream>>>(f,dX,batch,dy,dJ);
}
template <typename T, int N, int M, class F>
inline void launch_jacobian_hessians(F f, const T* dX, int batch, T* dy, T* dJ, T* dH, cudaStream_t stream=nullptr){
  int threads=256, blocks=(batch+threads-1)/threads;
  jacobian_hessians_kernel<T,N,M><<<blocks,threads,0,stream>>>(f,dX,batch,dy,dJ,dH);
}
} // namespace cudadual
