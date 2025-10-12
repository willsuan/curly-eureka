// include/cudual/advanced_kernels.cuh
#pragma once
#include <array>
#include <algorithm>
#include "cudual/cudual.cuh"
namespace cudadual {
enum class DataLayout : int { AoS=0, SoA=1 };
template <typename T, int N> CDUAL_HD inline void load_aos(const T* __restrict__ X, int row, int row_stride, T (&xh)[N]){ #pragma unroll
  for (int i=0;i<N;++i) xh[i]=X[row*row_stride + i]; }
template <typename T, int N> CDUAL_HD inline void load_soa(const T* __restrict__ X, int row, int col_stride, T (&xh)[N]){ #pragma unroll
  for (int i=0;i<N;++i) xh[i]=X[i*col_stride + row]; }
template <typename T, int N, class F, DataLayout L>
__global__ void gradient_kernel_layout(F f, const T* __restrict__ X, int batch, int stride, T* __restrict__ f_out, T* __restrict__ grad_out){
  int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= batch) return; T xh[N];
  if constexpr (L==DataLayout::AoS) load_aos<T,N>(X, idx, stride, xh); else load_soa<T,N>(X, idx, stride, xh);
  std::array< MultiDual<T,N>, N > x; #pragma unroll
  for (int i=0;i<N;++i) x[i]=make_variable<T,N>(xh[i],i);
  MultiDual<T,N> y=f(x); if (f_out) f_out[idx]=y.f; #pragma unroll
  for (int i=0;i<N;++i) grad_out[idx*N+i]=y.d[i];
}
template <typename T, int N, class F, DataLayout L>
__global__ void hessian_kernel_layout(F f, const T* __restrict__ X, int batch, int stride, T* __restrict__ hess_out){
  int b=blockIdx.x,i=blockIdx.y,j=threadIdx.x; if(b>=batch||i>=N||j>=N) return; if(j<i) return; T xh[N];
  if constexpr (L==DataLayout::AoS) load_aos<T,N>(X, b, stride, xh); else load_soa<T,N>(X, b, stride, xh);
  std::array< HyperDual<T>, N > x; #pragma unroll
  for (int k=0;k<N;++k) x[k]=make_hyper<T>(xh[k],k==i,k==j);
  HyperDual<T> y=f(x); T h=y.e12; hess_out[b*N*N+i*N+j]=h; hess_out[b*N*N+j*N+i]=h;
}
template <typename T, int N, class F, DataLayout L>
__global__ void grad_hess_kernel_layout(F f, const T* __restrict__ X, int batch, int stride, T* __restrict__ f_out, T* __restrict__ grad_out, T* __restrict__ hess_out){
  int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx>=batch) return; T xh[N];
  if constexpr (L==DataLayout::AoS) load_aos<T,N>(X, idx, stride, xh); else load_soa<T,N>(X, idx, stride, xh);
  std::array< MultiDual2<T,N>, N > x; #pragma unroll
  for (int i=0;i<N;++i) x[i]=make_variable2<T,N>(xh[i],i);
  MultiDual2<T,N> y=f(x); if(f_out) f_out[idx]=y.f; if(grad_out) for(int i=0;i<N;++i) grad_out[idx*N+i]=y.g[i];
  for(int i=0;i<N;++i) for(int j=0;j<N;++j) hess_out[idx*N*N+i*N+j]=y.H[i*N+j];
}
template <typename T, int N, class F>
inline void launch_gradient_layout(F f, const T* dX, int batch, int stride, DataLayout layout, T* df, T* dgrad, cudaStream_t stream=nullptr){
  int threads=256, blocks=(batch+threads-1)/threads;
  if(layout==DataLayout::AoS) gradient_kernel_layout<T,N,F,DataLayout::AoS><<<blocks,threads,0,stream>>>(f,dX,batch,stride,df,dgrad);
  else                        gradient_kernel_layout<T,N,F,DataLayout::SoA><<<blocks,threads,0,stream>>>(f,dX,batch,stride,df,dgrad);
}
template <typename T, int N, class F>
inline void launch_hessian_layout(F f, const T* dX, int batch, int stride, DataLayout layout, T* dH, cudaStream_t stream=nullptr){
  dim3 grid(batch,N); dim3 block(N);
  if(layout==DataLayout::AoS) hessian_kernel_layout<T,N,F,DataLayout::AoS><<<grid,block,0,stream>>>(f,dX,batch,stride,dH);
  else                        hessian_kernel_layout<T,N,F,DataLayout::SoA><<<grid,block,0,stream>>>(f,dX,batch,stride,dH);
}
template <typename T, int N, class F>
inline void launch_grad_hess_layout(F f, const T* dX, int batch, int stride, DataLayout layout, T* df, T* dgrad, T* dH, cudaStream_t stream=nullptr){
  int threads=256, blocks=(batch+threads-1)/threads;
  if(layout==DataLayout::AoS) grad_hess_kernel_layout<T,N,F,DataLayout::AoS><<<blocks,threads,0,stream>>>(f,dX,batch,stride,df,dgrad,dH);
  else                        grad_hess_kernel_layout<T,N,F,DataLayout::SoA><<<blocks,threads,0,stream>>>(f,dX,batch,stride,df,dgrad,dH);
}
template <typename T, int N, class F>
inline void launch_hessian_streamed(F f, const T* dX, int batch, int stride, DataLayout layout, T* dH, int chunk, cudaStream_t stream=nullptr){
  int offset=0; while(offset<batch){ int bs=(chunk>0? std::min(chunk, batch-offset) : batch-offset); const T* ptr=(layout==DataLayout::AoS)?(dX+offset*stride):(dX+offset);
    launch_hessian_layout<T,N>(f, ptr, bs, stride, layout, dH + offset*N*N, stream); offset+=bs; }
}
} // namespace cudadual
