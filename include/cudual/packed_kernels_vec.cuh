// include/cudual/packed_kernels_vec.cuh
#pragma once
#include <array>
#include "cudual/cudual.cuh"
namespace cudadual {
template <typename I> CDUAL_HD inline I upidx(I i, I j, I N){ return i*N - (i*(i-1))/2 + (j - i); }
template <typename T, int N, int M, class F>
__global__ void vec_hessian_hyper_packed_kernel(F f, const T* __restrict__ X, int batch, T* __restrict__ Hpack){
  int b=blockIdx.x,i=blockIdx.y,j=threadIdx.x; if(b>=batch||i>=N||j>=N) return; if(j<i) return;
  T xh[N]; for(int k=0;k<N;++k) xh[k]=X[b*N+k];
  std::array< HyperDual<T>, N > x; #pragma unroll
  for (int k=0;k<N;++k) x[k]=make_hyper<T>(xh[k],k==i,k==j);
  auto y=f(x); size_t L=N*(N+1)/2, off=upidx<int>(i,j,N); #pragma unroll
  for (int m=0;m<M;++m) Hpack[b*(M*L)+m*L+off]=y[m].e12;
}
template <typename T, int N, int M, class F>
__global__ void vec_hessian_multi2_packed_kernel(F f, const T* __restrict__ X, int batch, T* __restrict__ y_out, T* __restrict__ J_out, T* __restrict__ Hpack){
  int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=batch) return; T xh[N]; for(int k=0;k<N;++k) xh[k]=X[b*N+k];
  std::array< MultiDual2<T,N>, N > x; #pragma unroll
  for (int i=0;i<N;++i) x[i]=make_variable2<T,N>(xh[i],i);
  auto y=f(x); if(y_out) for(int m=0;m<M;++m) y_out[b*M+m]=y[m].f; if(J_out) for(int m=0;m<M;++m) for(int n=0;n<N;++n) J_out[b*(M*N)+m*N+n]=y[m].g[n];
  size_t L=N*(N+1)/2; for(int m=0;m<M;++m) for(int i=0;i<N;++i) for(int j=i;j<N;++j){ size_t off=upidx<int>(i,j,N); Hpack[b*(M*L)+m*L+off]=y[m].H[i*N+j]; }
}
template <typename T, int N, int M, class F>
inline void launch_vec_hessian_packed(F f, const T* dX, int batch, T* dHpack, cudaStream_t stream=nullptr){
  dim3 grid(batch,N), block(N); vec_hessian_hyper_packed_kernel<T,N,M><<<grid,block,0,stream>>>(f,dX,batch,dHpack);
}
template <typename T, int N, int M, class F>
inline void launch_vec_grad_hess_packed(F f, const T* dX, int batch, T* dy, T* dJ, T* dHpack, cudaStream_t stream=nullptr){
  int threads=256, blocks=(batch+threads-1)/threads; vec_hessian_multi2_packed_kernel<T,N,M><<<blocks,threads,0,stream>>>(f,dX,batch,dy,dJ,dHpack);
}
} // namespace cudadual
