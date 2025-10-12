// include/cudual/tiled_kernels_vec.cuh
#pragma once
#include <array>
#include <algorithm>
#include "cudual/cudual.cuh"
#include "cudual/advanced_kernels.cuh"
namespace cudadual {
CDUAL_HD inline size_t H_offset(int b, int m, int i, int j, int N, int M){
  return (size_t)b*(M*N*N) + (size_t)m*(N*N) + (size_t)i*N + j;
}
template <typename T, int N, int K, int M, class F, DataLayout L>
__global__ void vec_hess_diag_blocks_kernel(F f, const T* __restrict__ X, int batch, int stride, T* __restrict__ H){
  int tile=blockIdx.x; int idx=blockIdx.y*blockDim.x+threadIdx.x; if(idx>=batch) return; int t0=tile*K; if(t0>=N) return;
  T xh[N]; if constexpr (L==DataLayout::AoS) load_aos<T,N>(X, idx, stride, xh); else load_soa<T,N>(X, idx, stride, xh);
  std::array< MultiDual2<T,K>, N > x; #pragma unroll
  for (int i=0;i<N;++i) x[i]=MultiDual2<T,K>(xh[i]);
  #pragma unroll
  for (int k=0;k<K;++k){ int i=t0+k; if(i<N) x[i].g[k]=T(1); }
  auto y=f(x); // array<MultiDual2<T,K>,M>
  #pragma unroll
  for (int a=0;a<K;++a){ int i=t0+a; if(i>=N) break; #pragma unroll
  for (int b=0;b<K;++b){ int j=t0+b; if(j>=N) break;
    #pragma unroll
  for (int m=0;m<M;++m){ H[ H_offset(idx,m,i,j,N,M) ] = y[m].H[a*K+b]; } } }
}
template <typename T, int N, int K, int M, class F, DataLayout L>
__global__ void vec_hess_offdiag_tiles_kernel(F f, const T* __restrict__ X, int batch, int stride, int tileA, int tileB, T* __restrict__ H){
  int idx=blockIdx.x; if(idx>=batch) return; int tA0=tileA*K, tB0=tileB*K; if(tA0>=N||tB0>=N) return;
  int t=threadIdx.x; int i=tA0+(t/K); int j=tB0+(t%K); if(i>=N||j>=N) return;
  T xh[N]; if constexpr (L==DataLayout::AoS) load_aos<T,N>(X, idx, stride, xh); else load_soa<T,N>(X, idx, stride, xh);
  std::array< HyperDual<T>, N > x; #pragma unroll
  for (int u=0;u<N;++u) x[u]=make_hyper<T>(xh[u], u==i, u==j);
  auto y=f(x);
  #pragma unroll
  for (int m=0;m<M;++m){ T h=y[m].e12; H[H_offset(idx,m,i,j,N,M)]=h; H[H_offset(idx,m,j,i,N,M)]=h; }
}
CDUAL_HD inline void tilepair_from_linear(int p, int T, int& ta, int& tb){ ta=0; int rowlen=T-1; int rem=p; while(rem>=rowlen){ rem-=rowlen; ta++; rowlen--; } tb=ta+1+rem; }
template <typename T, int N, int K, int M, class F, DataLayout L>
__global__ void vec_hess_offdiag_pairs_kernel(F f, const T* __restrict__ X, int batch, int stride, int numTiles, int pair_start, int pair_count, T* __restrict__ H){
  int b=blockIdx.x; int p=pair_start+blockIdx.y; if(b>=batch||blockIdx.y>=pair_count) return;
  int ta,tb; tilepair_from_linear(p,numTiles,ta,tb); int tA0=ta*K, tB0=tb*K; if(tA0>=N||tB0>=N) return;
  int t=threadIdx.x; int i=tA0+(t/K); int j=tB0+(t%K); if(i>=N||j>=N) return;
  T xh[N]; if constexpr (L==DataLayout::AoS) load_aos<T,N>(X, b, stride, xh); else load_soa<T,N>(X, b, stride, xh);
  std::array< HyperDual<T>, N > x; #pragma unroll
  for (int u=0;u<N;++u) x[u]=make_hyper<T>(xh[u], u==i, u==j);
  auto y=f(x); #pragma unroll
  for (int m=0;m<M;++m){ T h=y[m].e12; H[H_offset(b,m,i,j,N,M)]=h; H[H_offset(b,m,j,i,N,M)]=h; }
}
template <typename T, int N, int K, int M, class F>
inline void launch_vec_hessians_tiled(F f, const T* dX, int batch, int stride, DataLayout layout, T* dH, cudaStream_t stream=nullptr){
  const int threads=128; const int blocksY=(batch+threads-1)/threads; const int numTiles=(N+K-1)/K;
  if(layout==DataLayout::AoS) vec_hess_diag_blocks_kernel<T,N,K,M,F,DataLayout::AoS><<<dim3(numTiles,blocksY),threads,0,stream>>>(f,dX,batch,stride,dH);
  else                        vec_hess_diag_blocks_kernel<T,N,K,M,F,DataLayout::SoA><<<dim3(numTiles,blocksY),threads,0,stream>>>(f,dX,batch,stride,dH);
  for(int ta=0; ta<numTiles; ++ta) for(int tb=ta+1; tb<numTiles; ++tb){
    if(layout==DataLayout::AoS) vec_hess_offdiag_tiles_kernel<T,N,K,M,F,DataLayout::AoS><<<batch,K*K,0,stream>>>(f,dX,batch,stride,ta,tb,dH);
    else                        vec_hess_offdiag_tiles_kernel<T,N,K,M,F,DataLayout::SoA><<<batch,K*K,0,stream>>>(f,dX,batch,stride,ta,tb,dH);
  }
}
template <typename T, int N, int K, int M, class F>
inline void launch_vec_hessians_tiled_streamed(F f, const T* dX, int batch, int stride, DataLayout layout, T* dH, int sample_chunk, int pair_chunk, cudaStream_t stream=nullptr){
  const int threads=128; const int numTiles=(N+K-1)/K; const int total_pairs=(numTiles*(numTiles-1))/2;
  int samp_off=0; while(samp_off<batch){ int bs=(sample_chunk>0? std::min(sample_chunk, batch-samp_off) : batch-samp_off); int blocksY=(bs+threads-1)/threads;
    if(layout==DataLayout::AoS) vec_hess_diag_blocks_kernel<T,N,K,M,F,DataLayout::AoS><<<dim3(numTiles,blocksY),threads,0,stream>>>(f,dX+samp_off*stride,bs,stride,dH + size_t(samp_off)*M*N*N);
    else                        vec_hess_diag_blocks_kernel<T,N,K,M,F,DataLayout::SoA><<<dim3(numTiles,blocksY),threads,0,stream>>>(f,dX+samp_off,bs,stride,dH + size_t(samp_off)*M*N*N);
    int pair_off=0; while(pair_off<total_pairs){ int pc=(pair_chunk>0? std::min(pair_chunk,total_pairs-pair_off) : total_pairs-pair_off); dim3 grid(bs,pc), block(K*K);
      if(layout==DataLayout::AoS) vec_hess_offdiag_pairs_kernel<T,N,K,M,F,DataLayout::AoS><<<grid,block,0,stream>>>(f,dX+samp_off*stride,bs,stride,numTiles,pair_off,pc,dH + size_t(samp_off)*M*N*N);
      else                        vec_hess_offdiag_pairs_kernel<T,N,K,M,F,DataLayout::SoA><<<grid,block,0,stream>>>(f,dX+samp_off,bs,stride,numTiles,pair_off,pc,dH + size_t(samp_off)*M*N*N);
      pair_off+=pc; } samp_off+=bs; }
}
inline size_t vec_hessian_tiled_workspace_bytes(int, int, int, int){ return 0; }
} // namespace cudadual
