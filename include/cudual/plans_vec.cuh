// include/cudual/plans_vec.cuh
#pragma once
#include <cuda_runtime.h>
#include "cudual/tiled_kernels_vec.cuh"
namespace cudadual {
template <typename T, int N, int K, int M>
struct VecHessianPlan {
  int Nvars=N, Ktile=K, Mout=M;
  DataLayout layout=DataLayout::AoS;
  int numTiles=(N+K-1)/K;
  int totalPairs=(numTiles*(numTiles-1))/2;
  int sample_chunk=0, pair_chunk=0;
  int2* d_pairs=nullptr;
  cudaError_t init_pairs(){
    if(d_pairs||totalPairs<=0) return cudaSuccess;
    int2* h=(int2*)malloc(sizeof(int2)*totalPairs); int idx=0;
    for(int ta=0;ta<numTiles;++ta) for(int tb=ta+1;tb<numTiles;++tb){ h[idx].x=ta; h[idx].y=tb; ++idx; }
    cudaError_t st=cudaMalloc((void**)&d_pairs,sizeof(int2)*totalPairs);
    if(st!=cudaSuccess){ free(h); return st; }
    st=cudaMemcpy(d_pairs,h,sizeof(int2)*totalPairs,cudaMemcpyHostToDevice); free(h); return st;
  }
  void destroy(){ if(d_pairs){ cudaFree(d_pairs); d_pairs=nullptr; } }
};
template <typename T, int N, int K, int M, class F, DataLayout L>
__global__ void vec_hess_offdiag_pairs_list_kernel(F f,const T* __restrict__ X,int batch,int stride,const int2* __restrict__ pairs,int pair_start,int pair_count,T* __restrict__ H){
  int b=blockIdx.x; int p=pair_start+blockIdx.y; if(b>=batch||blockIdx.y>=pair_count) return;
  int2 pr=pairs[p]; int tA0=pr.x*K, tB0=pr.y*K; if(tA0>=N||tB0>=N) return;
  int t=threadIdx.x; int i=tA0+(t/K); int j=tB0+(t%K); if(i>=N||j>=N) return;
  T xh[N]; if constexpr (L==DataLayout::AoS) load_aos<T,N>(X,b,stride,xh); else load_soa<T,N>(X,b,stride,xh);
  std::array< HyperDual<T>, N > x; #pragma unroll
  for (int u=0;u<N;++u) x[u]=make_hyper<T>(xh[u],u==i,u==j);
  auto y=f(x); #pragma unroll
  for (int m=0;m<M;++m){ T h=y[m].e12; H[H_offset(b,m,i,j,N,M)]=h; H[H_offset(b,m,j,i,N,M)]=h; }
}
template <typename T, int N, int K, int M, class F>
inline void execute_vec_hessians_plan(const VecHessianPlan<T,N,K,M>& plan, F f, const T* dX, int batch, int stride, T* dH, cudaStream_t streams[2]){
  const int threads=128; const int numTiles=plan.numTiles; const int total_pairs=plan.totalPairs;
  const int sample_chunk=(plan.sample_chunk>0? plan.sample_chunk : batch); const int pair_chunk=(plan.pair_chunk>0? plan.pair_chunk : total_pairs);
  cudaStream_t s0=streams?streams[0]:nullptr, s1=streams?streams[1]:nullptr;
  int samp_off=0; while(samp_off<batch){ int bs= (sample_chunk<batch? min(sample_chunk, batch-samp_off) : batch-samp_off);
    int blocksY=(bs+threads-1)/threads;
    if(plan.layout==DataLayout::AoS) vec_hess_diag_blocks_kernel<T,N,K,M,F,DataLayout::AoS><<<dim3(numTiles,blocksY),threads,0,s0>>>(f,dX+samp_off*stride,bs,stride,dH + size_t(samp_off)*M*N*N);
    else                             vec_hess_diag_blocks_kernel<T,N,K,M,F,DataLayout::SoA><<<dim3(numTiles,blocksY),threads,0,s0>>>(f,dX+samp_off,bs,stride,dH + size_t(samp_off)*M*N*N);
    int pair_off=0; while(pair_off<total_pairs){ int pc=(pair_chunk<total_pairs? min(pair_chunk,total_pairs-pair_off) : total_pairs-pair_off);
      dim3 grid(bs,pc), block(K*K);
      if(plan.d_pairs){
        if(plan.layout==DataLayout::AoS)
          vec_hess_offdiag_pairs_list_kernel<T,N,K,M,F,DataLayout::AoS><<<grid,block,0,s1>>>(f,dX+samp_off*stride,bs,stride,plan.d_pairs,pair_off,pc,dH + size_t(samp_off)*M*N*N);
        else
          vec_hess_offdiag_pairs_list_kernel<T,N,K,M,F,DataLayout::SoA><<<grid,block,0,s1>>>(f,dX+samp_off,bs,stride,plan.d_pairs,pair_off,pc,dH + size_t(samp_off)*M*N*N);
      } else {
        if(plan.layout==DataLayout::AoS)
          vec_hess_offdiag_pairs_kernel<T,N,K,M,F,DataLayout::AoS><<<grid,block,0,s1>>>(f,dX+samp_off*stride,bs,stride,numTiles,pair_off,pc,dH + size_t(samp_off)*M*N*N);
        else
          vec_hess_offdiag_pairs_kernel<T,N,K,M,F,DataLayout::SoA><<<grid,block,0,s1>>>(f,dX+samp_off,bs,stride,numTiles,pair_off,pc,dH + size_t(samp_off)*M*N*N);
      }
      pair_off+=pc;
    }
    samp_off+=bs;
  }
}
} // namespace cudadual
