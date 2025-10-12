#include <cstdio>
#include <array>
#include <vector>
#include "cudual/tiled_kernels_vec.cuh"
#include "cudual/plans_vec.cuh"
using namespace cudadual;
template <int N, int M>
struct VecF { template <class Num> CDUAL_HD std::array<Num,M> operator()(const std::array<Num,N>& x) const {
  std::array<Num,M> y; y[0]=sin(x[0]*x[1]) + exp(Num(0.1)*x[2]) + log1p(x[0]*x[2]);
  if constexpr (M>1) y[1]=exp2(x[1]) + log2(x[0]*x[0] + Num(1)); return y; } };
int main(){ using T=double; constexpr int N=12, M=2, K=4; const int batch=2048; std::vector<T> X(batch*N);
  for(int b=0;b<batch;++b) for(int i=0;i<N;++i) X[b*N+i]=T(0.01*b+0.1*i);
  T *dX=nullptr,*dH=nullptr; cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*M*N*N);
  cudaMemcpy(dX,X.data(),sizeof(T)*batch*N,cudaMemcpyHostToDevice);
  VecHessianPlan<T,N,K,M> plan; plan.layout=DataLayout::AoS; plan.sample_chunk=512; plan.pair_chunk=64; plan.init_pairs();
  cudaStream_t s0,s1; cudaStreamCreate(&s0); cudaStreamCreate(&s1); cudaStream_t ss[2]={s0,s1};
  VecF<N,M> f; execute_vec_hessians_plan(plan,f,dX,batch,/*stride=*/N,dH,ss); cudaDeviceSynchronize();
  plan.destroy(); cudaStreamDestroy(s0); cudaStreamDestroy(s1);
  cudaFree(dX); cudaFree(dH); printf("plan example completed.\n"); return 0; }