#include <cstdio>
#include <array>
#include "cudual/kernels.cuh"
using namespace cudadual;
struct F3 { template <class Num> CDUAL_HD Num operator()(const std::array<Num,3>& x) const { return sin(x[0]*x[1]) + exp(x[0]) * (x[2]*x[2]*x[2]); } };
int main(){ using T=double; constexpr int N=3; const int batch=1; T x[N]={1.0,0.5,2.0};
  T *dX=nullptr,*df=nullptr,*dg=nullptr,*dH=nullptr; cudaMalloc(&dX,sizeof(T)*batch*N);
  cudaMalloc(&df,sizeof(T)*batch); cudaMalloc(&dg,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*N*N);
  cudaMemcpy(dX,x,sizeof(T)*batch*N,cudaMemcpyHostToDevice);
  F3 f; launch_grad_hess<T,N>(f,dX,batch,df,dg,dH); cudaDeviceSynchronize();
  T f0=0; cudaMemcpy(&f0,df,sizeof(T),cudaMemcpyDeviceToHost);
  printf("f = %+.6f\n", f0);
  cudaFree(dX); cudaFree(df); cudaFree(dg); cudaFree(dH); return 0; }