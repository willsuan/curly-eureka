#include <cstdio>
#include <array>
#include <vector>
#include "cudual/packed_kernels_vec.cuh"
#include "cudual/kernels.cuh"
using namespace cudadual;
template <int N, int M>
struct VecF { template <class Num> CDUAL_HD std::array<Num,M> operator()(const std::array<Num,N>& x) const {
  std::array<Num,M> y; y[0]=x[0]*x[0] + sin(x[1]*x[2]); if constexpr (M>1) y[1]=exp(x[0])*cosh(x[2]) + log1p(x[1]); return y; } };
int main(){ using T=double; constexpr int N=5, M=2; const int batch=3; std::vector<T> X(batch*N);
  for(int b=0;b<batch;++b) for(int i=0;i<N;++i) X[b*N+i]=T(0.2*(i+1)+0.01*b);
  const size_t L=N*(N+1)/2;
  T *dX=nullptr,*dHpack=nullptr,*dy=nullptr,*dJ=nullptr,*dHref=nullptr;
  cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&dHpack,sizeof(T)*batch*M*L);
  cudaMalloc(&dy,sizeof(T)*batch*M); cudaMalloc(&dJ,sizeof(T)*batch*M*N);
  cudaMalloc(&dHref,sizeof(T)*batch*M*N*N);
  cudaMemcpy(dX,X.data(),sizeof(T)*batch*N,cudaMemcpyHostToDevice);
  VecF<N,M> f; launch_vec_grad_hess_packed<T,N,M>(f,dX,batch,dy,dJ,dHpack); launch_jacobian_hessians<T,N,M>(f,dX,batch,dy,dJ,dHref); cudaDeviceSynchronize();
  cudaFree(dX); cudaFree(dHpack); cudaFree(dy); cudaFree(dJ); cudaFree(dHref); return 0; }