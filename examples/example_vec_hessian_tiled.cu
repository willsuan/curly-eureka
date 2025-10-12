#include <cstdio>
#include <array>
#include <vector>
#include "cudual/tiled_kernels_vec.cuh"
using namespace cudadual;
template <int N, int M>
struct VecF { template <class Num> CDUAL_HD std::array<Num,M> operator()(const std::array<Num,N>& x) const {
  std::array<Num,M> y; y[0]=sin(x[0]*x[1]) + exp(Num(0.2)*x[0]) * log1p(x[2]*x[2]);
  if constexpr (M>1){ Num s=Num(0); for(int i=0;i<N;++i) s=s+x[i]*x[i]; y[1]=s + Num(0.3)*sin(x[0]*x[2]); } return y; } };
int main(){ using T=double; constexpr int N=8,M=2,K=4; const int batch=3; std::vector<T> X(batch*N);
  for(int b=0;b<batch;++b) for(int i=0;i<N;++i) X[b*N+i]=T(0.1*(i+1)+0.01*b);
  T *dX=nullptr,*dH=nullptr; cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*M*N*N);
  cudaMemcpy(dX,X.data(),sizeof(T)*batch*N,cudaMemcpyHostToDevice);
  VecF<N,M> f; launch_vec_hessians_tiled<T,N,K,M>(f,dX,batch,/*stride=*/N,DataLayout::AoS,dH); cudaDeviceSynchronize();
  std::vector<T> H(batch*M*N*N); cudaMemcpy(H.data(),dH,sizeof(T)*batch*M*N*N,cudaMemcpyDeviceToHost);
  for(int b=0;b<batch;++b){ for(int m=0;m<M;++m){ std::printf("Sample %d, out %d H:\n",b,m);
    for(int i=0;i<N;++i){ std::printf("  "); for(int j=0;j<N;++j) std::printf(" %+.6f", H[b*(M*N*N)+m*(N*N)+i*N+j]); std::printf("\n"); } } }
  cudaFree(dX); cudaFree(dH); return 0; }