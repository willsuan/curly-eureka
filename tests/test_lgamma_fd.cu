#include <cstdio>
#include <array>
#include <vector>
#include <cmath>
#include "cudual/kernels.cuh"
#include "cudual/cudual.cuh"
using namespace cudadual;
struct LG { template <class Num> CDUAL_HD Num operator()(const std::array<Num,1>& x) const { return lgamma(x[0]); } double host_eval(double v) const { return ::lgamma(v); } };
int main(){ using T=double; constexpr int N=1; const int batch=1; T x0=3.25;
  T *dX=nullptr,*df=nullptr,*dg=nullptr,*dH=nullptr;
  cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&df,sizeof(T)*batch);
  cudaMalloc(&dg,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*N*N);
  cudaMemcpy(dX,&x0,sizeof(T),cudaMemcpyHostToDevice);
  LG f; launch_grad_hess<T,N>(f,dX,batch,df,dg,dH); cudaDeviceSynchronize();
  T f_ad=0,g_ad=0,H_ad=0; cudaMemcpy(&f_ad,df,sizeof(T),cudaMemcpyDeviceToHost);
  cudaMemcpy(&g_ad,dg,sizeof(T),cudaMemcpyDeviceToHost); cudaMemcpy(&H_ad,dH,sizeof(T),cudaMemcpyDeviceToHost);
  auto feval=[&](T v){ return f.host_eval(v); }; T h=1e-6; T g_fd=(feval(x0+h)-feval(x0-h))/(2*h); T H_fd=(feval(x0+h)-2*feval(x0)+feval(x0-h))/(h*h);
  printf("lgamma: grad err=%.3e hess err=%.3e\n", fabs(g_ad-g_fd), fabs(H_ad-H_fd));
  bool ok = fabs(g_ad-g_fd)<1e-6 && fabs(H_ad-H_fd)<1e-4; printf("RESULT: %s\n", ok? "OK":"FAIL");
  cudaFree(dX); cudaFree(df); cudaFree(dg); cudaFree(dH); return ok?0:1; }