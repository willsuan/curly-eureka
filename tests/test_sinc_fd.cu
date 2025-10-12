// tests/test_sinc_fd.cu
#include <cstdio>
#include <array>
#include <cmath>
#include "cudual/kernels.cuh"
using namespace cudadual;

// f(x) = sinc(x) + 0.1*x*x
struct F {
  template <class Num>
  CDUAL_HD Num operator()(const std::array<Num,1>& x) const {
    return sinc(x[0]) + Num(0.1)*x[0]*x[0];
  }
  double host_eval(const double* v) const {
    double x=v[0]; double s = (std::fabs(x) > 1e-4) ? std::sin(x)/x : (1.0 - x*x/6.0 + (x*x)*(x*x)/120.0);
    return s + 0.1*x*x;
  }
};

int main(){
  using T=double; constexpr int N=1; const int batch=1; T x[N]={1e-6};
  T *dX=nullptr,*df=nullptr,*dg=nullptr,*dH=nullptr;
  cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&df,sizeof(T)*batch); cudaMalloc(&dg,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*N*N);
  cudaMemcpy(dX,x,sizeof(T)*batch*N,cudaMemcpyHostToDevice);
  F f; launch_grad_hess<T,N>(f,dX,batch,df,dg,dH); cudaDeviceSynchronize();
  T f_ad=0,g_ad[N],H_ad[N*N];
  cudaMemcpy(&f_ad,df,sizeof(T),cudaMemcpyDeviceToHost);
  cudaMemcpy(g_ad,dg,sizeof(T)*N,cudaMemcpyDeviceToHost);
  cudaMemcpy(H_ad,dH,sizeof(T)*N*N,cudaMemcpyDeviceToHost);

  auto fh=[&](const T* v){ return f.host_eval(v); };
  T h=1e-6,g_fd[N],H_fd[N*N]; T x0[1]={x[0]};
  for(int i=0;i<N;++i){ T xp[1]={x0[0]}; T xm[1]={x0[0]}; xp[i]+=h; xm[i]-=h; g_fd[i]=(fh(xp)-fh(xm))/(2*h); }
  for(int i=0;i<N;++i){ T xp[1]={x0[0]}; T xm[1]={x0[0]}; H_fd[i*N+i]=(fh(xp)-2*fh(x0)+fh(xm))/(h*h); }
  auto maxabs=[&](const T* a,const T* b,int n){ T m=0; for(int i=0;i<n;++i) m=fmax(m,fabs(a[i]-b[i])); return m; };
  T ge=maxabs(g_ad,g_fd,N), he=maxabs(H_ad,H_fd,N*N);
  std::printf("sinc: grad err=%.3e, hess err=%.3e\n", ge, he);
  bool ok=(ge<1e-8)&&(he<1e-6); std::printf("RESULT: %s\n", ok? "OK":"FAIL");
  cudaFree(dX); cudaFree(df); cudaFree(dg); cudaFree(dH);
  return ok?0:1;
}
