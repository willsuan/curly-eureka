// tests/test_atan2_fd.cu
#include <cstdio>
#include <array>
#include <cmath>
#include "cudual/kernels.cuh"
using namespace cudadual;

// f(x0,x1) = atan2(x1, x0) + 0.1*(x0*x0 + x1*x1)
struct F {
  template <class Num>
  CDUAL_HD Num operator()(const std::array<Num,2>& x) const {
    return atan2(x[1], x[0]) + Num(0.1)*(x[0]*x[0] + x[1]*x[1]);
  }
  double host_eval(const double* v) const {
    return std::atan2(v[1], v[0]) + 0.1*(v[0]*v[0] + v[1]*v[1]);
  }
};

int main(){
  using T=double; constexpr int N=2; const int batch=1; T x[N]={0.8, -0.5};
  T *dX=nullptr,*df=nullptr,*dg=nullptr,*dH=nullptr;
  cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&df,sizeof(T)*batch); cudaMalloc(&dg,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*N*N);
  cudaMemcpy(dX,x,sizeof(T)*batch*N,cudaMemcpyHostToDevice);
  F f; launch_grad_hess<T,N>(f,dX,batch,df,dg,dH); cudaDeviceSynchronize();
  T f_ad=0,g_ad[N],H_ad[N*N];
  cudaMemcpy(&f_ad,df,sizeof(T),cudaMemcpyDeviceToHost);
  cudaMemcpy(g_ad,dg,sizeof(T)*N,cudaMemcpyDeviceToHost);
  cudaMemcpy(H_ad,dH,sizeof(T)*N*N,cudaMemcpyDeviceToHost);

  auto fh=[&](const T* v){ return f.host_eval(v); };
  T h=1e-6,g_fd[N],H_fd[N*N]; T x0[2]={x[0],x[1]};
  for(int i=0;i<N;++i){ T xp[2]={x0[0],x0[1]}; T xm[2]={x0[0],x0[1]}; xp[i]+=h; xm[i]-=h; g_fd[i]=(fh(xp)-fh(xm))/(2*h); }
  for(int i=0;i<N;++i) for(int j=0;j<N;++j){
    if(i==j){ T xp[2]={x0[0],x0[1]}; T xm[2]={x0[0],x0[1]}; xp[i]+=h; xm[i]-=h; H_fd[i*N+i]=(fh(xp)-2*fh(x0)+fh(xm))/(h*h);
    } else {
      T xpp[2]={x0[0],x0[1]}; xpp[i]+=h; xpp[j]+=h;
      T xpm[2]={x0[0],x0[1]}; xpm[i]+=h; xpm[j]-=h;
      T xmp[2]={x0[0],x0[1]}; xmp[i]-=h; xmp[j]+=h;
      T xmm[2]={x0[0],x0[1]}; xmm[i]-=h; xmm[j]-=h;
      H_fd[i*N+j]=(fh(xpp)-fh(xpm)-fh(xmp)+fh(xmm))/(4*h*h);
    }}
  auto maxabs=[&](const T* a,const T* b,int n){ T m=0; for(int i=0;i<n;++i) m=fmax(m,fabs(a[i]-b[i])); return m; };
  T ge=maxabs(g_ad,g_fd,N), he=maxabs(H_ad,H_fd,N*N);
  std::printf("atan2: grad err=%.3e, hess err=%.3e\n", ge, he);
  bool ok=(ge<5e-8)&&(he<5e-6); std::printf("RESULT: %s\n", ok? "OK":"FAIL");
  cudaFree(dX); cudaFree(df); cudaFree(dg); cudaFree(dH);
  return ok?0:1;
}
