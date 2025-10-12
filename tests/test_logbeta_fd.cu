#include <cstdio>
#include <array>
#include <vector>
#include <cmath>
#include "cudual/kernels.cuh"
#include "cudual/cudual.cuh"
using namespace cudadual;
struct LB { template <class Num> CDUAL_HD Num operator()(const std::array<Num,2>& x) const { return lgamma(x[0]) + lgamma(x[1]) - lgamma(x[0]+x[1]); }
  double host_eval(const double* v) const { return ::lgamma(v[0]) + ::lgamma(v[1]) - ::lgamma(v[0]+v[1]); } };
int main(){ using T=double; constexpr int N=2; const int batch=1; T x[2]={2.7,1.9};
  T *dX=nullptr,*df=nullptr,*dg=nullptr,*dH=nullptr;
  cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&df,sizeof(T)*batch);
  cudaMalloc(&dg,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*N*N);
  cudaMemcpy(dX,x,sizeof(T)*N,cudaMemcpyHostToDevice);
  LB f; launch_grad_hess<T,N>(f,dX,batch,df,dg,dH); cudaDeviceSynchronize();
  T f_ad=0, g_ad[N], H_ad[N*N]; cudaMemcpy(&f_ad,df,sizeof(T),cudaMemcpyDeviceToHost);
  cudaMemcpy(g_ad,dg,sizeof(T)*N,cudaMemcpyDeviceToHost); cudaMemcpy(H_ad,dH,sizeof(T)*N*N,cudaMemcpyDeviceToHost);
  auto feval=[&](const T* v){ return f.host_eval(v); }; T h=1e-6, g_fd[N], H_fd[N*N];
  for(int i=0;i<N;++i){ T xp[2]={x[0],x[1]}, xm[2]={x[0],x[1]}; xp[i]+=h; xm[i]-=h; g_fd[i]=(feval(xp)-feval(xm))/(2*h); }
  for(int i=0;i<N;++i) for(int j=0;j<N;++j){
    if(i==j){ T xp[2]={x[0],x[1]}, xm[2]={x[0],x[1]}; xp[i]+=h; xm[i]-=h; H_fd[i*N+i]=(feval(xp)-2*feval(x)+feval(xm))/(h*h);
    } else { T xpp[2]={x[0],x[1]}; xpp[i]+=h; xpp[j]+=h; T xpm[2]={x[0],x[1]}; xpm[i]+=h; xpm[j]-=h;
      T xmp[2]={x[0],x[1]}; xmp[i]-=h; xmp[j]+=h; T xmm[2]={x[0],x[1]}; xmm[i]-=h; xmm[j]-=h;
      H_fd[i*N+j]=(feval(xpp)-feval(xpm)-feval(xmp)+feval(xmm))/(4*h*h); } }
  auto max_abs=[&](const T* a,const T* b,int n){ T m=0; for(int i=0;i<n;++i) m=fmax(m,fabs(a[i]-b[i])); return m; };
  T ge=max_abs(g_ad,g_fd,N), he=max_abs(H_ad,H_fd,N*N);
  printf("logbeta: grad err=%.3e hess err=%.3e\n", ge, he);
  bool ok=(ge<5e-7)&&(he<2e-6); printf("RESULT: %s\n", ok? "OK":"FAIL");
  cudaFree(dX); cudaFree(df); cudaFree(dg); cudaFree(dH); return ok?0:1; }