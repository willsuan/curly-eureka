#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <typeinfo>
#include <cuda_runtime.h>
#include "cudual/cudual.cuh"
#include "cudual/kernels.cuh"
#include "cudual/advanced_kernels.cuh"
#include "cudual/tiled_kernels_vec.cuh"
#include "cudual/plans_vec.cuh"
#include "cudual/packed_kernels_vec.cuh"

#ifndef DEF_N
#define DEF_N 64
#endif
#ifndef DEF_M
#define DEF_M 2
#endif
#ifndef DEF_K
#define DEF_K 16
#endif
#ifndef REAL_T
#define REAL_T double
#endif

using T = REAL_T;
constexpr int N = DEF_N;
constexpr int M = DEF_M;
constexpr int K = DEF_K;

using namespace cudadual;

static void die(const char* msg){ std::fprintf(stderr, "%s\n", msg); std::exit(1); }

template <int N>
struct FScalar {
  template <class Num>
  CDUAL_HD Num operator()(const std::array<Num,N>& x) const {
    Num s = Num(0);
    Num sum = Num(0);
    for (int i=0;i<N;++i){
      s = s + log1p(x[i]*x[i]) + sin(x[i]) * exp(Num(0.05)*x[(i+1)%N]);
      sum = sum + x[i];
    }
    s = s + lgamma(Num(0.5)*(sum + Num(2))) + normal_logcdf(sum * Num(0.1));
    return s;
  }
};
template <int N, int M>
struct FVec {
  template <class Num>
  CDUAL_HD std::array<Num,M> operator()(const std::array<Num,N>& x) const {
    std::array<Num,M> y;
    Num s0 = Num(0), s1 = Num(0);
    for (int i=0;i<N;++i){
      s0 = s0 + sin(Num(0.3)*x[i]) + softplus(Num(0.1)*x[i]);
      s1 = s1 + log1p(x[i]*x[i]) + cosh(Num(0.05)*x[i]);
    }
    y[0] = s0 + normal_cdf(x[0]);
    if constexpr (M>1) y[1] = s1 + lgamma(Num(1)+Num(0.2)*x[1]) - log1p(exp(-x[2]));
    if constexpr (M>2) y[2] = erfcx(Num(0.25)*x[0]) + sinc(x[1]);
    return y;
  }
};

float time_ms(std::function<void()> fn, int warmup, int repeat){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  for(int i=0;i<warmup;++i){ fn(); cudaDeviceSynchronize(); }
  std::vector<float> ms(repeat);
  for(int i=0;i<repeat;++i){
    cudaEventRecord(a);
    fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    cudaEventElapsedTime(&ms[i], a, b);
  }
  cudaEventDestroy(a); cudaEventDestroy(b);
  float mean=0; for(float v: ms) mean+=v; mean/=repeat;
  float var=0; for(float v: ms) var += (v-mean)*(v-mean); var/=std::max(1,repeat-1);
  std::printf("   times ms: mean=%.3f, std=%.3f (n=%d)\n", mean, std::sqrt(var), repeat);
  return mean;
}

void bench_hess_hyper(int batch){
  std::printf("[hess_hyper] N=%d batch=%d\n", N, batch);
  std::vector<T> X(batch*N); for (int b=0;b<batch;++b) for (int i=0;i<N;++i) X[b*N+i]=T(0.01*b+0.1*(i+1));
  T *dX=nullptr,*dH=nullptr; cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*N*N);
  cudaMemcpy(dX,X.data(),sizeof(T)*batch*N,cudaMemcpyHostToDevice);
  FScalar<N> f;
  auto fn=[&](){ launch_hessian<T,N>(f, dX, batch, dH); };
  float ms = time_ms(fn, 5, 30);
  double elems = double(batch)*N*N;
  std::printf("   H elements: %.3e  ->  %.3e elem/s\n", elems, elems/(ms*1e-3));
  cudaFree(dX); cudaFree(dH);
}

void bench_grad_hess(int batch){
  std::printf("[grad_hess (MultiDual2)] N=%d batch=%d\n", N, batch);
  std::vector<T> X(batch*N); for (int b=0;b<batch;++b) for (int i=0;i<N;++i) X[b*N+i]=T(0.01*b+0.1*(i+1));
  T *dX=nullptr,*df=nullptr,*dg=nullptr,*dH=nullptr; cudaMalloc(&dX,sizeof(T)*batch*N);
  cudaMalloc(&df,sizeof(T)*batch); cudaMalloc(&dg,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*N*N);
  cudaMemcpy(dX,X.data(),sizeof(T)*batch*N,cudaMemcpyHostToDevice);
  FScalar<N> f;
  auto fn=[&](){ launch_grad_hess<T,N>(f, dX, batch, df, dg, dH); };
  float ms = time_ms(fn, 5, 30);
  double elems = double(batch)*N*N;
  std::printf("   H elements: %.3e  ->  %.3e elem/s\n", elems, elems/(ms*1e-3));
  cudaFree(dX); cudaFree(df); cudaFree(dg); cudaFree(dH);
}

void bench_vec_tiled(int batch, DataLayout layout, int sample_chunk, int pair_chunk){
  std::printf("[vec_hess_tiled] N=%d M=%d K=%d batch=%d layout=%s chunkS=%d chunkP=%d\n",
              N, M, K, batch, layout==DataLayout::AoS?"AoS":"SoA", sample_chunk, pair_chunk);
  const int stride = (layout==DataLayout::AoS) ? N : batch;
  std::vector<T> Xa(batch*N); for (int b=0;b<batch;++b) for (int i=0;i<N;++i) Xa[b*N+i]=T(0.01*b+0.1*(i+1));
  std::vector<T> Xs(batch*N);
  if (layout==DataLayout::SoA){ for(int i=0;i<N;++i) for(int b=0;b<batch;++b) Xs[i*batch+b] = Xa[b*N+i]; }
  T *dX=nullptr,*dH=nullptr; cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*M*N*N);
  cudaMemcpy(dX, (layout==DataLayout::AoS? Xa.data(): Xs.data()), sizeof(T)*batch*N, cudaMemcpyHostToDevice);
  FVec<N,M> f;
  auto fn=[&](){ launch_vec_hessians_tiled_streamed<T,N,K,M>(f, dX, batch, stride, layout, dH, sample_chunk, pair_chunk); };
  float ms = time_ms(fn, 3, 20);
  double elems = double(batch)*M*N*N;
  std::printf("   H elements: %.3e  ->  %.3e elem/s\n", elems, elems/(ms*1e-3));
  cudaFree(dX); cudaFree(dH);
}

void bench_vec_plan(int batch, DataLayout layout, int sample_chunk, int pair_chunk, int use_streams){
  std::printf("[vec_hess_plan] N=%d M=%d K=%d batch=%d layout=%s chunkS=%d chunkP=%d streams=%d\n",
              N, M, K, batch, layout==DataLayout::AoS?"AoS":"SoA", sample_chunk, pair_chunk, use_streams);
  const int stride = (layout==DataLayout::AoS) ? N : batch;
  std::vector<T> Xa(batch*N); for (int b=0;b<batch;++b) for (int i=0;i<N;++i) Xa[b*N+i]=T(0.01*b+0.1*(i+1));
  std::vector<T> Xs(batch*N);
  if (layout==DataLayout::SoA){ for(int i=0;i<N;++i) for(int b=0;b<batch;++b) Xs[i*batch+b] = Xa[b*N+i]; }
  T *dX=nullptr,*dH=nullptr; cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&dH,sizeof(T)*batch*M*N*N);
  cudaMemcpy(dX, (layout==DataLayout::AoS? Xa.data(): Xs.data()), sizeof(T)*batch*N, cudaMemcpyHostToDevice);
  VecHessianPlan<T,N,K,M> plan; plan.layout=layout; plan.sample_chunk=sample_chunk; plan.pair_chunk=pair_chunk; plan.init_pairs();
  FVec<N,M> f;
  cudaStream_t s0=nullptr,s1=nullptr; cudaStream_t arr[2] = {nullptr,nullptr};
  if(use_streams){ cudaStreamCreate(&s0); cudaStreamCreate(&s1); arr[0]=s0; arr[1]=s1; }
  auto fn=[&](){ execute_vec_hessians_plan(plan, f, dX, batch, stride, dH, (use_streams? arr : nullptr)); };
  float ms = time_ms(fn, 3, 20);
  double elems = double(batch)*M*N*N;
  std::printf("   H elements: %.3e  ->  %.3e elem/s\n", elems, elems/(ms*1e-3));
  if(use_streams){ cudaStreamDestroy(s0); cudaStreamDestroy(s1); }
  plan.destroy(); cudaFree(dX); cudaFree(dH);
}

void bench_vec_packed(int batch){
  std::printf("[vec_hess_packed] N=%d M=%d batch=%d (AoS)\n", N, M, batch);
  const size_t L = N*(N+1)/2;
  std::vector<T> X(batch*N); for (int b=0;b<batch;++b) for (int i=0;i<N;++i) X[b*N+i]=T(0.01*b+0.1*(i+1));
  T *dX=nullptr,*dHpack=nullptr; cudaMalloc(&dX,sizeof(T)*batch*N); cudaMalloc(&dHpack,sizeof(T)*batch*M*L);
  cudaMemcpy(dX,X.data(),sizeof(T)*batch*N,cudaMemcpyHostToDevice);
  FVec<N,M> f;
  auto fn=[&](){ launch_vec_hessian_packed<T,N,M>(f, dX, batch, dHpack); };
  float ms = time_ms(fn, 3, 20);
  double elems = double(batch)*M*L;
  std::printf("   H(pack) elements: %.3e  ->  %.3e elem/s\n", elems, elems/(ms*1e-3));
  cudaFree(dX); cudaFree(dHpack);
}

int main(int argc, char** argv){
  int batch=4096;
  std::string kernel="vec_plan";
  DataLayout layout=DataLayout::AoS;
  int sample_chunk=1024, pair_chunk=128, use_streams=1;

  for(int i=1;i<argc;++i){
    if (!strcmp(argv[i],"--kernel") && i+1<argc){ kernel = argv[++i]; }
    else if (!strcmp(argv[i],"--batch") && i+1<argc){ batch = std::atoi(argv[++i]); }
    else if (!strcmp(argv[i],"--layout") && i+1<argc){ layout = (!strcmp(argv[++i],"SoA"))? DataLayout::SoA : DataLayout::AoS; }
    else if (!strcmp(argv[i],"--chunkS") && i+1<argc){ sample_chunk = std::atoi(argv[++i]); }
    else if (!strcmp(argv[i],"--chunkP") && i+1<argc){ pair_chunk = std::atoi(argv[++i]); }
    else if (!strcmp(argv[i],"--streams") && i+1<argc){ use_streams = std::atoi(argv[++i]); }
    else if (!strcmp(argv[i],"--help")){
      std::printf("Usage: %s [--kernel hess_hyper|grad_hess|vec_tiled|vec_plan|vec_packed] [--batch N]\n", argv[0]);
      std::printf("              [--layout AoS|SoA] [--chunkS n] [--chunkP n] [--streams 0|1]\n");
      std::printf("  Compiled constants: N=%d M=%d K=%d T=%s\n", N, M, K, typeid(T).name());
      return 0;
    }
  }

  std::printf("Benchmark compiled with: N=%d M=%d K=%d T=%s\n", N, M, K, typeid(T).name());
  if (kernel=="hess_hyper") bench_hess_hyper(batch);
  else if (kernel=="grad_hess") bench_grad_hess(batch);
  else if (kernel=="vec_tiled") bench_vec_tiled(batch, layout, sample_chunk, pair_chunk);
  else if (kernel=="vec_plan") bench_vec_plan(batch, layout, sample_chunk, pair_chunk, use_streams);
  else if (kernel=="vec_packed") bench_vec_packed(batch);
  else die("Unknown --kernel");
  return 0;
}
