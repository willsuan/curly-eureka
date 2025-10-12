// examples/benchmark_ad.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <cuda_runtime.h>
#include "cudual/advanced_kernels.cuh"
#include "cudual/kernels.cuh"
#include "cudual/tiled_kernels.cuh"
#include "cudual/tiled_kernels_vec.cuh"
#include "cudual/plans_vec.cuh"

using namespace cudadual;

struct Args {
  int N=32, M=2, K=8, batch=1<<14, repeats=20, warmup=5;
  int sample_chunk=0, pair_chunk=0;
  DataLayout layout=DataLayout::AoS;
  std::string mode="vec-hess-tiled"; // hess-hyper, hess-multi2, jac-hess-full, vec-hess-tiled, vec-hess-plan
  std::string csv="";
  bool sweepK=false;
};

static int to_int(const char* s){ return std::atoi(s); }
static bool starts_with(const char* s, const char* k){ return std::strncmp(s,k,std::strlen(k))==0; }

static void parse(int argc, char** argv, Args& a){
  for(int i=1;i<argc;++i){
    if (starts_with(argv[i],"--N=")) a.N = to_int(argv[i]+4);
    else if (starts_with(argv[i],"--M=")) a.M = to_int(argv[i]+4);
    else if (starts_with(argv[i],"--K=")) a.K = to_int(argv[i]+4);
    else if (starts_with(argv[i],"--batch=")) a.batch = to_int(argv[i]+8);
    else if (starts_with(argv[i],"--repeats=")) a.repeats = to_int(argv[i]+10);
    else if (starts_with(argv[i],"--warmup=")) a.warmup = to_int(argv[i]+9);
    else if (starts_with(argv[i],"--sample-chunk=")) a.sample_chunk = to_int(argv[i]+15);
    else if (starts_with(argv[i],"--pair-chunk=")) a.pair_chunk = to_int(argv[i]+13);
    else if (starts_with(argv[i],"--layout=")){
      const char* v = argv[i]+9;
      a.layout = (std::strcmp(v,"soa")==0) ? DataLayout::SoA : DataLayout::AoS;
    } else if (starts_with(argv[i],"--mode=")){
      a.mode = std::string(argv[i]+7);
    } else if (starts_with(argv[i],"--csv=")){
      a.csv = std::string(argv[i]+6);
    } else if (std::strcmp(argv[i],"--sweepK")==0){
      a.sweepK = true;
    } else {
      std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
    }
  }
}

static void print_device_info(){
  int dev=0; cudaDeviceProp p{}; cudaGetDevice(&dev); cudaGetDeviceProperties(&p,dev);
  printf("Device %d: %s, CC %d.%d, SMs=%d, Mem=%.1f GB\n", dev, p.name, p.major, p.minor, p.multiProcessorCount, p.totalGlobalMem/1e9);
}
static void maybe_csv_header(const std::string& csv){
  if (csv.empty()) return;
  FILE* f = std::fopen(csv.c_str(), "a");
  if (!f) return;
  static bool first=true;
  if (first){ std::fprintf(f, "mode,N,M,K,batch,layout,sample_chunk,pair_chunk,repeats,warmup,ms,elems_per_s\n"); }
  first=false; std::fclose(f);
}
static void csv_row(const std::string& csv, const std::string& mode,int N,int M,int K,int batch,const char* layout,int sample_chunk,int pair_chunk,int repeats,int warmup,double ms,double eps){
  if (csv.empty()) return;
  FILE* f = std::fopen(csv.c_str(), "a"); if (!f) return;
  std::fprintf(f, "%s,%d,%d,%d,%d,%s,%d,%d,%d,%d,%.6f,%.6e\n", mode.c_str(), N,M,K,batch,layout,sample_chunk,pair_chunk,repeats,warmup,ms,eps);
  std::fclose(f);
}

template <int N, int M>
struct BenchF {
  template <class Num>
  CDUAL_HD std::array<Num,M> operator()(const std::array<Num,N>& x) const {
    std::array<Num,M> y;
    Num s = Num(0);
    #pragma unroll
    for (int i=0;i<N;++i){
      s = s + sin(Num(0.2)*x[i]) + gelu(x[i]) + gelu_approx(Num(0.5)*x[i]) + softplus(x[i]);
      s = s + log1p(x[i]*x[i]) + exp(Num(0.01)*x[i]) + sigmoid(x[i]);
    }
    y[0] = s + lgamma(Num(2) + abs(x[0]) + Num(0.1)) + erfcx(Num(0.1)*x[1]+Num(2));
    if constexpr (M>1){
      Num t = Num(0);
      #pragma unroll
      for(int i=0;i<N;++i){
        t = t + tanh(Num(0.2)*x[i]) + silu(x[i]) + leaky_relu(x[i]);
      }
      // small-N logsumexp
      Num arr[4]; int nn = (N<4? N:4); for (int i=0;i<nn;++i) arr[i]=x[i];
      y[1] = t + logsumexp(arr, nn);
    }
    return y;
  }
};

template <typename F>
float time_kernel(std::function<void(F)> launcher, F f, int warmup, int repeats){
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for(int i=0;i<warmup;++i){ launcher(f); }
  cudaEventRecord(e0);
  for(int r=0;r<repeats;++r){ launcher(f); }
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  return ms / repeats;
}

template <int N, int M, int K>
void runK(const Args& a){
  using T=double;
  const int batch=a.batch;
  const int stride = (a.layout==DataLayout::AoS) ? N : batch;
  std::vector<T> hX((size_t)batch*N);
  for (int b=0;b<batch;++b) for (int i=0;i<N;++i) hX[b*N+i] = 0.01*b + 0.1*i;
  T *dX=nullptr; cudaMalloc(&dX, sizeof(T)*batch*N);
  cudaMemcpy(dX, hX.data(), sizeof(T)*batch*N, cudaMemcpyHostToDevice);

  // Outputs
  T *df=nullptr, *dJ=nullptr, *dH=nullptr, *dHvec=nullptr, *dy=nullptr;
  cudaMalloc(&df, sizeof(T)*batch);
  cudaMalloc(&dJ, sizeof(T)*batch*M*N);
  cudaMalloc(&dH, sizeof(T)*batch*N*N);
  cudaMalloc(&dHvec, sizeof(T)*batch*M*N*N);
  cudaMalloc(&dy, sizeof(T)*batch*M);

  BenchF<N,M> f;

  float ms=0.0f;
  size_t elems=0;

  if (a.mode == "hess-hyper"){
    struct Fwrap { BenchF<N,M> f; template <class Num> CDUAL_HD Num operator()(const std::array<Num,N>& x) const { return f(x)[0]; } } fw{f};
    auto launch = [&](BenchF<N,M> /*unused*/){
      if (a.layout==DataLayout::AoS) launch_hessian_layout<T,N>(fw, dX, batch, stride, DataLayout::AoS, dH);
      else                           launch_hessian_layout<T,N>(fw, dX, batch, stride, DataLayout::SoA, dH);
      cudaDeviceSynchronize();
    };
    ms = time_kernel<BenchF<N,M>>(launch, f, a.warmup, a.repeats);
    elems = (size_t)batch*N*N;
  } else if (a.mode == "hess-multi2"){
    struct Fwrap { BenchF<N,M> f; template <class Num> CDUAL_HD Num operator()(const std::array<Num,N>& x) const { return f(x)[0]; } } fw{f};
    auto launch = [&](BenchF<N,M> /*unused*/){
      if (a.layout==DataLayout::AoS) launch_grad_hess_layout<T,N>(fw, dX, batch, stride, DataLayout::AoS, df, dJ, dH);
      else                           launch_grad_hess_layout<T,N>(fw, dX, batch, stride, DataLayout::SoA, df, dJ, dH);
      cudaDeviceSynchronize();
    };
    ms = time_kernel<BenchF<N,M>>(launch, f, a.warmup, a.repeats);
    elems = (size_t)batch*N*N;
  } else if (a.mode == "jac-hess-full"){
    auto launch = [&](BenchF<N,M> /*unused*/){
      launch_jacobian_hessians<T,N,M>(f, dX, batch, dy, dJ, dHvec);
      cudaDeviceSynchronize();
    };
    ms = time_kernel<BenchF<N,M>>(launch, f, a.warmup, a.repeats);
    elems = (size_t)batch*M*N*N;
  } else if (a.mode == "vec-hess-tiled"){
    auto launch = [&](BenchF<N,M> /*unused*/){
      if (a.layout==DataLayout::AoS)
        launch_vec_hessians_tiled<T,N,K,M>(f, dX, batch, stride, DataLayout::AoS, dHvec);
      else
        launch_vec_hessians_tiled<T,N,K,M>(f, dX, batch, stride, DataLayout::SoA, dHvec);
      cudaDeviceSynchronize();
    };
    ms = time_kernel<BenchF<N,M>>(launch, f, a.warmup, a.repeats);
    elems = (size_t)batch*M*N*N;
  } else if (a.mode == "vec-hess-plan"){
    auto launch = [&](BenchF<N,M> /*unused*/){
      VecHessianPlan<T,N,K,M> plan;
      plan.layout = a.layout;
      plan.sample_chunk = (a.sample_chunk? a.sample_chunk : batch);
      int numTiles = (N + K - 1)/K;
      plan.pair_chunk   = (a.pair_chunk? a.pair_chunk : (numTiles*(numTiles-1))/2);
      plan.init_pairs();
      cudaStream_t s0, s1; cudaStreamCreate(&s0); cudaStreamCreate(&s1); cudaStream_t ss[2] = {s0,s1};
      execute_vec_hessians_plan(plan, f, dX, batch, stride, dHvec, ss);
      cudaStreamDestroy(s0); cudaStreamDestroy(s1);
      plan.destroy();
      cudaDeviceSynchronize();
    };
    ms = time_kernel<BenchF<N,M>>(launch, f, a.warmup, a.repeats);
    elems = (size_t)batch*M*N*N;
  } else {
    printf("Unknown mode: %s\n", a.mode.c_str());
  }

  double elts_per_s = (elems / (ms/1000.0));
  printf("Mode=%s N=%d M=%d K=%d batch=%d -> %.3f ms (avg), throughput=%.2e elems/s\n",
         a.mode.c_str(), N, M, K, batch, ms, elts_per_s);
  csv_row(a.csv, a.mode, N, M, K, batch, (a.layout==DataLayout::AoS?"aos":"soa"),
          a.sample_chunk, a.pair_chunk, a.repeats, a.warmup, ms, elts_per_s);

  cudaFree(dX); cudaFree(df); cudaFree(dJ); cudaFree(dH); cudaFree(dHvec); cudaFree(dy);
}

// Small dispatcher for K
template <int N, int M>
void run(const Args& a){
  switch(a.K){
    case 4:  runK<N,M,4>(a); break;
    case 8:  runK<N,M,8>(a); break;
    case 16: runK<N,M,16>(a); break;
    case 32: runK<N,M,32>(a); break;
    default: printf("Unsupported K=%d; use 4,8,16,32\n", a.K);
  }
}

int main(int argc, char** argv){
  Args a; parse(argc, argv, a);
  print_device_info();
  maybe_csv_header(a.csv);

  auto one = [&](int Kval){
    a.K = Kval;
    if      (a.N==16 && a.M==1) run<16,1>(a);
    else if (a.N==16 && a.M==2) run<16,2>(a);
    else if (a.N==32 && a.M==1) run<32,1>(a);
    else if (a.N==32 && a.M==2) run<32,2>(a);
    else if (a.N==64 && a.M==1) run<64,1>(a);
    else if (a.N==64 && a.M==2) run<64,2>(a);
    else if (a.N==128 && a.M==1) run<128,1>(a);
    else if (a.N==128 && a.M==2) run<128,2>(a);
    else printf("Unsupported N/M combo for this binary. Recompile with more instantiations if needed.\n");
  };

  if (a.sweepK){
    int Ks[] = {4,8,16,32};
    for (int k: Ks){ one(k); }
    return 0;
  }
  one(a.K);
  return 0;
}
