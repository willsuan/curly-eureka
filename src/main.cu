#include <cstdio>
#include <vector>
#include <array>
#include <random>
#include <cassert>
#include <cuda_runtime.h>
#include "runner.cuh"

// CPU reference

template <typename T>
T f_scalar(T x, T y){ return sin(x*y) + exp(x)/log(1+y); }

template <typename T>
void finite_diff(T x, T y, T& fx, T& dfx, T& dfy, T& dxy){
  const T h = 1e-6; const T hh = 1e-4;
  fx = f_scalar<T>(x,y);
  dfx = (f_scalar<T>(x+h,y) - f_scalar<T>(x-h,y)) / (2*h);
  dfy = (f_scalar<T>(x,y+h) - f_scalar<T>(x,y-h)) / (2*h);
  dxy = ( f_scalar<T>(x+hh,y+hh) - f_scalar<T>(x+hh,y-hh)
        - f_scalar<T>(x-hh,y+hh) + f_scalar<T>(x-hh,y-hh) ) / (4*hh*hh);
}

int main(){
  using T = double; // toggle float/double
  const int N = 1<<20;
  std::mt19937 rng(42);
  std::uniform_real_distribution<T> dist_x(0.1, 2.0);
  std::uniform_real_distribution<T> dist_y(0.05,1.5);
  std::vector<T> hx(N), hy(N);
  for(int i=0;i<N;++i){ hx[i]=dist_x(rng); hy[i]=dist_y(rng); }

  // Device buffers
  T *dx_x,*dx_y,*dr,*ddx,*ddy,*ddxy;
  T *jf0,*jf1,*j00,*j01,*j10,*j11;
  cudaMalloc(&dx_x,  N*sizeof(T));
  cudaMalloc(&dx_y,  N*sizeof(T));
  cudaMalloc(&dr,    N*sizeof(T));
  cudaMalloc(&ddx,   N*sizeof(T));
  cudaMalloc(&ddy,   N*sizeof(T));
  cudaMalloc(&ddxy,  N*sizeof(T));
  cudaMalloc(&jf0,   N*sizeof(T));
  cudaMalloc(&jf1,   N*sizeof(T));
  cudaMalloc(&j00,   N*sizeof(T));
  cudaMalloc(&j01,   N*sizeof(T));
  cudaMalloc(&j10,   N*sizeof(T));
  cudaMalloc(&j11,   N*sizeof(T));

  cudaMemcpy(dx_x, hx.data(), N*sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dx_y, hy.data(), N*sizeof(T), cudaMemcpyHostToDevice);
  cudaStream_t stream; cudaStreamCreate(&stream);

  // HD path
  launch_eval_batch_double(dx_x, dx_y, dr, ddx, ddy, ddxy, N, stream);
  cudaStreamSynchronize(stream);

  // MD generic K×M runner: K=2, M=2
  std::array<T*,2> inputs   = { dx_x, dx_y };
  std::array<T*,2> outF     = { jf0, jf1 };
  std::array<T*,4> outJ     = { j00, j01, j10, j11 }; // [m*K + j]
  Runner<2,2,T,Fun2From2>::run(inputs, outF, outJ, N, stream);
  cudaStreamSynchronize(stream);

  // Verify few samples
  std::vector<T> hr(5), hdx(5), hdy(5), hdxy(5), jf0_h(5), jf1_h(5), j00_h(5), j11_h(5);
  cudaMemcpy(hr.data(),   dr,   5*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(hdx.data(),  ddx,  5*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(hdy.data(),  ddy,  5*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(hdxy.data(), ddxy, 5*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(jf0_h.data(), jf0, 5*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(jf1_h.data(), jf1, 5*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(j00_h.data(), j00, 5*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(a
