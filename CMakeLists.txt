cmake_minimum_required(VERSION 3.22)
project(cuHyperDual LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enable reasonable defaults
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
add_definitions(-D_FORCE_INLINES)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(CUHD_USE_DOUBLE "Build examples in double precision" ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(hyperdual src/main.cu src/kernel.cu)

# Avoid unsafe fast-math; enable FMA contractions where legal
target_compile_options(hyperdual PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)


# File: include/hyperdual.cuh
#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

// Hyper-dual number: r + e1*ε1 + e2*ε2 + e12*ε1ε2
// εi^2 = 0, ε1ε2 = ε2ε1
//
// Stores four scalars; arithmetic is branchless and GPU-friendly.

template <typename T>
struct HD {
  T r, e1, e2, e12;
  __host__ __device__ HD() = default;
  __host__ __device__ HD(T r_, T e1_=T(0), T e2_=T(0), T e12_=T(0))
      : r(r_), e1(e1_), e2(e2_), e12(e12_) {}
};

// Seeds for ∂/∂x and ∂/∂y

template <typename T>
__host__ __device__ inline HD<T> make_seed_x(T x){ return HD<T>(x, T(1), T(0), T(0)); }

template <typename T>
__host__ __device__ inline HD<T> make_seed_y(T y){ return HD<T>(y, T(0), T(1), T(0)); }

// Basic arithmetic

template <typename T>
__host__ __device__ inline HD<T> operator+(const HD<T>& a, const HD<T>& b){
  return HD<T>(a.r+b.r, a.e1+b.e1, a.e2+b.e2, a.e12+b.e12);
}

template <typename T>
__host__ __device__ inline HD<T> operator-(const HD<T>& a, const HD<T>& b){
  return HD<T>(a.r-b.r, a.e1-b.e1, a.e2-b.e2, a.e12-b.e12);
}

template <typename T>
__host__ __device__ inline HD<T> operator*(const HD<T>& u, const HD<T>& v){
  return HD<T>(
    u.r * v.r,
    u.e1 * v.r + u.r * v.e1,
    u.e2 * v.r + u.r * v.e2,
    u.e12 * v.r + u.r * v.e12 + u.e1 * v.e2 + u.e2 * v.e1
  );
}

template <typename T>
__host__ __device__ inline HD<T> inv(const HD<T>& v){
  const T vr = v.r;
  const T vr2 = vr * vr;
  const T vr3 = vr2 * vr;
  return HD<T>(
    T(1)/vr,
    -v.e1/vr2,
    -v.e2/vr2,
    (T(2) * v.e1 * v.e2) / vr3 - v.e12 / vr2
  );
}

template <typename T>
__host__ __device__ inline HD<T> operator/(const HD<T>& u, const HD<T>& v){
  return u * inv(v);
}

// Unary function lifts: implement explicitly to avoid device-lambda issues

template <typename T>
__host__ __device__ inline HD<T> hsin(const HD<T>& x){
  const T r  = sin(x.r);
  const T fr = cos(x.r);
  const T f2 = -sin(x.r);
  return HD<T>(r, fr * x.e1, fr * x.e2, f2 * (x.e1 * x.e2) + fr * x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hcos(const HD<T>& x){
  const T r  = cos(x.r);
  const T fr = -sin(x.r);
  const T f2 = -cos(x.r);
  return HD<T>(r, fr * x.e1, fr * x.e2, f2 * (x.e1 * x.e2) + fr * x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hexp(const HD<T>& x){
  const T r  = exp(x.r);
  const T fr = r;
  const T f2 = r;
  return HD<T>(r, fr * x.e1, fr * x.e2, f2 * (x.e1 * x.e2) + fr * x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hlog(const HD<T>& x){
  const T r  = log(x.r);
  const T fr = T(1)/x.r;
  const T f2 = -T(1)/(x.r * x.r);
  return HD<T>(r, fr * x.e1, fr * x.e2, f2 * (x.e1 * x.e2) + fr * x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hpow(const HD<T>& x, T a){
  const T r  = pow(x.r, a);
  const T fr = a * pow(x.r, a - T(1));
  const T f2 = a * (a - T(1)) * pow(x.r, a - T(2));
  return HD<T>(r, fr * x.e1, fr * x.e2, f2 * (x.e1 * x.e2) + fr * x.e12);
}

// Convenience ops

template <typename T>
__host__ __device__ inline HD<T> hadd(const HD<T>& a, const HD<T>& b){ return a + b; }

template <typename T>
__host__ __device__ inline HD<T> hsub(const HD<T>& a, const HD<T>& b){ return a - b; }

//////////////////////////////////////////
// Example target function f(x,y)
//////////////////////////////////////////

template <typename T>
__host__ __device__ inline HD<T> f_xy(const HD<T>& x, const HD<T>& y){
  // f(x,y) = sin(x*y) + exp(x) / log(1 + y)
  HD<T> term1 = hsin(x * y);
  HD<T> denom = hlog( HD<T>(T(1)) + y );
  HD<T> term2 = hexp(x) / denom;
  return term1 + term2;
}


# File: src/kernel.cu
#include "hyperdual.cuh"

// Evaluate f(x,y) for a batch, returning value and derivatives via hyper-dual

template <typename T>
__global__ void eval_batch_kernel(
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ out_r,
    T* __restrict__ out_dx,
    T* __restrict__ out_dy,
    T* __restrict__ out_dxy,
    int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;

  HD<T> hx = make_seed_x<T>(x[i]);
  HD<T> hy = make_seed_y<T>(y[i]);

  HD<T> h = f_xy(hx, hy);

  out_r  [i] = h.r;
  out_dx [i] = h.e1;
  out_dy [i] = h.e2;
  out_dxy[i] = h.e12;
}

extern "C" void launch_eval_batch_float(
    const float* x, const float* y,
    float* r, float* dx, float* dy, float* dxy, int n,
    cudaStream_t stream)
{
  dim3 blk(256);
  dim3 grd((n + blk.x - 1) / blk.x);
  eval_batch_kernel<float><<<grd, blk, 0, stream>>>(x, y, r, dx, dy, dxy, n);
}

extern "C" void launch_eval_batch_double(
    const double* x, const double* y,
    double* r, double* dx, double* dy, double* dxy, int n,
    cudaStream_t stream)
{
  dim3 blk(256);
  dim3 grd((n + blk.x - 1) / blk.x);
  eval_batch_kernel<double><<<grd, blk, 0, stream>>>(x, y, r, dx, dy, dxy, n);
}


########################################
# File: src/main.cu
########################################
#include <cstdio>
#include <vector>
#include <random>
#include <cassert>
#include <cuda_runtime.h>

// Declarations from kernel.cu
extern "C" void launch_eval_batch_float(
    const float* x, const float* y,
    float* r, float* dx, float* dy, float* dxy, int n,
    cudaStream_t stream);
extern "C" void launch_eval_batch_double(
    const double* x, const double* y,
    double* r, double* dx, double* dy, double* dxy, int n,
    cudaStream_t stream);

// CPU reference: finite differences (central)
template <typename T>
T f_scalar(T x, T y){
  return sin(x*y) + exp(x)/log(1+y);
}

template <typename T>
void finite_diff(T x, T y, T& fx, T& dfx, T& dfy, T& dxy){
  const T h = 1e-6;
  const T hh = 1e-4; // larger step for mixed to reduce cancellation
  fx = f_scalar<T>(x,y);
  dfx = (f_scalar<T>(x+h,y) - f_scalar<T>(x-h,y)) / (2*h);
  dfy = (f_scalar<T>(x,y+h) - f_scalar<T>(x,y-h)) / (2*h);
  dxy = ( f_scalar<T>(x+hh,y+hh) - f_scalar<T>(x+hh,y-hh)
        - f_scalar<T>(x-hh,y+hh) + f_scalar<T>(x-hh,y-hh) ) / (4*hh*hh);
}

int main(){
  using T = double; // switch to float for speed tests
  const int N = 1<<20; // 1M samples

  std::mt19937 rng(42);
  std::uniform_real_distribution<T> dist_x(0.1, 2.0);
  std::uniform_real_distribution<T> dist_y(0.05, 1.5);

  std::vector<T> hx(N), hy(N);
  for(int i=0;i<N;++i){ hx[i]=dist_x(rng); hy[i]=dist_y(rng); }

  T *dx_x, *dx_y, *dr, *ddx, *ddy, *ddxy;
  cudaMalloc(&dx_x,  N*sizeof(T));
  cudaMalloc(&dx_y,  N*sizeof(T));
  cudaMalloc(&dr,    N*sizeof(T));
  cudaMalloc(&ddx,   N*sizeof(T));
  cudaMalloc(&ddy,   N*sizeof(T));
  cudaMalloc(&ddxy,  N*sizeof(T));

  cudaMemcpy(dx_x, hx.data(), N*sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dx_y, hy.data(), N*sizeof(T), cudaMemcpyHostToDevice);

  cudaStream_t stream; cudaStreamCreate(&stream);

  // Warm-up + run
  if constexpr (std::is_same<T,double>::value){
    launch_eval_batch_double(dx_x, dx_y, dr, ddx, ddy, ddxy, N, stream);
  } else {
    launch_eval_batch_float((float*)dx_x, (float*)dx_y, (float*)dr, (float*)ddx, (float*)ddy, (float*)ddxy, N, stream);
  }
  cudaStreamSynchronize(stream);

  // Copy back a small sample to verify numerics
  std::vector<T> hr(10), hdx(10), hdy(10), hdxy(10);
  cudaMemcpy(hr.data(),   dr,   10*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(hdx.data(),  ddx,  10*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(hdy.data(),  ddy,  10*sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(hdxy.data(), ddxy,10*sizeof(T), cudaMemcpyDeviceToHost);

  // Check first few against finite differences
  for(int i=0;i<5;++i){
    T fx, dfx, dfy, dxy;
    finite_diff<T>(hx[i], hy[i], fx, dfx, dfy, dxy);
    printf("i=%d\n  GPU: f=%.12e dx=%.12e dy=%.12e dxy=%.12e\n  CPU: f=%.12e dx=%.12e dy=%.12e dxy=%.12e\n",
           i, hr[i], hdx[i], hdy[i], hdxy[i], fx, dfx, dfy, dxy);
  }

  // Simple throughput timing
  const int iters = 10;
  float ms=0; cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a, stream);
  for(int t=0;t<iters;++t){
    if constexpr (std::is_same<T,double>::value){
      launch_eval_batch_double(dx_x, dx_y, dr, ddx, ddy, ddxy, N, stream);
    } else {
      launch_eval_batch_float((float*)dx_x, (float*)dx_y, (float*)dr, (float*)ddx, (float*)ddy, (float*)ddxy, N, stream);
    }
  }
  cudaEventRecord(b, stream); cudaEventSynchronize(b);
  cudaEventElapsedTime(&ms, a, b);
  double gsps = (double)N * iters / (ms*1e-3) / 1e9; // Gsamples/s
  printf("Throughput: %.3f Gsamples/s (%s)\n", gsps, std::is_same<T,double>::value?"fp64":"fp32");

  cudaFree(dx_x); cudaFree(dx_y); cudaFree(dr); cudaFree(ddx); cudaFree(ddy); cudaFree(ddxy);
  cudaStreamDestroy(stream);
  return 0;
}
