# cuHyperDual – minimal GPU hyper-dual engine (starter)
#
# Project layout (copy into files as named below)
#
# hyperdual-gpu/
# ├── CMakeLists.txt
# ├── include/
# │   ├── hyperdual.cuh
# │   ├── jacobian_kernels.cuh
# │   └── runner.cuh
# └── src/
#     ├── kernel.cu
#     └── main.cu

########################################
# File: CMakeLists.txt
########################################
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

########################################
# File: include/hyperdual.cuh
########################################
#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

// =====================
// Hyper-dual (2-dir, 2nd order mixed) type
// =====================

template <typename T>
struct HD {
  T r, e1, e2, e12;
  __host__ __device__ HD() = default;
  __host__ __device__ HD(T r_, T e1_=T(0), T e2_=T(0), T e12_=T(0))
      : r(r_), e1(e1_), e2(e2_), e12(e12_) {}
};

template <typename T>
__host__ __device__ inline HD<T> make_seed_x(T x){ return HD<T>(x, T(1), T(0), T(0)); }

template <typename T>
__host__ __device__ inline HD<T> make_seed_y(T y){ return HD<T>(y, T(0), T(1), T(0)); }

// Basic arithmetic (HD)

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

// Unary lifts (HD)

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

// Extra unary lifts (HD): tanh, erf, log1p, expm1

template <typename T>
__host__ __device__ inline HD<T> htanh(const HD<T>& x){
  const T t  = tanh(x.r);
  const T fr = T(1) - t*t;             // sech^2
  const T f2 = -T(2) * t * fr;         // d/dx(sech^2) = -2 tanh * sech^2
  return HD<T>(t, fr*x.e1, fr*x.e2, f2*(x.e1*x.e2) + fr*x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> herf(const HD<T>& x){
  const T r  = erf(x.r);
  const T fr = T(2.0/1.7724538509055160273) * exp(-x.r*x.r); // 2/sqrt(pi)
  const T f2 = -T(4.0/1.7724538509055160273) * x.r * exp(-x.r*x.r);
  return HD<T>(r, fr*x.e1, fr*x.e2, f2*(x.e1*x.e2) + fr*x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hlog1p(const HD<T>& x){
  const T r  = log1p(x.r);
  const T fr = T(1)/(T(1)+x.r);
  const T f2 = -fr*fr;
  return HD<T>(r, fr*x.e1, fr*x.e2, f2*(x.e1*x.e2) + fr*x.e12);
}

template <typename T>
__host__ __device__ inline HD<T> hexpm1(const HD<T>& x){
  const T r  = expm1(x.r);
  const T fr = r + T(1);
  const T f2 = fr;
  return HD<T>(r, fr*x.e1, fr*x.e2, f2*(x.e1*x.e2) + fr*x.e12);
}

// Example target function f(x,y) using HD

template <typename T>
__host__ __device__ inline HD<T> f_xy(const HD<T>& x, const HD<T>& y){
  HD<T> term1 = hsin(x * y);
  HD<T> denom = hlog( HD<T>(T(1)) + y );
  HD<T> term2 = hexp(x) / denom;
  return term1 + term2;
}

// =====================
// Multi-dual (K-dir, 1st order) type for full Jacobian columns
// =====================

template <int K, typename T>
struct MD {
  T r;            // primal
  T e[K];         // first-order components along K seed directions
  __host__ __device__ MD() = default;
  __host__ __device__ explicit MD(T r_): r(r_) { for(int i=0;i<K;++i) e[i]=T(0); }
  __host__ __device__ MD(T r_, const T* v){ r=r_; for(int i=0;i<K;++i) e[i]=v[i]; }
};

template <int K, typename T>
__host__ __device__ inline MD<K,T> make_seed(T x, int j){
  MD<K,T> a; a.r = x; for(int i=0;i<K;++i) a.e[i]=T(0); if(j>=0 && j<K) a.e[j]=T(1); return a;
}

// MD ops

template <int K, typename T>
__host__ __device__ inline MD<K,T> operator+(const MD<K,T>& a, const MD<K,T>& b){
  MD<K,T> c; c.r = a.r + b.r; for(int i=0;i<K;++i) c.e[i]=a.e[i]+b.e[i]; return c;
}

template <int K, typename T>
__host__ __device__ inline MD<K,T> operator-(const MD<K,T>& a, const MD<K,T>& b){
  MD<K,T> c; c.r = a.r - b.r; for(int i=0;i<K;++i) c.e[i]=a.e[i]-b.e[i]; return c;
}

template <int K, typename T>
__host__ __device__ inline MD<K,T> operator*(const MD<K,T>& u, const MD<K,T>& v){
  MD<K,T> c; c.r = u.r * v.r; for(int i=0;i<K;++i) c.e[i] = u.e[i]*v.r + u.r*v.e[i]; return c;
}

template <int K, typename T>
__host__ __device__ inline MD<K,T> inv(const MD<K,T>& v){
  MD<K,T> w; const T vr=v.r; const T vr2=vr*vr; w.r = T(1)/vr; for(int i=0;i<K;++i) w.e[i] = -v.e[i]/vr2; return w;
}

template <int K, typename T>
__host__ __device__ inline MD<K,T> operator/(const MD<K,T>& u, const MD<K,T>& v){ return u*inv(v); }

// Unary lifts (MD)

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_sin(const MD<K,T>& x){ MD<K,T> y; y.r=sin(x.r); const T fr=cos(x.r); for(int i=0;i<K;++i) y.e[i]=fr*x.e[i]; return y; }

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_exp(const MD<K,T>& x){ MD<K,T> y; y.r=exp(x.r); for(int i=0;i<K;++i) y.e[i]=y.r*x.e[i]; return y; }

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_log(const MD<K,T>& x){ MD<K,T> y; y.r=log(x.r); const T fr=T(1)/x.r; for(int i=0;i<K;++i) y.e[i]=fr*x.e[i]; return y; }

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_tanh(const MD<K,T>& x){ MD<K,T> y; const T t=tanh(x.r); const T fr=T(1)-t*t; y.r=t; for(int i=0;i<K;++i) y.e[i]=fr*x.e[i]; return y; }

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_log1p(const MD<K,T>& x){ MD<K,T> y; y.r=log1p(x.r); const T fr=T(1)/(T(1)+x.r); for(int i=0;i<K;++i) y.e[i]=fr*x.e[i]; return y; }

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_expm1(const MD<K,T>& x){ MD<K,T> y; y.r=expm1(x.r); const T fr=y.r+T(1); for(int i=0;i<K;++i) y.e[i]=fr*x.e[i]; return y; }

// Binary lifts (MD)

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_hypot(const MD<K,T>& x, const MD<K,T>& y){
  MD<K,T> z; z.r = hypot(x.r, y.r); if(z.r==T(0)){ for(int i=0;i<K;++i) z.e[i]=T(0); return z; }
  const T inv = T(1)/z.r; // ∂r/∂x = x/r, ∂r/∂y = y/r
  #pragma unroll
  for(int i=0;i<K;++i) z.e[i] = inv*( x.r*x.e[i] + y.r*y.e[i] );
  return z;
}

template <int K, typename T>
__host__ __device__ inline MD<K,T> md_atan2(const MD<K,T>& y, const MD<K,T>& x){
  MD<K,T> z; z.r = atan2(y.r, x.r); const T den = x.r*x.r + y.r*y.r; const T inv = T(1)/den;
  #pragma unroll
  for(int i=0;i<K;++i) z.e[i] = (-y.r*inv)*x.e[i] + (x.r*inv)*y.e[i];
  return z;
}

// Example f(x,y) using MD (K>=2)

template <int K, typename T>
__host__ __device__ inline MD<K,T> f_xy_md(const MD<K,T>& x, const MD<K,T>& y){
  MD<K,T> term1 = md_sin<K,T>(x * y);
  MD<K,T> denom = md_log<K,T>( MD<K,T>(T(1)) + y );
  MD<K,T> term2 = md_exp<K,T>(x) / denom;
  return term1 + term2;
}

########################################
# File: include/jacobian_kernels.cuh
########################################
#pragma once
#include "hyperdual.cuh"

// Generic KxM Jacobian kernel template

template<int K, int M, typename T, typename F>
__global__ void eval_jacobian_KM(const T* const* __restrict__ inputs,
                                 T* const* __restrict__ outF,
                                 T* const* __restrict__ outJ,
                                 int n, F f)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  MD<K,T> x[K];
  #pragma unroll
  for(int j=0;j<K;++j){ x[j] = make_seed<K,T>( inputs[j][i], j ); }
  MD<K,T> y[M];
  f.template operator()<K,T>(x, y);
  #pragma unroll
  for(int m=0;m<M;++m){ outF[m][i] = y[m].r; }
  #pragma unroll
  for(int m=0;m<M;++m){
    #pragma unroll
    for(int j=0;j<K;++j){ outJ[m*K + j][i] = y[m].e[j]; }
  }
}

// Example functor producing M=2 outputs from K=2 inputs
struct Fun2From2 {
  template<int K, typename T>
  __device__ void operator()(const MD<K,T>* x, MD<K,T>* y) const {
    y[0] = f_xy_md<K,T>(x[0], x[1]);
    y[1] = md_tanh<K,T>(x[0]) + md_hypot<K,T>(x[0], x[1]);
  }
};

########################################
# File: include/runner.cuh
########################################
#pragma once
#include <array>
#include <vector>
#include <cuda_runtime.h>
#include "jacobian_kernels.cuh"

// Build device arrays of pointers from host arrays

template <typename T>
static inline T** make_device_ptr_array(const std::vector<T*>& host_ptrs){
  T** d=nullptr; cudaMalloc(&d, host_ptrs.size()*sizeof(T*));
  cudaMemcpy(d, host_ptrs.data(), host_ptrs.size()*sizeof(T*), cudaMemcpyHostToDevice);
  return d;
}

template <typename T, size_t N>
static inline T** make_device_ptr_array(const std::array<T*,N>& arr){
  T** d=nullptr; cudaMalloc(&d, N*sizeof(T*));
  cudaMemcpy(d, arr.data(), N*sizeof(T*), cudaMemcpyHostToDevice);
  return d;
}

// Generic Runner for arbitrary K, M, T, Functor

template <int K, int M, typename T, typename Fun>
struct Runner {
  static void run(const std::array<T*,K>& inputs,
                  const std::array<T*,M>& outF,
                  const std::array<T*,M*K>& outJ,
                  int N, cudaStream_t s){
    T** d_inputs = make_device_ptr_array(inputs);
    T** d_outF   = make_device_ptr_array(outF);
    T** d_outJ   = make_device_ptr_array(outJ);
    dim3 blk(256), grd((N+255)/256);
    Fun F{};
    eval_jacobian_KM<K,M,T,Fun><<<grd, blk, 0, s>>>(
      const_cast<const T* const*>(d_inputs), d_outF, d_outJ, N, F);
    cudaFree(d_inputs); cudaFree(d_outF); cudaFree(d_outJ);
  }
};

########################################
# File: src/kernel.cu
########################################
#include "hyperdual.cuh"
#include "jacobian_kernels.cuh"

// ===== Existing HD batch kernel (value + dx,dy,dxy) =====

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
  dim3 blk(256); dim3 grd((n + blk.x - 1) / blk.x);
  eval_batch_kernel<float><<<grd, blk, 0, stream>>>(x, y, r, dx, dy, dxy, n);
}

extern "C" void launch_eval_batch_double(
    const double* x, const double* y,
    double* r, double* dx, double* dy, double* dxy, int n,
    cudaStream_t stream)
{
  dim3 blk(256); dim3 grd((n + blk.x - 1) / blk.x);
  eval_batch_kernel<double><<<grd, blk, 0, stream>>>(x, y, r, dx, dy, dxy, n);
}

// Mixed-precision HD path (primals half, derivatives float) 

__global__ void eval_batch_mixed_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ y,
    __half* __restrict__ out_r,
    float* __restrict__ out_dx,
    float* __restrict__ out_dy,
    float* __restrict__ out_dxy,
    int n)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= n) return;
  HD_mixed hx = make_seed_x_mixed(x[i]);
  HD_mixed hy = make_seed_y_mixed(y[i]);
  HD_mixed h  = hsin(hx * hy) + hexp(hx) / hlog( HD_mixed(1.f) + hy );
  out_r  [i] = __float2half(h.r);
  out_dx [i] = h.e1;
  out_dy [i] = h.e2;
  out_dxy[i] = h.e12;
}

extern "C" void launch_eval_batch_mixed(
    const __half* x, const __half* y,
    __half* r, float* dx, float* dy, float* dxy, int n, cudaStream_t stream)
{
  dim3 blk(256); dim3 grd((n + blk.x - 1) / blk.x);
  eval_batch_mixed_kernel<<<grd, blk, 0, stream>>>(x, y, r, dx, dy, dxy, n);
}


option(BUILD_PY \"Build Python module with pybind11\" OFF)
if(BUILD_PY)
  find_package(pybind11 REQUIRED)
  add_library(cuhd MODULE src/pybind_module.cpp)
  set_target_properties(cuhd PROPERTIES PREFIX \"\" OUTPUT_NAME \"cuhd\")
  target_link_libraries(cuhd PRIVATE pybind11::module)
  target_include_directories(cuhd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # Link CUDA runtime if needed
  target_link_libraries(cuhd PRIVATE cuda)
endif()
